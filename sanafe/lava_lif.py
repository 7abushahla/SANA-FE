"""Lava's bit-accurate fixed-point LIF and the ``cuba_lif_int24`` soma, step for step.

The soma unit ``loihi_cuba_lif`` of the Loihi 2 candidate runs the rule of
``AbstractPyLifModelFixed.subthr_dynamics`` with the spike and reset of
``PyLifModelBitAcc`` (Lava's ``LIF``) or ``PyLifResetModelBitAcc`` (Lava's
``LIFReset``). It is the neuron of published LIF networks. The thesis networks
run on the two IF units and never on this one.

Alignment of a Lava run with a SANA-FE run
------------------------------------------
Row ``t`` (zero-based) of an activation array ``a_in`` of shape ``[steps, n]``
is the activation the neuron receives at its update ``t``. In Lava that update
has the one-based ``time_step`` ``t + 1``. A fixed-point ``RingBuffer`` source
sends column ``time_step - 1`` of its data in the same time step, so the source
holds ``a_in.T``. In SANA-FE the activation is the ``currents`` attribute, whose
element ``t`` is consumed at update ``t``. Row ``t`` of every returned trace is
the state after update ``t``, which a Lava ``Monitor`` reads after the time step
and SANA-FE's spike and potential traces record for timestep ``t + 1``.

In a chain, Lava's ``Dense`` sends at each time step the activation it
accumulated from the spikes of the previous time step (its ``a_buff``), so a
spike emitted by layer ``k`` at update ``t`` reaches layer ``k + 1`` at update
``t + 1``. SANA-FE reproduces this delay of one update with the candidate's
buffer before the soma unit. A spike delivered at timestep ``t`` is stored and
consumed by the soma at timestep ``t + 1``. The first layer receives ``a_in``
directly, with no delay, in both.

Accepted activation range
-------------------------
Lava's fixed-point ``Dense`` and ``RingBuffer`` send signed 32-bit activations,
and the LIF shifts whatever it receives by 6 bits without a range check. The
current then wraps once by ``2**24`` into ``(-2**23, 2**23]``. The neuron
accepts every activation of magnitude at most ``ACTIVATION_BOUND`` (``2**18 + 32``),
the largest magnitude for which that single wrap keeps the current inside
``(-2**23, 2**23]`` from every reachable current and decay. Over this range Lava
is a 24-bit register and the neuron reproduces it exactly. Beyond it, Lava's
current leaves 24 bits and can later overflow 32-bit arithmetic, and the neuron
refuses the input. ``docs/loihi2-profile.md`` records the probe.
"""
import numpy as np
import sanafe
from sanafe.lava_integer import _integers, _scalar_integer as _shared_scalar_integer

#: The soma unit of the candidate that runs ``cuba_lif_int24``.
LIF_SOMA_UNIT = 'loihi_cuba_lif'
#: Largest accepted activation magnitude, 2**18 + 32 (see the module docstring).
ACTIVATION_BOUND = (1 << 18) + 32
_INT32 = (-(1 << 31), (1 << 31) - 1)


def _scalar_integer(value, name, low, high):
    """The shared scalar check, after unwrapping a size-1 array (Lava Vars hold scalars as such)."""
    data = np.asarray(value)
    if isinstance(value, np.ndarray) and data.size == 1:
        value = data.reshape(()).item()
    return _shared_scalar_integer(value, name, low, high)


def _activations(a_in, low, high):
    data = _integers(a_in, 'a_in', low, high)
    if data.ndim != 2 or data.shape[0] < 1 or data.shape[1] < 1:
        raise ValueError('a_in must be a nonempty [steps, neurons] array')
    return data


def _registers(du, dv, vth, bias_mant, bias_exp, reset_interval, reset_offset, initial_v):
    return {'du': _scalar_integer(du, 'du', 0, 4095),
            'dv': _scalar_integer(dv, 'dv', 0, 4095),
            'vth': _scalar_integer(vth, 'vth', 0, (1 << 17) - 1),
            'bias_mant': _scalar_integer(bias_mant, 'bias_mant', -(1 << 12), (1 << 12) - 1),
            'bias_exp': _scalar_integer(bias_exp, 'bias_exp', 0, 7),
            'reset_interval': _scalar_integer(reset_interval, 'reset_interval', 0, _INT32[1]),
            'reset_offset': _scalar_integer(reset_offset, 'reset_offset', 0, _INT32[1]),
            'initial_v': _scalar_integer(initial_v, 'initial_v', -(1 << 23), (1 << 23) - 1)}


def lava_lif_trace(a_in, *, du, dv, vth, bias_mant=0, bias_exp=0, reset_interval=0,
                   reset_offset=0, initial_v=0):
    """Run Lava's fixed-point ``LIF`` (``LIFReset`` when ``reset_interval > 0``).

    ``a_in[t]`` is the activation of each neuron at its update ``t``. It may hold
    any signed 32-bit value, which is what Lava's own sources send. Returns
    ``{'s': spikes, 'v': voltages}``, each ``[steps, n]`` int64, after each update.
    ``reset_interval == 0`` selects ``LIF``, because Lava's ``LIFReset`` has no
    disabled value (it requires an interval of at least 1).
    """
    from lava.proc.lif.process import LIF, LIFReset
    from lava.proc.io.source import RingBuffer
    from lava.proc.monitor.process import Monitor
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.run_configs import Loihi1SimCfg

    a = _activations(a_in, *_INT32)
    r = _registers(du, dv, vth, bias_mant, bias_exp, reset_interval, reset_offset, initial_v)
    steps, n = a.shape
    params = {'shape': (n,), 'du': r['du'], 'dv': r['dv'], 'vth': r['vth'],
              'bias_mant': r['bias_mant'], 'bias_exp': r['bias_exp'], 'v': r['initial_v']}
    if r['reset_interval'] == 0:
        lif = LIF(**params)
    else:
        lif = LIFReset(**params, reset_interval=r['reset_interval'],
                       reset_offset=r['reset_offset'])
    source = RingBuffer(data=a.T.astype(np.int32))
    source.s_out.connect(lif.a_in)
    voltage, spikes = Monitor(), Monitor()
    voltage.probe(lif.v, steps)
    spikes.probe(lif.s_out, steps)
    try:
        lif.run(condition=RunSteps(num_steps=steps), run_cfg=Loihi1SimCfg(select_tag='fixed_pt'))
        v = voltage.get_data()[lif.name]['v']
        s = spikes.get_data()[lif.name]['s_out']
    finally:
        lif.stop()
    # Monitors return float64, which holds every signed 32-bit value exactly.
    return {'s': np.asarray(s).astype(np.int64), 'v': np.asarray(v).astype(np.int64)}


def _spike_matrix(spike_trace, sizes):
    spikes = [np.zeros((len(spike_trace), size), dtype=np.int64) for size in sizes]
    for t, active in enumerate(spike_trace):
        for neuron in active:
            spikes[int(neuron.group_name.split('_')[-1])][t, neuron.neuron_offset] = 1
    return spikes


def sanafe_lif_trace(a_in, *, du, dv, vth, bias_mant=0, bias_exp=0, reset_interval=0,
                     reset_offset=0, initial_v=0):
    """Run ``cuba_lif_int24`` on the candidate's ``loihi_cuba_lif`` unit.

    Same arguments and return as ``lava_lif_trace``. ``a_in`` must lie within
    ``[-ACTIVATION_BOUND, ACTIVATION_BOUND]``. ``reset_interval == 0`` disables
    the periodic reset.
    """
    from sanafe.loihi2 import load_loihi2_candidate

    a = _activations(a_in, -ACTIVATION_BOUND, ACTIVATION_BOUND)
    r = _registers(du, dv, vth, bias_mant, bias_exp, reset_interval, reset_offset, initial_v)
    steps, n = a.shape
    arch = load_loihi2_candidate()
    net = sanafe.Network()
    group = net.create_neuron_group('layer_0', n, log_spikes=True, log_potential=True)
    for offset, neuron in enumerate(group):
        neuron.set_attributes(soma_hw_name=LIF_SOMA_UNIT,
                              model_attributes={**r, 'currents': a[:, offset].tolist()})
        neuron.map_to_core(arch.tiles[0].cores[0])
    chip = sanafe.SpikingChip(arch)
    chip.load(net)
    result = chip.sim(steps, spike_trace=True, potential_trace=True)
    return {'s': _spike_matrix(result['spike_trace'], [n])[0],
            'v': np.asarray(result['potential_trace'], dtype=np.int64)}


def lif_chain_to_sanafe(layers, connections, a_in, arch, placements):
    """Return ``(Network, manifest)`` for a fresh Lava LIF/Dense chain.

    ``layers`` are one-dimensional Lava ``LIF`` or ``LIFReset`` Processes and
    ``connections`` the plain ``Dense`` between each adjacent pair. ``a_in`` of
    shape ``[steps, layers[0] size]`` is the activation of the first layer at
    each update (see the module docstring for the alignment). Dense input must
    be binary (zero message bits) and the weight exponent nonnegative. The
    effective weights are the clipped and truncated weights of Lava's
    fixed-point Dense, scaled by ``2**weight_exp``, and every possible binary
    fan-in sum must lie within the accepted activation range. Initial currents
    must be zero, because the neuron starts and resets its current at zero.

    Export before attaching a source or observers. All checks precede network
    construction. The manifest records the numerical and resource assumptions.
    """
    from lava.proc.lif.process import LIF, LIFReset
    from lava.proc.dense.process import Dense
    from lava.utils.weightutils import (SignMode, determine_sign_mode,
                                       clip_weights, truncate_weights)
    from sanafe.loihi2 import (candidate_profile, validate_chain_resources,
                               validate_candidate_architecture)

    if not layers or len(connections) != len(layers) - 1:
        raise ValueError('Expected one Dense connection between each pair of layers')
    if len(placements) != len(layers):
        raise ValueError('Expected one placement per layer')
    processes = [*layers, *connections]
    if len({id(p) for p in processes}) != len(processes):
        raise ValueError('Each Process must appear once')
    for layer in layers:
        if type(layer) not in (LIF, LIFReset) or len(layer.v.shape) != 1:
            raise ValueError('Only one-dimensional LIF and LIFReset layers are supported')
    for process in processes:
        if process.runtime is not None or process._is_compiled:
            raise ValueError('Export fresh, uncompiled Processes only')
    if layers[0].a_in.in_connections:
        raise ValueError('Supply first-layer activations before attaching an input')
    for index, dense in enumerate(connections):
        if type(dense) is not Dense:
            raise ValueError('Only plain Dense connections are supported')
        if (dense.s_in.in_connections != [layers[index].s_out] or
                layers[index].s_out.out_connections != [dense.s_in] or
                dense.a_out.out_connections != [layers[index + 1].a_in] or
                layers[index + 1].a_in.in_connections != [dense.a_out]):
            raise ValueError('Only direct, unbranched chains are supported')
    if layers[-1].s_out.out_connections:
        raise ValueError('Attach observers after export')

    sizes = [layer.v.shape[0] for layer in layers]
    a = _activations(a_in, -ACTIVATION_BOUND, ACTIVATION_BOUND)
    if a.shape[1] != sizes[0]:
        raise ValueError('a_in must have one column per first-layer neuron')
    states = []
    for layer in layers:
        shape = layer.v.shape
        if np.any(_integers(layer.u.init, 'initial current', *_INT32, shape)):
            raise ValueError('Nonzero initial current is unsupported')
        if type(layer) is LIFReset:
            interval = _scalar_integer(layer.proc_params['reset_interval'],
                                       'reset_interval', 1, _INT32[1])
            offset = _scalar_integer(layer.proc_params['reset_offset'],
                                     'reset_offset', 0, _INT32[1])
        else:
            interval, offset = 0, 0
        states.append({
            'du': _scalar_integer(layer.du.init, 'du', 0, 4095),
            'dv': _scalar_integer(layer.dv.init, 'dv', 0, 4095),
            'vth': _scalar_integer(layer.vth.init, 'vth', 0, (1 << 17) - 1),
            'bias_mant': _integers(layer.bias_mant.init, 'bias_mant',
                                   -(1 << 12), (1 << 12) - 1, shape),
            'bias_exp': _integers(layer.bias_exp.init, 'bias_exp', 0, 7, shape),
            'initial_v': _integers(layer.v.init, 'initial voltage',
                                   -(1 << 23), (1 << 23) - 1, shape),
            'reset_interval': interval, 'reset_offset': offset,
        })
    effective, weight_specs = [], []
    for index, dense in enumerate(connections):
        buffer = _integers(dense.a_buff.init, 'initial Dense buffer', *_INT32,
                           (sizes[index + 1],))
        if np.any(buffer):
            raise ValueError('Nonzero initial Dense buffer is unsupported')
        if _integers(dense.num_message_bits.init, 'message bits', 0, 0).size != 1:
            raise ValueError('message bits must be scalar')
        exponent = _scalar_integer(dense.proc_params.get('weight_exp', 0),
                                   'weight exponent', 0, 15)
        bits = _scalar_integer(dense.proc_params.get('num_weight_bits', 8), 'weight bits', 1, 8)
        weights = _integers(dense.weights.init, 'Dense weights', *_INT32)
        if weights.shape != (sizes[index + 1], sizes[index]):
            raise ValueError('Dense matrix shape does not match adjacent layers')
        mode = dense.proc_params.get('sign_mode')
        if mode is None:
            mode = determine_sign_mode(weights)
        if mode not in (SignMode.MIXED, SignMode.EXCITATORY, SignMode.INHIBITORY):
            raise ValueError('Unsupported Dense sign mode')
        # The same two operations as AbstractPyDenseModelBitAcc. A nonnegative exponent
        # left-shifts the summed activation, which equals shifting each binary-gated weight.
        transformed = truncate_weights(clip_weights(weights, mode, num_bits=8), mode, bits)
        transformed = _integers(np.asarray(transformed, dtype=np.int64) * (1 << exponent),
                                'effective weights', -(1 << 15), (1 << 15) - 1)
        lower = np.minimum(transformed, 0).sum(axis=1, dtype=np.int64)
        upper = np.maximum(transformed, 0).sum(axis=1, dtype=np.int64)
        if np.any(lower < -ACTIVATION_BOUND) or np.any(upper > ACTIVATION_BOUND):
            raise ValueError('binary fan-in can exceed the accepted LIF activation range')
        effective.append(transformed)
        weight_specs.append({'num_weight_bits': bits, 'sign_mode': mode.name,
                             'weight_exp': exponent, 'num_message_bits': 0,
                             'fanin_min': lower.tolist(), 'fanin_max': upper.tolist()})

    resources = validate_chain_resources(sizes, effective, placements)
    if len(arch.tiles) != 32 or any(len(t.cores) != 4 for t in arch.tiles):
        raise ValueError('Expected candidate 32-router/four-core topology')
    fingerprint = validate_candidate_architecture(arch)
    cores = []
    for tile_id, core_id in placements:
        core = arch.tiles[int(tile_id)].cores[int(core_id)]
        somas = [u for u in core.pipeline_hw if u.implements_soma and u.name == LIF_SOMA_UNIT]
        if len(somas) != 1 or somas[0].model_info.name != 'cuba_lif_int24':
            raise ValueError(f'Expected one {LIF_SOMA_UNIT} unit per selected core')
        for flag, model in [('implements_dendrite', 'accumulator_int'),
                            ('implements_synapse', 'current_based_int')]:
            units = [u for u in core.pipeline_hw if getattr(u, flag)]
            if len(units) != 1 or units[0].model_info.name != model:
                raise ValueError(f'Expected one {model} unit per selected core')
        if core.buffer_position != sanafe.BufferPosition.buffer_before_soma_unit:
            raise ValueError('Expected external pre-soma buffer')
        required = sum(size for size, place in zip(sizes, placements)
                       if tuple(place) == (tile_id, core_id))
        if required > core.max_neurons_supported:
            raise ValueError('Placement exceeds configured neuron capacity')
        cores.append(core)

    network = sanafe.Network()
    groups = []
    for index, (size, state, core) in enumerate(zip(sizes, states, cores)):
        group = network.create_neuron_group(f'layer_{index}', size,
                                            log_spikes=True, log_potential=True)
        for offset, neuron in enumerate(group):
            attrs = {name: (int(value[offset]) if isinstance(value, np.ndarray) else value)
                     for name, value in state.items()}
            if index == 0:
                attrs['currents'] = a[:, offset].tolist()
            neuron.set_attributes(soma_hw_name=LIF_SOMA_UNIT, model_attributes=attrs)
            neuron.map_to_core(core)
        groups.append(group)
    for index, weights in enumerate(effective):
        for post, row in enumerate(weights):
            for pre, weight in enumerate(row):
                groups[index][pre].connect_to_neuron(groups[index + 1][post],
                                                    {'weight': int(weight)})
    profile = candidate_profile()
    profile['allocation'] = resources['allocation'].copy()
    # The profile's default semantics describe the IF units. This chain runs the LIF unit.
    profile['numerical_profile'] = 'qcfs-lif-int24-v1'
    profile['semantics'] = {
        'input_min': -ACTIVATION_BOUND, 'input_max': ACTIVATION_BOUND, 'input_shift_bits': 6,
        'current_bits': 24, 'current_overflow': 'single wrap by 2**24 into (-2**23, 2**23]',
        'decay_bits': 12, 'current_decay_offset': 1, 'voltage_decay_offset': 0,
        'voltage_bits': 24, 'voltage_clip': 'symmetric, [-(2**23 - 1), 2**23 - 1]',
        'threshold_bits': 17, 'threshold_shift_bits': 6, 'threshold_comparison': '>',
        'reset': 'voltage to zero', 'periodic_reset': 'u and v to zero, before the update',
        'bias': 'bias_mant * 2**bias_exp', 'weight_signed': True,
        'effective_weight_transport_bits': 16, 'accumulation_bits': 32,
        'accumulation_signed': True, 'accumulation_overflow': 'reject', 'events': 'binary',
        'edge_delay_steps': 1}
    manifest = {
        'schema_version': 1,
        'execution_profile': 'qcfs-lif-int24-v1',
        'soma_unit': LIF_SOMA_UNIT,
        'architecture_profile': profile, 'architecture_sha256': fingerprint,
        'hardware_validated': False, 'input_transfer_cost_included': False,
        'lava_models': [('lava.proc.lif.models.PyLifResetModelBitAcc' if type(layer) is LIFReset
                         else 'lava.proc.lif.models.PyLifModelBitAcc') for layer in layers]
                       + ['lava.proc.dense.models.PyDenseModelBitAcc'],
        'architecture_instance': repr(arch),
        'input': 'external activation per update (a_in[t] at update t)',
        'updates': a.shape[0],
        'dense_delay_updates': 1, 'layer_sizes': sizes,
        'placements': [[int(t), int(c)] for t, c in placements],
        'layers': [{name: (value.tolist() if isinstance(value, np.ndarray) else value)
                    for name, value in state.items()} for state in states],
        'a_in': a.tolist(), 'weights': weight_specs,
        'effective_weights': [w.tolist() for w in effective],
        'resource_report': resources,
    }
    return network, manifest
