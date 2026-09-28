"""Restricted integer Lava export with an explicit execution specification.

The numerical program is a CPU reference candidate. It is not a verified
Loihi 2 instruction program. Model allocation assumptions are reported.
"""
import numbers
import numpy as np
import sanafe


def _integers(value, name, low, high, shape=None):
    data = np.asarray(value)
    if not np.issubdtype(data.dtype, np.integer):
        raise ValueError(f'{name} must contain integers')
    if np.any(data < low) or np.any(data > high):
        raise ValueError(f'{name} outside declared range [{low}, {high}]')
    if shape is not None:
        try:
            data = np.broadcast_to(data, shape)
        except ValueError as exc:
            raise ValueError(f'{name} has incompatible shape') from exc
    return data.astype(np.int64, copy=True)


def _scalar_integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f'{name} must be an integer')
    if not low <= value <= high:
        raise ValueError(f'{name} outside supported range')
    return int(value)


def qcfs_fixed_chain_to_sanafe(layers, connections, currents, arch, placements,
                              *, reserved_cores=(), allocation=None):
    """Return ``(Network, manifest)`` for a fresh QCFSIFFixed/Dense chain.

    Binary Dense, zero weight exponent, 1..8 weight bits are supported. Lava's
    clipping and truncation are applied before mapping. Every possible binary
    fan-in sum must fit the IF input's signed16 range. The target must use the
    candidate integer pipeline and an external pre-soma buffer. No I/O costs
    are attributed to the direct numerical current injection.

    All checks precede network construction. The manifest is required when
    interpreting results and records the numerical and resource assumptions.
    """
    from lava.proc.qcfs import QCFSIFFixed
    from lava.proc.dense.process import Dense
    from lava.utils.weightutils import (SignMode, determine_sign_mode,
                                       clip_weights, truncate_weights)
    from sanafe.loihi2 import (audit_unit_scale_signed8_weights, candidate_profile,
                               validate_chain_resources,
                               validate_candidate_architecture)

    if not layers or len(connections) != len(layers) - 1:
        raise ValueError('Expected one Dense connection between each pair of layers')
    if len(placements) != len(layers):
        raise ValueError('Expected one placement per layer')
    processes = [*layers, *connections]
    if len({id(p) for p in processes}) != len(processes):
        raise ValueError('Each Process must appear once')
    for layer in layers:
        if type(layer) is not QCFSIFFixed or len(layer.v.shape) != 1:
            raise ValueError('Only one-dimensional QCFSIFFixed layers are supported')
    for process in processes:
        if process.runtime is not None or process._is_compiled:
            raise ValueError('Export fresh, uncompiled Processes only')
    if layers[0].a_in.in_connections:
        raise ValueError('Supply first-layer currents before attaching an input')
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
    currents = _integers(currents, 'currents', -(1 << 15), (1 << 15) - 1)
    if currents.ndim != 2 or currents.shape[0] != sizes[0] or currents.shape[1] < 1:
        raise ValueError('currents must be a nonempty [neurons, updates] array')
    states = []
    for layer in layers:
        theta = _integers(layer.threshold.init, 'threshold', 2, (1 << 23)-1, layer.v.shape)
        if np.any(theta % 2):
            raise ValueError('threshold must be even')
        start = _integers(layer.valid_start.init, 'valid_start', 0,
                          (1 << 31)-2)
        stop = _integers(layer.valid_stop.init, 'valid_stop', 1,
                         (1 << 31)-1)
        if start.size != 1 or stop.size != 1 or start.item() >= stop.item():
            raise ValueError('valid update window must be scalar [start, stop)')
        states.append({
            'threshold': theta,
            'bias': _integers(layer.bias.init, 'bias', -(1 << 15), (1 << 15)-1, layer.v.shape),
            'initial_voltage': _integers(layer.v.init, 'initial voltage', -(1 << 23),
                                         (1 << 23)-1, layer.v.shape),
            'valid_start': np.full(layer.v.shape, int(start.item()), dtype=np.int64),
            'valid_stop': np.full(layer.v.shape, int(stop.item()), dtype=np.int64),
        })
    effective, weight_specs = [], []
    for index, dense in enumerate(connections):
        buffer = _integers(dense.a_buff.init, 'initial Dense buffer',
                           -(1 << 15), (1 << 15)-1, (sizes[index+1],))
        if np.any(buffer):
            raise ValueError('Nonzero initial Dense buffer is unsupported')
        messages = _integers(dense.num_message_bits.init, 'message bits', 0, 0)
        if messages.size != 1:
            raise ValueError('message bits must be scalar')
        _scalar_integer(dense.proc_params.get('weight_exp', 0), 'weight exponent', 0, 0)
        bits = _scalar_integer(dense.proc_params.get('num_weight_bits', 8), 'weight bits', 1, 8)
        weights = _integers(dense.weights.init, 'Dense weights', -(1 << 31), (1 << 31)-1)
        if weights.shape != (sizes[index + 1], sizes[index]):
            raise ValueError('Dense matrix shape does not match adjacent layers')
        mode = dense.proc_params.get('sign_mode')
        if mode is None:
            mode = determine_sign_mode(weights)
        if mode not in (SignMode.MIXED, SignMode.EXCITATORY, SignMode.INHIBITORY):
            raise ValueError('Unsupported Dense sign mode')
        # Both operations match AbstractPyDenseModelBitAcc, including mixed-sign
        # truncation. Binary fan-in is bounded before execution, avoiding wraps.
        transformed = truncate_weights(clip_weights(weights, mode, num_bits=8), mode, bits)
        transformed = _integers(transformed, 'effective weights', -(1 << 15), (1 << 15)-1)
        lower = np.minimum(transformed, 0).sum(axis=1, dtype=np.int64)
        upper = np.maximum(transformed, 0).sum(axis=1, dtype=np.int64)
        if np.any(lower < -(1 << 15)) or np.any(upper > (1 << 15)-1):
            raise ValueError('binary fan-in can exceed the declared signed16 current range')
        effective.append(transformed)
        weight_specs.append({'num_weight_bits': bits, 'sign_mode': mode.name,
                             'weight_exp': 0, 'num_message_bits': 0,
                             'fanin_min': lower.tolist(), 'fanin_max': upper.tolist()})

    resources = validate_chain_resources(sizes, effective, placements,
                                         reserved_cores=reserved_cores, allocation=allocation)
    if len(arch.tiles) != 32 or any(len(t.cores) != 4 for t in arch.tiles):
        raise ValueError('Expected candidate 32-router/four-core topology')
    fingerprint = validate_candidate_architecture(arch)
    cores = []
    for tile_id, core_id in placements:
        core = arch.tiles[int(tile_id)].cores[int(core_id)]
        for flag, model in [('implements_soma', 'integrate_fire_int24'),
                            ('implements_dendrite', 'accumulator_int'),
                            ('implements_synapse', 'current_based_int')]:
            units = [u for u in core.pipeline_hw if getattr(u, flag)]
            if len(units) != 1 or units[0].model_info.name != model:
                raise ValueError(f'Expected one {model} unit per selected core')
        if core.buffer_position != sanafe.BufferPosition.buffer_before_soma_unit:
            raise ValueError('Expected external pre-soma buffer')
        # Capacity is also enforced by the C++ mapper. Check the selected
        # architecture here before constructing the graph.
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
            attrs = {name: int(values[offset]) for name, values in state.items()}
            if index == 0:
                attrs['currents'] = currents[offset].tolist()
            neuron.set_attributes(model_attributes=attrs)
            neuron.map_to_core(core)
        groups.append(group)
    for index, weights in enumerate(effective):
        for post, row in enumerate(weights):
            for pre, weight in enumerate(row):
                groups[index][pre].connect_to_neuron(groups[index + 1][post],
                                                    {'weight': int(weight)})
    profile = candidate_profile()
    profile['allocation'] = resources['allocation'].copy()
    windows = [[int(s['valid_start'].flat[0]), int(s['valid_stop'].flat[0])]
               for s in states]
    gated = any(start != 0 or stop != (1 << 31)-1 for start, stop in windows)
    manifest = {
        'schema_version': 1,
        'execution_profile': ('qcfs-if-int24-binary-windowed-v1' if gated
                              else 'qcfs-if-int24-binary-v1'),
        'architecture_profile': profile, 'architecture_sha256': fingerprint,
        'hardware_validated': False, 'input_transfer_cost_included': False,
        'lava_models': ['lava.proc.qcfs.models.PyQCFSIFFixed',
                        'lava.proc.dense.models.PyDenseModelBitAcc'],
        'architecture_instance': repr(arch),
        'input': 'external numeric current per update', 'updates': currents.shape[1],
        'valid_update_windows_zero_based': windows,
        'scheduling_status': 'candidate numerical retiming, not verified Loihi 2 hardware',
        'dense_delay_updates': 1, 'layer_sizes': sizes,
        'placements': [[int(t), int(c)] for t, c in placements],
        'initial_voltage': [s['initial_voltage'].tolist() for s in states],
        'thresholds': [s['threshold'].tolist() for s in states],
        'biases': [s['bias'].tolist() for s in states],
        'currents': currents.tolist(), 'weights': weight_specs,
        'effective_weights': [w.tolist() for w in effective],
        'unit_scale_signed8_report': audit_unit_scale_signed8_weights(effective),
        'resource_report': resources,
    }
    return network, manifest
