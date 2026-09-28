"""Restricted spatial QCFSIFFixed -> Lava Conv -> QCFSIFFixed export.

This bridge expands the convolution into individual synapses. It makes no
claim about Loihi 2 convolutional weight sharing, packing, or compiler fit.
"""
import numpy as np
import sanafe

from sanafe.lava_integer import _integers, _scalar_integer
from sanafe.loihi2 import (candidate_profile, validate_candidate_architecture,
                           validate_chain_resources)


def qcfs_fixed_conv_to_sanafe(first, conv, second, currents, arch, placements,
                              *, reserved_cores=(), allocation=None):
    """Export one fresh binary Conv edge between two spatial fixed IF layers.

    ``currents`` has shape ``first.v.shape + (updates,)``. The externally
    supplied current is applied to the first IF layer on every update.
    Conv output reaches the second IF one numbered update after source spikes.
    The resource check treats expanded connectivity as a dense upper bound.
    """
    from lava.proc.qcfs import QCFSIFFixed
    from lava.proc.conv.process import Conv
    from lava.proc.conv.utils import conv_to_sparse

    if type(first) is not QCFSIFFixed or type(second) is not QCFSIFFixed:
        raise ValueError('Expected exactly two QCFSIFFixed Processes')
    if type(conv) is not Conv:
        raise ValueError('Expected a plain Lava Conv Process')
    if len(first.v.shape) != 3 or len(second.v.shape) != 3:
        raise ValueError('Both IF layers must have three-dimensional WHC shape')
    if len(placements) != 2:
        raise ValueError('Expected two (tile, core) placements')
    for process in (first, conv, second):
        if process.runtime is not None or process._is_compiled:
            raise ValueError('Export fresh, uncompiled Processes only')
    if (first.a_in.in_connections or
            first.s_out.out_connections != [conv.s_in] or
            conv.s_in.in_connections != [first.s_out] or
            conv.a_out.out_connections != [second.a_in] or
            second.a_in.in_connections != [conv.a_out] or
            second.s_out.out_connections):
        raise ValueError('Only a direct, unbranched Conv chain is supported')
    if tuple(conv.input_shape) != tuple(first.v.shape) or tuple(conv.output_shape) != tuple(second.v.shape):
        raise ValueError('Conv input/output shape must match IF layers')

    message_bits = _integers(conv.num_message_bits.init, 'message bits', 0, 0)
    if message_bits.size != 1:
        raise ValueError('message bits must be scalar')
    _scalar_integer(conv.proc_params.get('weight_exp', 0), 'weight exponent', 0, 0)
    _scalar_integer(conv.proc_params.get('num_weight_bits', 8), 'weight bits', 8, 8)
    groups = int(np.asarray(conv.groups.init).item())
    if groups != 1 or any(int(v) != 1 for v in conv.dilation.init):
        raise ValueError('Only groups=1 and dilation=1 are supported')
    buffer = _integers(conv.a_buf.init, 'initial Conv buffer', -(1 << 15),
                       (1 << 15)-1, second.v.shape)
    if np.any(buffer):
        raise ValueError('Nonzero initial Conv buffer is unsupported')
    weight = _integers(conv.weight.init, 'Conv weights', -128, 127)
    if weight.shape != (second.v.shape[2], *tuple(conv.kernel_size.init), first.v.shape[2]):
        raise ValueError('Conv weight shape differs from kernel and channel dimensions')
    currents = _integers(currents, 'currents', -(1 << 15), (1 << 15)-1)
    if currents.ndim != 4 or tuple(currents.shape[:3]) != tuple(first.v.shape) or currents.shape[-1] < 1:
        raise ValueError('currents must have first-layer WHC shape plus updates')

    states = []
    for layer in (first, second):
        shape = layer.v.shape
        theta = _integers(layer.threshold.init, 'threshold', 2, (1 << 23)-1, shape)
        if np.any(theta % 2):
            raise ValueError('threshold must be even')
        start = _integers(layer.valid_start.init, 'valid_start', 0, (1 << 31)-2)
        stop = _integers(layer.valid_stop.init, 'valid_stop', 1, (1 << 31)-1)
        if start.size != 1 or stop.size != 1 or start.item() >= stop.item():
            raise ValueError('valid update window must be scalar [start, stop)')
        states.append({
            'threshold': theta.ravel(),
            'bias': _integers(layer.bias.init, 'bias', -(1 << 15), (1 << 15)-1, shape).ravel(),
            'initial_voltage': _integers(layer.v.init, 'initial voltage', -(1 << 23),
                                         (1 << 23)-1, shape).ravel(),
            'valid_start': np.full(int(np.prod(shape)), int(start.item()), dtype=np.int64),
            'valid_stop': np.full(int(np.prod(shape)), int(stop.item()), dtype=np.int64),
        })

    dst, src, edges = conv_to_sparse(tuple(first.v.shape), tuple(second.v.shape),
                                     weight, tuple(int(x) for x in conv.stride.init),
                                     tuple(int(x) for x in conv.padding.init), (1, 1), 1)
    edges = _integers(edges, 'expanded Conv weights', -128, 127)
    sizes = [int(np.prod(layer.v.shape)) for layer in (first, second)]
    matrix = np.zeros((sizes[1], sizes[0]), dtype=np.int64)
    np.add.at(matrix, (dst, src), edges)
    if np.any(matrix < -(1 << 15)) or np.any(matrix > (1 << 15)-1):
        raise ValueError('Expanded Conv weights exceed signed16 range')
    lower = np.minimum(matrix, 0).sum(axis=1, dtype=np.int64)
    upper = np.maximum(matrix, 0).sum(axis=1, dtype=np.int64)
    if np.any(lower < -(1 << 15)) or np.any(upper > (1 << 15)-1):
        raise ValueError('binary Conv fan-in can exceed signed16 current range')
    resources = validate_chain_resources(sizes, [matrix], placements,
                                         reserved_cores=reserved_cores, allocation=allocation)
    resources['storage_bound'] = 'dense expansion upper bound; actual emitted edges are spatial'
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
        cores.append(core)

    network = sanafe.Network()
    groups = []
    for index, (size, state, core) in enumerate(zip(sizes, states, cores)):
        group = network.create_neuron_group(f'layer_{index}', size,
                                            log_spikes=True, log_potential=True)
        for offset, neuron in enumerate(group):
            attrs = {name: int(values[offset]) for name, values in state.items()}
            if index == 0:
                attrs['currents'] = currents.reshape(size, -1)[offset].tolist()
            neuron.set_attributes(model_attributes=attrs)
            neuron.map_to_core(core)
        groups.append(group)
    for dest, source, value in zip(dst, src, edges):
        groups[0][int(source)].connect_to_neuron(groups[1][int(dest)],
                                                {'weight': int(value)})
    profile = candidate_profile()
    profile['allocation'] = resources['allocation'].copy()
    windows = [[int(s['valid_start'][0]), int(s['valid_stop'][0])] for s in states]
    manifest = {
        'schema_version': 1, 'execution_profile': 'qcfs-if-int24-binary-conv-expanded-v1',
        'architecture_profile': profile, 'architecture_sha256': fingerprint,
        'hardware_validated': False, 'input_transfer_cost_included': False,
        'lava_models': ['lava.proc.qcfs.models.PyQCFSIFFixed',
                        'lava.proc.conv.models.PyConvModelFixed'],
        'input': 'external numeric current per update', 'updates': currents.shape[-1],
        'valid_update_windows_zero_based': windows,
        'scheduling_status': 'candidate numerical retiming, not verified Loihi 2 hardware',
        'conv_delay_updates': 1, 'layer_sizes': sizes,
        'layer_shapes_whc': [[int(x) for x in first.v.shape],
                             [int(x) for x in second.v.shape]],
        'placements': [[int(t), int(c)] for t, c in placements],
        'initial_voltage': [s['initial_voltage'].tolist() for s in states],
        'thresholds': [s['threshold'].tolist() for s in states],
        'biases': [s['bias'].tolist() for s in states],
        'currents': currents.reshape(sizes[0], -1).tolist(),
        'conv': {'weight_layout': 'Cout, W, H, Cin',
                 'kernel_size_wh': [int(x) for x in conv.kernel_size.init],
                 'stride_wh': [int(x) for x in conv.stride.init],
                 'padding_wh': [int(x) for x in conv.padding.init],
                 'expanded_synapses': len(edges),
                 'lava_weight_precision_bits': 8,
                 'physical_weight_packing_verified': False},
        'resource_report': resources,
    }
    return network, manifest
