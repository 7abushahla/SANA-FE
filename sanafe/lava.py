# Copyright (c) 2026 - The University of Texas at Austin
#  This work was produced under contract #2317831 to National Technology and
#  Engineering Solutions of Sandia, LLC which is under contract
#  No. DE-NA0003525 with the U.S. Department of Energy.
# Implemented by Lance Lui as part of the capstone senior design project
"""
Lava to SANA-FE Backend

This module converts LAVA processes or serialization
object to the SNN representation runnable on SANA-FE

keep this file structure-
lava
├── src
│   ├── lava
│   │   ├── utils
│   │   │   ├── sanafe.py
│   │   │   │...
│   │   │...
│   │...
│...
SANAFE
├── sim.py
│...
"""
# As part of this backend, we have to access private Lava class details
# pylint: disable=protected-access

import lava
import sanafe


# Structural keys from Lava proc_params that describe the process shape
# rather than neuron model behavior. These are consumed during network
# construction and should NOT be forwarded as model_attributes.
_STRUCTURAL_KEYS = {"shape"}


def _extract_params(process):
    """Extract shape and model attributes from a Lava process.

    Splits ``proc_params._parameters`` into structural information (the
    neuron count derived from *shape*) and everything else, which is
    passed through to SANA-FE as ``model_attributes``.

    Args:
        process: A Lava ``AbstractProcess`` instance.

    Returns:
        tuple: (neuron_count, model_attributes_dict)
    """
    params = process.proc_params._parameters
    dim = params["shape"]
    neuron_count = dim[0] * (dim[1] if len(dim) > 1 else 1)

    model_attributes = {
        k: v for k, v in params.items() if k not in _STRUCTURAL_KEYS
    }
    return neuron_count, model_attributes


def _greedy_map_to_arch(network, arch):
    """Map every neuron in *network* to cores in *arch* using a greedy policy.

    Neurons are assigned to cores in the order they appear across groups.
    Each core is filled up to its ``max_neurons_supported`` limit before
    moving on to the next core.

    Args:
        network (sanafe.Network): The network whose neurons will be mapped.
        arch (sanafe.Architecture): Target architecture (e.g. from
            ``sanafe.load_loihi()``).

    Raises:
        RuntimeError: If the architecture runs out of cores before all
            neurons have been placed.
    """
    # Build a flat list of all cores across all tiles
    all_cores = []
    for tile in arch.tiles:
        for core in tile.cores:
            all_cores.append(core)

    core_idx = 0
    neurons_on_current_core = 0
    max_neurons = all_cores[core_idx].max_neurons_supported \
        if hasattr(all_cores[core_idx], "max_neurons_supported") else 1024

    for group_name in network.groups:
        group = network.groups[group_name]
        for neuron in group:
            if neurons_on_current_core >= max_neurons:
                core_idx += 1
                if core_idx >= len(all_cores):
                    raise RuntimeError(
                        f"Architecture has only {len(all_cores)} cores, "
                        f"which cannot hold all neurons in the network."
                    )
                neurons_on_current_core = 0
                max_neurons = (
                    all_cores[core_idx].max_neurons_supported
                    if hasattr(all_cores[core_idx], "max_neurons_supported")
                    else 1024
                )
            neuron.map_to_core(all_cores[core_idx])
            neurons_on_current_core += 1


def serial_to_sanafe(
    filename,
    arch=None,
    save_path="converted_network.yaml",
):
    """Load a serialized Lava process file and convert it to a SANA-FE network.

    If the serialized object is a single ``AbstractProcess``, it is
    forwarded to :func:`process_to_sanafe`.  Otherwise each element in
    the serialized list is turned into a neuron group and consecutive
    groups are connected pairwise.

    Args:
        filename (str): Path to the serialized Lava process file.
        arch (sanafe.Architecture, optional): Target architecture for
            hardware mapping.  Defaults to ``sanafe.load_loihi()``.
        save_path (str, optional): Where to save the resulting SANA-FE
            network.  Defaults to ``"converted_network.yaml"``.

    Returns:
        sanafe.Network: The constructed (and mapped) network.
    """
    p = lava.utils.serialization.load(filename)

    # Single process shortcut
    if isinstance(p[0], lava.magma.core.process.process.AbstractProcess):
        return process_to_sanafe(p[0], arch=arch, save_path=save_path)

    if arch is None:
        arch = sanafe.load_loihi()

    network = sanafe.Network()
    groups = []

    for i, proc in enumerate(p[0]):
        neuron_count, model_attrs = _extract_params(proc)
        group = network.create_neuron_group(
            f"layer_{i}",
            neuron_count,
            model_attributes=model_attrs,
        )
        groups.append(group)

        # Connect to the previous group (sequential topology)
        if len(groups) > 1:
            groups[-2].connect_neurons_dense(groups[-1], {})

    _greedy_map_to_arch(network, arch)
    network.save(save_path)
    return network


def process_to_sanafe(
    process,
    arch=None,
    save_path="converted_network.yaml",
):
    """Convert a live Lava ``AbstractProcess`` to a SANA-FE network.

    The process shape is interpreted as (*rows*, *cols*); each column
    becomes a separate neuron group and consecutive groups are connected.

    Args:
        process (AbstractProcess): A Lava process instance.
        arch (sanafe.Architecture, optional): Target architecture for
            hardware mapping.  Defaults to ``sanafe.load_loihi()``.
        save_path (str, optional): Where to save the resulting SANA-FE
            network.  Defaults to ``"converted_network.yaml"``.

    Returns:
        sanafe.Network: The constructed (and mapped) network.
    """
    if arch is None:
        arch = sanafe.load_loihi()

    network = sanafe.Network()
    params = process.proc_params._parameters
    dim = params["shape"]
    rows = dim[0]
    cols = dim[1] if len(dim) > 1 else 1

    model_attrs = {
        k: v for k, v in params.items() if k not in _STRUCTURAL_KEYS
    }

    groups = []
    for col_idx in range(cols):
        group = network.create_neuron_group(
            f"group_{col_idx}",
            rows,
            model_attributes=model_attrs,
        )
        groups.append(group)

        if len(groups) > 1:
            groups[-2].connect_neurons_dense(groups[-1], {})

    _greedy_map_to_arch(network, arch)
    network.save(save_path)
    return network


def qcfs_chain_to_sanafe(layers, connections, currents, arch, placements):
    """Export a fresh, explicitly connected QCFSIF/Dense float chain.

    This restricted bridge requires the optional Lava fork providing QCFSIF.
    ``currents`` is [first-layer neurons, updates], including any drain steps.
    The first input must be unconnected. Every other input and output must
    form exactly the supplied chain. Observer ports can be attached afterward.
    ``placements`` supplies one (tile, core) pair per layer. The architecture
    must use integrate_fire_float32 somas and a buffer before each soma.

    Dense weights are copied as configured, including any threshold scaling.
    Edges have one logical update of delay. This is a numerical export, not a
    Loihi compiler. Accumulation uses SANA-FE's double-precision dendrite, so
    arbitrary floating-point Dense reductions are not guaranteed bit-identical.
    External current injection has no modeled host/encoder transfer cost.
    """
    import numpy as np
    from lava.proc.dense.process import Dense
    from lava.proc.qcfs import QCFSIF

    if not layers or len(connections) != len(layers) - 1:
        raise ValueError('Expected one Dense connection between each pair of layers')
    if len(placements) != len(layers):
        raise ValueError('Expected one placement per layer')
    if len({id(p) for p in [*layers, *connections]}) != len(layers) + len(connections):
        raise ValueError('Each Process must appear once')
    for layer in layers:
        if type(layer) is not QCFSIF or len(layer.v.shape) != 1:
            raise ValueError('Only one-dimensional QCFSIF layers are supported')
    for process in [*layers, *connections]:
        if process.runtime is not None or process._is_compiled:
            raise ValueError('Export fresh, uncompiled Processes only')
    if layers[0].a_in.in_connections:
        raise ValueError('Supply first-layer currents explicitly, before attaching an input')
    for i, dense in enumerate(connections):
        if type(dense) is not Dense:
            raise ValueError('Only plain Dense connections are supported')
        if (dense.s_in.in_connections != [layers[i].s_out] or
                layers[i].s_out.out_connections != [dense.s_in] or
                dense.a_out.out_connections != [layers[i + 1].a_in] or
                layers[i + 1].a_in.in_connections != [dense.a_out]):
            raise ValueError('Only direct, unbranched chains are supported')
    if layers[-1].s_out.out_connections:
        raise ValueError('Attach observers after export')
    currents = np.asarray(currents, dtype=np.float32)
    if (currents.ndim != 2 or currents.shape[0] != layers[0].v.shape[0]
            or currents.shape[1] < 1 or not np.isfinite(currents).all()):
        raise ValueError('currents must be a finite [neurons, updates] array')
    counts = {}
    cores = []
    for layer, (tile_id, core_id) in zip(layers, placements):
        if not (0 <= tile_id < len(arch.tiles)):
            raise ValueError('Tile index outside architecture')
        tile = arch.tiles[tile_id]
        if not (0 <= core_id < len(tile.cores)):
            raise ValueError('Core index outside tile')
        core = tile.cores[core_id]
        somas = [unit for unit in core.pipeline_hw if unit.implements_soma]
        dendrites = [unit for unit in core.pipeline_hw if unit.implements_dendrite]
        synapses = [unit for unit in core.pipeline_hw if unit.implements_synapse]
        if (not somas or somas[0].model_info.name != 'integrate_fire_float32'
                or core.buffer_position != sanafe.BufferPosition.buffer_before_soma_unit):
            raise ValueError('Expected float32 IF soma with an external pre-soma buffer')
        if (not dendrites or dendrites[0].model_info.name != 'accumulator'
                or not synapses or synapses[0].model_info.name != 'current_based'):
            raise ValueError('Expected accumulator dendrite and current_based synapse')
        key = (tile_id, core_id)
        counts[key] = counts.get(key, 0) + layer.v.shape[0]
        if counts[key] > core.max_neurons_supported:
            raise ValueError('Placement exceeds core neuron capacity')
        cores.append(core)
    network = sanafe.Network()
    groups = []
    for index, (layer, core) in enumerate(zip(layers, cores)):
        group = network.create_neuron_group(
            f'layer_{index}', layer.v.shape[0], log_spikes=True, log_potential=True)
        values = {name: np.broadcast_to(getattr(layer, name).init, layer.v.shape)
                  for name in ('threshold', 'bias', 'v')}
        for offset, neuron in enumerate(group):
            attributes = {'threshold': float(values['threshold'][offset]),
                          'bias': float(values['bias'][offset]),
                          'initial_voltage': float(values['v'][offset])}
            if index == 0:
                attributes['currents'] = currents[offset].tolist()
            neuron.set_attributes(model_attributes=attributes)
            neuron.map_to_core(core)
        groups.append(group)
    for index, dense in enumerate(connections):
        weights = np.asarray(dense.weights.init, dtype=np.float64)
        if weights.shape != (len(groups[index + 1]), len(groups[index])):
            raise ValueError('Dense matrix shape does not match adjacent layers')
        if not np.isfinite(weights).all():
            raise ValueError('Dense weights must be finite')
        for post, row in enumerate(weights):
            for pre, weight in enumerate(row):
                # Keep zero-weight connections, matching the declared dense graph.
                groups[index][pre].connect_to_neuron(
                    groups[index + 1][post], {'weight': float(weight)})
    return network


# Keep the integer execution contract separate from the legacy and float paths.
from sanafe.lava_integer import qcfs_fixed_chain_to_sanafe
from sanafe.lava_conv_integer import qcfs_fixed_conv_to_sanafe
