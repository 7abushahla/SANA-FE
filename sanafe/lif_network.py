"""LIF networks with convolutions from a netx file, in SANA-FE and in Lava.

``netx_to_sanafe`` builds a SANA-FE network from a ``NetxNetwork`` (the reader
``sanafe.netx``). Every layer with a neuron runs on the soma unit
``loihi_cuba_lif``, which implements the rule of Lava's bit-accurate fixed-point
LIF (``sanafe.lava_lif``). ``lava_run`` runs the same network in Lava under
``Loihi2SimCfg(select_tag='fixed_pt')``, built either by lava-dl's netx from the
file (``builder="netx"``) or from the ``NetxNetwork`` (``builder="direct"``).
``sanafe_run`` runs the simulator. Both return the spikes and voltages of every
layer in the same form, so that the two compare index by index. Equality with
Lava's fixed-point model is what the tests establish. Nothing here is
validated against Loihi 2 hardware.

Conventions
-----------
* Layers are the layers of the file that have a neuron, in file order. Flatten
  layers have none and only fix the order in which a dense layer reads the
  previous layer.
* Neurons of a layer are indexed in flattened (channels, height, width) order,
  the order of the reader. Every returned array has shape ``[steps, n]`` with
  that column order. Row ``t`` is the state after update ``t``.
* Where a neuron sits is a separate matter. By default the builder splits each
  layer in reader order. ``netx_to_sanafe`` also accepts a partition, which may
  order a layer as Lava does (width, height, channels) before it is sliced over
  the cores (``neuron_positions``). The indices above do not change with it.
* The input layer is a LIF layer like the others, driven only by its bias, as
  netx builds it (its input port is not connected). ``input_bias`` is written
  into its bias mantissa, with the layer's bias exponent, on both sides. It must
  have exactly the input layer's (channels, height, width) shape. Lava's own
  order is (width, height, channels), and a flat array states no order, so
  both are refused.
* A spike of layer ``k - 1`` at update ``t`` reaches layer ``k`` at update
  ``t + 1``. Lava's Conv and Dense send the activation of the previous step
  (``proc/conv/models.py`` line 38, ``proc/dense/models.py`` line 96). SANA-FE
  realizes the same delay of one update with the candidate's buffer before the
  soma unit (see ``sanafe.lava_lif``).
* A convolution and a dense layer are expanded into individual synapses that
  carry the effective weights of the reader. Synapses of weight zero add
  nothing to any sum and are not emitted. The manifest records both counts.
* The periodic reset zeroes current and voltage before the update of the
  one-based step ``t`` when ``t mod reset_interval`` equals the layer's
  ``resets_at``, and that step's input is still integrated, as in Lava's
  ``LIFReset``.

Reading Lava's state
--------------------
Lava's ``Var.get`` and ``Monitor.get_data`` transfer an array one element per
message, and a Monitor run of 40 steps on a layer of 76,032 neurons deadlocked.
``lava_run`` therefore uses neither. Each layer's voltage is tapped after every
step by Lava's ``VarWire`` (a RefPort read in the management phase) and each
layer's spikes by Lava's ``Extractor`` (an InPort on ``s_out``). Both hand whole
arrays to this process through a channel. The run proceeds in chunks of at most
16 steps. A step that delivers nothing for ``stall_seconds`` ends the run with
``TimeoutError``. A chunk that fails for any reason is not stopped through
Lava's ``stop()``, which has no time limit. Every child process that this
Python process started through ``multiprocessing`` is terminated instead, which
includes Lava's runtime processes and any other such child.
"""
import threading
import time
from dataclasses import asdict

import numpy as np
import sanafe

from sanafe.lava_integer import _integers
from sanafe.lava_lif import (ACTIVATION_BOUND, LIF_EXECUTION_PROFILE, LIF_SOMA_UNIT,
                             _registers, _scalar_integer, check_lif_architecture,
                             check_lif_core, lif_architecture_profile)
from sanafe.loihi2 import Allocation

_INT32_MAX = (1 << 31) - 1
_BIAS_RANGE = (-(1 << 12), (1 << 12) - 1)
_WEIGHT_RANGE = (-(1 << 15), (1 << 15) - 1)
# One core kept free, as in the thesis's existing Loihi 2 route
# (mixed-horizon-snn/mhsnn/loihi2/export.py, RESERVED).
RESERVED_CORES = ((31, 3),)
_CONNECT_CHUNK = 1 << 20


# ---------------------------------------------------------------------------
# Structure of the network


def _neuron_layers(network):
    """The layers with a neuron, each with the flags the builders need.

    Returns a list of dicts with ``layer``, ``kind``, ``shape``, ``size`` and
    ``flatten`` (a flatten layer precedes it). Refuses structures that netx does
    not build or that this module does not implement.
    """
    out = []
    flatten = False
    for index, layer in enumerate(network.layers):
        if layer.kind == "flatten":
            if not out or len(out[-1]["shape"]) != 3 or flatten:
                raise ValueError(f"layer {index}: a flatten layer must follow an input or "
                                 "convolution layer")
            flatten = True
            continue
        if layer.neuron is None:
            raise ValueError(f"layer {index} ({layer.kind}): no neuron, so it cannot run")
        if (layer.kind == "input") != (index == 0):
            raise ValueError(f"layer {index}: the input layer must be the first layer, and "
                             "only the first")
        if layer.kind == "conv" and (flatten or len(out[-1]["shape"]) != 3):
            raise ValueError(f"layer {index}: a convolution must follow an input or "
                             "convolution layer")
        if layer.kind == "dense" and len(out[-1]["shape"]) == 3 and not flatten:
            raise ValueError(f"layer {index}: a dense layer after a three-dimensional layer "
                             "needs a flatten layer")
        if layer.kind not in ("input", "conv", "dense"):
            raise ValueError(f"layer {index}: layer type '{layer.kind}' is not implemented")
        shape = tuple(int(x) for x in layer.shape)
        if layer.kind != "dense" and len(shape) != 3:
            raise ValueError(f"layer {index}: shape {shape} is not (channels, height, width)")
        out.append({"layer": layer, "kind": layer.kind, "shape": shape,
                    "size": int(np.prod(shape)), "flatten": flatten, "file_index": index})
        flatten = False
    if flatten:
        raise ValueError("the last layer is a flatten layer")
    if not out:
        raise ValueError("the network has no layer with a neuron")
    return out


def _reset(layer):
    """(interval, phase) of the periodic reset, (0, 0) when there is none."""
    interval = layer.neuron.get("reset_interval")
    if interval is None:
        return 0, 0
    interval = _scalar_integer(interval, "reset_interval", 1, _INT32_MAX)
    return interval, int(layer.resets_at)


def _registers_of(entry):
    """Neuron registers checked against the soma's ranges, with the messages of lava_lif."""
    neuron = entry["layer"].neuron
    interval, phase = _reset(entry["layer"])
    r = _registers(neuron["du"], neuron["dv"], neuron["vth"], 0, neuron["bias_exp"],
                   interval, phase, 0)
    return {key: r[key] for key in ("du", "dv", "vth", "bias_exp", "reset_interval",
                                    "reset_offset")}


def _biases(entry, input_bias):
    """Bias mantissa of every neuron of a layer, in flattened (c, h, w) order."""
    layer, size = entry["layer"], entry["size"]
    if entry["kind"] == "input":
        # The shape must be exactly (channels, height, width). A flat array, or any
        # other array of the right size, would be taken in an order the caller never
        # stated. Lava's own arrays are (width, height, channels), the tutorial's frame
        # (width, height, channels, 1), and both have the same number of entries.
        bias = np.asarray(input_bias)
        if bias.shape != entry["shape"]:
            raise ValueError(
                f"input_bias has shape {bias.shape}, expected the input layer's "
                f"(channels, height, width) = {entry['shape']}. Lava's own order is "
                "(width, height, channels), so a Lava or netx array must be transposed "
                "to (channels, height, width) first")
        return _integers(bias.reshape(-1), "bias_mant", *_BIAS_RANGE)
    if layer.bias is None:
        return np.zeros(size, dtype=np.int64)
    bias = _integers(np.asarray(layer.bias).reshape(-1), "bias_mant", *_BIAS_RANGE)
    if entry["kind"] == "conv":
        channels = entry["shape"][0]
        if bias.size != channels:
            raise ValueError(f"a convolution bias has {bias.size} entries, the layer "
                             f"{channels} channels")
        return np.repeat(bias, size // channels)
    if bias.size != size:
        raise ValueError(f"a dense bias has {bias.size} entries, the layer {size} neurons")
    return bias


def _conv_geometry(previous, entry):
    layer = entry["layer"]
    weights = np.asarray(layer.weights)
    c_in, height, width = previous["shape"]
    c_out, out_h, out_w = entry["shape"]
    if weights.ndim != 4 or weights.shape[:2] != (c_out, c_in):
        raise ValueError(f"layer {entry['file_index']}: convolution weights {weights.shape} "
                         f"do not map {c_in} channels to {c_out}")
    (sh, sw), (ph, pw) = layer.stride, layer.padding
    kh, kw = weights.shape[2:]
    expected = (c_out, (height + 2 * ph - kh) // sh + 1, (width + 2 * pw - kw) // sw + 1)
    if expected != entry["shape"]:
        raise ValueError(f"layer {entry['file_index']}: shape {entry['shape']} differs from "
                         f"the convolution's output {expected}")
    return (int(sh), int(sw)), (int(ph), int(pw))


def conv_synapses(in_shape, weights, stride, padding, out_shape):
    """Expand a convolution into ``(source, destination, weight)`` int64 arrays.

    Shapes are (channels, height, width), ``weights`` is (out, in, kernel height,
    kernel width) with the meaning of a cross-correlation, and ``stride`` and
    ``padding`` are (height, width), the conventions of ``sanafe.netx``. Indices
    are flattened (c, h, w). Every kernel tap whose input position lies inside
    the input gives one synapse, weight zero included. The index computation is
    vectorized over channels and positions, one kernel tap at a time.
    """
    c_in, height, width = in_shape
    c_out, out_h, out_w = out_shape
    weights = np.asarray(weights, dtype=np.int64)
    (sh, sw), (ph, pw) = stride, padding
    out_c = np.arange(c_out)[:, None, None]
    in_c = np.arange(c_in)[None, :, None]
    sources, destinations, values = [], [], []
    for a in range(weights.shape[2]):
        iy = np.arange(out_h) * sh - ph + a
        rows = np.flatnonzero((iy >= 0) & (iy < height))
        for b in range(weights.shape[3]):
            ix = np.arange(out_w) * sw - pw + b
            cols = np.flatnonzero((ix >= 0) & (ix < width))
            y, x = np.repeat(rows, cols.size), np.tile(cols, rows.size)
            if y.size == 0:
                continue
            src = (in_c * height + iy[y][None, None, :]) * width + ix[x][None, None, :]
            dst = (out_c * out_h + y[None, None, :]) * out_w + x[None, None, :]
            full = (c_out, c_in, y.size)
            sources.append(np.broadcast_to(src, full).reshape(-1))
            destinations.append(np.broadcast_to(dst, full).reshape(-1))
            values.append(np.broadcast_to(weights[:, :, a, b][:, :, None], full).reshape(-1))
    if not sources:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy()
    return (np.concatenate(sources), np.concatenate(destinations), np.concatenate(values))


def _connection(previous, entry):
    """The synapses into ``entry`` and their record, refused outside lava_lif's ranges."""
    layer = entry["layer"]
    # The reader's effective weights, refused when a negative exponent would lose bits.
    effective = _integers(layer.effective_weights(), "effective weights", *_WEIGHT_RANGE)
    record = {"src": None, "dst": None, "kind": entry["kind"],
              "effective_weights_shape": list(effective.shape),
              "weight_exp": int(layer.weight_exp or 0), "weight_bits": layer.weight_bits}
    if entry["kind"] == "conv":
        stride, padding = _conv_geometry(previous, entry)
        src, dst, weight = conv_synapses(previous["shape"], effective, stride, padding,
                                         entry["shape"])
        record.update(stride_hw=list(stride), padding_hw=list(padding))
    else:
        if effective.shape != (entry["size"], previous["size"]):
            raise ValueError(f"layer {entry['file_index']}: dense weights {effective.shape} do "
                             f"not map {previous['size']} neurons to {entry['size']}")
        dst, src = (axis.reshape(-1) for axis in np.indices(effective.shape))
        weight = effective.reshape(-1)
    # Every binary fan-in sum must lie within the accepted LIF activation range. The
    # float64 sums of bincount are exact, since every sum is far below 2**53.
    lower = np.bincount(dst, weights=np.minimum(weight, 0), minlength=entry["size"])
    upper = np.bincount(dst, weights=np.maximum(weight, 0), minlength=entry["size"])
    lower, upper = lower.astype(np.int64), upper.astype(np.int64)
    if np.any(lower < -ACTIVATION_BOUND) or np.any(upper > ACTIVATION_BOUND):
        raise ValueError('binary fan-in can exceed the accepted LIF activation range')
    keep = weight != 0
    record.update(synapses_expanded=int(weight.size), synapses_emitted=int(keep.sum()),
                  fanin_min=int(lower.min()), fanin_max=int(upper.max()))
    return src[keep], dst[keep], weight[keep], record


# ---------------------------------------------------------------------------
# Placement


def core_order(order):
    """Cores in the order layers take them, the reserved core left out.

    ``packed`` fills tile by tile in row-major (tile, core) order. ``spread``
    takes core 0 of every tile, then core 1 of every tile, and so on, which
    distributes the layers' cores over the mesh. The two definitions are those
    of ``_core_order`` in mixed-horizon-snn/mhsnn/loihi2/export.py, reimplemented
    here so that the fork does not import that repository. ``packed_reversed``
    has the sequence of ``packed``. It differs only in that the layers take their
    cores in reverse layer order (see ``_placement``), the core order that the
    ledger record of the published PilotNet run infers from its memory plots
    (readout on the first cores, input layer on the last). That order is an
    inference, not a printed fact.
    """
    tiles, per_tile = 32, 4
    if order in ("packed", "packed_reversed"):
        cores = [(t, c) for t in range(tiles) for c in range(per_tile)]
    elif order == "spread":
        cores = [(t, c) for c in range(per_tile) for t in range(tiles)]
    else:
        raise ValueError(f"order {order!r} is not 'packed', 'packed_reversed' or 'spread'")
    return [place for place in cores if place not in RESERVED_CORES]


def slice_sizes(size, cores):
    """Contiguous slices of near-equal size, larger slices first, sizes differing by at most one."""
    size, cores = int(size), int(cores)
    if cores < 1:
        raise ValueError("a layer needs at least one core")
    if cores > size:
        raise ValueError(f"a layer of {size} neurons is split over more cores ({cores}) "
                         "than it has neurons")
    base, extra = divmod(size, cores)
    return [base + 1] * extra + [base] * (cores - extra)


ORDERS = ("reader", "lava")


def neuron_positions(shape, kind, order):
    """Position of every neuron (indexed in reader order) in the ordering ``order``.

    ``reader`` keeps the flattened (channels, height, width) index. ``lava`` gives
    the index in Lava's flattened (width, height, channels) order, (x * H + y) * C
    + c, for input and convolution layers. A dense layer has one order.
    """
    if order not in ORDERS:
        raise ValueError(f"order {order!r} is not one of {ORDERS}")
    n = int(np.prod(shape))
    if order == "reader" or kind == "dense":
        return np.arange(n, dtype=np.int64)
    if len(shape) != 3:
        raise ValueError(f"shape {shape} is not (channels, height, width)")
    c, h, w = (int(x) for x in shape)
    cc, yy, xx = np.indices((c, h, w))
    return ((xx * h + yy) * c + cc).reshape(-1).astype(np.int64)


def _partition_layers(entries, counts, partition):
    """Per layer, the (order, slice sizes) of ``partition``, or of the default when it is None.

    The default is reader order with the near-equal slices of ``slice_sizes``. A
    stated partition needs one entry per layer with a neuron, each with an order
    of ``ORDERS`` and one positive slice size per core of that layer, summing to
    the layer's size. These are the rules of ``sanafe.loihi2_counts``.
    """
    if partition is None:
        return [("reader", slice_sizes(entry["size"], n)) for entry, n in zip(entries, counts)]
    layers = list(partition["layers"])
    if len(layers) != len(entries):
        raise ValueError(f"the partition needs one entry per layer with a neuron "
                         f"({len(entries)}), got {len(layers)}")
    out = []
    for k, (spec, entry, n) in enumerate(zip(layers, entries, counts)):
        if spec["order"] not in ORDERS:
            raise ValueError(f"layer {k}: order {spec['order']!r} is not one of {ORDERS}")
        sizes = [_scalar_integer(v, "slice size", 0, _INT32_MAX) for v in spec["sizes"]]
        if len(sizes) != n or any(v < 1 for v in sizes) or sum(sizes) != entry["size"]:
            raise ValueError(f"layer {k}: the partition needs one positive slice size per core "
                             f"({n}) summing to {entry['size']}")
        out.append((spec["order"], sizes))
    return out


def _placement(entries, layer_cores, order, partition=None):
    counts = list(layer_cores)
    if len(counts) != len(entries):
        raise ValueError(f"layer_cores needs one core count per layer with a neuron "
                         f"({len(entries)}), got {len(counts)}")
    counts = [_scalar_integer(c, "layer core count", 1, _INT32_MAX) for c in counts]
    available = core_order(order)
    if sum(counts) > len(available):
        raise ValueError(f"the placement needs {sum(counts)} cores, the candidate has "
                         f"{len(available)} free")
    layers = _partition_layers(entries, counts, partition)
    # Layers take consecutive runs of the sequence, the last layer first under
    # packed_reversed. Within a layer the slices keep their order. A slice's start
    # and stop are positions in the layer's neuron order.
    turn = range(len(entries))
    if order == "packed_reversed":
        turn = reversed(turn)
    placement, cursor = [None] * len(entries), 0
    for k in turn:
        sizes = layers[k][1]
        start, slices = 0, []
        for place, n in zip(available[cursor:cursor + counts[k]], sizes):
            slices.append({"tile": int(place[0]), "core": int(place[1]),
                           "start": start, "stop": start + n})
            start += n
        cursor += counts[k]
        placement[k] = slices
    return placement, [neuron_order for neuron_order, _ in layers]


def _slice_index(slices, positions):
    """Index of the slice that holds each neuron of a layer, from its position in the layer's order."""
    stops = np.asarray([s["stop"] for s in slices], dtype=np.int64)
    return np.searchsorted(stops, positions, side="right").astype(np.int64)


def _per_core(src, dst, src_slice, dst_slice, n_src_cores, n_dst_cores, n_src):
    """Synapses and distinct source neurons per destination core, routes per source core."""
    dcore = dst_slice[dst]
    synapses = np.bincount(dcore, minlength=n_dst_cores)
    pairs = np.unique(dcore * n_src + src)  # distinct (destination core, source neuron)
    pair_core, pair_src = pairs // n_src, pairs % n_src
    axons = np.bincount(pair_core, minlength=n_dst_cores)
    routes = np.bincount(src_slice[pair_src], minlength=n_src_cores)
    return synapses.tolist(), axons.tolist(), routes.tolist()


# ---------------------------------------------------------------------------
# SANA-FE


def _group_name(k):
    return f"layer_{k:02d}"


def netx_to_sanafe(network, input_bias, arch, layer_cores, *, order="packed", log=True,
                   partition=None):
    """Return ``(Network, manifest)`` for a netx network on the candidate architecture.

    ``input_bias`` is the integer bias mantissa of the input layer for one frame,
    an array of exactly the layer's (channels, height, width) shape. Any other
    shape is refused, a flat array included. ``layer_cores`` is the number of
    cores of each layer with a neuron, in order. The cores are taken from the
    sequence of ``core_order(order)``, layer by layer in layer order, or in
    reverse layer order for ``packed_reversed``. ``log`` records spikes and
    voltages of every neuron. Weights, fan-in, biases and registers outside what
    the soma and ``sanafe.lava_lif`` accept are refused with the messages of
    ``lava_lif``. All checks precede network construction.

    ``partition`` states how the neurons of each layer are divided over its
    cores, in the form ``sanafe.loihi2_counts`` accepts,
    ``{"name": ..., "layers": [{"order": "reader" or "lava", "sizes": [...]}]}``
    with one entry per layer with a neuron and one slice size per core of that
    layer. The neurons of a layer are ordered by ``neuron_positions`` under the
    entry's order, and slice ``j`` (the next ``sizes[j]`` neurons of that order)
    sits on the layer's ``j``-th core. On each core the neurons are mapped in that
    order. Neuron indices, spikes and voltages stay in reader order. With
    ``partition=None`` (the default) the neurons are split in reader order into
    contiguous slices of near-equal size (sizes differ by at most one, larger
    slices first), and the network and the manifest are those the builder made
    before it accepted a partition. A stated partition adds the key
    ``partition`` to the manifest and the key ``order`` to each layer of its
    ``placement``, whose ``start`` and ``stop`` are then positions in that order.
    """
    entries = _neuron_layers(network)
    registers = [_registers_of(entry) for entry in entries]
    biases = [_biases(entry, input_bias) for entry in entries]
    connections = []
    for k in range(1, len(entries)):
        src, dst, weight, record = _connection(entries[k - 1], entries[k])
        record.update(src=k - 1, dst=k)
        connections.append((src, dst, weight, record))
    placement, neuron_orders = _placement(entries, layer_cores, order, partition)
    fingerprint = check_lif_architecture(arch)
    cores = []
    for k, slices in enumerate(placement):
        cores.append([arch.tiles[s["tile"]].cores[s["core"]] for s in slices])
        for j, (s, core) in enumerate(zip(slices, cores[-1])):
            check_lif_core(core, s["stop"] - s["start"],
                           what=f"layer {k} slice {j} on core ({s['tile']}, {s['core']})")
    positions = [neuron_positions(entry["shape"], entry["kind"], neuron_order)
                 for entry, neuron_order in zip(entries, neuron_orders)]
    slice_of = [_slice_index(slices, p) for slices, p in zip(placement, positions)]
    for src, dst, _, record in connections:
        k = record["dst"]
        synapses, axons, routes = _per_core(src, dst, slice_of[k - 1], slice_of[k],
                                            len(placement[k - 1]), len(placement[k]),
                                            entries[k - 1]["size"])
        record.update(synapses_per_dst_core=synapses, axons_per_dst_core=axons,
                      routes_per_src_core=routes)

    network_out = sanafe.Network()
    groups = []
    for k, (entry, regs, bias) in enumerate(zip(entries, registers, biases)):
        group = network_out.create_neuron_group(
            _group_name(k), entry["size"], model_attributes={**regs, "bias_mant": 0,
                                                             "initial_v": 0},
            soma_hw_name=LIF_SOMA_UNIT, log_spikes=bool(log), log_potential=bool(log))
        for offset in np.flatnonzero(bias).tolist():
            group[offset].set_attributes(model_attributes={"bias_mant": int(bias[offset])})
        # Reader index of the neuron at each position of the layer's order (the identity
        # in reader order).
        at_position = np.argsort(positions[k], kind="stable").tolist()
        for s, core in zip(placement[k], cores[k]):
            for position in range(s["start"], s["stop"]):
                group[at_position[position]].map_to_core(core)
        groups.append(group)
    for src, dst, weight, record in connections:
        pre, post = groups[record["src"]], groups[record["dst"]]
        for start in range(0, src.size, _CONNECT_CHUNK):
            stop = start + _CONNECT_CHUNK
            pre.connect_neurons_sparse(post, {"weight": weight[start:stop].tolist()},
                                       list(zip(src[start:stop].tolist(),
                                                dst[start:stop].tolist())))

    lif_model = ("lava.proc.lif.models.PyLifResetModelBitAcc"
                 if any(r["reset_interval"] for r in registers)
                 else "lava.proc.lif.models.PyLifModelBitAcc")
    kinds = [entry["kind"] for entry in entries]
    lava_models = [lif_model]
    if "conv" in kinds:
        lava_models.append("lava.proc.conv.models.PyConvModelFixed")
    if "dense" in kinds:
        lava_models.append("lava.proc.dense.models.PyDenseModelBitAcc")
    profile = lif_architecture_profile(asdict(Allocation()))
    manifest = {
        "schema_version": 1,
        "execution_profile": LIF_EXECUTION_PROFILE,
        "soma_unit": LIF_SOMA_UNIT,
        "architecture_profile": profile,
        # The allocation in the profile is the candidate's default byte layout. No
        # resource check was run on this placement.
        "allocation_status": ("default Allocation() of the candidate profile, not derived "
                              "from this placement"),
        "resources_validated": False,
        "architecture_sha256": fingerprint,
        "hardware_validated": False,
        "input_transfer_cost_included": False,
        "lava_models": lava_models,
        "input": ("input layer driven by its bias mantissa (input_bias) with the layer's "
                  "bias exponent, no external activation"),
        "connection_delay_updates": 1,
        "zero_weight_synapses": "not emitted",
        "logged": bool(log),
        "group_names": [_group_name(k) for k in range(len(entries))],
        "layer_kinds": kinds,
        "layer_shapes": [list(entry["shape"]) for entry in entries],
        "layer_sizes": [entry["size"] for entry in entries],
        # resets_at is the reduced phase (offset mod interval) that the soma receives as
        # its reset_offset. netx's unreduced offset is the reader's neuron["reset_offset"].
        "layers": [{**{key: value for key, value in regs.items() if key != "reset_offset"},
                    "resets_at": regs["reset_offset"],
                    "bias_mant_min": int(b.min()), "bias_mant_max": int(b.max()),
                    "bias_mant_nonzero": int(np.count_nonzero(b))}
                   for regs, b in zip(registers, biases)],
        "order": order,
        "layer_cores": [len(slices) for slices in placement],
        "reserved_cores": [list(p) for p in RESERVED_CORES],
        "placement": [{"layer": k, "slices": slices} for k, slices in enumerate(placement)],
        "connections": [record for _, _, _, record in connections],
    }
    if partition is not None:
        manifest["partition"] = {
            "name": partition.get("name"),
            "layers": [{"order": o, "sizes": [s["stop"] - s["start"] for s in slices]}
                       for o, slices in zip(neuron_orders, placement)]}
        for layer, neuron_order in zip(manifest["placement"], neuron_orders):
            layer["order"] = neuron_order
    return network_out, manifest


def sanafe_run(net, manifest, arch, steps, *, extra_traces=True):
    """Run a network built by ``netx_to_sanafe`` for ``steps`` updates.

    Returns ``{'spikes': [...], 'voltages': [...], 'trace': result}`` with one
    ``[steps, n]`` int64 array per layer (``None`` when the network was built
    with ``log=False``) and the simulator's result dict. ``extra_traces`` adds
    the perf and message traces that per-core counts need.
    """
    steps = _scalar_integer(steps, "steps", 1, _INT32_MAX)
    logged = bool(manifest["logged"])
    chip = sanafe.SpikingChip(arch)
    chip.load(net)
    options = {"spike_trace": logged, "potential_trace": logged}
    if extra_traces:
        options.update(perf_trace=True, message_trace=True)
    result = chip.sim(steps, **options)
    if not logged:
        return {"spikes": None, "voltages": None, "trace": result}
    names, sizes = manifest["group_names"], manifest["layer_sizes"]
    index = {name: k for k, name in enumerate(names)}
    spikes = [np.zeros((steps, n), dtype=np.int64) for n in sizes]
    for t, active in enumerate(result["spike_trace"]):
        for neuron in active:
            spikes[index[neuron.group_name]][t, neuron.neuron_offset] = 1
    # The potential trace lists groups in name order, then neurons by offset.
    potentials = np.asarray(result["potential_trace"], dtype=np.int64)
    if potentials.shape != (steps, sum(sizes)):
        raise ValueError(f"potential trace of shape {potentials.shape}, expected "
                         f"{(steps, sum(sizes))}")
    order = sorted(range(len(names)), key=lambda k: names[k])
    columns = np.split(potentials, np.cumsum([sizes[k] for k in order])[:-1], axis=1)
    voltages = [None] * len(names)
    for k, column in zip(order, columns):
        voltages[k] = column
    return {"spikes": spikes, "voltages": voltages, "trace": result}


# ---------------------------------------------------------------------------
# Lava


def _lava_bias(entry, bias):
    """A bias in flattened (c, h, w) order, as Lava's (x, y, c) array or a 1-D array.

    The flat bias comes from ``_biases``, which has already required the input
    bias to have exactly the layer's (c, h, w) shape.
    """
    if len(entry["shape"]) == 3:
        return bias.reshape(entry["shape"]).transpose(2, 1, 0).astype(np.int32)
    return bias.astype(np.int32)


def _lava_shape(entry):
    return tuple(reversed(entry["shape"])) if len(entry["shape"]) == 3 else entry["shape"]


def _direct_processes(entries, biases):
    """Lava LIF, Conv and Dense processes built from the reader's layers."""
    from lava.proc.lif.process import LIF, LIFReset
    from lava.proc.conv.process import Conv
    from lava.proc.dense.process import Dense

    neurons = []
    for k, (entry, bias) in enumerate(zip(entries, biases)):
        layer, neuron = entry["layer"], entry["layer"].neuron
        params = {"shape": _lava_shape(entry), "du": neuron["du"], "dv": neuron["dv"],
                  "vth": neuron["vth"], "bias_exp": neuron["bias_exp"],
                  "bias_mant": _lava_bias(entry, bias)}
        if neuron.get("reset_interval") is None:
            lif = LIF(**params)
        else:
            lif = LIFReset(**params, reset_interval=neuron["reset_interval"],
                           reset_offset=neuron["reset_offset"])
        if entry["kind"] == "conv":
            (sh, sw), (ph, pw) = layer.stride, layer.padding
            # Lava's Conv is a true convolution on (out, x, y, in) kernels. The flip and
            # the transposition give the reader's cross-correlation (netx hdf5.py lines
            # 411 and 412).
            weight = np.asarray(layer.weights)[:, :, ::-1, ::-1].transpose(0, 3, 2, 1)
            synapse = Conv(weight=weight.astype(np.int32), input_shape=_lava_shape(entries[k - 1]),
                           stride=(sw, sh), padding=(pw, ph))
            neurons[-1].s_out.connect(synapse.s_in)
            synapse.a_out.connect(lif.a_in)
        elif entry["kind"] == "dense":
            synapse = Dense(weights=np.asarray(layer.weights), weight_exp=int(layer.weight_exp),
                            num_weight_bits=int(layer.weight_bits))
            source = neurons[-1].s_out
            if entry["flatten"]:
                # The (c, h, w) flatten of netx (hdf5.py lines 553 and 554).
                source = source.transpose([2, 1, 0]).flatten()
            source.connect(synapse.s_in)
            synapse.a_out.connect(lif.a_in)
        neurons.append(lif)
    return neurons, neurons[0]


def _netx_processes(path, entries, biases):
    """lava-dl's netx network built from the file, with the input bias written in."""
    try:
        from lava.lib.dl import netx
    except ImportError as exc:
        raise ImportError("lava-dl's netx is not importable. Put data/calibration/lava-dl/src "
                          "on sys.path (fetched by SANA-FE-thesis/calibration/deployments/"
                          "pilotnet_lif/fetch.py)") from exc
    first = entries[0]["layer"].neuron
    interval = first.get("reset_interval")
    offset = 0 if interval is None else int(first["reset_offset"]) - 1
    net = netx.hdf5.Network(net_config=str(path), reset_interval=interval, reset_offset=offset)
    if len(net.layers) != len(entries):
        raise ValueError(f"netx built {len(net.layers)} layers, the network has {len(entries)}")
    for k, (block, entry) in enumerate(zip(net.layers, entries)):
        if tuple(block.shape) != _lava_shape(entry):
            raise ValueError(f"netx built a layer of shape {tuple(block.shape)}, the network "
                             f"has {_lava_shape(entry)}")
        # The reset arguments above are rebuilt from the reader. netx's own derivation
        # must give every layer the reader's interval and unreduced offset.
        neuron = entry["layer"].neuron
        built = block.neuron.proc_params
        expected = (neuron.get("reset_interval"), neuron.get("reset_offset"))
        found = (built.get("reset_interval"), built.get("reset_offset"))
        if expected[0] is None:
            expected = (None, None)  # a plain LIF, which carries neither parameter
        if found != expected:
            raise ValueError(f"layer {k}: netx built reset interval and offset {found}, the "
                             f"network has {expected}")
    # netx leaves the input layer's bias mantissa at 0 and its input port open. The
    # Var's init is what the compiler hands the process model.
    net.layers[0].neuron.bias_mant.init = _lava_bias(entries[0], biases[0])
    return [block.neuron for block in net.layers], net


def _flatten_rows(rows, entry):
    """Rows of Lava arrays in (x, y, c) order to a [steps, n] array in (c, h, w) order."""
    data = np.asarray(rows)
    if len(entry["shape"]) == 3:
        data = data.transpose(0, 3, 2, 1)
    return data.reshape(len(rows), entry["size"]).astype(np.int64)


def _receive(wires, extractors, steps, stall_seconds):
    """Receive ``steps`` rows from every tap, failing when no row arrives in time."""
    rows, failure = [], []

    def work():
        try:
            for _ in range(steps):
                rows.append(([w.receive() for w in wires], [e.receive() for e in extractors]))
        except BaseException as exc:  # handed to the caller
            failure.append(exc)

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    seen, last = 0, time.monotonic()
    while worker.is_alive():
        worker.join(timeout=0.5)
        if len(rows) != seen:
            seen, last = len(rows), time.monotonic()
        elif time.monotonic() - last > stall_seconds:
            raise TimeoutError(f"Lava delivered no step for {stall_seconds} s "
                               f"(after {len(rows)} of {steps} steps of this chunk)")
    if failure:
        raise failure[0]
    return rows


def _terminate_lava_children():
    """Terminate every multiprocessing child of this process, not only Lava's."""
    import multiprocessing
    for child in multiprocessing.active_children():
        child.terminate()


def lava_run(network, input_bias, steps, *, builder, path=None, chunk=16, stall_seconds=300):
    """Run the network in Lava's fixed-point model and read every step of every layer.

    ``builder="netx"`` builds the processes with lava-dl's netx from the file at
    ``path`` (which must be the file ``network`` was read from, with the same
    reset arguments). ``builder="direct"`` builds Lava's LIF, Conv and Dense
    processes from ``network``. lava-dl's ``src`` folder must be importable for
    ``netx``. Returns ``{'spikes': [...], 'voltages': [...]}``, one ``[steps, n]``
    int64 array per layer with a neuron, in flattened (c, h, w) order. The run
    goes in chunks of at most ``chunk`` steps (at most 16). See the module
    docstring for how the state is read.
    """
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.run_configs import Loihi2SimCfg
    from lava.proc.io.extractor import (Extractor, PyLoihiExtractorModel, PyLoihiVarWireModel,
                                        VarWire)

    steps = _scalar_integer(steps, "steps", 1, _INT32_MAX)
    chunk = _scalar_integer(chunk, "chunk", 1, 16)
    entries = _neuron_layers(network)
    for entry in entries:
        _registers_of(entry)
    biases = [_biases(entry, input_bias) for entry in entries]
    if builder == "netx":
        if path is None:
            raise ValueError("builder 'netx' needs the path of the netx file")
        neurons, top = _netx_processes(path, entries, biases)
    elif builder == "direct":
        neurons, top = _direct_processes(entries, biases)
    else:
        raise ValueError(f"builder {builder!r} is not 'netx' or 'direct'")
    wires, extractors = [], []
    for lif in neurons:
        wire = VarWire(buffer_size=chunk + 1)
        wire.connect_var(lif.v)
        extractor = Extractor(shape=tuple(lif.s_out.shape), buffer_size=chunk + 1)
        lif.s_out.connect(extractor.in_port)
        wires.append(wire)
        extractors.append(extractor)
    config = Loihi2SimCfg(select_tag="fixed_pt", exception_proc_model_map={
        Extractor: PyLoihiExtractorModel, VarWire: PyLoihiVarWireModel})
    rows = []
    in_flight = False
    try:
        while len(rows) < steps:
            n = min(chunk, steps - len(rows))
            in_flight = True
            top.run(condition=RunSteps(num_steps=n, blocking=False), run_cfg=config)
            rows.extend(_receive(wires, extractors, n, stall_seconds))
            top.wait()
            in_flight = False
    finally:
        if in_flight:
            # A chunk failed between run and wait (a stall, an error in a tap, or an
            # error of the run itself). stop() would send commands to a runtime that may
            # still be stepping or may be stuck, and it waits for replies without a time
            # limit, so it is not attempted. Every multiprocessing child of this process,
            # Lava's runtime among them, is terminated instead, which cannot block. The
            # exception propagates.
            _terminate_lava_children()
        else:
            # Every chunk completed with wait(), so the runtime is idle and stop() is
            # the shutdown Lava intends.
            top.stop()
    voltages = [_flatten_rows([row[0][k] for row in rows], entry)
                for k, entry in enumerate(entries)]
    spikes = [_flatten_rows([row[1][k] for row in rows], entry)
              for k, entry in enumerate(entries)]
    return {"spikes": spikes, "voltages": voltages}
