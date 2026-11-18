"""Per-core, per-step counts of the Loihi 2 max-affine runtime model, for networks split over cores.

``analyze_binary_chain`` (``sanafe/loihi2_runtime.py``) counts the operations of
a dense IF chain with one core per layer. ``step_counts`` of the thesis route
(``mixed-horizon-snn/mhsnn/loihi2/estimate.py``) counts from SANA-FE's message
trace. Neither covers a LIF network whose convolutions are split over several
cores. ``core_step_counts`` takes the counts from the spikes of every step and
the expanded connectivity of the network, under a placement, with the meaning
each count has in ``analyze_binary_chain``. ``workload_from_counts`` turns them
into the workload that ``max_affine_runtime`` consumes. ``message_trace_counts``
reads the same quantities from a message trace, for comparison. Nothing here
computes a time or holds a rate.

The runtime model is Eq. 1 of Timcheck, Pierro and Shrestha (arXiv:2601.10035v2,
ledger record ``timcheck2026runtime-v2``). Per step it takes the maximum of the
busiest core's DendOps, SynOps and SynMem reads, each times its rate, of the
heaviest link's bits over the link bandwidth, and of the barrier time. Its
microbenchmarks (Section III, summarized in the notes of the ledger record under
"Microbenchmark protocol") define what one operation is.

Attribution in time
-------------------
Every count of step ``t`` comes from the spikes emitted in step ``t``. The
synaptic work of a spike is attributed to its sending step, as SANA-FE processes
messages before the next soma update (``analyze_binary_chain`` docstring, lines
80 and 81, ``step_counts`` docstring). Neuron updates are those of step ``t``.

Definitions
-----------
Neuron update (DendOp), ``neuron_updates``.
    Every neuron mapped to a core counts one update in every step, with or
    without input. This is ``analyze_binary_chain`` lines 147 to 149 (the size of
    every layer valid in the step, and a LIF layer has no window, so every layer
    is valid in every step) and the rule ``idle="full"`` of ``step_counts``
    (``estimate.py`` lines 73 to 75). The other rule of ``estimate.py``,
    ``idle="skipped"`` (lines 76 to 80), counts a group only inside its window,
    which a network without windows does not have. The rates file names no idle
    rule. It is a structural hypothesis of ``estimate.py``. The runtime model's
    DendOp microbenchmark runs N non-spiking neurons without input on one core
    and divides the step time by N, so a DendOp is one neuron state update per
    step whether or not the neuron receives input. SANA-FE's LIF soma reports
    every neuron as updated in every step (``perf_trace['updated']``), so this
    rule equals the simulator's own count. A layer driven by its bias alone (the
    input layer) is counted like every other layer.

Neurons receiving input, ``neurons_receiving``.
    The definition is this module's, an event-driven alternative to the rule above and
    not used by ``workload_from_counts``. It counts the neurons of a core that
    receive at least one nonzero weight from the spikes of step ``t``. It is a
    lower bound of any event-driven rule, since a leaky neuron whose voltage is
    not zero also needs an update.

Synaptic operation (SynOp), ``synops``.
    Counted at the destination core, one per spike and synapse. Two conventions.
    ``nonzero`` counts only synapses of nonzero weight. This is
    ``analyze_binary_chain`` lines 182 and 183 (the nonzero weights of the row
    reached by the spike). ``expanded`` counts every expanded entry, zero weights
    included. ``step_counts`` line 89 counts the synapses SANA-FE emitted
    (``m["spikes"]``). That equals ``nonzero`` for ``netx_to_sanafe``, which does
    not emit zero weights, and ``expanded`` for the IF chain exporter, which does.
    The two implementations therefore differ, and ``nonzero`` follows
    ``analyze_binary_chain``. The runtime model's microbenchmarks are consistent
    with it, which is an inference from their description and not a statement of
    the paper. The SynMem read microbenchmark runs a dense-packed identity matrix
    with "few SynOps and O(N^2) reads", which suggests that a stored zero weight
    is read but is not a SynOp.

Synaptic memory read (SynMem read), ``synmem_reads`` and ``synmem_access_histograms``.
    A message arriving at a destination core is one access. Its entries are the
    stored weights of the source neuron on that core. Its reads are
    ceil(entries / weights_per_read), and a core's reads are the sum over its
    accesses. This is ``analyze_binary_chain`` lines 180 and 181 (one access per
    message, entries the whole stored row, zero weights included) and
    ``max_affine_runtime`` lines 254 to 256 (the ceiling per access). The two
    implementations differ in two ways. (i) ``analyze_binary_chain`` counts
    stored entries including zeros, ``step_counts`` line 90 counts the emitted
    synapses. (ii) ``analyze_binary_chain`` applies the packing to every access,
    while ``step_counts`` applies it only with ``kernel_sharing=True`` and reads
    one entry per read otherwise, and the rates file states that its packing
    "applies to shared kernels only". This module follows
    ``analyze_binary_chain`` and applies the packing to every access. The
    unshared rule of ``step_counts`` equals the hypothesis of one weight per
    read. The hypotheses of the rates file are 4 (``timcheck_fig4_digitized`` and
    the two tolerance scenarios), 1 (``packing_one_per_read``) and 8
    (``packing_eight_per_read``). Which entries are stored is a second
    hypothesis with the two conventions of the SynOps. ``expanded`` stores every
    expanded entry (the dense storage of ``analyze_binary_chain``), ``nonzero``
    stores only the synapses ``netx_to_sanafe`` emits. For a convolution the
    entries of one access are, by this module's read model, one weight per
    output channel on the core and kernel tap that lands inside the output, and
    the model counts the same number whether the kernel is stored once and
    shared or stored per synapse, so that sharing changes the storage
    (``validate_group_resources``) and not the number of entries read. This read
    model of a convolution is the thesis's extension of the runtime model, which
    was validated on dense and identity layers only. Whether shared kernel words
    pack differently is the weights-per-read hypothesis.
    The runtime model's SynMem read microbenchmark (dense-packed 8-bit identity
    weights) scales the reads with the stored entries, not with the SynOps.

Message, ``messages_out``, ``messages_in`` and ``messages_out_per_spike``.
    The rule of ``messages_out`` and ``messages_in`` is SANA-FE's. One message
    per spiking source neuron and destination core that stores at least one of
    its entries, sent in the step of the spike, counted out at the source core
    and in at the destination core. SANA-FE allocates one axon per source neuron
    and destination core (``sim_allocate_axon`` in ``src/chip.cpp``, ``Message``
    in ``src/message.cpp``), and ``step_counts`` (lines 83 to 86) counts one per
    SANA-FE message. ``analyze_binary_chain`` (lines 154 to 168 and 193) sends
    one message per spike of a layer that has a next layer, which is the same
    rule when each layer has one core. The runtime model does not decide the
    rule. Its link microbenchmark uses identity synapses, so every source has
    one destination, and per-core, per-axon and per-neuron counting cannot be
    told apart there. Which pairs exist follows the storage convention. Under
    ``nonzero`` (the network the builder emits, whose messages the simulator
    sends) a source whose entries on a core are all zero sends no message there.
    Under ``expanded`` (the dense chain of ``analyze_binary_chain``) it does.
    The ``expanded`` messages of a network split over cores are this module's
    definition. ``messages_out_per_spike`` is a second rule, under which a spike
    leaves its core once whatever its fan-out. It counts, at the source core,
    each spike of a neuron that has at least one entry under the convention.
    Under that rule each destination core still receives the spike once, so
    ``messages_in`` is the same under both rules, and the link loads of this
    module are defined for SANA-FE's rule only (the second rule would need a
    multicast route). Messages, accesses and link loads depend on the partition
    (see below) and on the placement. SynOps and neuron updates per layer do not.

Directed link bits, ``link_messages`` and ``link_bits``.
    A message between distinct cores loads the core-to-router link of its source
    core, the router-to-core link of its destination core, and every
    router-to-router link of its dimension-order route, horizontal first, with
    tile coordinates (tile // height, tile % height). A message within one core
    loads no link. Bits are messages times ``message_bits``. This is
    ``analyze_binary_chain`` lines 184 to 188 with ``_route_links`` (lines 64 to
    74) and ``step_counts`` lines 91 to 95, which count messages that
    ``estimate_A`` multiplies by the packet width. The two agree. The runtime
    model takes the heaviest link including the core-to-router links and
    aggregates the two physical fabrics (Section VI).

Spikes, ``spikes_out``.
    The spikes emitted by the neurons of a core in the step. No counterpart in
    ``analyze_binary_chain``. They serve the per-layer tables.

Partition
---------
How the neurons of a layer are divided over its cores is a structural
hypothesis. A partition gives, per layer, an ordering of the neurons and a list
of slice sizes over that ordering. Slice ``j`` sits on the ``j``-th core the
manifest gives that layer. The ordering ``reader`` is the flattened (channels,
height, width) order of ``sanafe.netx``. The ordering ``lava`` is Lava's
flattened (width, height, channels) order, in which contiguous slices are
spatial and hold every channel. Dense layers have one order. ``builder`` is the
partition ``netx_to_sanafe`` builds by default (reader order, near-equal slices).
The builder also accepts a partition in this form and records it in the
manifest. The default of the functions here is the partition the manifest was
built with. Counts under another partition come from the same per-neuron spikes,
since the function of the network does not depend on where a neuron sits.

Conventions of the returned arrays
----------------------------------
Columns are the occupied cores sorted by (tile, core), as ``analyze_binary_chain``
orders them. Rows are the steps, row ``t`` the one-based step ``t + 1``. Links are
labels ``("core_to_router", tile, core)``, ``("router_to_core", tile, core)`` and
``("mesh", x1, y1, x2, y2)``, sorted, the union of the links that carry a message
under either convention in any step.
"""
from collections import Counter

import numpy as np

from sanafe.lif_network import (ORDERS, _conv_geometry, _neuron_layers,  # noqa: F401 (ORDERS)
                                conv_synapses, neuron_positions)
from sanafe.loihi2_runtime import _route_links

CONVENTIONS = ("nonzero", "expanded")
_IF_PROFILES = ("qcfs-if-int24-binary-v1", "qcfs-if-int24-binary-windowed-v1")
_LIF_PROFILE = "lava-lif-int24-v1"


# ---------------------------------------------------------------------------
# Structure of a manifest


def expanded_connectivity(network):
    """Every expanded entry of every connection of a netx network, zero weights included.

    Returns one dict per connection with ``src`` and ``dst`` (layer indices of
    ``netx_to_sanafe``), and int64 arrays ``source``, ``target`` and ``weight``
    (the reader's effective weights). Convolutions are expanded by
    ``conv_synapses`` from the reader's weights, stride, padding and shapes.
    Dense layers give one entry per weight. The builder emits the entries of
    nonzero weight.
    """
    entries = _neuron_layers(network)
    out = []
    for k in range(1, len(entries)):
        previous, entry = entries[k - 1], entries[k]
        weights = np.asarray(entry["layer"].effective_weights(), dtype=np.int64)
        if entry["kind"] == "conv":
            stride, padding = _conv_geometry(previous, entry)
            source, target, weight = conv_synapses(previous["shape"], weights, stride,
                                                   padding, entry["shape"])
        else:
            if weights.shape != (entry["size"], previous["size"]):
                raise ValueError(f"dense weights {weights.shape} do not map "
                                 f"{previous['size']} neurons to {entry['size']}")
            target, source = (axis.reshape(-1) for axis in np.indices(weights.shape))
            weight = weights.reshape(-1)
        out.append({"src": k - 1, "dst": k, "source": source.astype(np.int64),
                    "target": target.astype(np.int64), "weight": weight.astype(np.int64)})
    return out


def _structure(manifest, connectivity):
    """Sizes, slices, connectivity, windows and topology of a supported manifest."""
    profile = manifest.get("execution_profile")
    topology = manifest["architecture_profile"]["topology"]
    sizes = [int(n) for n in manifest["layer_sizes"]]
    if profile in _IF_PROFILES:
        places = [tuple(int(v) for v in p) for p in manifest["placements"]]
        slices = [[(place[0], place[1], 0, n)] for place, n in zip(places, sizes)]
        if connectivity is not None:
            raise ValueError("the IF chain manifest carries its own weights")
        connectivity = []
        for k, matrix in enumerate(manifest["effective_weights"]):
            matrix = np.asarray(matrix, dtype=np.int64)
            target, source = (axis.reshape(-1) for axis in np.indices(matrix.shape))
            connectivity.append({"src": k, "dst": k + 1, "source": source, "target": target,
                                 "weight": matrix.reshape(-1)})
        windows = manifest.get("valid_update_windows_zero_based")
        emitted = "expanded"  # the IF chain exporter emits every dense weight
    elif profile == _LIF_PROFILE:
        slices = [[(int(s["tile"]), int(s["core"]), int(s["start"]), int(s["stop"]))
                   for s in layer["slices"]] for layer in manifest["placement"]]
        if connectivity is None:
            raise ValueError("a LIF manifest needs the expanded connectivity of its network "
                             "(expanded_connectivity)")
        windows = None
        if manifest.get("zero_weight_synapses") != "not emitted":
            raise ValueError("the LIF manifest does not state that zero weights are not emitted")
        emitted = "nonzero"
    else:
        raise ValueError(f"unsupported execution profile {profile!r}")
    if len(slices) != len(sizes) or any(sum(s[3] - s[2] for s in layer) != n
                                        for layer, n in zip(slices, sizes)):
        raise ValueError("placement does not cover the layers")
    if [(c["src"], c["dst"]) for c in connectivity] != [(k, k + 1)
                                                        for k in range(len(sizes) - 1)]:
        raise ValueError("connectivity must connect each layer to the next")
    for c in connectivity:
        if (c["source"].size != c["target"].size or c["source"].size != c["weight"].size or
                (c["source"].size and (c["source"].min() < 0 or
                                       c["source"].max() >= sizes[c["src"]] or
                                       c["target"].min() < 0 or
                                       c["target"].max() >= sizes[c["dst"]]))):
            raise ValueError("connectivity indices outside the layers")
    return {"sizes": sizes, "slices": slices, "connectivity": connectivity,
            "windows": windows, "emitted": emitted, "height": int(topology["height"]),
            "width": int(topology["width"])}


def _layer_index(name):
    if not isinstance(name, str) or not name.startswith("layer_"):
        raise ValueError(f"unexpected group name {name!r}")
    return int(name.removeprefix("layer_"))


def spikes_from_trace(manifest, result):
    """Per-layer ``[steps, n]`` int64 spike arrays from a simulator result's spike trace."""
    sizes = [int(n) for n in manifest["layer_sizes"]]
    trace = result["spike_trace"]
    spikes = [np.zeros((len(trace), n), dtype=np.int64) for n in sizes]
    for t, active in enumerate(trace):
        for neuron in active:
            spikes[_layer_index(neuron.group_name)][t, int(neuron.neuron_offset)] = 1
    return spikes


def _spikes_and_result(manifest, traces):
    # A simulator result also has a key "spikes", its total of synaptic events.
    if "spike_trace" in traces:
        spikes, result = spikes_from_trace(manifest, traces), traces
    elif "spikes" in traces:
        spikes, result = traces["spikes"], traces.get("trace")
        if spikes is None:
            raise ValueError("the run was not logged, so it holds no spikes")
    else:
        raise ValueError("traces need 'spikes' (sanafe_run) or 'spike_trace' (a result)")
    spikes = [np.asarray(s) != 0 for s in spikes]
    steps = {s.shape[0] for s in spikes}
    if len(steps) != 1 or [s.shape[1] for s in spikes] != list(manifest["layer_sizes"]):
        raise ValueError("spike arrays do not match the layers")
    return spikes, result, steps.pop()


def _links_of(src, dst, height):
    """Links loaded by one message from core ``src`` to core ``dst`` ((tile, core) pairs)."""
    if src == dst:
        return []
    out = [("core_to_router",) + src, ("router_to_core",) + dst]
    # The route of analyze_binary_chain itself. The fork imports private helpers across
    # its modules (lif_network imports lava_lif's), so the private name is used directly.
    out += [("mesh",) + link for link in _route_links((src[0] // height, src[0] % height),
                                                      (dst[0] // height, dst[0] % height))]
    return out


# ---------------------------------------------------------------------------
# Counts from spikes


def _pairs(conn, column_of_src, column_of_dst, n_cores):
    """Distinct (source neuron, destination core) pairs with their entry counts."""
    dcol = column_of_dst[conn["target"]]
    key = conn["source"] * n_cores + dcol
    pairs, inverse = np.unique(key, return_inverse=True)
    expanded = np.bincount(inverse, minlength=pairs.size).astype(np.int64)
    nonzero = np.bincount(inverse, weights=(conn["weight"] != 0).astype(np.float64),
                          minlength=pairs.size).astype(np.int64)
    source = pairs // n_cores
    return {"source": source, "dcol": pairs % n_cores, "scol": column_of_src[source],
            "entries": {"nonzero": nonzero, "expanded": expanded}}


def _check_manifest_fanout(manifest, structure, pair_tables, slice_of):
    """The builder's per-core fan-out in the manifest equals the connectivity's (nonzero)."""
    for record, conn, pairs in zip(manifest["connections"], structure["connectivity"],
                                   pair_tables):
        k = conn["dst"]
        nz = conn["weight"] != 0
        if (record["synapses_expanded"] != conn["weight"].size or
                record["synapses_emitted"] != int(nz.sum())):
            raise ValueError(f"connection into layer {k}: synapse totals differ from the manifest")
        dslice = slice_of[k][conn["target"][nz]]
        synapses = np.bincount(dslice, minlength=len(structure["slices"][k])).tolist()
        kept = pairs["entries"]["nonzero"] > 0
        src_slice = slice_of[k - 1][pairs["source"][kept]]
        dst_slices = structure["slice_of_column"][k][pairs["dcol"][kept]]
        axons = np.bincount(dst_slices, minlength=len(structure["slices"][k])).tolist()
        routes = np.bincount(src_slice, minlength=len(structure["slices"][k - 1])).tolist()
        if (synapses != record["synapses_per_dst_core"] or axons != record["axons_per_dst_core"]
                or routes != record["routes_per_src_core"]):
            raise ValueError(f"connection into layer {k}: per-core fan-out differs from the "
                             "manifest")


def builder_partition(manifest):
    """The partition ``netx_to_sanafe`` built, with the manifest's slice sizes.

    The partition the manifest records when the builder was given one, and
    otherwise reader order under the name ``builder``.
    """
    if manifest.get("partition") is not None:
        return manifest["partition"]
    return {"name": "builder",
            "layers": [{"order": "reader",
                        "sizes": [int(s["stop"]) - int(s["start"]) for s in layer["slices"]]}
                       for layer in manifest["placement"]]}


def _slice_indices(manifest, structure, partition):
    """Per layer, the slice index of every neuron (reader index) under ``partition``.

    ``None`` is the partition the manifest was built with, which is the one it
    records, or reader order when it records none.
    """
    out = []
    if partition is None and manifest.get("partition") is not None:
        partition = manifest["partition"]
    if partition is None:
        for k, layer in enumerate(structure["slices"]):
            sli = np.empty(structure["sizes"][k], dtype=np.int64)
            for j, (_, _, start, stop) in enumerate(layer):
                sli[start:stop] = j
            out.append(sli)
        return out
    if manifest.get("execution_profile") != _LIF_PROFILE:
        raise ValueError("a partition other than the builder's needs a LIF manifest")
    layers = partition["layers"]
    if len(layers) != len(structure["sizes"]):
        raise ValueError("the partition needs one entry per layer")
    for k, spec in enumerate(layers):
        sizes = [int(v) for v in spec["sizes"]]
        if (len(sizes) != len(structure["slices"][k]) or any(v < 1 for v in sizes) or
                sum(sizes) != structure["sizes"][k]):
            raise ValueError(f"layer {k}: the partition needs one positive slice size per core "
                             f"({len(structure['slices'][k])}) summing to "
                             f"{structure['sizes'][k]}")
        pos = neuron_positions(manifest["layer_shapes"][k], manifest["layer_kinds"][k],
                               spec["order"])
        out.append(np.searchsorted(np.cumsum(sizes), pos, side="right").astype(np.int64))
    return out


def core_step_counts(manifest, traces, *, weights_per_read, message_bits, connectivity=None,
                     partition=None):
    """Per-step, per-core counts under the manifest's placement (see the module docstring).

    ``manifest`` is from ``netx_to_sanafe`` or from the IF chain exporter.
    ``traces`` is the output of ``sanafe_run`` (``{'spikes': [...], 'trace': ...}``)
    or a simulator result with a spike trace. The spikes may come from a run
    under another placement, since spikes do not depend on placement. When a
    perf trace is present, its neuron updates and its synaptic events per step
    must equal the counts (``verified_against_perf_trace``), and the synaptic
    events are compared under the convention the builder emitted. A LIF
    manifest needs ``connectivity`` (``expanded_connectivity`` of its network),
    and under the builder's partition its per-core fan-out must equal the
    manifest's. ``partition`` is ``None`` for the builder's partition, or
    ``{"name": ..., "layers": [{"order": "reader" or "lava", "sizes": [...]}]}``
    with one slice size per core of the manifest's placement (see the module
    docstring). ``weights_per_read`` is one integer or several. Returns a dict of
    int64 arrays, ``[steps, cores]`` per count and ``[steps, links]`` per link
    count, with the partition and the message rule recorded.
    """
    per_read = ([int(weights_per_read)] if np.isscalar(weights_per_read)
                else [int(v) for v in weights_per_read])
    if not per_read or any(v < 1 for v in per_read):
        raise ValueError("weights_per_read must be positive integers")
    message_bits = int(message_bits)
    if message_bits < 1:
        raise ValueError("message_bits must be a positive integer")
    structure = _structure(manifest, connectivity)
    spikes, result, steps = _spikes_and_result(manifest, traces)
    sizes, height = structure["sizes"], structure["height"]

    cores = sorted({(s[0], s[1]) for layer in structure["slices"] for s in layer})
    if any(not (0 <= t < structure["width"] * height and 0 <= c < 4) for t, c in cores):
        raise ValueError("placement outside the declared topology")
    column = {place: i for i, place in enumerate(cores)}
    n_cores = len(cores)
    column_of = []      # neuron -> core column, per layer
    slice_of = _slice_indices(manifest, structure, partition)
    structure["slice_of_column"] = []
    core_layers = [[] for _ in cores]
    layer_columns = np.zeros((len(sizes), n_cores), dtype=np.int64)  # neurons per layer and core
    for k, layer in enumerate(structure["slices"]):
        layer_cols = np.asarray([column[(tile, core)] for tile, core, _, _ in layer],
                                dtype=np.int64)
        col = layer_cols[slice_of[k]]
        by_column = np.full(n_cores, -1, dtype=np.int64)
        by_column[layer_cols] = np.arange(len(layer))
        layer_columns[k] = np.bincount(col, minlength=n_cores)
        for c in layer_cols.tolist():
            if k not in core_layers[c]:
                core_layers[c].append(k)
        column_of.append(col)
        structure["slice_of_column"].append(by_column)

    pair_tables = [_pairs(conn, column_of[conn["src"]], column_of[conn["dst"]], n_cores)
                   for conn in structure["connectivity"]]
    if manifest.get("execution_profile") == _LIF_PROFILE and partition is None:
        _check_manifest_fanout(manifest, structure, pair_tables, slice_of)
    # Sources with at least one entry under each convention, for the per-spike rule.
    has_entry = [{"nonzero": np.bincount(conn["source"][conn["weight"] != 0],
                                         minlength=sizes[conn["src"]]) > 0,
                  "expanded": np.bincount(conn["source"], minlength=sizes[conn["src"]]) > 0}
                 for conn in structure["connectivity"]]
    nonzero_entries = [(conn["source"][conn["weight"] != 0], conn["target"][conn["weight"] != 0])
                       for conn in structure["connectivity"]]

    zeros = lambda: np.zeros((steps, n_cores), dtype=np.int64)  # noqa: E731
    neuron_updates, receiving, spikes_out = zeros(), zeros(), zeros()
    synops = {c: zeros() for c in CONVENTIONS}
    reads = {c: {v: zeros() for v in per_read} for c in CONVENTIONS}
    messages_out = {c: zeros() for c in CONVENTIONS}
    messages_in = {c: zeros() for c in CONVENTIONS}
    per_spike = {c: zeros() for c in CONVENTIONS}
    histograms = {c: [[{} for _ in cores] for _ in range(steps)] for c in CONVENTIONS}
    link_counts = {c: [Counter() for _ in range(steps)] for c in CONVENTIONS}
    windows = structure["windows"]

    for t in range(steps):
        for k in range(len(sizes)):
            if windows is None or windows[k][0] <= t < windows[k][1]:
                neuron_updates[t] += layer_columns[k]
            fired = np.flatnonzero(spikes[k][t])
            if fired.size:
                spikes_out[t] += np.bincount(column_of[k][fired], minlength=n_cores)
        for conn, pairs, (nz_src, nz_dst), has in zip(structure["connectivity"], pair_tables,
                                                      nonzero_entries, has_entry):
            active = spikes[conn["src"]][t]
            if not active.any():
                continue
            for convention in CONVENTIONS:
                senders = np.flatnonzero(active & has[convention])
                per_spike[convention][t] += np.bincount(column_of[conn["src"]][senders],
                                                        minlength=n_cores)
            hit = np.zeros(sizes[conn["dst"]], dtype=bool)
            hit[nz_dst[active[nz_src]]] = True
            receiving[t] += np.bincount(column_of[conn["dst"]][hit], minlength=n_cores)
            sel = active[pairs["source"]]
            for convention in CONVENTIONS:
                entries = pairs["entries"][convention]
                ok = sel & (entries > 0)
                dcol, scol, e = pairs["dcol"][ok], pairs["scol"][ok], entries[ok]
                synops[convention][t] += np.bincount(dcol, weights=e,
                                                     minlength=n_cores).astype(np.int64)
                messages_out[convention][t] += np.bincount(scol, minlength=n_cores)
                messages_in[convention][t] += np.bincount(dcol, minlength=n_cores)
                for v in per_read:
                    reads[convention][v][t] += np.bincount(
                        dcol, weights=-(-e // v), minlength=n_cores).astype(np.int64)
                if e.size:
                    span = int(e.max()) + 1
                    keys, mult = np.unique(dcol * span + e, return_counts=True)
                    for key, m in zip(keys.tolist(), mult.tolist()):
                        hist = histograms[convention][t][key // span]
                        hist[key % span] = hist.get(key % span, 0) + m
                    combos, mult = np.unique(scol * n_cores + dcol, return_counts=True)
                    for key, m in zip(combos.tolist(), mult.tolist()):
                        for link in _links_of(cores[key // n_cores], cores[key % n_cores], height):
                            link_counts[convention][t][link] += m

    # Sort each histogram by entries, so equal histograms compare equal as dicts and print alike.
    for convention in CONVENTIONS:
        for row in histograms[convention]:
            for i, hist in enumerate(row):
                row[i] = dict(sorted(hist.items()))
    links = sorted({link for convention in CONVENTIONS for row in link_counts[convention]
                    for link in row})
    link_messages = {c: np.asarray([[row.get(link, 0) for link in links]
                                    for row in link_counts[c]], dtype=np.int64).reshape(
                                        steps, len(links)) for c in CONVENTIONS}

    verified = False
    if result is not None and result.get("perf_trace"):
        perf = result["perf_trace"]
        if (len(perf["updated"]) != steps or len(perf["spikes"]) != steps or
                list(perf["updated"]) != neuron_updates.sum(axis=1).tolist() or
                list(perf["spikes"]) != synops[structure["emitted"]].sum(axis=1).tolist()):
            raise ValueError("neuron updates or synaptic events differ from the perf trace")
        verified = True

    return {
        "schema": "loihi2-core-step-counts-v2",
        "execution_profile": manifest["execution_profile"],
        "partition": (builder_partition(manifest) if partition is None and
                      manifest.get("execution_profile") == _LIF_PROFILE
                      else partition if partition is not None
                      else {"name": "builder", "layers": "one core per layer"}),
        "message_rule": "SANA-FE's, one message per spiking source neuron and destination core "
                        "holding at least one of its entries (messages_out, messages_in, links), "
                        "and one per spike at its source core (messages_out_per_spike)",
        "steps": steps,
        "cores": cores,
        "core_layers": core_layers,
        "message_bits": message_bits,
        "weights_per_read": per_read,
        "emitted_convention": structure["emitted"],
        "verified_against_perf_trace": verified,
        "neuron_updates": neuron_updates,
        "neurons_receiving": receiving,
        "spikes_out": spikes_out,
        "synops": synops,
        "synmem_reads": reads,
        "synmem_access_histograms": histograms,
        "messages_out": messages_out,
        "messages_in": messages_in,
        "messages_out_per_spike": per_spike,
        "links": links,
        "link_messages": link_messages,
        "link_bits": {c: link_messages[c] * message_bits for c in CONVENTIONS},
    }


# ---------------------------------------------------------------------------
# Counts read from a message trace


def message_trace_counts(manifest, result, *, message_bits):
    """Messages, synaptic events, accesses and link messages read from a message trace.

    The reading of ``step_counts``. Every real message counts out at its source
    core and in at its destination core, its ``spikes`` (the synapses the
    simulator emitted on that axon) count as synaptic events and as the entries
    of one access, and it loads the links of ``_links_of``. These equal the
    counts of ``core_step_counts`` under the convention the builder emitted.
    """
    if manifest.get("execution_profile") == _LIF_PROFILE:
        cores = sorted({(int(s["tile"]), int(s["core"])) for layer in manifest["placement"]
                        for s in layer["slices"]})
    else:
        cores = sorted({(int(p[0]), int(p[1])) for p in manifest["placements"]})
    height = int(manifest["architecture_profile"]["topology"]["height"])
    column = {place: i for i, place in enumerate(cores)}
    trace = result["message_trace"]
    steps = len(trace)
    shape = (steps, len(cores))
    out_, in_, syn = (np.zeros(shape, dtype=np.int64) for _ in range(3))
    histograms = [[{} for _ in cores] for _ in range(steps)]
    link_rows = [Counter() for _ in range(steps)]
    for t, row in enumerate(trace):
        for m in row:
            if m["placeholder"]:
                continue
            if m["timestep"] != t + 1:
                raise ValueError("message timestep differs from its row")
            src = (int(m["src_tile_id"]), int(m["src_core_offset"]))
            dst = (int(m["dest_tile_id"]), int(m["dest_core_offset"]))
            if (int(m["src_x"]), int(m["src_y"])) != (src[0] // height, src[0] % height):
                raise ValueError("message source coordinates differ from its tile")
            fan = int(m["spikes"])
            out_[t, column[src]] += 1
            in_[t, column[dst]] += 1
            syn[t, column[dst]] += fan
            hist = histograms[t][column[dst]]
            hist[fan] = hist.get(fan, 0) + 1
            for link in _links_of(src, dst, height):
                link_rows[t][link] += 1
    for row in histograms:
        for i, hist in enumerate(row):
            row[i] = dict(sorted(hist.items()))
    links = sorted({link for row in link_rows for link in row})
    link_messages = np.asarray([[row.get(link, 0) for link in links] for row in link_rows],
                               dtype=np.int64).reshape(steps, len(links))
    return {"cores": cores, "messages_out": out_, "messages_in": in_, "synops": syn,
            "synmem_access_histograms": histograms, "links": links,
            "link_messages": link_messages, "link_bits": link_messages * int(message_bits)}


# ---------------------------------------------------------------------------
# The workload of max_affine_runtime


def workload_from_counts(counts, *, synops, synmem):
    """The workload ``max_affine_runtime`` consumes, in the schema of ``analyze_binary_chain``.

    Both conventions must be stated. ``synops`` selects the convention of the
    SynOps and ``synmem`` the storage convention of the SynMem entries, the
    accesses, the messages and the links. ``synops="nonzero", synmem="expanded"``
    are the conventions of ``analyze_binary_chain``, under which a dense chain
    gives its workload. The network the simulator runs is ``synmem="nonzero"``.
    The messages and links of the workload follow SANA-FE's message rule (one
    message per spiking source neuron and destination core) under the partition
    and placement the counts were taken with, recorded in the workload as
    ``partition`` and ``message_rule``.
    """
    if synops not in CONVENTIONS or synmem not in CONVENTIONS:
        raise ValueError(f"conventions are {CONVENTIONS}")
    bits = counts["message_bits"]
    steps = []
    for t in range(counts["steps"]):
        cores = []
        for i, (tile, core) in enumerate(counts["cores"]):
            accesses = [entries for entries, m in
                        counts["synmem_access_histograms"][synmem][t][i].items()
                        for _ in range(m)]
            cores.append({"tile": tile, "core": core,
                          "dendops": int(counts["neuron_updates"][t, i]),
                          "synops": int(counts["synops"][synops][t, i]),
                          "dense_synmem_entries": int(counts["synops"][synmem][t, i]),
                          "dense_synmem_accesses": accesses})
        links, endpoints = [], []
        for label, n in zip(counts["links"], counts["link_messages"][synmem][t].tolist()):
            if not n:
                continue
            if label[0] == "mesh":
                links.append({"from": [label[1], label[2]], "to": [label[3], label[4]],
                              "bits": n * bits})
            else:
                endpoints.append({"kind": label[0], "tile": label[1], "core": label[2],
                                  "bits": n * bits})
        maxima = {name: max((c[name] for c in cores), default=0)
                  for name in ("dendops", "synops", "dense_synmem_entries")}
        maxima["interrouter_link_bits"] = max((x["bits"] for x in links), default=0)
        maxima["link_bits"] = max(maxima["interrouter_link_bits"],
                                  max((x["bits"] for x in endpoints), default=0))
        steps.append({"timestep": t + 1,
                      "messages": int(counts["messages_out"][synmem][t].sum()),
                      "cores": cores, "links": links, "endpoint_links": endpoints,
                      "maxima": maxima})
    return {"schema_version": 2, "source": "sanafe.loihi2_counts.core_step_counts",
            "updates": counts["steps"], "messages": sum(s["messages"] for s in steps),
            "packet_bits_assumption": bits,
            "conventions": {"synops": synops, "synmem": synmem},
            "partition": counts.get("partition"),
            "message_rule": "SANA-FE's, one message per spiking source neuron and destination "
                            "core holding at least one of its entries",
            "link_scope": "directed core-router endpoints and XY router-router links, "
                          "two physical fabrics aggregated, same-core bypass assumed",
            "synmem_measure": f"stored entries under the {synmem} convention, physical reads "
                              "require the packing assumption",
            "input_output_transfer_included": False, "calibrated_for_loihi2": False,
            "steps": steps}
