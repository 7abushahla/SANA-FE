"""Per-core, per-step counts of the Loihi 2 runtime model (``sanafe.loihi2_counts``).

The counts must mean what they mean in ``analyze_binary_chain``. A dense chain
is exported twice, by the integer IF chain exporter that ``analyze_binary_chain``
reads and by ``netx_to_sanafe`` as a LIF network with the same weights and
placement, and both give the counts of ``analyze_binary_chain``. A hand-worked
convolution gives literal synaptic operations, messages and reads. Counts taken
from the spikes of one run equal counts read from the message trace of a run
under each placement. The workload built from the counts gives the modeled time
of ``analyze_binary_chain`` for every rate scenario of the rates file. Nothing
here skips. Lava's integer IF extension, h5py and the rates file must be present.
"""
import json

import h5py
import numpy as np
import pytest
import sanafe
from lava.proc.dense.process import Dense
from lava.proc.qcfs import QCFSIFFixed

import sanafe.lava as bridge
from sanafe.lif_network import check_lif_architecture, netx_to_sanafe, sanafe_run
from sanafe.loihi2 import load_loihi2_candidate
from sanafe.loihi2_counts import (builder_partition, core_step_counts, expanded_connectivity,
                                  message_trace_counts, neuron_positions, spikes_from_trace,
                                  workload_from_counts)
from sanafe.loihi2_runtime import RuntimeRates, analyze_binary_chain, max_affine_runtime
from sanafe.netx import read_netx

from thesis_tree import RATES

pytestmark = pytest.mark.thesis
READS = (1, 4, 8)
BITS = 32


# ---------------------------------------------------------------------------
# netx files, in the layout of test_lif_network_lava.py


def _neuron(group, vth=25, dv=1024):
    group.create_dataset("type", data=np.bytes_("LOIHI"), dtype=h5py.string_dtype("ascii"))
    values = dict(iDecay=4096, refDelay=1, scaleRho=1.0, tauRho=0.5, theta=vth << 6,
                  vDecay=dv, vThMant=vth, wgtExp=0)
    for key, value in values.items():
        group.create_dataset(key, data=value)


def _write(path, layers):
    with h5py.File(path, "w") as f:
        f.create_group("simulation").create_dataset("Ts", data=1)
        group = f.create_group("layer")
        for index, (kind, shape, neuron, datasets) in enumerate(layers):
            lg = group.create_group(str(index))
            lg.create_dataset("type", data=np.array([kind.encode()]))
            lg.create_dataset("shape", data=np.array(shape, dtype=np.int64))
            for key, value in datasets.items():
                lg.create_dataset(key, data=value)
            if neuron is not None:
                _neuron(lg.create_group("neuron"), **neuron)
    return read_netx(path, reset_interval=16, reset_offset=3)


def _input(shape):
    return ("input", shape, {}, {"weight": np.float32(40), "bias": np.float32(36)})


def _conv(shape, weight, bias, stride, padding):
    datasets = {"weight": np.asarray(weight, dtype=np.float32), "stride": np.array(stride),
                "padding": np.array(padding), "dilation": np.array([1, 1]), "groups": 1}
    if bias is not None:
        datasets["bias"] = np.asarray(bias, dtype=np.float32)
    return ("conv", shape, {}, datasets)


def _flatten(n):
    return ("flatten", (1, 1, n), None, {})


def _dense(weight, bias=None):
    weight = np.asarray(weight, dtype=np.float32)
    datasets = {"weight": weight}
    if bias is not None:
        datasets["bias"] = np.asarray(bias, dtype=np.float32)
    return ("dense", (weight.shape[0], 1, 1), {}, datasets)


def _counts(manifest, traces, **kwargs):
    return core_step_counts(manifest, traces, weights_per_read=READS, message_bits=BITS,
                            **kwargs)


def _sorted_workload(work):
    """Workload steps with each core's access list sorted, for comparison."""
    steps = []
    for step in work["steps"]:
        cores = [dict(c, dense_synmem_accesses=sorted(int(a) for a in c["dense_synmem_accesses"]))
                 for c in step["cores"]]
        steps.append(dict(step, cores=cores))
    return steps


# ---------------------------------------------------------------------------
# The dense chain, exported by the IF chain exporter and by netx_to_sanafe

# Effective weights [destination, source], even, since Lava's 8-bit Dense truncates odd
# weights. Source 3 of the first matrix has only zero weights, so the two conventions of
# the counts differ on it.
W1 = 2 * np.array([[6, -2, 0, 0, 3, 1, 0, -1],
               [0, 4, 5, 0, 0, -3, 2, 0],
               [-6, 0, 1, 0, 2, 0, 0, 4],
               [2, 2, 0, 0, -1, 5, 3, 0],
               [0, -4, 3, 0, 6, 0, 1, 2],
               [1, 0, -5, 0, 0, 2, 0, 3]])
W2 = 2 * np.array([[3, 0, -2, 5, 1, 0],
               [0, 6, 1, 0, -3, 2],
               [4, -1, 0, 2, 0, 5],
               [-6, 3, 2, 0, 4, 1]])
CHAIN_STEPS = 12
CHAIN_PLACES = [(0, 0), (1, 0), (2, 0)]  # the cores of order "spread" for one core per layer
# Tiles 0, 6 and 31 lie at (0, 0), (1, 2) and (7, 3), so every route turns from the
# horizontal to the vertical, and the second layer shares no core with the others.
TURNING_PLACES = [(0, 0), (6, 1), (31, 2)]


def _run_if_chain(places):
    """The IF chain run with every trace under ``places``: manifest, result, spikes."""
    sizes = (8, 6, 4)
    layers = [QCFSIFFixed(shape=(n,), threshold=4) for n in sizes]
    dense = [Dense(weights=W1), Dense(weights=W2)]
    for pre, conn, post in zip(layers, dense, layers[1:]):
        pre.s_out.connect(conn.s_in)
        conn.a_out.connect(post.a_in)
    rng = np.random.default_rng(5)
    currents = rng.integers(0, 6, size=(8, CHAIN_STEPS))
    currents[3] = 5  # the source with only zero weights spikes
    arch = load_loihi2_candidate()
    net, manifest = bridge.qcfs_fixed_chain_to_sanafe(layers, dense, currents, arch, places)
    assert manifest["effective_weights"] == [W1.tolist(), W2.tolist()]
    chip = sanafe.SpikingChip(arch)
    chip.load(net)
    result = chip.sim(CHAIN_STEPS, spike_trace=True, message_trace=True, perf_trace=True)
    spikes = spikes_from_trace(manifest, result)
    assert all(s.any() for s in spikes[:2]), "every layer that sends must spike"
    assert spikes[0][:, 3].any()
    return manifest, result, spikes


@pytest.fixture(scope="module")
def if_chain():
    return _run_if_chain(CHAIN_PLACES)


@pytest.fixture(scope="module")
def if_chain_turning():
    return _run_if_chain(TURNING_PLACES)


def test_dense_chain_counts_equal_analyze_binary_chain(if_chain, if_chain_turning, tmp_path):
    # Exported by the IF chain exporter, under both placements.
    for manifest, result, _ in (if_chain_turning, if_chain):
        expected = analyze_binary_chain(manifest, result, packet_bits=BITS)
        counts = _counts(manifest, result)
        assert counts["verified_against_perf_trace"]
        work = workload_from_counts(counts, synops="nonzero", synmem="expanded")
        assert work["messages"] == expected["messages"]
        assert _sorted_workload(work) == _sorted_workload(expected)
        assert any(step["links"] for step in expected["steps"])
    assert {(x["from"][0] != x["to"][0]) for step in
            analyze_binary_chain(if_chain_turning[0], if_chain_turning[1],
                                 packet_bits=BITS)["steps"] for x in step["links"]} == {True, False}

    # The same chain exported by netx_to_sanafe as a LIF network, same weights and cores.
    manifest, result, spikes = if_chain
    network = _write(tmp_path / "chain.net", [_input((8, 1, 1)), _flatten(8),
                                              _dense(W1), _dense(W2)])
    assert [layer.effective_weights().tolist() for layer in network.layers[2:]] == \
        [W1.tolist(), W2.tolist()]
    _, lif_manifest = netx_to_sanafe(network, np.zeros((8, 1, 1), dtype=np.int64),
                                     load_loihi2_candidate(), [1, 1, 1], order="spread",
                                     log=False)
    assert [(s["tile"], s["core"]) for p in lif_manifest["placement"] for s in p["slices"]] \
        == CHAIN_PLACES
    lif_counts = _counts(lif_manifest, {"spikes": spikes},
                         connectivity=expanded_connectivity(network))
    assert _sorted_workload(workload_from_counts(lif_counts, synops="nonzero",
                                                 synmem="expanded")) == \
        _sorted_workload(expected)

    # The conventions differ on the source whose weights are all zero.
    assert counts["messages_out"]["nonzero"].sum() < counts["messages_out"]["expanded"].sum()
    # Per-spike rule: every spike of a layer with a next layer, except, under the nonzero
    # convention, the spikes of source 3, whose weights are all zero.
    sent = spikes[0].sum(axis=1) + spikes[1].sum(axis=1)
    assert np.array_equal(counts["messages_out_per_spike"]["expanded"].sum(axis=1), sent)
    assert np.array_equal(counts["messages_out_per_spike"]["nonzero"].sum(axis=1),
                          sent - spikes[0][:, 3])
    assert np.array_equal(counts["synops"]["nonzero"], lif_counts["synops"]["nonzero"])
    # The IF exporter emits zero weights, so its message trace follows the expanded convention.
    assert counts["emitted_convention"] == "expanded"
    traced = message_trace_counts(manifest, result, message_bits=BITS)
    for key in ("messages_out", "messages_in", "synops"):
        assert np.array_equal(traced[key], counts[key]["expanded"]), key


# ---------------------------------------------------------------------------
# The hand-worked convolution

# Input (1, 4, 4), kernel 3 by 3, stride 1, no padding, output (2, 2, 2). Channel 0 has
# no zero weight. Channel 1 has zeros at the corners and at the center.
KERNEL = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
                   [[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]])


def _hand_network(tmp_path):
    return _write(tmp_path / "hand.net", [_input((1, 4, 4)),
                                          _conv((2, 2, 2), KERNEL, None, (1, 1), (0, 0))])


def _hand_run(network, bias, order, steps=4):
    arch = load_loihi2_candidate()
    net, manifest = netx_to_sanafe(network, bias, arch, [1, 2], order=order)
    run = sanafe_run(net, manifest, arch, steps, extra_traces=True)
    return manifest, run


def test_hand_worked_convolution(tmp_path):
    """Two input spikes per step, at (0, 0) and (1, 1), into two channels on two cores.

    Input (0, 0) reaches output (0, 0) through tap (0, 0): weight 1 in channel 0 and
    weight 0 in channel 1. Input (1, 1) reaches the four outputs through taps (1, 1),
    (1, 0), (0, 1) and (0, 0): weights 5, 4, 2, 1 in channel 0 and 0, 1, 1, 0 in
    channel 1. Channel 0 is on the first conv core, channel 1 on the second.
    Nonzero synaptic operations per step are 1 + 4 = 5 and 0 + 2 = 2. Expanded entries
    are 5 and 5. Messages, nonzero convention: (0, 0) sends to the first core only,
    (1, 1) to both, so 3 out and 2 and 1 in. Expanded convention: 4 out, 2 and 2 in.
    Reads at four weights per read: nonzero accesses [1, 4] and [2] give 2 and 1,
    expanded accesses [1, 4] and [1, 4] give 2 and 2.
    """
    network = _hand_network(tmp_path)
    bias = np.zeros((1, 4, 4), dtype=np.int64)
    bias[0, 0, 0] = bias[0, 1, 1] = 30  # 30 * 64 = 1920 > 1600, a spike every step
    connectivity = expanded_connectivity(network)

    manifest, run = _hand_run(network, bias, "packed")
    assert [(s["tile"], s["core"], s["start"], s["stop"])
            for p in manifest["placement"] for s in p["slices"]] == \
        [(0, 0, 0, 16), (0, 1, 0, 4), (0, 2, 4, 8)]
    assert run["spikes"][0].sum(axis=1).tolist() == [2, 2, 2, 2]
    assert not run["spikes"][1].any()  # the convolution never reaches its threshold
    c = _counts(manifest, run, connectivity=connectivity)
    assert c["verified_against_perf_trace"]
    assert c["cores"] == [(0, 0), (0, 1), (0, 2)]
    every = np.ones((4, 1), dtype=np.int64)
    assert np.array_equal(c["neuron_updates"], every * [16, 4, 4])
    assert np.array_equal(c["spikes_out"], every * [2, 0, 0])
    assert np.array_equal(c["synops"]["nonzero"], every * [0, 5, 2])
    assert np.array_equal(c["synops"]["expanded"], every * [0, 5, 5])
    assert np.array_equal(c["messages_out"]["nonzero"], every * [3, 0, 0])
    assert np.array_equal(c["messages_in"]["nonzero"], every * [0, 2, 1])
    assert np.array_equal(c["messages_out"]["expanded"], every * [4, 0, 0])
    assert np.array_equal(c["messages_in"]["expanded"], every * [0, 2, 2])
    assert c["synmem_access_histograms"]["nonzero"][0] == [{}, {1: 1, 4: 1}, {2: 1}]
    assert c["synmem_access_histograms"]["expanded"][0] == [{}, {1: 1, 4: 1}, {1: 1, 4: 1}]
    assert np.array_equal(c["synmem_reads"]["nonzero"][1], every * [0, 5, 2])
    assert np.array_equal(c["synmem_reads"]["nonzero"][4], every * [0, 2, 1])
    assert np.array_equal(c["synmem_reads"]["nonzero"][8], every * [0, 2, 1])
    assert np.array_equal(c["synmem_reads"]["expanded"][4], every * [0, 2, 2])
    # All three cores share tile 0, so only the endpoint links carry the messages.
    assert c["links"] == [("core_to_router", 0, 0), ("router_to_core", 0, 1),
                          ("router_to_core", 0, 2)]
    assert np.array_equal(c["link_messages"]["nonzero"], every * [3, 2, 1])
    assert np.array_equal(c["link_bits"]["nonzero"], every * [3, 2, 1] * BITS)
    # 6 neurons of the convolution receive a nonzero weight: all 4 of channel 0 and
    # outputs (0, 1) and (1, 0) of channel 1.
    assert np.array_equal(c["neurons_receiving"], every * [0, 4, 2])

    # Spread: the conv cores are on tiles 1 and 2, one router column up from tile 0.
    manifest, run = _hand_run(network, bias, "spread")
    c = _counts(manifest, run, connectivity=connectivity)
    assert c["cores"] == [(0, 0), (1, 0), (2, 0)]
    labels = dict(zip(c["links"], c["link_messages"]["nonzero"][0].tolist()))
    assert labels == {("core_to_router", 0, 0): 3, ("router_to_core", 1, 0): 2,
                      ("router_to_core", 2, 0): 1, ("mesh", 0, 0, 0, 1): 3,
                      ("mesh", 0, 1, 0, 2): 1}

    # Lava's order: slices [4, 4] are the output columns x = 0 and x = 1, both channels.
    # From (0, 0): output (0, 0) only, weights 1 and 0, on the first core. From (1, 1):
    # column x = 0 gets outputs (0, 0) and (1, 0) with weights 5, 0 and 2, 1, column
    # x = 1 gets outputs (0, 1) and (1, 1) with weights 4, 1 and 1, 0.
    manifest, run = _hand_run(network, bias, "packed")
    lava = {"name": "lava columns", "layers": [{"order": "lava", "sizes": [16]},
                                               {"order": "lava", "sizes": [4, 4]}]}
    c = _counts(manifest, run, connectivity=connectivity, partition=lava)
    assert c["partition"] == lava
    assert np.array_equal(c["neuron_updates"], every * [16, 4, 4])
    assert np.array_equal(c["synops"]["nonzero"], every * [0, 4, 3])
    assert np.array_equal(c["synops"]["expanded"], every * [0, 6, 4])
    assert np.array_equal(c["messages_out"]["nonzero"], every * [3, 0, 0])
    assert np.array_equal(c["messages_in"]["nonzero"], every * [0, 2, 1])
    # Both channels of output (0, 0) are on the first core, so (0, 0) sends one message
    # there under either convention.
    assert np.array_equal(c["messages_out"]["expanded"], every * [3, 0, 0])
    assert np.array_equal(c["messages_out_per_spike"]["nonzero"], every * [2, 0, 0])
    assert c["synmem_access_histograms"]["nonzero"][0] == [{}, {1: 1, 3: 1}, {3: 1}]
    assert np.array_equal(c["neurons_receiving"], every * [0, 3, 3])


def test_a_layer_without_input_spikes_updates_its_neurons_and_has_no_synaptic_operation(
        tmp_path):
    network = _hand_network(tmp_path)
    manifest, run = _hand_run(network, np.zeros((1, 4, 4), dtype=np.int64), "packed")
    assert not run["spikes"][0].any()
    c = _counts(manifest, run, connectivity=expanded_connectivity(network))
    every = np.ones((4, 1), dtype=np.int64)
    assert np.array_equal(c["neuron_updates"], every * [16, 4, 4])
    for convention in ("nonzero", "expanded"):
        assert not c["synops"][convention].any()
        assert not c["messages_in"][convention].any()
        assert not c["link_messages"][convention].any()
        for per_read in READS:
            assert not c["synmem_reads"][convention][per_read].any()
    assert not c["neurons_receiving"].any()


# ---------------------------------------------------------------------------
# Counts from the spikes of one run against the message traces of runs under each order


def _conv_network(tmp_path):
    """Input (2, 9, 7), conv 2x3 stride (2, 1) padding (0, 1) to (3, 4, 7), conv 3x2 to
    (4, 2, 6), flatten, dense to 10, with zero weights in every connection."""
    rng = np.random.default_rng(0)
    w1 = rng.integers(-12, 21, size=(3, 2, 2, 3))
    w2 = rng.integers(-10, 13, size=(4, 3, 3, 2))
    w3 = 2 * rng.integers(-75, 81, size=(10, 48))
    w3.reshape(-1)[:2] = [160, -150]
    w1.reshape(-1)[:3] = 0
    w2.reshape(-1)[:3] = 0
    w3.reshape(-1)[2:6] = 0
    return _write(tmp_path / "conv.net", [
        _input((2, 9, 7)),
        _conv((3, 4, 7), w1, (4, -6, 9), (2, 1), (0, 1)),
        _conv((4, 2, 6), w2, (2, -3, 0, 5), (1, 1), (0, 0)),
        _flatten(48),
        _dense(w3, rng.integers(-5, 6, size=10))])


def test_counts_from_spikes_equal_counts_read_from_runs_under_each_order(tmp_path):
    network = _conv_network(tmp_path)
    bias = np.random.default_rng(3).integers(-10, 60, size=(2, 9, 7))
    connectivity = expanded_connectivity(network)
    arch = load_loihi2_candidate()
    runs = {}
    for order in ("packed", "spread", "packed_reversed"):
        net, manifest = netx_to_sanafe(network, bias, arch, [2, 4, 3, 3], order=order)
        runs[order] = (manifest, sanafe_run(net, manifest, arch, 20, extra_traces=True))
    reference = runs["packed"][1]["spikes"]
    assert all(s.any() for s in reference)
    # Sources reach up to three destination cores in the second convolution and in the dense
    # layer, which is split over three cores.
    manifest = runs["packed"][0]
    for conn in connectivity[1:]:
        slices = manifest["placement"][conn["dst"]]["slices"]
        stops = np.asarray([s["stop"] for s in slices])
        dst = np.searchsorted(stops, conn["target"][conn["weight"] != 0], side="right")
        reach = np.bincount(np.unique(conn["source"][conn["weight"] != 0] * 8 + dst) // 8)
        assert len(slices) == 3 and reach.max() == 3, conn["dst"]
    for order, (manifest, run) in runs.items():
        assert all(np.array_equal(a, b) for a, b in zip(run["spikes"], reference)), order
        # Spikes of the packed run, placement of this order, perf trace of this order's run.
        counts = _counts(manifest, {"spikes": reference, "trace": run["trace"]},
                         connectivity=connectivity)
        assert counts["verified_against_perf_trace"]
        assert counts["emitted_convention"] == "nonzero"
        traced = message_trace_counts(manifest, run["trace"], message_bits=BITS)
        assert traced["cores"] == counts["cores"]
        for key in ("messages_out", "messages_in", "synops"):
            assert np.array_equal(traced[key], counts[key]["nonzero"]), (order, key)
            assert traced[key].any(), (order, key)
        assert traced["synmem_access_histograms"] == counts["synmem_access_histograms"]["nonzero"]
        links = dict(zip(counts["links"], counts["link_messages"]["nonzero"].T.tolist()))
        traced_links = dict(zip(traced["links"], traced["link_messages"].T.tolist()))
        assert {k: v for k, v in links.items() if any(v)} == traced_links, order
        assert any(label[0] == "mesh" for label in traced_links), order
        # The zero weights change the expanded counts.
        assert counts["synops"]["expanded"].sum() > counts["synops"]["nonzero"].sum()
        # The builder's partition stated explicitly gives the same counts as the default.
        explicit = _counts(manifest, {"spikes": reference}, connectivity=connectivity,
                           partition=builder_partition(manifest))
        for key in ("neuron_updates", "spikes_out", "neurons_receiving"):
            assert np.array_equal(explicit[key], counts[key]), (order, key)
        for key in ("synops", "messages_out", "messages_in", "link_messages"):
            for convention in ("nonzero", "expanded"):
                assert np.array_equal(explicit[key][convention], counts[key][convention])


def _same_counts(a, b):
    """Every per-core and per-link count of two count dicts is equal."""
    assert a["cores"] == b["cores"] and a["links"] == b["links"]
    for key in ("neuron_updates", "spikes_out", "neurons_receiving"):
        assert np.array_equal(a[key], b[key]), key
    for key in ("synops", "messages_out", "messages_in", "messages_out_per_spike",
                "link_messages"):
        for convention in ("nonzero", "expanded"):
            assert np.array_equal(a[key][convention], b[key][convention]), (key, convention)
    for convention in ("nonzero", "expanded"):
        assert a["synmem_access_histograms"][convention] == \
            b["synmem_access_histograms"][convention]
        for per_read in READS:
            assert np.array_equal(a["synmem_reads"][convention][per_read],
                                  b["synmem_reads"][convention][per_read])


def test_a_partition_in_lava_order_equals_the_relabeled_network_in_reader_order(tmp_path):
    """The counts under a partition are checked here without the simulator running it. Every
    neuron of every layer is relabeled by its position in Lava's order, spikes and synapses
    alike. The relabeled network under a reader-order partition
    with the same slice sizes must give every count of the original under the Lava-order
    partition. The sizes have a remainder slice, as the published plots suggest."""
    network = _conv_network(tmp_path)
    bias = np.random.default_rng(3).integers(-10, 60, size=(2, 9, 7))
    connectivity = expanded_connectivity(network)
    arch = load_loihi2_candidate()
    net, manifest = netx_to_sanafe(network, bias, arch, [2, 4, 3, 3], order="spread")
    spikes = sanafe_run(net, manifest, arch, 20, extra_traces=False)["spikes"]
    sizes = [[64, 62], [24, 24, 24, 12], [21, 21, 6], [4, 4, 2]]
    lava = {"name": "lava remainder", "layers": [{"order": "lava", "sizes": s} for s in sizes]}
    reader = {"name": "reader remainder", "layers": [{"order": "reader", "sizes": s}
                                                     for s in sizes]}
    perms = [neuron_positions(shape, kind, "lava")
             for shape, kind in zip(manifest["layer_shapes"], manifest["layer_kinds"])]
    assert all(np.array_equal(np.sort(p), np.arange(p.size)) for p in perms)
    assert not np.array_equal(perms[1], np.arange(perms[1].size))
    relabeled_spikes = []
    for s, perm in zip(spikes, perms):
        moved = np.zeros_like(s)
        moved[:, perm] = s
        relabeled_spikes.append(moved)
    relabeled = [dict(c, source=perms[c["src"]][c["source"]], target=perms[c["dst"]][c["target"]])
                 for c in connectivity]
    original = _counts(manifest, {"spikes": spikes}, connectivity=connectivity, partition=lava)
    moved = _counts(manifest, {"spikes": relabeled_spikes}, connectivity=relabeled,
                    partition=reader)
    _same_counts(original, moved)
    # The Lava order changes the counts against the builder's partition.
    builder = _counts(manifest, {"spikes": spikes}, connectivity=connectivity)
    assert not np.array_equal(builder["messages_in"]["nonzero"],
                              original["messages_in"]["nonzero"])


# ---------------------------------------------------------------------------
# The builder under an explicit partition

CONV_CORES = [2, 4, 3, 3]
LAVA_REMAINDER = {"name": "lava remainder",
                  "layers": [{"order": "lava", "sizes": s} for s in
                             ([64, 62], [24, 24, 24, 12], [21, 21, 6], [4, 4, 2])]}
# The builder's manifest and saved netlist for _conv_network, recorded from the builder as
# it was before it accepted a partition. The default must reproduce the netlist byte for
# byte, and the manifest byte for byte apart from the record of the architecture profile it
# embeds, which belongs to the profile and changes whenever a cost of the profile does. That
# record is hashed out of the manifest and checked instead against the fingerprint of the
# packaged profile, computed with the function the builder itself uses.
PROFILE_KEYS = ("architecture_profile", "architecture_sha256")
DEFAULT_SHA256 = {
    "spread": ("f000647a906bb1df04fe4a6f807e412f703174f7cc8bc3b0c2ad9be30105ceee",
               "6ca042f85977488567a7aa638570fb60647ee328515c5e7d6ba9e272cdffccf1"),
    "packed_reversed": ("96949a20936e90236e6648c28b7e7000b2fdd45640e59946028a3b5dc485a53f",
                        "816244f178ecd0385e66920cb8e1b547137007a16cdcd0f6e5993451b56afbaa"),
}


def _conv_bias():
    return np.random.default_rng(3).integers(-10, 60, size=(2, 9, 7))


def _digests(net, manifest, path):
    """sha256 of the manifest without the profile's record (JSON, sorted keys) and of the
    network saved as a netlist."""
    import hashlib
    net.save(str(path), use_netlist_format=True)
    assert all(key in manifest for key in PROFILE_KEYS)
    own = {key: value for key, value in manifest.items() if key not in PROFILE_KEYS}
    return (hashlib.sha256(json.dumps(own, sort_keys=True).encode()).hexdigest(),
            hashlib.sha256(path.read_bytes()).hexdigest())


def test_the_default_partition_leaves_the_builder_unchanged(tmp_path):
    network = _conv_network(tmp_path)
    arch = load_loihi2_candidate()
    fingerprint = check_lif_architecture(load_loihi2_candidate())
    for order, expected in DEFAULT_SHA256.items():
        net, manifest = netx_to_sanafe(network, _conv_bias(), arch, CONV_CORES, order=order)
        assert _digests(net, manifest, tmp_path / f"default_{order}.net") == expected, order
        assert manifest["architecture_sha256"] == fingerprint, order
        assert "partition" not in manifest
        net, manifest = netx_to_sanafe(network, _conv_bias(), arch, CONV_CORES, order=order,
                                       partition=None)
        assert _digests(net, manifest, tmp_path / f"none_{order}.net") == expected, order
        # The builder's own partition stated explicitly builds the same network. Only the
        # manifest differs, since it then records the partition.
        explicit = builder_partition(manifest)
        net, stated = netx_to_sanafe(network, _conv_bias(), arch, CONV_CORES, order=order,
                                     partition=explicit)
        digests = _digests(net, stated, tmp_path / f"explicit_{order}.net")
        assert digests[1] == expected[1], order
        assert stated["partition"] == explicit
        assert stated["placement"] == [dict(p, order="reader") for p in manifest["placement"]]


def test_a_lava_order_partition_places_the_neurons_of_its_slice_table(tmp_path):
    network = _conv_network(tmp_path)
    arch = load_loihi2_candidate()
    _, builder = netx_to_sanafe(network, _conv_bias(), arch, CONV_CORES, order="spread")
    net, manifest = netx_to_sanafe(network, _conv_bias(), arch, CONV_CORES, order="spread",
                                   partition=LAVA_REMAINDER)
    assert manifest["partition"] == LAVA_REMAINDER
    names = manifest["group_names"]
    moved = 0
    for k, (spec, layer, kept) in enumerate(zip(LAVA_REMAINDER["layers"], manifest["placement"],
                                                builder["placement"])):
        # The same cores as the builder's, in the same order, with the table's slice sizes.
        assert [(s["tile"], s["core"]) for s in layer["slices"]] == \
            [(s["tile"], s["core"]) for s in kept["slices"]]
        assert [s["stop"] - s["start"] for s in layer["slices"]] == spec["sizes"]
        assert layer["order"] == "lava"
        position = neuron_positions(manifest["layer_shapes"][k], manifest["layer_kinds"][k],
                                    "lava")
        slot = np.searchsorted(np.cumsum(spec["sizes"]), position, side="right")
        group = net.groups[names[k]]
        mapped = [tuple(group[i].mapped_core) for i in range(manifest["layer_sizes"][k])]
        expected = [(layer["slices"][j]["tile"], layer["slices"][j]["core"]) for j in slot]
        assert mapped == expected, k
        # On each core the neurons are mapped in the partition's order.
        mapping = np.asarray([group[i].mapping_order for i in range(manifest["layer_sizes"][k])])
        for j in range(len(spec["sizes"])):
            on_core = np.flatnonzero(slot == j)
            assert np.all(np.diff(mapping[on_core[np.argsort(position[on_core])]]) > 0), (k, j)
        reader = np.searchsorted(np.cumsum(spec["sizes"]), np.arange(position.size), side="right")
        moved += int(np.count_nonzero(reader != slot))
    assert moved > 0, "the Lava order must move neurons in this network"
    # The per-core fan-out of the manifest follows the partition.
    connectivity = expanded_connectivity(network)
    for record, conn in zip(manifest["connections"], connectivity):
        k = conn["dst"]
        position = neuron_positions(manifest["layer_shapes"][k], manifest["layer_kinds"][k],
                                    "lava")
        slot = np.searchsorted(np.cumsum(LAVA_REMAINDER["layers"][k]["sizes"]), position,
                               side="right")
        nz = conn["weight"] != 0
        assert record["synapses_per_dst_core"] == np.bincount(
            slot[conn["target"][nz]], minlength=len(manifest["placement"][k]["slices"])).tolist()


def test_a_partition_is_refused_unless_it_covers_every_core_of_every_layer(tmp_path):
    network = _conv_network(tmp_path)
    arch = load_loihi2_candidate()
    layers = LAVA_REMAINDER["layers"]

    def build(partition):
        return netx_to_sanafe(network, _conv_bias(), arch, CONV_CORES, partition=partition)

    with pytest.raises(ValueError, match="one entry per layer"):
        build({"name": "short", "layers": layers[:3]})
    with pytest.raises(ValueError, match=r"layer 1: the partition needs one positive slice size "
                                         r"per core \(4\) summing to 84"):
        build({"name": "sum", "layers": [layers[0], {"order": "lava", "sizes": [24, 24, 24, 13]},
                                         *layers[2:]]})
    with pytest.raises(ValueError, match=r"layer 2: the partition needs one positive slice size "
                                         r"per core \(3\) summing to 48"):
        build({"name": "count", "layers": [*layers[:2], {"order": "lava", "sizes": [24, 24]},
                                           layers[3]]})
    with pytest.raises(ValueError, match="layer 0: order 'width' is not one of"):
        build({"name": "order", "layers": [{"order": "width", "sizes": [64, 62]}, *layers[1:]]})
    # A dense layer has one order, so both names place every neuron on the same core. The
    # slices hold only tile, core, start and stop, so the comparison is of the per-neuron cores.
    net_a, a = build({"name": "dense lava", "layers": layers})
    net_b, b = build({"name": "dense reader", "layers": [*layers[:3],
                                                         {"order": "reader", "sizes": [4, 4, 2]}]})
    name = a["group_names"][3]
    assert b["group_names"][3] == name
    cores_a = [tuple(net_a.groups[name][i].mapped_core) for i in range(10)]
    cores_b = [tuple(net_b.groups[name][i].mapped_core) for i in range(10)]
    assert cores_a == cores_b
    assert len(set(cores_a)) == 3


def _bad_layers(index, replacement):
    layers = [dict(layer) for layer in LAVA_REMAINDER["layers"]]
    layers[index] = replacement
    return {"name": "bad", "layers": layers}


_REFUSED_PARTITIONS = [
    (lambda: {"name": "long", "layers": [*LAVA_REMAINDER["layers"], LAVA_REMAINDER["layers"][0]]},
     "one entry per layer"),
    (lambda: _bad_layers(1, {"order": "lava", "sizes": [24, 24, 36, 0]}),
     r"layer 1: the partition needs one positive slice size"),
    (lambda: _bad_layers(1, {"order": "lava", "sizes": [24, 24, 48, -12]}),
     r"layer 1: the partition needs one positive slice size"),
    (lambda: _bad_layers(0, {"order": "lava", "sizes": [64.7, 61.3]}),
     r"layer 0: slice size must be an integer"),
    (lambda: _bad_layers(0, {"order": "lava", "sizes": [True, 125]}),
     r"layer 0: slice size must be an integer"),
    (lambda: _bad_layers(2, {"order": "lava", "sizes": ["21", 21, 6]}),
     r"layer 2: slice size must be an integer"),
    (lambda: _bad_layers(2, {"order": "lava", "sizes": 48}),
     r"layer 2: the slice sizes must be a list"),
    (lambda: _bad_layers(2, {"order": "lava"}), r"layer 2: the entry has no 'sizes'"),
    (lambda: _bad_layers(2, {"sizes": [21, 21, 6]}), r"layer 2: the entry has no 'order'"),
    (lambda: _bad_layers(2, [21, 21, 6]), r"layer 2: the entry must be a mapping"),
    (lambda: {"name": "no layers"}, "a partition is a mapping with a 'layers' list"),
    (lambda: [{"order": "lava", "sizes": [64, 62]}], "a partition is a mapping"),
    (lambda: {"name": "scalar", "layers": 4}, "'layers' must be a list"),
]


@pytest.mark.parametrize("make,message", _REFUSED_PARTITIONS)
def test_every_refusal_of_a_stated_partition_is_a_value_error_naming_its_layer(
        tmp_path, make, message):
    network = _conv_network(tmp_path)
    arch = load_loihi2_candidate()
    net, manifest = netx_to_sanafe(network, _conv_bias(), arch, CONV_CORES, order="spread")
    steps = [np.zeros((3, n), dtype=np.int64) for n in manifest["layer_sizes"]]
    with pytest.raises(ValueError, match=message) as built:
        netx_to_sanafe(network, _conv_bias(), arch, CONV_CORES, partition=make())
    # The counts refuse the same partition with the same message.
    with pytest.raises(ValueError, match=message) as counted:
        _counts(manifest, {"spikes": steps, "trace": None},
                connectivity=expanded_connectivity(network), partition=make())
    assert str(built.value) == str(counted.value)


def test_the_neuron_positions_of_a_multichannel_layer_equal_lavas_order():
    # Shape (C=2, H=2, W=3). The reader index of (c, y, x) is (c * 2 + y) * 3 + x, so the
    # reader indices laid out by (c, y, x) are the array below.
    reader_index_array = np.arange(12).reshape(2, 2, 3)
    # Lava's index of (c, y, x) is (x * H + y) * C + c. This array is written out by hand,
    # one entry per reader index, from that formula.
    literal = np.array([0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11])
    positions = neuron_positions((2, 2, 3), "conv", "lava")
    assert np.array_equal(positions, literal)
    # Second, independent formula. Transposing to (W, H, C) and flattening lists the reader
    # indices in Lava's order, which is the inverse permutation of the positions.
    in_lava_order = np.transpose(reader_index_array, (2, 1, 0)).reshape(-1)
    assert np.array_equal(positions[in_lava_order], np.arange(12))
    assert np.array_equal(neuron_positions((2, 2, 3), "conv", "reader"), np.arange(12))
    assert np.array_equal(neuron_positions((2, 2, 3), "dense", "lava"), np.arange(12))


def test_the_partition_returned_by_the_counts_and_the_builder_is_a_copy(tmp_path):
    network = _conv_network(tmp_path)
    arch = load_loihi2_candidate()
    net, manifest = netx_to_sanafe(network, _conv_bias(), arch, CONV_CORES, order="spread",
                                   partition=LAVA_REMAINDER)
    recorded = json.loads(json.dumps(manifest["partition"]))
    run = sanafe_run(net, manifest, arch, 6, extra_traces=False)
    own = _counts(manifest, run, connectivity=expanded_connectivity(network))
    stated = _counts(manifest, run, connectivity=expanded_connectivity(network),
                     partition=manifest["partition"])
    for returned in (builder_partition(manifest), own["partition"], stated["partition"]):
        returned["layers"][0]["sizes"][0] += 1
        returned["name"] = "changed"
    assert manifest["partition"] == recorded


@pytest.mark.parametrize("stated", [False, True])
def test_the_fan_out_check_runs_when_the_manifests_own_partition_is_stated(tmp_path, stated):
    network = _conv_network(tmp_path)
    arch = load_loihi2_candidate()
    net, manifest = netx_to_sanafe(network, _conv_bias(), arch, CONV_CORES, order="spread",
                                   partition=LAVA_REMAINDER)
    run = sanafe_run(net, manifest, arch, 6, extra_traces=False)
    connectivity = expanded_connectivity(network)
    kwargs = {"partition": builder_partition(manifest)} if stated else {}
    _counts(manifest, run, connectivity=connectivity, **kwargs)
    corrupted = json.loads(json.dumps(manifest))
    corrupted["connections"][0]["synapses_per_dst_core"][0] += 1
    with pytest.raises(ValueError, match="per-core fan-out differs"):
        _counts(corrupted, run, connectivity=connectivity, **kwargs)
    # Another partition is not the manifest's, so its counts are not checked against it.
    _counts(corrupted, run, connectivity=connectivity, partition=LAVA_REMAINDER | {
        "layers": [{"order": "reader", "sizes": s["sizes"]} for s in LAVA_REMAINDER["layers"]]})


def test_the_spikes_do_not_depend_on_the_partition(tmp_path):
    network = _conv_network(tmp_path)
    arch = load_loihi2_candidate()
    runs = []
    for partition in (None, LAVA_REMAINDER):
        net, manifest = netx_to_sanafe(network, _conv_bias(), arch, CONV_CORES,
                                       order="packed_reversed", partition=partition)
        runs.append(sanafe_run(net, manifest, arch, 20, extra_traces=False))
    assert all(s.any() for s in runs[0]["spikes"])
    for a, b in zip(runs[0]["spikes"], runs[1]["spikes"]):
        assert np.array_equal(a, b)
    for a, b in zip(runs[0]["voltages"], runs[1]["voltages"]):
        assert np.array_equal(a, b)


def test_the_message_trace_under_a_lava_order_partition_equals_its_counts(tmp_path):
    network = _conv_network(tmp_path)
    connectivity = expanded_connectivity(network)
    arch = load_loihi2_candidate()
    _, builder = netx_to_sanafe(network, _conv_bias(), arch, CONV_CORES, order="spread")
    net, manifest = netx_to_sanafe(network, _conv_bias(), arch, CONV_CORES, order="spread",
                                   partition=LAVA_REMAINDER)
    run = sanafe_run(net, manifest, arch, 20, extra_traces=True)
    traced = message_trace_counts(manifest, run["trace"], message_bits=BITS)
    # The manifest's own partition, read from the manifest, and the same partition stated on
    # the builder's manifest give the same counts.
    own = _counts(manifest, run, connectivity=connectivity)
    assert own["verified_against_perf_trace"]
    assert own["partition"] == LAVA_REMAINDER
    stated = _counts(builder, {"spikes": run["spikes"]}, connectivity=connectivity,
                     partition=LAVA_REMAINDER)
    _same_counts(own, stated)
    assert traced["cores"] == own["cores"]
    for key in ("messages_out", "messages_in", "synops"):
        assert np.array_equal(traced[key], own[key]["nonzero"]), key
        assert traced[key].any(), key
    assert traced["synmem_access_histograms"] == own["synmem_access_histograms"]["nonzero"]
    links = dict(zip(own["links"], own["link_messages"]["nonzero"].T.tolist()))
    traced_links = dict(zip(traced["links"], traced["link_messages"].T.tolist()))
    assert {k: v for k, v in links.items() if any(v)} == traced_links
    assert any(label[0] == "mesh" for label in traced_links)
    # The simulator's messages differ from those of the builder's partition.
    default = _counts(builder, {"spikes": run["spikes"]}, connectivity=connectivity)
    assert not np.array_equal(default["messages_in"]["nonzero"], traced["messages_in"])


def test_the_hand_worked_convolution_runs_under_lava_columns(tmp_path):
    """The literal counts of the Lava-order case of ``test_hand_worked_convolution``, now
    read from the message trace of a run that places the output columns on the two cores."""
    network = _hand_network(tmp_path)
    bias = np.zeros((1, 4, 4), dtype=np.int64)
    bias[0, 0, 0] = bias[0, 1, 1] = 30
    lava = {"name": "lava columns", "layers": [{"order": "lava", "sizes": [16]},
                                               {"order": "lava", "sizes": [4, 4]}]}
    arch = load_loihi2_candidate()
    net, manifest = netx_to_sanafe(network, bias, arch, [1, 2], order="packed", partition=lava)
    run = sanafe_run(net, manifest, arch, 4, extra_traces=True)
    traced = message_trace_counts(manifest, run["trace"], message_bits=BITS)
    every = np.ones((4, 1), dtype=np.int64)
    assert np.array_equal(traced["synops"], every * [0, 4, 3])
    assert np.array_equal(traced["messages_out"], every * [3, 0, 0])
    assert np.array_equal(traced["messages_in"], every * [0, 2, 1])
    assert traced["synmem_access_histograms"][0] == [{}, {1: 1, 3: 1}, {3: 1}]


# ---------------------------------------------------------------------------
# The workload of the counts gives the modeled time of analyze_binary_chain


def test_workload_from_counts_gives_the_modeled_time_of_analyze_binary_chain(if_chain_turning):
    manifest, result, _ = if_chain_turning
    assert RATES.is_file(), (f"{RATES} is missing. The test reads the rate scenarios of the "
                             "thesis route (mixed-horizon-snn), so the fork must sit inside the "
                             "Thesis repository or SANAFE_THESIS_ROOT must name it.")
    scenarios = {name: s for name, s in json.loads(RATES.read_text()).items()
                 if not name.startswith("_")}
    assert len(scenarios) == 5
    for name, s in scenarios.items():
        rates = RuntimeRates(**{key: s[key] for key in (
            "dendop_seconds", "synop_seconds", "synmem_read_seconds", "link_bits_per_second",
            "barrier_seconds", "dense_entries_per_read")})
        expected = max_affine_runtime(analyze_binary_chain(manifest, result,
                                                           packet_bits=s["packet_bits"]), rates)
        counts = core_step_counts(manifest, result, weights_per_read=READS,
                                  message_bits=s["packet_bits"])
        modeled = max_affine_runtime(workload_from_counts(counts, synops="nonzero",
                                                          synmem="expanded"), rates)
        # Every term of every step is compared, not only the bound.
        assert modeled == expected, name
        assert any(step["synmem_reads_assumed"] > 0 and step["terms_seconds"]["link"] > 0
                   for step in modeled["steps"]), name


# ---------------------------------------------------------------------------
# Refusals of malformed arguments and manifests


def _tiny_if_manifest():
    return {"execution_profile": "qcfs-if-int24-binary-v1", "layer_sizes": [2, 2],
            "placements": [[0, 0], [0, 1]], "effective_weights": [[[1, 0], [0, 1]]],
            "architecture_profile": {"topology": {"width": 1, "height": 1}}}


def _dense_lif_chain_manifest():
    # The layout of lif_chain_to_sanafe, which carries the profile id of the LIF manifests of
    # netx_to_sanafe but holds placements and effective weights, not a placement table.
    manifest = _tiny_if_manifest()
    manifest["execution_profile"] = "lava-lif-int24-v1"
    return manifest


def test_the_manifest_of_a_dense_lif_chain_is_refused_by_both_count_functions():
    manifest = _dense_lif_chain_manifest()
    traces = {"spikes": [np.zeros((3, 2)), np.zeros((3, 2))], "trace": None,
              "spike_trace": [[], [], []], "message_trace": [[], [], []]}
    with pytest.raises(ValueError, match="lif_chain_to_sanafe.*not supported.*'placement'"):
        _counts(manifest, traces, connectivity=[])
    with pytest.raises(ValueError, match="lif_chain_to_sanafe.*not supported.*'placement'"):
        message_trace_counts(manifest, traces, message_bits=BITS)


_BAD_SCALARS = [4.9, True, "8", 0, -1]
# weights_per_read may also be given per layer, message_bits is a single value.
_BAD_COUNT_ARGUMENTS = (
    [("weights_per_read", bad) for bad in
     _BAD_SCALARS + [[4.9], [True], ["8"], np.array(4), np.array(4.5)]]
    + [("message_bits", bad) for bad in _BAD_SCALARS]
)


@pytest.mark.parametrize("name,bad", _BAD_COUNT_ARGUMENTS)
def test_weights_per_read_and_message_bits_must_be_positive_integers(name, bad):
    kwargs = {"weights_per_read": 4, "message_bits": 8}
    kwargs[name] = bad
    traces = {"spikes": [np.zeros((3, 2)), np.zeros((3, 2))], "trace": None}
    with pytest.raises(ValueError, match=name):
        core_step_counts(_tiny_if_manifest(), traces, **kwargs)


def test_a_run_without_a_requested_trace_is_refused_with_a_message():
    # A simulator result always has both keys, set to None when the trace was not requested.
    result = {"spike_trace": None, "message_trace": None, "spikes": 0}
    with pytest.raises(ValueError, match="not logged"):
        _counts(_tiny_if_manifest(), result)
    with pytest.raises(ValueError, match="not logged"):
        message_trace_counts(_tiny_if_manifest(), result, message_bits=BITS)
    with pytest.raises(ValueError, match="not logged"):
        _counts(_tiny_if_manifest(), {"spikes": None, "trace": None})
    with pytest.raises(ValueError, match="not logged"):
        message_trace_counts(_tiny_if_manifest(), {"spikes": 0}, message_bits=BITS)
