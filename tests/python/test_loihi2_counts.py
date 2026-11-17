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
from pathlib import Path

import h5py
import numpy as np
import pytest
import sanafe
from lava.proc.dense.process import Dense
from lava.proc.qcfs import QCFSIFFixed

import sanafe.lava as bridge
from sanafe.lif_network import netx_to_sanafe, sanafe_run
from sanafe.loihi2 import load_loihi2_candidate
from sanafe.loihi2_counts import (core_step_counts, expanded_connectivity,
                                  message_trace_counts, spikes_from_trace,
                                  workload_from_counts)
from sanafe.loihi2_runtime import RuntimeRates, analyze_binary_chain, max_affine_runtime
from sanafe.netx import read_netx

REPO = Path(__file__).resolve().parents[3]
RATES = REPO / "mixed-horizon-snn" / "rates" / "loihi2_scenarios.json"
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
        work = workload_from_counts(counts)
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
    assert _sorted_workload(workload_from_counts(lif_counts)) == _sorted_workload(expected)

    # The conventions differ on the source whose weights are all zero.
    assert counts["messages_out"]["nonzero"].sum() < counts["messages_out"]["expanded"].sum()
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
    # 7 neurons of the convolution receive a nonzero weight: all 4 of channel 0 and
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
        net, manifest = netx_to_sanafe(network, bias, arch, [2, 2, 2, 1], order=order)
        runs[order] = (manifest, sanafe_run(net, manifest, arch, 20, extra_traces=True))
    reference = runs["packed"][1]["spikes"]
    assert all(s.any() for s in reference)
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


# ---------------------------------------------------------------------------
# The workload of the counts gives the modeled time of analyze_binary_chain


def test_workload_from_counts_gives_the_modeled_time_of_analyze_binary_chain(if_chain_turning):
    manifest, result, _ = if_chain_turning
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
        modeled = max_affine_runtime(workload_from_counts(counts), rates)
        # Every term of every step is compared, not only the bound.
        assert modeled == expected, name
        assert any(step["synmem_reads_assumed"] > 0 and step["terms_seconds"]["link"] > 0
                   for step in modeled["steps"]), name
