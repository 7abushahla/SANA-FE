"""LIF networks with convolutions from netx files: the simulator against Lava's fixed-point model.

Every network is written as a netx file with h5py into ``tmp_path``, in the
layout of the reader's own tests (``test_netx.py``). Lava runs it under
``Loihi2SimCfg(select_tag='fixed_pt')``, built either by lava-dl's netx from the
file or directly from the reader's ``NetxNetwork``. The simulator runs the same
file through ``netx_to_sanafe``. Every comparison is ``np.array_equal`` on the
whole ``[steps, n]`` arrays of spikes and of voltages of every layer, with no
tolerance. The tests need Lava, h5py and the git-ignored lava-dl checkout. They
fail, and never skip, when the checkout is absent.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from sanafe.loihi2 import load_loihi2_candidate
from sanafe.netx import NetxLayer, NetxNetwork, read_netx
from sanafe.lif_network import (conv_synapses, core_order, lava_run, netx_to_sanafe,
                                sanafe_run, slice_sizes)

REPO = Path(__file__).resolve().parents[3]
LAVA_DL_SRC = REPO / "data" / "calibration" / "lava-dl" / "src"
FETCH = "SANA-FE-thesis/calibration/deployments/pilotnet_lif/fetch.py"
STEPS = 48


@pytest.fixture(scope="module", autouse=True)
def lava_dl_on_path():
    assert LAVA_DL_SRC.is_dir(), f"{LAVA_DL_SRC} is missing. Run {FETCH} to fetch lava-dl."
    if str(LAVA_DL_SRC) not in sys.path:
        sys.path.insert(0, str(LAVA_DL_SRC))


# ---------------------------------------------------------------------------
# netx files


def _neuron(group, vth=25, dv=1024):
    group.create_dataset("type", data=np.bytes_("LOIHI"), dtype=h5py.string_dtype("ascii"))
    values = dict(iDecay=4096, refDelay=1, scaleRho=1.0, tauRho=0.5, theta=vth << 6,
                  vDecay=dv, vThMant=vth, wgtExp=0)
    for key, value in values.items():
        group.create_dataset(key, data=value)


def _write(path, layers):
    """``layers`` holds (kind, shape, neuron or None, datasets), neuron a dict of vth and dv."""
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
    return path


def _input(shape):
    return ("input", shape, {}, {"weight": np.float32(40), "bias": np.float32(36)})


def _conv(shape, weight, bias, stride, padding, neuron=None):
    datasets = {"weight": np.asarray(weight, dtype=np.float32),
                "stride": np.array(stride), "padding": np.array(padding),
                "dilation": np.array([1, 1]), "groups": 1}
    if bias is not None:
        datasets["bias"] = np.asarray(bias, dtype=np.float32)
    return ("conv", shape, neuron or {}, datasets)


def _flatten(n):
    return ("flatten", (1, 1, n), None, {})


def _dense(weight, bias, neuron=None):
    weight = np.asarray(weight, dtype=np.float32)
    datasets = {"weight": weight}
    if bias is not None:
        datasets["bias"] = np.asarray(bias, dtype=np.float32)
    return ("dense", (weight.shape[0], 1, 1), neuron or {}, datasets)


def _even(rng, low, high, size, extremes):
    """Even integers in [low, high] with ``extremes`` placed first, so netx's rule is known."""
    w = 2 * rng.integers(low // 2, high // 2 + 1, size=size)
    flat = w.reshape(-1)
    flat[:len(extremes)] = extremes
    return w


def net_a(path, *, dense="wide", readout=False, conv_bias=(4, -6, 9), seed=0):
    """Network (a): input (2, 9, 7), conv 2x3 stride (2, 1) padding (0, 1) to (3, 4, 7),
    conv 3x2 stride 1 to (4, 2, 6), flatten, dense to 10.

    ``dense="wide"`` gives even weights in [-150, 160], for which netx's rule gives
    exponent 0. ``dense="small"`` gives weights in [-6, 6] (exponent -5) and adds a
    second dense layer of 5 with weights in [-24, 20] (exponent -3). ``readout``
    appends a dense layer of 2 with dv 0 and a threshold it never reaches.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.integers(-12, 21, size=(3, 2, 2, 3))
    w2 = rng.integers(-10, 13, size=(4, 3, 3, 2))
    layers = [_input((2, 9, 7)),
              _conv((3, 4, 7), w1, conv_bias, (2, 1), (0, 1)),
              _conv((4, 2, 6), w2, [2, -3, 0, 5], (1, 1), (0, 0)),
              _flatten(48)]
    if dense == "wide":
        layers.append(_dense(_even(rng, -150, 160, (10, 48), [160, -150]),
                             rng.integers(-5, 6, size=10)))
        last = 10
    else:
        layers.append(_dense(_even(rng, -6, 6, (10, 48), [6, -6]), rng.integers(-5, 6, size=10)))
        layers.append(_dense(rng.integers(-24, 21, size=(5, 10)), None))
        last = 5
    if readout:
        layers.append(_dense(rng.integers(-40, 31, size=(2, last)), None,
                             neuron={"vth": 131071, "dv": 0}))
    return _write(path, layers)


def net_g(path, *, conv_bias=(3, -2, 0, 7, -5), seed=1):
    """Network (g): input (3, 8, 11), conv 3x3 stride (2, 1) padding (1, 0) to (5, 4, 9),
    the geometry of PilotNet's fourth convolution, then flatten and dense to 6."""
    rng = np.random.default_rng(seed)
    w = rng.integers(-14, 19, size=(5, 3, 3, 3))
    return _write(path, [_input((3, 8, 11)),
                         _conv((5, 4, 9), w, conv_bias, (2, 1), (1, 0)),
                         _flatten(180),
                         _dense(rng.integers(-12, 14, size=(6, 180)), [1, 0, -2, 3, 0, 4])])


def _read(path):
    return read_netx(path, reset_interval=16, reset_offset=3)


def _input_bias(network, seed=2, low=-10, high=61):
    shape = network.layers[0].shape
    return np.random.default_rng(seed).integers(low, high, size=int(np.prod(shape)))


def _cores(network, per_layer):
    return [per_layer] * sum(1 for layer in network.layers if layer.neuron is not None)


# ---------------------------------------------------------------------------
# Comparison


def _assert_equal_runs(first, second, sizes, steps=STEPS):
    for key in ("spikes", "voltages"):
        assert len(first[key]) == len(second[key]) == len(sizes)
        for k, n in enumerate(sizes):
            assert first[key][k].shape == second[key][k].shape == (steps, n), (key, k)
            assert np.array_equal(first[key][k], second[key][k]), (key, k)


def _assert_nonvacuous(run, spiking):
    for k in spiking:
        assert np.any(run["spikes"][k]), f"layer {k} never spikes"
    assert any(np.any(v < 0) for v in run["voltages"]), "no negative voltage anywhere"


def _sanafe(network, bias, cores, order="packed", steps=STEPS):
    arch = load_loihi2_candidate()
    net, manifest = netx_to_sanafe(network, bias, arch, cores, order=order)
    return sanafe_run(net, manifest, arch, steps), manifest


def _sizes(network):
    return [int(np.prod(layer.shape)) for layer in network.layers if layer.neuron is not None]


# ---------------------------------------------------------------------------
# The equality tests (a) to (g)


def test_a_two_convolutions_and_dense(tmp_path):
    path = net_a(tmp_path / "a.net")
    network = _read(path)
    dense = network.layers[4]
    assert dense.weight_exp == 0
    assert [layer.resets_at for layer in network.layers if layer.neuron] == [4, 5, 6, 7]
    bias = _input_bias(network)
    lava = lava_run(network, bias, STEPS, builder="netx", path=path)
    ours, _ = _sanafe(network, bias, _cores(network, 1))
    _assert_equal_runs(lava, ours, _sizes(network))
    _assert_nonvacuous(lava, spiking=range(4))


def test_b_dense_with_negative_exponent(tmp_path):
    path = net_a(tmp_path / "b.net", dense="small")
    network = _read(path)
    dense = [layer for layer in network.layers if layer.kind == "dense"]
    assert [layer.weight_exp for layer in dense] == [-5, -3]
    assert not np.array_equal(dense[0].weights, dense[0].effective_weights())
    bias = _input_bias(network)
    lava = lava_run(network, bias, STEPS, builder="netx", path=path)
    ours, _ = _sanafe(network, bias, _cores(network, 1))
    _assert_equal_runs(lava, ours, _sizes(network))
    _assert_nonvacuous(lava, spiking=range(5))


def test_c_readout_compared_on_voltage(tmp_path):
    path = net_a(tmp_path / "c.net", readout=True)
    network = _read(path)
    readout = network.layers[-1]
    assert (readout.neuron["dv"], readout.neuron["vth"]) == (0, 131071)
    bias = _input_bias(network)
    lava = lava_run(network, bias, STEPS, builder="netx", path=path)
    ours, _ = _sanafe(network, bias, _cores(network, 1))
    _assert_equal_runs(lava, ours, _sizes(network))
    _assert_nonvacuous(lava, spiking=range(4))
    v = lava["voltages"][-1]
    assert not np.any(lava["spikes"][-1]), "the readout never reaches its threshold"
    assert np.any(v > 0) and np.any(v < 0), "the readout integrates both signs"
    assert len(np.unique(v)) > 10


def test_d_two_cores_equal_one_core(tmp_path):
    path = net_a(tmp_path / "d.net")
    network = _read(path)
    bias = _input_bias(network)
    sizes = _sizes(network)
    lava = lava_run(network, bias, STEPS, builder="netx", path=path)
    one, m1 = _sanafe(network, bias, _cores(network, 1))
    two, m2 = _sanafe(network, bias, _cores(network, 2))
    spread, m3 = _sanafe(network, bias, _cores(network, 2), order="spread")
    _assert_equal_runs(one, two, sizes)
    _assert_equal_runs(one, spread, sizes)
    _assert_equal_runs(lava, two, sizes)
    _assert_nonvacuous(lava, spiking=range(4))
    # Non-vacuous: every layer really occupies two cores, and the placements differ.
    for manifest in (m2, m3):
        for layer in manifest["placement"]:
            assert len({(s["tile"], s["core"]) for s in layer["slices"]}) == 2
    assert m2["placement"] != m3["placement"]
    assert all(len(c["synapses_per_dst_core"]) == 2 for c in m2["connections"])
    assert all(min(c["synapses_per_dst_core"]) > 0 for c in m2["connections"])


def test_e_netx_and_direct_builders_agree(tmp_path):
    path = net_a(tmp_path / "e.net")
    network = _read(path)
    bias = _input_bias(network)
    sizes = _sizes(network)
    by_netx = lava_run(network, bias, STEPS, builder="netx", path=path)
    direct = lava_run(network, bias, STEPS, builder="direct")
    _assert_equal_runs(by_netx, direct, sizes)
    ours, _ = _sanafe(network, bias, _cores(network, 1))
    _assert_equal_runs(by_netx, ours, sizes)
    _assert_equal_runs(direct, ours, sizes)
    _assert_nonvacuous(by_netx, spiking=range(4))


def test_f_zero_input_bias_and_stored_biases(tmp_path):
    # Conv biases of 30 and 40 exceed the threshold of 25 on their own.
    path = net_a(tmp_path / "f.net", conv_bias=(30, -6, 40))
    network = _read(path)
    bias = np.zeros(_sizes(network)[0], dtype=np.int64)
    lava = lava_run(network, bias, STEPS, builder="netx", path=path)
    ours, _ = _sanafe(network, bias, _cores(network, 1))
    _assert_equal_runs(lava, ours, _sizes(network))
    assert not np.any(lava["spikes"][0]) and not np.any(lava["voltages"][0])
    _assert_nonvacuous(lava, spiking=range(1, 4))


def test_g_asymmetric_stride_and_padding(tmp_path):
    path = net_g(tmp_path / "g.net")
    network = _read(path)
    conv = network.layers[1]
    assert conv.stride == (2, 1) and conv.padding == (1, 0)
    assert len({*network.layers[0].shape}) == 3 and len({*conv.shape}) == 3
    bias = _input_bias(network)
    lava = lava_run(network, bias, STEPS, builder="netx", path=path)
    ours, _ = _sanafe(network, bias, _cores(network, 1))
    _assert_equal_runs(lava, ours, _sizes(network))
    _assert_nonvacuous(lava, spiking=range(3))
    direct = lava_run(network, bias, STEPS, builder="direct")
    _assert_equal_runs(lava, direct, _sizes(network))


# ---------------------------------------------------------------------------
# The Lava side: how state is read and how the input bias arrives


def test_reading_equals_lava_monitors(tmp_path):
    """lava_run reads through VarWire and Extractor. A Monitor run of 16 steps agrees."""
    from lava.lib.dl import netx
    from lava.proc.monitor.process import Monitor
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.run_configs import Loihi2SimCfg

    steps = 16
    path = net_g(tmp_path / "m.net")
    network = _read(path)
    bias = _input_bias(network)
    ours = lava_run(network, bias, steps, builder="netx", path=path)
    net = netx.hdf5.Network(net_config=str(path), reset_interval=16, reset_offset=3)
    c, h, w = network.layers[0].shape
    net.layers[0].neuron.bias_mant.init = bias.reshape(c, h, w).transpose(2, 1, 0).astype(np.int32)
    monitors = []
    for block in net.layers:
        mv, ms = Monitor(), Monitor()
        mv.probe(block.neuron.v, steps)
        ms.probe(block.neuron.s_out, steps)
        monitors.append((mv, ms))
    try:
        net.run(condition=RunSteps(num_steps=steps), run_cfg=Loihi2SimCfg(select_tag="fixed_pt"))
        data = [(mv.get_data()[b.neuron.name]["v"], ms.get_data()[b.neuron.name]["s_out"])
                for b, (mv, ms) in zip(net.layers, monitors)]
    finally:
        net.stop()
    for k, (v, s) in enumerate(data):
        v, s = np.asarray(v), np.asarray(s)
        if v.ndim == 4:  # Monitor rows are (x, y, c), lava_run's are flattened (c, y, x)
            v, s = (a.transpose(0, 3, 2, 1).reshape(steps, -1) for a in (v, s))
        assert v.shape == s.shape == ours["voltages"][k].shape
        assert np.array_equal(v.astype(np.int64), ours["voltages"][k]), k
        assert np.array_equal(s.astype(np.int64), ours["spikes"][k]), k
    assert np.any(ours["spikes"][1])


def test_input_bias_reaches_the_netx_input_layer(tmp_path):
    path = net_g(tmp_path / "i.net", conv_bias=(0, 0, 0, 0, 0))
    network = _read(path)
    zero = np.zeros(_sizes(network)[0], dtype=np.int64)
    bias = _input_bias(network)
    silent = lava_run(network, zero, 8, builder="netx", path=path)
    driven = lava_run(network, bias, 8, builder="netx", path=path)
    assert not np.any(silent["spikes"][0]) and not np.any(silent["voltages"][0])
    assert np.any(driven["spikes"][0])
    # After the first update the input layer holds exactly its bias, 64 times the
    # mantissa, or zero where that exceeds the threshold of 25 * 64 and it spiked.
    first = np.where(bias * 64 > 25 * 64, 0, bias * 64)
    assert np.array_equal(driven["voltages"][0][0], first)
    assert np.array_equal(driven["spikes"][0][0], (bias * 64 > 25 * 64).astype(np.int64))


def test_a_stalled_tap_ends_the_run():
    import threading
    from sanafe.lif_network import _receive

    class Silent:
        def __init__(self):
            self.release = threading.Event()

        def receive(self):
            self.release.wait()
            return np.zeros(1)

    tap = Silent()
    try:
        with pytest.raises(TimeoutError, match="no step for 1 s"):
            _receive([tap], [], 3, stall_seconds=1)
    finally:
        tap.release.set()


# ---------------------------------------------------------------------------
# Expansion, placement, manifest and refusals


def test_conv_synapses_equal_lava_conv_to_sparse():
    """The expansion against Lava's conv_to_sparse in its default (x, y, c) order.

    That is the order lava_conv_integer.py uses. Lava receives the kernel as netx
    hands it (flipped, (out, x, y, in)) and its indices are mapped back to (c, h, w).
    Lava's NCHW branch is not used: it flips the input-channel and kernel-height
    axes of an (out, in, h, w) kernel (proc/conv/utils.py, NCHW branch).
    """
    from lava.proc.conv.utils import conv_to_sparse

    rng = np.random.default_rng(3)
    weights = rng.integers(-5, 6, size=(5, 3, 3, 2))
    (c, h, w), (co, ho, wo) = (3, 8, 11), (5, 4, 10)
    src, dst, weight = conv_synapses((c, h, w), weights, (2, 1), (1, 0), (co, ho, wo))
    kernel = weights[:, :, ::-1, ::-1].transpose(0, 3, 2, 1)
    lava_dst, lava_src, lava_w = conv_to_sparse((w, h, c), (wo, ho, co), kernel, (1, 2), (0, 1),
                                                (1, 1), 1)

    def chw(index, height, width, channels):  # Lava's x * H * C + y * C + c
        x, rest = np.divmod(np.asarray(index), height * channels)
        y, ch = np.divmod(rest, channels)
        return (ch * height + y) * width + x

    theirs = sorted(zip(chw(lava_src, h, w, c).tolist(), chw(lava_dst, ho, wo, co).tolist(),
                        np.asarray(lava_w).astype(int).tolist()))
    ours = sorted(zip(src.tolist(), dst.tolist(), weight.tolist()))
    assert [t for t in ours if t[2]] == [t for t in theirs if t[2]]
    assert sum(1 for t in ours if t[2]) > 100


def test_slices_and_core_order():
    assert slice_sizes(10, 3) == [4, 3, 3]
    assert slice_sizes(39600, 13) == [3047, 3047] + [3046] * 11
    assert slice_sizes(5, 5) == [1] * 5
    with pytest.raises(ValueError, match="more cores"):
        slice_sizes(3, 4)
    packed, spread = core_order("packed"), core_order("spread")
    assert packed[:5] == [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]
    assert spread[:3] == [(0, 0), (1, 0), (2, 0)] and spread[32] == (0, 1)
    assert len(packed) == len(spread) == 127 and (31, 3) not in packed + spread
    with pytest.raises(ValueError, match="packed"):
        core_order("diagonal")


def test_manifest(tmp_path):
    path = net_a(tmp_path / "man.net")
    network = _read(path)
    arch = load_loihi2_candidate()
    _, manifest = netx_to_sanafe(network, _input_bias(network), arch, [3, 2, 2, 1], order="packed")
    assert manifest["execution_profile"] == "lava-lif-int24-v1"
    assert manifest["soma_unit"] == "loihi_cuba_lif"
    assert manifest["hardware_validated"] is False
    assert manifest["layer_sizes"] == [126, 84, 48, 10]
    slices = manifest["placement"]
    assert [[(s["tile"], s["core"]) for s in layer["slices"]] for layer in slices] == [
        [(0, 0), (0, 1), (0, 2)], [(0, 3), (1, 0)], [(1, 1), (1, 2)], [(1, 3)]]
    assert [[(s["start"], s["stop"]) for s in layer["slices"]] for layer in slices][0] == [
        (0, 42), (42, 84), (84, 126)]
    conns = manifest["connections"]
    assert [c["kind"] for c in conns] == ["conv", "conv", "dense"]
    assert [c["effective_weights_shape"] for c in conns] == [[3, 2, 2, 3], [4, 3, 3, 2], [10, 48]]
    w = network.layers[1].weights
    expanded = sum(1 for o in range(3) for i in range(2) for a in range(2) for b in range(3)
                   for y in range(4) for x in range(7) if 0 <= x - 1 + b < 7)
    assert conns[0]["synapses_expanded"] == expanded
    assert conns[0]["synapses_emitted"] == expanded - sum(
        1 for o in range(3) for i in range(2) for a in range(2) for b in range(3)
        for y in range(4) for x in range(7) if 0 <= x - 1 + b < 7 and w[o, i, a, b] == 0)
    for c in conns:
        assert sum(c["synapses_per_dst_core"]) == c["synapses_emitted"]
    assert manifest["layers"][0]["reset_offset"] == 4


def _tiny(weights, exp=0, bits=8):
    n = np.asarray(weights).shape[1]
    layers = [NetxLayer(kind="input", shape=(n, 1, 1), neuron=dict(
                  du=4095, dv=1024, vth=25, bias_exp=6, reset_interval=None, reset_offset=None)),
              NetxLayer(kind="flatten", shape=(1, 1, n)),
              NetxLayer(kind="dense", shape=(1,), weights=np.asarray(weights), weight_exp=exp,
                        weight_bits=bits, neuron=dict(du=4095, dv=1024, vth=25, bias_exp=6,
                                                      reset_interval=None, reset_offset=None))]
    return NetxNetwork(layers=layers)


def test_refusals_use_the_messages_of_lava_lif():
    arch = load_loihi2_candidate()
    zero = np.zeros(48, dtype=np.int64)
    with pytest.raises(ValueError, match="binary fan-in can exceed the accepted LIF activation range"):
        netx_to_sanafe(_tiny(np.full((1, 48), 6000)), zero, arch, [1, 1])
    with pytest.raises(ValueError, match=r"effective weights outside declared range \[-32768, 32767\]"):
        netx_to_sanafe(_tiny(np.full((1, 48), 40000)), zero, arch, [1, 1])
    with pytest.raises(ValueError, match=r"bias_mant outside declared range \[-4096, 4095\]"):
        netx_to_sanafe(_tiny(np.ones((1, 48), dtype=int)), zero + 5000, arch, [1, 1])
    with pytest.raises(ValueError, match="one core count per layer"):
        netx_to_sanafe(_tiny(np.ones((1, 48), dtype=int)), zero, arch, [1])
    with pytest.raises(ValueError, match="more cores"):
        netx_to_sanafe(_tiny(np.ones((1, 48), dtype=int)), zero, arch, [1, 2])
    with pytest.raises(ValueError, match="needs 128 cores"):
        netx_to_sanafe(_tiny(np.ones((1, 200), dtype=int)), np.zeros(200, dtype=np.int64),
                       arch, [127, 1])
