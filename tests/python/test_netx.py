"""Tests of the netx file reader (sanafe.netx).

Small files are written with h5py in the layout of lava-dl's netx files.
The weight rule is checked against lava-dl's own netx, which is imported
from the git-ignored checkout and is used by these tests only.
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from sanafe.netx import NetxLayer, NetxNetwork, read_netx

REPO = Path(__file__).resolve().parents[3]
FACTS_DIR = REPO / "SANA-FE-thesis" / "calibration" / "deployments" / "pilotnet_lif"
LAVA_DL_SRC = REPO / "data" / "calibration" / "lava-dl" / "src"
PILOTNET = (
    REPO / "data" / "calibration" / "lava-dl" / "tutorials" / "lava" / "lib" / "dl"
    / "netx" / "pilotnet_snn" / "network.net"
)


def _neuron(group, kind="LOIHI", **override):
    values = dict(iDecay=4096, refDelay=1, scaleRho=1.0, tauRho=0.5,
                  theta=1600, vDecay=1024, vThMant=25, wgtExp=0)
    values.update(override)
    group.create_dataset("type", data=np.bytes_(kind), dtype=h5py.string_dtype("ascii"))
    for key, value in values.items():
        group.create_dataset(key, data=value)


def _layer(f, index, kind, shape, neuron=True, **datasets):
    group = f.require_group("layer").create_group(str(index))
    group.create_dataset("type", data=np.array([kind.encode()]))
    group.create_dataset("shape", data=np.array(shape, dtype=np.int64))
    for key, value in datasets.items():
        group.create_dataset(key, data=value)
    if neuron:
        _neuron(group.create_group("neuron"))
    return group


def _write(path, layers):
    with h5py.File(path, "w") as f:
        f.require_group("layer")
        f.create_group("simulation").create_dataset("Ts", data=1)
        for index, (kind, shape, neuron, datasets) in enumerate(layers):
            _layer(f, index, kind, shape, neuron, **datasets)
    return path


def _input(shape=(1, 4, 4)):
    return ("input", shape, True, {"weight": np.float32(40), "bias": np.float32(36)})


def _dense_file(tmp_path, weights, name="net.net", shape=None):
    weights = np.asarray(weights, dtype=np.float32)
    out, inp = weights.shape
    return _write(tmp_path / name, [
        _input((inp, 1, 1)),
        ("flatten", (1, 1, inp), False, {}),
        ("dense", (out, 1, 1), True, {"weight": weights,
                                      "bias": np.arange(out, dtype=np.float32)}),
    ])


def test_dense_with_exponent_zero(tmp_path):
    # scale = min(127/150, 128/100) = 0.85, floor(log2) = -1, plus 1 for a signed
    # layer gives scale_bits 0 and exponent 0. Every weight is even, so n = 1 - 1.
    w = [[150, -100, 4], [2, -2, 0]]
    net = read_netx(_dense_file(tmp_path, w))
    dense = net.layers[2]
    assert dense.kind == "dense"
    assert dense.shape == (2,)
    assert dense.weight_exp == 0
    assert dense.weight_bits == 8
    assert np.array_equal(dense.weights, np.array(w))
    assert np.array_equal(dense.effective_weights(), np.array(w))
    assert np.array_equal(dense.bias, [0, 1])
    assert net.layers[0].kind == "input"


def test_dense_negative_exponent_divides(tmp_path):
    # max 6, min -6: scale = min(127/6, 128/6) = 21.2, scale_bits = floor(log2) + 1 = 5.
    w = [[6, -6, 2], [0, 4, -2]]
    dense = read_netx(_dense_file(tmp_path, w)).layers[2]
    assert dense.weight_exp == -5
    assert dense.weight_bits == 3
    assert np.array_equal(dense.weights, np.array(w) * 32)
    assert np.array_equal(dense.effective_weights(), np.array(w))


def test_dense_negative_exponent_not_divisible_raises(tmp_path):
    # A layer built directly with an applied weight that exponent -5 does not divide.
    layer = NetxLayer(kind="dense", shape=(1,), weights=np.array([[33, 32]]),
                      weight_exp=-5, weight_bits=3)
    with pytest.raises(ValueError, match="multiple"):
        layer.effective_weights()


def test_positive_exponent_shifts_left():
    layer = NetxLayer(kind="dense", shape=(1,), weights=np.array([[3, -1]]),
                      weight_exp=2, weight_bits=8)
    assert np.array_equal(layer.effective_weights(), np.array([[12, -4]]))


def test_fractional_weights_raise(tmp_path):
    with pytest.raises(ValueError, match="integer"):
        read_netx(_dense_file(tmp_path, [[1.5, 2.0]]))


def test_conv_stride_padding_and_weights(tmp_path):
    w = np.arange(2 * 1 * 3 * 3, dtype=np.float32).reshape(2, 1, 3, 3) - 9
    path = _write(tmp_path / "c.net", [
        _input((1, 5, 6)),
        ("conv", (2, 2, 6), True, {
            "weight": w, "bias": np.array([1, -2], dtype=np.float32),
            "stride": np.array([2, 1]), "padding": np.array([0, 1]),
            "dilation": np.array([1, 1]), "groups": 1,
            "kernelSize": np.array([3, 3]), "inChannels": 1, "outChannels": 2}),
    ])
    conv = read_netx(path).layers[1]
    assert conv.kind == "conv"
    assert conv.shape == (2, 2, 6)
    assert conv.stride == (2, 1)
    assert conv.padding == (0, 1)
    assert conv.dilation == (1, 1)
    assert conv.groups == 1
    assert conv.weight_exp == 0 and conv.weight_bits == 8
    assert conv.weights.shape == (2, 1, 3, 3)
    assert np.array_equal(conv.weights, w.astype(int))
    assert np.array_equal(conv.effective_weights(), w.astype(int))
    assert np.array_equal(conv.bias, [1, -2])


def test_reset_interval_and_offset(tmp_path):
    w = [[1, -1], [2, 0]]
    path = _write(tmp_path / "r.net", [
        _input((2, 1, 1)),
        ("dense", (2, 1, 1), True, {"weight": np.array(w, dtype=np.float32)}),
        ("dense", (2, 1, 1), True, {"weight": np.array(w, dtype=np.float32)}),
    ])
    net = read_netx(path, reset_interval=16, reset_offset=3)
    offsets = [layer.neuron["reset_offset"] for layer in net.layers]
    assert offsets == [4, 5, 6]
    assert [layer.neuron["reset_interval"] for layer in net.layers] == [16] * 3
    assert [layer.resets_at for layer in net.layers] == [4, 5, 6]
    n = net.layers[1].neuron
    assert (n["du"], n["dv"], n["vth"], n["bias_exp"]) == (4095, 1024, 25, 6)
    # An offset that reaches the interval wraps, as Lava stores offset mod interval.
    wrapped = read_netx(path, reset_interval=5, reset_offset=3)
    assert [layer.resets_at for layer in wrapped.layers] == [4, 0, 1]
    # No reset requested.
    plain = read_netx(path)
    assert plain.layers[1].neuron["reset_interval"] is None
    assert plain.layers[1].resets_at is None


def test_flatten_has_no_neuron_and_does_not_advance_offset(tmp_path):
    path = _write(tmp_path / "f.net", [
        _input((1, 2, 2)),
        ("flatten", (1, 1, 4), False, {}),
        ("dense", (1, 1, 1), True, {"weight": np.array([[1, 0, 0, 2]], dtype=np.float32)}),
    ])
    net = read_netx(path, reset_interval=16, reset_offset=0)
    kinds = [layer.kind for layer in net.layers]
    assert kinds == ["input", "flatten", "dense"]
    flat = net.layers[1]
    assert flat.neuron is None and flat.weights is None
    assert [net.layers[0].resets_at, net.layers[2].resets_at] == [1, 2]


def test_input_transform(tmp_path):
    path = _write(tmp_path / "t.net", [_input((1, 2, 2))])
    layer = read_netx(path).layers[0]
    assert layer.transform == {"weight": 40, "bias": 36}
    assert layer.shape == (1, 2, 2)
    assert layer.weights is None


def test_unknown_layer_type_raises(tmp_path):
    path = _write(tmp_path / "u.net", [_input(), ("pool", (1, 2, 2), False, {})])
    with pytest.raises(ValueError, match="pool"):
        read_netx(path)


def test_other_neuron_type_raises(tmp_path):
    path = tmp_path / "n.net"
    with h5py.File(path, "w") as f:
        g = _layer(f, 0, "input", (1, 2, 2), neuron=False)
        _neuron(g.create_group("neuron"), kind="SDNN")
    with pytest.raises(ValueError, match="SDNN"):
        read_netx(path)


def test_delay_raises(tmp_path):
    path = _write(tmp_path / "d.net", [
        _input((2, 1, 1)),
        ("dense", (1, 1, 1), True, {"weight": np.array([[1, 2]], dtype=np.float32),
                                    "delay": np.array([0.0])}),
    ])
    with pytest.raises(ValueError, match="delay"):
        read_netx(path)


def test_unknown_attribute_raises(tmp_path):
    path = _write(tmp_path / "a.net", [
        _input((2, 1, 1)),
        ("dense", (1, 1, 1), True, {"weight": np.array([[1, 2]], dtype=np.float32),
                                    "mystery": np.array([1])}),
    ])
    with pytest.raises(ValueError, match="mystery"):
        read_netx(path)


def test_dilation_and_groups_refused(tmp_path):
    def conv(dilation, groups):
        return _write(tmp_path / f"g{dilation}{groups}.net", [
            _input((1, 5, 5)),
            ("conv", (1, 3, 3), True, {
                "weight": np.ones((1, 1, 3, 3), dtype=np.float32),
                "stride": np.array([1, 1]), "padding": np.array([0, 0]),
                "dilation": np.array([dilation, dilation]), "groups": groups}),
        ])
    with pytest.raises(ValueError, match="dilation"):
        read_netx(conv(2, 1))
    with pytest.raises(ValueError, match="groups"):
        read_netx(conv(1, 2))


def test_reads_pilotnet():
    assert PILOTNET.is_file(), (
        f"{PILOTNET} is missing. Run "
        "SANA-FE-thesis/calibration/deployments/pilotnet_lif/fetch.py to fetch it."
    )
    facts = json.loads((FACTS_DIR / "network_facts.json").read_text())
    net = read_netx(PILOTNET, reset_interval=16, reset_offset=3)
    assert isinstance(net, NetxNetwork)
    assert len(net.layers) == len(facts["layers"])
    for layer, fact in zip(net.layers, facts["layers"]):
        assert layer.kind == fact["type"]
        n_shape = tuple(fact["shape"])
        if layer.kind == "dense":
            assert layer.shape == (int(np.prod(n_shape)),)
        elif layer.kind == "flatten":
            continue
        else:
            assert layer.shape == n_shape
    weights = sum(layer.weights.size for layer in net.layers if layer.weights is not None)
    biases = sum(layer.bias.size for layer in net.layers if layer.bias is not None)
    assert weights == 349_998
    assert biases == 397
    dense = [layer for layer in net.layers if layer.kind == "dense"]
    assert [layer.weight_exp for layer in dense] == [-5, -4, -3, -2]
    assert [layer.weight_bits for layer in dense] == [3, 4, 5, 6]
    for layer in net.layers:
        if layer.weights is not None:
            layer.effective_weights()
    with_neuron = [layer for layer in net.layers if layer.neuron is not None]
    assert [layer.resets_at for layer in with_neuron] == list(range(4, 14))
    conv4 = net.layers[4]
    assert conv4.stride == (2, 1) and conv4.padding == (1, 0)
    assert net.layers[0].transform == {"weight": 40, "bias": 36}


def _lava_netx():
    src = str(LAVA_DL_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    from lava.lib.dl import netx
    return netx


def test_weight_rule_agrees_with_lava_netx(tmp_path):
    """Independent check of the weight rule against lava-dl's own netx."""
    assert LAVA_DL_SRC.is_dir(), (
        f"{LAVA_DL_SRC} is missing. Run "
        "SANA-FE-thesis/calibration/deployments/pilotnet_lif/fetch.py to fetch it."
    )
    rng = np.random.default_rng(7)
    w1 = rng.integers(-6, 7, size=(5, 8)) * 2
    w2 = rng.integers(-9, 12, size=(3, 5))
    w3 = rng.integers(0, 5, size=(2, 3))  # excitatory only
    path = _write(tmp_path / "lava.net", [
        _input((8, 1, 1)),
        ("flatten", (1, 1, 8), False, {}),
        ("dense", (5, 1, 1), True, {"weight": w1.astype(np.float32),
                                    "bias": np.arange(5, dtype=np.float32)}),
        ("dense", (3, 1, 1), True, {"weight": w2.astype(np.float32)}),
        ("dense", (2, 1, 1), True, {"weight": w3.astype(np.float32)}),
    ])
    ours = read_netx(path, reset_interval=16, reset_offset=3)
    netx = _lava_netx()
    theirs = netx.hdf5.Network(net_config=str(path), reset_interval=16, reset_offset=3)
    ours_layers = [layer for layer in ours.layers if layer.kind != "flatten"]
    assert len(theirs.layers) == len(ours_layers)
    for mine, block in zip(ours_layers[1:], theirs.layers[1:]):
        syn = block.synapse
        assert np.array_equal(mine.weights, np.asarray(syn.weights.init))
        assert mine.weight_exp == syn.proc_params["weight_exp"]
        assert mine.weight_bits == syn.proc_params["num_weight_bits"]
        lif = block.neuron
        assert mine.neuron["reset_offset"] == lif.proc_params["reset_offset"]
        assert mine.neuron["reset_interval"] == lif.proc_params["reset_interval"]
        assert mine.neuron["vth"] == int(np.asarray(lif.vth.init).reshape(-1)[0])
        assert mine.neuron["du"] == int(np.asarray(lif.du.init).reshape(-1)[0])
        assert mine.neuron["dv"] == int(np.asarray(lif.dv.init).reshape(-1)[0])
        assert mine.neuron["bias_exp"] == int(np.asarray(lif.bias_exp.init).reshape(-1)[0])
        if mine.bias is not None:
            assert np.array_equal(mine.bias, np.asarray(block.neuron.bias_mant.init))
