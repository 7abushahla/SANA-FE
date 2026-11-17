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


# ---------------------------------------------------------------------------
# Convolution convention against lava-dl's netx and Lava's conv utility.


def _scalar(x):
    return int(np.asarray(x).reshape(-1)[0])


def _reader_conv(layer, x):
    """The cross-correlation of the module docstring, in plain loops.

    x has shape (in channels, height, width). Returns (out channels, out height,
    out width) as integers.
    """
    w = layer.weights
    n_out, n_in, kh, kw = w.shape
    sh, sw = layer.stride
    ph, pw = layer.padding
    _, height, width = x.shape
    out_h = (height + 2 * ph - kh) // sh + 1
    out_w = (width + 2 * pw - kw) // sw + 1
    out = np.zeros((n_out, out_h, out_w), dtype=np.int64)
    for o in range(n_out):
        for h in range(out_h):
            for v in range(out_w):
                total = 0
                for i in range(n_in):
                    for a in range(kh):
                        for b in range(kw):
                            r = h * sh + a - ph
                            c = v * sw + b - pw
                            if 0 <= r < height and 0 <= c < width:
                                total += w[o, i, a, b] * x[i, r, c]
                out[o, h, v] = total
    return out


def _lava_conv(block, x):
    """Lava's conv utility on netx's weights, input in Lava's (x, y, c) order.

    Returns the output mapped back to (out channels, height, width).
    """
    from lava.proc.conv import utils as conv_utils
    syn = block.synapse
    weight = np.asarray(syn.weight.init)
    x_lava = x.transpose(2, 1, 0).astype(np.float64)
    out = conv_utils.conv(
        x_lava, weight, tuple(weight.shape[1:3]),
        tuple(int(v) for v in syn.proc_params["stride"]),
        tuple(int(v) for v in syn.proc_params["padding"]),
        (1, 1), 1)
    return np.rint(out).astype(np.int64).transpose(2, 1, 0)


def test_conv_convention_agrees_with_lava_netx(tmp_path):
    """Axis order, kernel orientation, stride and padding against Lava.

    Every size differs (in channels 2, out channels 3, height 8, width 9, kernel
    height 2, kernel width 3), the kernel is asymmetric, the stride is (2, 1) and
    the padding is (0, 1), so no transposition or flip can pass by accident.
    """
    rng = np.random.default_rng(11)
    w = rng.integers(-5, 6, size=(3, 2, 2, 3))
    path = _write(tmp_path / "conv.net", [
        _input((2, 8, 9)),
        ("conv", (3, 4, 9), True, {
            "weight": w.astype(np.float32),
            "bias": np.array([1, -2, 3], dtype=np.float32),
            "stride": np.array([2, 1]), "padding": np.array([0, 1]),
            "dilation": np.array([1, 1]), "groups": 1,
            "kernelSize": np.array([2, 3]), "inChannels": 2, "outChannels": 3}),
    ])
    netx = _lava_netx()
    ours = read_netx(path).layers[1]
    block = netx.hdf5.Network(net_config=str(path)).layers[1]
    assert (ours.stride, ours.padding) == ((2, 1), (0, 1))
    # netx hands Lava (x, y) pairs, the reverse of the file's (height, width).
    assert tuple(int(v) for v in block.synapse.proc_params["stride"]) == (1, 2)
    assert tuple(int(v) for v in block.synapse.proc_params["padding"]) == (1, 0)
    for seed in range(3):
        x = np.random.default_rng(seed).integers(0, 2, size=(2, 8, 9))
        mine = _reader_conv(ours, x)
        theirs = _lava_conv(block, x)
        assert mine.shape == theirs.shape == (3, 4, 9)
        assert np.array_equal(mine, theirs)
        assert np.abs(mine).sum() > 0


def test_real_file_agrees_with_lava_netx_on_every_layer():
    """Every layer of the PilotNet file, reader against netx, after axis mapping."""
    assert PILOTNET.is_file(), (
        f"{PILOTNET} is missing. Run "
        "SANA-FE-thesis/calibration/deployments/pilotnet_lif/fetch.py to fetch it."
    )
    netx = _lava_netx()
    ours = read_netx(PILOTNET, reset_interval=16, reset_offset=3)
    theirs = netx.hdf5.Network(net_config=str(PILOTNET), reset_interval=16,
                               reset_offset=3)
    mine = [layer for layer in ours.layers if layer.kind != "flatten"]
    assert len(mine) == len(theirs.layers) == 10
    previous_shape = None
    for k, (m, block) in enumerate(zip(mine, theirs.layers)):
        lif = block.neuron
        for key, var in (("du", lif.du), ("dv", lif.dv), ("vth", lif.vth),
                         ("bias_exp", lif.bias_exp)):
            assert m.neuron[key] == _scalar(var.init), (k, key)
        assert m.neuron["reset_interval"] == lif.proc_params["reset_interval"]
        assert m.neuron["reset_offset"] == lif.proc_params["reset_offset"]
        if m.kind == "input":
            assert m.shape == tuple(block.shape[::-1])
            previous_shape = m.shape
            continue
        syn = block.synapse
        bias_mant = np.asarray(lif.bias_mant.init)
        if m.kind == "conv":
            assert m.shape == tuple(block.shape[::-1])
            w_lava = np.asarray(syn.weight.init)  # (out, x, y, in), kernel flipped
            assert np.array_equal(m.weights, w_lava[:, ::-1, ::-1, :].transpose(0, 3, 2, 1))
            assert m.weight_exp == _scalar(syn.weight_exp.init)
            assert m.weight_bits == _scalar(syn.num_weight_bits.init)
            assert tuple(m.stride) == tuple(int(v) for v in syn.proc_params["stride"])[::-1]
            assert tuple(m.padding) == tuple(int(v) for v in syn.proc_params["padding"])[::-1]
            assert np.array_equal(np.broadcast_to(m.bias, bias_mant.shape), bias_mant)
            # The same random input through both conventions on this layer.
            x = np.random.default_rng(k).integers(0, 2, size=previous_shape)
            assert np.array_equal(_reader_conv(m, x), _lava_conv(block, x)), k
            previous_shape = m.shape
        else:
            assert m.shape == (block.shape[0],)
            assert np.array_equal(m.weights, np.asarray(syn.weights.init))
            assert m.weight_exp == syn.proc_params["weight_exp"]
            assert m.weight_bits == syn.proc_params["num_weight_bits"]
            assert np.array_equal(m.bias, bias_mant)


# ---------------------------------------------------------------------------
# Refusals and consistency checks.


def _one_dense(tmp_path, weights, name="one.net", neuron=True, **extra):
    weights = np.asarray(weights, dtype=np.float32)
    out, inp = weights.shape
    return _write(tmp_path / name, [
        _input((inp, 1, 1)),
        ("flatten", (1, 1, inp), False, {}),
        ("dense", (out, 1, 1), neuron, {"weight": weights, **extra}),
    ])


def test_more_than_eight_weight_bits_refused(tmp_path):
    # Odd weights with scale_bits 0 give n = -1 and 9 bits under netx's rule.
    with pytest.raises(ValueError, match="9 weight bits"):
        read_netx(_one_dense(tmp_path, [[149, -101, 1]]))


def test_lava_clip_refused_when_sign_mode_differs(tmp_path):
    # netx sees a mixed layer (max 0 is not below 0). Lava sees an inhibitory
    # layer (max <= 0), negates it, and clips the weight -256 to 255.
    with pytest.raises(ValueError, match="clip or truncate"):
        read_netx(_one_dense(tmp_path, [[-4, 0, -2]]))


def test_lava_clip_and_truncate_keep_ordinary_layers(tmp_path):
    inhibitory = read_netx(_one_dense(tmp_path, [[-4, -2, -6]], "inh.net")).layers[2]
    assert np.array_equal(inhibitory.weights, np.array([[-128, -64, -192]]))
    excitatory = read_netx(_one_dense(tmp_path, [[4, 0, 6]], "exc.net")).layers[2]
    assert excitatory.weight_exp < 0
    assert np.array_equal(excitatory.effective_weights(), np.array([[4, 0, 6]]))


def test_all_zero_dense_weights_refused(tmp_path):
    with pytest.raises(ValueError, match="all zero"):
        read_netx(_one_dense(tmp_path, [[0, 0], [0, 0]]))


def test_nonzero_message_bits_refused(tmp_path):
    path = tmp_path / "m.net"
    with h5py.File(path, "w") as f:
        g = _layer(f, 0, "input", (1, 2, 2), neuron=False)
        _neuron(g.create_group("neuron"), messageBits=8)
    with pytest.raises(ValueError, match="messageBits"):
        read_netx(path)
    ok = tmp_path / "m0.net"
    with h5py.File(ok, "w") as f:
        g = _layer(f, 0, "input", (1, 2, 2), neuron=False)
        _neuron(g.create_group("neuron"), messageBits=0)
    assert read_netx(ok).layers[0].neuron["vth"] == 25


def test_missing_neuron_group_refused(tmp_path):
    with pytest.raises(ValueError, match="no neuron group"):
        read_netx(_one_dense(tmp_path, [[1, 2]], neuron=False))


def _conv_file(tmp_path, name="cv.net", weight=None, **override):
    weight = np.ones((2, 3, 2, 2), dtype=np.float32) if weight is None else weight
    datasets = {
        "weight": weight, "stride": np.array([1, 1]), "padding": np.array([0, 0]),
        "dilation": np.array([1, 1]), "groups": 1, "kernelSize": np.array([2, 2]),
        "inChannels": 3, "outChannels": 2,
    }
    datasets.update(override)
    datasets = {k: v for k, v in datasets.items() if v is not None}
    return _write(tmp_path / name, [_input((3, 4, 4)),
                                    ("conv", (2, 3, 3), True, datasets)])


def test_size_attributes_checked_against_weights(tmp_path):
    assert read_netx(_conv_file(tmp_path)).layers[1].weights.shape == (2, 3, 2, 2)
    with pytest.raises(ValueError, match="outChannels"):
        read_netx(_conv_file(tmp_path, "a.net", outChannels=5))
    with pytest.raises(ValueError, match="inChannels"):
        read_netx(_conv_file(tmp_path, "b.net", inChannels=4))
    with pytest.raises(ValueError, match="kernelSize"):
        read_netx(_conv_file(tmp_path, "c.net", kernelSize=np.array([3, 2])))
    with pytest.raises(ValueError, match="inFeatures"):
        read_netx(_one_dense(tmp_path, [[1, 2]], "d.net", inFeatures=3))
    with pytest.raises(ValueError, match="outFeatures"):
        read_netx(_one_dense(tmp_path, [[1, 2]], "e.net", outFeatures=2))
    assert read_netx(_one_dense(tmp_path, [[1, 2]], "f.net", inFeatures=2,
                                outFeatures=1)).layers[2].weights.shape == (1, 2)


@pytest.mark.parametrize("missing", ["stride", "padding", "dilation", "groups"])
def test_missing_conv_attribute_is_value_error(tmp_path, missing):
    with pytest.raises(ValueError, match=missing):
        read_netx(_conv_file(tmp_path, f"{missing}.net", **{missing: None}))


def test_scalar_stride_applies_to_both_axes(tmp_path):
    layer = read_netx(_conv_file(tmp_path, stride=np.int64(2))).layers[1]
    assert layer.stride == (2, 2)


def test_dense_weight_with_extra_axes_refused(tmp_path):
    path = _write(tmp_path / "x.net", [
        _input((2, 1, 1)), ("flatten", (1, 1, 2), False, {}),
        ("dense", (1, 1, 1), True, {"weight": np.ones((1, 2, 1), dtype=np.float32)}),
    ])
    with pytest.raises(ValueError, match="axes"):
        read_netx(path)


def test_conv_weights_outside_eight_bits_refused(tmp_path):
    edge = np.full((2, 3, 2, 2), 127, dtype=np.float32)
    edge[0, 0, 0, 0] = -128
    assert read_netx(_conv_file(tmp_path, "edge.net", weight=edge)).layers[1].weights.min() == -128
    for bad in (128, -129):
        w = np.ones((2, 3, 2, 2), dtype=np.float32)
        w[1, 2, 1, 1] = bad
        with pytest.raises(ValueError, match="8-bit"):
            read_netx(_conv_file(tmp_path, f"bad{bad}.net", weight=w))
