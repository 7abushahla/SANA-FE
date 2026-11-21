"""The sigma-delta soma ``sigma_delta_int32`` and the sigma soma ``sigma_int32`` against Lava's
fixed-point ``SigmaDelta`` and ``Sigma`` processes, fed by a ``RingBuffer`` through ``Dense`` or
``Conv`` with 16 message bits.

Alignment in time. A ``RingBuffer`` sends its value of step t at Lava update t. ``Dense`` and
``Conv`` send at each update the activation they accumulated on the previous one, so that value
reaches the neuron at update t + 1, and the neuron receives zero at update 0. A graded source
sends ``graded_values[t]`` at update t, and the buffer before the soma delivers it at update
t + 1, so the soma receives nothing at update 0. In both, the first update therefore runs on
the bias alone, and a bias at or above the threshold sends at that update in both. Row t of the
two traces is compared directly, row 0 included. Each run lasts one update more than the values,
so that the last value sent is received. At that extra update the ``RingBuffer`` sends its first
value again, which would arrive after the run ends. ``test_one_neuron_shows_the_alignment``
shows the alignment on one neuron whose bias sends at update 0.

Alignment in scale. Lava's process multiplies ``vth`` and ``bias`` by
``2**(spike_exp + state_exp)``, here ``2**12``. The neuron takes the multiplied values, so Lava's
``SigmaDelta(vth=6, bias=b, spike_exp=6, state_exp=6)`` is the soma with ``vth = 6 << 12``,
``bias = b << 12`` and ``state_exp = 6``. Lava's ``s_out`` is the soma's ``sent``.

Range. Lava's fixed-point ``Conv`` wraps its output to signed 24 bits. The integer accumulator
does not, and the soma reports the largest magnitude of its input in ``max_abs_input``, so a
convolution whose sum leaves 24 bits is visible there.
"""
import numpy as np
import pytest

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.conv.process import Conv
from lava.proc.conv.utils import conv_to_sparse, signed_clamp
from lava.proc.dense.process import Dense
from lava.proc.io.source import RingBuffer
from lava.proc.monitor.process import Monitor
from lava.proc.sdn.models import PySigmaDeltaModelFixed, PySigmaDeltaModelFixedCorrected
from lava.proc.sdn.process import Sigma, SigmaDelta
from lava.utils.weightutils import clip_weights, determine_sign_mode, truncate_weights

from sanafe.lava_sdnn import load_loihi2_sdnn_candidate
from test_graded_messages import make_chip, run, sources

pytestmark = pytest.mark.thesis

MODELS = [(PySigmaDeltaModelFixed, False), (PySigmaDeltaModelFixedCorrected, True)]
EXP = 6                                          # spike_exp and state_exp of every SigmaDelta
STATES = ("sigma", "act", "residue")
INPUT_SHAPE = (5, 5, 3)


def lava_run(port, process, values, steps, run_cfg, names):
    """Send ``values`` (``[sources, steps]`` or ``[*shape, steps]``) from a ``RingBuffer`` into
    ``port`` and run ``steps`` updates.

    Returns the Vars or ports of ``process`` named in ``names``, each as ``[steps, neurons]`` in
    int64, with the neurons in Lava's flattened order.
    """
    source = RingBuffer(data=np.asarray(values, dtype=np.int32))
    source.s_out.connect(port)
    monitors = {name: Monitor() for name in names}
    for name, monitor in monitors.items():
        monitor.probe(getattr(process, name), steps)
    try:
        process.run(condition=RunSteps(num_steps=steps), run_cfg=run_cfg)
        data = {name: np.asarray(monitor.get_data()[process.name][name]).astype(np.int64)
                for name, monitor in monitors.items()}
    finally:
        process.stop()
    return {name: array.reshape(steps, -1) for name, array in data.items()}


def lava_sigma_delta(connection, port, shape, bias, values, steps, model):
    """``connection`` into ``SigmaDelta(vth=6, bias, spike_exp=6, state_exp=6)`` under ``model``."""
    layer = SigmaDelta(shape=shape, vth=6, bias=np.asarray(bias).reshape(shape), spike_exp=EXP,
                       state_exp=EXP)
    connection.a_out.connect(layer.a_in)
    cfg = Loihi2SimCfg(select_tag="fixed_pt", exception_proc_model_map={SigmaDelta: model})
    return lava_run(port, layer, values, steps, cfg, [*STATES, "s_out"])


def sanafe_layer(values, edges, neurons, unit, steps):
    """Graded sources holding the rows of ``values`` into one neuron per entry of ``neurons``
    on ``unit``, connected by ``edges`` of ``(source, destination, weight)``, on the derived
    architecture. Returns every neuron trace as ``[steps, neurons]`` in int64."""
    count = len(neurons)
    chip, _ = make_chip([sources(np.asarray(values).tolist()),
                         ("layer_0", unit, neurons, [(0, 1)] * count)],
                        [("input", int(s), "layer_0", int(d), int(w)) for s, d, w in edges],
                        arch=load_loihi2_sdnn_candidate())
    traces = run(chip, steps, neuron_trace=True)["neuron_trace"]
    out = {k: np.asarray(v, dtype=np.int64) for k, v in traces.items()}
    assert all(v.shape == (steps, count) for v in out.values())
    return out


def sanafe_sigma_delta(values, edges, bias, steps, corrected):
    neurons = [dict(vth=6 << 12, bias=int(b) << 12, state_exp=EXP, corrected=corrected)
               for b in np.ravel(bias)]
    return sanafe_layer(values, edges, neurons, "loihi_sigma_delta", steps)


def dense_fixture():
    """Six sources over eight steps, values integers in [-8192, 8192] with about a third set to
    zero, weights (5, 6) even integers in [-30, 30], and a bias per neuron in [-20, 60].

    The values are not all multiples of 64. With multiples of 64 every change of the activation
    is a multiple of ``2**state_exp``, the shift drops nothing, and the two fixed-point models
    send the same values.
    """
    rng = np.random.default_rng(3)
    values = rng.integers(-8192, 8192, size=(6, 8), endpoint=True)
    values[rng.random(values.shape) < 1 / 3] = 0
    weights = 2 * rng.integers(-15, 15, size=(5, 6), endpoint=True)
    bias = rng.integers(-20, 60, size=5, endpoint=True)
    return values, weights, bias


def dense_edges(weights):
    """The weights Lava's ``Dense`` applies, as ``sanafe.lava_integer._dense_weights``
    computes them, as edges."""
    mode = determine_sign_mode(weights)
    effective = truncate_weights(clip_weights(weights, mode, num_bits=8), mode, 8)
    return [(s, d, effective[d, s]) for d in range(weights.shape[0])
            for s in range(weights.shape[1])]


def dense_lava(values, weights, bias, model):
    dense = Dense(weights=weights, num_message_bits=16)
    return lava_sigma_delta(dense, dense.s_in, (weights.shape[0],), bias, values,
                            values.shape[1] + 1, model)


def conv_fixture():
    """75 sources in the shape (5, 5, 3) over six steps, values as in the dense fixture, weight
    (4, 3, 3, 3) even integers in [-30, 30], and a bias per output neuron in [-20, 60]."""
    rng = np.random.default_rng(4)
    values = rng.integers(-8192, 8192, size=(*INPUT_SHAPE, 6), endpoint=True)
    values[rng.random(values.shape) < 1 / 3] = 0
    weight = 2 * rng.integers(-15, 15, size=(4, 3, 3, 3), endpoint=True)
    bias = rng.integers(-20, 60, size=5 * 5 * 4, endpoint=True)
    return values, weight, bias


def conv_both(values, weight, bias, model, corrected):
    """The convolution in Lava and on the derived architecture, with the sources and the
    destinations both in Lava's flattened order, as in ``sanafe/lava_conv_integer.py``."""
    steps = values.shape[-1] + 1
    conv = Conv(weight=weight, input_shape=INPUT_SHAPE, padding=(1, 1), num_message_bits=16)
    shape = tuple(conv.output_shape)
    lava = lava_sigma_delta(conv, conv.s_in, shape, bias, values, steps, model)
    dst, src, kernel = conv_to_sparse(INPUT_SHAPE, shape, weight, (1, 1), (1, 1), (1, 1), 1)
    ours = sanafe_sigma_delta(values.reshape(-1, values.shape[-1]), list(zip(src, dst, kernel)),
                              bias, steps, corrected)
    return ours, lava


def test_one_neuron_shows_the_alignment():
    # One source, weight 2, bias 10 << 12 above the threshold 6 << 12. Both neurons send
    # (10 << 12) >> 6 = 640 at update 0, before any input has arrived, and the value sent at
    # update t arrives at update t + 1.
    values = np.array([[64 * 500, 0, -64 * 900, 64 * 700]])
    weights = np.array([[2]])
    for model, corrected in MODELS:
        lava = dense_lava(values, weights, [10], model)
        ours = sanafe_sigma_delta(values, [(0, 0, 2)], [10], 5, corrected)
        assert ours["input"].ravel().tolist() == [0, 64000, 0, -115200, 89600]
        assert lava["s_out"].ravel().tolist() == [640, 1000, 0, -1640, 1240]
        for k in STATES:
            assert np.array_equal(ours[k], lava[k]), k
        assert np.array_equal(ours["sent"], lava["s_out"])


@pytest.mark.parametrize("model,corrected", MODELS)
def test_dense_into_sigma_delta_matches_lava(model, corrected):
    values, weights, bias = dense_fixture()
    lava = dense_lava(values, weights, bias, model)
    ours = sanafe_sigma_delta(values, dense_edges(weights), bias, values.shape[1] + 1, corrected)
    for k in STATES:
        assert np.array_equal(ours[k], lava[k]), k
    assert np.array_equal(ours["sent"], lava["s_out"]) and np.any(lava["s_out"] < 0)


@pytest.mark.parametrize("model,corrected", MODELS)
def test_conv_into_sigma_delta_matches_lava(model, corrected):
    ours, lava = conv_both(*conv_fixture(), model, corrected)
    for k in STATES:
        assert np.array_equal(ours[k], lava[k]), k
    assert np.array_equal(ours["sent"], lava["s_out"])
    assert np.abs(ours["max_abs_input"][-1]).max() < (1 << 23)   # inside Lava's 24-bit wrap


def test_dense_into_sigma_matches_lava():
    rng = np.random.default_rng(5)
    values = rng.integers(-8192, 8192, size=(6, 8), endpoint=True)
    values[rng.random(values.shape) < 1 / 3] = 0
    weights = 2 * rng.integers(-15, 15, size=(3, 6), endpoint=True)
    steps = values.shape[1] + 1
    dense = Dense(weights=weights, num_message_bits=16)
    layer = Sigma(shape=(3,))
    dense.a_out.connect(layer.a_in)
    lava = lava_run(dense.s_in, layer, values, steps, Loihi2SimCfg(select_tag="fixed_pt"),
                    ["sigma", "s_out"])
    ours = sanafe_layer(values, dense_edges(weights), [{}] * 3, "loihi_sigma", steps)
    assert np.array_equal(ours["sigma"], lava["sigma"])
    assert np.array_equal(ours["sent"], lava["s_out"]) and np.any(lava["s_out"] < 0)


def test_the_two_lava_models_differ_on_the_fixture():
    fixed = dense_lava(*dense_fixture(), PySigmaDeltaModelFixed)
    corrected = dense_lava(*dense_fixture(), PySigmaDeltaModelFixedCorrected)
    assert not np.array_equal(fixed["s_out"], corrected["s_out"])


def test_a_conv_sum_beyond_24_bits_wraps_in_lava_and_is_flagged_by_the_soma():
    # Every weight 30 and every source sending 2**20 at update 0 only, over three updates. An
    # inner output neuron sums 27 * 30 * 2**20 at update 1, beyond 24 bits and inside 32.
    _, _, bias = conv_fixture()
    weight = np.full((4, 3, 3, 3), 30)
    values = np.zeros((*INPUT_SHAPE, 2), dtype=np.int64)
    values[..., 0] = 1 << 20
    ours, lava = conv_both(values, weight, bias, PySigmaDeltaModelFixed, False)
    assert len(ours["input"]) == 3
    assert ours["max_abs_input"][-1].max() > (1 << 23) - 1
    assert ours["max_abs_input"][-1].max() == 27 * 30 * (1 << 20)
    assert np.array_equal(lava["sigma"][1], signed_clamp(ours["input"][1], bits=24))
    assert not np.array_equal(lava["sigma"], ours["sigma"])
