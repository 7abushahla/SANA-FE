"""Graded messages against Lava's fixed-point models.

A graded source sends a signed value that multiplies the weight at each synapse, as Lava's
``Dense`` and ``Conv`` do with the value a ``RingBuffer`` sends when ``num_message_bits`` is
above zero. A destination with ``hold_input`` adds its input to a register that drives the
membrane at every update of its window, as Lava's LIF keeps its current u when ``du`` is -1,
which makes its decay factor exactly one.

Alignment in time. ``Dense`` and ``Conv`` send at each update the activation they accumulated
on the previous one, so a ``RingBuffer`` value sent at Lava update t reaches the destination at
update t + 1. A graded source sends ``graded_values[t]`` at update t, and the buffer before the
soma delivers it at update t + 1, so row t of the two traces is compared directly. Each run
lasts one update more than the values, so that the last value sent is received. A
``RingBuffer`` that feeds ``LIF.a_in`` reaches it at the same update, one update before a graded
source reaches its destination, so row t of Lava is compared with row t + 1 of SANA-FE.

Alignment in scale. Lava's LIF shifts its input left by 6 bits before adding it to u, so its u
and v are 64 times the held current and the membrane.
"""
import numpy as np
import pytest

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg, Loihi2SimCfg
from lava.proc.conv.process import Conv
from lava.proc.conv.utils import conv_to_sparse
from lava.proc.dense.process import Dense
from lava.proc.io.source import RingBuffer
from lava.proc.lif.process import LIF
from lava.proc.monitor.process import Monitor
from lava.proc.qcfs import QCFSIFGated
from lava.utils.weightutils import clip_weights, determine_sign_mode, truncate_weights

from test_graded_messages import DST, make_chip, run, sources

pytestmark = pytest.mark.thesis


def lava_run(port, process, values, steps, run_cfg, names):
    """Send ``values`` from a ``RingBuffer`` into ``port`` and run ``steps`` updates.

    Returns the Vars of ``process`` named in ``names``, each as ``[steps, neurons]`` in int64,
    with the neurons in Lava's flattened order.
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
    return {name: values.reshape(steps, -1) for name, values in data.items()}


def sanafe_run(values, edges, count, steps, destination, *, neuron_trace=False):
    """Graded sources holding ``values`` into ``count`` destinations with the attributes
    ``destination``, connected by ``edges`` of ``(source, destination, weight)``.

    Returns the potentials of the destinations as ``[steps, count]`` in int64, and the run.
    """
    target = ("layer_0", "loihi_if_gated", [dict(destination) for _ in range(count)],
              [(0, 1)] * count)
    chip, _ = make_chip([sources(values), target],
                        [("input", int(s), "layer_0", int(d), w) for s, d, w in edges])
    out = run(chip, steps, neuron_trace=neuron_trace)
    return np.asarray(out["potential_trace"], dtype=np.int64), out


def dense_lava_v(weights, values, steps, message_bits):
    dense = Dense(weights=weights, num_message_bits=message_bits)
    layer = QCFSIFGated(shape=(weights.shape[0],), threshold=8388606, initial_voltage=0)
    dense.a_out.connect(layer.a_in)
    return lava_run(dense.s_in, layer, values, steps, Loihi2SimCfg(select_tag="fixed_pt"),
                    ["v"])["v"]


def test_dense_graded_matches_lava_fixed_point():
    rng = np.random.default_rng(11)
    sent = 6
    values = rng.integers(-31, 32, size=(6, sent))
    values[rng.random(values.shape) < 1 / 3] = 0
    weights = rng.integers(-128, 128, size=(5, 6))
    assert np.any(weights % 2) and np.any(values < 0) and np.any(values == 0)
    steps = sent + 1
    lava_v = dense_lava_v(weights, values, steps, 16)
    binary_lava_v = dense_lava_v(weights, values, steps, 0)

    # The weights Lava's Dense applies, as sanafe.lava_integer._dense_weights computes them.
    mode = determine_sign_mode(weights)
    effective = truncate_weights(clip_weights(weights, mode, num_bits=8), mode, 8)
    edges = [(s, d, effective[d, s]) for d in range(5) for s in range(6)]
    ours_v, _ = sanafe_run(values.tolist(), edges, 5, steps, dict(DST, strict=True))
    assert np.array_equal(ours_v, lava_v)
    # The same run with num_message_bits=0 differs.
    assert not np.array_equal(lava_v, binary_lava_v)


def test_conv_graded_matches_lava_fixed_point():
    rng = np.random.default_rng(12)
    sent = 4
    input_shape = (5, 5, 3)
    weight = rng.integers(-128, 128, size=(4, 3, 3, 3))
    values = rng.integers(-31, 32, size=(*input_shape, sent))
    steps = sent + 1
    conv = Conv(weight=weight, input_shape=input_shape, padding=(1, 1), num_message_bits=16)
    layer = QCFSIFGated(shape=conv.output_shape, threshold=8388606, initial_voltage=0)
    conv.a_out.connect(layer.a_in)
    lava_v = lava_run(conv.s_in, layer, values, steps, Loihi2SimCfg(select_tag="fixed_pt"),
                      ["v"])["v"]
    assert np.abs(lava_v).max() < (1 << 23)                # inside Lava's 24-bit wrap
    # The same convolution with num_message_bits=0 sends the binary activation instead.
    binary_conv = Conv(weight=weight, input_shape=input_shape, padding=(1, 1), num_message_bits=0)
    binary_layer = QCFSIFGated(shape=binary_conv.output_shape, threshold=8388606,
                               initial_voltage=0)
    binary_conv.a_out.connect(binary_layer.a_in)
    binary_lava_v = lava_run(binary_conv.s_in, binary_layer, values, steps,
                             Loihi2SimCfg(select_tag="fixed_pt"), ["v"])["v"]

    # Sources and destinations both in Lava's flattened order, as in
    # sanafe.lava_conv_integer.qcfs_fixed_conv_to_sanafe.
    dst, src, kernel = conv_to_sparse(input_shape, tuple(conv.output_shape), weight, (1, 1),
                                      (1, 1), (1, 1), 1)
    count = int(np.prod(conv.output_shape))
    ours_v, _ = sanafe_run(values.reshape(-1, sent).tolist(), list(zip(src, dst, kernel)),
                           count, steps, dict(DST, strict=True))
    assert np.array_equal(ours_v, lava_v)
    # The binary run differs.
    assert not np.array_equal(lava_v, binary_lava_v)


def test_held_current_matches_lava_lif_with_du_minus_one():
    steps = 20
    first = [37, -12, 0, 1000]
    values = np.zeros((4, steps), dtype=np.int32)
    values[:, 0] = first                                   # the zero columns make the input arrive once
    lif = LIF(shape=(4,), du=-1, dv=0, vth=131071, bias_mant=0)
    lava = lava_run(lif.a_in, lif, values, steps, Loihi1SimCfg(select_tag="fixed_pt"),
                    ["u", "v"])
    lava_u, lava_v = lava["u"], lava["v"]

    held_destination = dict(DST, window_start=1, window_stop=21, hold_input=1, strict=1)
    ours_v, out = sanafe_run([[a] for a in first], [(n, n, 1) for n in range(4)], 4, steps + 1,
                             held_destination, neuron_trace=True)
    held = np.asarray(out["neuron_trace"]["held_current"], dtype=np.int64)
    assert np.all(lava_u % 64 == 0) and np.all(lava_v % 64 == 0)
    assert np.array_equal(held[1:], lava_u // 64) and np.array_equal(ours_v[1:], lava_v // 64)
