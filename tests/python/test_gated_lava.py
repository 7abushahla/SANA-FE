"""integrate_fire_int24_gated against the Lava process QCFSIFGated, on every spike and voltage.

The Lava process is a second implementation of the same gated IF program. The
alignment of the two runs is that of ``sanafe.lava_integer.qcfs_gated_chain_to_sanafe``.
"""
import unittest
import numpy as np

try:
    from lava.proc.qcfs import QCFSIFGated
    from lava.proc.dense.process import Dense
    from lava.proc.io.source import RingBuffer
    from lava.proc.monitor.process import Monitor
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.run_configs import Loihi2SimCfg
    HAVE_LAVA = True
except ImportError:
    HAVE_LAVA = False

import sanafe
from sanafe.lava_integer import qcfs_gated_chain_to_sanafe
from sanafe.loihi2 import load_loihi2_candidate


def run_both(layers, connections, currents, steps):
    """Spikes and voltages per layer, ``[steps, n]`` each, from SANA-FE and from Lava.

    The chain is exported before Lava attaches its source and monitors.
    """
    arch = load_loihi2_candidate()
    placements = [(index, 0) for index in range(len(layers))]
    net, manifest = qcfs_gated_chain_to_sanafe(layers, connections, currents, arch, placements)
    chip = sanafe.SpikingChip(arch)
    chip.load(net)
    result = chip.sim(steps, spike_trace=True, potential_trace=True)
    sizes = manifest['layer_sizes']
    ours_s = [np.zeros((steps, n), dtype=np.int64) for n in sizes]
    for t, active in enumerate(result['spike_trace']):
        for neuron in active:
            ours_s[int(neuron.group_name.split('_')[-1])][t, neuron.neuron_offset] = 1
    ours_v = np.split(np.asarray(result['potential_trace'], dtype=np.int64),
                      np.cumsum(sizes)[:-1], axis=1)

    source = RingBuffer(data=np.asarray(currents, dtype=np.int32))
    source.s_out.connect(layers[0].a_in)
    monitors = []
    for layer in layers:
        v_mon, s_mon = Monitor(), Monitor()
        v_mon.probe(layer.v, steps)
        s_mon.probe(layer.s_out, steps)
        monitors.append((v_mon, s_mon))
    try:
        layers[0].run(condition=RunSteps(num_steps=steps),
                      run_cfg=Loihi2SimCfg(select_tag='fixed_pt'))
        lava_v = [np.asarray(m.get_data()[l.name]['v']).astype(np.int64)
                  for l, (m, _) in zip(layers, monitors)]
        lava_s = [np.asarray(m.get_data()[l.name]['s_out']).astype(np.int64)
                  for l, (_, m) in zip(layers, monitors)]
    finally:
        layers[0].stop()
    return (ours_s, ours_v), (lava_s, lava_v), manifest


@unittest.skipUnless(HAVE_LAVA, 'requires Lava with QCFSIFGated')
class TestGatedAgainstLava(unittest.TestCase):
    def test_single_neurons_over_windows_strides_and_biases(self):
        # 16 neurons, 60 updates, inputs uniform in [-300, 300] (seed 3), thresholds even in [2, 400],
        # initial voltage threshold / 2, for (window_start, window_stop, stride) in
        # (0, 60, 1), (10, 40, 1), (10, 40, 3), (59, 60, 1) and bias_at_start in (0, 123), bias_per_check in (0, -2).
        # Spikes and voltages of SANA-FE's integrate_fire_int24_gated equal Lava's at every update.
        rng = np.random.default_rng(3)
        n, steps = 16, 60
        currents = rng.integers(-300, 301, size=(n, steps))
        theta = 2 * rng.integers(1, 201, size=n)
        compared = 0
        for start, stop, stride in ((0, 60, 1), (10, 40, 1), (10, 40, 3), (59, 60, 1)):
            for at_start in (0, 123):
                for per_check in (0, -2):
                    cell = (start, stop, stride, at_start, per_check)
                    with self.subTest(cell=cell):
                        layer = QCFSIFGated(shape=(n,), threshold=theta, initial_voltage=theta // 2,
                                            bias_at_start=at_start, bias_per_check=per_check,
                                            window_start=start, window_stop=stop, stride=stride)
                        (ours_s, ours_v), (lava_s, lava_v), _ = run_both([layer], [], currents, steps)
                        self.assertEqual(lava_s[0].shape, (steps, n))
                        self.assertEqual(lava_v[0].shape, (steps, n))
                        self.assertTrue(np.array_equal(ours_s[0], lava_s[0]), cell)
                        self.assertTrue(np.array_equal(ours_v[0], lava_v[0]), cell)
                        self.assertTrue(np.any(lava_s[0]), f'{cell} never spiked')
                        u = np.arange(steps)
                        checks = (u >= start) & (u < stop) & ((u - start) % stride == 0)
                        if not checks.all():
                            # Some neuron sits at or above threshold after an unchecked update,
                            # so the gate, not the input, decides when it fires.
                            self.assertTrue(np.any(lava_v[0][~checks] >= theta), cell)
                        if stride > 1:
                            # The same holds on the updates inside the window that the stride skips.
                            skipped = (u >= start) & (u < stop) & ((u - start) % stride != 0)
                            self.assertTrue(np.any(lava_v[0][skipped] >= theta), cell)
                        compared += lava_v[0].size
        self.assertEqual(compared, 16 * 16 * 60)

    def test_three_layer_completion_schedule(self):
        # layers of 8, 6, 4 gated neurons, integer weights in [-20, 20], horizons (4, 2, 3),
        # windows opened one after another as under completion. Every layer's spikes equal Lava's.
        # Lava's Dense delivers a spike one update after it is sent, so layer k + 1 opens its
        # window on the update after the last check of layer k has reached it.
        rng = np.random.default_rng(5)
        sizes, horizons = (8, 6, 4), (4, 2, 3)
        starts = np.concatenate([[0], np.cumsum(horizons)[:-1]])
        steps = int(starts[-1] + horizons[-1]) + 3          # three unchecked updates after the last window
        layers = []
        for size, start, horizon in zip(sizes, starts, horizons):
            theta = 2 * rng.integers(4, 16, size=size)
            layers.append(QCFSIFGated(
                shape=(size,), threshold=theta, initial_voltage=theta // 2,
                bias_at_start=rng.integers(-10, 11, size=size),
                bias_per_check=rng.integers(-3, 4, size=size),
                window_start=int(start), window_stop=int(start + horizon), stride=1))
        dense = [Dense(weights=rng.integers(-20, 21, size=(post, pre)))
                 for pre, post in zip(sizes, sizes[1:])]
        for pre, conn, post in zip(layers, dense, layers[1:]):
            pre.s_out.connect(conn.s_in)
            conn.a_out.connect(post.a_in)
        currents = rng.integers(-5, 30, size=(sizes[0], steps))
        (ours_s, ours_v), (lava_s, lava_v), manifest = run_both(layers, dense, currents, steps)
        windows = [sorted({(a, b, c) for a, b, c in zip(layer['window_start'], layer['window_stop'],
                                                         layer['stride'])})
                   for layer in manifest['layers']]
        self.assertEqual(windows, [[(0, 4, 1)], [(4, 6, 1)], [(6, 9, 1)]])
        for index, size in enumerate(sizes):
            self.assertEqual(lava_s[index].shape, (steps, size))
            self.assertTrue(np.any(lava_s[index]), f'layer {index} never spiked')
            self.assertTrue(np.array_equal(ours_s[index], lava_s[index]), index)
            self.assertTrue(np.array_equal(ours_v[index], lava_v[index]), index)

    def test_voltage_saturates_at_both_rails_and_fires_from_the_upper_rail(self):
        # Inputs at the signed 24-bit limits drive the voltage onto the upper rail 2**23 - 1 and the
        # lower rail -2**23 before a late window [6, 10). Neuron 0 fires from the upper rail, neuron 1
        # stays on the lower rail through a negative bias_per_check, neuron 2 crosses from the lower
        # rail to the upper one and fires, neuron 3 is clamped by bias_at_start inside the window.
        v_max, v_min = (1 << 23) - 1, -(1 << 23)
        steps = 10
        currents = np.zeros((4, steps), dtype=np.int64)
        currents[0, :4] = v_max
        currents[1, :4] = v_min
        currents[2, :3], currents[2, 3:6] = v_min, v_max
        currents[3, 5:7] = v_max
        layer = QCFSIFGated(shape=(4,), threshold=np.array([1000, 1000, v_max - 1, 2]),
                            initial_voltage=np.array([v_max - 1, v_min + 1, 0, 0]),
                            bias_at_start=np.array([0, 0, 0, v_max]),
                            bias_per_check=np.array([0, -(1 << 15), 0, 0]),
                            window_start=6, window_stop=10)
        (ours_s, ours_v), (lava_s, lava_v), _ = run_both([layer], [], currents, steps)
        self.assertTrue(np.array_equal(ours_s[0], lava_s[0]))
        self.assertTrue(np.array_equal(ours_v[0], lava_v[0]))
        for name, s, v in (('SANA-FE', ours_s[0], ours_v[0]), ('Lava', lava_s[0], lava_v[0])):
            self.assertTrue(np.any(v == v_max), f'{name} never reached the upper rail')
            self.assertTrue(np.any(v == v_min), f'{name} never reached the lower rail')
            railed = (v == v_max) | (v == v_min)
            fired_after_clamp = [n for n in range(4) if railed[:, n].any() and
                                 s[int(np.argmax(railed[:, n])) + 1:, n].any()]
            self.assertTrue(fired_after_clamp, f'{name}: no neuron fired after being clamped')

    def test_a_neuron_without_input_keeps_the_window_phase(self):
        # The Lava process counts updates once for the whole layer, the soma once per neuron. They agree
        # only if SANA-FE updates every gated neuron on every timestep, input or not. Layer 0 never fires,
        # so layer 1 receives no input at all. Each of its neurons fires once, through bias_at_start alone,
        # exactly on its own window_start.
        steps = 12
        first = QCFSIFGated(shape=(2,), threshold=(1 << 23) - 2, initial_voltage=0)
        starts = np.array([0, 5, 7, 11])
        second = QCFSIFGated(shape=(4,), threshold=4, initial_voltage=2, bias_at_start=2,
                             window_start=starts, window_stop=starts + 1)
        dense = Dense(weights=np.ones((4, 2), dtype=int))
        first.s_out.connect(dense.s_in)
        dense.a_out.connect(second.a_in)
        (ours_s, ours_v), (lava_s, lava_v), _ = run_both([first, second], [dense],
                                                         np.zeros((2, steps), dtype=int), steps)
        for index in range(2):
            self.assertTrue(np.array_equal(ours_s[index], lava_s[index]), index)
            self.assertTrue(np.array_equal(ours_v[index], lava_v[index]), index)
        self.assertFalse(np.any(ours_s[0]))
        expected = np.zeros((steps, 4), dtype=np.int64)
        expected[starts, np.arange(4)] = 1
        self.assertTrue(np.array_equal(ours_s[1], expected))

    def test_export_refuses_what_the_gated_soma_cannot_run(self):
        from lava.proc.qcfs import QCFSIFFixed
        arch = load_loihi2_candidate()
        layer = QCFSIFGated(shape=(1,), threshold=4, initial_voltage=2)
        with self.assertRaisesRegex(ValueError, 'currents outside'):
            qcfs_gated_chain_to_sanafe([layer], [], [[1 << 23]], arch, [(0, 0)])
        _, manifest = qcfs_gated_chain_to_sanafe([layer], [], [[(1 << 23) - 1]], arch, [(0, 0)])
        self.assertEqual(manifest['architecture_profile']['semantics']['input_bits'], 24)
        with self.assertRaisesRegex(ValueError, 'QCFSIFGated'):
            qcfs_gated_chain_to_sanafe([QCFSIFFixed(shape=(1,), threshold=4)], [], [[0]],
                                       arch, [(0, 0)])
        # 32,897 inputs of effective weight 255 can sum to 8,388,735, beyond signed 24 bits.
        wide, post = QCFSIFGated(shape=(32897,), threshold=4, initial_voltage=2), \
            QCFSIFGated(shape=(1,), threshold=4, initial_voltage=2)
        dense = Dense(weights=np.full((1, 32897), 255))
        wide.s_out.connect(dense.s_in)
        dense.a_out.connect(post.a_in)
        with self.assertRaisesRegex(ValueError, 'fan-in'):
            qcfs_gated_chain_to_sanafe([wide, post], [dense], np.zeros((32897, 1), dtype=int),
                                       arch, [(0, 0), (1, 0)])


if __name__ == '__main__':
    unittest.main()
