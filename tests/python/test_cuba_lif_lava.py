"""cuba_lif_int24 against Lava's bit-accurate LIF and LIFReset, on every spike and voltage."""
import unittest
import numpy as np

try:
    from lava.proc.lif.process import LIF, LIFReset
    from lava.proc.dense.process import Dense
    from lava.proc.io.source import RingBuffer
    from lava.proc.monitor.process import Monitor
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.run_configs import Loihi1SimCfg
    HAVE_LAVA = True
except ImportError:
    HAVE_LAVA = False

import sanafe
from sanafe.lava_lif import (ACTIVATION_BOUND, lava_lif_trace, lif_chain_to_sanafe,
                             sanafe_lif_trace)
from sanafe.loihi2 import load_loihi2_candidate


@unittest.skipUnless(HAVE_LAVA, 'requires Lava')
class TestCubaLifAgainstLava(unittest.TestCase):
    def both(self, a_in, **p):
        return lava_lif_trace(a_in, **p), sanafe_lif_trace(a_in, **p)

    def test_decay_grid(self):
        a = np.random.default_rng(0).integers(-2000, 2001, size=(200, 16))
        # The whole accepted input range, with its two ends and the 24-bit boundary of the
        # current after the 6-bit shift (|a| = 2**17), so that the current wraps.
        wide = np.random.default_rng(5).integers(-ACTIVATION_BOUND, ACTIVATION_BOUND + 1,
                                                 size=(200, 16))
        wide[:3, :6] = [40000, -40000, 1 << 17, -(1 << 17), ACTIVATION_BOUND, -ACTIVATION_BOUND]
        for name, activations in (('narrow', a), ('wide', wide)):
            for du in (0, 1, 2047, 4095):
                for dv in (0, 1, 2048, 4095):
                    for vth in (1, 1000):
                        lava, ours = self.both(activations, du=du, dv=dv, vth=vth)
                        self.assertTrue(np.array_equal(lava['s'], ours['s']), (name, du, dv, vth))
                        self.assertTrue(np.array_equal(lava['v'], ours['v']), (name, du, dv, vth))

    def test_current_wraps_as_in_lava(self):
        a = np.full((300, 4), 32767)
        lava, ours = self.both(a, du=0, dv=4095, vth=131071)
        self.assertTrue(np.array_equal(lava['s'], ours['s']))
        self.assertTrue(np.array_equal(lava['v'], ours['v']))

    def test_bias_grid(self):
        a = np.zeros((64, 4), dtype=int)
        for mant in (-4096, 37, 4095):
            for exp in (0, 3, 7):
                lava, ours = self.both(a, du=4095, dv=64, vth=500, bias_mant=mant, bias_exp=exp)
                self.assertTrue(np.array_equal(lava['s'], ours['s']), (mant, exp))
                self.assertTrue(np.array_equal(lava['v'], ours['v']), (mant, exp))

    def test_periodic_reset_matches_lifreset(self):
        a = np.random.default_rng(1).integers(-50, 200, size=(100, 8))
        for offset in (0, 3, 16):
            lava, ours = self.both(a, du=4095, dv=0, vth=2000, reset_interval=16, reset_offset=offset)
            self.assertTrue(np.array_equal(lava['s'], ours['s']), offset)
            self.assertTrue(np.array_equal(lava['v'], ours['v']), offset)

    def test_initial_voltage_matches_lava(self):
        a = np.random.default_rng(4).integers(-10, 30, size=(50, 8))
        lava, ours = self.both(a, du=4095, dv=0, vth=39, initial_v=32 * 40)
        self.assertTrue(np.array_equal(lava['s'], ours['s']))
        self.assertTrue(np.array_equal(lava['v'], ours['v']))

    def test_three_layer_chain_with_odd_weights(self):
        # LIF layers of 8, 6 and 4 neurons, Dense weights uniform in [-127, 127] with odd values,
        # num_weight_bits 8, 100 steps. Every layer's spikes equal Lava's, and the manifest's
        # effective weights equal what Lava's Dense applies. Two further cells run a positive
        # weight exponent, which Lava applies as a left shift of the summed activation.
        for bits, exponent in ((8, 0), (8, 1), (6, 2)):
            with self.subTest(num_weight_bits=bits, weight_exp=exponent):
                self.chain(bits, exponent)

    def chain(self, bits, exponent):
        rng = np.random.default_rng(3)
        steps = 100
        layers = [LIF(shape=(8,), du=1024, dv=256, vth=40,
                      bias_mant=rng.integers(-20, 21, size=8), bias_exp=2,
                      v=rng.integers(-2000, 2001, size=8)),
                  LIFReset(shape=(6,), du=2048, dv=128, vth=30, reset_interval=16, reset_offset=5),
                  LIF(shape=(4,), du=4095, dv=0, vth=20, bias_mant=rng.integers(-3, 4, size=4))]
        raw = [rng.integers(-127, 128, size=(6, 8)), rng.integers(-127, 128, size=(4, 6))]
        self.assertTrue(all(np.any(w % 2) for w in raw), 'an odd weight exercises truncation')
        dense = [Dense(weights=w, num_weight_bits=bits, weight_exp=exponent) for w in raw]
        for pre, conn, post in zip(layers, dense, layers[1:]):
            pre.s_out.connect(conn.s_in)
            conn.a_out.connect(post.a_in)
        a_in = rng.integers(-40, 120, size=(steps, 8))
        arch = load_loihi2_candidate()
        net, manifest = lif_chain_to_sanafe(layers, dense, a_in, arch, [(0, 0), (1, 0), (2, 0)])
        semantics = manifest['architecture_profile']['semantics']
        self.assertEqual((semantics['input_min'], semantics['input_max']),
                         (-ACTIVATION_BOUND, ACTIVATION_BOUND))
        self.assertNotIn('input_bits', semantics)       # the IF units' 16-bit input is not stated

        source = RingBuffer(data=a_in.T.astype(np.int32))
        source.s_out.connect(layers[0].a_in)
        monitors = []
        for layer in layers:
            v_mon, s_mon = Monitor(), Monitor()
            v_mon.probe(layer.v, steps)
            s_mon.probe(layer.s_out, steps)
            monitors.append((v_mon, s_mon))
        try:
            layers[0].run(condition=RunSteps(num_steps=steps),
                          run_cfg=Loihi1SimCfg(select_tag='fixed_pt'))
            lava_v = [m.get_data()[l.name]['v'] for l, (m, _) in zip(layers, monitors)]
            lava_s = [m.get_data()[l.name]['s_out'] for l, (_, m) in zip(layers, monitors)]
            applied = [d.weights.get() * (1 << d.proc_params.get('weight_exp', 0)) for d in dense]
        finally:
            layers[0].stop()

        self.assertTrue(np.array_equal(manifest['effective_weights'][0], applied[0]))
        self.assertTrue(np.array_equal(manifest['effective_weights'][1], applied[1]))
        self.assertTrue(any(np.any(w != a) for w, a in zip(raw, applied)), 'Lava truncated a weight')

        chip = sanafe.SpikingChip(arch)
        chip.load(net)
        result = chip.sim(steps, spike_trace=True, potential_trace=True)
        sizes = manifest['layer_sizes']
        ours_s = [np.zeros((steps, n), dtype=np.int64) for n in sizes]
        for t, active in enumerate(result['spike_trace']):
            for neuron in active:
                ours_s[int(neuron.group_name.split('_')[-1])][t, neuron.neuron_offset] = 1
        ours_v = np.split(np.asarray(result['potential_trace']), np.cumsum(sizes)[:-1], axis=1)
        for index in range(3):
            self.assertTrue(np.any(lava_s[index]), f'layer {index} never spiked')
            self.assertTrue(np.array_equal(ours_s[index], lava_s[index]), index)
            self.assertTrue(np.array_equal(ours_v[index], lava_v[index]), index)


if __name__ == '__main__':
    unittest.main()
