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
from test_integer_models import TestIfAgainstLeakFreeLif, if_trace


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

    def test_single_layer_manifest_lists_no_dense_model(self):
        layer = LIF(shape=(3,), du=4095, dv=0, vth=10)
        _, manifest = lif_chain_to_sanafe([layer], [], np.ones((4, 3), dtype=int),
                                          load_loihi2_candidate(), [(0, 0)])
        self.assertEqual(manifest['lava_models'], ['lava.proc.lif.models.PyLifModelBitAcc'])

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
            truncated = [d.weights.get() for d in dense]      # Lava's weights before the exponent
            applied = [w * (1 << d.proc_params.get('weight_exp', 0)) for w, d in zip(truncated, dense)]
        finally:
            layers[0].stop()

        self.assertTrue(np.array_equal(manifest['effective_weights'][0], applied[0]))
        self.assertTrue(np.array_equal(manifest['effective_weights'][1], applied[1]))
        if bits < 8:
            self.assertTrue(any(np.any(w != t) for w, t in zip(raw, truncated)),
                            'Lava truncated a weight')

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


def large_theta_cases():
    """Cases at the top of the mapping's domain, theta 2**17 and the extreme bias mantissas.

    The drifts are gentle so that the membrane stays inside +-131,071 before the first spike.
    """
    return ((131072, 4095, lambda rng, theta: rng.integers(-2000, 3001, size=200)),
            (131072, -4096, lambda rng, theta: rng.integers(3000, 9000, size=200)),
            (131072, 0, lambda rng, theta: rng.integers(-3000, 6001, size=200)))


@unittest.skipUnless(HAVE_LAVA, 'requires Lava')
class TestLeakFreeMappingOnLava(unittest.TestCase):
    """Lava's own LIF under the leak-free mapping of TestIfAgainstLeakFreeLif, against the soft-reset IF.

    The mapping is du = 4095, dv = 0, vth = theta - 1, initial_v = 32 theta and bias exponent 6.
    Its domain is theta at most 131,072 (17-bit vth), bias in [-4096, 4095] (13-bit mantissa)
    and membrane magnitude at most 131,071 (the LIF clips 64 V at +-(2^23 - 1)). Up to the
    first spike the LIF voltage is 64 times the IF voltage. At the first spike the two differ by
    design (the IF subtracts theta, the LIF resets to zero), so voltages are compared on every
    update before it and spikes on every update up to and including it.
    """
    NEURONS = 50

    def test_voltages_and_first_spike_agree_with_the_soft_reset_if(self):
        rng = np.random.default_rng(2)
        negative_before_spike = late_first_spike = exact_landing = runs = fired = 0
        lowest = 0
        for theta, bias, draw in TestIfAgainstLeakFreeLif.cases() + large_theta_cases():
            a = np.stack([draw(rng, theta) for _ in range(self.NEURONS)], axis=1)
            lava = lava_lif_trace(a, du=4095, dv=0, vth=theta - 1, bias_mant=bias, bias_exp=6,
                                  initial_v=32 * theta)
            for column in range(self.NEURONS):
                column_a = a[:, column].tolist()
                s_if, v_if = if_trace({'threshold': theta, 'initial_voltage': theta // 2,
                                       'bias': bias, 'currents': column_a}, 200)
                k = s_if.index(1) if 1 in s_if else 200
                with self.subTest(theta=theta, bias=bias, column=column):
                    self.assertEqual(lava['s'][:k + 1, column].tolist(), s_if[:k + 1])
                    self.assertEqual(lava['v'][:k, column].tolist(), [64 * v for v in v_if[:k]])
                runs += 1
                lowest = min([lowest, *v_if[:k]])
                if k < 200:
                    fired += 1
                    negative_before_spike += any(v < 0 for v in v_if[:k])
                    late_first_spike += k > 20
                    before = v_if[k - 1] if k > 0 else theta // 2
                    exact_landing += before + column_a[k] + bias == theta
        self.assertGreaterEqual(lowest, -131071, 'every membrane stays inside the domain')
        self.assertGreater(fired, 0, 'some run fires')
        self.assertGreater(negative_before_spike, 0, 'a run that goes negative and then fires')
        self.assertGreater(late_first_spike, 0, 'a first spike after step 20')
        self.assertGreater(exact_landing, 0, 'a membrane exactly at the threshold at the firing step')
        self.counters = (runs, fired, negative_before_spike, late_first_spike, exact_landing)


if __name__ == '__main__':
    unittest.main()
