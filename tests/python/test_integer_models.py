"""Integer numerical models and dynamic accumulator regression tests."""
import tempfile
import subprocess
import sys
import unittest
from importlib.resources import files
from pathlib import Path

import numpy as np

import sanafe


def architecture(soma='integrate_fire_int24', dendrite='accumulator_int',
                 synapse='current_based_int', capacity=2048):
    text = (files('sanafe.examples') / 'loihi.yaml').read_text()
    text = text.replace('loihi_tile[0..31]', 'loihi_tile[0..0]')
    text = text.replace('loihi_core[0..3]', 'loihi_core[0..0]')
    text = text.replace('loihi_inputs[0..1023]', 'loihi_inputs[0..0]')
    text = text.replace('model: leaky_integrate_fire', 'model: ' + soma)
    text = text.replace('model: accumulator\n', 'model: ' + dendrite + '\n')
    text = text.replace('model: current_based', 'model: ' + synapse)
    text = text.replace('max_neurons_supported: 1024',
                        'max_neurons_supported: ' + str(capacity))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'arch.yaml'
        path.write_text(text)
        return sanafe.load_arch(str(path))


def chip_for(attributes, **kwargs):
    arch = architecture(**kwargs)
    net = sanafe.Network()
    group = net.create_neuron_group('if', len(attributes), log_spikes=True,
                                    log_potential=True)
    for neuron, attrs in zip(group, attributes):
        neuron.set_attributes(model_attributes=attrs)
        neuron.map_to_core(arch.tiles[0].cores[0])
    chip = sanafe.SpikingChip(arch)
    chip.load(net)
    return chip


def connected_chip(weights, **kwargs):
    arch = architecture(**kwargs)
    net = sanafe.Network()
    src = net.create_neuron_group('src', 1,
        model_attributes={'threshold': 2, 'currents': [2, 0, 2, 0]})
    dst = net.create_neuron_group('dst', 1,
        model_attributes={'threshold': 100, 'initial_voltage': 0},
        log_potential=True, log_spikes=True)
    src[0].map_to_core(arch.tiles[0].cores[0])
    dst[0].map_to_core(arch.tiles[0].cores[0])
    for weight in weights:
        src[0].connect_to_neuron(dst[0], {'weight': weight})
    chip = sanafe.SpikingChip(arch)
    chip.load(net)
    return chip


class TestIntegerIF(unittest.TestCase):
    def test_candidate_valid_window_gates_bias_and_reset(self):
        chip = chip_for([{'threshold': 4, 'initial_voltage': 2, 'bias': 2,
                         'valid_start': 1, 'valid_stop': 3,
                         'currents': [0, 0, 0, 0]}])
        full = chip.sim(4, potential_trace=True, spike_trace=True)
        self.assertEqual(full['potential_trace'], [[2], [0], [2], [2]])
        self.assertEqual([len(s) for s in full['spike_trace']], [0, 1, 0, 0])
        chip.reset()
        first = chip.sim(1, potential_trace=True, spike_trace=True)
        rest = chip.sim(3, potential_trace=True, spike_trace=True)
        self.assertEqual(first['potential_trace'] + rest['potential_trace'],
                         full['potential_trace'])
        self.assertEqual([len(s) for s in first['spike_trace'] + rest['spike_trace']],
                         [0, 1, 0, 0])

    def test_candidate_valid_window_rejects_invalid_bounds(self):
        for name, value in [('valid_start', -1), ('valid_start', .5),
                            ('valid_stop', 0), ('valid_stop', 1 << 31)]:
            with self.subTest(name=name, value=value), self.assertRaises((ValueError, RuntimeError)):
                chip_for([{'threshold': 4, name: value}])
        chip = chip_for([{'threshold': 4, 'valid_start': 2, 'valid_stop': 2}])
        with self.assertRaisesRegex((ValueError, RuntimeError), 'window'):
            chip.sim(1)

    def test_equality_negative_voltage_and_single_subtraction(self):
        chip = chip_for([
            {'threshold': 2, 'initial_voltage': 1, 'currents': [1, -4, 6, 0]},
            {'threshold': 4, 'initial_voltage': 2, 'currents': [10, 0, 0, 0]},
        ])
        result = chip.sim(4, potential_trace=True, spike_trace=True)
        self.assertEqual(result['potential_trace'], [[0, 8], [-4, 4], [0, 0], [0, 0]])
        self.assertEqual([len(s) for s in result['spike_trace']], [2, 1, 2, 0])

    def test_clip_before_threshold_and_widened_bias_addition(self):
        chip = chip_for([
            {'threshold': 8388606, 'initial_voltage': 8388600,
             'bias': 32767, 'currents': [32767]},
            {'threshold': 2, 'initial_voltage': -8388600,
             'bias': -32768, 'currents': [-32768]},
        ])
        result = chip.sim(1, potential_trace=True, spike_trace=True)
        self.assertEqual(result['potential_trace'], [[1, -8388608]])
        self.assertEqual(len(result['spike_trace'][0]), 1)

    def test_reset_restores_state_stream_and_independent_neurons(self):
        chip = chip_for([
            {'threshold': 4, 'initial_voltage': 2, 'bias': 1, 'currents': [1, -6, 3, 0]},
            {'threshold': 6, 'initial_voltage': -2, 'currents': [-1, 6]},
        ])
        full = chip.sim(5, potential_trace=True)['potential_trace']
        self.assertEqual(full, [[0, -3], [-5, 3], [-1, 3], [0, 3], [1, 3]])
        chip.reset()
        chunks = chip.sim(2, potential_trace=True)['potential_trace']
        chunks += chip.sim(3, potential_trace=True)['potential_trace']
        self.assertEqual(full, chunks)

    def test_invalid_attributes(self):
        invalid = {
            'threshold': [0, -2, 3, 8388608, 2.5, float('inf'), float('nan')],
            'initial_voltage': [-8388609, 8388608, .5, float('nan')],
            'bias': [-32769, 32768, .5, float('inf')],
            'currents': [[-32769], [32768], [.5], [float('nan')]],
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    with self.assertRaises((ValueError, RuntimeError)):
                        chip_for([{'threshold': 2, name: value}])

    def test_synapse_rejects_noninteger_and_out_of_range_weights(self):
        for weight in [.5, 32768, -32769, float('inf'), float('nan')]:
            with self.subTest(weight=weight):
                with self.assertRaises((ValueError, RuntimeError)):
                    connected_chip([weight])

    def test_parallel_validation_errors_reach_python_without_abort(self):
        # On an OpenMP build these exercise soma and message worker exceptions.
        script = "\n".join([
            "import sys",
            "sys.path.insert(0, " + repr(str(Path(__file__).resolve().parent)) + ")",
            "from test_integer_models import connected_chip",
            "cases = [([32767, 1], {}, ValueError, 'signed 16'),",
            "         ([2147483647, 1], {'synapse': 'current_based'}, OverflowError, 'signed 32')]",
            "for weights, kwargs, error, text in cases:",
            "    chip = connected_chip(weights, **kwargs)",
            "    try:",
            "        chip.sim(3, processing_threads=2)",
            "    except error as exc:",
            "        assert text in str(exc), str(exc)",
            "    else:",
            "        raise AssertionError('invalid current was accepted')",
        ])
        result = subprocess.run([sys.executable, '-c', script],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_synaptic_sum_rejected_at_soma_signed16_boundary(self):
        chip = connected_chip([32767, 1])
        with self.assertRaisesRegex((ValueError, RuntimeError), 'signed 16'):
            chip.sim(3)

    def test_soma_rejects_fractional_transport_current(self):
        chip = connected_chip([.5], dendrite='accumulator', synapse='current_based')
        with self.assertRaisesRegex((ValueError, RuntimeError), 'signed 16'):
            chip.sim(3)

    def test_effective_signed_weights_and_accumulator_reset(self):
        chip = connected_chip([12, -5])
        full = chip.sim(5, potential_trace=True)['potential_trace']
        self.assertEqual(full, [[0], [7], [7], [14], [14]])
        chip.reset()
        self.assertEqual(full, chip.sim(5, potential_trace=True)['potential_trace'])

    def test_large_integer_population(self):
        chip = chip_for([{'threshold': 2, 'initial_voltage': 1, 'bias': 1}] * 1025)
        result = chip.sim(2, potential_trace=True, spike_trace=True)
        self.assertEqual(result['potential_trace'], [[0] * 1025, [1] * 1025])
        self.assertEqual([len(s) for s in result['spike_trace']], [1025, 0])


class TestDynamicAccumulators(unittest.TestCase):
    def test_accumulator_population_above_1024(self):
        for soma, dendrite, synapse in [
                ('integrate_fire_float32', 'accumulator', 'current_based'),
                ('integrate_fire_int24', 'accumulator_int', 'current_based_int')]:
            with self.subTest(dendrite=dendrite):
                arch = architecture(soma=soma, dendrite=dendrite, synapse=synapse)
                net = sanafe.Network()
                src = net.create_neuron_group('src', 1,
                    model_attributes={'threshold': 2., 'currents': [2.]})
                dst = net.create_neuron_group('dst', 1025,
                    model_attributes={'threshold': 100.}, log_potential=True)
                src[0].map_to_core(arch.tiles[0].cores[0])
                for neuron in dst:
                    neuron.map_to_core(arch.tiles[0].cores[0])
                    src[0].connect_to_neuron(neuron, {'weight': 7.})
                chip = sanafe.SpikingChip(arch)
                chip.load(net)
                result = chip.sim(4, potential_trace=True)['potential_trace']
                self.assertEqual(len(result[-1]), 1025)
                self.assertEqual(set(result[-1]), {7.})

    def test_float_accumulator_reset_timestamps(self):
        for dendrite in ['accumulator']:
            with self.subTest(dendrite=dendrite):
                chip = connected_chip([7], soma='integrate_fire_float32',
                                      dendrite=dendrite, synapse='current_based')
                full = chip.sim(6, potential_trace=True)['potential_trace']
                chip.reset()
                self.assertEqual(full, chip.sim(6, potential_trace=True)['potential_trace'])


if __name__ == '__main__':
    unittest.main()


GATED = 'integrate_fire_int24_gated'


class TestGatedIF(unittest.TestCase):
    """Integrates every update; bias and threshold check only in a strided window (Loihi 2 route)."""

    def test_input_before_window_is_integrated_and_checked_at_the_first_window_step(self):
        chip = chip_for([{'threshold': 4, 'initial_voltage': 2, 'bias_at_start': 0, 'bias_per_check': 0,
                          'window_start': 2, 'window_stop': 4, 'stride': 1,
                          'currents': [3, 0, 0, 0]}], soma=GATED)
        out = chip.sim(4, potential_trace=True, spike_trace=True)
        self.assertEqual(out['potential_trace'], [[5], [5], [1], [1]])
        self.assertEqual([len(s) for s in out['spike_trace']], [0, 0, 1, 0])

    def test_bias_at_start_is_added_once_and_bias_per_check_every_check(self):
        chip = chip_for([{'threshold': 10, 'initial_voltage': 5, 'bias_at_start': 6, 'bias_per_check': 2,
                          'window_start': 1, 'window_stop': 3, 'stride': 1, 'currents': [0, 0, 0]}],
                        soma=GATED)
        out = chip.sim(3, potential_trace=True, spike_trace=True)
        self.assertEqual(out['potential_trace'], [[5], [3], [5]])
        self.assertEqual([len(s) for s in out['spike_trace']], [0, 1, 0])

    def test_stride_checks_every_k_steps_and_input_on_the_stop_step_is_kept(self):
        chip = chip_for([{'threshold': 4, 'initial_voltage': 0, 'bias_at_start': 0, 'bias_per_check': 0,
                          'window_start': 1, 'window_stop': 4, 'stride': 2,
                          'currents': [3, 0, 3, 0, 3]}], soma=GATED)
        out = chip.sim(5, potential_trace=True, spike_trace=True)
        # Update 2 is in the window but skipped by the stride, so 6 waits for the check at update 3.
        # With stride 1 the neuron would fire at update 2 and end at [3, 3, 2, 2, 5].
        self.assertEqual(out['potential_trace'], [[3], [3], [6], [2], [5]])
        self.assertEqual([len(s) for s in out['spike_trace']], [0, 0, 0, 1, 0])

    def test_voltage_is_clamped_to_signed_24_bits_before_the_check(self):
        top, bottom = (1 << 23) - 1, -(1 << 23)
        chip = chip_for([{'threshold': 4, 'initial_voltage': top - 1,
                          'window_start': 2, 'window_stop': 3, 'currents': [top, 5, 0]},
                         {'threshold': 4, 'initial_voltage': bottom + 1, 'bias_per_check': -1,
                          'window_start': 2, 'window_stop': 3, 'currents': [bottom, -5, 0]}],
                        soma=GATED)
        out = chip.sim(3, potential_trace=True, spike_trace=True)
        # Both voltages are held at their rail, then the check at update 2 fires from the upper rail
        # and subtracts the threshold once, while bias_per_check cannot push below the lower rail.
        self.assertEqual(out['potential_trace'], [[top, bottom], [top, bottom], [top - 4, bottom]])
        self.assertEqual([len(s) for s in out['spike_trace']], [0, 0, 1])

    def test_a_neuron_without_input_is_updated_every_timestep(self):
        chip = chip_for([{'threshold': 4, 'initial_voltage': 2, 'bias_at_start': 2,
                          'window_start': 5, 'window_stop': 6}], soma=GATED)
        out = chip.sim(7, potential_trace=True, spike_trace=True)
        # No currents and no synapses. The update index still advances on every timestep, so the
        # window opens at update 5, where bias_at_start alone reaches the threshold.
        self.assertEqual(out['potential_trace'], [[2]] * 5 + [[0], [0]])
        self.assertEqual([len(s) for s in out['spike_trace']], [0, 0, 0, 0, 0, 1, 0])

    def test_accepts_signed_24_bit_current_and_rejects_beyond(self):
        chip_for([{'threshold': 2, 'currents': [(1 << 23) - 1]}], soma=GATED).sim(1)
        with self.assertRaisesRegex((ValueError, RuntimeError), 'signed 24'):
            chip_for([{'threshold': 2, 'currents': [1 << 23]}], soma=GATED).sim(1)

    def test_synaptic_input_reaches_a_gated_neuron_outside_its_window(self):
        arch = architecture(soma=GATED)
        net = sanafe.Network()
        src = net.create_neuron_group('src', 1, model_attributes={'threshold': 2, 'currents': [2, 0, 0]})
        dst = net.create_neuron_group('dst', 1, model_attributes={
            'threshold': 100, 'initial_voltage': 0, 'window_start': 2, 'window_stop': 3},
            log_potential=True, log_spikes=True)
        src[0].map_to_core(arch.tiles[0].cores[0])
        dst[0].map_to_core(arch.tiles[0].cores[0])
        src[0].connect_to_neuron(dst[0], {'weight': 7})
        chip = sanafe.SpikingChip(arch)
        chip.load(net)
        out = chip.sim(3, potential_trace=True)
        self.assertEqual(out["potential_trace"], [[0], [7], [7]])

    def test_reset_restores_initial_voltage_and_window_phase(self):
        chip = chip_for([{'threshold': 4, 'initial_voltage': 2, 'window_start': 1, 'window_stop': 2,
                          'currents': [3, 0]}], soma=GATED)
        a = chip.sim(2, potential_trace=True)
        chip.reset()
        b = chip.sim(2, potential_trace=True)
        self.assertEqual(a['potential_trace'], b['potential_trace'])
        self.assertEqual(a['potential_trace'], [[5], [1]])


class TestGatedIFSynapticRange(unittest.TestCase):
    """A fan-in above signed 16 bits reaches the gated soma intact; one beyond signed 24 bits is refused."""

    def _fanin(self, n_src, weight):
        arch = architecture(soma=GATED, capacity=n_src + 1)
        net = sanafe.Network()
        src = net.create_neuron_group('src', n_src, model_attributes={'threshold': 2, 'currents': [2]})
        dst = net.create_neuron_group('dst', 1, model_attributes={'threshold': 1 << 22, 'window_start': 5},
                                      log_potential=True)
        for n in src:
            n.map_to_core(arch.tiles[0].cores[0])
            n.connect_to_neuron(dst[0], {'weight': weight})
        dst[0].map_to_core(arch.tiles[0].cores[0])
        chip = sanafe.SpikingChip(arch)
        chip.load(net)
        return chip.sim(2, potential_trace=True)

    def test_a_fan_in_above_16_bits_is_integrated_exactly(self):
        out = self._fanin(300, 127)                                  # 38,100 in one step
        self.assertEqual(out['potential_trace'][1][-1], 300 * 127)

    def test_a_fan_in_beyond_24_bits_is_refused(self):
        with self.assertRaisesRegex((ValueError, RuntimeError), 'signed 24'):
            self._fanin(257, 32767)                                  # 8,421,119 > 2^23 - 1


LIF = 'cuba_lif_int24'


def neuron_trace(attributes: dict, steps: int, chunks: tuple[int, ...] | None = None,
                 **chip_kwargs) -> tuple[list[int], list[int]]:
    """Spike (0 or 1) and voltage after each update of one neuron, run in the given chunks of updates."""
    chip = chip_for([attributes], **chip_kwargs)
    spikes, potentials = [], []
    for n in (chunks if chunks is not None else (steps,)):
        out = chip.sim(n, potential_trace=True, spike_trace=True)
        spikes += [len(s) for s in out['spike_trace']]
        potentials += [p[0] for p in out['potential_trace']]
    return spikes, potentials


def lif_trace(attributes: dict, steps: int, chunks: tuple[int, ...] | None = None) -> tuple[list[int], list[int]]:
    """Spike (0 or 1) and Lava-scaled voltage after each update of one cuba_lif_int24 neuron."""
    return neuron_trace(attributes, steps, chunks, soma=LIF)


class TestCubaLif(unittest.TestCase):
    """Lava's bit-accurate fixed-point LIF (PyLifModelBitAcc, PyLifResetModelBitAcc), step for step."""

    def test_a_leak_free_accumulation_and_strict_threshold(self):
        s, v = lif_trace({'du': 4095, 'dv': 0, 'vth': 10, 'currents': [5] * 6}, 6)
        self.assertEqual(s, [0, 0, 1, 0, 0, 1])
        self.assertEqual(v, [320, 640, 0, 320, 640, 0])

    def test_b_negative_voltage_decays_toward_zero(self):
        s, v = lif_trace({'du': 4095, 'dv': 2048, 'vth': 100, 'currents': [-3]}, 9)
        self.assertEqual(v, [-192, -96, -48, -24, -12, -6, -3, -1, 0])

    def test_c_current_decay(self):
        s, v = lif_trace({'du': 2047, 'dv': 4095, 'vth': 100, 'currents': [4]}, 4)
        self.assertEqual(v, [256, 128, 64, 32])

    def test_d_bias_mantissa_and_exponent(self):
        s, v = lif_trace({'du': 4095, 'dv': 0, 'vth': 5, 'bias_mant': 3, 'bias_exp': 6}, 4)
        self.assertEqual(s, [0, 1, 0, 1])

    def test_e_voltage_clips_at_the_24_bit_bound(self):
        s, v = lif_trace({'du': 4095, 'dv': 0, 'vth': 131071,
                          'bias_mant': -4096, 'bias_exp': 7}, 16)
        self.assertEqual(v[14], -7864320)
        self.assertEqual(v[15], -8388607)

    def test_f_periodic_reset(self):
        s, v = lif_trace({'du': 4095, 'dv': 0, 'vth': 1000, 'reset_interval': 4,
                          'reset_offset': 0, 'currents': [1] * 8}, 8)
        self.assertEqual(v, [64, 128, 192, 64, 128, 192, 256, 64])

    def test_g_reset_phase_survives_chunked_simulation(self):
        attrs = {'du': 4095, 'dv': 0, 'vth': 1000, 'reset_interval': 4,
                 'reset_offset': 0, 'currents': [1] * 8}
        self.assertEqual(lif_trace(attrs, 8, chunks=(3, 5)), lif_trace(attrs, 8))

    def test_i_initial_voltage(self):
        s, v = lif_trace({'du': 4095, 'dv': 0, 'vth': 10, 'initial_v': 640, 'currents': [1]}, 1)
        self.assertEqual((s, v), ([1], [0]))        # 640 + 64 > 640

    def _summed_input(self, external):
        """A spike at update 1 reaches the target through eight synapses of 32767 at update 2."""
        arch = architecture(soma=LIF)
        net = sanafe.Network()
        src = net.create_neuron_group('src', 1, model_attributes={
            'du': 4095, 'dv': 0, 'vth': 0, 'currents': [1]})
        dst = net.create_neuron_group('dst', 1, log_potential=True, model_attributes={
            'du': 4095, 'dv': 4095, 'vth': 131071, 'currents': [0, external]})
        src[0].map_to_core(arch.tiles[0].cores[0])
        dst[0].map_to_core(arch.tiles[0].cores[0])
        for _ in range(8):
            src[0].connect_to_neuron(dst[0], {'weight': 32767})
        chip = sanafe.SpikingChip(arch)
        chip.load(net)
        return chip.sim(2, potential_trace=True)['potential_trace'][1][-1]

    def test_h_rejects_out_of_range_input_and_attributes(self):
        # Positive controls: the base configuration, both ends of the accepted activation
        # range (2^18 + 32) and a synaptic plus external sum at the bound all run.
        base = {'du': 0, 'dv': 0, 'vth': 1}
        self.assertEqual(lif_trace(base, 1), ([0], [0]))
        for edge, v in ((262176, 2048), (-262176, -2048)):
            self.assertEqual(lif_trace({'du': 4095, 'dv': 0, 'vth': 131071, 'currents': [edge]}, 1),
                             ([0], [v]))
        self.assertEqual(self._summed_input(40), 2048)              # 8 * 32767 + 40 = 262176
        for value in (262177, -262177):
            with self.assertRaisesRegex(ValueError, 'LIF external activation must be'):
                lif_trace({'du': 4095, 'dv': 0, 'vth': 10, 'currents': [value]}, 1)
        with self.assertRaisesRegex(ValueError, 'LIF input activation must be'):
            self._summed_input(41)
        for bad, fragment in (({'du': 4096}, 'LIF unsigned 12-bit du must be'),
                              ({'dv': -1}, 'LIF unsigned 12-bit dv must be'),
                              ({'vth': 131072}, 'LIF unsigned 17-bit vth must be'),
                              ({'bias_exp': 8}, 'LIF unsigned 3-bit bias_exp must be'),
                              ({'bias_mant': 4096}, 'LIF signed 13-bit bias_mant must be')):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, fragment):
                lif_trace({**base, **bad}, 1)

    def test_j_current_wraps_once_at_the_24_bit_boundary(self):
        # Values observed from Lava's PyLifModelBitAcc (du = 4095, dv = 0, one update from rest).
        for a, s, v in ((131071, 0, 8388544), (131072, 1, 0), (131073, 0, -8388544),
                        (-131072, 1, 0), (-131073, 0, 8388544), (262144, 0, 0)):
            with self.subTest(a=a):
                self.assertEqual(lif_trace({'du': 4095, 'dv': 0, 'vth': 131071,
                                            'currents': [a]}, 1), ([s], [v]))


def if_trace(attributes: dict, steps: int) -> tuple[list[int], list[int]]:
    """Spike (0 or 1) and voltage after each update of one integrate_fire_int24 neuron."""
    return neuron_trace(attributes, steps)


class TestIfAgainstLeakFreeLif(unittest.TestCase):
    """The soft-reset IF and Loihi's built-in leak-free LIF agree up to the first spike and differ only in the reset."""

    def pair(self, theta, bias, a, steps):
        s_if, v_if = if_trace({'threshold': theta, 'initial_voltage': theta // 2,
                               'bias': bias, 'currents': a}, steps)
        s_lif, v_lif = lif_trace({'du': 4095, 'dv': 0, 'vth': theta - 1, 'initial_v': 32 * theta,
                                  'bias_mant': bias, 'bias_exp': 6, 'currents': a}, steps)
        return s_if, v_if, s_lif, v_lif

    @staticmethod
    def drift_positive(rng, theta):
        return rng.integers((-theta) // 4, theta // 2 + 1, size=200)

    @staticmethod
    def drift_zero(rng, theta):
        return rng.integers(-(theta // 4), theta // 4 + 1, size=200)

    @staticmethod
    def drift_negative_with_jumps(rng, theta):
        jumps = rng.random(200) < 0.05
        return np.where(jumps, rng.integers(theta // 3, 3 * theta // 2 + 1, size=200),
                        rng.integers(-(theta // 16) - 1, 1, size=200))

    def test_voltages_and_first_spike_agree(self):
        rng = np.random.default_rng(2)
        cases = ((40, 0, self.drift_positive), (40, 3, self.drift_positive),
                 (2, 0, self.drift_positive), (1000, -2, self.drift_positive),
                 (1000, 0, self.drift_zero), (1000, 1, self.drift_zero),
                 (1000, 0, self.drift_negative_with_jumps),
                 (40, -1, self.drift_negative_with_jumps),
                 (2, 1, self.drift_positive), (2, -1, self.drift_positive),
                 (4, 1, self.drift_zero), (4, -1, self.drift_zero))
        negative_before_spike = late_first_spike = exact_landing = 0
        for theta, bias, draw in cases:
            for _ in range(50):
                a = draw(rng, theta).tolist()
                s_if, v_if, s_lif, v_lif = self.pair(theta, bias, a, 200)
                k = s_if.index(1) if 1 in s_if else 200
                self.assertEqual(s_lif[:k + 1], s_if[:k + 1])
                self.assertEqual(v_lif[:k], [64 * v for v in v_if[:k]])
                if k < 200:
                    negative_before_spike += any(v < 0 for v in v_if[:k])
                    late_first_spike += k > 20
                    before = v_if[k - 1] if k > 0 else theta // 2
                    # the membrane at the firing step, where >= on the IF and > on vth = theta - 1 must agree
                    exact_landing += before + a[k] + bias == theta
        self.assertGreater(negative_before_spike, 0, 'a run that goes negative and then fires')
        self.assertGreater(late_first_spike, 0, 'a first spike after step 20')
        self.assertGreater(exact_landing, 0, 'a membrane exactly at the threshold at the firing step')

    def test_membrane_exactly_at_the_threshold_fires_in_both(self):
        # theta 40, initial 20, one input of 20: membrane 40
        s_if, v_if, s_lif, v_lif = self.pair(40, 0, [20], 1)
        self.assertEqual((s_if, s_lif), ([1], [1]))
        self.assertEqual((v_if, v_lif), ([0], [0]))

    def test_membrane_one_below_the_threshold_fires_in_neither(self):
        s_if, v_if, s_lif, v_lif = self.pair(40, 0, [19], 1)
        self.assertEqual((s_if, s_lif), ([0], [0]))
        self.assertEqual(v_lif, [64 * 39])

    def test_the_two_differ_only_in_the_reset(self):
        # theta 40, initial 20, one input of 30: membrane 50
        s_if, v_if, s_lif, v_lif = self.pair(40, 0, [30], 1)
        self.assertEqual((s_if, s_lif), ([1], [1]))
        self.assertEqual(v_if, [10])        # subtraction, the soft reset of conversion
        self.assertEqual(v_lif, [0])        # reset to zero, Loihi's built-in LIF
