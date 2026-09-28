"""Integer numerical models and dynamic accumulator regression tests."""
import tempfile
import subprocess
import sys
import unittest
from importlib.resources import files
from pathlib import Path

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
