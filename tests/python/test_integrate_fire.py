"""Float32 IF regression tests, run with unittest discovery."""
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path

import sanafe


def architecture(model, dendrite_model="accumulator"):
    # Keep upstream architecture costs unchanged; replace only soma semantics.
    text = (files('sanafe.examples') / 'loihi.yaml').read_text()
    text = text.replace('model: leaky_integrate_fire', 'model: ' + model)
    text = text.replace('model: accumulator\n', 'model: ' + dendrite_model + '\n')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'arch.yaml'
        path.write_text(text)
        return sanafe.load_arch(str(path))


def chip_for(attributes, model='integrate_fire_float32'):
    arch = architecture(model)
    net = sanafe.Network()
    group = net.create_neuron_group('if', len(attributes), log_spikes=True,
                                    log_potential=True)
    for neuron, attrs in zip(group, attributes):
        neuron.set_attributes(model_attributes=attrs)
        neuron.map_to_core(arch.tiles[0].cores[0])
    chip = sanafe.SpikingChip(arch)
    chip.load(net)
    return chip


class TestFloatIF(unittest.TestCase):
    def test_existing_loihi_equality_is_different(self):
        chip = chip_for([{'threshold': 1., 'bias': 1., 'reset_mode': 'soft'}],
                        model='leaky_integrate_fire')
        self.assertEqual(chip.sim(1)['spikes'], 0)

    def test_equality_negative_state_and_single_spike_backlog(self):
        chip = chip_for([
            {'threshold': 1., 'initial_voltage': .5, 'currents': [.5, -2., 3., 0.]},
            {'threshold': 2., 'initial_voltage': 1., 'currents': [5., 0., 0., 0.]},
        ])
        result = chip.sim(4, spike_trace=True, potential_trace=True)
        self.assertEqual(result['potential_trace'], [[0., 4.], [-2., 2.], [0., 0.], [0., 0.]])
        self.assertEqual([len(s) for s in result['spike_trace']], [2, 1, 2, 0])

    def test_float32_rounding(self):
        import struct
        f32 = lambda x: struct.unpack('f', struct.pack('f', x))[0]
        expected = []
        voltage = 0.
        for _ in range(3):
            voltage = f32(voltage + f32(.1))
            expected.append([voltage])
        chip = chip_for([{'threshold': 1., 'currents': [.1, .1, .1]}])
        self.assertEqual(chip.sim(3, potential_trace=True)['potential_trace'], expected)

    def test_chunking_and_reset(self):
        attrs = [{'threshold': 1., 'initial_voltage': .5,
                  'currents': [.5, -.5, .25, 2.]}]
        chip = chip_for(attrs)
        full = chip.sim(4, spike_trace=True, potential_trace=True)
        chip.reset()
        first = chip.sim(2, spike_trace=True, potential_trace=True)
        last = chip.sim(2, spike_trace=True, potential_trace=True)
        self.assertEqual(full['potential_trace'], first['potential_trace'] + last['potential_trace'])
        addresses = lambda trace: [[(n.group_name, n.neuron_offset) for n in step] for step in trace]
        self.assertEqual(addresses(full['spike_trace']), addresses(first['spike_trace'] + last['spike_trace']))


if __name__ == '__main__':
    unittest.main()
