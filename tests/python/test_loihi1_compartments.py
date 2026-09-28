"""Restricted source-derived compartment candidate, not silicon validation."""
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path

import sanafe


def chip_for(attributes, edges=()):
    text = (files('sanafe.examples') / 'loihi.yaml').read_text()
    text = text.replace('loihi_tile[0..31]', 'loihi_tile[0..0]')
    text = text.replace('loihi_core[0..3]', 'loihi_core[0..0]')
    text = text.replace('loihi_inputs[0..1023]', 'loihi_inputs[0..0]')
    text = text.replace('model: leaky_integrate_fire', 'model: paired_compartment_if')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'arch.yaml'
        path.write_text(text)
        arch = sanafe.load_arch(str(path))
    net = sanafe.Network()
    group = net.create_neuron_group('cx', len(attributes), log_potential=True,
                                    log_spikes=True)
    for neuron, attrs in zip(group, attributes):
        neuron.set_attributes(model_attributes=attrs)
        neuron.map_to_core(arch.tiles[0].cores[0])
    for pre, post, weight in edges:
        group[pre].connect_to_neuron(group[post], {'weight': weight})
    chip = sanafe.SpikingChip(arch)
    chip.load(net)
    return chip


def pair(bias=8192, threshold=16320):
    return [{'pair_role': 'dendrite', 'bias': bias},
            {'pair_role': 'soma', 'threshold': threshold}]


class TestPairedCompartmentIF(unittest.TestCase):
    def test_explicit_recurrent_reset_has_one_step_delay(self):
        self.assertIn('paired_compartment_if', sanafe.model_attributes)
        chip = chip_for(pair(), [(1, 0, -16320)])
        result = chip.sim(4, potential_trace=True, spike_trace=True, neuron_trace=True)
        self.assertEqual(result['potential_trace'],
                         [[8192, 8192], [16384, 0], [8256, 8256], [16448, 0]])
        self.assertEqual([len(s) for s in result['spike_trace']], [0, 1, 0, 1])
        self.assertEqual(result['neurons_updated'], 8)
        self.assertEqual(result['neuron_trace']['u'],
                         [[0, 0], [0, 0], [-16320, 0], [0, 0]])
        self.assertEqual(result['neuron_trace']['v'],
                         [[8192, 8192], [16384, 0], [8256, 8256], [16448, 0]])

    def test_no_hidden_subtraction_and_strict_threshold(self):
        result = chip_for(pair(bias=16320)).sim(
            3, potential_trace=True, spike_trace=True)
        self.assertEqual(result['potential_trace'],
                         [[16320, 16320], [32640, 0], [48960, 0]])
        self.assertEqual([len(s) for s in result['spike_trace']], [0, 1, 1])

    def test_negative_current_and_no_retained_current(self):
        attrs = [{'pair_role': 'hard', 'bias': 2, 'threshold': 1}, *pair(bias=0)]
        chip = chip_for(attrs, [(0, 1, -64)])
        result = chip.sim(3, potential_trace=True)
        self.assertEqual(result['potential_trace'], [[0, 0, 0], [0, -64, -64],
                                                    [0, -128, -128]])

    def test_hard_reset_and_padding_remain_separate(self):
        attrs = [{'pair_role': 'hard', 'bias': 64, 'threshold': 64},
                 {'pair_role': 'dummy'}, *pair()]
        result = chip_for(attrs).sim(2, potential_trace=True, spike_trace=True)
        self.assertEqual(result['potential_trace'],
                         [[64, 0, 8192, 8192], [0, 0, 16384, 0]])
        self.assertEqual(result['neurons_updated'], 6)

    def test_reset_and_chunked_execution_preserve_pair_state(self):
        chip = chip_for(pair(), [(1, 0, -16320)])
        expected = chip.sim(5, potential_trace=True, spike_trace=True)
        chip.reset()
        first = chip.sim(2, potential_trace=True, spike_trace=True)
        rest = chip.sim(3, potential_trace=True, spike_trace=True)
        self.assertEqual(first['potential_trace'] + rest['potential_trace'],
                         expected['potential_trace'])
        self.assertEqual([len(s) for s in first['spike_trace'] + rest['spike_trace']],
                         [0, 1, 0, 1, 0])

    def test_invalid_pair_order_and_missing_roles_fail(self):
        for attrs in ([{'pair_role': 'soma', 'threshold': 64}],
                      [{'pair_role': 'dendrite'}],
                      [{'pair_role': 'dummy'}, {'pair_role': 'soma', 'threshold': 64}],
                      [{'pair_role': 'hard', 'threshold': 64},
                       {'pair_role': 'soma', 'threshold': 64}],
                      [{'threshold': 64}], [{'pair_role': 'hard'}]):
            with self.subTest(attrs=attrs), self.assertRaises((ValueError, RuntimeError)):
                chip_for(attrs).sim(1)

    def test_invalid_scalar_attributes_fail(self):
        for name, values in {'threshold': [0, -1, 1.5, 8388608],
                             'bias': [.5, -8388609, 8388608, float('nan')],
                             'initial_voltage': [-1, 1, .5],
                             'pair_role': ['other']}.items():
            for value in values:
                attrs = {'pair_role': 'hard', 'threshold': 64, name: value}
                with self.subTest(name=name, value=value), self.assertRaises((ValueError, RuntimeError)):
                    chip_for([attrs]).sim(1)

    def test_soma_and_dummy_reject_bias_and_input(self):
        for role in ('soma', 'dummy'):
            attrs = pair() if role == 'soma' else [{'pair_role': 'dummy'}]
            attrs[-1]['bias'] = 1
            with self.subTest(role=role, kind='bias'), self.assertRaises((ValueError, RuntimeError)):
                chip_for(attrs).sim(1)
            attrs = [{'pair_role': 'hard', 'bias': 2, 'threshold': 1}]
            attrs += pair(bias=0) if role == 'soma' else [{'pair_role': 'dummy'}]
            with self.subTest(role=role, kind='input'), self.assertRaises((ValueError, RuntimeError)):
                chip_for(attrs, [(0, len(attrs)-1, 64)]).sim(2)

    def test_current_and_accumulated_voltage_bounds_fail(self):
        for weight in (.5, 8388608, -8388609):
            attrs = [{'pair_role': 'hard', 'bias': 2, 'threshold': 1}, *pair(bias=0)]
            with self.subTest(weight=weight), self.assertRaises((ValueError, RuntimeError)):
                chip_for(attrs, [(0, 1, weight)]).sim(2)
        for bias in (8388607, -8388608):
            with self.subTest(bias=bias), self.assertRaises((ValueError, RuntimeError)):
                chip_for(pair(bias=bias)).sim(2)


if __name__ == '__main__':
    unittest.main()
