"""Connectivity summary, neuron detail, and the candidate budget gate."""
import unittest

import sanafe
from sanafe.loihi2 import load_loihi2_candidate
from sanafe.studio.engine import BuiltWorkload, Connectivity, Session, neuron_map

from studio_helpers import PLACEMENTS, REPO, ChainWorkload, candidate_yaml, chain_network


def loaded(placement):
    arch = load_loihi2_candidate()
    network = chain_network(arch, PLACEMENTS[placement], 6)
    chip = sanafe.SpikingChip(arch)
    chip.load(network)
    return Connectivity.from_network(network, neuron_map(chip))


class TestConnectivity(unittest.TestCase):
    def test_group_edges_and_core_links(self):
        far = loaded('far')
        self.assertEqual(far.group_edges, [
            {'src': 'layer_0', 'dst': 'layer_1', 'synapses': 32},
            {'src': 'layer_1', 'dst': 'layer_2', 'synapses': 8}])
        self.assertEqual(far.core_links, [
            {'src': '0.0', 'dst': '16.0', 'synapses': 32, 'axons': 8},
            {'src': '16.0', 'dst': '31.0', 'synapses': 8, 'axons': 4}])
        self.assertEqual(loaded('one_router').core_links[0]['dst'], '0.1')

    def test_core_stats_neurons_and_attributes(self):
        far = loaded('far')
        stats = {(s['tile'], s['core']): s for s in far.core_stats()}
        self.assertEqual(stats[(16, 0)], {'tile': 16, 'core': 0, 'neurons': 4,
                                          'edges': 32, 'outgoing_neurons': 4})
        self.assertEqual(stats[(0, 0)]['edges'], 0)
        self.assertEqual(stats[(31, 0)]['outgoing_neurons'], 0)
        self.assertEqual(far.core_neurons['0.0'], [['layer_0', 0, 7]])
        self.assertEqual(far.group_attributes['layer_1'], {'threshold': 4})

    def test_neuron_detail(self):
        detail = loaded('far').neuron('layer_1', 2)
        self.assertEqual((detail['group'], detail['offset'], detail['core']),
                         ('layer_1', 2, '16.0'))
        self.assertEqual(detail['attributes'],
                         {'threshold': 4, 'bias': 0, 'initial_voltage': 2})
        self.assertEqual(detail['fan_in_total'], 8)
        self.assertEqual(detail['fan_in'][0], {'neuron': 'layer_0.0', 'core': '0.0',
                                               'weight': 3})
        self.assertEqual(detail['fan_out_total'], 2)
        self.assertEqual([e['neuron'] for e in detail['fan_out']],
                         ['layer_2.0', 'layer_2.1'])
        self.assertTrue(detail['log_spikes'] and detail['log_potential'])
        limited = loaded('far').neuron('layer_1', 2, limit=3)
        self.assertEqual((len(limited['fan_in']), limited['fan_in_total']), (3, 8))
        for group, offset in (('layer_9', 0), ('layer_1', 4)):
            with self.assertRaises(KeyError):
                loaded('far').neuron(group, offset)


class OverBudgetWorkload(ChainWorkload):
    """200 x 200 dense edges into one core: 160000 synapse bytes > 131072."""

    name = 'over-budget'

    def build(self, params):
        arch = load_loihi2_candidate()
        network = chain_network(arch, PLACEMENTS['near'], 2, sizes=(200, 200, 1))
        return BuiltWorkload(arch_yaml=candidate_yaml(), arch=arch, network=network,
                             horizon=2)


class TestSessionConnectivity(unittest.TestCase):
    def test_summary_carries_connectivity(self):
        session = Session(ChainWorkload(), {})
        self.addCleanup(session.close)
        summary = session.describe()['network']
        self.assertEqual(len(summary['connections']), 2)
        self.assertEqual(summary['core_links'][0]['src'], '0.0')
        self.assertEqual(summary['core_neurons']['31.0'], [['layer_2', 0, 1]])
        self.assertEqual(summary['group_attributes']['layer_0'], {'threshold': 4})
        self.assertEqual(session.neuron_detail('layer_2', 1)['fan_in_total'], 4)
        self.assertNotIn('reference', session.neuron_detail('layer_2', 1))

    def test_candidate_budget_fails_the_build(self):
        with self.assertRaisesRegex(ValueError, r'core \(1, 0\) exceeds the synapse'):
            Session(OverBudgetWorkload(), {})

    def test_other_architectures_skip_the_candidate_budget(self):
        from sanafe.studio.engine import SanafeFiles
        session = Session(SanafeFiles(), {
            'arch_yaml': str(REPO / 'arch' / 'example_chip.yaml'),
            'net_file': str(REPO / 'snn' / 'example.net')})
        self.addCleanup(session.close)
        self.assertEqual(sorted(session.network_summary()['core_neurons']), ['0.0', '0.1'])

    def test_series_is_offered_when_the_checker_has_it(self):
        class Series:
            def check(self, record):
                return {'status': 'match', 'references': ['fake']}

            def series(self, key):
                return {'fake': {'potential': [key], 'spike': [None]}}

        class WithSeries(ChainWorkload):
            def build(self, params):
                built = super().build(params)
                built.reference = Series()
                return built

        session = Session(WithSeries(), {})
        self.addCleanup(session.close)
        self.assertEqual(session.neuron_detail('layer_0', 3)['reference'],
                         {'fake': {'potential': ['layer_0.3'], 'spike': [None]}})
        [record] = session.step(1)
        self.assertEqual(record.reference, {'status': 'match', 'references': ['fake']})


if __name__ == '__main__':
    unittest.main()
