"""The platform catalog: curated facts, structural facts read from the YAML, provenance."""
from importlib.resources import files
from pathlib import Path
import unittest

import yaml
import sanafe


def example(name):
    return files('sanafe.examples') / name


class TestDocumentedTrueNorth(unittest.TestCase):
    def test_differs_from_shipped_file_in_exactly_two_attributes(self):
        shipped = yaml.safe_load(example('truenorth.yaml').read_text())['architecture']
        documented = yaml.safe_load(example('truenorth_documented.yaml').read_text())['architecture']
        self.assertEqual(documented['attributes']['sync_model'], 'fixed')
        self.assertEqual(documented['attributes']['latency_sync'], 1.0e-3)
        synapse = documented['tile'][0]['core'][0]['synapse'][0]['attributes']
        self.assertEqual(synapse['energy_process_spike'], 26.0e-12)
        # Everything else is the shipped file.
        documented['attributes']['latency_sync'] = shipped['attributes']['latency_sync']
        synapse['energy_process_spike'] = 0.0
        documented['name'] = shipped['name']
        self.assertEqual(documented, shipped)

    def test_packaged_copy_matches_repository_copy_and_loads(self):
        repo = Path(__file__).resolve().parents[2] / 'arch' / 'truenorth_documented.yaml'
        self.assertEqual(repo.read_text(), example('truenorth_documented.yaml').read_text())
        arch = sanafe.load_arch(str(repo))
        # A fixed sync model is a one-entry table; timestep_delay is a separate field.
        self.assertEqual(list(arch.configuration()['sync_table'].values()), [1.0e-3])


import json
import random

from sanafe import platforms as P


class TestRegistry(unittest.TestCase):
    def test_four_platforms_in_order(self):
        self.assertEqual([p.id for p in P.registry()],
                         ['loihi', 'loihi2', 'truenorth', 'truenorth_documented'])
        with self.assertRaises(KeyError):
            P.get('speck')

    def test_structure_facts_come_from_the_yaml(self):
        for platform in P.registry():
            with self.subTest(platform=platform.id):
                data = yaml.safe_load(platform.arch_yaml.read_text())['architecture']
                card = P.describe(platform)
                self.assertEqual(card['id'], platform.id)
                self.assertEqual((card['structure']['width'], card['structure']['height']),
                                 (data['attributes']['width'], data['attributes']['height']))
                chip = sanafe.SpikingChip(sanafe.load_arch(str(platform.arch_yaml)))
                described = chip.describe()
                self.assertEqual(card['structure']['tiles'], len(described['tiles']))
                self.assertEqual(card['structure']['cores'], described['core_count'])
                self.assertEqual(card['structure']['max_neurons_per_core'],
                                 data['tile'][0]['core'][0]['attributes']['max_neurons_supported'])
                models = {u['name']: u['model'] for u in card['units'] if u['role'] != 'axon'}
                for role in ('soma', 'synapse', 'dendrite'):
                    for unit in data['tile'][0]['core'][0][role]:
                        self.assertEqual(models[unit['name'].split('[')[0]], unit['attributes']['model'])
                self.assertTrue(card['not_hardware'].startswith('Nothing on this card'))
                if platform.id == 'loihi':
                    inputs = next(u for u in card['units'] if u['name'] == 'loihi_inputs')
                    self.assertEqual(inputs['instances'], 1024)

    def test_units_list_model_attributes_not_costs(self):
        card = P.describe(P.get('loihi'))
        soma = next(u for u in card['units'] if u['name'] == 'loihi_lif')
        names = {a['name'] for a in soma['attributes']}
        self.assertIn('leak_decay', names)
        self.assertIn('threshold', names)
        self.assertNotIn('energy_access_neuron', names)
        self.assertNotIn('soma_hw_name', names)
        self.assertEqual(soma['model'], 'leaky_integrate_fire')
        axon = next(u for u in card['units'] if u['name'] == 'loihi_in')
        self.assertEqual(axon['role'], 'axon')

    def test_every_cost_attribute_in_each_yaml_has_a_status(self):
        for platform in P.registry():
            with self.subTest(platform=platform.id):
                rows = P.describe(platform)['costs']
                self.assertTrue(rows)
                self.assertTrue(all(r['status'] in P.STATUSES for r in rows), rows)
                names = {r['attribute'] for r in rows}
                self.assertIn('latency_sync', names)
                self.assertIn('energy_process_spike', names)
                self.assertIn('latency_north_hop', names)

    def test_statuses_match_the_audit(self):
        loihi = {r['attribute']: r for r in P.describe(P.get('loihi'))['costs'] if r['unit'] == 'loihi_tile'}
        self.assertEqual(loihi['latency_north_hop']['status'], 'documented')
        self.assertEqual(loihi['energy_east_hop']['status'], 'documented')
        self.assertEqual(loihi['energy_north_hop']['status'], 'fitted')
        loihi2 = {(r['unit'], r['attribute']): r for r in P.describe(P.get('loihi2'))['costs']}
        self.assertEqual(loihi2[('loihi_lif', 'latency_spike_out')]['status'], 'scaled')
        self.assertEqual(loihi2[('loihi_lif', 'latency_spike_out')]['factor'], 10.0)
        self.assertEqual(loihi2[('loihi_lif', 'energy_spike_out')]['status'], 'inherited')
        self.assertEqual(loihi2[('loihi_dense_synapse', 'latency_process_spike')]['value'], 0.76e-9)
        tn = {r['status'] for r in P.describe(P.get('truenorth'))['costs']}
        self.assertEqual(tn, {'none'})
        doc = {r['attribute']: r for r in P.describe(P.get('truenorth_documented'))['costs']}
        self.assertEqual(doc['energy_process_spike']['status'], 'documented')
        self.assertEqual(doc['latency_sync']['status'], 'documented')
        self.assertEqual(doc['energy_access_neuron']['status'], 'none')

    def test_unlisted_cost_attribute_is_an_error(self):
        broken = P.Platform(**{**P.get('truenorth').__dict__, 'costs': {}})
        with self.assertRaises(ValueError):
            P.describe(broken)

    def test_badges(self):
        self.assertEqual(P.badge(P.get('loihi2')),
                         'Intel Loihi 2 candidate · latencies scaled from Loihi 1 by Intel-stated factors · '
                         'energy inherited from Loihi 1 · not hardware')
        self.assertEqual(P.badge(P.get('loihi')),
                         'Intel Loihi 1 · costs fitted on Nahuku (TCAD 2025, within 11.7% energy and '
                         '24.3% latency) · not hardware')
        self.assertEqual(P.badge(P.get('truenorth')),
                         'IBM TrueNorth (functional) · no cost model: energy and time are zero · not hardware')
        self.assertEqual(P.badge(P.get('truenorth_documented')),
                         'IBM TrueNorth (documented costs) · 26 pJ per synaptic event and a 1 ms tick '
                         'from published figures · not hardware')

    def test_demo_neurons_use_each_platforms_soma_model(self):
        rng = random.Random(1)
        for platform in P.registry():
            with self.subTest(platform=platform.id):
                card = P.describe(platform)
                soma = next(u for u in card['units'] if u['role'] == 'soma')
                accepted = {a['name'] for a in soma['attributes']}
                for kind in ('input', 'hidden'):
                    attrs = platform.demo_neuron(kind, rng)
                    self.assertTrue(set(attrs) <= accepted | {'reset_mode'}, (platform.id, attrs))
                    self.assertIn('threshold', attrs)
                self.assertIn('bias', platform.demo_neuron('input', rng))
                self.assertNotIn('bias', platform.demo_neuron('hidden', rng))
                weight = platform.demo_weight(rng)
                if platform.id == 'loihi2':
                    self.assertIsInstance(weight, int)
                else:
                    self.assertIsInstance(weight, float)

    def test_card_is_json_serializable(self):
        for platform in P.registry():
            json.dumps(P.describe(platform))


class TestCatalogCost(unittest.TestCase):
    def test_facts_never_build_a_full_chip(self):
        """The card needs only one core's units, so the catalog must not
        construct the whole chip (seconds and gigabytes for Loihi 1)."""
        from unittest import mock
        seen = []
        real = sanafe.SpikingChip

        def recording(arch, *args, **kwargs):
            seen.append(arch.configuration()['core_count'])
            return real(arch, *args, **kwargs)

        P._facts.cache_clear()
        with mock.patch.object(sanafe, 'SpikingChip', recording):
            card = P.describe(P.get('loihi'))
        self.assertEqual(seen, [1])
        self.assertEqual(card['structure']['cores'], 128)
        soma = next(u for u in card['units'] if u['name'] == 'loihi_lif')
        self.assertIn('leak_decay', {a['name'] for a in soma['attributes']})


class TestCatalogMatching(unittest.TestCase):
    def test_input_placeholder_unit_is_not_labeled_fitted(self):
        rows = {(r['unit'], r['attribute']): r for r in P.describe(P.get('loihi'))['costs']}
        self.assertEqual(rows[('loihi_inputs', 'energy_access_neuron')]['status'], 'none')
        self.assertEqual(rows[('loihi_lif', 'energy_access_neuron')]['status'], 'fitted')

    def test_match_identifies_each_catalog_architecture_and_nothing_else(self):
        for platform in P.registry():
            with self.subTest(platform=platform.id):
                self.assertEqual(P.match(sanafe.load_arch(str(platform.arch_yaml))), platform.id)
        repo = Path(__file__).resolve().parents[2]
        self.assertIsNone(P.match(sanafe.load_arch(str(repo / 'arch' / 'example_chip.yaml'))))
        text = P.get('loihi2').arch_yaml.read_text().replace(
            'energy_spike_out: 69.3e-12', 'energy_spike_out: 70.0e-12')
        import tempfile
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / 'modified.yaml'
            path.write_text(text)
            self.assertIsNone(P.match(sanafe.load_arch(str(path))))
