"""Resource gates for the explicitly assumed candidate storage layout."""
import json
from pathlib import Path
from importlib.resources import files
import unittest
import numpy as np
from sanafe.loihi2 import (Allocation, audit_unit_scale_signed8_weights,
                          candidate_profile, load_loihi2_candidate,
                          validate_chain_resources)


class TestLoihi2Profile(unittest.TestCase):
    def test_yaml_copies_and_direct_loading_match_helper(self):
        from sanafecpp import load_arch
        packaged = files('sanafe.examples') / 'loihi2.yaml'
        repository = Path(__file__).resolve().parents[2] / 'arch' / 'loihi2.yaml'
        self.assertEqual(repository.read_text(), packaged.read_text())
        self.assertEqual(load_arch(str(repository)).configuration(),
                         load_loihi2_candidate().configuration())

    def test_unit_scale_weight_audit_is_not_a_physical_mapping_certificate(self):
        report = audit_unit_scale_signed8_weights([[[127, -128, 255, -256, 0]]])
        self.assertFalse(report['physical_mapping_verified'])
        self.assertEqual(report['out_of_range'], 2)
        self.assertEqual(report['nonzero'], 4)
        self.assertEqual(report['matrices'][0]['out_of_range'], 2)
        self.assertEqual(report['range'], [-128, 127])
        with self.assertRaises(ValueError):
            audit_unit_scale_signed8_weights([[[1.5]]])

    def test_small_chain_and_dense_zero_accounting(self):
        report = validate_chain_resources([2, 3], [np.zeros((3, 2), dtype=int)], [(0, 0), (1, 0)])
        self.assertFalse(report['physical_fit_verified'])
        first, second = report['cores']
        self.assertEqual((first['neurons'], first['outgoing_routes'], first['total_bytes']), (2, 2, 304))
        self.assertEqual((second['synapses'], second['incoming_axons'], second['total_bytes']), (6, 2, 344))
        json.dumps(report)
        json.dumps(candidate_profile())

    def test_colocation_counts_sources_once_per_destination_core(self):
        report = validate_chain_resources([2, 3, 1], [np.zeros((3, 2), dtype=int), np.zeros((1, 3), dtype=int)], [(0, 0)] * 3)
        core = report['cores'][0]
        self.assertEqual((core['neurons'], core['synapses'], core['incoming_axons'], core['outgoing_routes'], core['total_bytes']), (6, 9, 5, 5, 468))

    def test_synaptic_and_total_memory_are_separate_gates(self):
        with self.assertRaisesRegex(ValueError, 'synapse'):
            validate_chain_resources([182, 182], [np.zeros((182, 182), dtype=int)], [(0, 0), (1, 0)])
        with self.assertRaisesRegex(ValueError, 'total'):
            validate_chain_resources([8192], [], [(0, 0)], allocation=Allocation(state_bytes_per_neuron=24))

    def test_reservations_and_neuron_limit(self):
        with self.assertRaisesRegex(ValueError, 'reserved'):
            validate_chain_resources([1], [], [(0, 0)], reserved_cores=[(0, 0)])
        with self.assertRaisesRegex(ValueError, 'neuron'):
            validate_chain_resources([8193], [], [(0, 0)])

    def test_exact_memory_boundary_and_malformed_containers(self):
        report = validate_chain_resources([256, 128], [np.zeros((128, 256), dtype=np.int16)], [(0, 0), (1, 0)])
        self.assertEqual(report['cores'][1]['synapse_bytes'], 131072)
        for args, kwargs in [((None, [], []), {}), (([1], None, [(0, 0)]), {}),
                             (([1], [], None), {}), (([1], [], [(0, 0)]), {'reserved_cores': None})]:
            with self.subTest(args=args, kwargs=kwargs), self.assertRaises(ValueError):
                validate_chain_resources(*args, **kwargs)

    def test_invalid_inputs_are_rejected(self):
        for sizes, weights, places in [([], [], []), ([True], [], [(0, 0)]), ([1.0], [], [(0, 0)]), ([1], [], [(32, 0)]), ([1], [], [(0, 4)]), ([1], [], [(0, False)]), ([1, 1], [[[1.5]]], [(0, 0), (1, 0)]), ([1, 1], [[[32768]]], [(0, 0), (1, 0)]), ([1, 1], [[[True]]], [(0, 0), (1, 0)]), ([1, 2], [[[1]]], [(0, 0), (1, 0)])]:
            with self.subTest(sizes=sizes, weights=weights, places=places), self.assertRaises(ValueError):
                validate_chain_resources(sizes, weights, places)
        for kwargs in [{'synapse_bytes': 0}, {'program_bytes_per_core': -1}, {'incoming_axon_bytes': 1.5}, {'state_bytes_per_neuron': True}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Allocation(**kwargs)

    def test_candidate_costs_derive_from_loihi1_by_the_stated_factors(self):
        """Every cost is the fitted Loihi 1 value, scaled only where a factor is declared."""
        import yaml
        from sanafe.loihi2 import (LATENCY_SCALING, SYNC_SCALING, INHERITED_COSTS,
                                   candidate_cost_provenance)
        base = yaml.safe_load((files('sanafe.examples') / 'loihi.yaml').read_text())['architecture']
        cand = yaml.safe_load((files('sanafe.examples') / 'loihi2.yaml').read_text())['architecture']

        def flat(node, path=()):
            if isinstance(node, dict):
                for key, value in node.items():
                    yield from flat(value, path + (key,))
            elif isinstance(node, list):
                for item in node:
                    yield from flat(item, path + (item.get('name', '?') if isinstance(item, dict) else '?',))
            else:
                yield path, node

        base_costs = {path: value for path, value in flat(base)
                      if path[-1] in LATENCY_SCALING or path[-1] in INHERITED_COSTS}
        seen = set()
        for path, value in flat(cand):
            name = path[-1]
            if name in LATENCY_SCALING:
                factor, _ = LATENCY_SCALING[name]
                self.assertIn(path, base_costs, path)
                self.assertAlmostEqual(value, base_costs[path] / factor, delta=abs(value) * 1e-3)
                seen.add(name)
            elif name in INHERITED_COSTS:
                self.assertIn(path, base_costs, path)
                self.assertEqual(value, base_costs[path], path)
                seen.add(name)
        self.assertEqual(seen, set(LATENCY_SCALING) | set(INHERITED_COSTS))
        base_sync = base['attributes']['latency_sync']
        cand_sync = cand['attributes']['latency_sync']
        self.assertEqual(set(cand_sync), set(base_sync))
        for tiles, value in cand_sync.items():
            self.assertAlmostEqual(value, base_sync[tiles] / SYNC_SCALING[0], delta=value * 1e-3)
        self.assertAlmostEqual(cand_sync[max(cand_sync)], 200e-9, delta=1e-12)
        provenance = candidate_cost_provenance()
        self.assertEqual({k for k, v in provenance.items() if v['status'] == 'scaled'},
                         set(LATENCY_SCALING) | {'latency_sync'})
        self.assertEqual(provenance, candidate_profile()['cost_provenance'])

    def test_candidate_architecture_uses_only_integer_models(self):
        profile = candidate_profile()
        self.assertEqual(profile['limits']['native_stored_weight_bits_max'], 8)
        self.assertFalse(profile['physical_fit_verified'])
        self.assertIn('multiple 8-bit synapses', profile['weight_mapping'])
        self.assertIn('documented 8 by 4', profile['provenance']['mesh_coordinates']['status'])
        arch = load_loihi2_candidate()
        self.assertEqual(len(arch.tiles), 32)
        for tile in arch.tiles:
            self.assertEqual(len(tile.cores), 4)
            for core in tile.cores:
                self.assertEqual(core.max_neurons_supported, 8192)
                self.assertEqual([x.model_info.name for x in core.pipeline_hw if x.implements_soma], ['integrate_fire_int24'])
                self.assertEqual([x.model_info.name for x in core.pipeline_hw if x.implements_dendrite], ['accumulator_int'])
                self.assertEqual([x.model_info.name for x in core.pipeline_hw if x.implements_synapse], ['current_based_int'])


if __name__ == '__main__':
    unittest.main()


class TestCoreBudgets(unittest.TestCase):
    """Per-core budget check shared by the chunked QCFS mappers."""

    def stat(self, **counts):
        base = {'tile': 3, 'core': 1, 'neurons': 10, 'edges': 20, 'outgoing_neurons': 3}
        base.update(counts)
        return base

    def test_bytes_follow_the_assumed_layout(self):
        from sanafe.loihi2 import validate_core_budgets
        [checked] = validate_core_budgets([self.stat()])
        self.assertEqual(checked['assumed_synapse_bytes'], 80)
        self.assertEqual(checked['assumed_total_bytes'], 10*16 + 20*(4+8) + 3*8 + 256)
        self.assertEqual((checked['tile'], checked['core'], checked['neurons']), (3, 1, 10))

    def test_each_ceiling_raises(self):
        from sanafe.loihi2 import validate_core_budgets
        cases = [(self.stat(neurons=8193), 'neuron'),
                 (self.stat(edges=32769, neurons=1, outgoing_neurons=0), 'synapse'),
                 (self.stat(neurons=8000, edges=5000, outgoing_neurons=8000), 'total')]
        for stat, budget in cases:
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, budget):
                validate_core_budgets([stat])
        with self.assertRaisesRegex(ValueError, r'core \(3, 1\)'):
            validate_core_budgets([self.stat(neurons=8193)])

    def test_input_is_not_mutated(self):
        from sanafe.loihi2 import validate_core_budgets
        stat = self.stat()
        validate_core_budgets([stat])
        self.assertNotIn('assumed_total_bytes', stat)

    def test_rejects_non_integer_counts(self):
        from sanafe.loihi2 import validate_core_budgets
        for bad in (self.stat(neurons=1.5), self.stat(edges=-1), self.stat(neurons=True)):
            with self.assertRaises(ValueError):
                validate_core_budgets([bad])
