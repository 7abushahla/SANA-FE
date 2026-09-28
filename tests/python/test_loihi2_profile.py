"""Resource gates for the explicitly assumed candidate storage layout."""
import json
import unittest
import numpy as np
from sanafe.loihi2 import Allocation, candidate_profile, load_loihi2_candidate, validate_chain_resources


class TestLoihi2Profile(unittest.TestCase):
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

    def test_candidate_architecture_uses_only_integer_models(self):
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
