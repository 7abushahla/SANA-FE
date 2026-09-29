"""UpdateRecord construction from one-step SANA-FE results."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import sanafe
from sanafe.loihi2 import load_loihi2_candidate
from sanafe.studio.engine import (ChipLayout, UpdateRecord, build_update_record,
                                  instrument_arch_yaml, neuron_map)

from studio_helpers import PLACEMENTS, candidate_yaml, chain_network

TRACES = dict(spike_trace=True, potential_trace=True, message_trace=True,
              perf_trace=True)


def stepped_records(placement, steps=6, network=None):
    directory = tempfile.TemporaryDirectory()
    path = Path(directory.name) / 'instrumented.yaml'
    instrument_arch_yaml(candidate_yaml(), path)
    canonical = load_loihi2_candidate()
    chip = sanafe.SpikingChip(sanafe.load_arch(str(path)))
    chip.load(network or chain_network(canonical, PLACEMENTS[placement], steps))
    layout, neurons = ChipLayout.from_chip(chip), neuron_map(chip)
    records = [build_update_record(update, chip.sim(1, **TRACES), layout, neurons)
               for update in range(1, steps + 1)]
    directory.cleanup()
    return records, layout


class TestUpdateRecords(unittest.TestCase):
    def test_counts_match_recorded_totals(self):
        records, _ = stepped_records('far')
        for record in records:
            self.assertEqual(record.counts['fired'], len(record.fired))
            self.assertEqual(record.counts['spikes'],
                             sum(m.spikes for m in record.messages))
            self.assertEqual(record.counts['hops'], sum(m.hops for m in record.messages))
            self.assertEqual(record.counts['messages'], len(record.messages))

    def test_every_occupied_core_finishes(self):
        records, _ = stepped_records('far')
        for record in records:
            self.assertEqual(set(record.core_finish), {'0.0', '16.0', '31.0'})
            self.assertTrue(all(value > 0 for value in record.core_finish.values()))

    def test_barrier_equals_sync_table_entry(self):
        for placement, barrier in (('far', 1.0e-6), ('one_router', 0.6e-6)):
            with self.subTest(placement=placement):
                records, _ = stepped_records(placement)
                for record in records:
                    self.assertAlmostEqual(record.barrier, barrier, delta=1e-12)

    def test_energy_closes(self):
        records, _ = stepped_records('far')
        for record in records:
            cores = sum(entry['total'] for entry in record.core_energy.values())
            network = sum(record.tile_network_energy.values())
            self.assertAlmostEqual((cores + network) / record.energy['total'], 1.0,
                                   places=9)
            for entry in record.core_energy.values():
                self.assertEqual(set(entry['units']),
                                 {'loihi_lif', 'loihi_dendrites', 'loihi_dense_synapse'})
                self.assertGreaterEqual(entry['axon'], -1e-18)

    def test_paths_and_potentials(self):
        records, layout = stepped_records('far')
        for record in records:
            self.assertEqual(len(record.potentials), 14)
            for message in record.messages:
                self.assertEqual(len(message.path) - 1, message.hops)
                self.assertEqual(message.path[0], int(message.src.split('.')[0]))
                self.assertEqual(message.path[-1], int(message.dst.split('.')[0]))
        arch = load_loihi2_candidate()
        chip = sanafe.SpikingChip(arch)
        chip.load(chain_network(arch, PLACEMENTS['far'], 6))
        batch = np.asarray(chip.sim(6, potential_trace=True)['potential_trace'])
        stepped = np.array([list(r.potentials.values()) for r in records])
        np.testing.assert_array_equal(stepped, batch)

    def test_silent_update_builds_record(self):
        arch = load_loihi2_candidate()
        net = sanafe.Network()
        group = net.create_neuron_group('quiet', 2, log_spikes=True, log_potential=True)
        for neuron in group:
            neuron.set_attributes(model_attributes={'threshold': 100, 'bias': 0,
                                                    'initial_voltage': 0})
            neuron.map_to_core(arch.tiles[3].cores[2])
        records, _ = stepped_records('far', steps=2, network=net)
        for record in records:
            self.assertEqual(record.messages, [])
            self.assertEqual(record.counts['messages'], 0)
            self.assertEqual(set(record.core_counts), {'3.2'})
            self.assertEqual(record.core_counts['3.2']['packets_out'], 0)

    def test_round_trip_through_json(self):
        records, _ = stepped_records('far', steps=2)
        for record in records:
            data = json.loads(json.dumps(record.to_dict(), allow_nan=False))
            self.assertEqual(UpdateRecord.from_dict(data), record)
            self.assertEqual(record.provenance['message.path'], 'X')
            self.assertEqual(record.provenance['core_energy.total'], 'R')


if __name__ == '__main__':
    unittest.main()
