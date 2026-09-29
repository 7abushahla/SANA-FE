"""Read-only mapped-neuron identity used by the Studio neuron map."""
from pathlib import Path
import unittest

import sanafe

REPO = Path(__file__).resolve().parents[2]


class TestMappedNeuronIdentity(unittest.TestCase):
    def setUp(self):
        arch = sanafe.load_arch(str(REPO / 'arch' / 'example_chip.yaml'))
        net = sanafe.load_net(str(REPO / 'snn' / 'example.net'), arch,
                              use_netlist_format=True)
        self.chip = sanafe.SpikingChip(arch)
        self.chip.load(net)

    def test_groups_offsets_and_cores_match_netlist(self):
        # snn/example.net maps offsets 0 and 1 of each group to core 0.0 and
        # offset 2 to core 0.1.
        groups = self.chip.mapped_neuron_groups
        self.assertEqual(list(groups), ['0', '1'])
        for name, members in groups.items():
            self.assertEqual([n.group_name for n in members], [name] * 3)
            self.assertEqual([n.offset for n in members], [0, 1, 2])
            self.assertEqual([(n.tile_id, n.core_offset) for n in members],
                             [(0, 0), (0, 0), (0, 1)])
            self.assertEqual([n.core_id for n in members], [0, 0, 1])

    def test_log_potential_matches_trace_width(self):
        logged = [n for members in self.chip.mapped_neuron_groups.values()
                  for n in members if n.log_potential]
        result = self.chip.sim(1, potential_trace=True)
        self.assertEqual(len(logged), len(result['potential_trace'][0]))
        self.assertTrue(all(n.log_spikes for members in
                            self.chip.mapped_neuron_groups.values()
                            for n in members))

    def test_attributes_are_read_only(self):
        neuron = self.chip.mapped_neuron_groups['0'][0]
        with self.assertRaises(AttributeError):
            neuron.offset = 5


class TestNetworkNeuronAttributes(unittest.TestCase):
    def test_network_neuron_model_attributes(self):
        net = sanafe.Network()
        group = net.create_neuron_group('g', 2)
        group[1].set_attributes(model_attributes={'threshold': 4, 'bias': 0,
                                                  'initial_voltage': 2})
        self.assertEqual(group[1].model_attributes,
                         {'threshold': 4, 'bias': 0, 'initial_voltage': 2})
        self.assertEqual(group[0].model_attributes, {})
        with self.assertRaises(AttributeError):
            group[1].model_attributes = {}


if __name__ == '__main__':
    unittest.main()
