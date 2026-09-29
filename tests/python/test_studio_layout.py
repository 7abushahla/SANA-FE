"""Chip geometry from describe() and SANA-FE's x-then-y routes."""
import json
import unittest

import sanafe
from sanafe.loihi2 import load_loihi2_candidate
from sanafe.studio.engine import ChipLayout, xy_path

from studio_helpers import PLACEMENTS, REPO, chain_network


def layout_for(name):
    return ChipLayout.from_chip(sanafe.SpikingChip(
        sanafe.load_arch(str(REPO / 'arch' / f'{name}.yaml'))))


class TestChipLayout(unittest.TestCase):
    def test_dimensions_for_bundled_architectures(self):
        expected = {'example_chip': (2, 1, 4), 'loihi': (8, 4, 4),
                    'loihi2': (8, 4, 4), 'truenorth': (64, 64, 1)}
        for name, (width, height, cores) in expected.items():
            with self.subTest(name=name):
                layout = layout_for(name)
                self.assertEqual((layout.width, layout.height), (width, height))
                self.assertEqual(len(layout.tiles), width * height)
                self.assertEqual(len(layout.tiles[0].cores), cores)

    def test_partially_filled_mesh(self):
        # demo_with_dendrites.yaml declares a 2 x 1 mesh but defines one tile.
        layout = layout_for('demo_with_dendrites')
        self.assertEqual((layout.width, layout.height, len(layout.tiles)), (2, 1, 1))
        self.assertEqual(layout.tile_at(0, 0).tile_id, 0)
        with self.assertRaisesRegex(ValueError, r'no tile at \(1, 0\)'):
            layout.tile_at(1, 0)

    def test_tile_coordinates_follow_arch_cpp(self):
        layout = layout_for('loihi2')
        self.assertEqual((layout.tiles[16].x, layout.tiles[16].y), (4, 0))
        self.assertEqual((layout.tiles[31].x, layout.tiles[31].y), (7, 3))
        self.assertEqual(layout.tile_at(7, 3).tile_id, 31)
        core = layout.core('9.1')
        self.assertEqual((core.tile_id, core.offset, core.key), (9, 1, '9.1'))
        # describe() lists units in pipeline order: synapse, dendrite, soma.
        self.assertEqual(core.units, ('loihi_dense_synapse', 'loihi_dendrites',
                                      'loihi_lif'))

    def test_xy_path_moves_x_first(self):
        layout = layout_for('loihi2')
        self.assertEqual(xy_path(layout, 0, 31), [0, 4, 8, 12, 16, 20, 24, 28, 29, 30, 31])
        self.assertEqual(xy_path(layout, 31, 0), [31, 27, 23, 19, 15, 11, 7, 3, 2, 1, 0])
        self.assertEqual(xy_path(layout, 5, 5), [5])

    def test_path_length_matches_recorded_hops(self):
        arch = load_loihi2_candidate()
        chip = sanafe.SpikingChip(arch)
        chip.load(chain_network(arch, PLACEMENTS['far'], 3))
        layout = ChipLayout.from_chip(chip)
        result = chip.sim(3, message_trace=True)
        messages = [m for step in result['message_trace'] for m in step
                    if not m['placeholder']]
        self.assertTrue(messages)
        for message in messages:
            path = xy_path(layout, message['src_tile_id'], message['dest_tile_id'])
            self.assertEqual(len(path) - 1, message['hops'])

    def test_pipeline_roles_axons_and_buffer(self):
        arch = load_loihi2_candidate()
        core = ChipLayout.from_chip(sanafe.SpikingChip(arch), arch).tiles[0].cores[0]
        self.assertEqual(core.pipeline, (('loihi_dense_synapse', 'synapse'),
                                         ('loihi_dendrites', 'dendrite'),
                                         ('loihi_lif', 'soma')))
        self.assertEqual((core.axon_in, core.axon_out), (('loihi_in',), ('loihi_out',)))
        self.assertEqual(core.buffer, 'soma')
        self.assertIsNone(layout_for('loihi2').tiles[0].cores[0].buffer)
        data = ChipLayout.from_chip(sanafe.SpikingChip(arch), arch).to_dict()
        described = data['tiles'][0]['cores'][0]
        self.assertEqual(described['pipeline'][2], {'name': 'loihi_lif', 'role': 'soma'})
        self.assertEqual((described['axon_in'], described['buffer']), (['loihi_in'], 'soma'))

    def test_to_dict_is_json_ready(self):
        data = layout_for('example_chip').to_dict()
        self.assertEqual(json.loads(json.dumps(data)), data)
        self.assertEqual(data['tiles'][0]['cores'][0]['key'], '0.0')


if __name__ == '__main__':
    unittest.main()
