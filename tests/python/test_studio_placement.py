"""Re-mapping whole cores before a session loads its network."""
import unittest

from sanafe.studio.engine import Session

from studio_helpers import ChainWorkload


def spikes(records):
    return [sorted(map(tuple, r.fired)) for r in records]


class TestCoreMap(unittest.TestCase):
    def session(self, **kwargs):
        session = Session(ChainWorkload(), {'placement': 'far'}, **kwargs)
        self.addCleanup(session.close)
        return session

    def test_swap_moves_neurons_and_keeps_spikes(self):
        plain = self.session()
        swapped = self.session(core_map={'0.0': '5.1', '5.1': '0.0'})
        groups = {g['name']: g['cores'] for g in swapped.network_summary()['groups']}
        self.assertEqual(groups['layer_0'], {'5.1': 8})
        self.assertEqual(swapped.describe()['core_map'], {'0.0': '5.1', '5.1': '0.0'})
        self.assertEqual(swapped.manifest()['core_map'], {'0.0': '5.1', '5.1': '0.0'})
        self.assertEqual(spikes(plain.run_to_horizon()), spikes(swapped.run_to_horizon()))
        self.assertEqual([n.offset for n in swapped.neurons if n.group == 'layer_0'],
                         list(range(8)))

    def test_merge_places_both_groups_on_one_core(self):
        merged = self.session(core_map={'16.0': '0.0'})
        self.assertEqual(merged.network_summary()['core_neurons']['0.0'],
                         [['layer_0', 0, 7], ['layer_1', 0, 3]])
        self.assertEqual(len(merged.run_to_horizon()), 6)

    def test_unknown_core_is_rejected(self):
        for core_map in ({'0.0': '99.0'}, {'x': '0.0'}, {'0.0': 5}):
            with self.subTest(core_map=core_map), self.assertRaisesRegex(ValueError, 'core_map'):
                self.session(core_map=core_map)


if __name__ == '__main__':
    unittest.main()
