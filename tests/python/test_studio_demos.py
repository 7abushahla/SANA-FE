"""The random-network demo workload runs on every platform."""
import unittest

from sanafe.studio.engine import RandomSNN, Session, resolve_parameters
from sanafe import platforms as P


RUNNABLE = [p for p in P.registry() if not p.preview]  # previews cannot run


class TestRandomSNN(unittest.TestCase):
    def build(self, platform, **overrides):
        workload = RandomSNN()
        params = resolve_parameters(workload, overrides)
        params['platform'] = platform
        return workload.build(params)

    def test_declares_every_platform(self):
        self.assertEqual(RandomSNN.platforms, tuple(p.id for p in P.registry() if not p.preview))
        self.assertNotIn('speck', RandomSNN.platforms)

    def test_runs_with_spikes_in_every_group_on_every_platform(self):
        for platform in RUNNABLE:
            with self.subTest(platform=platform.id):
                session = Session(RandomSNN(), {'groups': 3, 'neurons_per_group': 16, 'seed': 3},
                                  platform=platform.id, store_dir=None)
                try:
                    session.run_to_horizon()
                    fired = {group for record in session.records for group, _ in record.fired}
                    self.assertEqual(fired, {'group_0', 'group_1', 'group_2'})
                    self.assertEqual(session.built.metadata['platform'], platform.id)
                    if platform.id == 'truenorth':
                        self.assertTrue(all(r.step_time == 0 for r in session.records))
                    if platform.id == 'truenorth_documented':
                        self.assertTrue(all(abs(r.step_time - 1e-3) < 1e-12 for r in session.records))
                finally:
                    session.close()

    def test_same_seed_gives_the_same_edges_on_every_platform(self):
        edges = {p.id: self.build(p.id, seed=7).metadata['edges'] for p in RUNNABLE}
        self.assertEqual(len(set(edges.values())), 1)
        self.assertNotEqual(self.build('loihi', seed=8).metadata['edges'], edges['loihi'])

    def test_spread_places_one_group_per_distant_tile(self):
        built = self.build('loihi', placement='spread', groups=3)
        self.assertEqual(len(built.metadata['cores']), 3)
        tiles = [int(core.split('.')[0]) for core in built.metadata['cores']]
        self.assertEqual(tiles, sorted(tiles))
        self.assertGreater(tiles[-1] - tiles[0], 1)

    def test_group_larger_than_a_core_is_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, 'neurons_per_group'):
            self.build('truenorth', neurons_per_group=300)

    def test_packed_shares_a_core_when_groups_fit(self):
        built = self.build('loihi2', neurons_per_group=8, groups=4)
        self.assertEqual(len(built.metadata['cores']), 1)

    def test_spread_on_a_large_mesh_stays_within_eight_columns(self):
        built = self.build('truenorth', placement='spread', groups=8, neurons_per_group=8)
        tiles = [int(core.split('.')[0]) for core in built.metadata['cores']]
        self.assertEqual(len(tiles), 8)
        self.assertEqual(tiles, sorted(set(tiles)))
        self.assertTrue(all(tile % 64 == 0 for tile in tiles), tiles)   # y = 0
        self.assertLessEqual(max(tiles) // 64, 7)                        # x <= 7

    def test_drive_neurons_fire_with_the_same_exact_period_on_every_platform(self):
        periods = {}
        for platform in RUNNABLE:
            session = Session(RandomSNN(), {'groups': 2, 'neurons_per_group': 12, 'seed': 4,
                                            'connection_percent': 0, 'horizon': 45},
                              platform=platform.id, store_dir=None)
            try:
                session.run_to_horizon()
                trains = {}
                for record in session.records:
                    for group, offset in record.fired:
                        if group == 'group_0':
                            trains.setdefault(offset, []).append(record.update)
            finally:
                session.close()
            found = {}
            for offset in range(12):
                train = trains.get(offset, [])
                self.assertTrue(train, (platform.id, offset, 'never fired'))
                period = train[0]
                self.assertTrue(1 <= period <= 20, (platform.id, offset, period))
                self.assertEqual(train, list(range(period, 46, period)), (platform.id, offset))
                found[offset] = period
            periods[platform.id] = found
        self.assertEqual(len({tuple(sorted(v.items())) for v in periods.values()}), 1, periods)
