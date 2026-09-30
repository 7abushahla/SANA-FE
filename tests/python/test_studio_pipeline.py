"""Pipeline view support: per-group spike counts and the readout hook."""
import unittest

from sanafe.studio.engine import Session, UpdateRecord

from studio_helpers import ChainWorkload

GROUPS = ('layer_0', 'layer_1', 'layer_2')


def by_group(fired):
    return {g: sum(1 for f in fired if f[0] == g) for g in GROUPS}


class TestGroupFired(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        full = Session(ChainWorkload(), {'placement': 'far'})
        cls.full = full.run_to_horizon()
        full.close()

    def test_full_level_counts_each_group(self):
        for record in self.full:
            self.assertEqual(record.group_fired, by_group(record.fired))
            self.assertEqual(record.provenance['group_fired'], 'D')
            self.assertEqual(UpdateRecord.from_dict(record.to_dict()).group_fired, record.group_fired)
        self.assertTrue(any(sum(r.group_fired.values()) for r in self.full))

    def test_aggregate_level_matches_full(self):
        session = Session(ChainWorkload(), {'placement': 'far'}, trace_level='aggregate')
        self.addCleanup(session.close)
        for full, agg in zip(self.full, session.run_to_horizon()):
            self.assertEqual(agg.group_fired, full.group_fired)

    def test_group_without_spike_logging_is_none(self):
        class Unlogged(ChainWorkload):
            def build(self, params):
                built = super().build(params)
                for neuron in built.network.groups['layer_2']:
                    neuron.set_attributes(log_spikes=False)
                return built

        session = Session(Unlogged(), {'placement': 'far'})
        self.addCleanup(session.close)
        [record] = session.step(1)
        self.assertIsNone(record.group_fired['layer_2'])
        self.assertIsInstance(record.group_fired['layer_0'], int)


if __name__ == '__main__':
    unittest.main()
