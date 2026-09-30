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


from sanafe.studio.engine import SessionFault


class CountingReadout:
    """Test readout: layer_2 spike counts per update, accumulated."""

    def __init__(self, raise_at=None):
        self.calls, self.states, self.raise_at, self.total = [], [], raise_at, None

    def bind(self, session):
        self.lookup = session.lookup

    def decode(self, record):
        if record.update == self.raise_at:
            raise RuntimeError('readout failed')
        self.calls.append(record.update)
        self.states.append(getattr(record, 'state', None) is not None)
        count = record.group_fired['layer_2']
        self.total = (self.total or 0) + count
        return {'step': record.update - 1, 'scores': [count], 'cumulative': [self.total],
                'predicted': 0, 'reference': {'status': 'waiting', 'max_abs_diff': None},
                'quantity': 'spike counts'}


def with_readout(readout, **kwargs):
    class Read(ChainWorkload):
        def build(self, params):
            built = super().build(params)
            built.readout = readout()
            return built
    return Read(**kwargs)


class TestReadout(unittest.TestCase):
    def test_called_once_per_update_at_both_levels(self):
        for level, has_state in (('full', False), ('aggregate', True)):
            with self.subTest(level=level):
                made = []
                session = Session(with_readout(lambda: made.append(CountingReadout()) or made[-1]),
                                  {'placement': 'far'}, trace_level=level)
                self.addCleanup(session.close)
                records = session.run_to_horizon()
                readout = made[-1]
                self.assertEqual(readout.calls, [r.update for r in records])
                self.assertEqual(set(readout.states), {has_state})
                self.assertEqual(records[-1].readout['cumulative'],
                                 [sum(r.group_fired['layer_2'] for r in records)])
                self.assertEqual(UpdateRecord.from_dict(records[-1].to_dict()).readout,
                                 records[-1].readout)

    def test_reset_starts_the_readout_again(self):
        session = Session(with_readout(CountingReadout), {'placement': 'far'})
        self.addCleanup(session.close)
        first = session.run_to_horizon()[-1].readout['cumulative']
        session.reset()
        self.assertEqual(session.run_to_horizon()[-1].readout['cumulative'], first)

    def test_a_readout_error_faults_the_session(self):
        session = Session(with_readout(lambda: CountingReadout(raise_at=2)), {'placement': 'far'})
        self.addCleanup(session.close)
        with self.assertRaises(SessionFault):
            session.step(3)
        self.assertEqual(session.state.value, 'faulted')

    def test_pipeline_metadata_reaches_describe(self):
        class Declared(ChainWorkload):
            def build(self, params):
                built = super().build(params)
                built.metadata['pipeline'] = {'T': 2, 'rows': [{'group': 'layer_0', 'depth': 0, 'window': [0, 2]}]}
                return built
        session = Session(Declared(), {'placement': 'far'})
        self.addCleanup(session.close)
        self.assertEqual(session.describe()['metadata']['pipeline']['rows'][0]['group'], 'layer_0')


if __name__ == '__main__':
    unittest.main()
