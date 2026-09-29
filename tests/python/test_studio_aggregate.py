"""The aggregate trace level: counts per core and link, watched neurons only."""
import json
import unittest

import numpy as np
from sanafe.studio.engine import Session, to_strict_json

from studio_helpers import ChainWorkload


class TestAggregate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        full = Session(ChainWorkload(), {'placement': 'far'})
        cls.full = full.run_to_horizon()
        full.close()

    def session(self, **kwargs):
        session = Session(ChainWorkload(), {'placement': 'far'}, trace_level='aggregate', **kwargs)
        self.addCleanup(session.close)
        return session

    def test_counts_match_the_full_level(self):
        session = self.session()
        records = session.run_to_horizon()
        for full, agg in zip(self.full, records):
            self.assertEqual(agg.messages, [])
            self.assertEqual(agg.provenance['messages'], 'not kept (aggregate)')
            for field in ('core_counts', 'core_finish', 'barrier', 'last_activity', 'counts',
                          'step_time', 'energy', 'core_energy'):
                self.assertEqual(getattr(agg, field), getattr(full, field), field)
            self.assertEqual(sum(agg.links.values()), sum(m.hops for m in full.messages))
            self.assertEqual(agg.provenance['links'], 'X')
        self.assertEqual(session.describe()['trace_level'], 'aggregate')

    def test_only_watched_neurons_are_shipped(self):
        session = self.session()
        session.watch(['layer_2.0', 'layer_1.3'])
        records = session.run_to_horizon()
        for full, agg in zip(self.full, records):
            self.assertEqual(set(agg.potentials), {'layer_2.0', 'layer_1.3'})
            self.assertEqual(agg.potentials['layer_1.3'], full.potentials['layer_1.3'])
            expected = {tuple(f) for f in full.fired} & {('layer_2', 0), ('layer_1', 3)}
            self.assertEqual({tuple(f) for f in agg.fired}, expected)
        agg_size = len(json.dumps(to_strict_json(records[2].to_dict())))
        full_size = len(json.dumps(to_strict_json(self.full[2].to_dict())))
        self.assertLess(agg_size, full_size)

    def test_core_state_at_a_past_update(self):
        session = self.session()
        session.step(4)
        state = session.core_state('16.0', 2)
        self.assertEqual(state['neurons'], [f'layer_1.{i}' for i in range(4)])
        self.assertEqual(state['potentials'],
                         [self.full[1].potentials[k] for k in state['neurons']])
        self.assertEqual(state['fired'], sorted(f'{g}.{o}' for g, o in self.full[1].fired
                                                if g == 'layer_1'))
        for bad in (('16.0', 5), ('16.0', 0), ('99.0', 1)):
            with self.subTest(bad=bad), self.assertRaises((KeyError, ValueError)):
                session.core_state(*bad)

    def test_history_backfills_a_late_watch(self):
        session = self.session()
        session.run_to_horizon()
        history = session.neuron_detail('layer_2.1'.split('.')[0], 1)['history']
        self.assertEqual(history['potential'], [r.potentials['layer_2.1'] for r in self.full])
        self.assertEqual(history['fired'],
                         [('layer_2', 1) in {tuple(f) for f in r.fired} for r in self.full])

    def test_watch_rejects_unlogged_neurons(self):
        class Unlogged(ChainWorkload):
            def build(self, params):
                built = super().build(params)
                for neuron in built.network.groups['layer_2']:
                    neuron.set_attributes(log_potential=False)
                return built

        session = Session(Unlogged(), {}, trace_level='aggregate')
        self.addCleanup(session.close)
        with self.assertRaisesRegex(ValueError, 'does not log its potential'):
            session.watch(['layer_2.0'])
        with self.assertRaisesRegex(ValueError, 'no neuron'):
            session.watch(['layer_7.0'])

    def test_workload_default_trace_level(self):
        class Aggregated(ChainWorkload):
            default_trace_level = 'aggregate'

        session = Session(Aggregated(), {})
        self.addCleanup(session.close)
        self.assertEqual(session.trace_level, 'aggregate')
        with self.assertRaisesRegex(ValueError, 'trace level'):
            Session(ChainWorkload(), {}, trace_level='sparse')

    def test_checkers_bind_and_read_full_state(self):
        seen = {}

        class Checker:
            def bind(self, session):
                seen['lookup'] = session.lookup

            def check(self, record):
                state = record.state
                position = seen['lookup'].logged_pos['layer_0.3']
                seen.setdefault('values', []).append(float(state['potentials'][position]))
                return {'status': 'match', 'references': ['fake']}

        class Checked(ChainWorkload):
            def build(self, params):
                built = super().build(params)
                built.reference = Checker()
                return built

        session = Session(Checked(), {'placement': 'far'}, trace_level='aggregate')
        self.addCleanup(session.close)
        records = session.run_to_horizon()
        self.assertEqual(seen['values'], [r.potentials['layer_0.3'] for r in self.full])
        self.assertFalse(hasattr(records[0], 'state'))

    def test_full_level_core_state_and_history(self):
        session = Session(ChainWorkload(), {'placement': 'far'})
        self.addCleanup(session.close)
        session.step(3)
        self.assertEqual(session.core_state('31.0', 3)['potentials'],
                         [self.full[2].potentials[f'layer_2.{i}'] for i in range(2)])
        self.assertEqual(len(session.neuron_detail('layer_0', 0)['history']['potential']), 3)
        np.testing.assert_array_equal(
            session.neuron_detail('layer_0', 0)['history']['potential'],
            [r.potentials['layer_0.0'] for r in self.full[:3]])


if __name__ == '__main__':
    unittest.main()
