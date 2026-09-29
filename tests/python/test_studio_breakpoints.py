"""Declarative breakpoints stop a session at the correct update."""
import threading
import time
import unittest

from sanafe.studio.engine import Session, SessionState

from studio_helpers import ChainWorkload, SlowFirstUpdate


class TestBreakpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        free = Session(ChainWorkload(), {'steps': 12})
        cls.free = free.run_to_horizon()
        free.close()

    def session(self, workload=None, **parameters):
        session = Session(workload or ChainWorkload(), {'steps': 12, **parameters})
        self.addCleanup(session.close)
        return session

    def first(self, predicate):
        return next(r.update for r in self.free if predicate(r))

    def stops_at(self, specs, expected, reason):
        session = self.session()
        session.set_breakpoints(specs)
        produced = session.run_to_horizon()
        self.assertEqual(session.state, SessionState.STOPPED)
        self.assertEqual((produced[-1].update, session.update), (expected, expected))
        self.assertRegex(session.stop_reason, reason)

    def test_neuron_fires(self):
        expected = self.first(lambda r: ['layer_2', 0] in r.fired or ('layer_2', 0) in r.fired)
        self.stops_at([{'id': 'b1', 'kind': 'neuron_fires', 'neuron': 'layer_2.0'}],
                      expected, r'^breakpoint b1: layer_2\.0 fires$')

    def test_core_sends(self):
        expected = self.first(lambda r: r.core_counts['0.0']['packets_out'] > 3)
        self.stops_at([{'id': 'b2', 'kind': 'core_sends', 'core': '0.0', 'more_than': 3}],
                      expected, r'core 0\.0 sends more than 3 packets')

    def test_step_time(self):
        limit = min(r.step_time for r in self.free)
        expected = self.first(lambda r: r.step_time > limit)
        self.stops_at([{'id': 'b3', 'kind': 'step_time', 'more_than': limit}],
                      expected, 'step time exceeds')

    def test_update_equals(self):
        self.stops_at([{'id': 'b4', 'kind': 'update', 'equals': 4}], 4, 'update 4')

    def test_disabled_breakpoints_do_not_stop(self):
        session = self.session()
        session.set_breakpoints([{'id': 'b4', 'kind': 'update', 'equals': 4, 'enabled': False}])
        self.assertEqual(len(session.run_to_horizon()), 12)
        self.assertEqual(session.state, SessionState.FINISHED)
        self.assertEqual(session.describe()['breakpoints'][0]['enabled'], False)

    def test_invalid_breakpoints_are_rejected(self):
        session = self.session()
        cases = [({'id': 'x', 'kind': 'weather'}, 'kind'),
                 ({'id': 'x', 'kind': 'neuron_fires', 'neuron': 'layer_9.0'}, 'no neuron'),
                 ({'id': 'x', 'kind': 'core_sends', 'core': '99.0', 'more_than': 1}, 'no core'),
                 ({'id': 'x', 'kind': 'core_sends', 'core': '0.0', 'more_than': -1}, 'more_than'),
                 ({'id': 'x', 'kind': 'update', 'equals': 0}, 'equals'),
                 ({'kind': 'update', 'equals': 2}, 'id'),
                 ('update 2', 'object')]
        for spec, message in cases:
            with self.subTest(spec=spec), self.assertRaisesRegex(ValueError, message):
                session.set_breakpoints([spec])
        self.assertEqual(session.describe()['breakpoints'], [])

    def test_neuron_without_spike_logging_is_rejected(self):
        class Unlogged(ChainWorkload):
            def build(self, params):
                built = super().build(params)
                for neuron in built.network.groups['layer_2']:
                    neuron.set_attributes(log_spikes=False)
                return built

        session = self.session(Unlogged())
        with self.assertRaisesRegex(ValueError, 'does not log spikes'):
            session.set_breakpoints([{'id': 'b', 'kind': 'neuron_fires', 'neuron': 'layer_2.0'}])

    def test_replaced_during_a_run_takes_effect_next_update(self):
        session = self.session(SlowFirstUpdate())
        runner = threading.Thread(target=session.run_to_horizon)
        runner.start()
        time.sleep(1.0)  # update 1 sleeps for 3 s in its reference check
        session.set_breakpoints([{'id': 'late', 'kind': 'update', 'equals': 2}])
        runner.join(timeout=30)
        self.assertEqual((session.state, session.update), (SessionState.STOPPED, 2))

    def test_breakpoints_survive_reset(self):
        session = self.session()
        session.set_breakpoints([{'id': 'b4', 'kind': 'update', 'equals': 4}])
        session.reset()
        session.run_to_horizon()
        self.assertEqual(session.update, 4)


if __name__ == '__main__':
    unittest.main()
