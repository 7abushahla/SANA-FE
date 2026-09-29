"""Session stepping, stopping, faults, storage, and stepwise-batch equality."""
import tempfile
import unittest

import numpy as np
import sanafe
from sanafe.loihi2 import architecture_fingerprint, load_loihi2_candidate
from sanafe.studio.engine import (SanafeFiles, Session, SessionFault, SessionState,
                                  TraceStore)

from studio_helpers import PLACEMENTS, REPO, ChainWorkload, chain_network


class TestSession(unittest.TestCase):
    def session(self, **kwargs):
        session = Session(ChainWorkload(), kwargs.pop('parameters', {}), **kwargs)
        self.addCleanup(session.close)
        return session

    def test_stepwise_equals_batch_for_three_placements(self):
        for placement in PLACEMENTS:
            with self.subTest(placement=placement):
                session = self.session(parameters={'placement': placement})
                records = session.run_to_horizon()
                arch = load_loihi2_candidate()
                chip = sanafe.SpikingChip(arch)
                chip.load(chain_network(arch, PLACEMENTS[placement], 6))
                batch = chip.sim(6, spike_trace=True, potential_trace=True,
                                 perf_trace=True)
                np.testing.assert_array_equal(
                    np.array([list(r.potentials.values()) for r in records]),
                    np.asarray(batch['potential_trace']))
                self.assertEqual([r.counts['fired'] for r in records],
                                 [len(s) for s in batch['spike_trace']])
                # Instrumentation adds logging only; totals agree to rounding.
                np.testing.assert_allclose([r.energy['total'] for r in records],
                                           batch['perf_trace']['total_energy'],
                                           rtol=1e-12, atol=0)
                self.assertEqual(session.state, SessionState.FINISHED)

    def test_step_and_run_to_horizon(self):
        session = self.session()
        self.assertEqual(session.state, SessionState.IDLE)
        self.assertEqual(len(session.step(2)), 2)
        self.assertEqual(session.state, SessionState.PAUSED)
        session.step(3)
        self.assertEqual([r.update for r in session.records], [1, 2, 3, 4, 5])
        self.assertEqual(len(session.run_to_horizon()), 1)
        self.assertEqual(session.state, SessionState.FINISHED)

    def test_step_past_horizon(self):
        session = self.session()
        session.run_to_horizon()
        extra = session.step(3)
        self.assertEqual([r.update for r in extra], [7, 8, 9])
        self.assertEqual(session.run_to_horizon(), [])
        self.assertEqual(session.state, SessionState.FINISHED)

    def test_stop_when_and_pause(self):
        session = self.session()
        records = session.run_to_horizon(
            stop_when=lambda r: 'ten fired' if r.counts['fired'] >= 10 else None)
        self.assertEqual(session.state, SessionState.STOPPED)
        self.assertEqual(session.stop_reason, 'ten fired')
        self.assertEqual(records[-1].counts['fired'], 10)

        paused = self.session()
        produced = paused.run_to_horizon(stop_when=lambda r: paused.pause())
        self.assertEqual(len(produced), 1)
        self.assertEqual(paused.state, SessionState.PAUSED)

    def test_fault_then_reset(self):
        session = self.session()
        real_simulate = session._simulate
        calls = []

        def fails_once():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError('range error in core processing')
            return real_simulate()  # bound to the session, so it sees the rebuilt chip

        # pybind11 objects reject attribute assignment, so the session routes
        # every simulation call through its own _simulate method.
        session._simulate = fails_once
        with self.assertRaisesRegex(SessionFault, 'range error'):
            session.step()
        self.assertEqual(session.state, SessionState.FAULTED)
        with self.assertRaisesRegex(SessionFault, r'reset\(\) is required'):
            session.step()
        session.reset()
        self.assertEqual(session.state, SessionState.IDLE)
        self.assertEqual(session.records, [])
        self.assertEqual(len(session.step()), 1)

    def test_post_simulation_error_faults(self):
        raised = []

        class RaisesOnUpdateTwo:
            def check(self, record):
                if record.update == 2 and not raised:
                    raised.append(True)  # once only, so reset() can recover
                    raise KeyError('reference lookup failed')
                return None

        class CheckedChain(ChainWorkload):
            def build(self, params):
                built = super().build(params)
                built.reference = RaisesOnUpdateTwo()
                return built

        session = Session(CheckedChain(), {})
        self.addCleanup(session.close)
        with self.assertRaisesRegex(SessionFault, 'reference lookup failed'):
            session.run_to_horizon()
        # The chip advanced past update 2, so the session must not continue
        # with shifted update numbers.
        self.assertEqual(session.state, SessionState.FAULTED)
        self.assertEqual([r.update for r in session.records], [1])
        with self.assertRaisesRegex(SessionFault, r'reset\(\) is required'):
            session.step()
        session.reset()
        self.assertEqual([r.update for r in session.step(2)], [1, 2])

    def test_pause_before_run_is_honored(self):
        session = self.session()
        session.pause()
        self.assertEqual(session.run_to_horizon(), [])
        self.assertEqual(session.state, SessionState.PAUSED)
        self.assertEqual(len(session.run_to_horizon()), 6)  # the pause was consumed
        self.assertEqual(session.state, SessionState.FINISHED)

    def test_pause_during_last_update_reports_finished(self):
        session = self.session()
        session.step(5)
        produced = session.run_to_horizon(stop_when=lambda r: session.pause())
        self.assertEqual(len(produced), 1)
        self.assertEqual(session.state, SessionState.FINISHED)
        self.assertEqual(len(session.step(1)), 1)  # no stale pause remains

    def test_cancel_pause(self):
        session = self.session()
        session.pause()
        session.cancel_pause()
        self.assertEqual(len(session.step(2)), 2)

    def test_step_accepts_stop_when(self):
        session = self.session()
        produced = session.step(5, stop_when=lambda r: 'two' if r.update == 2 else None)
        self.assertEqual([r.update for r in produced], [1, 2])
        self.assertEqual((session.state, session.stop_reason), (SessionState.STOPPED, 'two'))

    def test_network_summary_badge_and_describe(self):
        session = self.session()
        summary = session.network_summary()
        self.assertEqual(summary['groups'], [
            {'name': 'layer_0', 'size': 8, 'cores': {'0.0': 8}},
            {'name': 'layer_1', 'size': 4, 'cores': {'16.0': 4}},
            {'name': 'layer_2', 'size': 2, 'cores': {'31.0': 2}}])
        self.assertEqual(summary['occupied'], ['0.0', '16.0', '31.0'])
        self.assertEqual(session.badge(),
                         'Loihi 2 candidate · costs inherited from Loihi 1 · not hardware')
        described = session.describe()
        self.assertEqual(set(described), {'layout', 'network', 'horizon', 'badge', 'manifest',
                                          'metadata', 'state', 'update'})
        self.assertEqual((described['state'], described['update'], described['horizon']),
                         ('idle', 0, 6))
        import json
        from sanafe.studio.engine import to_strict_json
        json.dumps(to_strict_json(described), allow_nan=False)

    def test_badge_for_other_architectures(self):
        session = Session(SanafeFiles(), {
            'arch_yaml': str(REPO / 'arch' / 'example_chip.yaml'),
            'net_file': str(REPO / 'snn' / 'example.net')})
        self.addCleanup(session.close)
        self.assertEqual(session.badge(),
                         'example_chip.yaml · modeled costs from this file · not measurements')

    def test_aggregate_trace_level_is_stage_five(self):
        with self.assertRaisesRegex(ValueError, 'stage 5'):
            Session(ChainWorkload(), {}, trace_level='aggregate')

    def test_horizon_override(self):
        session = self.session(horizon=2)
        self.assertEqual(len(session.run_to_horizon()), 2)
        with self.assertRaisesRegex(ValueError, 'horizon: must be at least 1'):
            Session(ChainWorkload(), {}, horizon=0)

    def test_store_manifest_and_records(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self.session(store_dir=directory)
            session.step(2)
            manifest, records = TraceStore.load(session.store.directory)
            self.assertEqual(records, session.records)
            self.assertEqual(manifest['workload'], 'test-chain')
            self.assertEqual(manifest['parameters'], {'placement': 'far', 'steps': 6})
            self.assertEqual(manifest['architecture_sha256'],
                             architecture_fingerprint(load_loihi2_candidate()))
            self.assertNotEqual(manifest['simulated_architecture_sha256'],
                                manifest['architecture_sha256'])
            self.assertEqual(manifest['trace_level'], 'full')
            self.assertEqual(manifest['timing_model'], 'detailed')
            first = session.store.directory
            session.reset()
            self.assertNotEqual(session.store.directory, first)

    def test_sanafe_files_end_to_end(self):
        session = Session(SanafeFiles(), {
            'arch_yaml': str(REPO / 'arch' / 'example_chip.yaml'),
            'net_file': str(REPO / 'snn' / 'example.net'), 'horizon': 4})
        self.addCleanup(session.close)
        records = session.run_to_horizon()
        self.assertEqual(len(records), 4)
        self.assertEqual((session.layout.width, session.layout.height), (2, 1))
        self.assertEqual(len(records[0].potentials), 6)


if __name__ == '__main__':
    unittest.main()
