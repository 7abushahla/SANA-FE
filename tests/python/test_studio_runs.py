"""Saved runs: listing, loading, comparison, and sanafe.viz export."""
import shutil
import tempfile
import unittest
from pathlib import Path

from sanafe.studio.engine import (EXPORT_KINDS, SanafeFiles, Session, compare_runs,
                                  export_plot, list_runs, load_run)

from studio_helpers import REPO, ChainWorkload


class TestRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = Path(tempfile.mkdtemp())
        cls.ids = {}
        for name, workload, parameters in (
                ('far', ChainWorkload(), {'placement': 'far'}),
                ('near', ChainWorkload(), {'placement': 'near'}),
                ('files', SanafeFiles(), {'arch_yaml': str(REPO / 'arch' / 'example_chip.yaml'),
                                          'net_file': str(REPO / 'snn' / 'example.net'),
                                          'horizon': 6})):
            session = Session(workload, parameters, store_dir=cls.store)
            session.run_to_horizon()
            cls.ids[name] = session.store.directory.name
            session.close()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.store)

    def test_list_runs_newest_first(self):
        runs = list_runs(self.store)
        self.assertEqual([r['id'] for r in runs],
                         [self.ids['files'], self.ids['near'], self.ids['far']])
        far = runs[2]
        self.assertEqual((far['workload'], far['updates'], far['horizon']), ('test-chain', 6, 6))
        self.assertEqual(far['parameters']['placement'], 'far')
        self.assertEqual(far['core_map'], {})
        self.assertEqual(far['architecture'], 'loihi2.yaml')
        (self.store / 'not-a-run').mkdir(exist_ok=True)
        self.assertEqual(len(list_runs(self.store)), 3)
        self.assertEqual(list_runs(self.store / 'missing'), [])

    def test_placements_compare_with_identical_spikes(self):
        result = compare_runs(load_run(self.store, self.ids['far']),
                              load_run(self.store, self.ids['near']))
        self.assertTrue(result['comparable'])
        self.assertTrue(result['identical_spikes'])
        self.assertIsNone(result['first_difference'])
        self.assertEqual(len(result['updates']), 6)
        hops = [(u['a']['hops'], u['b']['hops']) for u in result['updates']]
        self.assertTrue(any(a != b for a, b in hops), hops)
        self.assertGreater(result['totals']['a']['hops'], result['totals']['b']['hops'])

    def test_different_networks_are_not_comparable(self):
        result = compare_runs(load_run(self.store, self.ids['far']),
                              load_run(self.store, self.ids['files']))
        self.assertFalse(result['comparable'])
        self.assertFalse(result['identical_spikes'])

    def test_first_difference(self):
        manifest, records = load_run(self.store, self.ids['far'])
        self.assertTrue(records[2].fired)
        group, offset = records[2].fired[0]
        records[2].fired = records[2].fired[1:]
        result = compare_runs((manifest, records), load_run(self.store, self.ids['near']))
        self.assertFalse(result['identical_spikes'])
        self.assertEqual(result['first_difference'],
                         {'update': 3, 'neuron': f'{group}.{offset}', 'in': 'b'})

    def test_empty_runs_are_not_called_identical(self):
        manifest, _ = load_run(self.store, self.ids['far'])
        result = compare_runs((manifest, []), (manifest, []))
        self.assertFalse(result['identical_spikes'])
        self.assertEqual(result['note'], 'no updates to compare')

    def test_export_forces_a_headless_backend(self):
        import os
        import subprocess
        import sys
        script = (
            'import sys, threading; sys.path.insert(0, %r)\n'
            'from sanafe.studio.engine import load_run, export_plot\n'
            '_, records = load_run(%r, %r)\n'
            'out = []\n'
            't = threading.Thread(target=lambda: out.append(export_plot(records, "energy")))\n'
            't.start(); t.join()\n'
            'assert out and out[0].lstrip().startswith((b"<?xml", b"<svg"))\n'
        ) % (str(Path(__file__).parent), str(self.store), self.ids['far'])
        env = dict(os.environ, MPLBACKEND='macosx')
        done = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True,
                              text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr[-1500:])

    def test_aggregate_runs_do_not_claim_spike_identity(self):
        store = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, store)
        ids = []
        for placement in ('far', 'near'):
            session = Session(ChainWorkload(), {'placement': placement}, store_dir=store,
                              trace_level='aggregate')
            session.run_to_horizon()
            ids.append(session.store.directory.name)
            session.close()
        result = compare_runs(load_run(store, ids[0]), load_run(store, ids[1]))
        self.assertFalse(result['identical_spikes'])
        self.assertIsNone(result['first_difference'])
        self.assertIn('aggregate', result['note'])
        self.assertEqual(len(result['updates']), 6)
        with self.assertRaisesRegex(ValueError, 'aggregate'):
            export_plot(load_run(store, ids[0])[1], 'raster', load_run(store, ids[0])[0])
        self.assertTrue(export_plot(load_run(store, ids[0])[1], 'energy').lstrip().startswith(
            (b'<?xml', b'<svg')))

    def test_unsafe_or_unknown_ids(self):
        for bad in ('../x', 'a/b', '', '.', '..'):
            with self.subTest(run_id=bad), self.assertRaises(ValueError):
                load_run(self.store, bad)
        with self.assertRaises(KeyError):
            load_run(self.store, 'no-such-run')

    def test_export_every_kind(self):
        _, records = load_run(self.store, self.ids['far'])
        self.assertEqual(set(EXPORT_KINDS), {'raster', 'potential', 'energy', 'throughput',
                                             'latency'})
        for kind in EXPORT_KINDS:
            with self.subTest(kind=kind):
                svg = export_plot(records, kind)
                self.assertTrue(svg.lstrip().startswith((b'<?xml', b'<svg')), svg[:40])
        with self.assertRaises(ValueError):
            export_plot([], 'raster')
        with self.assertRaises(KeyError):
            export_plot(records, 'weather')


if __name__ == '__main__':
    unittest.main()
