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
