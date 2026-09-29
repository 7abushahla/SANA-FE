"""Workload parameters, the generic file workload, and JSON Lines storage."""
from pathlib import Path
import tempfile
import unittest

from sanafe.studio.engine import (TRACE_FORMAT, ParameterSpec, SanafeFiles,
                                  TraceStore, UpdateRecord, resolve_parameters)

from studio_helpers import REPO


class TestParameters(unittest.TestCase):
    def test_rejects_non_integer_and_bool(self):
        spec = ParameterSpec('T', 'int', minimum=1)
        self.assertEqual(spec.validate(3), 3)
        for bad in (4.0, True, '4', None):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ValueError, '^T: expected an integer'):
                    spec.validate(bad)
        with self.assertRaisesRegex(ValueError, '^T: must be at least 1, got 0'):
            spec.validate(0)

    def test_missing_file_named(self):
        with self.assertRaisesRegex(ValueError, '^net_file: file not found: .*nope.net'):
            ParameterSpec('net_file', 'path').validate('/tmp/nope.net')

    def test_choice(self):
        spec = ParameterSpec('placement', 'choice', choices=('near', 'far'))
        self.assertEqual(spec.validate('far'), 'far')
        with self.assertRaisesRegex(ValueError, 'expected one of near, far'):
            spec.validate('middle')

    def test_resolve_defaults_unknown_and_missing(self):
        workload = SanafeFiles()
        with self.assertRaisesRegex(ValueError, "unknown parameter 'Tee'"):
            resolve_parameters(workload, {'Tee': 3})
        with self.assertRaisesRegex(ValueError, 'arch_yaml: required'):
            resolve_parameters(workload, {})
        params = resolve_parameters(workload, {
            'arch_yaml': str(REPO / 'arch' / 'example_chip.yaml'),
            'net_file': str(REPO / 'snn' / 'example.net')})
        self.assertEqual(params['horizon'], 10)
        self.assertIsInstance(params['arch_yaml'], Path)


class TestSanafeFiles(unittest.TestCase):
    def test_builds_yaml_and_netlist_networks(self):
        for net_file in ('example_snn.yaml', 'example.net'):
            with self.subTest(net_file=net_file):
                workload = SanafeFiles()
                built = workload.build(resolve_parameters(workload, {
                    'arch_yaml': str(REPO / 'arch' / 'example_chip.yaml'),
                    'net_file': str(REPO / 'snn' / net_file), 'horizon': 3}))
                self.assertEqual(built.horizon, 3)
                self.assertEqual(built.arch_yaml.name, 'example_chip.yaml')
                self.assertEqual(built.metadata['network'], net_file)
                self.assertIsNone(built.reference)


class TestTraceStore(unittest.TestCase):
    def record(self, update):
        return UpdateRecord(update=update, step_time=1e-6,
                            energy={'total': 2e-9}, counts={'fired': 1},
                            messages=[], core_finish={'0.0': 5e-7},
                            last_activity=5e-7, barrier=5e-7, fired=[('a', 0)],
                            potentials={'a.0': 1.0}, core_counts={},
                            core_energy={}, tile_network_energy={0: 0.0})

    def test_round_trip_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'run'
            store = TraceStore.create(run, {'workload': 'test'})
            store.append(self.record(1))
            store.append(self.record(2))
            manifest, records = TraceStore.load(run)
            self.assertEqual(manifest['format'], TRACE_FORMAT)
            self.assertEqual(manifest['workload'], 'test')
            self.assertEqual(records, [self.record(1), self.record(2)])
            with self.assertRaises(FileExistsError):
                TraceStore.create(run, {'workload': 'test'})

    def test_rejects_non_finite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TraceStore.create(Path(directory) / 'run', {})
            bad = self.record(1)
            bad.step_time = float('inf')
            with self.assertRaises(ValueError):
                store.append(bad)


if __name__ == '__main__':
    unittest.main()
