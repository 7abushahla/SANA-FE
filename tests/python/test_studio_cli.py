"""Workload registry parsing and the module entry point."""
from pathlib import Path
import subprocess
import sys
import unittest

from sanafe.studio.server.cli import DEFAULT_WORKLOADS, build_registry
from sanafe.studio.server.worker import WorkloadRef

TESTS = Path(__file__).resolve().parent


class TestCli(unittest.TestCase):
    def test_registry_defaults_and_additions(self):
        registry = build_registry(['test-chain=studio_helpers:ChainWorkload'], [str(TESTS)])
        self.assertEqual(registry['sanafe-files'], DEFAULT_WORKLOADS['sanafe-files'])
        self.assertEqual(registry['test-chain'],
                         WorkloadRef('studio_helpers:ChainWorkload', (str(TESTS),)))
        extra = {'x': WorkloadRef('m:C')}
        self.assertEqual(build_registry([], [], extra)['x'], extra['x'])

    def test_registry_rejects_malformed_workloads(self):
        for bad in ('noequals', '=m:C', 'name=nocolon'):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, 'NAME=module:Class'):
                    build_registry([bad], [])

    def test_module_entry_point_help(self):
        result = subprocess.run([sys.executable, '-m', 'sanafe.studio', '--help'],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--workload', result.stdout)
        self.assertIn('.sanafe-studio', result.stdout)  # runs are saved by default
        self.assertIn('--no-store', result.stdout)


if __name__ == '__main__':
    unittest.main()
