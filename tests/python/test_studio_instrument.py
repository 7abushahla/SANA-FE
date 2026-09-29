"""Instrumented architecture copies differ only in energy logging flags."""
from pathlib import Path
import tempfile
import unittest

import sanafe
from sanafe.studio.engine import MAX_LOGGED_UNITS_PER_CORE, instrument_arch_yaml

REPO = Path(__file__).resolve().parents[2]


def without_log_flags(value):
    if isinstance(value, dict):
        return {k: without_log_flags(v) for k, v in value.items()
                if k not in ('log_energy', 'log_latency')}
    if isinstance(value, list):
        return [without_log_flags(v) for v in value]
    return value


class TestInstrumentArchYaml(unittest.TestCase):
    def instrument(self, name):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        destination = Path(directory.name) / f'{name}.yaml'
        summary = instrument_arch_yaml(REPO / 'arch' / f'{name}.yaml', destination)
        original = sanafe.load_arch(str(REPO / 'arch' / f'{name}.yaml'))
        instrumented = sanafe.load_arch(str(destination))
        return summary, original.configuration(), instrumented.configuration()

    def test_configuration_unchanged_except_flags(self):
        for name in ('example_chip', 'loihi', 'loihi2', 'truenorth'):
            with self.subTest(name=name):
                _, before, after = self.instrument(name)
                self.assertEqual(without_log_flags(before), without_log_flags(after))
                self.assertTrue(all(tile['log_energy'] for tile in after['tiles']))
                self.assertTrue(all(core['log_energy'] for tile in after['tiles']
                                    for core in tile['cores']))

    def test_candidate_units_logged(self):
        summary, _, after = self.instrument('loihi2')
        self.assertTrue(summary['units'])
        flags = [unit['log_energy'] for unit in after['tiles'][0]['cores'][0]['pipeline']]
        self.assertEqual(flags, [True, True, True])

    def test_loihi1_units_not_logged(self):
        summary, _, after = self.instrument('loihi')
        self.assertFalse(summary['units'])
        self.assertGreater(len(after['tiles'][0]['cores'][0]['pipeline']),
                           MAX_LOGGED_UNITS_PER_CORE)
        self.assertFalse(any(unit['log_energy'] for tile in after['tiles']
                             for core in tile['cores'] for unit in core['pipeline']))

    def test_rejects_non_architecture_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                instrument_arch_yaml(REPO / 'snn' / 'example_snn.yaml',
                                     Path(directory) / 'out.yaml')


if __name__ == '__main__':
    unittest.main()
