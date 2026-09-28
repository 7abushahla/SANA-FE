"""Regression checks for the optional BookSim network timing model."""

import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import sanafe


class TestBookSimLifecycle(unittest.TestCase):
    def test_two_cycle_timesteps_complete(self):
        # An assertion in BookSim terminates the process, so isolate the run.
        script = (
            "import sanafe\n"
            "arch, net = sanafe.load_example()\n"
            "chip = sanafe.SpikingChip(arch)\n"
            "chip.load(net)\n"
            "result = chip.sim(2, timing_model='cycle')\n"
            "assert result['timesteps_executed'] == 2\n"
        )
        run = subprocess.run([sys.executable, "-c", script],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr[-2000:])


class TestBookSimConfiguration(unittest.TestCase):
    def test_loaded_architecture_exposes_effective_booksim_settings(self):
        source = (files('sanafe.examples') / 'example_chip.yaml').read_text()
        custom = source.replace('    height: 1\n',
                                '    height: 1\n'
                                '    booksim:\n'
                                '      subnets: 1\n'
                                '      packet_size: 3\n'
                                '      clock_period: 2.5e-9\n'
                                '      num_vcs: 2\n'
                                '      vc_buf_size: 12\n'
                                '      use_noc_latency: true\n', 1)
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'chip.yaml'
            path.write_text(custom)
            actual = sanafe.load_arch(path).configuration()['booksim']
        self.assertEqual(actual['topology'], 'cmesh')
        self.assertEqual(actual['x'], 2)
        self.assertEqual(actual['y'], 1)
        self.assertEqual(actual['c'], 4)
        self.assertEqual(actual['subnets'], 1)
        self.assertEqual(actual['packet_size_flits'], 3)
        self.assertEqual(actual['clock_period_seconds'], 2.5e-9)
        self.assertEqual(actual['num_vcs'], 2)
        self.assertEqual(actual['vc_buf_size_flits'], 12)
        self.assertTrue(actual['use_noc_latency'])


if __name__ == '__main__':
    unittest.main()
