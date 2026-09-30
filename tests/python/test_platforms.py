"""The platform catalog: curated facts, structural facts read from the YAML, provenance."""
from importlib.resources import files
from pathlib import Path
import unittest

import yaml
import sanafe


def example(name):
    return files('sanafe.examples') / name


class TestDocumentedTrueNorth(unittest.TestCase):
    def test_differs_from_shipped_file_in_exactly_two_attributes(self):
        shipped = yaml.safe_load(example('truenorth.yaml').read_text())['architecture']
        documented = yaml.safe_load(example('truenorth_documented.yaml').read_text())['architecture']
        self.assertEqual(documented['attributes']['sync_model'], 'fixed')
        self.assertEqual(documented['attributes']['latency_sync'], 1.0e-3)
        synapse = documented['tile'][0]['core'][0]['synapse'][0]['attributes']
        self.assertEqual(synapse['energy_process_spike'], 26.0e-12)
        # Everything else is the shipped file.
        documented['attributes']['latency_sync'] = shipped['attributes']['latency_sync']
        synapse['energy_process_spike'] = 0.0
        documented['name'] = shipped['name']
        self.assertEqual(documented, shipped)

    def test_packaged_copy_matches_repository_copy_and_loads(self):
        repo = Path(__file__).resolve().parents[2] / 'arch' / 'truenorth_documented.yaml'
        self.assertEqual(repo.read_text(), example('truenorth_documented.yaml').read_text())
        arch = sanafe.load_arch(str(repo))
        # A fixed sync model is a one-entry table; timestep_delay is a separate field.
        self.assertEqual(list(arch.configuration()['sync_table'].values()), [1.0e-3])
