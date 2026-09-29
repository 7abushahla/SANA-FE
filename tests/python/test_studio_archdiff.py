"""Architecture YAML differences for the Architecture panel."""
from importlib.resources import files
import unittest

from sanafe.studio.engine import architecture_diff, bundled_architectures


def example(name):
    return str(files('sanafe.examples') / name)


class TestArchitectureDiff(unittest.TestCase):
    def test_candidate_against_loihi(self):
        diff = architecture_diff(example('loihi2.yaml'), example('loihi.yaml'))
        self.assertEqual((diff['loaded'], diff['baseline']), ('loihi2.yaml', 'loihi.yaml'))
        rows = {row['path']: row for row in diff['rows']}
        self.assertEqual(rows['architecture.name'],
                         {'path': 'architecture.name', 'baseline': 'loihi_chip',
                          'loaded': 'loihi2_candidate', 'change': 'changed'})
        core = 'architecture.tile[loihi_tile[0..31]].core[loihi_core[0..3]]'
        self.assertEqual(rows[core + '.attributes.max_neurons_supported']['loaded'], 8192)
        somas = [row for path, row in rows.items()
                 if path.startswith(core + '.soma[') and path.endswith('.attributes.model')]
        self.assertIn('integrate_fire_int24', [row['loaded'] for row in somas])
        removed = rows[core + '.soma[loihi_inputs[0..1023]]']
        self.assertEqual((removed['change'], removed['loaded']), ('removed', None))
        self.assertEqual(removed['baseline']['attributes']['model'], 'input')
        self.assertEqual(len([r for r in diff['rows'] if r['path'].startswith(
            core + '.soma[loihi_inputs')]), 1)

    def test_identical_files_have_no_rows(self):
        self.assertEqual(architecture_diff(example('loihi.yaml'), example('loihi.yaml'))['rows'], [])

    def test_bundled_architectures(self):
        bundled = bundled_architectures()
        self.assertTrue({'loihi', 'loihi2', 'truenorth', 'example_chip'} <= set(bundled))
        self.assertNotIn('example_snn', bundled)


if __name__ == '__main__':
    unittest.main()
