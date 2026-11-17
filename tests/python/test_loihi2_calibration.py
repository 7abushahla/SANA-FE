"""Static power and the Speck terms for the Loihi 2 route."""
import copy
import unittest
from pathlib import Path
import tempfile

import yaml

from sanafe.loihi2_calibration import load_calibration, report, static_power

PROVENANCE = {'sources': 3, 'rows': ['record:a', 'record:b'], 'note': 'Test rows.'}
EMPTY = {'sources': 0, 'rows': [], 'note': 'Not anchored.'}


def per_chip(watts=0.15, status='fitted'):
    return {'name': 'per_chip', 'watts_per_chip': watts, 'status': status,
            'provenance': copy.deepcopy(PROVENANCE)}


def per_core(watts=0.0008, status='fitted'):
    return {'name': 'per_core', 'watts_per_core': watts, 'status': status,
            'provenance': copy.deepcopy(PROVENANCE)}


def affine(chip=0.1, core=0.0005):
    return {'name': 'affine', 'watts_per_chip': chip, 'watts_per_core': core,
            'status': 'fitted', 'provenance': copy.deepcopy(PROVENANCE)}


def document(rules=None):
    return {
        'schema': 'loihi2-calibration-v1',
        'represents': 'A single chip of unknown revision.',
        'readings': {'neuron_update_ns': 6.27, 'link_bits_per_s': 9.05e9,
                     'alternative': {'neuron_update_ns': 7.4, 'link_bits_per_s': 8.42e9}},
        'static': {'rules': rules if rules is not None else [per_chip()]},
        'time': {'latency_scale': 1.0, 'status': 'inherited', 'provenance': copy.deepcopy(EMPTY)},
        'dynamic': {'energy_scale': 1.0, 'status': 'inherited', 'provenance': copy.deepcopy(EMPTY)},
        'transfer': [],
    }


class Base(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.tmp = Path(self._dir.name)

    def load(self, doc, name='cal.yaml'):
        path = self.tmp / name
        path.write_text(yaml.safe_dump(doc))
        return load_calibration(path)

    def refused(self, doc, fragment=None):
        with self.assertRaises(ValueError) as caught:
            self.load(doc)
        if fragment is not None:
            self.assertIn(fragment, str(caught.exception))


class TestLoading(Base):
    def test_valid_file_loads_and_is_frozen(self):
        cal = self.load(document())
        self.assertEqual(cal.represents, 'A single chip of unknown revision.')
        with self.assertRaises(Exception):
            cal.represents = 'other'

    def test_unknown_schema_refused_naming_value(self):
        doc = document()
        doc['schema'] = 'loihi2-calibration-v9'
        self.refused(doc, 'loihi2-calibration-v9')

    def test_unknown_keys_refused_naming_key(self):
        for where in (None, 'static', 'time', 'dynamic'):
            doc = document()
            (doc if where is None else doc[where])['surprise'] = 1
            self.refused(doc, 'surprise')

    def test_negative_or_nonfinite_coefficient_refused(self):
        for bad in (-0.1, float('inf'), float('nan'), '0.1', True):
            self.refused(document([per_chip(bad)]))

    def test_scale_must_be_positive_finite_number(self):
        for bad in (0.0, -1.0, float('nan'), True):
            doc = document()
            doc['time']['latency_scale'] = bad
            doc['time']['status'] = 'anchored'
            doc['time']['provenance'] = copy.deepcopy(PROVENANCE)
            self.refused(doc)

    def test_rule_name_and_fields(self):
        bad = per_chip()
        bad['name'] = 'per_neuron'
        self.refused(document([bad]), 'per_neuron')
        extra = per_chip()
        extra['watts_per_core'] = 0.1
        self.refused(document([extra]))
        missing = affine()
        del missing['watts_per_core']
        self.refused(document([missing]))
        self.refused(document([per_core(), {**per_chip(), 'name': 'per_core',
                                            'watts_per_core': 0.1}]))
        self.refused(document([]))
        self.refused(document([per_chip(), per_core(), affine()]))

    def test_rule_names_unique(self):
        self.refused(document([per_chip(), per_chip(0.2)]), 'per_chip')

    def test_bad_status_refused(self):
        self.refused(document([per_chip(status='guessed')]), 'guessed')

    def test_fitted_or_anchored_requires_provenance(self):
        for status in ('fitted', 'anchored'):
            for prov in ({'sources': 0, 'rows': ['x:1'], 'note': 'n'},
                         {'sources': 1, 'rows': [], 'note': 'n'},
                         {'sources': 1, 'rows': [''], 'note': 'n'},
                         {'sources': 1, 'rows': [3], 'note': 'n'}):
                rule = per_chip(status=status)
                rule['provenance'] = prov
                self.refused(document([rule]))
        doc = document()
        doc['time']['status'] = 'anchored'
        self.refused(doc)

    def test_scale_other_than_one_requires_anchoring(self):
        doc = document()
        doc['dynamic']['energy_scale'] = 0.5
        self.refused(doc, 'energy_scale')
        doc['dynamic']['status'] = 'anchored'
        doc['dynamic']['provenance'] = copy.deepcopy(PROVENANCE)
        self.load(doc)

    def test_transfer_entries_need_four_fields(self):
        entry = {'coefficient': 'message_in', 'from': 'lif', 'to': ['if'], 'assumption': 'Same.'}
        doc = document()
        doc['transfer'] = [entry]
        self.assertEqual(self.load(doc).transfer[0]['to'], ['if'])
        for key in entry:
            broken = {k: v for k, v in entry.items() if k != key}
            doc['transfer'] = [broken]
            self.refused(doc, key)

    def test_missing_top_level_section_refused(self):
        doc = document()
        del doc['dynamic']
        self.refused(doc, 'dynamic')


class TestStaticPower(Base):
    def test_per_chip_ignores_cores(self):
        cal = self.load(document([per_chip(0.15)]))
        for cores in (0, 1, 74):
            self.assertAlmostEqual(static_power(cal, chips=1, cores=cores), 0.15)
        self.assertAlmostEqual(static_power(cal, chips=3, cores=5), 0.45)

    def test_per_core(self):
        cal = self.load(document([per_core(0.0008)]))
        self.assertAlmostEqual(static_power(cal, chips=1, cores=74), 0.0592)

    def test_affine(self):
        cal = self.load(document([affine(0.1, 0.0005)]))
        self.assertAlmostEqual(static_power(cal, chips=2, cores=10), 0.2 + 0.005)

    def test_several_rules_need_a_name(self):
        cal = self.load(document([per_chip(0.15), per_core(0.0008)]))
        with self.assertRaises(ValueError) as caught:
            static_power(cal, chips=1, cores=74)
        self.assertIn('several rules', str(caught.exception))
        self.assertAlmostEqual(static_power(cal, chips=1, cores=74, rule='per_core'), 0.0592)
        with self.assertRaises(ValueError):
            static_power(cal, chips=1, cores=74, rule='affine')

    def test_counts_validated(self):
        cal = self.load(document())
        for chips, cores in ((0, 1), (-1, 1), (1.5, 1), (True, 1), (1, -1), (1, 2.5), (1, True)):
            with self.assertRaises(ValueError):
                static_power(cal, chips=chips, cores=cores)
        static_power(cal, chips=1, cores=0)


class TestReport(Base):
    def test_brief_numbers(self):
        cal = self.load(document([per_chip(0.15)]))
        out = report(cal, sim_time_s=2e-3, dynamic_energy_j=1e-3, chips=1, cores=74)
        self.assertAlmostEqual(out['inference_time_s'], 2e-3)
        self.assertAlmostEqual(out['power_dynamic_w'], 0.5)
        self.assertAlmostEqual(out['power_static_w'], 0.15)
        self.assertAlmostEqual(out['power_total_w'], 0.65)
        self.assertAlmostEqual(out['energy_inference_j'], 1.3e-3)
        self.assertAlmostEqual(out['energy_total_j'], 1.3e-3)
        self.assertAlmostEqual(out['energy_dynamic_j'], 1e-3)
        self.assertEqual((out['samples'], out['chips'], out['cores'], out['host_time_s']),
                         (1, 1, 74, 0.0))

    def test_scales_and_samples(self):
        doc = document([per_chip(0.15)])
        doc['time'] = {'latency_scale': 2.0, 'status': 'anchored',
                       'provenance': copy.deepcopy(PROVENANCE)}
        doc['dynamic'] = {'energy_scale': 0.5, 'status': 'anchored',
                          'provenance': copy.deepcopy(PROVENANCE)}
        cal = self.load(doc)
        out = report(cal, sim_time_s=2e-3, dynamic_energy_j=1e-3, chips=1, cores=1)
        self.assertAlmostEqual(out['inference_time_s'], 4e-3)
        self.assertAlmostEqual(out['energy_dynamic_j'], 0.5e-3)
        self.assertAlmostEqual(out['power_dynamic_w'], 0.125)
        self.assertAlmostEqual(out['power_total_w'], 0.275)
        self.assertAlmostEqual(out['energy_inference_j'], 0.275 * 4e-3)
        self.assertEqual(out['scales']['time'], {'value': 2.0, 'status': 'anchored'})
        self.assertEqual(out['scales']['dynamic'], {'value': 0.5, 'status': 'anchored'})
        many = report(cal, sim_time_s=2e-3 * 10, dynamic_energy_j=1e-3 * 10, chips=1, cores=1,
                      samples=10)
        self.assertAlmostEqual(many['inference_time_s'], 4e-3)
        self.assertAlmostEqual(many['energy_dynamic_j'], 0.5e-3)
        self.assertAlmostEqual(many['power_total_w'], 0.275)

    def test_host_time_adds_static_energy(self):
        cal = self.load(document([per_chip(0.15)]))
        out = report(cal, sim_time_s=2e-3, dynamic_energy_j=1e-3, chips=2, cores=4,
                     host_time_s=0.1)
        self.assertAlmostEqual(out['power_static_w'], 0.3)
        self.assertAlmostEqual(out['energy_inference_j'], (0.3 + 0.5) * 2e-3)
        self.assertAlmostEqual(out['energy_total_j'], (0.3 + 0.5) * 2e-3 + 0.3 * 0.1)

    def test_one_rule_appears_at_top_level_and_under_by_rule(self):
        cal = self.load(document([per_chip(0.15)]))
        out = report(cal, sim_time_s=2e-3, dynamic_energy_j=1e-3, chips=1, cores=74)
        self.assertEqual(list(out['by_rule']), ['per_chip'])
        for key in ('power_static_w', 'power_total_w', 'energy_inference_j', 'energy_total_j'):
            self.assertEqual(out[key], out['by_rule']['per_chip'][key])

    def test_two_rules_appear_only_under_by_rule(self):
        cal = self.load(document([per_chip(0.15), per_core(0.0008)]))
        out = report(cal, sim_time_s=2e-3, dynamic_energy_j=1e-3, chips=1, cores=74)
        for key in ('power_static_w', 'power_total_w', 'energy_inference_j', 'energy_total_j'):
            self.assertNotIn(key, out)
        self.assertAlmostEqual(out['by_rule']['per_chip']['power_total_w'], 0.65)
        self.assertAlmostEqual(out['by_rule']['per_core']['power_static_w'], 0.0592)
        self.assertAlmostEqual(out['by_rule']['per_core']['power_total_w'], 0.5592)
        self.assertNotEqual(out['by_rule']['per_chip']['energy_inference_j'],
                            out['by_rule']['per_core']['energy_inference_j'])
        self.assertAlmostEqual(out['inference_time_s'], 2e-3)

    def test_report_states_what_it_rests_on(self):
        doc = document()
        doc['transfer'] = [{'coefficient': 'message_in', 'from': 'lif', 'to': ['if'],
                            'assumption': 'Same cost.'}]
        out = report(self.load(doc), sim_time_s=1e-3, dynamic_energy_j=0.0, chips=1, cores=1)
        self.assertEqual(out['represents'], 'A single chip of unknown revision.')
        self.assertEqual(out['readings']['neuron_update_ns'], 6.27)
        self.assertEqual(out['readings']['alternative']['link_bits_per_s'], 8.42e9)
        self.assertEqual(out['transfer'][0]['coefficient'], 'message_in')
        self.assertEqual(out['scales']['time']['status'], 'inherited')
        self.assertEqual(out['power_dynamic_w'], 0.0)

    def test_inputs_validated(self):
        cal = self.load(document())
        good = dict(sim_time_s=1e-3, dynamic_energy_j=1e-3, chips=1, cores=1)
        report(cal, **good)
        for key, bad in (('sim_time_s', 0.0), ('sim_time_s', -1.0), ('sim_time_s', float('nan')),
                         ('sim_time_s', float('inf')), ('dynamic_energy_j', -1e-3),
                         ('dynamic_energy_j', float('nan')), ('host_time_s', -1.0),
                         ('host_time_s', float('inf')), ('samples', 0), ('samples', 1.5),
                         ('samples', True), ('chips', 0), ('cores', -1)):
            with self.assertRaises(ValueError, msg=key):
                report(cal, **{**good, key: bad})


if __name__ == '__main__':
    unittest.main()
