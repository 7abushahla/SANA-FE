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

    def test_no_bracket_leaves_the_report_unchanged(self):
        cal = self.load(document([per_chip(0.15)]))
        self.assertIsNone(cal.dynamic.low)
        self.assertIsNone(cal.dynamic.high)
        out = report(cal, sim_time_s=2e-3, dynamic_energy_j=1e-3, chips=1, cores=74)
        self.assertNotIn('by_energy_scale', out)
        self.assertEqual(out['scales']['dynamic'], {'value': 1.0, 'status': 'inherited'})

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


def bracketed(low=0.4, carried=0.6, high=0.7, rules=None):
    doc = document(rules or [per_chip(0.15)])
    doc['dynamic'] = {'energy_scale': carried, 'energy_scale_low': low, 'energy_scale_high': high,
                      'status': 'anchored', 'provenance': copy.deepcopy(PROVENANCE)}
    return doc


class TestEnergyScaleBracket(Base):
    def test_bracket_loads(self):
        cal = self.load(bracketed())
        self.assertEqual((cal.dynamic.low, cal.dynamic.scale, cal.dynamic.high), (0.4, 0.6, 0.7))

    def test_bracket_needs_all_three_values(self):
        for missing in ('energy_scale_low', 'energy_scale_high'):
            doc = bracketed()
            del doc['dynamic'][missing]
            self.refused(doc, missing)

    def test_bracket_must_be_ordered(self):
        self.refused(bracketed(low=0.65), 'energy_scale_low')
        self.refused(bracketed(high=0.55), 'energy_scale_high')
        self.load(bracketed(low=0.6, carried=0.6, high=0.6))

    def test_bracket_values_must_be_positive_finite_numbers(self):
        for key in ('energy_scale_low', 'energy_scale_high'):
            for bad in (0.0, -0.5, float('nan'), float('inf'), True, '0.5'):
                doc = bracketed()
                doc['dynamic'][key] = bad
                self.refused(doc, key)

    def test_bracket_needs_an_anchored_or_fitted_scale(self):
        doc = bracketed(low=0.9, carried=1.0, high=1.1)
        doc['dynamic']['status'] = 'inherited'
        doc['dynamic']['provenance'] = copy.deepcopy(EMPTY)
        self.refused(doc, 'inherited')

    def test_time_takes_no_bracket(self):
        doc = document()
        doc['time']['latency_scale_low'] = 0.5
        self.refused(doc, 'latency_scale_low')
        doc = document()
        doc['time']['energy_scale_low'] = 0.5
        self.refused(doc, 'energy_scale_low')

    def test_report_at_the_three_scales(self):
        cal = self.load(bracketed())
        out = report(cal, sim_time_s=2e-3, dynamic_energy_j=1e-3, chips=1, cores=74)
        by = out['by_energy_scale']
        self.assertEqual(list(by), ['low', 'carried', 'high'])
        for name, scale in (('low', 0.4), ('carried', 0.6), ('high', 0.7)):
            self.assertEqual(by[name]['energy_scale'], scale)
            self.assertAlmostEqual(by[name]['energy_dynamic_j'], scale * 1e-3)
            self.assertAlmostEqual(by[name]['power_dynamic_w'], scale * 0.5)
            self.assertAlmostEqual(by[name]['power_static_w'], 0.15)
            self.assertAlmostEqual(by[name]['power_total_w'], 0.15 + scale * 0.5)
            self.assertAlmostEqual(by[name]['energy_inference_j'], (0.15 + scale * 0.5) * 2e-3)
            self.assertEqual(by[name]['by_rule']['per_chip']['power_total_w'],
                             by[name]['power_total_w'])
        for key in ('energy_dynamic_j', 'power_dynamic_w', 'power_total_w', 'energy_inference_j'):
            self.assertEqual(out[key], by['carried'][key])
        self.assertEqual(out['scales']['dynamic'],
                         {'value': 0.6, 'status': 'anchored', 'low': 0.4, 'high': 0.7})

    def test_report_at_the_three_scales_with_two_rules(self):
        cal = self.load(bracketed(rules=[per_chip(0.15), per_core(0.0008)]))
        out = report(cal, sim_time_s=2e-3, dynamic_energy_j=1e-3, chips=1, cores=74,
                     host_time_s=0.1)
        for name, scale in (('low', 0.4), ('carried', 0.6), ('high', 0.7)):
            level = out['by_energy_scale'][name]
            self.assertNotIn('power_total_w', level)
            core = level['by_rule']['per_core']
            self.assertAlmostEqual(core['power_static_w'], 0.0592)
            self.assertAlmostEqual(core['power_total_w'], 0.0592 + scale * 0.5)
            self.assertAlmostEqual(core['energy_total_j'],
                                   (0.0592 + scale * 0.5) * 2e-3 + 0.0592 * 0.1)
        self.assertEqual(out['by_rule'], out['by_energy_scale']['carried']['by_rule'])


class TestReportConvention(Base):
    """Time, energy and host time are totals over the run, each divided by `samples`."""

    def test_host_time_is_divided_by_samples_like_the_other_totals(self):
        cal = self.load(document([per_chip(0.15)]))
        one = report(cal, sim_time_s=2e-3, dynamic_energy_j=1e-3, chips=1, cores=4,
                     samples=1, host_time_s=0.1)
        many = report(cal, sim_time_s=2e-3 * 10, dynamic_energy_j=1e-3 * 10, chips=1, cores=4,
                      samples=10, host_time_s=0.1 * 10)
        self.assertAlmostEqual(many['host_time_s'], 0.1)
        for key in ('inference_time_s', 'energy_dynamic_j', 'energy_total_j',
                    'energy_inference_j', 'host_time_s'):
            self.assertAlmostEqual(many[key], one[key], msg=key)
        self.assertAlmostEqual(many['energy_total_j'], many['energy_inference_j'] + 0.15 * 0.1)

    def test_a_report_shares_no_mutable_part_with_the_calibration_or_itself(self):
        cal = self.load(bracketed(rules=[per_chip(0.15)]))
        out = report(cal, sim_time_s=2e-3, dynamic_energy_j=1e-3, chips=1, cores=74)
        readings = copy.deepcopy(cal.readings)
        out['readings']['injected'] = 1
        self.assertEqual(cal.readings, readings)
        self.assertIsNot(out['by_rule'], out['by_energy_scale']['carried']['by_rule'])
        before = copy.deepcopy(out['by_energy_scale']['carried']['by_rule'])
        out['by_rule']['per_chip']['power_total_w'] = -1.0
        self.assertEqual(out['by_energy_scale']['carried']['by_rule'], before)
        again = report(cal, sim_time_s=2e-3, dynamic_energy_j=1e-3, chips=1, cores=74)
        self.assertEqual(again['readings'], readings)


if __name__ == '__main__':
    unittest.main()
