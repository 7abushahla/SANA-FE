"""Trace-derived binary workload and conditional Loihi 2 runtime tests."""
import math
from types import SimpleNamespace
import unittest

from sanafe.loihi2_runtime import RuntimeRates, analyze_binary_chain, max_affine_runtime


def event(layer, offset=0):
    return SimpleNamespace(group_name=f'layer_{layer}', neuron_offset=offset)


def message(layer, offset, src_tile, dst_tile, fanout):
    return {'timestep': 1, 'placeholder': False,
            'src_neuron_group_id': f'layer_{layer}', 'src_neuron_offset': offset,
            'src_tile_id': src_tile, 'src_core_offset': 0,
            'dest_tile_id': dst_tile, 'dest_core_offset': 0,
            'src_x': src_tile // 4, 'src_y': src_tile % 4,
            'dest_x': dst_tile // 4, 'dest_y': dst_tile % 4,
            'spikes': fanout}


def manifest(sizes=(2, 2), places=((0, 0), (4, 0)), weights=(((2, 0), (0, 3)),)):
    return {'schema_version': 1, 'execution_profile': 'qcfs-if-int24-binary-v1',
            'architecture_profile': {'topology': {'width': 8, 'height': 4}},
            'layer_sizes': list(sizes), 'placements': [list(p) for p in places],
            'effective_weights': weights, 'updates': 2}


def result(messages, spikes, updated=None, fanout=None):
    updates = len(messages)
    return {'message_trace': messages, 'spike_trace': spikes,
            'perf_trace': {'timestep': list(range(1, updates + 1)),
                           'updated': updated or [4] * updates,
                           'spikes': fanout or [sum(m['spikes'] for m in row) for row in messages]}}


class TestBinaryWorkload(unittest.TestCase):
    def test_nonzero_synops_dense_zero_reads_link_and_empty_step(self):
        trace = result([[message(0, 0, 0, 4, 2)], []], [[event(0)], []])
        work = analyze_binary_chain(manifest(), trace, packet_bits=32)
        first, second = work['steps']
        self.assertEqual(first['maxima'], {'dendops': 2, 'synops': 1,
                                           'dense_synmem_entries': 2, 'link_bits': 32})
        self.assertEqual(first['links'], [{'from': [0, 0], 'to': [1, 0], 'bits': 32}])
        self.assertEqual(second['maxima'], {'dendops': 2, 'synops': 0,
                                            'dense_synmem_entries': 0, 'link_bits': 0})
        self.assertEqual(work['messages'], 1)
        self.assertEqual(work['packet_bits_assumption'], 32)

    def test_colocated_layer_still_computes_without_mesh_traffic(self):
        local = manifest(places=((0, 0), (0, 0)))
        trace = result([[message(0, 0, 0, 0, 2)], []], [[event(0)], []])
        first = analyze_binary_chain(local, trace, packet_bits=32)['steps'][0]
        self.assertEqual(first['maxima']['dendops'], 4)
        self.assertEqual(first['maxima']['synops'], 1)
        self.assertEqual(first['maxima']['link_bits'], 0)

    def test_directed_shared_link_load_is_summed(self):
        chain = manifest(sizes=(1, 1, 1, 1), places=((0, 0), (8, 0), (0, 0), (8, 0)),
                         weights=(((1,),), ((1,),), ((1,),)))
        chain['updates'] = 1
        trace = result([[message(0, 0, 0, 8, 1), message(2, 0, 0, 8, 1)]],
                       [[event(0), event(2)]], updated=[4])
        first = analyze_binary_chain(chain, trace, packet_bits=32)['steps'][0]
        self.assertEqual(first['maxima']['link_bits'], 64)
        self.assertEqual(first['links'], [
            {'from': [0, 0], 'to': [1, 0], 'bits': 64},
            {'from': [1, 0], 'to': [2, 0], 'bits': 64}])

    def test_horizontal_before_vertical_then_reverse_directions(self):
        chain = manifest(sizes=(1, 1, 1), places=((0, 0), (5, 0), (0, 0)),
                         weights=(((1,),), ((1,),)))
        chain['updates'] = 1
        trace = result([[message(0, 0, 0, 5, 1), message(1, 0, 5, 0, 1)]],
                       [[event(0), event(1)]], updated=[3])
        links = analyze_binary_chain(chain, trace, packet_bits=8)['steps'][0]['links']
        self.assertEqual(links, [
            {'from': [0, 0], 'to': [1, 0], 'bits': 8},
            {'from': [0, 1], 'to': [0, 0], 'bits': 8},
            {'from': [1, 0], 'to': [1, 1], 'bits': 8},
            {'from': [1, 1], 'to': [0, 1], 'bits': 8}])

    def test_rejects_inconsistent_trace_and_packet_assumptions(self):
        trace = result([[message(0, 0, 0, 4, 2)], []], [[event(0)], []])
        for bits in (0, -1, 1.5, True):
            with self.subTest(bits=bits), self.assertRaises(ValueError):
                analyze_binary_chain(manifest(), trace, packet_bits=bits)
        bad = result([[message(0, 0, 0, 4, 3)], []], [[event(0)], []])
        with self.assertRaisesRegex(ValueError, 'fanout'):
            analyze_binary_chain(manifest(), bad, packet_bits=32)
        bad = result([[message(0, 0, 0, 4, 2)], []], [[], []])
        with self.assertRaisesRegex(ValueError, 'spike'):
            analyze_binary_chain(manifest(), bad, packet_bits=32)


class TestMaxAffineRuntime(unittest.TestCase):
    def test_terms_bottleneck_and_sum_preserve_empty_step_barrier(self):
        trace = result([[message(0, 0, 0, 4, 2)], []], [[event(0)], []])
        work = analyze_binary_chain(manifest(), trace, packet_bits=32)
        rates = RuntimeRates(dendop_seconds=2e-9, synop_seconds=3e-9,
                             synmem_read_seconds=5e-9, link_bits_per_second=8e9,
                             barrier_seconds=1e-9, dense_entries_per_read=1)
        estimate = max_affine_runtime(work, rates)
        self.assertEqual(estimate['steps'][0]['bottleneck'], 'synmem')
        self.assertAlmostEqual(estimate['steps'][0]['lower_bound_seconds'], 10e-9)
        self.assertEqual(estimate['steps'][1]['bottleneck'], 'dendop')
        self.assertAlmostEqual(estimate['total_lower_bound_seconds'], 14e-9)
        self.assertFalse(estimate['calibrated_for_loihi2'])

    def test_each_independent_term_can_dominate(self):
        trace = result([[message(0, 0, 0, 4, 2)]], [[event(0)]], updated=[4])
        chain = manifest()
        chain['updates'] = 1
        work = analyze_binary_chain(chain, trace, packet_bits=32)
        common = {'dendop_seconds': 1e-12, 'synop_seconds': 1e-12,
                  'synmem_read_seconds': 1e-12, 'link_bits_per_second': 1e15,
                  'barrier_seconds': 1e-12, 'dense_entries_per_read': 2}
        for name, override in [('dendop', {'dendop_seconds': 1e-6}),
                               ('synop', {'synop_seconds': 1e-6}),
                               ('synmem', {'synmem_read_seconds': 1e-6}),
                               ('link', {'link_bits_per_second': 1e3}),
                               ('barrier', {'barrier_seconds': 1e-3})]:
            with self.subTest(name=name):
                rates = RuntimeRates(**(common | override))
                self.assertEqual(max_affine_runtime(work, rates)['steps'][0]['bottleneck'], name)

    def test_all_coefficients_and_packing_must_be_explicit_and_positive(self):
        values = {'dendop_seconds': 1., 'synop_seconds': 1., 'synmem_read_seconds': 1.,
                  'link_bits_per_second': 1., 'barrier_seconds': 1.,
                  'dense_entries_per_read': 1}
        for field, bad in [('dendop_seconds', 0), ('synop_seconds', -1),
                           ('synmem_read_seconds', math.nan),
                           ('link_bits_per_second', math.inf),
                           ('barrier_seconds', True), ('dense_entries_per_read', 0),
                           ('dense_entries_per_read', 1.5)]:
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                RuntimeRates(**(values | {field: bad}))

    def test_memory_reads_round_per_incoming_axon_not_after_core_aggregation(self):
        chain = manifest(sizes=(2, 1), places=((0, 0), (4, 0)),
                         weights=(((1, 1),),))
        chain['updates'] = 1
        trace = result([[message(0, 0, 0, 4, 1), message(0, 1, 0, 4, 1)]],
                       [[event(0, 0), event(0, 1)]], updated=[3])
        work = analyze_binary_chain(chain, trace, packet_bits=64)
        rates = RuntimeRates(dendop_seconds=1e-12, synop_seconds=1e-12,
                             synmem_read_seconds=1e-6, link_bits_per_second=1e15,
                             barrier_seconds=1e-12, dense_entries_per_read=2)
        self.assertEqual(max_affine_runtime(work, rates)['steps'][0]['synmem_reads_assumed'], 2)


if __name__ == '__main__':
    unittest.main()
