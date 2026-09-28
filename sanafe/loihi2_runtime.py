"""Trace-derived workload and a conditional Loihi 2 max-affine runtime bound.

This module does not reuse SANA-FE's inherited Loihi 1 latency coefficients.
It supports the restricted binary, dense IF/Dense chain exporter only. The
packet width, dense-memory packing, and effective rates must be supplied as
assumptions or measurements before the returned time has numerical meaning.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real

RUNTIME_PAPER = 'https://arxiv.org/html/2601.10035v2'


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _positive_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite number')
    return float(value)


@dataclass(frozen=True)
class RuntimeRates:
    """Explicit model coefficients, with units and dense-packing assumption.

    These values are never filled from the bundled Loihi 1 architecture.
    Passing published Loihi 2 rates still requires checking hardware revision,
    neuron program, and workload compatibility before calling the estimate
    calibrated for the thesis network.
    """
    dendop_seconds: float
    synop_seconds: float
    synmem_read_seconds: float
    link_bits_per_second: float
    barrier_seconds: float
    dense_entries_per_read: int

    def __post_init__(self):
        for name in ('dendop_seconds', 'synop_seconds', 'synmem_read_seconds',
                     'link_bits_per_second', 'barrier_seconds'):
            object.__setattr__(self, name, _positive_finite(getattr(self, name), name))
        object.__setattr__(self, 'dense_entries_per_read',
                           _positive_integer(self.dense_entries_per_read,
                                             'dense_entries_per_read'))


def _neuron_address(item):
    try:
        name, offset = item.group_name, item.neuron_offset
        if not name.startswith('layer_'):
            raise ValueError('Unexpected spike group')
        return int(name.removeprefix('layer_')), int(offset)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError('Malformed spike address in trace') from exc


def _route_links(src, dest):
    """Directed dimension-order route, horizontal before vertical."""
    x, y = src
    while x != dest[0]:
        next_x = x + (1 if dest[0] > x else -1)
        yield (x, y, next_x, y)
        x = next_x
    while y != dest[1]:
        next_y = y + (1 if dest[1] > y else -1)
        yield (x, y, x, next_y)
        y = next_y


def analyze_binary_chain(manifest, result, *, packet_bits):
    """Extract per-step busiest-core work and directed-link traffic.

    Message processing is attributed to the message's sending step, matching
    SANA-FE's trace: it applies synapses before the next numbered soma update.
    DendOps count every mapped IF update. Dense SynMem *entries* count all
    stored weights in a triggered row, including zero; the number of physical
    memory reads depends on packing and is resolved only by RuntimeRates.
    Distinct cores also load directed core-to-router and router-to-core links,
    including when they share a router. Same-core connections bypass the NoC
    in this model. Two physical fabrics are aggregated into one effective link
    as in the analytical Loihi 2 model; actual fabric assignment is unknown.
    """
    packet_bits = _positive_integer(packet_bits, 'packet_bits')
    if manifest.get('execution_profile') not in ('qcfs-if-int24-binary-v1',
                                                 'qcfs-if-int24-binary-windowed-v1'):
        raise ValueError('Expected the restricted integer binary chain manifest')
    sizes = manifest['layer_sizes']
    places = [tuple(p) for p in manifest['placements']]
    weights = manifest['effective_weights']
    topology = manifest['architecture_profile']['topology']
    width, height = topology['width'], topology['height']
    if (not sizes or len(sizes) != len(places) or len(weights) != len(sizes) - 1 or
            any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in sizes) or
            any(len(w) != sizes[i + 1] or any(len(row) != sizes[i] for row in w)
                for i, w in enumerate(weights))):
        raise ValueError('Malformed chain manifest')
    messages = result['message_trace']
    spikes = result['spike_trace']
    perf = result['perf_trace']
    count = manifest['updates']
    windows = manifest.get('valid_update_windows_zero_based',
                           [[0, count] for _ in sizes])
    if (len(windows) != len(sizes) or
            any(len(window) != 2 or
                any(isinstance(value, bool) or not isinstance(value, Integral)
                    for value in window) or
                window[0] < 0 or window[1] <= window[0]
                for window in windows)):
        raise ValueError('Malformed valid update windows')
    if (manifest['execution_profile'] == 'qcfs-if-int24-binary-windowed-v1' and
            'valid_update_windows_zero_based' not in manifest):
        raise ValueError('Windowed profile requires valid update windows')
    if (len(messages) != count or len(spikes) != count or
            any(len(perf[key]) != count for key in ('timestep', 'updated', 'spikes'))):
        raise ValueError('Trace length does not match manifest updates')
    if any(len(p) != 2 or p[0] < 0 or p[0] >= width * height or p[1] < 0 or p[1] >= 4
           for p in places):
        raise ValueError('Placement outside declared topology')

    steps = []
    messages_total = 0
    for index, (row, active) in enumerate(zip(messages, spikes)):
        t = index + 1
        valid_layers = [start <= index < stop for start, stop in windows]
        expected_updates = sum(size for size, valid in zip(sizes, valid_layers)
                               if valid)
        if perf['timestep'][index] != t or perf['updated'][index] != expected_updates:
            raise ValueError('Timestep or DendOp trace disagrees with mapped chain')
        fired = [_neuron_address(item) for item in active]
        if len(fired) != len(set(fired)) or any(i < 0 or i >= len(sizes) or
                                               j < 0 or j >= sizes[i] for i, j in fired):
            raise ValueError('Invalid or duplicate emitted spike in trace')
        if any(not valid_layers[i] for i, _ in fired):
            raise ValueError('Inactive layer emitted a spike')
        expected_sources = {address for address in fired if address[0] < len(sizes) - 1}
        seen_sources = set()
        core_counts = {p: {'dendops': 0, 'synops': 0,
                           'dense_synmem_entries': 0, 'dense_synmem_accesses': []}
                       for p in places}
        for size, place, valid in zip(sizes, places, valid_layers):
            if valid:
                core_counts[place]['dendops'] += size
        link_bits = Counter()
        endpoint_bits = Counter()
        fanout_total = 0
        real_messages = 0
        for m in row:
            if m['placeholder']:
                continue
            real_messages += 1
            if m['timestep'] != t or not m['src_neuron_group_id'].startswith('layer_'):
                raise ValueError('Message timestep or source group disagrees with trace')
            try:
                layer = int(m['src_neuron_group_id'].removeprefix('layer_'))
                offset = int(m['src_neuron_offset'])
            except (TypeError, ValueError) as exc:
                raise ValueError('Malformed message source') from exc
            source = (layer, offset)
            if source not in expected_sources or source in seen_sources:
                raise ValueError('Message does not correspond to one emitted source spike')
            seen_sources.add(source)
            src, dest = places[layer], places[layer + 1]
            expected_src_xy = (src[0] // height, src[0] % height)
            expected_dest_xy = (dest[0] // height, dest[0] % height)
            if ((m['src_tile_id'], m['src_core_offset']) != src or
                    (m['dest_tile_id'], m['dest_core_offset']) != dest or
                    (m['src_x'], m['src_y']) != expected_src_xy or
                    (m['dest_x'], m['dest_y']) != expected_dest_xy):
                raise ValueError('Message placement differs from export manifest')
            if m['spikes'] != sizes[layer + 1]:
                raise ValueError('Message fanout differs from dense layer')
            fanout_total += m['spikes']
            core_counts[dest]['dense_synmem_entries'] += sizes[layer + 1]
            core_counts[dest]['dense_synmem_accesses'].append(sizes[layer + 1])
            core_counts[dest]['synops'] += sum(weights[layer][post][offset] != 0
                                                for post in range(sizes[layer + 1]))
            if src != dest:
                endpoint_bits[('core_to_router', src[0], src[1])] += packet_bits
                endpoint_bits[('router_to_core', dest[0], dest[1])] += packet_bits
            for link in _route_links(expected_src_xy, expected_dest_xy):
                link_bits[link] += packet_bits
        if seen_sources != expected_sources:
            raise ValueError('Emitted source spike is missing its message')
        if perf['spikes'][index] != fanout_total:
            raise ValueError('Synapse fanout disagrees with performance trace')
        messages_total += real_messages
        ordered_cores = [{'tile': p[0], 'core': p[1], **counts}
                         for p, counts in sorted(core_counts.items())]
        links = [{'from': [x1, y1], 'to': [x2, y2], 'bits': bits}
                 for (x1, y1, x2, y2), bits in sorted(link_bits.items())]
        endpoint_links = [{'kind': kind, 'tile': tile, 'core': core, 'bits': bits}
                          for (kind, tile, core), bits in sorted(endpoint_bits.items())]
        maxima = {name: max(c[name] for c in core_counts.values())
                  for name in ('dendops', 'synops', 'dense_synmem_entries')}
        maxima['interrouter_link_bits'] = max(link_bits.values(), default=0)
        maxima['link_bits'] = max(maxima['interrouter_link_bits'],
                                  max(endpoint_bits.values(), default=0))
        steps.append({'timestep': t, 'messages': real_messages,
                      'cores': ordered_cores, 'links': links,
                      'endpoint_links': endpoint_links, 'maxima': maxima})
    return {'schema_version': 2, 'source': 'SANA-FE binary message and spike traces',
            'updates': count, 'messages': messages_total,
            'packet_bits_assumption': packet_bits,
            'link_scope': 'directed core-router endpoints and XY router-router links; '
                          'two physical fabrics aggregated; same-core bypass assumed',
            'synmem_measure': 'dense stored entries; physical reads require packing assumption',
            'input_output_transfer_included': False,
            'calibrated_for_loihi2': False, 'steps': steps}


def rescale_binary_packet_bits(workload, packet_bits):
    """Apply another assumed packet width to unchanged binary message counts.

    This is a sensitivity transformation. It does not model physical packet
    headers, flits, fabric assignment, or a Loihi 2 wire protocol.
    """
    packet_bits = _positive_integer(packet_bits, 'packet_bits')
    if workload.get('schema_version') != 2:
        raise ValueError('Expected endpoint-aware binary workload schema 2')
    previous = _positive_integer(workload.get('packet_bits_assumption'),
                                 'previous packet_bits_assumption')
    scaled = deepcopy(workload)

    def change(value):
        if isinstance(value, bool) or not isinstance(value, Integral) or value % previous:
            raise ValueError('Link bits disagree with original packet width')
        return int(value // previous * packet_bits)

    for step in scaled['steps']:
        for link in step['links']:
            link['bits'] = change(link['bits'])
        for link in step['endpoint_links']:
            link['bits'] = change(link['bits'])
        for key in ('interrouter_link_bits', 'link_bits'):
            step['maxima'][key] = change(step['maxima'][key])
    scaled['packet_bits_assumption'] = packet_bits
    return scaled


def max_affine_runtime(workload, rates):
    """Apply Timcheck et al. Eq. 1 independently of SANA-FE's cost model."""
    if not isinstance(rates, RuntimeRates):
        raise ValueError('rates must be an explicit RuntimeRates instance')
    steps = []
    for step in workload['steps']:
        maxima = step['maxima']
        reads = max((sum(math.ceil(access / rates.dense_entries_per_read)
                         for access in core['dense_synmem_accesses'])
                     for core in step['cores']), default=0)
        terms = {
            'dendop': maxima['dendops'] * rates.dendop_seconds,
            'synop': maxima['synops'] * rates.synop_seconds,
            'synmem': reads * rates.synmem_read_seconds,
            'link': maxima['link_bits'] / rates.link_bits_per_second,
            'barrier': rates.barrier_seconds,
        }
        bottleneck = max(terms, key=terms.get)
        steps.append({'timestep': step['timestep'], 'terms_seconds': terms,
                      'synmem_reads_assumed': reads, 'bottleneck': bottleneck,
                      'lower_bound_seconds': terms[bottleneck]})
    return {'model': 'Timcheck-et-al-2026-max-affine-equation-1',
            'source': RUNTIME_PAPER, 'rates': asdict(rates),
            'calibrated_for_loihi2': False,
            'scope': 'conditional lower bound for restricted binary dense chain; no host I/O',
            'steps': steps,
            'total_lower_bound_seconds': sum(s['lower_bound_seconds'] for s in steps)}
