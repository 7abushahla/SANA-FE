"""Saved runs: the trace stores under one directory, and their comparison."""
import json
import re
from pathlib import Path

from .store import TRACE_FORMAT, TraceStore

_SAFE_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')


def _run_dir(store_dir, run_id):
    if not isinstance(run_id, str) or not _SAFE_ID.fullmatch(run_id):
        raise ValueError(f'invalid run id {run_id!r}')
    directory = Path(store_dir) / run_id
    if not (directory / 'manifest.json').is_file():
        raise KeyError(f'no run {run_id!r}')
    return directory


def list_runs(store_dir):
    """Summaries of every readable run, newest first."""
    store_dir = Path(store_dir)
    if not store_dir.is_dir():
        return []
    runs = []
    for directory in store_dir.iterdir():
        try:
            manifest = json.loads((directory / 'manifest.json').read_text())
        except (OSError, ValueError):
            continue
        if manifest.get('format') != TRACE_FORMAT:
            continue
        records = directory / 'records.jsonl'
        updates = sum(1 for line in records.open() if line.strip()) if records.exists() else 0
        runs.append({'id': directory.name, 'workload': manifest.get('workload'),
                     'parameters': manifest.get('parameters', {}),
                     'core_map': manifest.get('core_map', {}),
                     'created': manifest.get('created'), 'updates': updates,
                     'horizon': manifest.get('horizon'),
                     'platform': manifest.get('platform'),
                     'architecture': Path(manifest.get('architecture_yaml', '')).name})
    return sorted(runs, key=lambda run: run['created'] or '', reverse=True)


def load_run(store_dir, run_id):
    """``(manifest, records)`` of one run. Raises ValueError or KeyError."""
    return TraceStore.load(_run_dir(store_dir, run_id))


def _metrics(record):
    packets = [counts['packets_out'] for counts in record.core_counts.values()]
    return {'step_time': record.step_time, 'energy': record.energy['total'],
            'hops': record.counts['hops'], 'messages': record.counts['messages'],
            'occupied': len(record.core_finish),
            'max_core_finish': max(record.core_finish.values(), default=0.0),
            'busiest_core_packets': max(packets, default=0)}


def _network(manifest, records):
    if manifest.get('network') is not None:
        return [(g['name'], g['size']) for g in manifest['network']]
    return sorted(records[0].potentials) if records else []  # stores before stage 4


def compare_runs(a, b):
    """Spike-train identity and per-update differences between two runs."""
    (manifest_a, records_a), (manifest_b, records_b) = a, b
    comparable = _network(manifest_a, records_a) == _network(manifest_b, records_b)
    # Aggregate stores keep only watched neurons' spikes: no identity claim.
    aggregate = 'aggregate' in (manifest_a.get('trace_level'), manifest_b.get('trace_level'))
    first = None
    common = min(len(records_a), len(records_b))
    if comparable and not aggregate:
        for index in range(common):
            left = {f'{g}.{o}' for g, o in records_a[index].fired}
            right = {f'{g}.{o}' for g, o in records_b[index].fired}
            if left != right:
                neuron = min(left ^ right)
                first = {'update': index + 1, 'neuron': neuron,
                         'in': 'a' if neuron in left else 'b'}
                break
        if first is None and len(records_a) != len(records_b):
            first = {'update': common + 1, 'neuron': None,
                     'in': 'a' if len(records_a) > len(records_b) else 'b'}
    updates = [{'update': index + 1, 'a': _metrics(records_a[index]),
                'b': _metrics(records_b[index])} for index in range(common)]

    def totals(records):
        return {key: sum(_metrics(r)[key] for r in records)
                for key in ('step_time', 'energy', 'hops', 'messages')}

    note = 'no updates to compare' if common == 0 else None
    if aggregate and note is None:
        note = ('spike trains not compared: aggregate runs keep only watched neurons; '
                'time, energy, hops, and per-core load are compared')
    platform_a, platform_b = manifest_a.get('platform'), manifest_b.get('platform')
    if platform_a != platform_b:
        warning = (f'different platforms ({platform_a or "file"} against {platform_b or "file"}): '
                   'time and energy come from different cost models and are not a '
                   'generation comparison')
        note = warning if note is None else note + '; ' + warning
    return {'comparable': comparable, 'note': note,
            'identical_spikes': comparable and not aggregate and first is None and common > 0,
            'first_difference': first, 'updates': updates,
            'updates_a': len(records_a), 'updates_b': len(records_b),
            'totals': {'a': totals(records_a), 'b': totals(records_b)},
            'a': {'workload': manifest_a.get('workload'), 'platform': platform_a,
                  'parameters': manifest_a.get('parameters'),
                  'core_map': manifest_a.get('core_map', {})},
            'b': {'workload': manifest_b.get('workload'), 'platform': platform_b,
                  'parameters': manifest_b.get('parameters'),
                  'core_map': manifest_b.get('core_map', {})}}
