"""Run manifest plus one JSON line per UpdateRecord.

Lines stay strict JSON, so browsers can parse them. Non-finite floats, such as
a diverging membrane at -inf, are written as ``{"$float": "-inf"}`` and
restored on load.
"""
import json
import math
from pathlib import Path

from .records import UpdateRecord

TRACE_FORMAT = 'sanafe-studio-trace-v1'
_NON_FINITE = {'inf': math.inf, '-inf': -math.inf, 'nan': math.nan}


def to_strict_json(value):
    """Replace non-finite floats with {"$float": ...} tags, recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return {'$float': 'nan'}
        return {'$float': 'inf' if value > 0 else '-inf'}
    if isinstance(value, dict):
        return {key: to_strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_strict_json(item) for item in value]
    return value


def from_strict_json(value):
    """Restore floats tagged by to_strict_json, recursively."""
    if isinstance(value, dict):
        if value.keys() == {'$float'}:
            return _NON_FINITE[value['$float']]
        return {key: from_strict_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [from_strict_json(item) for item in value]
    return value


class TraceStore:
    def __init__(self, directory, manifest):
        self.directory = Path(directory)
        self.manifest = manifest

    @classmethod
    def create(cls, directory, manifest):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {'format': TRACE_FORMAT, **manifest}
        (directory / 'manifest.json').write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
        return cls(directory, manifest)

    def append(self, record):
        line = json.dumps(to_strict_json(record.to_dict()), allow_nan=False)
        with open(self.directory / 'records.jsonl', 'a', encoding='utf-8') as handle:
            handle.write(line + '\n')

    @staticmethod
    def load(directory):
        directory = Path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest.get('format') != TRACE_FORMAT:
            raise ValueError(f'{directory}: not a {TRACE_FORMAT} store')
        path = directory / 'records.jsonl'
        lines = path.read_text().splitlines() if path.exists() else []
        return manifest, [UpdateRecord.from_dict(from_strict_json(json.loads(line)))
                          for line in lines if line]
