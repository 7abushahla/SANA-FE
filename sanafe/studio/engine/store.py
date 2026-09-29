"""Run manifest plus one JSON line per UpdateRecord."""
import json
from pathlib import Path

from .records import UpdateRecord

TRACE_FORMAT = 'sanafe-studio-trace-v1'


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
        line = json.dumps(record.to_dict(), allow_nan=False)
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
        return manifest, [UpdateRecord.from_dict(json.loads(line))
                          for line in lines if line]
