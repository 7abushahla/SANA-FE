"""Differences between two architecture YAML files, for the Architecture panel.

List items that carry a ``name`` (tiles, cores, pipeline units) are matched
by name; other lists by position. An item present on one side only is one
row holding the whole item. Rows follow the baseline's order, with items
only the loaded file has placed after them.
"""
from importlib.resources import files
from pathlib import Path

import yaml


def _keyed(items):
    if all(isinstance(item, dict) and 'name' in item for item in items):
        return {f'[{item["name"]}]': item for item in items}
    return {f'[{index}]': item for index, item in enumerate(items)}


def _walk(path, baseline, loaded, rows):
    if isinstance(baseline, dict) and isinstance(loaded, dict):
        pairs = [(f'{path}.{key}' if path else str(key), key) for key in
                 [*baseline, *(k for k in loaded if k not in baseline)]]
        for child, key in pairs:
            _compare(child, key in baseline, baseline.get(key),
                     key in loaded, loaded.get(key), rows)
        return
    if isinstance(baseline, list) and isinstance(loaded, list):
        left, right = _keyed(baseline), _keyed(loaded)
        for key in [*left, *(k for k in right if k not in left)]:
            _compare(path + key, key in left, left.get(key),
                     key in right, right.get(key), rows)
        return
    if baseline != loaded:
        rows.append({'path': path, 'baseline': baseline, 'loaded': loaded,
                     'change': 'changed'})


def _compare(path, in_baseline, baseline, in_loaded, loaded, rows):
    if not in_loaded:
        rows.append({'path': path, 'baseline': baseline, 'loaded': None,
                     'change': 'removed'})
    elif not in_baseline:
        rows.append({'path': path, 'baseline': None, 'loaded': loaded,
                     'change': 'added'})
    else:
        _walk(path, baseline, loaded, rows)


def diff_texts(loaded_name, loaded_text, baseline_name, baseline_text):
    rows = []
    _walk('', yaml.safe_load(baseline_text), yaml.safe_load(loaded_text), rows)
    return {'loaded': loaded_name, 'baseline': baseline_name, 'rows': rows}


def architecture_diff(loaded_path, baseline_path):
    loaded_path, baseline_path = Path(loaded_path), Path(baseline_path)
    return diff_texts(loaded_path.name, loaded_path.read_text(),
                      baseline_path.name, baseline_path.read_text())


def bundled_architectures():
    """Architecture files shipped in ``sanafe.examples``, by stem."""
    found = {}
    for entry in sorted(files('sanafe.examples').iterdir(), key=lambda e: e.name):
        if not entry.name.endswith('.yaml'):
            continue
        path = Path(str(entry))
        try:
            data = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
        if isinstance(data, dict) and 'architecture' in data:
            found[path.stem] = path
    return found
