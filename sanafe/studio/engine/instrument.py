"""Copy an architecture YAML with SANA-FE's per-update energy logging enabled.

The copy differs from its source only in ``log_energy`` flags. Scalars are
written back as their original text, because SANA-FE's YAML reader will not
cast a quoted scalar such as ``'8'`` to an integer.
"""
from pathlib import Path
import re

import yaml

MAX_LOGGED_UNITS_PER_CORE = 8
_UNIT_KINDS = ('synapse', 'dendrite', 'soma')
_RANGE = re.compile(r'\[(\d+)\.\.(\d+)\]$')


class _TextDumper(yaml.emitter.Emitter, yaml.serializer.Serializer,
                  yaml.representer.SafeRepresenter, yaml.resolver.BaseResolver):
    """Emit every scalar as plain text, without implicit type quoting."""

    def __init__(self, stream, **_):
        yaml.emitter.Emitter.__init__(self, stream, width=10000)
        yaml.serializer.Serializer.__init__(self)
        yaml.representer.SafeRepresenter.__init__(
            self, default_flow_style=False, sort_keys=False)
        yaml.resolver.BaseResolver.__init__(self)


def _as_list(node):
    return node if isinstance(node, list) else [node]


def _instances(unit):
    """Number of hardware units a YAML entry names, e.g. ``x[0..1023]`` is 1024."""
    match = _RANGE.search(str(unit.get('name', '')))
    return int(match[2]) - int(match[1]) + 1 if match else 1


def _enable(node):
    attributes = node.get('attributes')
    if not isinstance(attributes, dict):
        attributes = {}
        node['attributes'] = attributes
    attributes['log_energy'] = 'true'


def instrument_arch_yaml(source, destination):
    """Write ``destination``: ``source`` with tile, core, and unit energy logging.

    Units are logged only for cores with at most ``MAX_LOGGED_UNITS_PER_CORE``
    synapse, dendrite, and soma unit instances. Returns what was instrumented.
    """
    document = yaml.load(Path(source).read_text(), Loader=yaml.BaseLoader)
    try:
        tiles = _as_list(document['architecture']['tile'])
    except (KeyError, TypeError) as error:
        raise ValueError(f'{source}: not a SANA-FE architecture file') from error
    units_logged = True
    for tile in tiles:
        _enable(tile)
        for core in _as_list(tile['core']):
            _enable(core)
            units = [unit for kind in _UNIT_KINDS for unit in _as_list(core.get(kind, []))]
            if sum(_instances(unit) for unit in units) > MAX_LOGGED_UNITS_PER_CORE:
                units_logged = False
                continue
            for unit in units:
                _enable(unit)
    with open(destination, 'w', encoding='utf-8') as handle:
        yaml.dump(document, handle, Dumper=_TextDumper)
    return {'tiles': True, 'cores': True, 'units': units_logged}
