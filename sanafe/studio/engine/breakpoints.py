"""Declarative breakpoints: JSON conditions checked after every update.

Kinds: ``neuron_fires {neuron}``, ``core_sends {core, more_than}``,
``step_time {more_than}`` (seconds), ``update {equals}``, and
``reference_mismatch {}``. Every breakpoint has an ``id`` and may carry
``enabled: false``. Conditions are data, never code.
"""
from numbers import Real

KINDS = {
    'neuron_fires': ('neuron',),
    'core_sends': ('core', 'more_than'),
    'step_time': ('more_than',),
    'update': ('equals',),
    'reference_mismatch': (),
}


def _number(spec, field, integer=False, minimum=0):
    value = spec.get(field)
    ok = (isinstance(value, int) if integer else isinstance(value, Real))
    if isinstance(value, bool) or not ok or value < minimum:
        kind = 'an integer' if integer else 'a number'
        raise ValueError(f'breakpoint {spec.get("id")}: {field} must be {kind} '
                         f'of at least {minimum}, got {value!r}')
    return value


def describe(spec):
    kind = spec['kind']
    if kind == 'neuron_fires':
        return f'{spec["neuron"]} fires'
    if kind == 'core_sends':
        return f'core {spec["core"]} sends more than {spec["more_than"]} packets'
    if kind == 'step_time':
        return f'step time exceeds {spec["more_than"] * 1e9:g} ns'
    if kind == 'update':
        return f'update {spec["equals"]}'
    return 'reference mismatch'


class Breakpoints:
    def __init__(self, specs=()):
        self.specs = list(specs)

    @classmethod
    def compile(cls, specs, session):
        """Validate against the session's neurons and cores; raise ValueError."""
        if not isinstance(specs, list):
            raise ValueError('breakpoints: expected a list')
        logged = {f'{n.group}.{n.offset}': n.log_spikes for n in session.neurons}
        cores = {core.key for tile in session.layout.tiles for core in tile.cores}
        checked, seen = [], set()
        for spec in specs:
            if not isinstance(spec, dict):
                raise ValueError(f'breakpoint: expected an object, got {spec!r}')
            ident = spec.get('id')
            if not isinstance(ident, str) or not ident or ident in seen:
                raise ValueError(f'breakpoint: each needs a unique string id, got {ident!r}')
            seen.add(ident)
            kind = spec.get('kind')
            if kind not in KINDS:
                raise ValueError(f'breakpoint {ident}: unknown kind {kind!r}; '
                                 f'choose one of {", ".join(KINDS)}')
            enabled = spec.get('enabled', True)
            if not isinstance(enabled, bool):
                raise ValueError(f'breakpoint {ident}: enabled must be true or false, '
                                 f'got {enabled!r}')
            clean = {'id': ident, 'kind': kind, 'enabled': enabled}
            if kind == 'reference_mismatch' and getattr(session.built, 'reference', None) is None:
                raise ValueError(f'breakpoint {ident}: this workload has no reference check, '
                                 'so a reference mismatch cannot occur')
            if kind == 'neuron_fires':
                neuron = spec.get('neuron')
                if neuron not in logged:
                    raise ValueError(f'breakpoint {ident}: no neuron {neuron!r}')
                if not logged[neuron]:
                    raise ValueError(f'breakpoint {ident}: {neuron} does not log spikes, '
                                     'so its firing is not recorded')
                clean['neuron'] = neuron
            if kind == 'core_sends':
                if spec.get('core') not in cores:
                    raise ValueError(f'breakpoint {ident}: no core {spec.get("core")!r}')
                clean['core'] = spec['core']
                clean['more_than'] = _number(spec, 'more_than', integer=True)
            if kind == 'step_time':
                clean['more_than'] = float(_number(spec, 'more_than'))
            if kind == 'update':
                clean['equals'] = _number(spec, 'equals', integer=True, minimum=1)
            checked.append(clean)
        return cls(checked)

    @classmethod
    def compile_valid(cls, specs, session):
        """Keep the specs that validate; return them with the rejections' reasons."""
        kept, rejected = [], []
        for spec in specs if isinstance(specs, list) else []:
            try:
                kept.extend(cls.compile(kept + [spec], session).specs[len(kept):])
            except ValueError as error:
                rejected.append(str(error))
        return cls(kept), rejected

    def check(self, record):
        """The first enabled breakpoint this record hits, as a reason, or None."""
        for spec in self.specs:
            if spec['enabled'] and self._hit(spec, record):
                return f'breakpoint {spec["id"]}: {describe(spec)}'
        return None

    @staticmethod
    def _hit(spec, record):
        kind = spec['kind']
        if kind == 'neuron_fires':
            group, _, offset = spec['neuron'].rpartition('.')
            return any(g == group and int(o) == int(offset) for g, o in record.fired)
        if kind == 'core_sends':
            counts = record.core_counts.get(spec['core'])
            return bool(counts) and counts['packets_out'] > spec['more_than']
        if kind == 'step_time':
            return record.step_time > spec['more_than']
        if kind == 'update':
            return record.update == spec['equals']
        reference = record.reference
        return bool(reference) and reference.get('status') == 'mismatch'
