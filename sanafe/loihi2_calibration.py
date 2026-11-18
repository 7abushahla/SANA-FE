"""Static power and the reported terms of the Loihi 2 route.

The simulator models dynamic energy and time. It has no static power. This
module is the only place where static power enters. It reads a calibration
file and turns a modeled time and a modeled dynamic energy into inference
time, static, dynamic and total power, and energy as power times time.

The calibration file is data produced elsewhere. It does not touch any
architecture file. Example of the schema (YAML)::

    schema: loihi2-calibration-v1
    represents: One Oheo Gulch chip of unknown revision.
    readings:                     # informational, copied into every report
      neuron_update_ns: 6.27
      link_bits_per_s: 9.05e9
      alternative: {neuron_update_ns: 7.4, link_bits_per_s: 8.42e9}
    static:
      rules:                      # one or two entries, see below
        - name: per_chip          # per_chip | per_core | affine
          watts_per_chip: 0.174   # affine carries both coefficients
          status: fitted          # fitted | anchored | inherited
          provenance: {sources: 3, rows: ["record:id"], note: "One sentence."}
        - name: per_core
          watts_per_core: 0.000847
          status: fitted
          provenance: {sources: 1, rows: ["record:id"], note: "One sentence."}
    time:
      latency_scale: 1.0          # multiplies a modeled time, 1.0 is not anchored
      status: inherited           # anchored requires at least one row
      provenance: {sources: 0, rows: [], note: "One sentence."}
    dynamic:
      energy_scale: 1.0           # multiplies a modeled dynamic energy
      status: inherited
      provenance: {sources: 0, rows: [], note: "One sentence."}
    transfer:                     # may be empty
      - {coefficient: "name", from: "unit", to: ["unit"], assumption: "One sentence."}

More than one static rule means that the data cannot decide between them.
Then no single static power is returned unless a rule is named, because one
number would hide the disagreement.

The dynamic energy scale may be a bracket. Then ``dynamic`` also carries
``energy_scale_low`` and ``energy_scale_high``, both or neither, with
low <= energy_scale <= high and a status of anchored or fitted, and the
provenance note says what each value rests on. ``report`` then gives every
quantity that depends on the scale at all three values (``by_energy_scale``).
The time section takes no bracket.
"""
from dataclasses import dataclass
import math
from numbers import Integral, Real
from pathlib import Path

import yaml

SCHEMA = 'loihi2-calibration-v1'
STATUSES = ('fitted', 'anchored', 'inherited')
RULE_FIELDS = {
    'per_chip': ('watts_per_chip',),
    'per_core': ('watts_per_core',),
    'affine': ('watts_per_chip', 'watts_per_core'),
}
TOP_KEYS = {'schema', 'represents', 'readings', 'static', 'time', 'dynamic', 'transfer'}
TRANSFER_KEYS = ('coefficient', 'from', 'to', 'assumption')
RULE_QUANTITIES = ('power_static_w', 'power_total_w', 'energy_inference_j', 'energy_total_j')
BRACKET_KEYS = ('energy_scale_low', 'energy_scale_high')
SCALE_LEVELS = ('low', 'carried', 'high')


@dataclass(frozen=True)
class StaticRule:
    name: str
    watts_per_chip: float
    watts_per_core: float
    status: str
    provenance: dict


@dataclass(frozen=True)
class Anchor:
    scale: float
    status: str
    provenance: dict
    low: float = None
    high: float = None


@dataclass(frozen=True)
class Calibration:
    represents: str
    readings: dict
    rules: tuple
    time: Anchor
    dynamic: Anchor
    transfer: tuple


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number, got {value!r}')
    if positive and value <= 0:
        raise ValueError(f'{name} must be positive, got {value!r}')
    if not positive and value < 0:
        raise ValueError(f'{name} must be nonnegative, got {value!r}')
    return float(value)


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer of at least {minimum}, got {value!r}')
    return int(value)


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be a mapping')
    return value


def _check_keys(mapping, allowed, where):
    for key in mapping:
        if key not in allowed:
            raise ValueError(f'unknown key {key!r} in {where}')
    for key in allowed:
        if key not in mapping:
            raise ValueError(f'missing key {key!r} in {where}')


def _status_and_provenance(owner, where):
    status = owner.get('status')
    if status not in STATUSES:
        raise ValueError(f'{where} status must be one of {STATUSES}, got {status!r}')
    provenance = owner.get('provenance')
    _mapping(provenance, f'{where} provenance')
    if status in ('fitted', 'anchored'):
        rows = provenance.get('rows')
        if (_integer(provenance.get('sources'), f'{where} provenance sources', 1) < 1
                or not isinstance(rows, list) or not rows
                or not all(isinstance(row, str) and row for row in rows)):
            raise ValueError(f'{where} has status {status!r} and needs provenance '
                             'with sources of at least 1 and a non-empty list of row ids')
    return status, dict(provenance)


def _rule(entry, index):
    where = f'static rule {index}'
    _mapping(entry, where)
    name = entry.get('name')
    if name not in RULE_FIELDS:
        raise ValueError(f'{where} has unknown rule name {name!r}')
    fields = RULE_FIELDS[name]
    _check_keys(entry, {'name', 'status', 'provenance', *fields}, f'rule {name!r}')
    values = {f: _number(entry[f], f'{name}.{f}') for f in fields}
    status, provenance = _status_and_provenance(entry, f'rule {name!r}')
    return StaticRule(name, values.get('watts_per_chip', 0.0), values.get('watts_per_core', 0.0),
                      status, provenance)


def _anchor(section, scale_key, where, bracket_keys=()):
    _mapping(section, where)
    present = [key for key in bracket_keys if key in section]
    if present and len(present) != len(bracket_keys):
        missing = [key for key in bracket_keys if key not in section]
        raise ValueError(f'{where} carries {present} and is missing {missing}, a bracket '
                         'needs both ends')
    _check_keys(section, {scale_key, 'status', 'provenance', *present}, where)
    scale = _number(section[scale_key], f'{where}.{scale_key}', positive=True)
    status, provenance = _status_and_provenance(section, where)
    if scale != 1.0 and status == 'inherited':
        raise ValueError(f'{where}.{scale_key} is {scale} and needs a status of '
                         "'anchored' or 'fitted'")
    if not present:
        return Anchor(scale, status, provenance)
    low_key, high_key = bracket_keys
    low = _number(section[low_key], f'{where}.{low_key}', positive=True)
    high = _number(section[high_key], f'{where}.{high_key}', positive=True)
    if status == 'inherited':
        raise ValueError(f'{where} carries a bracket and has status inherited, a bracket '
                         "needs a status of 'anchored' or 'fitted'")
    if low > scale:
        raise ValueError(f'{where}.{low_key} is {low}, above {scale_key} {scale}')
    if high < scale:
        raise ValueError(f'{where}.{high_key} is {high}, below {scale_key} {scale}')
    return Anchor(scale, status, provenance, low, high)


def _transfer(entry):
    _mapping(entry, 'transfer entry')
    for key in TRANSFER_KEYS:
        if key not in entry:
            raise ValueError(f'transfer entry is missing {key!r}')
    for key in entry:
        if key not in TRANSFER_KEYS:
            raise ValueError(f'unknown key {key!r} in transfer entry')
    return {key: (list(entry[key]) if isinstance(entry[key], list) else entry[key])
            for key in TRANSFER_KEYS}


def load_calibration(path):
    """Read and validate a calibration file. Raise ValueError on any violation."""
    with open(Path(path)) as handle:
        doc = yaml.safe_load(handle)
    _mapping(doc, 'calibration file')
    if doc.get('schema') != SCHEMA:
        raise ValueError(f'unsupported schema {doc.get("schema")!r}, expected {SCHEMA!r}')
    _check_keys(doc, TOP_KEYS, 'calibration file')
    if not isinstance(doc['represents'], str) or not doc['represents']:
        raise ValueError('represents must be a non-empty sentence')
    readings = dict(_mapping(doc['readings'], 'readings'))
    static = _mapping(doc['static'], 'static')
    _check_keys(static, {'rules'}, 'static')
    entries = static['rules']
    if not isinstance(entries, list) or not 1 <= len(entries) <= 2:
        raise ValueError('static.rules must list one or two rules')
    rules = tuple(_rule(entry, i) for i, entry in enumerate(entries))
    names = [rule.name for rule in rules]
    if len(set(names)) != len(names):
        raise ValueError(f'rule names must be unique, got {names}')
    if not isinstance(doc['transfer'], list):
        raise ValueError('transfer must be a list')
    return Calibration(
        represents=doc['represents'], readings=readings, rules=rules,
        time=_anchor(doc['time'], 'latency_scale', 'time'),
        dynamic=_anchor(doc['dynamic'], 'energy_scale', 'dynamic', BRACKET_KEYS),
        transfer=tuple(_transfer(entry) for entry in doc['transfer']))


def _select(calibration, rule):
    if rule is None:
        if len(calibration.rules) > 1:
            raise ValueError('the file carries several rules and one must be named, since a '
                             'single number would hide that the data cannot decide '
                             f'(rules: {[r.name for r in calibration.rules]})')
        return calibration.rules[0]
    for candidate in calibration.rules:
        if candidate.name == rule:
            return candidate
    raise ValueError(f'rule {rule!r} is not in the file '
                     f'(rules: {[r.name for r in calibration.rules]})')


def _power(rule, chips, cores):
    return rule.watts_per_chip * chips + rule.watts_per_core * cores


def static_power(calibration, *, chips, cores, rule=None):
    """Static power in watts for `chips` chips and `cores` used cores.

    `rule` names one of the file's rules. It may be omitted only when the file
    carries exactly one rule.
    """
    chips = _integer(chips, 'chips', 1)
    cores = _integer(cores, 'cores', 0)
    return _power(_select(calibration, rule), chips, cores)


def report(calibration, *, sim_time_s, dynamic_energy_j, chips, cores, samples=1,
           host_time_s=0.0):
    """Turn a modeled time and dynamic energy into the reported terms.

    `sim_time_s` and `dynamic_energy_j` are the modeled totals over `samples`
    inferences, as the simulator reports them, before any scale. The inference
    time is `sim_time_s` times the latency scale, divided by `samples`. The
    dynamic energy of one inference is `dynamic_energy_j` times the energy
    scale, divided by `samples`. Dynamic power is that energy over the
    inference time. For each static rule, total power is static plus dynamic
    power, inference energy is total power times inference time, and total
    energy adds static power times `host_time_s`.

    The single-valued quantities are at the top level. The four rule-dependent
    quantities are under `by_rule`, and also at the top level when the file
    carries exactly one rule.

    When the energy scale is a bracket, `by_energy_scale` gives, under `low`,
    `carried` and `high`, the scale, the dynamic energy and power and the
    rule-dependent quantities at that scale, laid out as at the top level. The
    top level holds the carried scale.
    """
    sim_time_s = _number(sim_time_s, 'sim_time_s', positive=True)
    dynamic_energy_j = _number(dynamic_energy_j, 'dynamic_energy_j')
    host_time_s = _number(host_time_s, 'host_time_s')
    samples = _integer(samples, 'samples', 1)
    chips = _integer(chips, 'chips', 1)
    cores = _integer(cores, 'cores', 0)

    time_s = sim_time_s * calibration.time.scale / samples

    def at_scale(scale):
        energy_j = dynamic_energy_j * scale / samples
        power_dynamic = energy_j / time_s
        by_rule = {}
        for rule in calibration.rules:
            static = _power(rule, chips, cores)
            total = static + power_dynamic
            inference = total * time_s
            by_rule[rule.name] = {'power_static_w': static, 'power_total_w': total,
                                  'energy_inference_j': inference,
                                  'energy_total_j': inference + static * host_time_s}
        level = {'power_dynamic_w': power_dynamic, 'energy_dynamic_j': energy_j,
                 'by_rule': by_rule}
        if len(by_rule) == 1:
            level.update(next(iter(by_rule.values())))
        return level

    dynamic = calibration.dynamic
    carried = at_scale(dynamic.scale)
    scales_dynamic = {'value': dynamic.scale, 'status': dynamic.status}
    out = {
        'inference_time_s': time_s,
        'power_dynamic_w': carried['power_dynamic_w'],
        'energy_dynamic_j': carried['energy_dynamic_j'],
        'samples': samples,
        'chips': chips,
        'cores': cores,
        'host_time_s': host_time_s,
        'by_rule': carried['by_rule'],
        'scales': {
            'time': {'value': calibration.time.scale, 'status': calibration.time.status},
            'dynamic': scales_dynamic,
        },
        'readings': calibration.readings,
        'represents': calibration.represents,
        'transfer': [dict(entry) for entry in calibration.transfer],
    }
    if len(carried['by_rule']) == 1:
        out.update(next(iter(carried['by_rule'].values())))
    if dynamic.low is not None:
        scales_dynamic.update(low=dynamic.low, high=dynamic.high)
        out['by_energy_scale'] = {
            name: {'energy_scale': scale, **(carried if name == 'carried' else at_scale(scale))}
            for name, scale in zip(SCALE_LEVELS, (dynamic.low, dynamic.scale, dynamic.high))}
    return out
