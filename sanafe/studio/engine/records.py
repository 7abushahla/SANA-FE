"""One numbered update, converted from a SANA-FE one-step result.

Every field has a provenance mark in ``PROVENANCE``: R is recorded by
SANA-FE, D is derived exactly from records, X is reconstructed.
"""
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
from sanafe.data import performance_to_dataframe

from .layout import xy_path

PROVENANCE = {
    'step_time': 'R', 'energy': 'R', 'counts': 'R', 'messages': 'R',
    'message.path': 'X', 'core_finish': 'D', 'last_activity': 'D',
    'barrier': 'D', 'fired': 'R', 'potentials': 'R', 'core_counts': 'D',
    'core_energy.total': 'R', 'core_energy.units': 'R', 'core_energy.axon': 'D',
    'tile_network_energy': 'D', 'links': 'X', 'sample': 'R',
    'group_fired': 'D', 'readout': 'D',
}
SAMPLE_PAIRS = 256  # aggregate records keep at most this many sample messages
AGGREGATE = 'not kept (aggregate)'


@dataclass(frozen=True)
class MappedNeuronInfo:
    group: str
    offset: int
    core: str
    log_spikes: bool
    log_potential: bool


def neuron_map(chip):
    """Mapped neurons in SANA-FE trace order (group order as stored, then offset)."""
    return tuple(
        MappedNeuronInfo(neuron.group_name, int(neuron.offset),
                         f'{neuron.tile_id}.{neuron.core_offset}',
                         bool(neuron.log_spikes), bool(neuron.log_potential))
        for members in chip.mapped_neuron_groups.values() for neuron in members)


@dataclass
class MessageRecord:
    mid: int
    src: str
    dst: str
    src_neuron: str
    hops: int
    spikes: int
    generation_delay: float
    network_delay: float
    processing_delay: float
    blocking_delay: float
    send: float
    receive: float
    processed: float
    path: list


@dataclass
class UpdateRecord:
    update: int
    step_time: float
    energy: dict
    counts: dict
    messages: list
    core_finish: dict
    last_activity: float
    barrier: float
    fired: list
    potentials: dict
    core_counts: dict
    core_energy: dict
    tile_network_energy: dict
    provenance: dict = field(default_factory=lambda: dict(PROVENANCE))
    reference: Optional[dict] = None
    links: dict = field(default_factory=dict)  # 'a>b' adjacent tiles: packets (X)
    # Aggregate level: the earliest message of each core pair, for animation
    sample: list = field(default_factory=list)
    group_fired: dict = field(default_factory=dict)  # spikes per group (D)
    readout: Optional[dict] = None  # workload readout of this update (D)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data['messages'] = [MessageRecord(**message) for message in data['messages']]
        data['sample'] = [MessageRecord(**message) for message in data.get('sample', [])]
        data['fired'] = [tuple(item) for item in data['fired']]
        data['tile_network_energy'] = {int(tile): value for tile, value
                                       in data['tile_network_energy'].items()}
        return cls(**data)


def _core_key(tile, offset):
    return f'{int(tile)}.{int(offset)}'


class RecordCache:
    """Per-session lookups, so building a record never rescans every neuron."""

    def __init__(self, neurons, layout):
        self.layout = layout
        self.neurons = neurons
        self.keys = [f'{n.group}.{n.offset}' for n in neurons]
        self.index = {(n.group, n.offset): i for i, n in enumerate(neurons)}
        self.core_index = [n.core for n in neurons]
        self.logged = [i for i, n in enumerate(neurons) if n.log_potential]
        self.logged_pos = {self.keys[i]: j for j, i in enumerate(self.logged)}
        self.occupied = sorted({n.core for n in neurons})
        spikes = {}
        for n in neurons:
            spikes[n.core] = spikes.get(n.core, True) and n.log_spikes
        self.complete = spikes  # a core's fired count is exact only if all log spikes
        # Per-group spike counts; exact only for groups whose neurons all log spikes.
        self.groups = list(dict.fromkeys(n.group for n in neurons))
        position = {g: i for i, g in enumerate(self.groups)}
        self.group_id = np.array([position[n.group] for n in neurons], dtype=np.int32)
        logged = {}
        for n in neurons:
            logged[n.group] = logged.get(n.group, True) and n.log_spikes
        self.group_complete = logged
        self._paths = {}

    def group_counts(self, indices):
        """Spikes per group for these neuron indices; None where not all log spikes."""
        counts = np.bincount(self.group_id[np.asarray(indices, dtype=np.int64)],
                             minlength=len(self.groups))
        return {g: (int(counts[i]) if self.group_complete[g] else None)
                for i, g in enumerate(self.groups)}

    def path(self, src, dst):
        if (src, dst) not in self._paths:
            self._paths[(src, dst)] = xy_path(self.layout, src, dst)
        return self._paths[(src, dst)]


def _timing(raw, cache):
    finish = {core: 0.0 for core in cache.occupied}
    for message in raw:
        key = _core_key(message['src_tile_id'], message['src_core_offset'])
        finish[key] = max(finish.get(key, 0.0), float(message['send_timestamp']))
    return finish


def _energy(perf, layout, core_counts):
    core_energy, tile_network_energy = {}, {}
    for tile in layout.tiles:
        tile_total = perf.get(f'{tile.name}.energy')
        cores_in_tile = 0.0
        for core in tile.cores:
            total = perf.get(f'{tile.name}.{core.name}.energy')
            if total is None:
                continue
            cores_in_tile += float(total)
            if core.key not in core_counts and float(total) == 0.0:
                continue
            units = {unit: float(perf[f'{tile.name}.{core.name}.{unit}.energy'])
                     for unit in core.units
                     if f'{tile.name}.{core.name}.{unit}.energy' in perf}
            core_energy[core.key] = {
                'total': float(total), 'units': units,
                'axon': float(total) - sum(units.values()) if units else None}
        if tile_total is not None:
            network = float(tile_total) - cores_in_tile
            if network != 0.0 or any(c.key in core_counts for c in tile.cores):
                tile_network_energy[tile.tile_id] = network
    return core_energy, tile_network_energy


def _chip(perf, messages):
    return ({kind: float(perf[f'{kind}_energy']) for kind in
             ('synapse', 'dendrite', 'soma', 'network', 'total')},
            {'fired': int(perf['fired']), 'updated': int(perf['updated']),
             'spikes': int(perf['spikes']), 'hops': int(perf['hops']),
             'messages': messages})


def _core_counts(cache, fired_cores, real):
    counts = {core: {'fired': 0 if cache.complete[core] else None, 'packets_in': 0,
                     'packets_out': 0, 'spikes_in': 0} for core in cache.occupied}
    for core in fired_cores:
        if counts[core]['fired'] is not None:
            counts[core]['fired'] += 1
    blank = {'fired': 0, 'packets_in': 0, 'packets_out': 0, 'spikes_in': 0}
    for src, dst, spikes in real:
        counts.setdefault(src, dict(blank))['packets_out'] += 1
        entry = counts.setdefault(dst, dict(blank))
        entry['packets_in'] += 1
        entry['spikes_in'] += spikes
    return counts


def _message(m, cache):
    return MessageRecord(
        mid=int(m['mid']),
        src=_core_key(m['src_tile_id'], m['src_core_offset']),
        dst=_core_key(m['dest_tile_id'], m['dest_core_offset']),
        src_neuron=f"{m['src_neuron_group_id']}.{m['src_neuron_offset']}",
        hops=int(m['hops']), spikes=int(m['spikes']),
        generation_delay=float(m['generation_delay']),
        network_delay=float(m['network_delay']),
        processing_delay=float(m['processing_delay']),
        blocking_delay=float(m['blocking_delay']),
        send=float(m['send_timestamp']), receive=float(m['received_timestamp']),
        processed=float(m['processed_timestamp']),
        path=list(cache.path(int(m['src_tile_id']), int(m['dest_tile_id']))))


def build_update_record(update, result, layout, neurons, cache=None):
    """Convert one ``chip.sim(1, ...)`` result with all traces into a record."""
    cache = cache or RecordCache(neurons, layout)
    perf = performance_to_dataframe(result).iloc[-1]
    raw = [message for step in result.get('message_trace', []) for message in step]
    finish = _timing(raw, cache)
    messages = [_message(m, cache) for m in raw if not m['placeholder']]
    last_activity = max([*finish.values(), *(m.processed for m in messages)], default=0.0)
    step_time = float(perf['sim_time'])

    spike_steps = result.get('spike_trace') or [[]]
    fired = [(address.group_name, int(address.neuron_offset)) for address in spike_steps[-1]]
    values = (result.get('potential_trace') or [[]])[-1]
    if len(values) != len(cache.logged):
        raise ValueError(f'potential trace has {len(values)} values for '
                         f'{len(cache.logged)} logged neurons')
    potentials = {cache.keys[i]: float(v) for i, v in zip(cache.logged, values)}

    # The spike trace lists only neurons with log_spikes, so a core's fired
    # count is exact only when all of its neurons log spikes; otherwise None.
    core_counts = _core_counts(cache, [cache.core_index[cache.index[a]] for a in fired],
                               [(m.src, m.dst, m.spikes) for m in messages])
    core_energy, tile_network_energy = _energy(perf, layout, core_counts)
    energy, counts = _chip(perf, len(messages))
    return UpdateRecord(
        update=int(update), step_time=step_time, energy=energy, counts=counts,
        messages=messages, core_finish=finish, last_activity=last_activity,
        barrier=step_time - last_activity, fired=fired, potentials=potentials,
        core_counts=core_counts, core_energy=core_energy,
        tile_network_energy=tile_network_energy,
        group_fired=cache.group_counts([cache.index[a] for a in fired]))


def build_aggregate_record(update, result, layout, cache, watched=()):
    """One update as counts per core and per link, plus the watched neurons.

    The complete membranes and spikes stay on the record as ``state`` (not
    serialized) for reference checks and on-demand core views.
    """
    perf = performance_to_dataframe(result).iloc[-1]
    raw = [message for step in result.get('message_trace', []) for message in step]
    finish = _timing(raw, cache)
    real, links, last, earliest = [], Counter(), 0.0, {}
    for m in raw:
        if m['placeholder']:
            continue
        src_tile, dst_tile = int(m['src_tile_id']), int(m['dest_tile_id'])
        pair = (_core_key(src_tile, m['src_core_offset']),
                _core_key(dst_tile, m['dest_core_offset']))
        real.append((*pair, int(m['spikes'])))
        if pair not in earliest or m['send_timestamp'] < earliest[pair]['send_timestamp']:
            earliest[pair] = m
        path = cache.path(src_tile, dst_tile)
        for a, b in zip(path, path[1:]):
            links[f'{a}>{b}'] += 1
        last = max(last, float(m['processed_timestamp']))
    last_activity = max([*finish.values(), last], default=0.0)
    step_time = float(perf['sim_time'])

    fired_mask = np.zeros(len(cache.keys), dtype=bool)
    for address in (result.get('spike_trace') or [[]])[-1]:
        fired_mask[cache.index[(address.group_name, int(address.neuron_offset))]] = True
    values = np.asarray((result.get('potential_trace') or [[]])[-1], dtype=np.float64)
    if len(values) != len(cache.logged):
        raise ValueError(f'potential trace has {len(values)} values for '
                         f'{len(cache.logged)} logged neurons')
    core_counts = _core_counts(cache, [cache.core_index[i] for i in np.flatnonzero(fired_mask)],
                               real)
    core_energy, tile_network_energy = _energy(perf, layout, core_counts)
    energy, counts = _chip(perf, len(real))
    watched_fired = []
    for key in watched:
        group, _, offset = key.rpartition('.')
        if fired_mask[cache.index[(group, int(offset))]]:
            watched_fired.append((group, int(offset)))
    record = UpdateRecord(
        update=int(update), step_time=step_time, energy=energy, counts=counts,
        messages=[], core_finish=finish, last_activity=last_activity,
        barrier=step_time - last_activity, fired=watched_fired,
        potentials={key: float(values[cache.logged_pos[key]]) for key in watched},
        core_counts=core_counts, core_energy=core_energy,
        tile_network_energy=tile_network_energy, links=dict(links),
        group_fired=cache.group_counts(np.flatnonzero(fired_mask)),
        sample=[_message(m, cache) for m in sorted(
            earliest.values(), key=lambda m: m['send_timestamp'])[:SAMPLE_PAIRS]])
    record.provenance['messages'] = AGGREGATE
    record.state = {'potentials': values, 'fired': fired_mask}
    return record
