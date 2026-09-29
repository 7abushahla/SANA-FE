"""One numbered update, converted from a SANA-FE one-step result.

Every field has a provenance mark in ``PROVENANCE``: R is recorded by
SANA-FE, D is derived exactly from records, X is reconstructed.
"""
from dataclasses import asdict, dataclass, field
from typing import Optional

from sanafe.data import performance_to_dataframe

from .layout import xy_path

PROVENANCE = {
    'step_time': 'R', 'energy': 'R', 'counts': 'R', 'messages': 'R',
    'message.path': 'X', 'core_finish': 'D', 'last_activity': 'D',
    'barrier': 'D', 'fired': 'R', 'potentials': 'R', 'core_counts': 'D',
    'core_energy.total': 'R', 'core_energy.units': 'R', 'core_energy.axon': 'D',
    'tile_network_energy': 'D',
}


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

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data['messages'] = [MessageRecord(**message) for message in data['messages']]
        data['fired'] = [tuple(item) for item in data['fired']]
        data['tile_network_energy'] = {int(tile): value for tile, value
                                       in data['tile_network_energy'].items()}
        return cls(**data)


def _core_key(tile, offset):
    return f'{int(tile)}.{int(offset)}'


def build_update_record(update, result, layout, neurons):
    """Convert one ``chip.sim(1, ...)`` result with all traces into a record."""
    perf = performance_to_dataframe(result).iloc[-1]
    raw = [message for step in result.get('message_trace', []) for message in step]
    occupied = sorted({neuron.core for neuron in neurons})

    finish = {core: 0.0 for core in occupied}
    for message in raw:
        key = _core_key(message['src_tile_id'], message['src_core_offset'])
        finish[key] = max(finish.get(key, 0.0), float(message['send_timestamp']))

    messages = [
        MessageRecord(
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
            path=xy_path(layout, int(m['src_tile_id']), int(m['dest_tile_id'])))
        for m in raw if not m['placeholder']]

    last_activity = max([*finish.values(), *(m.processed for m in messages)],
                        default=0.0)
    step_time = float(perf['sim_time'])

    spike_steps = result.get('spike_trace') or [[]]
    fired = [(address.group_name, int(address.neuron_offset))
             for address in spike_steps[-1]]

    logged = [neuron for neuron in neurons if neuron.log_potential]
    values = (result.get('potential_trace') or [[]])[-1]
    if len(values) != len(logged):
        raise ValueError(f'potential trace has {len(values)} values for '
                         f'{len(logged)} logged neurons')
    potentials = {f'{n.group}.{n.offset}': float(v) for n, v in zip(logged, values)}

    # The spike trace lists only neurons with log_spikes, so a core's fired
    # count is exact only when all of its neurons log spikes; otherwise None.
    where = {(n.group, n.offset): n.core for n in neurons}
    complete = {core: all(n.log_spikes for n in neurons if n.core == core)
                for core in occupied}
    core_counts = {core: {'fired': 0 if complete[core] else None, 'packets_in': 0,
                          'packets_out': 0, 'spikes_in': 0} for core in occupied}
    for address in fired:
        counts = core_counts[where[address]]
        if counts['fired'] is not None:
            counts['fired'] += 1
    for message in messages:
        core_counts.setdefault(message.src, {'fired': 0, 'packets_in': 0,
                                             'packets_out': 0, 'spikes_in': 0})
        core_counts.setdefault(message.dst, {'fired': 0, 'packets_in': 0,
                                             'packets_out': 0, 'spikes_in': 0})
        core_counts[message.src]['packets_out'] += 1
        core_counts[message.dst]['packets_in'] += 1
        core_counts[message.dst]['spikes_in'] += message.spikes

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

    return UpdateRecord(
        update=int(update), step_time=step_time,
        energy={kind: float(perf[f'{kind}_energy']) for kind in
                ('synapse', 'dendrite', 'soma', 'network', 'total')},
        counts={'fired': int(perf['fired']), 'updated': int(perf['updated']),
                'spikes': int(perf['spikes']), 'hops': int(perf['hops']),
                'messages': len(messages)},
        messages=messages, core_finish=finish, last_activity=last_activity,
        barrier=step_time - last_activity, fired=fired, potentials=potentials,
        core_counts=core_counts, core_energy=core_energy,
        tile_network_energy=tile_network_energy)
