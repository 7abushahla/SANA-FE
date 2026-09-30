"""Public-information candidate profile, with explicitly assumed storage costs.

This profile is not a physical-fit certificate or calibrated Loihi 2 simulator.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from importlib.resources import as_file, files
from numbers import Integral
from pathlib import Path

INTEL_BRIEF = 'https://www.intel.com/content/dam/www/central-libraries/us/en/documents/neuromorphic-computing-loihi-2-brief.pdf'
RUNTIME_PAPER = 'https://arxiv.org/html/2601.10035v2'
LOIHI1_COSTS = 'https://github.com/SLAM-Lab/SANA-FE/blob/93926ec8019206c1c6e6709448ac4c67f46d57db/sanafe/examples/loihi.yaml'

# Latency scaling from the fitted Loihi 1 file to the candidate, decided
# 2026-09-30. Factors are Intel's stated Loihi 2 over Loihi 1 speed-ups
# (technology brief, "Faster circuit speeds" and footnote 2: Loihi 1 from
# silicon characterization on Nahuku-32, Loihi 2 from N3B1 silicon and
# pre-silicon circuit simulation). The barrier table is scaled so that its
# chip-wide entry equals the stated "minimum chip-wide time steps under
# 200ns". Everything not listed here is inherited from Loihi 1 unchanged.
LATENCY_SCALING = {
    'latency_access_neuron': (2.0, 'neuron state update 2x'),
    'latency_update_neuron': (2.0, 'neuron state update 2x'),
    'latency_process_spike': (5.0, 'synaptic operation 5x'),
    'latency_spike_out': (10.0, 'spike generation 10x'),
}
SYNC_SCALING = (9.0, 'chip-wide barrier table scaled to the stated 200 ns floor '
                     '(fitted Loihi 1 chip-wide entry 1.8 us / 9)')
INHERITED_COSTS = ('energy_access_neuron', 'energy_update_neuron', 'energy_spike_out',
                   'energy_process_spike', 'energy_message_in', 'latency_message_in',
                   'energy_message_out', 'latency_message_out', 'energy_update',
                   'latency_update', 'energy_north_hop', 'latency_north_hop',
                   'energy_east_hop', 'latency_east_hop', 'energy_south_hop',
                   'latency_south_hop', 'energy_west_hop', 'latency_west_hop',
                   'link_buffer_size')


def candidate_cost_provenance():
    """Per-attribute origin of every cost coefficient in the candidate YAML."""
    scaled = {name: {'status': 'scaled', 'factor': factor, 'basis': basis,
                     'base': 'fitted Loihi 1 (Nahuku) value in loihi.yaml',
                     'source': INTEL_BRIEF}
              for name, (factor, basis) in LATENCY_SCALING.items()}
    scaled['latency_sync'] = {'status': 'scaled', 'factor': SYNC_SCALING[0],
                              'basis': SYNC_SCALING[1],
                              'base': 'fitted Loihi 1 (Nahuku) table in loihi.yaml',
                              'source': INTEL_BRIEF}
    inherited = {name: {'status': 'inherited', 'base': 'fitted Loihi 1 (Nahuku) value in loihi.yaml',
                        'source': LOIHI1_COSTS,
                        'reason': 'no public Loihi 2 datum scales this coefficient'}
                 for name in INHERITED_COSTS}
    return {**scaled, **inherited}


def _integer(value, name, minimum, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer, excluding bool')
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name} outside supported range')
    return value


@dataclass(frozen=True)
class Allocation:
    """Assumed allocation sizes in bytes, not Intel packing specifications."""
    state_bytes_per_neuron: int = 16
    synapse_bytes: int = 4
    incoming_axon_bytes: int = 8
    outgoing_route_bytes: int = 8
    program_bytes_per_core: int = 256

    def __post_init__(self):
        for key, value in asdict(self).items():
            object.__setattr__(self, key, _integer(value, key, 0 if key == 'program_bytes_per_core' else 1))


def audit_unit_scale_signed8_weights(matrices):
    """Check direct signed8 values under an explicit unit-scale hypothesis.

    This is a software representability audit, not a physical Loihi 2 mapping
    check. Hardware exponents, parallel synapses, rounding, and packing are
    outside the test. Each matrix is indexed by destination then source.
    """
    try:
        matrix_list = list(matrices)
        details = []
        for matrix in matrix_list:
            rows = list(matrix)
            if not rows:
                raise ValueError('Weight matrices must have a nonempty row')
            width = None
            values = []
            for row in rows:
                entries = list(row)
                if not entries or (width is not None and len(entries) != width):
                    raise ValueError('Weight matrices must be rectangular and nonempty')
                width = len(entries)
                values.extend(_integer(weight, 'weight', -(1 << 15), (1 << 15)-1)
                              for weight in entries)
            details.append({'shape': [len(rows), width], 'entries': len(values),
                            'nonzero': sum(value != 0 for value in values),
                            'out_of_range': sum(value < -128 or value > 127
                                                for value in values),
                            'minimum': min(values), 'maximum': max(values)})
    except TypeError as exc:
        raise ValueError('Weight matrices must be iterable') from exc
    return {'hypothesis': 'one signed8 stored value per nonzero weight at unit scale',
            'range': [-128, 127], 'physical_mapping_verified': False,
            'matrices': details, 'entries': sum(item['entries'] for item in details),
            'nonzero': sum(item['nonzero'] for item in details),
            'out_of_range': sum(item['out_of_range'] for item in details)}


def candidate_profile():
    """Return a fresh JSON-serializable specification and evidence ledger."""
    return {
        'profile': 'loihi2_candidate', 'profile_version': 1,
        'numerical_profile': 'qcfs-if-int24-binary-v1', 'physical_fit_verified': False,
        'topology': {'width': 8, 'height': 4, 'cores_per_tile': 4, 'cores': 128},
        'limits': {'neurons_per_core': 8192, 'total_bytes_per_core': 192 * 1024,
                   'synapse_bytes_per_core': 128 * 1024,
                   'native_stored_weight_bits_max': 8},
        'semantics': {'input_bits': 16, 'input_signed': True, 'bias_bits': 16,
                      'effective_weight_transport_bits': 16, 'weight_signed': True,
                      'voltage_bits': 24, 'voltage_signed': True,
                      'clip_before_threshold': True, 'threshold_comparison': '>=',
                      'reset': 'single threshold subtraction',
                      'accumulation_bits': 32, 'accumulation_signed': True,
                      'accumulation_overflow': 'reject', 'events': 'binary',
                      'edge_delay_steps': 1},
        'allocation': asdict(Allocation()),
        'storage_layout': 'assumed; not verified Intel allocation or packing',
        'weight_mapping': 'signed16 effective software weights are not certified as '
                          'one native stored synapse; wider values may require '
                          'multiple 8-bit synapses and change workload',
        'timing': 'fitted Loihi 1 latencies scaled by Intel-stated Loihi 2 factors '
                  '(neuron 2x, synapse 5x, spike 10x, barrier to a 200 ns chip-wide '
                  'floor); hop and message latencies inherited; not calibrated for Loihi 2',
        'energy': 'inherited Loihi 1 costs; no public Loihi 2 datum; not calibrated for Loihi 2',
        'cost_provenance': candidate_cost_provenance(),
        'provenance': {
            'limits': {'status': 'public specification', 'source': INTEL_BRIEF,
                       'locator': 'resource comparison table and NeuroCore memory discussion',
                       'units': 'neurons and bytes; KiB = 1024 bytes'},
            'cores': {'status': 'documented 128 cores', 'source': INTEL_BRIEF,
                      'locator': 'Comparison of Loihi to Loihi 2 Resources/Features table', 'units': 'cores'},
            'cores_per_tile': {'status': 'documented four NeuroCores per router', 'source': RUNTIME_PAPER,
                               'locator': 'Section II-1 and Figure 1', 'units': 'cores/router'},
            'mesh_coordinates': {'status': 'documented 8 by 4 router grid; tile address mapping assumed',
                                 'source': RUNTIME_PAPER, 'locator': 'Section VI, paragraph on n=8 and m=4',
                                 'units': 'routers'},
            'native_weight_bits': {'status': 'documented maximum stored weight precision',
                                   'source': RUNTIME_PAPER, 'locator': 'Section III-5',
                                   'units': 'bits per native synapse'},
            'semantics': {'status': 'restricted functional candidate; not verified neuron program'},
            'allocation': {'status': 'assumed byte layout'},
            'timing': {'status': 'scaled from fitted Loihi 1 where Intel states a factor; '
                                 'inherited elsewhere', 'source': INTEL_BRIEF,
                       'locator': '"Faster circuit speeds" and footnote 2; see cost_provenance',
                       'units': 'seconds'},
            'energy': {'status': 'inherited Loihi 1', 'source': LOIHI1_COSTS,
                       'locator': 'energy_* attributes', 'units': 'joules'}},
    }


def load_loihi2_candidate():
    """Load the packaged candidate architecture (Loihi 1 costs, latencies scaled)."""
    from sanafecpp import load_arch
    with as_file(files('sanafe.examples') / 'loihi2.yaml') as path:
        return load_arch(str(path))


def _placement(value):
    try:
        pair = tuple(value)
    except TypeError as exc:
        raise ValueError('Placement must be a (tile, core) pair') from exc
    if len(pair) != 2:
        raise ValueError('Placement must be a (tile, core) pair')
    return (_integer(pair[0], 'tile', 0, 31), _integer(pair[1], 'core', 0, 3))


def validate_chain_resources(layer_sizes, weights, placements, *, reserved_cores=(), allocation=None):
    """Validate dense-chain storage under an assumed layout; raise ValueError.

    Weights are signed-16 integer matrices indexed [destination, source].
    All dense entries, including zeros, occupy storage. Layer placement is
    atomic, but multiple layers may share a core. Route counts include self.
    """
    allocation = Allocation() if allocation is None else allocation
    if not isinstance(allocation, Allocation):
        raise ValueError('allocation must be an Allocation')
    try:
        sizes = [_integer(n, 'layer size', 1) for n in layer_sizes]
        places = [_placement(p) for p in placements]
        matrices = list(weights)
        reserved = {_placement(p) for p in reserved_cores}
    except TypeError as exc:
        raise ValueError('Sizes, weights, placements and reservations must be iterable') from exc
    if not sizes or len(places) != len(sizes) or len(matrices) != len(sizes) - 1:
        raise ValueError('Expected nonempty layers, one placement per layer and one weight matrix per edge')
    cores = {}
    for index, (size, place) in enumerate(zip(sizes, places)):
        if place in reserved:
            raise ValueError(f'Core {place} is reserved')
        core = cores.setdefault(place, {'tile': place[0], 'core': place[1], 'layers': [],
                                       'neurons': 0, 'synapses': 0, 'incoming_axons': 0, 'outgoing_routes': 0})
        core['layers'].append(index)
        core['neurons'] += size
    for index, matrix in enumerate(matrices):
        try:
            rows = list(matrix)
            if len(rows) != sizes[index + 1]:
                raise ValueError('Weight matrix has incorrect destination dimension')
            for row in rows:
                entries = list(row)
                if len(entries) != sizes[index]:
                    raise ValueError('Weight matrix has incorrect source dimension')
                for weight in entries:
                    _integer(weight, 'weight', -(1 << 15), (1 << 15) - 1)
        except TypeError as exc:
            raise ValueError('Weights must be integer matrices') from exc
        # A chain source neuron has one following layer, hence one destination
        # core. Distinct layer indices retain distinct identities when colocated.
        cores[places[index + 1]]['synapses'] += sizes[index] * sizes[index + 1]
        cores[places[index + 1]]['incoming_axons'] += sizes[index]
        cores[places[index]]['outgoing_routes'] += sizes[index]
    for place, core in cores.items():
        core.update(state_bytes=core['neurons'] * allocation.state_bytes_per_neuron,
                    synapse_bytes=core['synapses'] * allocation.synapse_bytes,
                    incoming_axon_bytes=core['incoming_axons'] * allocation.incoming_axon_bytes,
                    outgoing_route_bytes=core['outgoing_routes'] * allocation.outgoing_route_bytes,
                    program_bytes=allocation.program_bytes_per_core)
        core['total_bytes'] = sum(core[k] for k in ('state_bytes', 'synapse_bytes', 'incoming_axon_bytes', 'outgoing_route_bytes', 'program_bytes'))
        if core['neurons'] > 8192:
            raise ValueError(f'Core {place} exceeds neuron limit')
        if core['synapse_bytes'] > 128 * 1024:
            raise ValueError(f'Core {place} exceeds synapse memory limit under assumed layout')
        if core['total_bytes'] > 192 * 1024:
            raise ValueError(f'Core {place} exceeds total memory limit under assumed layout')
    return {'profile': 'loihi2_candidate', 'feasibility': 'model-feasible under assumed layout',
            'physical_fit_verified': False, 'allocation': asdict(allocation),
            'cores': [cores[p] for p in sorted(cores)]}


def validate_core_budgets(core_stats, *, allocation=None):
    """Check per-core counts of a chunked mapping; raise ValueError.

    Each entry gives ``tile``, ``core``, ``neurons``, ``edges`` (synapses
    stored at the core) and ``outgoing_neurons`` (neurons with at least one
    outgoing edge). Returns new entries with the assumed synapse and total
    bytes added. Unlike ``validate_chain_resources``, only emitted edges
    occupy storage.
    """
    allocation = Allocation() if allocation is None else allocation
    if not isinstance(allocation, Allocation):
        raise ValueError('allocation must be an Allocation')
    checked = []
    for stat in core_stats:
        place = (stat['tile'], stat['core'])
        neurons = _integer(stat['neurons'], 'neurons', 0)
        edges = _integer(stat['edges'], 'edges', 0)
        outgoing = _integer(stat['outgoing_neurons'], 'outgoing_neurons', 0)
        entry = dict(stat)
        entry['assumed_synapse_bytes'] = edges * allocation.synapse_bytes
        entry['assumed_total_bytes'] = (
            neurons * allocation.state_bytes_per_neuron +
            edges * (allocation.synapse_bytes + allocation.incoming_axon_bytes) +
            outgoing * allocation.outgoing_route_bytes +
            allocation.program_bytes_per_core)
        if neurons > 8192:
            raise ValueError(f'core {place} exceeds the neuron limit: {neurons} > 8192')
        if entry['assumed_synapse_bytes'] > 128 * 1024:
            raise ValueError(f'core {place} exceeds the synapse memory limit under the '
                             f'assumed layout: {entry["assumed_synapse_bytes"]} > 131072 bytes')
        if entry['assumed_total_bytes'] > 192 * 1024:
            raise ValueError(f'core {place} exceeds the total memory limit under the '
                             f'assumed layout: {entry["assumed_total_bytes"]} > 196608 bytes')
        checked.append(entry)
    return checked


def architecture_fingerprint(arch):
    """Hash actual topology, buffering, units, parameters, and cost coefficients."""
    data = json.dumps(arch.configuration(), sort_keys=True, separators=(',', ':'),
                      allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


@lru_cache(maxsize=1)
def _candidate_fingerprint():
    return architecture_fingerprint(load_loihi2_candidate())


def validate_candidate_architecture(arch):
    """Reject a changed architecture before attaching candidate provenance."""
    fingerprint = architecture_fingerprint(arch)
    if fingerprint != _candidate_fingerprint():
        raise ValueError('Unsupported architecture configuration for this candidate profile')
    return fingerprint
