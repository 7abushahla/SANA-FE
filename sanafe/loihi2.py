"""Public-information candidate profile, with explicitly assumed storage costs.

This profile is not a physical-fit certificate or calibrated Loihi 2 simulator.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from importlib.resources import files
from numbers import Integral
from pathlib import Path
import tempfile

INTEL_BRIEF = 'https://www.intel.com/content/dam/www/central-libraries/us/en/documents/neuromorphic-computing-loihi-2-brief.pdf'
RUNTIME_PAPER = 'https://arxiv.org/html/2601.10035v2'
LOIHI1_COSTS = 'https://github.com/SLAM-Lab/SANA-FE/blob/93926ec8019206c1c6e6709448ac4c67f46d57db/sanafe/examples/loihi.yaml'


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


def candidate_profile():
    """Return a fresh JSON-serializable specification and evidence ledger."""
    return {
        'profile': 'loihi2_candidate', 'profile_version': 1,
        'numerical_profile': 'qcfs-if-int24-binary-v1', 'physical_fit_verified': False,
        'topology': {'width': 8, 'height': 4, 'cores_per_tile': 4, 'cores': 128},
        'limits': {'neurons_per_core': 8192, 'total_bytes_per_core': 192 * 1024,
                   'synapse_bytes_per_core': 128 * 1024},
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
        'timing': 'inherited Loihi 1 costs; not calibrated for Loihi 2',
        'energy': 'inherited Loihi 1 costs; not calibrated for Loihi 2',
        'provenance': {
            'limits': {'status': 'public specification', 'source': INTEL_BRIEF,
                       'locator': 'resource comparison table and NeuroCore memory discussion',
                       'units': 'neurons and bytes; KiB = 1024 bytes'},
            'cores': {'status': 'documented 128 cores', 'source': INTEL_BRIEF,
                      'locator': 'Comparison of Loihi to Loihi 2 Resources/Features table', 'units': 'cores'},
            'cores_per_tile': {'status': 'documented four NeuroCores per router', 'source': RUNTIME_PAPER,
                               'locator': 'Section II-1 and Figure 1', 'units': 'cores/router'},
            'mesh_coordinates': {'status': 'inherited 8 by 4 coordinate-shape assumption',
                                 'source': LOIHI1_COSTS, 'locator': 'architecture.attributes.width/height',
                                 'units': 'routers'},
            'semantics': {'status': 'restricted functional candidate; not verified neuron program'},
            'allocation': {'status': 'assumed byte layout'},
            'timing_and_energy': {'status': 'inherited Loihi 1', 'source': LOIHI1_COSTS,
                                  'locator': 'latency_* and energy_* attributes; latency_sync table',
                                  'units': 'seconds and joules'}},
    }


def load_loihi2_candidate():
    """Build the integer candidate while retaining identified Loihi 1 costs."""
    from sanafecpp import load_arch
    text = (files('sanafe.examples') / 'loihi.yaml').read_text()
    text = text.replace('name: loihi_chip', 'name: loihi2_candidate')
    text = text.replace('max_neurons_supported: 1024', 'max_neurons_supported: 8192')
    text = text.replace('model: leaky_integrate_fire', 'model: integrate_fire_int24')
    # Retain the original coefficients of only the selected pipeline units.
    start = text.index('            - name: loihi_inputs')
    end = text.index('          dendrite:', start)
    text = text[:start] + text[end:]
    start = text.index('            - name: loihi_dendrites_delay')
    end = text.index('          synapse:', start)
    text = text[:start] + text[end:]
    start = text.index('            - name: loihi_sparse_synapse')
    end = text.index('          axon_out:', start)
    text = text[:start] + text[end:]
    text = text.replace('model: accumulator\n', 'model: accumulator_int\n')
    text = text.replace('model: current_based\n', 'model: current_based_int\n')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'candidate.yaml'
        path.write_text(text)
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
