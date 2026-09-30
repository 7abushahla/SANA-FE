"""The platform catalog: what each bundled architecture is, what SANA-FE models
of it, and where every cost coefficient comes from.

Curated facts (titles, provenance, validation, omissions, references) live
here. Structural facts (mesh, cores, units, models, accepted attributes, cost
values) are read from the loaded architecture so they cannot drift from the
YAML. Nothing here is a hardware measurement.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
import re
import tempfile

import yaml

import sanafe
from sanafe.loihi2 import (INTEL_BRIEF, LOIHI1_COSTS, RUNTIME_PAPER, architecture_fingerprint,
                           candidate_cost_provenance)

STATUSES = ('fitted', 'scaled', 'inherited', 'documented', 'none', 'planned')
NOT_HARDWARE = ('Nothing on this card is a hardware measurement: structure and values are '
                'read from the architecture file SANA-FE simulates.')

DAVIES_2018 = ('M. Davies et al., "Loihi: A neuromorphic manycore processor with on-chip '
               'learning," IEEE Micro, vol. 38, no. 1, pp. 82-99, 2018, Table 2. '
               'doi:10.1109/MM.2018.112130359')
TCAD_2025 = ('J. A. Boyle, M. Plagge, S. G. Cardwell, F. S. Chance, and A. Gerstlauer, '
             '"SANA-FE: Simulating advanced neuromorphic architectures for fast exploration," '
             'IEEE Trans. Comput.-Aided Design Integr. Circuits Syst., vol. 44, no. 8, '
             'pp. 3165-3178, 2025, Sec. VI and Table IV. doi:10.1109/TCAD.2025.3537971')
INTEL_BRIEF_REF = ('Intel Corporation, "Taking neuromorphic computing to the next level with '
                   'Loihi 2," technology brief, 2021, Table 2 and "Faster circuit speeds". '
                   + INTEL_BRIEF)
TIMCHECK_2026 = ('J. Timcheck, A. Pierro, and S. B. Shrestha, "A compute and communication '
                 'runtime model for Loihi 2," arXiv:2601.10035v2, 2026. ' + RUNTIME_PAPER)
MEROLLA_2014 = ('P. A. Merolla et al., "A million spiking-neuron integrated circuit with a '
                'scalable communication network and interface," Science, vol. 345, no. 6197, '
                'pp. 668-673, 2014, Fig. 4 and supplementary S1, S7. doi:10.1126/science.1254642')
CASSIDY_2013 = ('A. S. Cassidy et al., "Cognitive computing building block: A versatile and '
                'efficient digital neuron model for neurosynaptic cores," in Proc. IJCNN, 2013. '
                'doi:10.1109/IJCNN.2013.6707077')
AUDIT_NOTE = ('Thesis vault, "2026-09-30 03 SANA-FE Platform Basis Audit" (source audit of '
              'these profiles).')
RICHTER_2024 = ('O. Richter et al., "Speck: A smart event-based vision sensor with a low latency '
                '327K neuron convolutional neuronal network processing pipeline," Neuromorphic '
                'Comput. Eng., vol. 4, 2024. doi:10.1088/2634-4386/ad2ce3')
YAO_2024 = ('M. Yao et al., "Spike-based dynamic computing with asynchronous sensing-computing '
            'neuromorphic chip," Nature Communications, vol. 15, 4464, 2024, and supplement. '
            'doi:10.1038/s41467-024-47811-6')
SPECK_PLAN = ('Thesis vault, "2026-09-30 02 Speck Event Engine Feasibility and Plan" (chip facts, '
              'engine design, calibration protocol, staged plan).')

# Cost attribute names as they appear in architecture YAML files.
NOC_COSTS = ('link_buffer_size', 'latency_sync')
HOP_COSTS = ('energy_north_hop', 'latency_north_hop', 'energy_east_hop', 'latency_east_hop',
             'energy_south_hop', 'latency_south_hop', 'energy_west_hop', 'latency_west_hop')
AXON_IN_COSTS = ('energy_message_in', 'latency_message_in')
AXON_OUT_COSTS = ('energy_message_out', 'latency_message_out')
UNIT_COSTS = ('energy_process_spike', 'latency_process_spike', 'energy_update', 'latency_update',
              'energy_access_neuron', 'latency_access_neuron', 'energy_update_neuron',
              'latency_update_neuron', 'energy_spike_out', 'latency_spike_out')
COST_ATTRIBUTES = frozenset(NOC_COSTS + HOP_COSTS + AXON_IN_COSTS + AXON_OUT_COSTS + UNIT_COSTS)
ALL_COST_NAMES = HOP_COSTS + AXON_IN_COSTS + AXON_OUT_COSTS + UNIT_COSTS + NOC_COSTS


@dataclass(frozen=True)
class Platform:
    id: str
    title: str
    vendor: str
    generation: str
    yaml_name: str
    execution: str        # 'timestep' now; 'event' reserved for Speck
    time_rule: str        # one sentence on how modeled time advances
    summary: str
    cost_summary: str     # the middle of the badge
    costs: dict           # attribute, or 'unit.attribute' override -> {'status', 'source', 'note', ['factor']}
    time_word: str        # provenance word shown beside modeled times
    energy_word: str      # provenance word shown beside modeled energies
    validation: str
    not_modeled: tuple
    references: tuple
    demo_neuron: object = field(compare=False)   # (kind, rng) -> soma attributes
    demo_weight: object = field(compare=False)   # rng -> weight value
    preview: bool = False          # card and picture only: no engine can run it
    preview_note: str = ''
    preview_layout: dict = field(default=None, compare=False)  # blocks and cores to draw

    @property
    def arch_yaml(self):
        return Path(str(files('sanafe.examples') / self.yaml_name))


def _status(status, source, note='', **extra):
    return {'status': status, 'source': source, 'note': note, **extra}


def _all(status, source, note=''):
    return {name: _status(status, source, note) for name in ALL_COST_NAMES}


# --- Loihi 1 ----------------------------------------------------------------
_loihi_costs = _all('fitted', LOIHI1_COSTS,
                    'fitted upstream against Nahuku measurements (TCAD 2025, Sec. VI); '
                    'the paper does not print the value')
for _name in ('latency_north_hop', 'latency_south_hop', 'latency_east_hop', 'latency_west_hop',
              'energy_east_hop', 'energy_west_hop'):
    _loihi_costs[_name] = _status('documented', DAVIES_2018, 'matches Davies et al. 2018, Table 2')
# The loihi_inputs unit (model: input) is a zero-cost placeholder for spike
# sources, not a fitted unit.
for _name in ('energy_access_neuron', 'latency_access_neuron', 'energy_update_neuron',
              'latency_update_neuron', 'energy_spike_out', 'latency_spike_out'):
    _loihi_costs[f'loihi_inputs.{_name}'] = _status(
        'none', LOIHI1_COSTS, 'input placeholder unit (model: input): zero by construction, not fitted')

# Drive neurons: one draw n in 1..20 per neuron, the same draw on every
# platform, and integer state so each fires exactly every n updates (no
# rounding or 1/64 truncation can stretch the period).


def _loihi_neuron(kind, rng):
    if kind == 'input':  # potential k after k updates; fires when k > n - 0.5
        return {'threshold': rng.randint(1, 20) - 0.5, 'leak_decay': 1.0,
                'reset_mode': 'hard', 'bias': 1.0}
    return {'threshold': 1.0, 'leak_decay': 1.0, 'reset_mode': 'hard'}


def _float_weight(rng):
    return round(rng.uniform(-0.5, 1.0), 3)


# --- Loihi 2 candidate ------------------------------------------------------
def _loihi2_costs():
    base = candidate_cost_provenance()
    costs = {}
    for name in ALL_COST_NAMES:
        entry = base[name]
        if entry['status'] == 'scaled':
            costs[name] = _status('scaled', INTEL_BRIEF, entry['basis'], factor=entry['factor'])
        else:
            costs[name] = _status('inherited', LOIHI1_COSTS, entry['reason'])
    return costs


def _loihi2_neuron(kind, rng):
    if kind == 'input':  # potential 2k; fires when 2k >= 2n (thresholds must be even)
        return {'threshold': 2 * rng.randint(1, 20), 'bias': 2}
    return {'threshold': 20}


def _int_weight(rng):
    return rng.randint(-10, 20)


# --- TrueNorth --------------------------------------------------------------
def _truenorth_neuron(kind, rng):
    if kind == 'input':  # potential k; fires when k >= n
        return {'threshold': float(rng.randint(1, 20)), 'leak': 0.0,
                'reset_mode': 'hard', 'bias': 1.0}
    return {'threshold': 1.0, 'leak': 0.0, 'reset_mode': 'hard'}


_truenorth_costs = _all('none', AUDIT_NOTE, 'the shipped file sets every cost to zero; '
                        'upstream validated spike traces against NeMo only')
_truenorth_documented_costs = _all('none', AUDIT_NOTE, 'no public per-operation figure')
_truenorth_documented_costs['energy_process_spike'] = _status(
    'documented', MEROLLA_2014, '26 pJ per synaptic event: total energy at the 20 Hz, '
    '128-active-synapse operating point (72 mW at 0.775 V); includes static power')
_truenorth_documented_costs['latency_sync'] = _status(
    'documented', MEROLLA_2014, 'the 1 ms global tick; every processing latency is zero, '
    'so the modeled step time equals the tick')

_TIMESTEP_RULE = ('Numbered updates. Each core processes its neurons and messages, then the '
                  'chip synchronizes; SANA-FE adds the synchronization cost after the '
                  'scheduled step time.')

# --- Speck (preview) -----------------------------------------------------------
_SPECK_CORES = [(0, 16, 64), (1, 16, 64), (2, 16, 64), (3, 32, 32), (4, 32, 32),
                (5, 64, 16), (6, 64, 16), (7, 16, 16), (8, 16, 16)]  # id, kernel Ki, neuron Ki
_SPECK_LAYOUT = {
    'blocks': [
        {'id': 'dvs', 'label': 'DVS 128 × 128', 'sub': 'event pixels on the die'},
        {'id': 'preprocess', 'label': 'event pre-processing', 'sub': 'filter · decimation · pooling'},
        {'id': 'noc', 'label': 'star NoC', 'sub': 'one router · nine sCNN cores · fan-out 2'},
        {'id': 'readout', 'label': 'readout', 'sub': '15 classes · slow-clock average'},
    ],
    'cores': [{'id': i, 'kernel_words': k * 1024, 'neuron_words': n * 1024, 'leak_words': 1024,
               'label': f'core {i}', 'memory': f'{n} Ki neur · {k} Ki kern'}
              for i, k, n in _SPECK_CORES],
}
_SPECK_NOTE = ('Speck preview: no execution engine yet. The picture and the card come from the '
               'published architecture; nothing can be started on this platform. The event '
               'engine is planned (see the plan note).')

PLATFORMS = (
    Platform(
        id='loihi', title='Intel Loihi 1', vendor='Intel', generation='first generation, 2018',
        yaml_name='loihi.yaml', execution='timestep', time_rule=_TIMESTEP_RULE,
        summary=('128 neuromorphic cores in an 8 x 4 mesh of four-core tiles, 1,024 '
                 'compartments per core, binary spikes, barrier synchronization per '
                 'algorithmic time-step. SANA-FE runs a floating-point leaky '
                 'integrate-and-fire soma with a 1/64 truncation, not the chip\'s '
                 'fixed-point compartment arithmetic.'),
        cost_summary='costs fitted on Nahuku (TCAD 2025, within 11.7% energy and 24.3% latency)',
        costs=_loihi_costs, time_word='fitted', energy_word='fitted',
        validation=('Upstream calibrated this file against Intel\'s Nahuku board with '
                    'micro-benchmarks and reports average absolute errors within 11.7% for '
                    'dynamic energy and 24.3% for time-step latency over its benchmark '
                    'applications (TCAD 2025, Table IV); DVS-gesture latency is consistently '
                    'underestimated. Static power is not modeled. Functional check: spike '
                    'traces against Loihi.'),
        not_modeled=('Loihi\'s 12-bit decay and 24-bit fixed-point compartment state',
                     'dendritic compartment trees and join functions', 'learning engines',
                     'embedded x86 cores and host input/output', 'static power',
                     'the two physical mesh fabrics (one effective mesh)'),
        references=(DAVIES_2018, TCAD_2025, AUDIT_NOTE),
        demo_neuron=_loihi_neuron, demo_weight=_float_weight),
    Platform(
        id='loihi2', title='Intel Loihi 2 candidate', vendor='Intel',
        generation='second generation, 2021; candidate profile', yaml_name='loihi2.yaml',
        execution='timestep', time_rule=_TIMESTEP_RULE,
        summary=('128 neurocores as 8 x 4 routers with four cores each, dimension-order '
                 'routing, up to 8,192 neurons and 192 KB per core. SANA-FE runs a '
                 'signed 24-bit integer integrate-and-fire soma with integer synapses and '
                 'accumulators as a candidate neuron program; graded spikes and microcode '
                 'are not modeled.'),
        cost_summary=('latencies scaled from Loihi 1 by Intel-stated factors · energy '
                      'inherited from Loihi 1'),
        costs=_loihi2_costs(), time_word='scaled', energy_word='inherited',
        validation=('No Loihi 2 measurement. Latencies are the fitted Loihi 1 values scaled '
                    'by Intel\'s stated factors (neuron update 2x, synaptic operation 5x, '
                    'spike generation 10x, chip-wide step floor 200 ns); energies, hop and '
                    'message costs are the Loihi 1 values. Scaled estimates, not calibrated '
                    'predictions.'),
        not_modeled=('graded (integer-payload) spikes', 'programmable neuron microcode',
                     'learning engines', 'the two physical mesh fabrics',
                     'embedded processors and host input/output', 'static power',
                     'Loihi 2 per-operation energy (no public figure exists)'),
        references=(INTEL_BRIEF_REF, TIMCHECK_2026, TCAD_2025, AUDIT_NOTE),
        demo_neuron=_loihi2_neuron, demo_weight=_int_weight),
    Platform(
        id='truenorth', title='IBM TrueNorth (functional)', vendor='IBM',
        generation='2014', yaml_name='truenorth.yaml', execution='timestep',
        time_rule=('Numbered updates with a fixed zero synchronization cost and zero '
                   'processing costs: every modeled step time is zero.'),
        summary=('4,096 single-core tiles in a 64 x 64 mesh, 256 neurons per core. SANA-FE '
                 'runs the deterministic subset of the TrueNorth neuron: additive leak, '
                 'bias, positive and negative thresholds with hard, linear or saturating '
                 'resets, and a random-mask threshold. The shipped file carries no costs.'),
        cost_summary='no cost model: energy and time are zero',
        costs=_truenorth_costs, time_word='none', energy_word='none',
        validation=('Upstream compared spike traces with the NeMo simulator and measured '
                    'simulator speed only; no TrueNorth energy or latency was claimed '
                    '(TCAD 2025).'),
        not_modeled=('the four axon types with per-neuron signed 9-bit weights (SANA-FE '
                     'stores a weight per synapse)', 'stochastic synapse and leak modes',
                     '1 to 15 tick axonal delays (plain accumulator in this file)',
                     'the 256 x 256 binary crossbar fan-in limit', 'the 1 ms tick',
                     'any energy or latency'),
        references=(MEROLLA_2014, CASSIDY_2013, TCAD_2025, AUDIT_NOTE),
        demo_neuron=_truenorth_neuron, demo_weight=_float_weight),
    Platform(
        id='truenorth_documented', title='IBM TrueNorth (documented costs)', vendor='IBM',
        generation='2014', yaml_name='truenorth_documented.yaml', execution='timestep',
        time_rule=('Numbered updates on a fixed 1 ms tick: every processing latency is '
                   'zero, so each modeled step time equals the tick.'),
        summary=('The functional TrueNorth profile plus the two published figures that '
                 'exist: 26 pJ per synaptic event and the 1 ms tick. The energy figure is a '
                 'whole-chip operating point that includes static power, not a per-operation '
                 'fit.'),
        cost_summary='26 pJ per synaptic event and a 1 ms tick from published figures',
        costs=_truenorth_documented_costs, time_word='documented', energy_word='documented',
        validation=('Same functional validation as the shipped profile (spike traces against '
                    'NeMo). Costs are published figures at one operating point, not fitted '
                    'and not validated against a chip by SANA-FE.'),
        not_modeled=('everything the functional profile omits', 'neuron-update, hop and '
                     'message energy (no public figures)', 'static power as a separate term'),
        references=(MEROLLA_2014, CASSIDY_2013, TCAD_2025, AUDIT_NOTE),
        demo_neuron=_truenorth_neuron, demo_weight=_float_weight),
    Platform(
        id='speck', title='SynSense Speck (preview)', vendor='SynSense',
        generation='Speck2f dev-kit generation; architecture as published for Speck1',
        yaml_name='speck.yaml', execution='event',
        time_rule=('No global time-step. Events move between blocks as AER packets on '
                   'four-phase QDI handshakes and each core processes them on arrival; only '
                   'the readout back end is clocked. SANA-FE\'s engine is time-stepped, so this '
                   'entry is a picture and a card, not something that runs.'),
        summary=('One 65 nm die with a 128 × 128 DVS, an event pre-processing core, a '
                 'star-topology NoC and nine spiking-convolution cores holding 327,680 neurons '
                 '(16-bit state) and 272 KB of 8-bit kernel memory, plus a 15-class readout. A '
                 'core sweeps kernel positions per input event, skips zero weights at the read, '
                 'updates one 16-bit neuron word per synaptic event (read, add, check, write), '
                 'emits at most one spike per update with subtract-or-reset and a threshold_low '
                 'clamp, applies bias and leak on a slow clock through the same path, and '
                 'routes to at most two destinations. Cores differ only in memory: 64 Ki '
                 'neuron words on cores 0 to 2, 32 Ki on 3 and 4, 16 Ki on 5 to 8.'),
        cost_summary='preview: no engine, no costs',
        costs=_all('planned', SPECK_PLAN, 'to be calibrated on the physical dev kit once the '
                   'event engine exists (plan §9); nothing is modeled today'),
        time_word='planned', energy_word='planned',
        validation=('None. This entry simulates nothing. Published anchors for the future '
                    'engine: 1.58 µs for one 3 × 3 layer and 3.36 µs for nine layers pad to '
                    'pad (Richter), 120 ns to 7 µs per layer by kernel size (Yao), about 30 M '
                    'events/s per neuron compute unit, 100 M SynOps/s on core 0 and 30 M on '
                    'the others, 0.47 to 0.6 mW on N-MNIST (Speck1). The papers disagree on '
                    'the 3.36 µs, the energy per SynOp and which cores are parallel; the '
                    'calibration protocol settles them.'),
        not_modeled=('everything: no event engine, no timing, no energy, no congestion',
                     'the Speck neuron (int16 state, threshold_low clamp, leak sweeps on a '
                     'slow clock); the stand-in unit is the int24 IF',
                     'the kernel-memory sweep and sum pooling (stand-ins: integer synapse and '
                     'accumulator)', 'the DVS, the pre-processing core and the readout',
                     'fan-out 2, the congestion balancer, the output decimator',
                     'the update-boundary buffer in the file does not exist on the chip'),
        references=(RICHTER_2024, YAO_2024, SPECK_PLAN),
        demo_neuron=None, demo_weight=None, preview=True, preview_note=_SPECK_NOTE,
        preview_layout=_SPECK_LAYOUT),
)


def registry():
    return PLATFORMS


def get(platform_id):
    for platform in PLATFORMS:
        if platform.id == platform_id:
            return platform
    raise KeyError(f'unknown platform {platform_id!r}; choose one of '
                   f'{", ".join(p.id for p in PLATFORMS)}')


def badge(platform):
    return f'{platform.title} · {platform.cost_summary} · not hardware'


@lru_cache(maxsize=None)
def fingerprint(platform_id):
    """Fingerprint of the platform's packaged architecture (see sanafe.loihi2)."""
    return architecture_fingerprint(sanafe.load_arch(str(get(platform_id).arch_yaml)))


@lru_cache(maxsize=None)
def _arch_name(platform_id):
    return yaml.safe_load(get(platform_id).arch_yaml.read_text())['architecture']['name']


def match(arch, candidates=None):
    """The catalog platform whose packaged architecture equals ``arch`` in every
    configured value, or None. Only platforms with the same architecture name
    are loaded, so an unrelated file costs nothing."""
    ids = list(candidates) if candidates is not None else [p.id for p in PLATFORMS]
    named = [pid for pid in ids if _arch_name(pid) == arch.configuration()['name']]
    if not named:
        return None
    wanted = architecture_fingerprint(arch)
    return next((pid for pid in named if fingerprint(pid) == wanted), None)


_BUFFER_NAMES = {'buffer_before_dendrite_unit': 'before the dendrite unit',
                 'buffer_before_soma_unit': 'before the soma unit',
                 'buffer_before_axon_out_unit': 'before the axon-out unit'}


def _buffer_name(value):
    try:
        return _BUFFER_NAMES.get(sanafe.BufferPosition(value).name, str(value))
    except (ValueError, TypeError):
        return str(value)


_RANGE = re.compile(r'\[\d+\.\.\d+\]$')


def _one_core_description(yaml_path):
    """``SpikingChip.describe()`` of the same file reduced to one tile with one
    core. Unit roles and accepted attributes are per-unit-template facts, so a
    one-core chip reports them without building the full mesh (seconds and
    gigabytes for Loihi 1)."""
    data = yaml.safe_load(Path(yaml_path).read_text())
    arch = data['architecture']
    arch['attributes']['width'] = 1
    arch['attributes']['height'] = 1
    tile = arch['tile'][0]
    tile['name'] = _RANGE.sub('', tile['name'])
    core = tile['core'][0]
    core['name'] = _RANGE.sub('', core['name'])
    for role in ('axon_in', 'synapse', 'dendrite', 'soma', 'axon_out'):
        for unit in core.get(role, []) or []:
            unit['name'] = _RANGE.sub('', unit['name'])
    arch['tile'] = [tile]
    tile['core'] = [core]
    with tempfile.TemporaryDirectory(prefix='sanafe-platform-') as scratch:
        reduced = Path(scratch) / 'one_core.yaml'
        reduced.write_text(yaml.safe_dump(data, sort_keys=False))
        return sanafe.SpikingChip(sanafe.load_arch(str(reduced))).describe()


@lru_cache(maxsize=None)
def _facts(platform_id):
    """Structure, units and cost values read from the loaded architecture."""
    platform = get(platform_id)
    config = sanafe.load_arch(str(platform.arch_yaml)).configuration()
    described = _one_core_description(platform.arch_yaml)
    tile, core = config['tiles'][0], config['tiles'][0]['cores'][0]
    dcore = described['tiles'][0]['cores'][0]
    framework = sanafe.framework_attributes
    framework = set(framework() if callable(framework) else framework)
    sync_table = {str(k): v for k, v in config['sync_table'].items()}
    structure = {'width': config['width'], 'height': config['height'],
                 'tiles': len(config['tiles']), 'cores': config['core_count'],
                 'cores_per_tile': len(tile['cores']),
                 'max_neurons_per_core': core['max_neurons_supported'],
                 'buffer_position': _buffer_name(core['buffer_position']),
                 'sync_model': 'table' if len(sync_table) > 1 else 'fixed',
                 'sync_table': sync_table,
                 'timestep_delay': config['timestep_delay'],
                 'link_buffer_size': config['link_buffer_size']}
    models = {u['name'].split('[')[0]: u for u in reversed(core['pipeline'])}
    # A YAML unit such as loihi_inputs[0..1023] expands to one instance per
    # index in the full chip; the card lists the template once with its
    # instance count (from the full configuration, not the one-core chip).
    instances = {}
    for unit in core['pipeline']:
        base = unit['name'].split('[')[0]
        instances[base] = instances.get(base, 0) + 1
    units, by_base = [], {}
    for unit in dcore['pipeline_units']:
        base = unit['name'].split('[')[0]
        if base in by_base:
            continue
        role = '+'.join(r for r in ('synapse', 'dendrite', 'soma') if unit[f'implements_{r}'])
        accepted = [{'name': name, 'help': text}
                    for name, text in sorted(unit['supported_attributes'].items())
                    if name not in framework and name not in COST_ATTRIBUTES]
        entry = {'name': base, 'role': role,
                 'model': models.get(base, {}).get('model', ''),
                 'plugin': models.get(base, {}).get('plugin', ''),
                 'instances': instances.get(base, 1), 'attributes': accepted}
        by_base[base] = entry
        units.append(entry)
    for name, _energy, _latency in core['axon_in']:
        units.append({'name': name, 'role': 'axon', 'model': 'axon in', 'plugin': '', 'attributes': []})
    for name, _energy, _latency in core['axon_out']:
        units.append({'name': name, 'role': 'axon', 'model': 'axon out', 'plugin': '', 'attributes': []})
    values = [(config['name'], 'link_buffer_size', config['link_buffer_size']),
              (config['name'], 'latency_sync', sync_table)]
    tile_name = tile['name'].split('[')[0]
    for name, value in zip(HOP_COSTS, tile['link_costs']):
        values.append((tile_name, name, value))
    for name, energy, latency in core['axon_in']:
        values += [(name, 'energy_message_in', energy), (name, 'latency_message_in', latency)]
    for name, energy, latency in core['axon_out']:
        values += [(name, 'energy_message_out', energy), (name, 'latency_message_out', latency)]
    seen_units = set()
    for unit in core['pipeline']:
        base = unit['name'].split('[')[0]
        if base in seen_units:
            continue
        seen_units.add(base)
        for name, value in unit['attributes'].items():
            if name in COST_ATTRIBUTES:
                values.append((base, name, value))
    return structure, units, values


def describe(platform):
    """The JSON card the Studio shows for a platform."""
    structure, units, values = _facts(platform.id)
    rows = []
    for unit, attribute, value in values:
        entry = platform.costs.get(f'{unit}.{attribute}') or platform.costs.get(attribute)
        if entry is None or entry['status'] not in STATUSES:
            raise ValueError(f'{platform.id}: cost attribute {attribute!r} has no provenance entry')
        row = {'unit': unit, 'attribute': attribute, 'value': value, 'status': entry['status'],
               'source': entry['source'], 'note': entry['note']}
        if 'factor' in entry:
            row['factor'] = entry['factor']
        rows.append(row)
    return {'id': platform.id, 'title': platform.title, 'vendor': platform.vendor,
            'generation': platform.generation, 'yaml': platform.yaml_name,
            'execution': platform.execution, 'time_rule': platform.time_rule,
            'summary': platform.summary, 'badge': badge(platform),
            'time_word': platform.time_word, 'energy_word': platform.energy_word,
            'not_hardware': NOT_HARDWARE, 'structure': structure, 'units': units,
            'costs': rows, 'validation': platform.validation,
            'not_modeled': list(platform.not_modeled), 'references': list(platform.references),
            'preview': platform.preview, 'preview_note': platform.preview_note,
            'preview_layout': platform.preview_layout}
