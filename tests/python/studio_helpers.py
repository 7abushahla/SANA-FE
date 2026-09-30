"""Shared fixtures for the Studio engine tests (not a test module)."""
from importlib.resources import files
import base64
import os
from pathlib import Path

import sanafe
from sanafe.loihi2 import load_loihi2_candidate
from sanafe.studio.engine import BuiltWorkload, ParameterSpec

REPO = Path(__file__).resolve().parents[2]
PLACEMENTS = {
    'near': [(0, 0), (1, 0), (2, 0)],
    'far': [(0, 0), (16, 0), (31, 0)],
    'one_router': [(0, 0), (0, 1), (0, 2)],
}
PADDING = 4  # Extra zero-current updates so runs may continue past the drive.


def candidate_yaml():
    return Path(str(files('sanafe.examples') / 'loihi2.yaml'))


def chain_network(arch, placements, steps, sizes=(8, 4, 2)):
    """Three integer IF layers, fully connected, one layer per placed core.

    Layer 0 receives a constant drive for ``steps`` updates, then zero.
    """
    net = sanafe.Network()
    groups = []
    for index, (size, (tile, core)) in enumerate(zip(sizes, placements)):
        group = net.create_neuron_group(f'layer_{index}', size,
                                        log_spikes=True, log_potential=True)
        for offset, neuron in enumerate(group):
            attributes = {'threshold': 4, 'bias': 0, 'initial_voltage': 2}
            if index == 0:
                attributes['currents'] = [offset % 5 + 1] * steps + [0] * PADDING
            neuron.set_attributes(model_attributes=attributes)
            neuron.map_to_core(arch.tiles[tile].cores[core])
        groups.append(group)
    for source, destination in zip(groups, groups[1:]):
        for pre in source:
            for post in destination:
                pre.connect_to_neuron(post, {'weight': 3})
    return net



class ChainWorkload:
    """The three-layer test chain on the Loihi 2 candidate."""

    name = 'test-chain'
    platforms = ('loihi2',)

    def parameters(self):
        return (ParameterSpec('placement', 'choice', default='far',
                              choices=tuple(PLACEMENTS)),
                ParameterSpec('steps', 'int', default=6, minimum=1))

    def build(self, params):
        arch = load_loihi2_candidate()
        network = chain_network(arch, PLACEMENTS[params['placement']], params['steps'])
        return BuiltWorkload(arch_yaml=candidate_yaml(), arch=arch, network=network,
                             horizon=params['steps'],
                             metadata={'placement': params['placement']})


class _ExitOnUpdateTwo:
    def check(self, record):
        if record.update == 2:
            os._exit(7)  # simulate a native crash inside the worker
        return None


class CrashOnSecondUpdate(ChainWorkload):
    """The test chain, whose worker process dies during update 2."""

    name = 'test-crash'

    def build(self, params):
        built = super().build(params)
        built.reference = _ExitOnUpdateTwo()
        return built


class _SlowFirstUpdate:
    def check(self, record):
        if record.update == 1:
            import time
            time.sleep(3)  # one update that takes a while, as a large network would
        return None


class SlowFirstUpdate(ChainWorkload):
    """The test chain, whose first update takes three seconds."""

    name = 'test-slow'

    def build(self, params):
        built = super().build(params)
        built.reference = _SlowFirstUpdate()
        return built


class ChainReadout:
    """layer_2 spike counts per step over its window, lowest index wins ties."""

    def __init__(self, T):
        self.T, self.total = T, None

    def decode(self, record):
        step = record.update - 1 - 2
        if not 0 <= step < self.T:
            return {'step': None, 'scores': None, 'cumulative': self.total,
                    'predicted': None if self.total is None else self.total.index(max(self.total)),
                    'reference': {'status': 'waiting', 'max_abs_diff': None}, 'quantity': 'spike counts'}
        fired = {o for g, o in record.fired if g == 'layer_2'}
        scores = [int(o in fired) for o in range(2)]
        self.total = [a + b for a, b in zip(self.total or [0, 0], scores)]
        return {'step': step, 'scores': scores, 'cumulative': list(self.total),
                'predicted': self.total.index(max(self.total)),
                'reference': {'status': 'match', 'max_abs_diff': 0}, 'quantity': 'spike counts'}


class PipelineChainWorkload(ChainWorkload):
    """The test chain with a declared pipeline and a readout."""

    name = 'test-pipeline'

    def build(self, params):
        built = super().build(params)
        T = params['steps']
        built.readout = ChainReadout(T)
        pixels = bytes((x * 8 + c * 60) % 256 for x in range(32 * 32) for c in range(3))
        built.metadata['pipeline'] = {
            'T': T, 'output_group': 'layer_2', 'classes': ['even', 'odd'], 'fixture': True,
            'rows': [{'group': f'layer_{i}', 'depth': i, 'window': [i, i + T]} for i in range(3)],
            'input': {'index': 0, 'label': 1, 'pixels': base64.b64encode(pixels).decode(),
                      'drive': 'constant test currents', 'currents': 8, 'group': 'layer_0'}}
        return built
