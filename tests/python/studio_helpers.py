"""Shared fixtures for the Studio engine tests (not a test module)."""
from importlib.resources import files
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
