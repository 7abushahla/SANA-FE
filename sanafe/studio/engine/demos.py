"""Demo workloads that run on every catalog platform (no reference, no readout)."""
import random

import sanafe
from sanafe import platforms as P

from .workload import BuiltWorkload, ParameterSpec


class RandomSNN:
    """A seeded random feed-forward network in the shape of the TCAD 2025
    randomized benchmark. Group 0 is the drive: each neuron carries a constant
    bias drawn from the seed so it fires every 1 to 20 updates. Each later group
    connects to the previous one with the connection percentage and weights
    drawn from the seed. Connectivity depends on the seed only; neuron
    attributes and weight kinds come from the platform's demo recipe.
    """

    name = 'random-snn'
    platforms = tuple(p.id for p in P.registry())
    default_trace_level = 'full'

    def parameters(self):
        return (
            ParameterSpec('groups', 'int', default=3, minimum=2, maximum=8,
                          help='Feed-forward groups; group_0 is the bias-driven input'),
            ParameterSpec('neurons_per_group', 'int', default=32, minimum=1, maximum=8192,
                          help='Neurons in every group; must fit one core of the platform'),
            ParameterSpec('connection_percent', 'int', default=25, minimum=0, maximum=100,
                          help='Chance that a neuron connects to each neuron of the next group'),
            ParameterSpec('placement', 'choice', default='packed', choices=('packed', 'spread'),
                          help='packed: fill cores in order; spread: one group per distant tile'),
            ParameterSpec('seed', 'int', default=1, minimum=0, help='Seed for connectivity and drive'),
            ParameterSpec('horizon', 'int', default=20, minimum=1, help='Updates executed by Run to horizon'),
        )

    def build(self, params):
        platform = P.get(params['platform'])
        arch = sanafe.load_arch(str(platform.arch_yaml))
        groups, size = params['groups'], params['neurons_per_group']
        edge_rng = random.Random(params['seed'])          # connectivity: platform-independent
        attr_rng = random.Random(params['seed'] + 1000)   # neuron attributes and weights
        placed = self._cores(arch, groups, size, params['placement'])  # (core, key) per group
        net = sanafe.Network()
        made = []
        for index in range(groups):
            group = net.create_neuron_group(f'group_{index}', size, log_spikes=True, log_potential=True)
            kind = 'input' if index == 0 else 'hidden'
            for neuron in group:
                neuron.set_attributes(model_attributes=platform.demo_neuron(kind, attr_rng))
                neuron.map_to_core(placed[index][0])
            made.append(group)
        edges = 0
        chance = params['connection_percent'] / 100.0
        for source, destination in zip(made, made[1:]):
            for pre in source:
                for post in destination:
                    if edge_rng.random() < chance:
                        pre.connect_to_neuron(post, {'weight': platform.demo_weight(attr_rng)})
                        edges += 1
        core_keys = []
        for _, key in placed:
            if key not in core_keys:
                core_keys.append(key)
        return BuiltWorkload(arch_yaml=platform.arch_yaml, arch=arch, network=net,
                             horizon=params['horizon'],
                             metadata={'platform': platform.id, 'edges': edges,
                                       'placement': params['placement'],
                                       'layer_sizes': [size] * groups, 'cores': core_keys,
                                       'seed': params['seed'], 'reference': None})

    @staticmethod
    def _cores(arch, groups, size, placement):
        """(core, 'tile.core' key) per group; packed shares cores while groups fit."""
        limit = arch.tiles[0].cores[0].max_neurons_supported
        if size > limit:
            raise ValueError(f"neurons_per_group={size} exceeds this platform's {limit} "
                             'neurons per core; lower neurons_per_group')
        if placement == 'spread':
            stride = max(1, len(arch.tiles) // groups)
            out = []
            for i in range(groups):
                tile = arch.tiles[min(i * stride, len(arch.tiles) - 1)]
                out.append((tile.cores[0], f'{tile.id}.0'))
            return out
        out, used = [], 0
        tile_index, core_index = 0, 0
        for _ in range(groups):
            if used + size > limit:
                core_index += 1
                if core_index >= len(arch.tiles[tile_index].cores):
                    tile_index, core_index = tile_index + 1, 0
                used = 0
            tile = arch.tiles[tile_index]
            out.append((tile.cores[core_index], f'{tile.id}.{core_index}'))
            used += size
        return out
