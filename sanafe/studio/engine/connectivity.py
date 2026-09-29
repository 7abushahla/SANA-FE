"""Who connects to whom, summarized once from the mapped network.

Everything here is read from the network SANA-FE was given and the
placement it reports (provenance R). Fan-in is indexed lazily on the first
neuron query, because only inspection needs it.
"""
from collections import Counter, defaultdict


def _key(group, offset):
    return f'{group}.{offset}'


class Connectivity:
    def __init__(self, network, neurons):
        self._network = network
        self._core_of = {(n.group, n.offset): n.core for n in neurons}
        self._flags = {(n.group, n.offset): (n.log_spikes, n.log_potential)
                       for n in neurons}
        self._fan_in = None

    @classmethod
    def from_network(cls, network, neurons):
        self = cls(network, neurons)
        group_edges = Counter()
        link_synapses = Counter()
        link_axons = defaultdict(set)
        incoming = Counter()
        outgoing = defaultdict(set)
        thresholds = {}
        for name, group in network.groups.items():
            values = set()
            for offset, neuron in enumerate(group):
                values.add(neuron.model_attributes.get('threshold'))
                src_core = self._core_of[(name, offset)]
                for edge in neuron.edges_out:
                    post = (edge.post_neuron.group_name, int(edge.post_neuron.neuron_offset))
                    dst_core = self._core_of[post]
                    group_edges[(name, post[0])] += 1
                    link_synapses[(src_core, dst_core)] += 1
                    link_axons[(src_core, dst_core)].add((name, offset))
                    incoming[dst_core] += 1
                    outgoing[src_core].add((name, offset))
            uniform = len(values) == 1 and isinstance(next(iter(values)), int)
            thresholds[name] = {'threshold': next(iter(values)) if uniform else None}
        self.group_edges = [{'src': src, 'dst': dst, 'synapses': count}
                            for (src, dst), count in group_edges.items()]
        self.core_links = [{'src': src, 'dst': dst, 'synapses': count,
                            'axons': len(link_axons[(src, dst)])}
                           for (src, dst), count in link_synapses.items()]
        self.group_attributes = thresholds
        self._incoming = incoming
        self._outgoing = {core: len(members) for core, members in outgoing.items()}
        self.core_neurons = {}
        for neuron in neurons:
            ranges = self.core_neurons.setdefault(neuron.core, [])
            last = ranges[-1] if ranges else None
            if last and last[0] == neuron.group and last[2] + 1 == neuron.offset:
                last[2] = neuron.offset
            else:
                ranges.append([neuron.group, neuron.offset, neuron.offset])
        return self

    def core_stats(self):
        """Per-core counts in the form ``validate_core_budgets`` takes."""
        stats = []
        for core, ranges in self.core_neurons.items():
            tile, offset = (int(part) for part in core.split('.'))
            stats.append({'tile': tile, 'core': offset,
                          'neurons': sum(last - first + 1 for _, first, last in ranges),
                          'edges': self._incoming.get(core, 0),
                          'outgoing_neurons': self._outgoing.get(core, 0)})
        return stats

    def _index_fan_in(self):
        fan_in = defaultdict(list)
        for name, group in self._network.groups.items():
            for offset, neuron in enumerate(group):
                for edge in neuron.edges_out:
                    post = (edge.post_neuron.group_name, int(edge.post_neuron.neuron_offset))
                    fan_in[post].append(((name, offset), edge.synapse_attributes.get('weight')))
        return fan_in

    def _entry(self, address, weight):
        return {'neuron': _key(*address), 'core': self._core_of[address], 'weight': weight}

    def neuron(self, group, offset, limit=256):
        """One neuron's attributes, core, and (capped) fan-in and fan-out."""
        address = (group, int(offset))
        if address not in self._core_of:
            raise KeyError(f'no neuron {_key(group, offset)}')
        if self._fan_in is None:
            self._fan_in = self._index_fan_in()
        neuron = self._network.groups[group][int(offset)]
        fan_out = [((edge.post_neuron.group_name, int(edge.post_neuron.neuron_offset)),
                    edge.synapse_attributes.get('weight')) for edge in neuron.edges_out]
        fan_in = self._fan_in.get(address, [])
        log_spikes, log_potential = self._flags[address]
        return {'group': group, 'offset': int(offset), 'core': self._core_of[address],
                'attributes': neuron.model_attributes,
                'log_spikes': log_spikes, 'log_potential': log_potential,
                'fan_in': [self._entry(a, w) for a, w in fan_in[:limit]],
                'fan_in_total': len(fan_in),
                'fan_out': [self._entry(a, w) for a, w in fan_out[:limit]],
                'fan_out_total': len(fan_out)}
