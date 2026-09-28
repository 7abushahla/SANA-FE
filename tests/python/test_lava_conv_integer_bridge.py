"""Spatial integer Conv export against Lava's actual fixed ProcessModels."""
import unittest
import numpy as np
from numpy.testing import assert_array_equal

try:
    import sanafe
    from sanafe.lava import qcfs_fixed_conv_to_sanafe
    from sanafe.loihi2 import load_loihi2_candidate
    from lava.proc.qcfs import QCFSIFFixed
    from lava.proc.conv.process import Conv
    from lava.proc.conv import utils
    from lava.proc.io import source, sink
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.run_configs import Loihi2SimCfg
    HAVE_LAVA = True
except ImportError:
    HAVE_LAVA = False


@unittest.skipUnless(HAVE_LAVA, 'requires optional Lava QCFSIFFixed extension')
class TestConvIntegerBridge(unittest.TestCase):
    def graph(self, T=2, *, weight=None, conv_kwargs=None):
        rng = np.random.default_rng(22)
        image = rng.integers(0, 7, size=(4, 5, 2), dtype=np.int32)
        first = QCFSIFFixed(shape=image.shape, threshold=8)
        if weight is None:
            weight = rng.integers(-3, 4, size=(3, 2, 3, 2), dtype=np.int32)
        params = {'weight': weight, 'input_shape': image.shape, 'padding': (0, 1)}
        params.update(conv_kwargs or {})
        conv = Conv(**params)
        second = QCFSIFFixed(shape=conv.output_shape, threshold=10)
        first.s_out.connect(conv.s_in)
        conv.a_out.connect(second.a_in)
        currents = np.repeat(image[..., None], T + 1, axis=-1)
        currents[..., T:] = 0
        return first, conv, second, currents

    def export(self, graph, places=((0, 0), (1, 0))):
        first, conv, second, currents = graph
        arch = load_loihi2_candidate()
        net, manifest = qcfs_fixed_conv_to_sanafe(first, conv, second,
                                                  currents, arch, places)
        return arch, net, manifest

    def test_spatial_edges_match_lava_convolution(self):
        first, conv, second, _ = self.graph()
        arch, net, manifest = self.export((first, conv, second,
                                          np.zeros((*first.v.shape, 2), dtype=np.int32)))
        self.assertFalse(manifest['conv']['physical_weight_packing_verified'])
        self.assertEqual(manifest['resource_report']['storage_bound'],
                         'dense expansion upper bound; actual emitted edges are spatial')
        rng = np.random.default_rng(25)
        dst, src, weights = utils.conv_to_sparse(first.v.shape, second.v.shape,
                                                 conv.weight.init, conv.stride.init,
                                                 conv.padding.init, (1, 1), 1)
        self.assertEqual(len(dst), manifest['conv']['expanded_synapses'])
        for _ in range(4):
            spike = rng.integers(0, 2, size=first.v.shape)
            actual = np.zeros(np.prod(second.v.shape), dtype=np.int64)
            np.add.at(actual, dst, spike.ravel()[src] * weights)
            expected = utils.conv(spike, conv.weight.init, conv.kernel_size.init,
                                  conv.stride.init, conv.padding.init, (1, 1), 1)
            assert_array_equal(actual.reshape(second.v.shape), expected)

    def test_lava_and_sanafe_match_each_update_with_drain(self):
        for T in (1, 2, 4):
            with self.subTest(T=T):
                first, conv, second, currents = self.graph(T)
                arch, net, manifest = self.export((first, conv, second, currents))
                count = T + 1
                src = source.RingBuffer(data=currents)
                src.s_out.connect(first.a_in)
                sinks = [sink.RingBuffer(shape=p.v.shape, buffer=count)
                         for p in (first, second)]
                for p, obs in zip((first, second), sinks):
                    p.s_out.connect(obs.a_in)
                voltages = []
                try:
                    for _ in range(count):
                        first.run(condition=RunSteps(num_steps=1),
                                  run_cfg=Loihi2SimCfg(select_tag='fixed_pt'))
                        voltages.append(np.concatenate((first.v.get().ravel(),
                                                        second.v.get().ravel())))
                    lava_spikes = np.concatenate([obs.data.get().reshape(-1, count)
                                                   for obs in sinks], axis=0)
                finally:
                    first.stop()
                chip = sanafe.SpikingChip(arch)
                chip.load(net)
                result = chip.sim(count, spike_trace=True, potential_trace=True)
                output = np.zeros_like(lava_spikes)
                offsets = (0, first.v.init.size)
                for t, active in enumerate(result['spike_trace']):
                    for neuron in active:
                        layer = int(neuron.group_name.split('_')[-1])
                        output[offsets[layer] + neuron.neuron_offset, t] = 1
                assert_array_equal(output, lava_spikes)
                assert_array_equal(np.asarray(result['potential_trace']), voltages)
                self.assertEqual(manifest['updates'], T + 1)
                self.assertEqual(manifest['conv_delay_updates'], 1)
                self.assertTrue(np.all(lava_spikes[offsets[1]:, 0] == 0))

    def test_one_presentation_reaches_next_layer_on_drain_update(self):
        first = QCFSIFFixed(shape=(1, 1, 1), threshold=2)
        conv = Conv(weight=np.array([[[[2]]]], dtype=np.int32),
                    input_shape=first.v.shape)
        second = QCFSIFFixed(shape=conv.output_shape, threshold=2)
        first.s_out.connect(conv.s_in)
        conv.a_out.connect(second.a_in)
        currents = np.array([[[[2, 0]]]], dtype=np.int32)
        arch, net, manifest = self.export((first, conv, second, currents))
        chip = sanafe.SpikingChip(arch)
        chip.load(net)
        result = chip.sim(2, spike_trace=True, potential_trace=True)
        self.assertEqual([[n.group_name for n in step] for step in result['spike_trace']],
                         [['layer_0'], ['layer_1']])
        self.assertEqual(manifest['updates'], 2)
        self.assertEqual(manifest['conv_delay_updates'], 1)

    def test_unsupported_graphs_and_precision_rejected(self):
        for params, label in [({'groups': 2}, 'groups'),
                              ({'dilation': 2}, 'dilation'),
                              ({'weight_exp': 1}, 'weight exponent'),
                              ({'num_weight_bits': 4}, 'weight bits'),
                              ({'num_message_bits': 4}, 'message bits')]:
            with self.subTest(params=params), self.assertRaisesRegex(ValueError, label):
                self.export(self.graph(conv_kwargs=params))
        graph = self.graph()
        graph[1].a_buf.init = np.ones(graph[2].v.shape, dtype=np.int32)
        with self.assertRaisesRegex(ValueError, 'initial Conv buffer'):
            self.export(graph)
        graph = self.graph()
        extra = Conv(weight=graph[1].weight.init, input_shape=graph[0].v.shape,
                     padding=(0, 1))
        graph[0].s_out.connect(extra.s_in)
        with self.assertRaisesRegex(ValueError, 'unbranched'):
            self.export(graph)


if __name__ == '__main__':
    unittest.main()
