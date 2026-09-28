"""Integer export preserves configured Lava arithmetic and rejects unsafe graphs."""
import unittest
import numpy as np
try:
    import sanafe.lava as bridge
    from lava.proc.qcfs import QCFSIFFixed
    from lava.proc.dense.process import Dense
    HAVE_LAVA = True
except ImportError:
    HAVE_LAVA = False


@unittest.skipUnless(HAVE_LAVA, 'requires optional Lava QCFSIFFixed extension')
class TestIntegerBridge(unittest.TestCase):
    def graph(self, weights=None, **dense_params):
        weights = np.array([[3, -3]], dtype=np.int32) if weights is None else weights
        first = QCFSIFFixed(shape=(weights.shape[1],), threshold=4)
        second = QCFSIFFixed(shape=(weights.shape[0],), threshold=4)
        dense = Dense(weights=weights, **dense_params)
        first.s_out.connect(dense.s_in)
        dense.a_out.connect(second.a_in)
        return [first, second], [dense]

    def export(self, layers=None, connections=None, currents=None, places=None, **kwargs):
        self.assertTrue(hasattr(bridge, 'qcfs_fixed_chain_to_sanafe'),
                        'integer chain exporter is missing')
        from sanafe.loihi2 import load_loihi2_candidate
        if layers is None:
            layers, connections = self.graph()
        if currents is None:
            currents = np.full((layers[0].v.shape[0], 3), 2, dtype=np.int32)
        return bridge.qcfs_fixed_chain_to_sanafe(
            layers, connections, currents, load_loihi2_candidate(),
            [(0, 0), (1, 0)] if places is None else places, **kwargs)

    def test_effective_weights_and_integer_execution(self):
        import sanafe
        net, manifest = self.export()
        from sanafe.loihi2 import load_loihi2_candidate
        self.assertEqual(manifest['effective_weights'], [[[2, -4]]])
        self.assertFalse(manifest['hardware_validated'])
        chip = sanafe.SpikingChip(load_loihi2_candidate())
        chip.load(net)
        self.assertEqual(chip.sim(3, potential_trace=True)['potential_trace'],
                         [[0, 0, 2], [2, 2, 0], [0, 0, 0]])

    def test_configured_weight_precision_is_honored(self):
        layers, connections = self.graph(np.array([[127, -127]]), num_weight_bits=4)
        _, manifest = self.export(layers, connections)
        self.assertEqual(manifest['effective_weights'], [[[96, -128]]])

    def test_invalid_current_rejected(self):
        for value in [1.5, np.nan, 32768, -32769, True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.export(currents=np.full((2, 3), value))

    def test_unsafe_fanin_rejected(self):
        layers, dense = self.graph(np.full((1, 130), 255, dtype=np.int32))
        with self.assertRaisesRegex(ValueError, 'fan-in'):
            self.export(layers, dense)

    def test_message_precision_and_exponent_are_rejected(self):
        for params in [{'num_message_bits': 4}, {'weight_exp': 1}, {'num_weight_bits': 9}]:
            with self.subTest(params=params), self.assertRaises(ValueError):
                self.export(*self.graph(**params))

    def test_fractional_weight_rejected(self):
        with self.assertRaisesRegex(ValueError, 'integer'):
            self.export(*self.graph(np.array([[0.5, 1.0]])))

    def test_reserved_core_and_resource_overflow_rejected(self):
        with self.assertRaisesRegex(ValueError, 'reserved'):
            self.export(reserved_cores=[(0, 0)])
        from sanafe.loihi2 import Allocation
        with self.assertRaises(ValueError):
            self.export(allocation=Allocation(program_bytes_per_core=192*1024))

    def test_branch_and_disconnection_rejected(self):
        layers, dense = self.graph()
        extra = Dense(weights=np.ones((1, 2), dtype=np.int32))
        layers[0].s_out.connect(extra.s_in)
        with self.assertRaisesRegex(ValueError, 'unbranched'):
            self.export(layers, dense)

    def test_pending_dense_current_rejected(self):
        layers, dense = self.graph()
        dense[0].a_buff.init = np.array([4], dtype=np.int32)
        with self.assertRaisesRegex(ValueError, 'initial Dense buffer'):
            self.export(layers, dense)

    def test_changed_architecture_rejected(self):
        import sanafe
        from sanafe.loihi2 import load_loihi2_candidate
        layers, dense = self.graph()
        arch = load_loihi2_candidate()
        tiles = list(arch.tiles)
        replacement = sanafe.Tile(tiles[0].name, 0, latency_east_hop=99.)
        replacement.cores = tiles[0].cores
        tiles[0] = replacement
        arch.tiles = tiles
        with self.assertRaisesRegex(ValueError, 'architecture configuration'):
            bridge.qcfs_fixed_chain_to_sanafe(layers, dense,
                np.ones((2, 3), dtype=np.int32), arch, [(0, 0), (1, 0)])

    def test_custom_allocation_consistent_in_manifest(self):
        from sanafe.loihi2 import Allocation
        _, manifest = self.export(allocation=Allocation(state_bytes_per_neuron=20))
        self.assertEqual(manifest['architecture_profile']['allocation'],
                         manifest['resource_report']['allocation'])

    def test_custom_initial_state_copied(self):
        layers, dense = self.graph()
        layers[0].v.init = np.array([-4, 7], dtype=np.int32)
        _, manifest = self.export(layers, dense)
        self.assertEqual(manifest['initial_voltage'][0], [-4, 7])


if __name__ == '__main__':
    unittest.main()
