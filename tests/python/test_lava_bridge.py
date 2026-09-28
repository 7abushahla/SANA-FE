"""Restricted graph-export validation; requires the QCFSIF Lava extension."""
import unittest
try:
    import numpy as np
    from lava.proc.qcfs import QCFSIF
    from lava.proc.dense.process import Dense
    from sanafe.lava import qcfs_chain_to_sanafe
    HAVE_LAVA = True
except ImportError:
    HAVE_LAVA = False
from test_integrate_fire import architecture


@unittest.skipUnless(HAVE_LAVA, 'requires optional Lava QCFSIF extension and NumPy')
class TestLavaBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arch = architecture('integrate_fire_float32')

    def graph(self):
        first = QCFSIF(shape=(2,))
        second = QCFSIF(shape=(1,))
        dense = Dense(weights=np.array([[.5, -.25]]))
        first.s_out.connect(dense.s_in)
        dense.a_out.connect(second.a_in)
        return [first, second], [dense]

    def test_weights_initial_state_and_mapping(self):
        layers, dense = self.graph()
        net = qcfs_chain_to_sanafe(layers, dense, np.ones((2, 3)),
                                  self.arch, [(0, 0), (1, 0)])
        self.assertEqual(len(net['layer_0'][0].edges_out), 1)
        self.assertEqual(len(net['layer_0'][1].edges_out), 1)
        import sanafe
        chip = sanafe.SpikingChip(self.arch)
        chip.load(net)
        result = chip.sim(2, potential_trace=True)
        self.assertEqual(result['potential_trace'], [[.5, .5, .5], [.5, .5, .75]])

    def test_delayed_dendrite_rejected(self):
        layers, dense = self.graph()
        arch = architecture('integrate_fire_float32', 'accumulator_with_delay')
        with self.assertRaisesRegex(ValueError, 'accumulator'):
            qcfs_chain_to_sanafe(layers, dense, np.ones((2, 2)),
                                 arch, [(0, 0), (1, 0)])

    def test_disconnected_graph_rejected(self):
        first, second = QCFSIF(shape=(2,)), QCFSIF(shape=(1,))
        with self.assertRaisesRegex(ValueError, 'unbranched'):
            qcfs_chain_to_sanafe([first, second], [Dense(weights=np.ones((1, 2)))],
                                 np.ones((2, 2)), self.arch, [(0, 0), (1, 0)])

    def test_invalid_current_and_placement_rejected(self):
        for currents, places in [(np.full((2, 2), np.nan), [(0, 0), (1, 0)]),
                                 (np.ones((2, 2)), [(-1, 0), (1, 0)])]:
            with self.subTest(placements=places):
                layers, dense = self.graph()
                with self.assertRaises(ValueError):
                    qcfs_chain_to_sanafe(layers, dense, currents, self.arch, places)


if __name__ == '__main__':
    unittest.main()
