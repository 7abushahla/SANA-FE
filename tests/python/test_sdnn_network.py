"""Sigma-delta networks from netx files: the reader, the builder, the two runners and the
comparison of ``sanafe.sdnn_network``.

The synthetic network is the file of ``sdnn_fixture.write_sdnn_file``. Lava runs it as the
PilotNet SDNN tutorial of lava-dl does, built by lava-dl's netx with ``skip_layers=1`` and
fed through Lava's ``DeltaEncoder`` (dense model), under either fixed-point sigma-delta model.
The simulator runs the same file through ``sdnn_to_sanafe``, with the values Lava's encoder
sent as the graded sources. Every comparison covers every layer at every step, the first
included, with no tolerance. The released PilotNet SDNN file is read only, to check the
reader against netx. The tests need Lava, h5py, the lava-dl checkout and that file, and fail,
never skip, when one is missing.
"""
import copy
import sys

import h5py
import numpy as np
import pytest

from sanafe.lava_sdnn import COST_ASSUMPTIONS, load_loihi2_sdnn_candidate
from sanafe.loihi2 import load_loihi2_candidate
from sanafe.loihi2_counts import core_step_counts, expanded_connectivity
from sanafe.sdnn_network import (MODELS, _stack, compare_sdnn, fits, read_sdnn, sdnn_lava_run,
                                 sdnn_sanafe_run, sdnn_to_sanafe)

from sdnn_fixture import FIXTURE_CORES, fixture_frames, write_sdnn_file
from thesis_tree import FETCH, LAVA_DL_SRC, PILOTNET_SDNN

pytestmark = pytest.mark.thesis
STEPS = 24
N_INPUT = 2 * 8 * 10
LAVA_MODELS = {"fixed": "PySigmaDeltaModelFixed", "corrected": "PySigmaDeltaModelFixedCorrected"}
PARTITION = {"name": "t", "layers": [{"order": "lava", "sizes": [160]},
                                     {"order": "lava", "sizes": [48, 48]},
                                     {"order": "reader", "sizes": [128]},
                                     {"order": "reader", "sizes": [6]},
                                     {"order": "reader", "sizes": [1]}]}


@pytest.fixture(scope="module", autouse=True)
def lava_dl_on_path():
    assert LAVA_DL_SRC.is_dir(), f"{LAVA_DL_SRC} is missing. Run {FETCH} to fetch lava-dl."
    if str(LAVA_DL_SRC) not in sys.path:
        sys.path.insert(0, str(LAVA_DL_SRC))


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """The synthetic file and its Lava runs, each run once per module and kept."""
    path = write_sdnn_file(tmp_path_factory.mktemp("sdnn") / "f.net")
    runs = {}

    def lava(model="fixed", keep="all"):
        if (model, keep) not in runs:
            runs[(model, keep)] = sdnn_lava_run(path, fixture_frames(), STEPS, model=model,
                                                keep=keep)
        return runs[(model, keep)]

    return path, lava


def source_values(seed=1):
    """Arbitrary source values for the tests that build and do not run."""
    return np.random.default_rng(seed).integers(-900, 900, size=(STEPS, N_INPUT), endpoint=True)


def ours_of(path, lava_run, model="fixed", **kwargs):
    """The simulator's run of the synthetic file on the values Lava's encoder sent."""
    arch = load_loihi2_sdnn_candidate()
    net, man = sdnn_to_sanafe(read_sdnn(path), lava_run["layers"][0]["sent"], arch,
                              FIXTURE_CORES, model=model)
    return sdnn_sanafe_run(net, man, arch, STEPS, **kwargs), man


def change(path, what):
    """Edit one dataset of the synthetic file in place."""
    with h5py.File(path, "r+") as f:
        if what == "type LOIHI":
            del f["layer/1/neuron/type"]
            f["layer/1/neuron"].create_dataset("type", data="LOIHI",
                                               dtype=h5py.string_dtype("utf-8"))
        elif what == "activation tanh":
            del f["layer/2/neuron/activation"]
            f["layer/2/neuron"].create_dataset("activation", data="tanh",
                                               dtype=h5py.string_dtype("utf-8"))
        elif what == "sigma_output on layer 4":
            f["layer/4/neuron"].create_dataset("sigma_output", data=np.bool_(True))
        elif what == "messageBits 8":
            f["layer/1/neuron"].create_dataset("messageBits", data=np.int64(8))
        elif what == "bias on the sigma_output layer":
            f["layer/5"].create_dataset("bias", data=np.ones(1, dtype=np.float32))
        else:
            raise AssertionError(what)


# ---------------------------------------------------------------------------
# The reader


def test_the_reader_scales_threshold_and_bias_as_lava_does(tmp_path):
    net = read_sdnn(write_sdnn_file(tmp_path / "f.net"))
    first, conv, last = net.layers[0], net.layers[1], net.layers[-1]
    assert first.transform == {"weight": 64, "bias": 0} and first.neuron["vth_mant"] == 6
    assert (conv.neuron["vth"], conv.neuron["bias_scale"], conv.neuron["state_exp"]) == (
        6 << 12, 1 << 12, 6)
    assert conv.neuron["message_bits"] == 16 and conv.neuron["sigma_output"] is False
    assert last.neuron["sigma_output"] is True and last.bias is None
    with h5py.File(tmp_path / "f.net", "r") as f:
        assert np.array_equal(conv.bias, f["layer/1/bias"][()].astype(np.int64))


@pytest.mark.parametrize("what,message", [
    ("type LOIHI", "SDNN"), ("activation tanh", "activation"),
    ("sigma_output on layer 4", "sigma_output"), ("messageBits 8", "messageBits"),
    ("bias on the sigma_output layer", "bias")])
def test_the_reader_refuses_what_it_does_not_implement(tmp_path, what, message):
    path = write_sdnn_file(tmp_path / "f.net")
    read_sdnn(path)
    change(path, what)
    with pytest.raises(ValueError, match=message):
        read_sdnn(path)


def _lava_dense_weights(synapse):
    """The weights Lava's fixed-point ``Dense`` multiplies by, after the clip to 8 bits and the
    truncation to ``num_weight_bits`` of its first step (``proc/dense/models.py`` 81-90)."""
    from lava.utils.weightutils import clip_weights, determine_sign_mode, truncate_weights

    weights = np.asarray(synapse.weights.init).astype(np.int64)
    mode = determine_sign_mode(weights)
    clipped = clip_weights(weights, mode, num_bits=8)
    bits = int(synapse.proc_params["num_weight_bits"])
    return np.asarray(truncate_weights(clipped, mode, bits)).astype(np.int64)


def test_the_reader_agrees_with_netx_on_the_released_network():
    from lava.lib.dl import netx
    from lava.magma.core.process.ports.ports import ReshapePort, TransposePort

    network = read_sdnn(PILOTNET_SDNN)
    layers = [layer for layer in network.layers if layer.kind != "flatten"]
    built = netx.hdf5.Network(net_config=str(PILOTNET_SDNN), skip_layers=1)
    assert len(built.layers) == len(layers) - 1 == 9
    assert tuple(built.inp.shape) == tuple(reversed(layers[0].shape))
    for k, (block, layer) in enumerate(zip(built.layers, layers[1:]), start=1):
        shape = tuple(reversed(layer.shape)) if len(layer.shape) == 3 else tuple(layer.shape)
        assert tuple(block.shape) == shape, k
        synapse, params = block.synapse, block.synapse.proc_params
        assert int(params["num_message_bits"]) == layer.neuron["message_bits"] == 16, k
        assert int(params["weight_exp"]) == layer.weight_exp, k
        assert int(params["num_weight_bits"]) == layer.weight_bits, k
        if layer.kind == "conv":
            # netx flips the kernel and transposes it to Lava's (out, x, y, in) true convolution
            # (hdf5.py 411-412). Undone, it is the reader's (out, in, h, w) cross-correlation.
            kernel = np.asarray(synapse.weight.init).transpose(0, 3, 2, 1)[:, :, ::-1, ::-1]
            assert np.array_equal(kernel, layer.weights), k
            # Lava's stride and padding are in (x, y) order, the reader's in (height, width).
            assert tuple(int(v) for v in params["stride"]) == tuple(reversed(layer.stride)), k
            assert tuple(int(v) for v in params["padding"]) == tuple(reversed(layer.padding)), k
        else:
            # Lava's Dense shifts the summed input by weight_exp. The shift is exact, and equals
            # the reader's effective weights, when every applied weight is a multiple of the
            # shift.
            applied = _lava_dense_weights(synapse)
            assert np.array_equal(applied, np.asarray(synapse.weights.init)), k
            exp = int(params["weight_exp"])
            if exp < 0:
                assert np.all(applied % (1 << -exp) == 0), k
                applied = applied >> -exp
            else:
                applied = applied << exp
            assert np.array_equal(applied, layer.effective_weights()), k
        if layer.neuron["sigma_output"]:
            assert type(block.neuron).__name__ == "Sigma" and k == 9
            continue
        assert type(block.neuron).__name__ == "SigmaDelta", k
        assert int(np.asarray(block.neuron.vth.init).reshape(-1)[0]) == layer.neuron["vth"]
        # A convolution's bias is per channel, the last axis of Lava's (x, y, c) order.
        expected = np.broadcast_to(layer.neuron["bias_scale"] * layer.bias, shape)
        assert np.array_equal(np.broadcast_to(block.neuron.bias.init, shape), expected), k
    # The first dense layer reads the last convolution through out.transpose([2, 1, 0])
    # .flatten() (hdf5.py 552-556), which turns Lava's (x, y, c) order into the reader's
    # (c, h, w) order, the column order of the reader's dense weights.
    first_dense = next(k for k, layer in enumerate(layers[1:]) if layer.kind == "dense")
    (reshape,) = built.layers[first_dense].inp.in_connections
    (transpose,) = reshape.in_connections
    assert isinstance(reshape, ReshapePort) and tuple(reshape.shape) == (2560,)
    assert isinstance(transpose, TransposePort) and tuple(transpose.axes) == (2, 1, 0)
    assert transpose.in_connections == [built.layers[first_dense - 1].out]
    assert network.layers[first_dense + 1].kind == "flatten"
    assert layers[first_dense + 1].weights.shape[1] == 2560


# ---------------------------------------------------------------------------
# The builder


def test_the_builder_states_sources_partition_and_assumptions(tmp_path):
    path = write_sdnn_file(tmp_path / "f.net")
    net, man = sdnn_to_sanafe(read_sdnn(path), source_values(), load_loihi2_sdnn_candidate(),
                              FIXTURE_CORES, partition=PARTITION)
    assert net is not None
    assert man["execution_profile"] == "lava-sigma-delta-int32-v1"
    assert man["sources"]["group"] == "layer_00" and man["sources"]["steps"] == STEPS
    assert man["cost_assumptions"] == list(COST_ASSUMPTIONS)
    assert man["partition"]["layers"][1]["sizes"] == [48, 48]
    assert len({(s["tile"], s["core"]) for l in man["placement"] for s in l["slices"]}) == 6
    assert man["layer_sizes"] == [160, 96, 128, 6, 1] and man["sigma_output_layer"] == 4
    assert man["model"] == "fixed" and man["hardware_validated"] is False
    assert man["resources_validated"] is False
    assert man["allocation_status"] == ("default Allocation() of the candidate profile, not "
                                        "derived from this placement")


def test_a_manifest_without_construction_equals_the_built_one(tmp_path):
    network = read_sdnn(write_sdnn_file(tmp_path / "f.net"))
    arch = load_loihi2_sdnn_candidate()
    built = sdnn_to_sanafe(network, source_values(), arch, FIXTURE_CORES, partition=PARTITION,
                           model="corrected")
    assert sdnn_to_sanafe(network, source_values(), arch, FIXTURE_CORES, partition=PARTITION,
                          model="corrected", build=False) == (None, built[1])


def test_the_builder_refuses_the_candidate_architecture(tmp_path):
    network = read_sdnn(write_sdnn_file(tmp_path / "f.net"))
    with pytest.raises(ValueError, match="sigma-delta profile"):
        sdnn_to_sanafe(network, source_values(), load_loihi2_candidate(), FIXTURE_CORES)


def test_the_builder_refuses_source_values_of_another_width(tmp_path):
    network = read_sdnn(write_sdnn_file(tmp_path / "f.net"))
    with pytest.raises(ValueError, match="source_values"):
        sdnn_to_sanafe(network, source_values()[:, 1:], load_loihi2_sdnn_candidate(),
                       FIXTURE_CORES)


def test_the_builder_refuses_a_weight_beyond_signed_16_bits(tmp_path):
    network = read_sdnn(write_sdnn_file(tmp_path / "f.net"))
    arch = load_loihi2_sdnn_candidate()
    network.layers[2].weights[0, 0, 1, 1] = (1 << 15) - 1
    sdnn_to_sanafe(network, source_values(), arch, FIXTURE_CORES, build=False)
    network.layers[2].weights[0, 0, 1, 1] = 1 << 15
    with pytest.raises(ValueError, match="effective weights outside declared range"):
        sdnn_to_sanafe(network, source_values(), arch, FIXTURE_CORES, build=False)


def test_the_builder_refuses_a_scaled_bias_beyond_signed_32_bits(tmp_path):
    network = read_sdnn(write_sdnn_file(tmp_path / "f.net"))
    arch = load_loihi2_sdnn_candidate()
    network.layers[4].bias[0] = -(1 << 19)            # times 4096 is -2**31, inside
    sdnn_to_sanafe(network, source_values(), arch, FIXTURE_CORES, build=False)
    network.layers[4].bias[0] = 1 << 19               # times 4096 is 2**31, outside
    with pytest.raises(ValueError, match="layer 4: scaled bias outside declared range"):
        sdnn_to_sanafe(network, source_values(), arch, FIXTURE_CORES, build=False)


def test_the_builder_refuses_a_source_value_beyond_signed_32_bits(tmp_path):
    network = read_sdnn(write_sdnn_file(tmp_path / "f.net"))
    arch = load_loihi2_sdnn_candidate()
    values = source_values()
    values[3, 17] = -(1 << 31)
    sdnn_to_sanafe(network, values, arch, FIXTURE_CORES, build=False)
    values[3, 17] = 1 << 31
    with pytest.raises(ValueError, match="source_values outside declared range"):
        sdnn_to_sanafe(network, values, arch, FIXTURE_CORES, build=False)


# ---------------------------------------------------------------------------
# The runners and the comparison


@pytest.mark.parametrize("model", MODELS)
def test_the_simulator_equals_lava_on_a_synthetic_network(synthetic, model):
    path, lava_of = synthetic
    lava = lava_of(model)
    ours, _ = ours_of(path, lava, model)
    result = compare_sdnn(ours, lava)
    assert result["total_mismatches"] == 0
    assert result["neuron_steps"] == STEPS * (160 + 96 + 128 + 6 + 1)
    assert [layer["compared"] for layer in result["layers"]] == (
        [["sent"]] + [["sigma", "act", "residue", "sent", "input"]] * 3
        + [["sigma", "sent", "input"]])
    assert np.any(lava["layers"][-2]["sent"] != 0) and np.any(lava["layers"][-1]["sigma"] != 0)
    assert set(lava["models"][1:-1]) == {LAVA_MODELS[model]}
    assert lava["models"][0] == "PyDeltaEncoderModelDense"
    assert lava["models"][-1] == "PySigmaModelFixed"


def test_the_two_models_differ_on_the_synthetic_network(synthetic):
    _, lava_of = synthetic
    fixed, corrected = lava_of("fixed"), lava_of("corrected")
    assert any(not np.array_equal(f["sent"], c["sent"])
               for f, c in zip(fixed["layers"][1:], corrected["layers"][1:]))


def test_the_comparison_finds_one_changed_value(synthetic):
    path, lava_of = synthetic
    lava = lava_of()
    ours, _ = ours_of(path, lava)
    changed = copy.deepcopy(ours)
    changed["layers"][2]["act"][5, 7] += 1
    result = compare_sdnn(changed, lava)
    assert result["total_mismatches"] == 1
    assert result["layers"][2]["mismatches"]["act"] == 1
    a, b = int(changed["layers"][2]["act"][5, 7]), int(lava["layers"][2]["act"][5, 7])
    assert result["layers"][2]["first"]["act"] == [5, 7, a, b]
    assert result["layers"][2]["first"]["sigma"] is None


def test_a_chunked_run_equals_one_run(synthetic):
    path, lava_of = synthetic
    a, _ = ours_of(path, lava_of(), chunk=3)
    b, _ = ours_of(path, lava_of(), chunk=24)
    assert len(a["wall_seconds"]) == 8 and len(b["wall_seconds"]) == 1
    assert compare_sdnn(a, b)["total_mismatches"] == 0
    assert a["trace"]["perf_trace"] == b["trace"]["perf_trace"]
    assert a["trace"]["sim_time"] == b["trace"]["sim_time"]
    assert a["trace"]["energy"] == b["trace"]["energy"]


def test_an_extremes_run_reports_the_extremes_of_a_full_run(synthetic):
    path, lava_of = synthetic
    full, _ = ours_of(path, lava_of(), keep="all")
    extremes, _ = ours_of(path, lava_of(), keep="extremes")
    assert extremes["extremes"] == full["extremes"]
    assert extremes["layers"] is None and extremes["spikes"] is None
    assert full["trace"]["perf_trace"] == extremes["trace"]["perf_trace"]
    for k, layer in enumerate(full["layers"]):
        for state, array in layer.items():
            assert full["extremes"][k][state] == int(np.abs(array.astype(np.int64)).max())


def test_lavas_nonzero_run_gives_the_simulators_spikes_and_extremes(synthetic):
    path, lava_of = synthetic
    nz = lava_of(keep="nonzero")
    ours, _ = ours_of(path, lava_of())
    assert all(set(n) == {"sent"} and n["sent"].dtype == bool for n in nz["layers"])
    assert all(np.array_equal(n["sent"], s) for n, s in zip(nz["layers"], ours["spikes"]))
    assert nz["extremes"] == ours["extremes"] == lava_of()["extremes"]


def test_the_fits_read_the_extremes_against_24_and_16_bits():
    extremes = [{"sent": (1 << 15) - 1},
                {"sigma": (1 << 23) - 1, "act": 5, "residue": 0, "sent": 1 << 15,
                 "input": 1 << 23},
                {"sigma": 1 << 23, "sent": 3, "input": (1 << 23) - 1}]
    assert fits(extremes) == [
        {"states_fit_24": None, "sent_fits_16": True, "sent_fits_24": True,
         "input_fits_24": None},
        {"states_fit_24": True, "sent_fits_16": False, "sent_fits_24": True,
         "input_fits_24": False},
        {"states_fit_24": False, "sent_fits_16": True, "sent_fits_24": True,
         "input_fits_24": True}]
    # A sigma_output layer sends its sigma, which the tutorial decodes as 24 bits.
    output = fits([{"sigma": (1 << 23) - 1, "sent": (1 << 23) - 1, "input": 5},
                   {"sigma": 1 << 23, "sent": 1 << 23, "input": 5}])
    assert [(f["sent_fits_16"], f["sent_fits_24"]) for f in output] == [(False, True),
                                                                        (False, False)]


def test_the_stack_of_chunks_equals_their_concatenation_and_releases_them():
    rng = np.random.default_rng(2)
    parts = [rng.integers(-9, 9, size=(n, 5)).astype(np.int32) for n in (3, 1, 4)]
    expected = np.concatenate(parts, axis=0)
    stacked = _stack(parts)
    assert parts == [] and stacked.dtype == np.int32 and np.array_equal(stacked, expected)


def test_counts_of_an_sdnn_manifest_agree_with_the_perf_trace(synthetic):
    path, lava_of = synthetic
    ours, man = ours_of(path, lava_of())
    counts = core_step_counts(man, ours, weights_per_read=4, message_bits=32,
                              connectivity=expanded_connectivity(read_sdnn(path)))
    assert counts["verified_against_perf_trace"] is True
    assert counts["execution_profile"] == "lava-sigma-delta-int32-v1"
    assert counts["emitted_convention"] == "nonzero"
    assert counts["partition"]["name"] == "builder"
