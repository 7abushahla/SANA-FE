"""The sigma-delta soma ``sigma_delta_int32`` and the sigma soma ``sigma_int32`` against the
rules of Lava's fixed-point models, and the derived architecture that offers them.

Row ``t`` of an input array ``a_in`` is the input of update ``t``.
"""
import hashlib
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from sanafe.lava_sdnn import (_sigma_delta_chip, lava_sigma_delta_trace, lava_sigma_trace,
                              load_loihi2_sdnn_candidate, sanafe_sigma_delta_trace,
                              sdnn_fingerprint)
from sanafe.loihi2 import architecture_fingerprint, load_loihi2_candidate
from test_graded_messages import DST, make_chip, run, sources

ROOT = Path(__file__).resolve().parents[2]
S = dict(vth=24576, bias=0, state_exp=6)
SD_UNIT, SIGMA_UNIT = "loihi_sigma_delta", "loihi_sigma"
STATES = ("sigma", "act", "residue", "sent")


def sdnn_chip(groups, edges):
    return make_chip(groups, edges, arch=load_loihi2_sdnn_candidate())[0]


def sd_neuron(attrs, name="layer_0", place=(0, 1)):
    return (name, SD_UNIT, [dict(attrs)], [place])


def destination(name="layer_1", place=(0, 2)):
    return (name, "loihi_if_gated", [dict(DST)], [place])


def column(out, index):
    """The potential of the logged neuron at ``index``, one entry per step."""
    return [[row[index]] for row in out["potential_trace"]]


def random_case():
    """40 neurons, 60 updates, inputs in [-60000, 60000] with about 30 percent set to zero, and
    a bias of 4096 times an integer in [-32, 118] per neuron."""
    rng = np.random.default_rng(0)
    a_in = rng.integers(-60000, 60000, size=(60, 40), endpoint=True)
    a_in[rng.random(a_in.shape) < 0.3] = 0
    bias = 4096 * rng.integers(-32, 118, size=40, endpoint=True)
    return a_in, dict(vth=24576, bias=bias, state_exp=6)


def test_a_negative_change_is_sent_floored():
    for corrected, act in ((False, [40000, 15423]), (True, [40000, 15360])):
        t = sanafe_sigma_delta_trace([[40000], [-24577]], **S, corrected=corrected)
        # -24577 / 64 = -384.02, whose floor is -385
        assert t["sent"].ravel().tolist() == [625, -385]
        assert t["sigma"].ravel().tolist() == [40000, 15423]
        assert t["act"].ravel().tolist() == act


def test_a_change_whose_shift_leaves_zero_sends_nothing():
    attrs = dict(vth=10, bias=0, state_exp=6)
    chip = sdnn_chip([sources([[40, 0, 0]]), sd_neuron(attrs), destination()],
                     [("input", 0, "layer_0", 0, 1), ("layer_0", 0, "layer_1", 0, 1)])
    out = run(chip, 5)
    assert column(out, 1) == [[0]] * 5
    assert not any(m["src_neuron_group_id"] == "layer_0" and not m["placeholder"]
                   for row in out["message_trace"] for m in row)
    fixed = sanafe_sigma_delta_trace([[40], [0], [0]], **attrs)
    corr = sanafe_sigma_delta_trace([[40], [0], [0]], **attrs, corrected=True)
    assert fixed["sent"].ravel().tolist() == corr["sent"].ravel().tolist() == [0, 0, 0]
    assert fixed["act"].ravel().tolist() == [40, 40, 40]
    assert fixed["residue"].ravel().tolist() == [0, 0, 0]
    assert corr["act"].ravel().tolist() == [0, 0, 0]


def test_a_bias_alone_drives_one_message_and_the_remainder_differs():
    zeros = [[0]] * 6
    fixed = sanafe_sigma_delta_trace(zeros, vth=24576, bias=30000, state_exp=6)
    corr = sanafe_sigma_delta_trace(zeros, vth=24576, bias=30000, state_exp=6, corrected=True)
    assert fixed["sent"].ravel().tolist() == corr["sent"].ravel().tolist() == [468, 0, 0, 0, 0, 0]
    assert fixed["act"].ravel().tolist() == [30000] * 6
    assert corr["act"].ravel().tolist() == [29952] * 6


@pytest.mark.parametrize("corrected", [False, True])
def test_the_rules_equal_lavas_transcription(corrected):
    a_in, attrs = random_case()
    ours = sanafe_sigma_delta_trace(a_in, **attrs, corrected=corrected)
    lava = lava_sigma_delta_trace(a_in, **attrs, corrected=corrected)
    assert all(np.array_equal(ours[k], lava[k]) for k in STATES)
    assert np.any(lava["sent"] < 0) and np.any(lava["sent"] > 0)


def test_graded_input_equals_the_transcription_one_update_later():
    # The same input sent by one graded source per neuron, weight 1. Each value reaches its
    # neuron one update after it is sent, so the neuron's first update takes no input, and its
    # bias acts there. The neuron follows the rule on the input with a row of zeros in front.
    a_in, attrs = random_case()
    steps, n = a_in.shape
    neurons = [dict(vth=attrs["vth"], bias=int(b), state_exp=attrs["state_exp"])
               for b in attrs["bias"]]
    chip = sdnn_chip([sources(a_in.T.tolist()), ("layer_0", SD_UNIT, neurons, [(0, 1)] * n)],
                     [("input", i, "layer_0", i, 1) for i in range(n)])
    traces = run(chip, steps + 1, neuron_trace=True)["neuron_trace"]
    delayed = lava_sigma_delta_trace(np.vstack([np.zeros((1, n), dtype=np.int64), a_in]), **attrs)
    for k in STATES:
        assert np.array_equal(np.asarray(traces[k], dtype=np.int64), delayed[k]), k


def test_the_two_rules_differ_on_the_same_input():
    a_in, attrs = random_case()
    assert not np.array_equal(lava_sigma_delta_trace(a_in, **attrs)["sent"],
                              lava_sigma_delta_trace(a_in, **attrs, corrected=True)["sent"])


def test_extremes_are_the_largest_magnitudes_since_the_reset():
    a_in, attrs = random_case()
    ours = sanafe_sigma_delta_trace(a_in, **attrs)
    lava = lava_sigma_delta_trace(a_in, **attrs)
    for k in STATES:
        assert np.array_equal(ours[f"max_abs_{k}"][-1], np.abs(lava[k]).max(axis=0)), k
    assert np.array_equal(ours["max_abs_input"][-1], np.abs(a_in).max(axis=0))
    # One neuron whose first input is 5, then larger ones. After the reset its first update
    # takes the 5 again.
    chip = _sigma_delta_chip([[5], [40000], [-60000]], **S)
    before = chip.sim(3, neuron_trace=True)["neuron_trace"]
    # sigma is 5, 40005 and -19995, and the sent value 0, 625 and floor(-40005 / 64) = -626
    assert before["max_abs_sigma"][-1] == [40005] and before["max_abs_sent"][-1] == [626]
    chip.reset()
    after = chip.sim(1, neuron_trace=True)["neuron_trace"]
    for k in (*STATES, "input"):
        assert after[f"max_abs_{k}"][-1] == [abs(after[k][-1][0])], k
    assert [after[k][-1][0] for k in (*STATES, "input")] == [5, 5, 5, 0, 5]


def test_the_sigma_soma_accumulates_and_sends_its_sum():
    chip = sdnn_chip([sources([[5, -3, 0, -2]]), ("layer_0", SIGMA_UNIT, [{}], [(0, 1)]),
                      destination()],
                     [("input", 0, "layer_0", 0, 1), ("layer_0", 0, "layer_1", 0, 3)])
    out = run(chip, 6)
    assert out["potential_trace"] == [[0, 0], [5, 0], [2, 15], [2, 21], [0, 27], [0, 27]]


def test_a_sigma_neuron_without_destination_reports_its_sum():
    chip = sdnn_chip([sources([[7]]), ("layer_0", SIGMA_UNIT, [{}], [(0, 1)])],
                     [("input", 0, "layer_0", 0, 1)])
    out = run(chip, 3, neuron_trace=True)
    assert out["neuron_trace"]["sigma"] == [[0], [7], [7]]
    assert out["neuron_trace"]["sent"] == [[0], [7], [7]]


def test_the_sigma_rule_equals_lavas_transcription():
    a_in = [[5], [-3], [0], [-2]]
    t = lava_sigma_trace(a_in)
    assert t["sigma"].ravel().tolist() == t["sent"].ravel().tolist() == [5, 2, 2, 0]


def test_a_sent_value_multiplies_the_weight():
    chip = sdnn_chip([sources([[40000]]), sd_neuron(S), destination()],
                     [("input", 0, "layer_0", 0, 1), ("layer_0", 0, "layer_1", 0, -3)])
    out = run(chip, 4)
    assert column(out, 1) == [[0], [0], [-1875], [-1875]]


def test_a_state_beyond_32_bits_stops_the_run():
    chip = sdnn_chip([sources([[2**30, 2**30]]), sd_neuron(S)], [("input", 0, "layer_0", 0, 1)])
    with pytest.raises(OverflowError, match="sigma 2147483648 leaves signed 32 bits"):
        run(chip, 3)


def test_out_of_range_attributes_are_refused():
    with pytest.raises(ValueError, match="vth"):
        sdnn_chip([sd_neuron(dict(S, vth=-1))], [])
    with pytest.raises(ValueError, match="state_exp"):
        sdnn_chip([sd_neuron(dict(S, state_exp=31))], [])


def test_the_sdnn_architecture_adds_only_the_two_units():
    base = load_loihi2_candidate().configuration()
    derived = load_loihi2_sdnn_candidate().configuration()
    for tile in derived["tiles"]:
        for core in tile["cores"]:
            core["pipeline"] = [u for u in core["pipeline"]
                                if u["name"] not in (SD_UNIT, SIGMA_UNIT)]
    assert derived == base


def test_the_two_units_cost_what_the_gated_unit_costs():
    derived = load_loihi2_sdnn_candidate().configuration()
    for tile in derived["tiles"]:
        for core in tile["cores"]:
            units = {u["name"]: u for u in core["pipeline"]}
            costs = {name: {k: v for k, v in units[name]["attributes"].items() if k != "model"}
                     for name in (SD_UNIT, SIGMA_UNIT, "loihi_if_gated")}
            assert costs[SD_UNIT] == costs[SIGMA_UNIT] == costs["loihi_if_gated"]
            assert units[SD_UNIT]["model"] == "sigma_delta_int32"
            assert units[SIGMA_UNIT]["model"] == "sigma_int32"


def test_the_sdnn_fingerprint_differs_from_the_candidate_and_repeats():
    assert sdnn_fingerprint() != architecture_fingerprint(load_loihi2_candidate())
    assert architecture_fingerprint(load_loihi2_sdnn_candidate()) == sdnn_fingerprint()


def test_the_packaged_candidate_files_are_unchanged():
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert digest(files("sanafe.examples") / "loihi2.yaml") == digest(ROOT / "arch" / "loihi2.yaml") \
        == "fd56fa2df3bc7b8fe4105450d9f8d04667436651bd888785d6fcfd466e2e85dc"
    assert architecture_fingerprint(load_loihi2_candidate()) == \
        "a30997364736668b9b2bb7639c561d808a37ac63bd5a164cd274695d5bc0a849"
