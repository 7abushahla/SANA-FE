"""Graded messages: a message that carries a signed integer value, which multiplies the weight
at every synapse it reaches, sent by a gated IF neuron that holds ``graded_values``.

A binary network takes the path it took before graded messages existed, pinned by a digest
of a binary run.
"""
import hashlib
import json
from importlib.resources import files

import numpy as np
import pytest

import sanafe
from sanafe.loihi2 import load_loihi2_candidate
from sanafe.loihi2_counts import input_delivery_counts

# A destination whose threshold is out of reach, so its potential shows every input it sums.
DST = dict(threshold=8388606, initial_voltage=0, window_start=0)


def make_chip(groups, edges, *, arch=None):
    """Return ``(chip, arch)`` for groups of neurons on the Loihi 2 candidate.

    ``groups`` is a list of ``(name, unit, attrs_per_neuron, places)``, with ``places`` one
    ``(tile, core)`` per neuron. ``edges`` is a list of
    ``(src_group, src_index, dst_group, dst_index, weight)``. Every group logs its spikes and
    potentials except a group named ``"input"``. ``arch`` replaces the candidate when given.
    """
    arch = load_loihi2_candidate() if arch is None else arch
    network = sanafe.Network()
    made = {}
    for name, unit, attrs_per_neuron, places in groups:
        assert len(attrs_per_neuron) == len(places), name
        logged = name != "input"
        group = network.create_neuron_group(name, len(attrs_per_neuron),
                                            log_spikes=logged, log_potential=logged)
        for neuron, attrs, (tile, core) in zip(group, attrs_per_neuron, places):
            neuron.set_attributes(soma_hw_name=unit, model_attributes=attrs)
            neuron.map_to_core(arch.tiles[tile].cores[core])
        made[name] = group
    for src_group, src_index, dst_group, dst_index, weight in edges:
        made[src_group][src_index].connect_to_neuron(made[dst_group][dst_index],
                                                     {"weight": int(weight)})
    chip = sanafe.SpikingChip(arch)
    chip.load(network)
    return chip, arch


def run(chip, steps, *, neuron_trace=False):
    """Run ``steps`` updates with the spike, potential, message and perf traces."""
    return chip.sim(steps, spike_trace=True, potential_trace=True, message_trace=True,
                    perf_trace=True, neuron_trace=True if neuron_trace else None)


def sources(values, places=None):
    """An ``input`` group of graded sources, one per list of values, on (0, 0) by default."""
    places = [(0, 0)] * len(values) if places is None else places
    return ("input", "loihi_if_gated", [dict(graded_values=list(v)) for v in values], places)


def destinations(count, places=None):
    """A ``layer_0`` group of ``DST`` neurons, on (0, 1) by default."""
    places = [(0, 1)] * count if places is None else places
    return ("layer_0", "loihi_if_gated", [dict(DST) for _ in range(count)], places)


def binary_network(third, *, strict=False, neuron_trace=False):
    """A binary network of three layers on the candidate, run for six steps.

    ``input``: four gated neurons on (0, 0), neurons 0 and 2 driven once by a current of 2.
    ``layer_0``: three gated neurons on (0, 1), (0, 1) and ``third``, with a window [1, 4),
    in strict mode when ``strict`` is set.
    ``layer_1``: two ungated IF neurons on (2, 0). Every pair of adjacent layers is connected.
    """
    driven = dict(threshold=2, initial_voltage=0, window_start=0, window_stop=1, currents=[2])
    silent = dict(threshold=2, initial_voltage=0, window_start=0, window_stop=1, currents=[0])
    hidden = dict(threshold=6, initial_voltage=3, window_start=1, window_stop=4, stride=1,
                  bias_per_check=1)
    if strict:
        hidden["strict"] = True
    output = dict(threshold=4, initial_voltage=0)
    w0 = np.random.default_rng(0).integers(-5, 9, (4, 3))
    w1 = np.random.default_rng(1).integers(-3, 7, (3, 2))
    groups = [
        ("input", "loihi_if_gated", [dict(driven), dict(silent), dict(driven), dict(silent)],
         [(0, 0)] * 4),
        ("layer_0", "loihi_if_gated", [dict(hidden) for _ in range(3)], [(0, 1), (0, 1), third]),
        ("layer_1", "loihi_lif", [dict(output) for _ in range(2)], [(2, 0), (2, 0)]),
    ]
    edges = [("input", i, "layer_0", j, w0[i, j]) for i in range(4) for j in range(3)]
    edges += [("layer_0", i, "layer_1", j, w1[i, j]) for i in range(3) for j in range(2)]
    chip, _ = make_chip(groups, edges)
    return run(chip, 6, neuron_trace=neuron_trace)


def spikes_of(out):
    return [[[n.group_name, n.neuron_offset] for n in step] for step in out["spike_trace"]]


def digest_of(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def full_payload(out):
    return {
        "spike_trace": spikes_of(out),
        "potential_trace": out["potential_trace"],
        "message_trace": out["message_trace"],
        "perf_trace": out["perf_trace"],
        "sim_time": out["sim_time"],
        "energy": out["energy"],
    }


MESSAGE_FIELDS = (
    "src_neuron_group_id", "src_neuron_offset", "dest_tile_id", "dest_core_offset",
    "placeholder", "spikes",
    "dest_core_id", "dest_axon_hw", "dest_axon_id", "dest_x", "dest_y", "hops",
    "src_tile_id", "src_core_id", "src_core_offset", "src_x", "src_y", "timestep",
    "processing_delay", "blocking_delay",
    "mid", "generation_delay", "network_delay", "send_timestamp", "received_timestamp",
    "processed_timestamp",
)

BINARY_DIGEST = "3ac3cb00d8b9771e29b6ccc542ff3a88c84cec482f76cee202c48bd3a7a96a19"
BINARY_ACROSS_TILES_DIGEST = "1a99292795fe57e2497891bd28eae402454ad3f384ac534462755b3cf1cf1d42"


def fanout_network():
    """Three gated input neurons on (0, 0), each connected to eight ``DST`` neurons of
    ``layer_0`` on core 1 of tiles 7, 3, 5, 1, 6, 2, 4 and 0, listed in that order."""
    driven = dict(threshold=2, initial_voltage=0, window_start=0, window_stop=1, currents=[2])
    tiles = [7, 3, 5, 1, 6, 2, 4, 0]
    weights = np.random.default_rng(2).integers(1, 9, (3, 8))
    groups = [
        ("input", "loihi_if_gated", [dict(driven) for _ in range(3)], [(0, 0)] * 3),
        ("layer_0", "loihi_if_gated", [dict(DST) for _ in tiles], [(t, 1) for t in tiles]),
    ]
    edges = [("input", i, "layer_0", j, weights[i, j]) for i in range(3) for j in range(8)]
    chip, _ = make_chip(groups, edges)
    return run(chip, 3)


def test_a_binary_network_is_unchanged_bit_for_bit():
    # Every fan-out stays inside one tile.
    out = binary_network((0, 2))
    assert digest_of(full_payload(out)) == BINARY_DIGEST


def test_a_binary_network_across_tiles_is_unchanged_bit_for_bit():
    """The same network with the third layer_0 neuron on (1, 0), so that each input neuron
    sends to cores on two tiles. Every field of the run is pinned, the modeled time included."""
    out = binary_network((1, 0))
    assert set(out["message_trace"][0][0]) == set(MESSAGE_FIELDS)
    assert digest_of(full_payload(out)) == BINARY_ACROSS_TILES_DIGEST


def test_a_neuron_sends_to_its_cores_in_ascending_core_id():
    out = fanout_network()
    for step in out["message_trace"]:
        real = [m for m in step if not m["placeholder"]]
        for src in {m["src_neuron_offset"] for m in real}:
            sent = [m for m in real if m["src_neuron_offset"] == src]
            places = [(m["dest_tile_id"], m["dest_core_offset"]) for m in sent]
            assert len(sent) == 8 and places == sorted(places)
            assert [m["mid"] for m in sent] == sorted(m["mid"] for m in sent)


def test_two_constructions_of_a_multi_tile_chip_give_the_same_run():
    runs = [fanout_network() for _ in range(5)]
    assert len({digest_of(full_payload(r)) for r in runs}) == 1
    assert len({r["sim_time"] for r in runs}) == 1


def sent_per_step(out):
    """The number of messages that are not placeholders, per step."""
    return [sum(not m["placeholder"] for m in row) for row in out["message_trace"]]


def test_a_graded_value_multiplies_the_weight():
    chip, _ = make_chip([sources([[5]]), destinations(1)], [("input", 0, "layer_0", 0, 3)])
    out = run(chip, 3)
    assert out["potential_trace"] == [[0], [15], [15]]


def test_negative_values_and_weights_multiply_exactly():
    chip, _ = make_chip([sources([[-7]]), destinations(2)],
                        [("input", 0, "layer_0", 0, 3), ("input", 0, "layer_0", 1, -4)])
    out = run(chip, 2)
    assert out["potential_trace"][1] == [-21, 28]


def test_a_zero_value_sends_nothing():
    chip, _ = make_chip([sources([[0, 4, 0]]), destinations(1)], [("input", 0, "layer_0", 0, 2)])
    out = run(chip, 4)
    assert out["potential_trace"] == [[0], [0], [8], [8]]
    assert sent_per_step(out) == [0, 1, 0, 0]


def test_values_may_differ_from_step_to_step():
    chip, _ = make_chip([sources([[1, 2, 3]]), destinations(1)], [("input", 0, "layer_0", 0, 5)])
    out = run(chip, 4)
    assert out["potential_trace"] == [[0], [5], [15], [30]]


def test_a_graded_spike_reaching_two_cores_carries_its_value_to_both():
    chip, _ = make_chip([sources([[6]]), destinations(2, [(0, 1), (1, 0)])],
                        [("input", 0, "layer_0", 0, 2), ("input", 0, "layer_0", 1, -3)])
    out = run(chip, 2)
    assert out["potential_trace"][1] == [12, -18]
    sent = [m for m in out["message_trace"][0] if not m["placeholder"]]
    assert len(sent) == 2
    assert {(m["dest_tile_id"], m["dest_core_offset"]) for m in sent} == {(0, 1), (1, 0)}


def test_values_beyond_24_bits_travel_exactly():
    # Each value alone is beyond the signed 24-bit input of the soma. Their sum, 5, is not.
    chip, _ = make_chip([sources([[2**30], [-(2**30) + 5]]), destinations(1)],
                        [("input", 0, "layer_0", 0, 1), ("input", 1, "layer_0", 0, 1)])
    out = run(chip, 2)
    assert out["potential_trace"][1] == [5]


def test_a_source_takes_no_synaptic_input():
    second = ("layer_0", "loihi_if_gated", [dict(graded_values=[0])], [(0, 1)])
    chip, _ = make_chip([sources([[3]]), second], [("input", 0, "layer_0", 0, 1)])
    with pytest.raises(ValueError, match="graded source"):
        run(chip, 2)


def test_a_value_beyond_32_bits_is_refused_when_set():
    network = sanafe.Network()
    neuron = network.create_neuron_group("input", 1)[0]
    with pytest.raises(RuntimeError):
        neuron.set_attributes(model_attributes={"graded_values": [1 << 31]})


def test_reset_restarts_the_values_of_a_source():
    chip, _ = make_chip([sources([[5, 1]]), destinations(1)], [("input", 0, "layer_0", 0, 1)])
    first = run(chip, 3)["potential_trace"]
    chip.reset()
    second = run(chip, 3)["potential_trace"]
    assert first == second == [[0], [5], [6]]


def test_an_empty_list_makes_a_source_that_sends_nothing():
    # Without graded_values the current of 2 would make the neuron fire at its first update.
    source = ("layer_0", "loihi_if_gated",
              [dict(graded_values=[], threshold=2, initial_voltage=0, window_start=0,
                    currents=[2, 2])], [(0, 0)])
    target = ("layer_1", "loihi_if_gated", [dict(DST)], [(0, 1)])
    chip, _ = make_chip([source, target], [("layer_0", 0, "layer_1", 0, 3)])
    out = run(chip, 3)
    assert out["potential_trace"] == [[0, 0], [0, 0], [0, 0]]
    assert sent_per_step(out) == [0, 0, 0]


def test_a_source_ignores_threshold_window_biases_and_currents():
    # As an IF neuron these attributes would make it fire at every update of its window.
    source = ("layer_0", "loihi_if_gated",
              [dict(graded_values=[4], threshold=2, initial_voltage=6, window_start=0,
                    window_stop=3, stride=1, bias_at_start=10, bias_per_check=5,
                    currents=[2, 2, 2])], [(0, 0)])
    target = ("layer_1", "loihi_if_gated", [dict(DST)], [(0, 1)])
    chip, _ = make_chip([source, target], [("layer_0", 0, "layer_1", 0, 1)])
    out = run(chip, 4)
    assert out["potential_trace"] == [[6, 0], [6, 4], [6, 4], [6, 4]]
    assert sent_per_step(out) == [1, 0, 0, 0]


def candidate_with_buffer(position, inside):
    """The candidate architecture with every core's time-step buffer moved, loaded from a copy
    of its description in which only the two buffer lines of the core differ."""
    import tempfile
    from pathlib import Path
    text = (files("sanafe.examples") / "loihi2.yaml").read_text()
    stanza = "buffer_position: soma\n            buffer_inside_unit: false"
    assert text.count(stanza) == 1
    moved = text.replace(stanza, f"buffer_position: {position}\n"
                                 f"            buffer_inside_unit: {str(inside).lower()}")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "loihi2_buffer.yaml"
        path.write_text(moved)
        return sanafe.load_arch(str(path))


# A neuron refused where its soma runs on the message path, with the start of its message.
# A strict neuron there would stop on the message path, outside the step that names its layer.
# A held input there would be added to its register once per arriving message.
REFUSED_ON_THE_MESSAGE_PATH = {
    "source": (dict(graded_values=[5, 7]), "A graded source requires"),
    "strict": (dict(threshold=2, initial_voltage=0, window_start=0, currents=[2, 2, 2],
                    strict=True), "A strict neuron requires"),
    "hold": (dict(threshold=2, initial_voltage=0, window_start=0, currents=[2, 2, 2],
                  hold_input=True), "A neuron with hold_input requires"),
}


@pytest.mark.parametrize("kind", sorted(REFUSED_ON_THE_MESSAGE_PATH))
@pytest.mark.parametrize("position, inside, expected, feeder", [
    # Before the axon-out unit the soma runs only when a message arrives, so a source
    # without input would never send.
    ("axon_out", False, 4, False),
    # Inside the soma unit a source reached by a message runs twice per step, so a
    # message of zero weight would make it skip a value.
    ("soma", True, 3, True),
])
def test_a_source_a_strict_or_a_holding_neuron_is_refused_where_its_soma_runs_on_the_message_path(
        position, inside, expected, feeder, kind):
    attrs, message = REFUSED_ON_THE_MESSAGE_PATH[kind]
    arch = candidate_with_buffer(position, inside)
    assert int(arch.tiles[0].cores[0].buffer_position) == expected
    groups = [("input", "loihi_if_gated", [dict(attrs)], [(0, 2)]),
              destinations(1)]
    edges = [("input", 0, "layer_0", 0, 3)]
    if feeder:
        groups.append(("feeder", "loihi_if_gated",
                       [dict(threshold=2, initial_voltage=0, window_start=0,
                             currents=[2, 2, 2])], [(0, 0)]))
        edges.append(("feeder", 0, "input", 0, 0))
    chip, _ = make_chip(groups, edges, arch=arch)
    with pytest.raises(ValueError, match=r"input\[0\]: " + message):
        run(chip, 3)


EXTREMES = ("input_min", "input_max", "voltage_min", "voltage_max")
# One update takes the membrane to 8388610, beyond signed 24 bits, before any threshold check.
BEYOND = dict(threshold=8388606, initial_voltage=8388600, window_start=5, window_stop=6,
              currents=[10])


def one_neuron(attrs, name="layer_0"):
    return (name, "loihi_if_gated", [attrs], [(0, 1)])


def test_strict_stops_when_the_membrane_leaves_24_bits():
    chip, _ = make_chip([one_neuron(dict(BEYOND, strict=True))], [])
    with pytest.raises(OverflowError, match=r"layer_0\[0\]: strict mode: membrane 8388610 "
                                            r"leaves the signed 24-bit range"):
        run(chip, 1)
    chip, _ = make_chip([one_neuron(dict(BEYOND))], [])
    non_strict = run(chip, 1)
    # The same neuron without strict saturates.
    assert non_strict["potential_trace"] == [[8388607]]


def test_strict_stops_when_the_input_leaves_24_bits():
    # The input of 2^23 reaches the neuron as a graded value of 2^23 on a weight of 1.
    def chip_with(strict):
        target = one_neuron(dict(DST, strict=True) if strict else dict(DST))
        chip, _ = make_chip([sources([[8388608]]), target], [("input", 0, "layer_0", 0, 1)])
        return chip
    with pytest.raises(OverflowError, match=r"input 8388608"):
        run(chip_with(True), 2)
    # Without strict the input is refused as before.
    with pytest.raises(ValueError, match="signed 24"):
        run(chip_with(False), 2)


def test_strict_names_the_group_and_offset():
    inside = dict(threshold=6, initial_voltage=0, window_start=0, currents=[10], strict=True)
    group = ("layer_3", "loihi_if_gated", [dict(inside), dict(inside), dict(BEYOND, strict=True)],
             [(0, 1)] * 3)
    chip, _ = make_chip([group], [])
    with pytest.raises(OverflowError, match=r"^layer_3\[2\]: "):
        run(chip, 1)


def test_extremes_are_recorded_before_the_reset_and_cleared_by_it():
    attrs = dict(threshold=6, initial_voltage=0, window_start=0, currents=[3, -5, 10, 0],
                 strict=True)
    chip, _ = make_chip([one_neuron(attrs)], [])
    # v = 3, -2, 8 (a spike, leaving 2), 2
    out = run(chip, 4, neuron_trace=True)
    assert out["potential_trace"] == [[3], [-2], [2], [2]]
    last = {k: out["neuron_trace"][k][-1] for k in EXTREMES}
    assert last == {"input_min": [-5], "input_max": [10], "voltage_min": [-2], "voltage_max": [8]}
    chip.reset()
    again = run(chip, 1, neuron_trace=True)
    assert [again["neuron_trace"][k][-1] for k in EXTREMES] == [[3]] * 4


def test_a_neuron_without_strict_reports_no_trace():
    chip, _ = make_chip([sources([[4]]), destinations(1)], [("input", 0, "layer_0", 0, 3)])
    assert run(chip, 2, neuron_trace=True)["neuron_trace"] == {}


def test_a_strict_source_reports_no_trace():
    source = ("input", "loihi_if_gated", [dict(graded_values=[4], strict=True)], [(0, 0)])
    chip, _ = make_chip([source, destinations(1)], [("input", 0, "layer_0", 0, 3)])
    assert run(chip, 2, neuron_trace=True)["neuron_trace"] == {}


def test_strict_inside_24_bits_changes_nothing():
    plain = binary_network((0, 2))
    strict = binary_network((0, 2), strict=True)
    assert strict["potential_trace"] == plain["potential_trace"]
    assert spikes_of(strict) == spikes_of(plain)
    # Strict mode adds no cost, so the whole run still matches the pinned digest.
    assert digest_of(full_payload(strict)) == BINARY_DIGEST
    # Strict mode was on: each of the three layer_0 neurons reports its four extremes at
    # every step, and no other neuron reports any.
    traced = binary_network((0, 2), strict=True, neuron_trace=True)
    assert set(traced["neuron_trace"]) == set(EXTREMES)
    for name in EXTREMES:
        assert [len(row) for row in traced["neuron_trace"][name]] == [3] * 6, name


# A destination that holds its input current over the window [1, 4), in strict mode.
HOLD = dict(DST, window_start=1, window_stop=4, hold_input=1, strict=1)


def held_run(source_values, destination, steps):
    chip, _ = make_chip([sources([source_values]), one_neuron(destination)],
                        [("input", 0, "layer_0", 0, 3)])
    return chip, run(chip, steps, neuron_trace=True)


def test_a_value_sent_once_drives_every_step_of_the_window():
    _, out = held_run([5], HOLD, 6)
    assert out["potential_trace"] == [[0], [15], [30], [45], [45], [45]]
    assert out["neuron_trace"]["held_current"] == [[0], [15], [15], [15], [15], [15]]


def test_without_hold_input_the_neuron_is_unchanged():
    _, out = held_run([5], dict(HOLD, hold_input=0), 6)
    assert out["potential_trace"] == [[0], [15], [15], [15], [15], [15]]
    # Off, the neuron reports the traces of the strict mode and no held current.
    assert set(out["neuron_trace"]) == set(EXTREMES)


def test_reset_clears_the_held_current():
    chip, _ = held_run([5], HOLD, 6)
    chip.reset()
    again = run(chip, 1, neuron_trace=True)
    assert again["potential_trace"] == [[0]] and again["neuron_trace"]["held_current"] == [[0]]


def test_input_after_the_window_bypasses_the_held_current():
    _, out = held_run([5, 0, 0, 0, 0, 2], HOLD, 8)
    assert out["potential_trace"] == [[0], [15], [30], [45], [45], [45], [51], [51]]
    assert out["neuron_trace"]["held_current"][-1] == [15]


def test_values_arriving_at_the_two_edges_of_the_window():
    # Each value reaches the destination one update after it is sent, as an input of 3 times
    # the value. The 5 sent at update 2 arrives at window_start 3 as 15 and is held from there.
    # The 1 sent at update 5 arrives at update 6, the last of the window, and the register
    # becomes 18. The 2 sent at update 6 arrives at window_stop 7 as 6, which leaves the
    # register at 18 and is added once.
    destination = dict(DST, window_start=3, window_stop=7, hold_input=1, strict=1)
    _, out = held_run([0, 0, 5, 0, 0, 1, 2], destination, 10)
    assert out["potential_trace"] == [[0], [0], [0], [15], [30], [45], [63], [69], [69], [69]]
    assert out["neuron_trace"]["held_current"] == [[0], [0], [0], [15], [15], [15], [18], [18],
                                                   [18], [18]]


def test_the_two_deliveries_give_identical_membranes():
    destination = dict(threshold=40, initial_voltage=20, window_start=1, window_stop=4)
    chip, _ = make_chip([sources([[5, 5, 5]]), one_neuron(dict(destination, hold_input=0))],
                        [("input", 0, "layer_0", 0, 3)])
    a = run(chip, 6)
    chip, _ = make_chip([sources([[5]]), one_neuron(dict(destination, hold_input=1))],
                        [("input", 0, "layer_0", 0, 3)])
    b = run(chip, 6)
    assert a["potential_trace"] == b["potential_trace"] == [[20], [35], [10], [25], [25], [25]]
    assert ([len(step) for step in spikes_of(a)] == [len(step) for step in spikes_of(b)]
            == [0, 0, 1, 0, 0, 0])


def test_strict_stops_when_the_held_current_leaves_24_bits():
    attrs = dict(threshold=8388606, initial_voltage=-8000000, window_start=0, window_stop=4,
                 currents=[4194304, 4194304], hold_input=1)
    chip, _ = make_chip([one_neuron(dict(attrs, strict=1))], [])
    with pytest.raises(OverflowError, match=r"^layer_0\[0\]: strict mode: held current 8388608 "
                                            r"leaves the signed 24-bit range"):
        run(chip, 2)
    # Without strict the held current saturates at the top of signed 24 bits.
    chip, _ = make_chip([one_neuron(attrs)], [])
    out = run(chip, 2, neuron_trace=True)
    assert out["neuron_trace"] == {"held_current": [[4194304], [8388607]]}
    assert out["potential_trace"] == [[-3805696], [4582911]]


def test_a_source_does_not_take_hold_input():
    source = ("input", "loihi_if_gated", [dict(graded_values=[5], hold_input=1)], [(0, 0)])
    with pytest.raises(ValueError, match="graded source does not take hold_input"):
        make_chip([source, destinations(1)], [("input", 0, "layer_0", 0, 3)])
    # Set to 0 it changes nothing.
    source = ("input", "loihi_if_gated", [dict(graded_values=[5], hold_input=0)], [(0, 0)])
    chip, _ = make_chip([source, destinations(1)], [("input", 0, "layer_0", 0, 3)])
    assert run(chip, 3)["potential_trace"] == [[0], [15], [15]]


def test_input_delivery_counts_separate_the_source():
    # The source in0 sends 4 to n0 on (0, 1) and to n2 on (1, 0), so two messages and two
    # synaptic events. in1 sends 0 and so nothing. in2 sends -2 to n2 and n3, which share the
    # core (1, 0), so one message and two synaptic events. At the next update n0 receives 4
    # and n2 receives 4 - 2 = 2, so both reach their threshold of 2 inside the window and
    # each sends one message with one synaptic event to the output neuron. n1 and n3 do not.
    hidden = dict(threshold=2, initial_voltage=0, window_start=1, window_stop=2)
    groups = [sources([[4], [0], [-2]]),
              ("layer_0", "loihi_if_gated", [dict(hidden) for _ in range(4)],
               [(0, 1), (0, 1), (1, 0), (1, 0)]),
              ("layer_1", "loihi_if_gated", [dict(DST)], [(2, 0)])]
    edges = [("input", 0, "layer_0", 0, 1), ("input", 0, "layer_0", 2, 1),
             ("input", 1, "layer_0", 1, 1), ("input", 2, "layer_0", 2, 1),
             ("input", 2, "layer_0", 3, 1), ("layer_0", 0, "layer_1", 0, 1),
             ("layer_0", 2, "layer_1", 0, 1)]
    chip, _ = make_chip(groups, edges)
    out = run(chip, 3)
    counts = input_delivery_counts(out, ["input"])
    assert counts["source_groups"] == ["input"] and counts["steps"] == 3
    assert counts["input"] == {"messages": 3, "synaptic_events": 4,
                               "messages_per_step": [3, 0, 0],
                               "synaptic_events_per_step": [4, 0, 0]}
    assert counts["layers"] == {"messages": 2, "synaptic_events": 2}
    # Naming no source group counts every message under the layers.
    assert input_delivery_counts(out, [])["layers"] == {"messages": 5, "synaptic_events": 6}


def test_input_delivery_counts_need_a_message_trace():
    with pytest.raises(ValueError, match="message trace"):
        input_delivery_counts({"message_trace": None}, ["input"])


def test_input_delivery_counts_take_a_string_as_one_group_name():
    chip, _ = make_chip([sources([[4]]), destinations(1)], [("input", 0, "layer_0", 0, 1)])
    out = run(chip, 2)
    assert input_delivery_counts(out, "input")["input"]["messages"] == 1
    assert input_delivery_counts(out, "input")["source_groups"] == ["input"]
    assert input_delivery_counts(out, ["absent"])["input"]["messages"] == 0
