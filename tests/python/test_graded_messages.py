"""Graded messages: a message that carries a signed integer value, which multiplies the weight
at every synapse it reaches, sent by a gated IF neuron that holds ``graded_values``.

A binary network takes the path it took before graded messages existed, pinned by a digest
of a binary run.
"""
import hashlib
import json
import math
from importlib.resources import files

import numpy as np
import pytest

import sanafe
from sanafe.loihi2 import load_loihi2_candidate

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


def binary_network(third):
    """A binary network of three layers on the candidate, run for six steps.

    ``input``: four gated neurons on (0, 0), neurons 0 and 2 driven once by a current of 2.
    ``layer_0``: three gated neurons on (0, 1), (0, 1) and ``third``, with a window [1, 4).
    ``layer_1``: two ungated IF neurons on (2, 0). Every pair of adjacent layers is connected.
    """
    driven = dict(threshold=2, initial_voltage=0, window_start=0, window_stop=1, currents=[2])
    silent = dict(threshold=2, initial_voltage=0, window_start=0, window_stop=1, currents=[0])
    hidden = dict(threshold=6, initial_voltage=3, window_start=1, window_stop=4, stride=1,
                  bias_per_check=1)
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
    return run(chip, 6)


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


# Message fields that are reproducible when a neuron sends to cores on several tiles, and the
# fields that are not. The first six order the messages of a step.
STABLE_MESSAGE_FIELDS = (
    "src_neuron_group_id", "src_neuron_offset", "dest_tile_id", "dest_core_offset",
    "placeholder", "spikes",
    "dest_core_id", "dest_axon_hw", "dest_axon_id", "dest_x", "dest_y", "hops",
    "src_tile_id", "src_core_id", "src_core_offset", "src_x", "src_y", "timestep",
    "processing_delay", "blocking_delay",
)
UNSTABLE_MESSAGE_FIELDS = (
    "mid", "generation_delay", "network_delay", "send_timestamp", "received_timestamp",
    "processed_timestamp",
)


def reproducible_payload(out):
    messages = [sorted([m[name] for name in STABLE_MESSAGE_FIELDS] for m in step)
                for step in out["message_trace"]]
    perf = {name: values for name, values in out["perf_trace"].items() if name != "sim_time"}
    return {
        "spike_trace": spikes_of(out),
        "potential_trace": out["potential_trace"],
        "message_multisets": messages,
        "perf_trace": perf,
        "energy": out["energy"],
    }


BINARY_DIGEST = "3ac3cb00d8b9771e29b6ccc542ff3a88c84cec482f76cee202c48bd3a7a96a19"
BINARY_ACROSS_TILES_DIGEST = "c4b45396177b73d8886b896204129873087647c0b06ff5dadd98f9ac17a9a0a0"


def test_a_binary_network_is_unchanged_bit_for_bit():
    # Every fan-out stays inside one tile, so every field of the run is reproducible.
    out = binary_network((0, 2))
    assert digest_of(full_payload(out)) == BINARY_DIGEST


def test_a_binary_network_across_tiles_is_unchanged_in_every_reproducible_field():
    """The same network with the third layer_0 neuron on (1, 0), so that each input neuron
    sends to cores on two tiles.

    The order in which a neuron's messages to different tiles are emitted is the pointer
    order of a set of cores, which differs from one chip construction to the next on this
    build. Left out for that reason are the message fields ``mid``, ``generation_delay``,
    ``network_delay``, ``send_timestamp``, ``received_timestamp`` and
    ``processed_timestamp``, the order of the messages within a step (each step is compared
    as a sorted multiset), the perf trace's ``sim_time`` and the run's ``sim_time``. Every
    other message field, every other perf field, the spike and potential traces and the
    energy are pinned.
    """
    out = binary_network((1, 0))
    assert set(out["message_trace"][0][0]) == (set(STABLE_MESSAGE_FIELDS)
                                               | set(UNSTABLE_MESSAGE_FIELDS))
    assert math.isfinite(out["sim_time"]) and out["sim_time"] > 0
    assert digest_of(reproducible_payload(out)) == BINARY_ACROSS_TILES_DIGEST


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


@pytest.mark.parametrize("position, inside, expected, feeder", [
    # Before the axon-out unit the soma runs only when a message arrives, so a source
    # without input would never send.
    ("axon_out", False, 4, False),
    # Inside the soma unit a source reached by a message runs twice per step, so a
    # message of zero weight would make it skip a value.
    ("soma", True, 3, True),
])
def test_a_source_is_refused_where_its_soma_runs_on_the_message_path(
        position, inside, expected, feeder):
    arch = candidate_with_buffer(position, inside)
    assert int(arch.tiles[0].cores[0].buffer_position) == expected
    groups = [("input", "loihi_if_gated", [dict(graded_values=[5, 7])], [(0, 2)]),
              destinations(1)]
    edges = [("input", 0, "layer_0", 0, 3)]
    if feeder:
        groups.append(("feeder", "loihi_if_gated",
                       [dict(threshold=2, initial_voltage=0, window_start=0,
                             currents=[2, 2, 2])], [(0, 0)]))
        edges.append(("feeder", 0, "input", 0, 0))
    chip, _ = make_chip(groups, edges, arch=arch)
    with pytest.raises(ValueError, match=r"input\[0\]: A graded source requires"):
        run(chip, 3)
