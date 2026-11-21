"""The detailed timing model's bookkeeping of messages in flight, and the network description
after the chip has loaded it.

The detailed timing model keeps, per destination core, the messages that are in the network
and removes each one once the time of the message being scheduled reaches its received
timestamp. The pins below fix the complete output of networks that stress this bookkeeping,
the modeled time included. They were recorded on the build of commit 0594785, before the
bookkeeping visited only the messages that leave, and every later build must reproduce them.

The last tests show that a run does not read the network description after ``chip.load``. A
caller may delete it to free its memory and obtains the same output.
"""
import gc
import tempfile
import weakref
from importlib.resources import files
from pathlib import Path

import numpy as np

import sanafe
from sanafe.loihi2 import load_loihi2_candidate
from test_graded_messages import (BINARY_ACROSS_TILES_DIGEST, DST, digest_of, full_payload,
                                  make_chip, run, sources)

SPREAD_DETAILED_DIGEST = "19e286226f6ae1370b0a9cc0caebe4a18d52e8cc3a6a0b374e2e422572514342"
SPREAD_SIMPLE_DIGEST = "1c90320b5e5ad0b2f45a93fd4d399fbdaac48395c69fef3a074b60a9fe8d498a"
BURST_DIGEST = "0725091507ad9324cc37cb4d0418b07557af9b0fc53074207dab2cf35800ca6f"
BURST_FREE_RECEIVE_DIGEST = "bf87d8c1871ee20cce705d095e3869337d93fb1320249b152ec9bc5ef16f7229"
OUT_OF_ORDER_DIGEST = "03be2550975cf26d7607205779de6da017071da4e72295907d2b5942fb01b66f"


def run_simple(chip, steps):
    """``run`` under the simple timing model."""
    return chip.sim(steps, timing_model="simple", spike_trace=True, potential_trace=True,
                    message_trace=True, perf_trace=True)


def candidate_with(replacements):
    """The candidate architecture loaded from a copy of its description in which each line of
    ``replacements`` (old, new) is replaced. Each old line must occur exactly once."""
    text = (files("sanafe.examples") / "loihi2.yaml").read_text()
    for old, new in replacements:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "loihi2_variant.yaml"
        path.write_text(text)
        return sanafe.load_arch(str(path))


MESSAGE_IN = "latency_message_in: 3.536e-9"
PROCESS_SPIKE = "latency_process_spike: 0.76e-9"


def real_messages(out):
    """The messages of each step that are not placeholders."""
    return [[m for m in step if not m["placeholder"]] for step in out["message_trace"]]


def most_in_flight(messages, core=None):
    """The largest number of messages of one step in flight at the moment one of them is
    sent, counting a message from its send timestamp until its received timestamp, over all
    destination cores or to ``core`` only."""
    if core is not None:
        messages = [m for m in messages if m["dest_core_id"] == core]
    if not messages:
        return 0
    sent = np.sort([m["send_timestamp"] for m in messages])
    received = np.sort([m["received_timestamp"] for m in messages])
    entered = np.searchsorted(sent, sent, side="right")
    left = np.searchsorted(received, sent, side="right")
    return int(np.max(entered - left))


def spread_network():
    """448 neurons over 112 cores on all 32 tiles, every firing neuron reaching many
    cores. ``input``: 64 graded sources, four on core 0 of each of tiles 0 to 15, sending
    random values in [0, 3] for eight steps (a zero sends nothing). Each reaches 48 random
    neurons of ``layer_0``: 256 gated IF neurons checked at every update, four per core on
    cores 1 and 2 of every tile. Each of those reaches 24 random neurons of ``layer_1``: 128
    LIF neurons, four on core 3 of every tile."""
    rng = np.random.default_rng(20261122)
    steps = 8
    values = rng.integers(0, 4, (64, steps))
    hidden = dict(threshold=24, initial_voltage=0, window_start=0, stride=1)
    output = dict(threshold=16, initial_voltage=0)
    groups = [
        sources([list(map(int, v)) for v in values], [(t, 0) for t in range(16) for _ in range(4)]),
        ("layer_0", "loihi_if_gated", [dict(hidden) for _ in range(256)],
         [(t, c) for t in range(32) for c in (1, 2) for _ in range(4)]),
        ("layer_1", "loihi_lif", [dict(output) for _ in range(128)],
         [(t, 3) for t in range(32) for _ in range(4)]),
    ]
    edges = []
    for i in range(64):
        for j, w in zip(rng.choice(256, 48, replace=False), rng.integers(-2, 7, 48)):
            edges.append(("input", i, "layer_0", int(j), int(w)))
    for i in range(256):
        for j, w in zip(rng.choice(128, 24, replace=False), rng.integers(-2, 7, 24)):
            edges.append(("layer_0", i, "layer_1", int(j), int(w)))
    return groups, edges, steps


def burst_network():
    """Sixteen graded sources, two on each core of tiles 0 and 2. The first eight send 1 at
    every step to four ``layer_0`` neurons on core 1 of tile 1. The last eight send nothing
    for three steps, then 2 at steps 4 and 5, then nothing, to four ``layer_1`` neurons on
    core 0 of tile 3, which therefore receives nothing and then a burst of eight messages.
    Sources at the same position on different cores of one tile send at the same time and
    reach the destination tile after the same number of hops."""
    steps = 8
    steady = [[1] * steps for _ in range(8)]
    burst = [[0, 0, 0, 2, 2, 0, 0, 0] for _ in range(8)]
    places = [(t, c) for t in (0, 2) for c in range(4) for _ in range(2)]
    groups = [
        sources(steady + burst, places),
        ("layer_0", "loihi_if_gated", [dict(DST) for _ in range(4)], [(1, 1)] * 4),
        ("layer_1", "loihi_if_gated", [dict(DST) for _ in range(4)], [(3, 0)] * 4),
    ]
    weights = np.random.default_rng(3).integers(1, 9, (16, 4))
    edges = [("input", i, "layer_0", j, weights[i, j]) for i in range(8) for j in range(4)]
    edges += [("input", i, "layer_1", j, weights[i, j]) for i in range(8, 16) for j in range(4)]
    return groups, edges, steps


def test_a_network_spread_over_all_tiles_is_unchanged_bit_for_bit():
    """Detailed timing model. Hundreds of messages are in flight at once, to one core and
    over the chip, and a core's messages arrive from many source cores."""
    groups, edges, steps = spread_network()
    chip, _ = make_chip(groups, edges)
    out = run(chip, steps)
    messages = real_messages(out)
    assert len({m["dest_core_id"] for step in messages for m in step}) == 96
    assert len({m["dest_tile_id"] for step in messages for m in step}) == 32
    assert max(most_in_flight(step) for step in messages) >= 200
    assert max(most_in_flight(step, core) for step in messages for core in range(128)) >= 20
    assert digest_of(full_payload(out)) == SPREAD_DETAILED_DIGEST


def test_scheduler_threads_give_the_pinned_run_of_the_spread_network():
    """Detailed timing model on two scheduler threads, which schedule each step with the same
    function on a thread of their own."""
    groups, edges, steps = spread_network()
    chip, _ = make_chip(groups, edges)
    out = chip.sim(steps, scheduler_threads=2, spike_trace=True, potential_trace=True,
                   message_trace=True, perf_trace=True)
    assert digest_of(full_payload(out)) == SPREAD_DETAILED_DIGEST


def test_the_spread_network_under_the_simple_timing_model_is_unchanged_bit_for_bit():
    """Simple timing model, which sums delays per core and keeps no messages in flight."""
    groups, edges, steps = spread_network()
    chip, _ = make_chip(groups, edges)
    assert digest_of(full_payload(run_simple(chip, steps))) == SPREAD_SIMPLE_DIGEST


def test_messages_arriving_together_and_a_burst_after_silence_are_unchanged_bit_for_bit():
    """Several messages reach one core at the same earliest time (sent at the same time over
    the same number of hops), and a core receives nothing for three steps, then a burst."""
    groups, edges, steps = burst_network()
    chip, _ = make_chip(groups, edges)
    out = run(chip, steps)
    messages = real_messages(out)
    burst_core = messages[3][-1]["dest_core_id"]
    per_step = [sum(m["dest_core_id"] == burst_core for m in step) for step in messages]
    assert per_step == [0, 0, 0, 8, 8, 0, 0, 0]
    earliest = [(m["dest_core_id"], m["send_timestamp"] + m["network_delay"])
                for step in messages for m in step]
    assert len(earliest) > len(set(earliest))
    assert digest_of(full_payload(out)) == BURST_DIGEST


def test_messages_received_at_the_same_instant_are_unchanged_bit_for_bit():
    """The same network on a copy of the candidate whose cores take no time to receive a
    message, so that messages arriving together share one received timestamp and leave the
    network at the same instant."""
    arch = candidate_with([(MESSAGE_IN, "latency_message_in: 0.0"),
                           (PROCESS_SPIKE, "latency_process_spike: 0.0")])
    groups, edges, steps = burst_network()
    chip, _ = make_chip(groups, edges, arch=arch)
    out = run(chip, steps)
    received = [(m["dest_core_id"], m["received_timestamp"])
                for step in real_messages(out) for m in step]
    assert len(received) > len(set(received))
    assert digest_of(full_payload(out)) == BURST_FREE_RECEIVE_DIGEST


def near_and_far_network():
    """Twelve graded sources sending 1 at each of four steps. Four on core 0 of tile 31 reach
    four ``layer_0`` neurons on core 1 of tile 0, ten hops away. Eight on core 0 of tile 0
    alternate between those neurons and four ``layer_1`` neurons on core 2 of tile 0."""
    steps = 4
    places = [(31, 0)] * 4 + [(0, 0)] * 8
    groups = [sources([[1] * steps for _ in range(12)], places),
              ("layer_0", "loihi_if_gated", [dict(DST) for _ in range(4)], [(0, 1)] * 4),
              ("layer_1", "loihi_if_gated", [dict(DST) for _ in range(4)], [(0, 2)] * 4)]
    edges = [("input", i, "layer_0", j, 1) for i in range(4) for j in range(4)]
    for k in range(8):
        target = "layer_0" if k % 2 else "layer_1"
        edges += [("input", 4 + k, target, j, 1) for j in range(4)]
    return groups, edges, steps


def leaves_behind_a_message_that_stays(out):
    """The number of times a message leaves the network while a message scheduled before it,
    to the same core, stays. Messages are scheduled in the order of their send timestamps
    before blocking, and each one scheduled first removes every message received by then."""
    count = 0
    for step in real_messages(out):
        scheduled = [m["send_timestamp"] - m["blocking_delay"] for m in step]
        for a, ka in zip(step, scheduled):
            for b, kb in zip(step, scheduled):
                if (a["dest_core_id"] == b["dest_core_id"] and ka < kb
                        and b["received_timestamp"] < a["received_timestamp"]):
                    count += sum(kb < t and b["received_timestamp"] <= t < a["received_timestamp"]
                                 for t in scheduled)
    return count


def test_a_core_whose_messages_are_received_out_of_order_is_unchanged_bit_for_bit():
    """A negative receive latency, which no architecture should state but which the simulator
    accepts, lets a message from a near source be received before an earlier message from a
    far source on the same core. The scheduler then removes the near message while the far
    one stays, which the scan of a core's whole list handles. With links that buffer no
    message, every message whose route holds messages in flight is blocked by an amount that
    depends on exactly which messages are in flight, so the pin covers that removal."""
    arch = candidate_with([(MESSAGE_IN, "latency_message_in: -3.0e-8"),
                           ("link_buffer_size: 16", "link_buffer_size: 0")])
    groups, edges, steps = near_and_far_network()
    chip, _ = make_chip(groups, edges, arch=arch)
    out = run(chip, steps)
    assert leaves_behind_a_message_that_stays(out) > 0
    assert any(m["blocking_delay"] != 0 for step in real_messages(out) for m in step)
    assert digest_of(full_payload(out)) == OUT_OF_ORDER_DIGEST


def load_with_description(groups, edges):
    """``make_chip`` that also returns the network description, the only Python reference to
    it once this function returns."""
    arch = load_loihi2_candidate()
    network = sanafe.Network()
    made = {}
    for name, unit, attrs_per_neuron, places in groups:
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
    return chip, arch, network


def every_trace(out):
    """The complete output with the neuron trace."""
    return dict(full_payload(out), neuron_trace=out["neuron_trace"])


def kept_and_deleted(groups, edges, steps):
    """Two runs of one network, the first keeping its description, the second deleting
    every Python reference to it after ``chip.load`` and collecting garbage."""
    chip, arch, network = load_with_description(groups, edges)
    kept = run(chip, steps, neuron_trace=True)
    assert network.groups
    del chip, arch, network
    chip, arch, network = load_with_description(groups, edges)
    description = weakref.ref(network)
    del network
    gc.collect()
    assert description() is None
    deleted = run(chip, steps, neuron_trace=True)
    return kept, deleted


def binary_network_across_tiles():
    """The network of ``binary_network((1, 0), strict=True)`` in ``test_graded_messages``."""
    driven = dict(threshold=2, initial_voltage=0, window_start=0, window_stop=1, currents=[2])
    silent = dict(threshold=2, initial_voltage=0, window_start=0, window_stop=1, currents=[0])
    hidden = dict(threshold=6, initial_voltage=3, window_start=1, window_stop=4, stride=1,
                  bias_per_check=1, strict=True)
    output = dict(threshold=4, initial_voltage=0)
    w0 = np.random.default_rng(0).integers(-5, 9, (4, 3))
    w1 = np.random.default_rng(1).integers(-3, 7, (3, 2))
    groups = [
        ("input", "loihi_if_gated", [dict(driven), dict(silent), dict(driven), dict(silent)],
         [(0, 0)] * 4),
        ("layer_0", "loihi_if_gated", [dict(hidden) for _ in range(3)], [(0, 1), (0, 1), (1, 0)]),
        ("layer_1", "loihi_lif", [dict(output) for _ in range(2)], [(2, 0), (2, 0)]),
    ]
    edges = [("input", i, "layer_0", j, w0[i, j]) for i in range(4) for j in range(3)]
    edges += [("layer_0", i, "layer_1", j, w1[i, j]) for i in range(3) for j in range(2)]
    return groups, edges, 6


def test_a_binary_network_runs_the_same_after_its_description_is_deleted():
    """Binary messages across tiles, with strict neurons so that the neuron trace is not
    empty. Every trace is requested."""
    kept, deleted = kept_and_deleted(*binary_network_across_tiles())
    assert digest_of(full_payload(kept)) == BINARY_ACROSS_TILES_DIGEST
    assert set(kept["neuron_trace"]) == {"input_min", "input_max", "voltage_min", "voltage_max"}
    for name in ("spike_trace", "potential_trace", "message_trace", "perf_trace",
                 "neuron_trace", "sim_time", "energy"):
        assert every_trace(deleted)[name] == every_trace(kept)[name], name
    assert digest_of(every_trace(deleted)) == digest_of(every_trace(kept))


def test_a_graded_network_runs_the_same_after_its_description_is_deleted():
    """Graded messages across tiles into a strict neuron and a neuron that holds its input.
    Every trace is requested."""
    strict = dict(DST, strict=1)
    hold = dict(DST, window_start=1, window_stop=4, hold_input=1, strict=1)
    groups = [sources([[5, 0, 3, -2], [0, 7, 1, 4]], [(0, 0), (4, 2)]),
              ("layer_0", "loihi_if_gated", [strict, hold], [(9, 1), (30, 3)])]
    edges = [("input", 0, "layer_0", 0, 3), ("input", 0, "layer_0", 1, -2),
             ("input", 1, "layer_0", 0, 4), ("input", 1, "layer_0", 1, 5)]
    kept, deleted = kept_and_deleted(groups, edges, 6)
    assert {"held_current", "input_max"} <= set(kept["neuron_trace"])
    # The first value, 5, reaches the strict neuron over a weight of 3 one step later.
    assert kept["potential_trace"][1][0] == 15
    for name in ("spike_trace", "potential_trace", "message_trace", "perf_trace",
                 "neuron_trace", "sim_time", "energy"):
        assert every_trace(deleted)[name] == every_trace(kept)[name], name
    assert digest_of(every_trace(deleted)) == digest_of(every_trace(kept))
