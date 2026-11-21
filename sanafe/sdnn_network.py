"""Sigma-delta networks from a netx file, in SANA-FE and in Lava.

``read_sdnn`` reads a netx file whose neurons are of type ``SDNN``, with the meaning lava-dl's
netx gives every field. ``sdnn_to_sanafe`` builds a SANA-FE network on the derived
architecture of ``sanafe.lava_sdnn``. The input layer runs on the host, so its neurons become
graded sources that send the values of the host's encoder. Every other layer runs the
``sigma_delta_int32`` soma, and a ``sigma_output`` layer runs the ``sigma_int32`` soma.
``sdnn_sanafe_run`` runs the simulator and ``sdnn_lava_run`` runs the same file in Lava's
fixed-point simulation, built as the PilotNet SDNN tutorial of lava-dl builds it. Both return
the states of every layer in the same form, and ``compare_sdnn`` compares them. Equality with
Lava's fixed-point models is what the tests establish. Nothing here is validated against
Loihi 2 hardware.

References are to lava-dl at commit b9e3691 (``src/lava/lib/dl/netx/hdf5.py``, written
``hdf5.py``, and the tutorial folder ``tutorials/lava/lib/dl/netx/pilotnet_sdnn``) and to
Lava at commit 74840cb8 (``src/lava/proc``).

Conventions
-----------
* Layers are the layers of the file that have a neuron, in file order, as in
  ``sanafe.lif_network``. Flatten layers have none and only fix the order in which a dense
  layer reads the previous layer. Layer 0 is the input layer. netx builds it and drops it
  (``skip_layers=1``, ``hdf5.py`` 495-500), so on the chip it is the group of graded sources
  and in Lava it is the host's encoder.
* Neurons of a layer are indexed in flattened (channels, height, width) order, the order of
  the reader. Every returned array has shape ``[steps, n]`` with that column order. Row ``t``
  is the state after simulator update ``t``. Where a neuron sits is a separate matter, decided
  by the placement and the partition as in ``netx_to_sanafe``.
* Scale. netx gives every sigma-delta layer ``vth = vThMant``, ``spike_exp = 6`` and
  ``state_exp = 6`` (``hdf5.py`` 154-174), and Lava's ``SigmaDelta`` multiplies ``vth`` and
  the bias by ``2**(spike_exp + state_exp)`` (``proc/sdn/process.py`` 179-180). The reader
  stores ``vth = vThMant << 12`` and ``bias_scale = 4096``, and the soma receives ``vth`` and
  ``bias_scale * bias``. An absent ``messageBits`` is netx's SDNN default of 16, and the
  ``activation`` dataset, which netx never reads, must say ``relu``, the default of Lava's
  ``SigmaDelta``.
* A value sent by layer ``k - 1`` at update ``t`` reaches layer ``k`` at update ``t + 1``.
  Lava's ``Conv`` and ``Dense`` send at each step what they accumulated at the previous one
  (``proc/conv/models.py`` 38, ``proc/dense/models.py`` 93-96), and the buffer of the derived
  architecture before the soma unit gives the same delay of one update.

Alignment of the chain
----------------------
Lava counts its steps from 1 and the simulator its updates from 0, so simulator update ``t``
is Lava step ``t + 1``. The chain of the tutorial's simulation (``utils.py`` 112-125) is a
data source, the ``DeltaEncoder`` with its dense model, then the network. Here a
``RingBuffer`` is the data source. At step ``s`` it sends column ``(s - 1) mod T`` of its data
(``proc/io/source.py`` 41-43), so row ``t`` of ``frames`` reaches the encoder at step
``t + 1``. ``PyDeltaEncoderModelDense.run_spk`` (``proc/io/encoder.py`` 145-149) first sends
the buffer it computed at the previous step and then encodes what it received. So at step 1
it sends zeros, and at step ``t + 1`` it sends the encoding of frame ``t - 1``. Row ``t`` of
the source values is the value the encoder sends at Lava step ``t + 1``, so row 0 of values
taken from Lava is zero. A graded source sends row ``t`` at simulator update ``t``, the step
at which the encoder sends it. ``Conv`` of the first layer receives that value at the same
step and hands it to ``SigmaDelta`` at the next, and the buffer before the soma does the same
in the simulator, so layer 1 integrates row ``t`` at update ``t + 1`` in both. Each further
layer adds one update in both. A sigma-delta neuron therefore runs its first update on its
bias alone in both, and a bias at or above the threshold sends at update 0 in both. Row ``t``
of every layer of one run is compared with row ``t`` of the other, row 0 included. Lava's
dataloader, which the tutorial uses on the real frames, adds one step of its own in front
(``proc/io/dataloader.py`` 169-199). That step belongs to the frames and not to this
alignment.

The encoder
-----------
The tutorial's simulation encodes with ``DeltaEncoder(shape, vth=vThMant, spike_exp=6)`` and
``PyDeltaEncoderModelDense`` (``utils.py`` 112-125). Its threshold is ``vThMant << 6``, and
``num_bits`` is not set, so nothing is clipped (``proc/io/encoder.py`` 83, 92-98, 119-126).
Its values are pixel changes shifted left by 6 with the residue of earlier changes, and they
are the values of the graded sources. The tutorial's hardware run encodes differently, on the
host with 8-bit compression (``utils.py`` 128-146). That path is not modeled.

The two fixed-point models
--------------------------
Lava's simulation runs ``PySigmaDeltaModelFixed``, the first model carrying the tag
``fixed_pt`` (see ``sanafe.lava_sdnn``). ``model="corrected"`` selects
``PySigmaDeltaModelFixedCorrected`` instead, on both sides. In Lava a run configuration whose
``select`` returns the chosen class for every ``SigmaDelta`` does it, and in the simulator the
soma's ``corrected`` attribute. The two models send the same values whenever every input and
bias of a neuron is a multiple of ``2**state_exp``. The encoder sends multiples of 64 and the
biases are multiples of 4096, so the first layer after the sources cannot tell the models
apart, and a difference shows from the second layer on.

Reading Lava's state
--------------------
Lava's ``Var.get`` and ``Monitor.get_data`` transfer an array one element per message, which
deadlocked on a large layer (``sanafe.lif_network``). ``sdnn_lava_run`` therefore taps
``sigma``, ``act`` and ``residue`` of every layer with ``VarWire`` and the ``s_out`` of the
encoder and of every layer with ``Extractor``, in chunks of at most 16 steps, with the stall
rule and the shutdown of ``lif_network.lava_run``. Lava has no Var for the input of an
update, so it is read as the change of ``sigma`` from one step to the next.

Range
-----
The somas hold 32-bit states and stop where Lava's int32 arithmetic would wrap. Lava's ``Conv``
wraps the sum of one projection at signed 24 bits (``proc/conv/utils.py`` 67-84), while the
integer accumulator of the simulator does not. A sum beyond 24 bits therefore makes the two
differ, and the soma's ``max_abs_input`` shows it. ``fits`` reads the extremes against 24 and
16 bits.

Costs
-----
The derived architecture charges the costs of the gated soma unit. Every time and energy of a
run rests on ``COST_ASSUMPTIONS``, which the manifest records, and excludes the transfer of
the input from the host.
"""
import hashlib
import time
from dataclasses import asdict

import h5py
import numpy as np
import sanafe

from sanafe.lava_integer import _integers, _scalar_integer
from sanafe.lava_sdnn import (COST_ASSUMPTIONS, SDNN_EXECUTION_PROFILE, SDNN_STATES,
                              SIGMA_DELTA_SOMA_UNIT, SIGMA_SOMA_UNIT, check_sdnn_architecture,
                              check_sdnn_core, sdnn_architecture_profile)
from sanafe.lif_network import (_CONNECT_CHUNK, RESERVED_CORES, _connection, _flatten_rows,
                                _group_name, _lava_shape, _neuron_layers, _per_core,
                                _placement, _receive, _slice_index, _terminate_lava_children,
                                neuron_positions)
from sanafe.loihi2 import Allocation
from sanafe.netx import (_ALLOWED, NetxLayer, NetxNetwork, _check_keys, _read, _read_conv,
                         _read_dense, _text)

#: The fixed-point sigma-delta models of Lava that a run may select.
MODELS = ("fixed", "corrected")

_INT32 = (-(1 << 31), (1 << 31) - 1)
_INT32_MAX = _INT32[1]
_SPIKE_EXP = 6
_STATE_EXP = 6
_MESSAGE_BITS = 16
_SOURCE_UNIT = "loihi_if_gated"
_SDNN_NEURON_KEYS = {"type", "activation", "vThMant", "sigma_output", "messageBits"}
_SIGMA_STATES = ("sigma", "sent", "input")
_LAVA_MODELS = {"fixed": "PySigmaDeltaModelFixed",
                "corrected": "PySigmaDeltaModelFixedCorrected"}


# ---------------------------------------------------------------------------
# The reader


def _sdnn_neuron(group, index):
    """The neuron of a layer, with the values netx passes to Lava (``hdf5.py`` 154-174)."""
    kind = _text(group["type"]) if "type" in group else None
    if kind != "SDNN":
        raise ValueError(f"layer {index}: neuron type '{kind}' is not implemented, only the "
                         "'SDNN' sigma-delta neuron of netx files")
    unknown = set(group.keys()) - _SDNN_NEURON_KEYS
    if unknown:
        raise ValueError(f"layer {index}: neuron attribute(s) {sorted(unknown)} are not "
                         "implemented")
    activation = _text(group["activation"]) if "activation" in group else "relu"
    if activation != "relu":
        raise ValueError(f"layer {index}: neuron activation '{activation}' is not "
                         "implemented. netx never reads it and Lava's SigmaDelta applies "
                         "the ReLU")
    if "messageBits" in group and int(_read(group["messageBits"])) != _MESSAGE_BITS:
        raise ValueError(f"layer {index}: neuron messageBits {int(_read(group['messageBits']))} "
                         f"is not implemented, only {_MESSAGE_BITS}, netx's default for SDNN")
    sigma_output = "sigma_output" in group
    if sigma_output and not bool(_read(group["sigma_output"])):
        raise ValueError(f"layer {index}: a stored sigma_output of False is not implemented. "
                         "netx builds a Sigma layer whenever the key is present")
    if "vThMant" not in group:
        if not sigma_output:
            raise ValueError(f"layer {index}: required neuron attribute 'vThMant' is missing")
        vth_mant = None
    else:
        vth_mant = int(_read(group["vThMant"]))
    shift = _SPIKE_EXP + _STATE_EXP
    return {"type": "SDNN", "vth_mant": vth_mant, "spike_exp": _SPIKE_EXP,
            "state_exp": _STATE_EXP, "vth": None if vth_mant is None else vth_mant << shift,
            "bias_scale": 1 << shift, "message_bits": _MESSAGE_BITS, "activation": "relu",
            "sigma_output": sigma_output}


def read_sdnn(path) -> NetxNetwork:
    """Read a netx file of sigma-delta neurons.

    Returns a ``NetxNetwork`` with the conventions of ``sanafe.netx``. Every layer with a
    neuron gets ``neuron = {"type": "SDNN", "vth_mant", "spike_exp": 6, "state_exp": 6,
    "vth": vth_mant << 12, "bias_scale": 4096, "message_bits": 16, "activation": "relu",
    "sigma_output": bool}``, the values netx gives (``hdf5.py`` 154-174). The input layer
    keeps its neuron and its ``transform`` (``{"weight": 64, "bias": 0}`` when the file
    stores none). ``bias`` holds the file's integers, unscaled. A ``ValueError`` refuses a
    neuron type other than ``SDNN``, an activation other than ``relu``, a ``sigma_output``
    on any layer but the last, a stored ``messageBits`` other than 16, a bias on the
    ``sigma_output`` layer, an input layer without a neuron, and everything ``sanafe.netx``
    refuses.
    """
    layers = []
    with h5py.File(path, "r") as f:
        group = f["layer"]
        for index in range(len(group)):
            lg = group[str(index)]
            kind = _text(lg["type"])
            if kind not in _ALLOWED:
                raise ValueError(f"layer {index}: layer type '{kind}' is not implemented")
            _check_keys(lg, kind, index)
            layer = NetxLayer(kind=kind, shape=tuple(int(x) for x in _read(lg["shape"])))
            if kind == "flatten":
                layers.append(layer)
                continue
            if "neuron" not in lg:
                # Without a neuron netx leaves the first convolution's input binary.
                raise ValueError(f"layer {index} ({kind}): no neuron group")
            layer.neuron = _sdnn_neuron(lg["neuron"], index)
            if kind == "input":
                # hdf5.py 261-271. netx stores the transform and never applies it.
                layer.transform = {"weight": int(_read(lg["weight"])) if "weight" in lg else 64,
                                   "bias": int(_read(lg["bias"])) if "bias" in lg else 0}
            elif kind == "conv":
                _read_conv(lg, layer, index)
            else:
                _read_dense(lg, layer, index)
            layers.append(layer)
    with_neuron = [i for i, layer in enumerate(layers) if layer.neuron is not None]
    for i in with_neuron:
        if not layers[i].neuron["sigma_output"]:
            continue
        if i != with_neuron[-1]:
            raise ValueError(f"layer {i}: sigma_output on a layer other than the last is not "
                             "implemented")
        if layers[i].bias is not None:
            raise ValueError(f"layer {i}: a bias on the sigma_output layer is not implemented, "
                             "since netx's Sigma takes none")
    return NetxNetwork(layers=layers)


# ---------------------------------------------------------------------------
# The builder


def _sdnn_entries(network):
    entries = _neuron_layers(network)
    for entry in entries:
        if entry["layer"].neuron.get("type") != "SDNN":
            raise ValueError(f"layer {entry['file_index']}: not a sigma-delta neuron (read the "
                             "file with read_sdnn)")
    return entries


def _source_array(source_values, n_input):
    values = np.asarray(source_values)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != n_input:
        raise ValueError(f"source_values has shape {values.shape}, expected [steps, {n_input}] "
                         "with one column per input neuron in reader order")
    return _integers(values, "source_values", *_INT32)


def _layer_bias(entry):
    """``bias_scale`` times the file's bias, one per neuron in reader order."""
    layer, size = entry["layer"], entry["size"]
    if layer.bias is None:
        return np.zeros(size, dtype=np.int64)
    bias = np.asarray(layer.bias, dtype=np.int64).reshape(-1) * layer.neuron["bias_scale"]
    if entry["kind"] == "conv":
        channels = entry["shape"][0]
        if bias.size != channels:
            raise ValueError(f"a convolution bias has {bias.size} entries, the layer "
                             f"{channels} channels")
        bias = np.repeat(bias, size // channels)
    elif bias.size != size:
        raise ValueError(f"a dense bias has {bias.size} entries, the layer {size} neurons")
    return _integers(bias, f"layer {entry['file_index']}: scaled bias", *_INT32)


def _soma(entry, corrected):
    """The soma unit and the group attributes of a layer with a neuron."""
    neuron = entry["layer"].neuron
    if entry["kind"] == "input":
        return _SOURCE_UNIT, {}
    if neuron["sigma_output"]:
        return SIGMA_SOMA_UNIT, {}
    if neuron["vth"] is None or not 0 <= neuron["vth"] <= _INT32_MAX:
        raise ValueError(f"layer {entry['file_index']}: vth {neuron['vth']} outside "
                         f"[0, {_INT32_MAX}]")
    return SIGMA_DELTA_SOMA_UNIT, {"vth": int(neuron["vth"]),
                                   "state_exp": int(neuron["state_exp"]), "corrected": corrected}


def sdnn_to_sanafe(network, source_values, arch, layer_cores, *, order="packed_reversed",
                   partition=None, model="fixed", build=True):
    """Return ``(Network, manifest)`` for a sigma-delta network on the derived architecture.

    ``network`` is from ``read_sdnn``. ``source_values`` is an integer array
    ``[steps, n_input]``, column ``j`` the values of input neuron ``j`` in reader order and row
    ``t`` the value sent at update ``t`` (see the module docstring for its alignment with
    Lava's encoder). Group ``layer_00`` holds one graded source per input neuron on the unit
    ``loihi_if_gated``. The groups ``layer_01`` onward hold the other layers, a
    ``sigma_output`` layer on ``loihi_sigma`` and every other layer on ``loihi_sigma_delta``
    with the reader's ``vth``, the scaled bias (per channel for a convolution), ``state_exp``
    and ``corrected = (model == "corrected")``. Spikes and potentials are not logged.
    ``layer_cores``, ``order`` and ``partition`` place the neurons as in ``netx_to_sanafe``.

    Every check precedes construction. They are the architecture (``check_sdnn_architecture``),
    each slice's core (``check_sdnn_core``), the weights within signed 16 bits, the scaled
    biases within signed 32 bits, ``source_values`` of width ``n_input`` within signed 32 bits,
    and the binary fan-in bound that ``_connection`` of the LIF builder applies. That bound is
    not a limit of the sigma-delta soma, and the released PilotNet SDNN lies far inside it.
    With ``build=False`` every check runs and ``None`` is returned in place of the network,
    so that a manifest for many steps needs no network that holds every value.
    """
    if model not in MODELS:
        raise ValueError(f"model {model!r} is not one of {MODELS}")
    entries = _sdnn_entries(network)
    values = _source_array(source_values, entries[0]["size"])
    somas = [_soma(entry, model == "corrected") for entry in entries]
    biases = [None] + [_layer_bias(entry) for entry in entries[1:]]
    connections = []
    for k in range(1, len(entries)):
        src, dst, weight, record = _connection(entries[k - 1], entries[k])
        record.update(src=k - 1, dst=k)
        connections.append((src, dst, weight, record))
    placement, neuron_orders = _placement(entries, layer_cores, order, partition)
    fingerprint = check_sdnn_architecture(arch)
    cores = []
    for k, slices in enumerate(placement):
        cores.append([arch.tiles[s["tile"]].cores[s["core"]] for s in slices])
        for j, (s, core) in enumerate(zip(slices, cores[-1])):
            check_sdnn_core(core, s["stop"] - s["start"],
                            what=f"layer {k} slice {j} on core ({s['tile']}, {s['core']})")
    positions = [neuron_positions(entry["shape"], entry["kind"], neuron_order)
                 for entry, neuron_order in zip(entries, neuron_orders)]
    slice_of = [_slice_index(slices, p) for slices, p in zip(placement, positions)]
    for src, dst, _, record in connections:
        k = record["dst"]
        synapses, axons, routes = _per_core(src, dst, slice_of[k - 1], slice_of[k],
                                            len(placement[k - 1]), len(placement[k]),
                                            entries[k - 1]["size"])
        record.update(synapses_per_dst_core=synapses, axons_per_dst_core=axons,
                      routes_per_src_core=routes)

    sigma_layers = [k for k, entry in enumerate(entries) if entry["layer"].neuron["sigma_output"]]
    kinds = [entry["kind"] for entry in entries]
    lava_models = ["lava.proc.io.encoder.PyDeltaEncoderModelDense",
                   f"lava.proc.sdn.models.{_LAVA_MODELS[model]}"]
    if sigma_layers:
        lava_models.append("lava.proc.sdn.models.PySigmaModelFixed")
    if "conv" in kinds:
        lava_models.append("lava.proc.conv.models.PyConvModelFixed")
    if "dense" in kinds:
        lava_models.append("lava.proc.dense.models.PyDenseModelBitAcc")
    stored = np.ascontiguousarray(values, dtype="<i4")
    manifest = {
        "schema_version": 1,
        "execution_profile": SDNN_EXECUTION_PROFILE,
        "architecture_profile": sdnn_architecture_profile(asdict(Allocation())),
        "architecture_sha256": fingerprint,
        "hardware_validated": False,
        "input_transfer_cost_included": False,
        "model": model,
        "lava_models": lava_models,
        "sources": {"group": _group_name(0), "steps": int(values.shape[0]),
                    # sha256 of the values as little-endian int32, C order [steps, n_input].
                    "values_sha256": hashlib.sha256(stored.tobytes()).hexdigest()},
        "sigma_output_layer": sigma_layers[0] if sigma_layers else None,
        "cost_assumptions": list(COST_ASSUMPTIONS),
        "connection_delay_updates": 1,
        "zero_weight_synapses": "not emitted",
        "group_names": [_group_name(k) for k in range(len(entries))],
        "layer_kinds": kinds,
        "layer_shapes": [list(entry["shape"]) for entry in entries],
        "layer_sizes": [entry["size"] for entry in entries],
        "order": order,
        "layer_cores": [len(slices) for slices in placement],
        "reserved_cores": [list(p) for p in RESERVED_CORES],
        "placement": [{"layer": k, "slices": slices} for k, slices in enumerate(placement)],
        "connections": [record for _, _, _, record in connections],
    }
    if partition is not None:
        manifest["partition"] = {
            "name": partition.get("name"),
            "layers": [{"order": o, "sizes": [s["stop"] - s["start"] for s in slices]}
                       for o, slices in zip(neuron_orders, placement)]}
        for layer, neuron_order in zip(manifest["placement"], neuron_orders):
            layer["order"] = neuron_order
    if not build:
        return None, manifest

    network_out = sanafe.Network()
    groups = []
    for k, (entry, (unit, attributes)) in enumerate(zip(entries, somas)):
        if biases[k] is not None and unit == SIGMA_DELTA_SOMA_UNIT:
            attributes = {**attributes, "bias": 0}
        group = network_out.create_neuron_group(_group_name(k), entry["size"],
                                                model_attributes=attributes,
                                                soma_hw_name=unit, log_spikes=False,
                                                log_potential=False)
        if k == 0:
            # A source sends zero after its last entry, so trailing zeros are left out.
            for j in range(entry["size"]):
                column = values[:, j]
                nonzero = np.flatnonzero(column)
                stop = int(nonzero[-1]) + 1 if nonzero.size else 0
                group[j].set_attributes(model_attributes={"graded_values":
                                                          column[:stop].tolist()})
        elif unit == SIGMA_DELTA_SOMA_UNIT:
            for offset in np.flatnonzero(biases[k]).tolist():
                group[offset].set_attributes(model_attributes={"bias": int(biases[k][offset])})
        at_position = np.argsort(positions[k], kind="stable").tolist()
        for s, core in zip(placement[k], cores[k]):
            for position in range(s["start"], s["stop"]):
                group[at_position[position]].map_to_core(core)
        groups.append(group)
    for src, dst, weight, record in connections:
        pre, post = groups[record["src"]], groups[record["dst"]]
        for start in range(0, src.size, _CONNECT_CHUNK):
            stop = start + _CONNECT_CHUNK
            pre.connect_neurons_sparse(post, {"weight": weight[start:stop].tolist()},
                                       list(zip(src[start:stop].tolist(),
                                                dst[start:stop].tolist())))
    return network_out, manifest


# ---------------------------------------------------------------------------
# The simulator


def _layer_states(manifest):
    """The states each layer reports, layer 0 the sources."""
    sigma = manifest["sigma_output_layer"]
    return [("sent",)] + [_SIGMA_STATES if k == sigma else SDNN_STATES
                          for k in range(1, len(manifest["layer_sizes"]))]


def _split_trace(rows, key, layers, sizes, names):
    """One trace key of one call, split into ``{layer: [steps, n] int64}``.

    The columns of a key are the neurons that report it, by group name, then by offset.
    """
    data = np.asarray(rows, dtype=np.float64)
    order = sorted(layers, key=lambda k: names[k])
    width = sum(sizes[k] for k in order)
    if data.ndim != 2 or data.shape[1] != width:
        raise ValueError(f"neuron trace {key} of shape {data.shape}, expected {width} columns")
    ints = data.astype(np.int64)
    if np.any(ints != data) or np.any(ints < _INT32[0]) or np.any(ints > _INT32[1]):
        raise ValueError(f"neuron trace {key} holds a value that is not a signed 32-bit integer")
    columns = np.split(ints, np.cumsum([sizes[k] for k in order])[:-1], axis=1)
    return dict(zip(order, columns))


def _sources_read(net, manifest, steps, keep_values):
    """The values the sources send over ``steps`` updates, read from the network.

    Returns ``(values or None, largest magnitude)``. A source sends zero after its last
    entry.
    """
    n = int(manifest["layer_sizes"][0])
    group = net[manifest["sources"]["group"]]
    values = np.zeros((steps, n), dtype=np.int32) if keep_values else None
    largest = 0
    for j in range(n):
        column = np.asarray(group[j].model_attributes.get("graded_values", [])[:steps],
                            dtype=np.int64)
        if column.size:
            largest = max(largest, int(np.abs(column).max()))
            if keep_values:
                values[:column.size, j] = column
    return values, largest


def sdnn_sanafe_run(net, manifest, arch, steps, *, keep="all", chunk=8,
                    timing_model="detailed"):
    """Run a network built by ``sdnn_to_sanafe`` for ``steps`` updates.

    ``arch`` is the architecture the chip is built on. It is not checked, so a run may use an
    architecture that differs from the one the network was built for in its costs only, and
    the caller records its fingerprint. The chip runs in calls of at most ``chunk`` steps with
    the perf trace and, under ``keep="all"``, the neuron trace, and each call is converted to
    int32 before the next.

    Returns a dict. ``layers`` holds, per layer, the states it has among ``SDNN_STATES`` as
    ``[steps, n]`` int32 arrays under ``keep="all"``, layer 0 holding ``sent``, the values of
    the sources. It is ``None`` under ``keep="extremes"``. ``spikes`` holds per layer
    ``sent != 0`` as bool under ``keep="all"``, else ``None``. ``extremes`` holds per layer the
    largest magnitude of each state over the run, read from the ``max_abs_*`` traces of the
    last step, which is the only step traced under ``keep="extremes"``, and for layer 0 from
    the values of the sources. ``trace`` holds ``perf_trace`` (a dict of per-step lists over
    the run), ``sim_time`` and ``energy``, each summed over the steps of the perf trace in
    step order, which is the simulator's own order of summation in one call. ``wall_seconds``
    holds the wall time of each call.
    """
    steps = _scalar_integer(steps, "steps", 1, _INT32_MAX)
    chunk = _scalar_integer(chunk, "chunk", 1, _INT32_MAX)
    if keep not in ("all", "extremes"):
        raise ValueError(f"keep {keep!r} is not 'all' or 'extremes'")
    names, sizes = manifest["group_names"], [int(n) for n in manifest["layer_sizes"]]
    states = _layer_states(manifest)
    reporting = {key: [k for k in range(1, len(sizes)) if key in states[k]]
                 for key in SDNN_STATES}
    chip = sanafe.SpikingChip(arch)
    chip.load(net)
    # Under keep="extremes" only the last step is traced, by a call of its own.
    calls = []
    untraced = steps - 1 if keep == "extremes" else steps
    while sum(calls) < untraced:
        calls.append(min(chunk, untraced - sum(calls)))
    if keep == "extremes":
        calls.append(1)
    held = [{key: [] for key in states[k]} for k in range(len(sizes))]
    perf, wall, last_traces = {}, [], None
    for i, n in enumerate(calls):
        traced = keep == "all" or i == len(calls) - 1
        start = time.perf_counter()
        result = chip.sim(n, timing_model=timing_model, perf_trace=True,
                          neuron_trace=True if traced else None)
        wall.append(time.perf_counter() - start)
        for key, column in result["perf_trace"].items():
            perf.setdefault(key, []).extend(column)
        if not traced:
            continue
        traces = result["neuron_trace"] or {}
        if keep == "all":
            for key in SDNN_STATES:
                if reporting[key]:
                    for k, part in _split_trace(traces[key], key, reporting[key], sizes,
                                                names).items():
                        held[k][key].append(part.astype(np.int32))
        last_traces = traces
    extremes = [dict() for _ in sizes]
    for key in SDNN_STATES:
        if reporting[key]:
            parts = _split_trace(last_traces[f"max_abs_{key}"][-1:], f"max_abs_{key}",
                                 reporting[key], sizes, names)
            for k, part in parts.items():
                extremes[k][key] = int(part.max())
    source_values, extremes[0]["sent"] = _sources_read(net, manifest, steps, keep == "all")
    trace = {"perf_trace": perf, "sim_time": 0.0,
             "energy": {"total": 0.0, "synapse": 0.0, "dendrite": 0.0, "soma": 0.0,
                        "network": 0.0}}
    for value in perf.get("sim_time", []):
        trace["sim_time"] += value
    for name in trace["energy"]:
        for value in perf.get(f"{name}_energy", []):
            trace["energy"][name] += value
    if keep != "all":
        return {"layers": None, "spikes": None, "extremes": extremes, "trace": trace,
                "wall_seconds": wall}
    layers = [{"sent": source_values}]
    for k in range(1, len(sizes)):
        layers.append({key: np.concatenate(held[k][key], axis=0) for key in states[k]})
    return {"layers": layers, "spikes": [layer["sent"] != 0 for layer in layers],
            "extremes": extremes, "trace": trace, "wall_seconds": wall}


# ---------------------------------------------------------------------------
# Lava


def _import_netx():
    try:
        from lava.lib.dl import netx
    except ImportError as exc:
        raise ImportError("lava-dl's netx is not importable. Put data/calibration/lava-dl/src "
                          "on sys.path (fetched by SANA-FE-thesis/calibration/deployments/"
                          "pilotnet_lif/fetch.py)") from exc
    return netx


def _lava_network(netx, path, entries):
    """lava-dl's netx network of the file, checked against the reader's layers."""
    net = netx.hdf5.Network(net_config=str(path), skip_layers=1)
    if len(net.layers) != len(entries) - 1:
        raise ValueError(f"netx built {len(net.layers)} layers, the network has "
                         f"{len(entries) - 1} after the input")
    if tuple(net.inp.shape) != _lava_shape(entries[0]):
        raise ValueError(f"netx's input has shape {tuple(net.inp.shape)}, the input layer "
                         f"{_lava_shape(entries[0])}")
    for k, (block, entry) in enumerate(zip(net.layers, entries[1:]), start=1):
        expected = "Sigma" if entry["layer"].neuron["sigma_output"] else "SigmaDelta"
        if tuple(block.shape) != _lava_shape(entry) or type(block.neuron).__name__ != expected:
            raise ValueError(f"layer {k}: netx built {type(block.neuron).__name__} of shape "
                             f"{tuple(block.shape)}, the network has {expected} of shape "
                             f"{_lava_shape(entry)}")
    return net


class _Extremes:
    """Running largest magnitudes per layer and state, and the input read from sigma."""

    def __init__(self, entries):
        self.values = [dict() for _ in entries]
        self.sigma = [np.zeros(entry["size"], dtype=np.int64) for entry in entries]

    def add(self, k, chunk):
        if "sigma" in chunk:
            sigma = chunk["sigma"]
            previous = np.vstack([self.sigma[k][None, :], sigma[:-1]])
            chunk["input"] = sigma - previous
            self.sigma[k] = sigma[-1].copy()
        for state, array in chunk.items():
            largest = int(np.abs(array).max()) if array.size else 0
            self.values[k][state] = max(self.values[k].get(state, 0), largest)


def sdnn_lava_run(path, frames, steps, *, model="fixed", keep="all", chunk=16,
                  stall_seconds=300):
    """Run the netx file at ``path`` in Lava's fixed-point simulation and read every step.

    lava-dl's netx builds the network with ``skip_layers=1``, and ``frames`` (integers
    ``[steps, c, h, w]``, row ``t`` the frame the encoder receives at its step ``t + 1``)
    feed it from a ``RingBuffer`` through ``DeltaEncoder(shape, vth=vThMant,
    spike_exp=spike_exp)`` of the input layer with ``PyDeltaEncoderModelDense``. Every
    ``SigmaDelta`` runs ``PySigmaDeltaModelFixed``, or ``PySigmaDeltaModelFixedCorrected``
    under ``model="corrected"``. lava-dl's ``src`` folder must be importable. The run goes in
    chunks of at most ``chunk`` steps (at most 16), with the stall rule and the shutdown of
    ``sanafe.lif_network.lava_run``.

    Returns a dict. ``layers`` has the layout of ``sdnn_sanafe_run``, in reader order, layer 0
    holding ``sent``, the encoder's values, row ``t`` the value sent at step ``t + 1``. Under
    ``keep="all"`` every state is int32, the input being the change of ``sigma`` from the
    previous step. Under ``keep="nonzero"`` each layer holds only ``sent != 0`` as bool.
    ``extremes`` has the layout of ``sdnn_sanafe_run``. ``models`` names the process model
    class each layer ran, the encoder's for layer 0.
    """
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.run_configs import Loihi2SimCfg
    from lava.proc.io.encoder import DeltaEncoder, PyDeltaEncoderModelDense
    from lava.proc.io.extractor import (Extractor, PyLoihiExtractorModel, PyLoihiVarWireModel,
                                        VarWire)
    from lava.proc.io.source import RingBuffer
    from lava.proc.sdn import models as sdn_models
    from lava.proc.sdn.process import SigmaDelta

    if model not in MODELS:
        raise ValueError(f"model {model!r} is not one of {MODELS}")
    if keep not in ("all", "nonzero"):
        raise ValueError(f"keep {keep!r} is not 'all' or 'nonzero'")
    steps = _scalar_integer(steps, "steps", 1, _INT32_MAX)
    chunk = _scalar_integer(chunk, "chunk", 1, 16)
    entries = _sdnn_entries(read_sdnn(path))
    data = np.asarray(frames)
    if data.shape != (steps, *entries[0]["shape"]):
        raise ValueError(f"frames has shape {data.shape}, expected (steps, channels, height, "
                         f"width) = {(steps, *entries[0]['shape'])}")
    data = _integers(data, "frames", *_INT32)
    netx = _import_netx()
    first = entries[0]["layer"].neuron
    net = _lava_network(netx, path, entries)
    source = RingBuffer(data=data.transpose(3, 2, 1, 0).astype(np.int32))
    encoder = DeltaEncoder(shape=tuple(net.inp.shape), vth=first["vth_mant"],
                           spike_exp=first["spike_exp"])
    source.s_out.connect(encoder.a_in)
    encoder.s_out.connect(net.inp)

    neurons = [encoder] + [block.neuron for block in net.layers]
    wires, extractors = [], []
    for k, process in enumerate(neurons):
        extractor = Extractor(shape=tuple(process.s_out.shape), buffer_size=chunk + 1)
        process.s_out.connect(extractor.in_port)
        extractors.append((k, extractor))
        for state in ("sigma", "act", "residue"):
            if k > 0 and hasattr(process, state):
                wire = VarWire(buffer_size=chunk + 1)
                wire.connect_var(getattr(process, state))
                wires.append((k, state, wire))

    chosen = getattr(sdn_models, _LAVA_MODELS[model])
    fixed = {SigmaDelta: chosen, DeltaEncoder: PyDeltaEncoderModelDense,
             Extractor: PyLoihiExtractorModel, VarWire: PyLoihiVarWireModel}
    index = {id(process): k for k, process in enumerate(neurons)}
    models = [None] * len(neurons)

    class _Config(Loihi2SimCfg):
        """``Loihi2SimCfg(select_tag='fixed_pt')`` with the selections above, recording the
        model of every layer's process."""

        def select(self, proc, proc_models):
            selected = fixed.get(type(proc)) or super().select(proc, proc_models)
            if id(proc) in index:
                models[index[id(proc)]] = selected.__name__
            return selected

    config = _Config(select_tag="fixed_pt")
    extremes = _Extremes(entries)
    held = [dict() for _ in entries]
    done, in_flight = 0, False
    try:
        while done < steps:
            n = min(chunk, steps - done)
            in_flight = True
            net.run(condition=RunSteps(num_steps=n, blocking=False), run_cfg=config)
            rows = _receive([w for _, _, w in wires], [e for _, e in extractors], n,
                            stall_seconds)
            net.wait()
            in_flight = False
            done += n
            parts = [dict() for _ in entries]
            for i, (k, state, _) in enumerate(wires):
                parts[k][state] = _flatten_rows([row[0][i] for row in rows], entries[k])
            for i, (k, _) in enumerate(extractors):
                parts[k]["sent"] = _flatten_rows([row[1][i] for row in rows], entries[k])
            for k, part in enumerate(parts):
                extremes.add(k, part)
                kept = part if keep == "all" else {"sent": part["sent"] != 0}
                for state, array in kept.items():
                    held[k].setdefault(state, []).append(
                        array if array.dtype == bool else array.astype(np.int32))
    finally:
        if in_flight:
            # As in lif_network.lava_run. A chunk failed between run and wait, so stop(),
            # which has no time limit, is not attempted, and every multiprocessing child of
            # this process is terminated instead. The exception propagates.
            _terminate_lava_children()
        else:
            net.stop()
    if any(m is None for m in models):
        raise RuntimeError("the run configuration did not select a model for every layer")
    order = ("sent",) if keep == "nonzero" else (*SDNN_STATES,)
    layers = [{state: np.concatenate(layer[state], axis=0) for state in order if state in layer}
              for layer in held]
    return {"layers": layers, "extremes": extremes.values, "models": models}


# ---------------------------------------------------------------------------
# Comparison and ranges


def compare_sdnn(a, b):
    """Compare the layers of two runs state by state.

    Each layer compares the states both runs hold, in the order of ``SDNN_STATES``. Returns
    ``layers`` (per layer ``compared``, the list of states compared, ``mismatches``, the
    number of differing entries per state, and ``first``, per state the first difference in
    step-major order as ``[step, neuron, a, b]`` or ``None``), ``total_mismatches`` and
    ``neuron_steps``, the steps times the neurons of every layer with a state compared.
    """
    if a["layers"] is None or b["layers"] is None:
        raise ValueError("both runs must hold their layers (keep='all' or keep='nonzero')")
    if len(a["layers"]) != len(b["layers"]):
        raise ValueError(f"{len(a['layers'])} layers against {len(b['layers'])}")
    out, total, neuron_steps = [], 0, 0
    for k, (x, y) in enumerate(zip(a["layers"], b["layers"])):
        compared = [state for state in SDNN_STATES if state in x and state in y]
        record = {"compared": compared, "mismatches": {}, "first": {}}
        for state in compared:
            u, v = np.asarray(x[state]), np.asarray(y[state])
            if u.shape != v.shape or (u.dtype == bool) != (v.dtype == bool):
                raise ValueError(f"layer {k} {state}: {u.shape} {u.dtype} against "
                                 f"{v.shape} {v.dtype}")
            differ = u != v
            record["mismatches"][state] = int(differ.sum())
            where = np.argwhere(differ)
            record["first"][state] = None
            if where.size:
                t, n = (int(i) for i in where[0])
                record["first"][state] = [t, n, int(u[t, n]), int(v[t, n])]
            total += record["mismatches"][state]
        if compared:
            neuron_steps += int(np.asarray(x[compared[0]]).size)
        out.append(record)
    return {"layers": out, "total_mismatches": total, "neuron_steps": neuron_steps}


def fits(extremes):
    """Per layer, whether the extremes fit the widths of Loihi 2's documented formats.

    ``states_fit_24`` holds when ``sigma``, ``act`` and ``residue`` are at most 2^23 - 1 in
    magnitude, ``sent_fits_16`` when ``sent`` is at most 2^15 - 1 and ``input_fits_24`` when
    the input is at most 2^23 - 1. A magnitude of exactly 2^(b - 1) is counted as not fitting,
    although the negative value -2^(b - 1) fits b signed bits, a conservative reading forced by
    the extremes holding magnitudes only. An entry is ``None`` when the layer has none of its
    states.
    """
    def check(layer, states, bits):
        found = [layer[s] for s in states if s in layer]
        return None if not found else all(v <= (1 << (bits - 1)) - 1 for v in found)

    return [{"states_fit_24": check(layer, ("sigma", "act", "residue"), 24),
             "sent_fits_16": check(layer, ("sent",), 16),
             "input_fits_24": check(layer, ("input",), 24)} for layer in extremes]
