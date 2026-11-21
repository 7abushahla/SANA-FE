"""Lava's fixed-point sigma-delta and sigma neurons, the ``sigma_delta_int32`` and ``sigma_int32``
somas, and the derived architecture that offers them.

The rules
---------
Both rules are transcribed from ``lava/proc/sdn/models.py`` of Lava at ``74840cb8``. The soma
``sigma_delta_int32`` runs ``PySigmaDeltaModelFixed.run_spk`` (lines 279-284) with
``dynamics`` (150-156), ``sigma_dynamics`` (52), the ReLU of ``activation_dynamics``
(140-143, with ``ReLU`` at 17-31) and ``delta_dynamics`` without cumulative error (85 and
95-100)::

    sigma    = sigma + a_in
    act_new  = max(sigma + bias, 0)
    delta    = act_new - act + residue
    s_scaled = delta if abs(delta) >= vth else 0
    residue  = delta - s_scaled
    act      = act_new
    sent     = floor(s_scaled / 2**state_exp)

With ``corrected`` set it runs ``PySigmaDeltaModelFixedCorrected.run_spk`` (306-316), which
leaves the residue untouched and carries the remainder of the shift in ``act``::

    sigma    = sigma + a_in
    act_new  = max(sigma + bias, 0)
    delta    = act_new - act
    s_scaled = delta if abs(delta) >= vth else 0
    sent     = floor(s_scaled / 2**state_exp)
    act      = act + sent * 2**state_exp

The soma ``sigma_int32`` runs ``PySigmaModelFixed`` (172-176, with ``run_spk`` at 54-57),
``sigma = sigma + a_in`` and ``sent = sigma``.

Lava's process scales ``vth`` and ``bias`` by ``2**(spike_exp + state_exp)``
(``lava/proc/sdn/process.py`` 179-180). The neuron holds them as given, already scaled.

Which model Lava runs
---------------------
Both fixed-point sigma-delta models are tagged ``fixed_pt``. With ``select_tag='fixed_pt'``
Lava's simulation takes the first model carrying the tag (``run_configs.py`` 314-326), in the
order in which the classes are defined in the module (``compiler_graphs.py`` 778-791). That is
``PySigmaDeltaModelFixed``, which the soma runs when ``corrected`` is not set.

Range
-----
Lava holds every state in ``np.int32``, which wraps silently. The somas compute every quantity
in 64 bits and stop with ``OverflowError`` ("<group>[<offset>]: sigma-delta state <name> <value>
leaves signed 32 bits") where Lava would wrap. Over the range in which Lava does not wrap the
two agree exactly. A sigma-delta neuron without ``vth`` stops at its first update with a
``ValueError`` that names it the same way.

Delivery
--------
A nonzero sent value leaves the soma as a graded spike, which multiplies the weight of every
synapse it reaches. A zero value sends nothing. The somas assume the time-step buffer before
the soma unit, where the input of a step is the sum of its synaptic products and a value sent
at one update reaches its destinations at the next. Where the buffer is inside the soma unit or
before the axon-out unit the soma runs on the message path, and the first step of a run refuses
each neuron of either soma with a ``ValueError`` that names it.
"""
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
import tempfile

import numpy as np
import sanafe
from sanafe.lava_integer import _integers, _scalar_integer

#: The soma unit of the derived architecture that runs ``sigma_delta_int32``.
SIGMA_DELTA_SOMA_UNIT = "loihi_sigma_delta"
#: The soma unit of the derived architecture that runs ``sigma_int32``.
SIGMA_SOMA_UNIT = "loihi_sigma"
#: Execution profile of the networks these units run, recorded in their manifests.
SDNN_EXECUTION_PROFILE = "lava-sigma-delta-int32-v1"
#: The states a sigma-delta neuron reports after each update.
SDNN_STATES = ("sigma", "act", "residue", "sent", "input")
#: What the cost of a sigma-delta network assumes.
COST_ASSUMPTIONS = ("A graded message is charged as a binary message.",
                    "A sigma-delta update is charged as an update of the existing somas.")

_INT32 = (-(1 << 31), (1 << 31) - 1)
_MODELS = {SIGMA_DELTA_SOMA_UNIT: "sigma_delta_int32", SIGMA_SOMA_UNIT: "sigma_int32"}
_TRACES = (*SDNN_STATES, *(f"max_abs_{k}" for k in SDNN_STATES))


def _inputs(a_in):
    data = _integers(a_in, "a_in", *_INT32)
    if data.ndim != 2 or data.shape[0] < 1 or data.shape[1] < 1:
        raise ValueError("a_in must be a nonempty [steps, neurons] array")
    return data


def _parameters(n, vth, bias, state_exp, corrected):
    if not isinstance(corrected, (bool, np.bool_)):
        raise ValueError("corrected must be a bool")
    return {"vth": _scalar_integer(vth, "vth", 0, _INT32[1]),
            "bias": _integers(bias, "bias", *_INT32, shape=(n,)),
            "state_exp": _scalar_integer(state_exp, "state_exp", 0, 30),
            "corrected": bool(corrected)}


def _stored(name, value):
    """``value``, or ``OverflowError`` where Lava's int32 arithmetic would wrap."""
    outside = (value < _INT32[0]) | (value > _INT32[1])
    if np.any(outside):
        raise OverflowError(f"sigma-delta state {name} {int(value[outside][0])} "
                            "leaves signed 32 bits")
    return value


def lava_sigma_delta_trace(a_in, *, vth, bias, state_exp, corrected=False):
    """The rule of ``PySigmaDeltaModelFixed``, or of ``PySigmaDeltaModelFixedCorrected`` when
    ``corrected`` is set, in 64-bit integers.

    ``a_in`` is ``[steps, n]``, row ``t`` the input of update ``t``. ``bias`` is one value or
    one per neuron. Returns int64 arrays ``[steps, n]`` under ``sigma``, ``act``, ``residue``
    and ``sent``, row ``t`` the state after update ``t``. Raises ``OverflowError`` where Lava's
    int32 states would wrap.
    """
    a = _inputs(a_in)
    steps, n = a.shape
    p = _parameters(n, vth, bias, state_exp, corrected)
    sigma, act, residue = (np.zeros(n, dtype=np.int64) for _ in range(3))
    out = {k: np.zeros((steps, n), dtype=np.int64) for k in ("sigma", "act", "residue", "sent")}
    for t in range(steps):
        sigma = _stored("sigma", sigma + a[t])
        act_new = np.maximum(_stored("activation", sigma + p["bias"]), 0)
        if p["corrected"]:
            delta = _stored("delta", act_new - act)
            s_scaled = np.where(np.abs(delta) >= p["vth"], delta, 0)
            sent = np.right_shift(s_scaled, p["state_exp"])
            act = _stored("act", act + np.left_shift(sent, p["state_exp"]))
        else:
            delta = _stored("delta", act_new - act + residue)
            s_scaled = np.where(np.abs(delta) >= p["vth"], delta, 0)
            residue = delta - s_scaled
            act = act_new
            sent = np.right_shift(s_scaled, p["state_exp"])
        for name, value in (("sigma", sigma), ("act", act), ("residue", residue),
                            ("sent", sent)):
            out[name][t] = value
    return out


def lava_sigma_trace(a_in):
    """The rule of ``PySigmaModelFixed`` in 64-bit integers.

    Returns int64 arrays ``[steps, n]`` under ``sigma`` and ``sent``, row ``t`` the state after
    update ``t``. Raises ``OverflowError`` where Lava's int32 sigma would wrap.
    """
    a = _inputs(a_in)
    sigma = np.zeros(a.shape[1], dtype=np.int64)
    out = np.zeros(a.shape, dtype=np.int64)
    for t in range(a.shape[0]):
        sigma = _stored("sigma", sigma + a[t])
        out[t] = sigma
    return {"sigma": out, "sent": out.copy()}


def _sigma_delta_chip(a_in, *, vth, bias, state_exp, corrected=False):
    """A chip of the derived architecture with one ``sigma_delta_int32`` neuron per column of
    ``a_in``, which receives row ``t`` at its update ``t`` as its external input. The neurons
    fill the cores in order, tile by tile, each core up to its capacity."""
    a = _inputs(a_in)
    steps, n = a.shape
    p = _parameters(n, vth, bias, state_exp, corrected)
    arch = load_loihi2_sdnn_candidate()
    capacity = arch.tiles[0].cores[0].max_neurons_supported
    per_tile = len(arch.tiles[0].cores)
    net = sanafe.Network()
    group = net.create_neuron_group("layer_0", n)
    for offset, neuron in enumerate(group):
        slot = offset // capacity
        neuron.set_attributes(soma_hw_name=SIGMA_DELTA_SOMA_UNIT, model_attributes={
            "vth": p["vth"], "bias": int(p["bias"][offset]), "state_exp": p["state_exp"],
            "corrected": p["corrected"], "currents": a[:, offset].tolist()})
        neuron.map_to_core(arch.tiles[slot // per_tile].cores[slot % per_tile])
    chip = sanafe.SpikingChip(arch)
    chip.load(net)
    return chip


def sanafe_sigma_delta_trace(a_in, *, vth, bias, state_exp, corrected=False):
    """Run ``sigma_delta_int32`` neurons on the derived architecture.

    Same arguments as ``lava_sigma_delta_trace``. Row ``t`` of ``a_in`` is the external input
    of update ``t``, which enters sigma without delay. Returns int64 arrays ``[steps, n]``
    under every neuron trace of the soma, row ``t`` the value after update ``t``.
    """
    steps = _inputs(a_in).shape[0]
    chip = _sigma_delta_chip(a_in, vth=vth, bias=bias, state_exp=state_exp,
                             corrected=corrected)
    traces = chip.sim(steps, neuron_trace=True)["neuron_trace"]
    # Every trace value is within 2**31 in magnitude, so float64 holds it exactly.
    return {k: np.asarray(traces[k], dtype=np.float64).astype(np.int64) for k in _TRACES}


def sdnn_candidate_yaml():
    """The text of the candidate architecture with the two soma units added after its last soma
    unit. Each copies the costs of ``loihi_if_gated`` and names its own model."""
    text = (files("sanafe.examples") / "loihi2.yaml").read_text()
    start = text.index("            - name: loihi_if_gated\n")
    gated = text[start:text.index("            - name:", start + 1)]
    anchor = "          dendrite:\n"
    if text.count(anchor) != 1 or gated.count("model: integrate_fire_int24_gated") != 1:
        raise ValueError("The candidate architecture text has an unexpected layout")
    units = ""
    for unit, model in _MODELS.items():
        stanza = gated.replace("name: loihi_if_gated", f"name: {unit}")
        head, tail = stanza.split("model: integrate_fire_int24_gated", 1)
        units += f"{head}model: {model}  # Lava fixed-point rule, the costs of loihi_if_gated\n"
        units += tail.split("\n", 1)[1]
    return text.replace(anchor, units + anchor)


def load_loihi2_sdnn_candidate():
    """Load the derived architecture of ``sdnn_candidate_yaml``."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "loihi2_sdnn.yaml"
        path.write_text(sdnn_candidate_yaml())
        return sanafe.load_arch(str(path))


@lru_cache(maxsize=1)
def sdnn_fingerprint():
    """The architecture fingerprint of the derived architecture."""
    from sanafe.loihi2 import architecture_fingerprint

    return architecture_fingerprint(load_loihi2_sdnn_candidate())


def check_sdnn_architecture(arch):
    """Refuse an architecture other than the derived one. Returns its fingerprint."""
    from sanafe.loihi2 import architecture_fingerprint

    fingerprint = architecture_fingerprint(arch)
    if fingerprint != sdnn_fingerprint():
        raise ValueError("Unsupported architecture configuration for the sigma-delta profile")
    return fingerprint


def check_sdnn_core(core, required, what=None):
    """Refuse a core without the sigma-delta pipeline or with fewer than ``required`` slots.

    The core must hold one unit each of ``loihi_sigma_delta``, ``loihi_sigma`` and
    ``loihi_if_gated``, one integer synapse and one integer accumulator, and the time-step
    buffer before the soma unit. ``what`` names the placed neurons in the capacity message.
    """
    for unit, model in {**_MODELS, "loihi_if_gated": "integrate_fire_int24_gated"}.items():
        somas = [u for u in core.pipeline_hw if u.implements_soma and u.name == unit]
        if len(somas) != 1 or somas[0].model_info.name != model:
            raise ValueError(f"Expected one {unit} unit per selected core")
    for flag, model in [("implements_dendrite", "accumulator_int"),
                        ("implements_synapse", "current_based_int")]:
        units = [u for u in core.pipeline_hw if getattr(u, flag)]
        if len(units) != 1 or units[0].model_info.name != model:
            raise ValueError(f"Expected one {model} unit per selected core")
    if core.buffer_position != sanafe.BufferPosition.buffer_before_soma_unit:
        raise ValueError("Expected external pre-soma buffer")
    if required > core.max_neurons_supported:
        placed = f"{what}, " if what else ""
        raise ValueError(f"Placement exceeds configured neuron capacity ({placed}{required} "
                         f"neurons on a core of capacity {core.max_neurons_supported})")


def sdnn_architecture_profile(allocation):
    """The candidate profile with the numerical profile, units and semantics of the sigma-delta
    somas, and the assumptions of their costs."""
    from sanafe.loihi2 import candidate_profile

    profile = candidate_profile()
    profile["allocation"] = dict(allocation)
    profile["numerical_profile"] = SDNN_EXECUTION_PROFILE
    profile["numerical_profiles"] = [*profile["numerical_profiles"], SDNN_EXECUTION_PROFILE]
    profile["soma_units"] = {**profile["soma_units"], **_MODELS}
    profile["semantics"] = {
        "describes_unit": f"{SIGMA_DELTA_SOMA_UNIT} and {SIGMA_SOMA_UNIT}",
        "state_bits": 32, "state_signed": True,
        "state_overflow": "stop with OverflowError where Lava's int32 states wrap",
        "wrap": False, "activation": "ReLU", "cumulative_error": False,
        "threshold_comparison": ">=", "threshold_operand": "abs(delta)",
        "sent_value": "floor(s_scaled / 2**state_exp), the arithmetic shift right by state_exp",
        "default_rule": "PySigmaDeltaModelFixed",
        "corrected_rule": "PySigmaDeltaModelFixedCorrected",
        "events": "graded", "graded_value_bits": 32, "graded_value_signed": True,
        "zero_value": "sends nothing", "weight_signed": True,
        "effective_weight_transport_bits": 16, "accumulation_bits": 32,
        "accumulation_signed": True, "accumulation_overflow": "reject",
        "edge_delay_steps": 1}
    profile["cost_assumptions"] = list(COST_ASSUMPTIONS)
    return profile
