# Loihi 2 candidate profile

`sanafe.loihi2.load_loihi2_candidate()` returns an architectural candidate with 128 cores in an 8 × 4 mesh of four-core tiles. It selects `integrate_fire_int24`, `accumulator_int`, and `current_based_int` with an external pre-soma buffer. All timing, synchronization, and energy coefficients remain inherited from the bundled Loihi 1 configuration. They are not Loihi 2 measurements or calibrated predictions.

`candidate_profile()` returns a serializable specification with parameter provenance, profile version 1, and numerical profile identifier `qcfs-if-int24-binary-v1`. The restricted numerical contract uses signed 16-bit effective weight transport, input, and bias, signed 32-bit accumulation with overflow rejection, and signed 24-bit voltage clipping before a `>=` threshold comparison. Effective weight transport is not a statement of physical Loihi 2 weight precision. Each update emits at most one binary event and subtracts the threshold once. Connections have one logical update of delay. These are candidate software semantics, not a verified Loihi 2 neuron program or a complete implementation of Loihi 2 capabilities.

## Resource validation

```python
from sanafe.loihi2 import Allocation, validate_chain_resources

report = validate_chain_resources(
    [2, 1], [[[1, -1]]], [(0, 0), (1, 0)],
    reserved_cores=[(31, 3)], allocation=Allocation(),
)
```

The function rejects malformed inputs, noninteger weights, reserved-core placement, and capacity violations with `ValueError`. Weight matrices use destination rows and source columns. Integer-valued floats and Boolean values are rejected. Every dense weight occupies storage, including zero weights. Whole layers are mapped to specified cores and colocated layers share the same capacity budget.

The public resource ceilings are 8,192 neurons, 192 KiB total memory, and 128 KiB synaptic memory per core [1]. No undocumented incoming-axon or outgoing-route cap is imposed. Each source neuron has one incoming axon on its destination core and one outgoing route per destination core. Self-routes are conservatively counted. Source identities remain distinct across colocated layers.

`Allocation` specifies assumed byte costs. Defaults are 16 bytes per neuron, four per synapse, eight per incoming axon, eight per outgoing route, and 256 program bytes per occupied core. These are explicit modeling assumptions. They do not reproduce verified Intel packing, alignment, compression, or allocation rules. Program storage is charged once per occupied core. Alternative models can pass another validated `Allocation`. Program bytes may be zero for an explicitly assumed lower-bound scenario. Other allocation sizes must be positive integers.

A successful report means **model-feasible under the assumed layout**. It does not guarantee physical fit. The report lists occupied cores sorted by tile and core, their layer indices, neuron/synapse/axon/route counts, each storage subtotal, and total bytes. It records the allocation assumptions and `physical_fit_verified=False`.

## Evidence and limits

[1] Intel, “Taking Neuromorphic Computing to the Next Level with Loihi 2,” technology brief. [Public document](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/neuromorphic-computing-loihi-2-brief.pdf). Source for resource ceilings. Maximum neuron counts remain conditional on state and storage requirements.

[2] J. Timcheck, A. Pierro, and S. B. Shrestha, “A Compute and Communication Runtime Model for Loihi 2,” arXiv:2601.10035v2, 2026. [Paper](https://arxiv.org/html/2601.10035v2). Section II-1 and Figure 1 document four-core routers and dimension-order routing. The 8 × 4 coordinate shape is an inherited assumption from the bundled configuration, distinct from the documented core count and cores per router. The candidate does not implement or claim validation against this paper's runtime model.

[3] SLAM Lab, “SANA-FE,” bundled Loihi configuration. [Pinned source](https://github.com/SLAM-Lab/SANA-FE/blob/93926ec8019206c1c6e6709448ac4c67f46d57db/sanafe/examples/loihi.yaml). Source for all inherited cost coefficients, attributed there to Davies et al. (2018).

Graded events, instruction execution, packed synaptic-memory traffic, detailed multicast, physical input/output transfer, and Loihi 2 cost calibration are outside this profile. Using the candidate with SANA-FE's scheduler does not execute Intel compiler output or Intel hardware.

## Integer Lava export

The optional Lava integration requires the `QCFSIFFixed` Process extension. Export a fresh, directly connected one-dimensional IF/Dense chain through `sanafe.lava.qcfs_fixed_chain_to_sanafe`. It returns the mapped `Network` and a required manifest. Attach input and observer Processes only after exporting.

The adapter supports binary Dense input, zero weight exponent, and 1–8 configured weight bits. It applies the selected Lava Dense clipping and truncation rules before storing effective integer weights. Positive and negative row sums must independently fit signed16, guaranteeing every binary fan-in combination fits the IF interface. Nonzero initial Dense buffers, branching, recurrent graphs, virtual ports, already compiled Processes, fractional values, and unsupported models are rejected. Explicit neuron initial voltages are preserved.

The exporter validates shared resource budgets before network construction. These checks currently cover the supported chain adapter. Direct use of the general SANA-FE Network API does not automatically enforce this Python resource profile.

`Architecture.configuration()` provides a complete configuration snapshot for provenance. The candidate adapter verifies its fingerprint, including topology, buffering, pipeline parameters, synchronization, and costs, against the loader's configuration. Modified architectures are rejected rather than receiving incorrect provenance. The manifest records the fingerprint, numerical model classes, effective weights, state, currents, placement, and actual allocation assumptions. Save the configuration snapshot alongside experiment results.

The integer pipeline carries bounded integers through SANA-FE's existing double-valued pipeline interface. All transported signed32 values are exactly representable. Neuron integration uses widened integer arithmetic, with explicit clipping at the voltage boundary. This does not add graded emitted messages.

Range errors in core processing are captured inside worker loops and rethrown after synchronization. After an execution error, discard or reset the simulation before reuse. The implementation has been checked on a macOS build without OpenMP; an OpenMP-enabled Linux runtime remains a separate portability check.

## Trace-derived workload and conditional runtime bound

The reusable Loihi 2 candidate modules live beside each other in `sanafe/`. `loihi2.py` defines the architectural and resource candidate. `lava_integer.py` exports the supported numerical graph. `loihi2_runtime.py` analyzes a mapped simulation and applies an independent runtime equation. The C++ scheduler remains in `src/`; `sanafe/data.py` prepares generic trace tables and `sanafe/viz/` plots them. The thesis-specific sweep and evidence files belong outside this public package.

Run the integer chain with `spike_trace=True`, `message_trace=True`, and `perf_trace=True`, then call:

```python
from sanafe.loihi2_runtime import analyze_binary_chain, RuntimeRates, max_affine_runtime

workload = analyze_binary_chain(manifest, result, packet_bits=64)
# The 64-bit packet size is an explicit hypothesis, not a Loihi 2 specification.
# Supply source-qualified coefficients before requesting a numerical bound.
rates = RuntimeRates(
    dendop_seconds=..., synop_seconds=..., synmem_read_seconds=...,
    link_bits_per_second=..., barrier_seconds=...,
    dense_entries_per_read=...,
)
estimate = max_affine_runtime(workload, rates)
```

The workload report gives each numbered step's operations on every occupied core and bits on every directed XY mesh link. It also reports the busiest core and link. DendOps count each IF state update. SynOps count nonzero effective weights reached by an emitted binary message. Dense SynMem entries include stored zero weights. The `dense_entries_per_read` assumption converts each incoming axon's dense entries to an assumed number of reads, with rounding per axon. The packet-width assumption turns message counts into link bits. Neither conversion is a verified Intel packing or packet format.

`max_affine_runtime` evaluates the five-term maximum in Timcheck *et al.* [2], Equation (1), on each step. It adds the resulting step bounds for a whole run. The model is separate from SANA-FE's inherited Loihi 1 `sim_time`. Its coefficients have no defaults; the module rejects missing, nonfinite, or nonpositive rates. Published Loihi 2 microbenchmarks characterize simple programs on specified hardware and software revisions [2]. Applying their rates to this custom IF program requires validation. The current output is a conditional lower bound, not calibrated Loihi 2 latency. It excludes host input and output transfer, physical instruction costs outside the five terms, and energy.

The trace analyzer supports the restricted unbranched binary dense chain and its one-step buffered update convention. It checks spike/message correspondence, mapped source and destination cores, dense fanout, and simulation update counts. It does not model general multicast, graded spikes, or the full Loihi 2 packet protocol. Two parallel physical meshes are represented by an assumed effective link bandwidth, not assigned individual traffic by this analyzer. The existing BookSim configuration is unchanged.
