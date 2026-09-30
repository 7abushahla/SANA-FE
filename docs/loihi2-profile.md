# Loihi 2 candidate profile

`sanafe.loihi2.load_loihi2_candidate()` returns an architectural candidate with 128 cores in an 8 × 4 mesh of four-core tiles. It selects `integrate_fire_int24`, `accumulator_int`, and `current_based_int` with an external pre-soma buffer. Its cost coefficients start from the bundled, Nahuku-fitted Loihi 1 configuration. Four latencies and the barrier table are scaled by the factors Intel states for Loihi 2 over Loihi 1; every energy, hop and message coefficient is inherited unchanged (see [Cost derivation](#cost-derivation)). They are scaled estimates, not Loihi 2 measurements or calibrated predictions.

`candidate_profile()` returns a serializable specification with parameter provenance, profile version 1, and numerical profile identifier `qcfs-if-int24-binary-v1`. The restricted numerical contract uses signed 16-bit effective weight transport, input, and bias, signed 32-bit accumulation with overflow rejection, and signed 24-bit voltage clipping before a `>=` threshold comparison. Effective weight transport is not a statement of physical Loihi 2 weight precision. Documented native stored synapses have at most 8-bit weights [2]. `audit_unit_scale_signed8_weights()` flags effective software values outside a direct signed8 unit-scale scenario; it does not test exponent selection, parallel synapses, or physical packing. Each valid update emits at most one binary event and subtracts the threshold once. Connections have one logical update of delay. These are candidate software semantics, not a verified Loihi 2 neuron program or a complete implementation of Loihi 2 capabilities.

## Cost derivation

Decided 2026-09-30. The candidate's costs are the fitted Loihi 1 coefficients of `arch/loihi.yaml` (see that file's header for their Nahuku provenance), with the following changes and nothing else. `sanafe.loihi2.candidate_cost_provenance()` returns the same table programmatically and `test_loihi2_profile` checks the YAML against it.

| Coefficient | Loihi 1 fitted | Factor | Candidate | Basis |
| --- | --- | --- | --- | --- |
| `latency_access_neuron`, `latency_update_neuron` | 6.0 ns, 3.7 ns | ÷ 2 | 3.0 ns, 1.85 ns | "2x for simple neuron state updates" [1] |
| `latency_process_spike` | 3.8 ns | ÷ 5 | 0.76 ns | "5x for synaptic operations" [1] |
| `latency_spike_out` | 30 ns | ÷ 10 | 3.0 ns | "10x for spike generation" [1] |
| `latency_sync` table, 1 / 2 / 4 / 29 tiles | 0.6 / 1.0 / 1.4 / 1.8 µs | ÷ 9 | 66.7 / 111 / 156 / 200 ns | chip-wide entry set to the stated "minimum chip-wide time steps under 200ns" [1] |
| every `energy_*`, hop and message coefficient, `link_buffer_size` | as Loihi 1 | 1 | unchanged | no public Loihi 2 datum |

Intel's footnote to the factors states that they are "based on comparisons between barrier synchronization time, synaptic update time, neuron update time, and neuron spike times between Loihi 1 and 2", with Loihi 1 measured on Nahuku-32 silicon and Loihi 2 measured on N3B1 silicon and pre-silicon circuit simulation [1]. The mapping of those four quantities onto SANA-FE's soma, synapse and synchronization attributes is ours. Limits of the derivation: it compounds the Loihi 1 fit error (TCAD 2025: within 24.3% on latency) with the uncertainty of vendor-stated ratios; the 200 ns floor is an upper bound of "under 200ns"; NoC hop latencies are inherited because Intel's "4x faster" signaling figure concerns chip-to-chip links [1], not the on-chip mesh; energy has no scaling datum at all, so candidate energies remain Loihi 1 energies on Intel 4 silicon and must not be read as Loihi 2 energy. Runtime rates measured on Loihi 2 exist only as Figure 4 of Timcheck et al. [2]; when they are read off or measured, they belong in `loihi2_runtime.py`, not in this file.

## Architecture files and authoritative specifications

`arch/loihi2.yaml` is the repository architecture description. Its identical
packaged copy, `sanafe/examples/loihi2.yaml`, is included by the existing
`examples/*.yaml` package-data rule. `load_loihi2_candidate()` loads that
resource directly. It no longer modifies the Loihi 1 YAML at runtime.

```python
from sanafecpp import load_arch
arch = load_arch("arch/loihi2.yaml")  # Run from the SANA-FE repository root.
```

The Python profile retains capacity checks, assumed storage allocation, and
provenance. Loading YAML directly does not automatically apply those checks.
The file migration preserves the previous candidate configuration exactly.

Use Intel's technology brief [1] for advertised capabilities and resource
ceilings. Use the detailed runtime paper [2] for router connectivity and its
analytical topology. These descriptions address different architectural levels.

| Property | Confirmed hardware description | Current candidate |
| --- | --- | --- |
| Core and router grids | Intel describes an 8 × 16-core mesh [1]. The runtime paper describes an 8 × 4 router grid with four neurocores per router [2]. Both count 128 neurocores. | 32 tiles, four cores per tile. Physical tile numbering remains assumed. |
| Routing | Dimension-order XY routing [2]. | Existing mesh routing; physical fabric assignment is unverified. |
| Fabrics | Two physical fabrics [1]. The runtime analysis aggregates their links [2]. | One effective mesh. Separate fabric scheduling is not implemented. |
| Scaling | Multichip expansion to thousands of cores [1]. | This YAML describes one chip. |
| Neuron capacity | Up to 8,192 neurons per neurocore [1]. | 8,192 ceiling, with additional Python resource checks. |
| Memory | Up to 128 KB synaptic memory and 192 KB total memory per neurocore, with flexible allocation [1]. Maxima depend on model requirements. | Python capacity checks use KiB and explicitly assumed allocation sizes. |
| Neuron execution | Programmable neuron microcode [1]. | Direct integer IF implementation, without an instruction interpreter. |
| Learning | Programmable learning [1]. | Not implemented in this profile. |
| Management CPUs | Up to six embedded processors [1]. Intel also describes x86 and RISC-V management cores [4]. | Not modeled. |
| “32-bit instruction set” | Do not use this as a description of the neurocores. Management CPU ISA, neuron microcode, and the documented 32-bit graded spike payload are separate properties [1], [4]. | Binary output events; no management CPU or neuron ISA emulation. |

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

[2] J. Timcheck, A. Pierro, and S. B. Shrestha, “A Compute and Communication Runtime Model for Loihi 2,” arXiv:2601.10035v2, 2026. [Paper](https://arxiv.org/html/2601.10035v2). Section II-1 and Figure 1 document four-core routers and dimension-order routing. Section VI documents the 8 × 4 router grid and counts core-to-router links in its heaviest-link load. The candidate tile address mapping and physical fabric assignment remain unverified.

[3] SLAM Lab, “SANA-FE,” bundled Loihi configuration. [Pinned source](https://github.com/SLAM-Lab/SANA-FE/blob/93926ec8019206c1c6e6709448ac4c67f46d57db/sanafe/examples/loihi.yaml). Source for all inherited cost coefficients. The file's original header attributed them to Davies et al. (2018); only the tile-hop latencies do. The rest were fitted upstream on Intel's Nahuku Loihi 1 board (Boyle et al., TCAD 2025, Sec. VI); see the header of `arch/loihi.yaml`.

[4] A. Rao Mangalore, Intel Labs, Loihi presentation, slide “The Latest Loihi chip: Loihi 2.” [Intel-authored slides hosted by TUM](https://www.tum-venture-labs.de/media/images/Labs/Quantum/Events/HWfAI_presentations/HWfAI_Intel_Labs_Ashish_Rao_Mangalore.pdf). The slide identifies six microprocessor cores and asynchronous x86 and RISC-V. Processor ISA claims should identify the hardware revision rather than treating every Loihi 2 revision as all-x86.

Graded events, instruction execution, packed synaptic-memory traffic, detailed multicast, physical input/output transfer, and Loihi 2 cost calibration are outside this profile. Using the candidate with SANA-FE's scheduler does not execute Intel compiler output or Intel hardware.

## Integer Lava export

The optional Lava integration requires the `QCFSIFFixed` Process extension. Export a fresh, directly connected one-dimensional IF/Dense chain through `sanafe.lava.qcfs_fixed_chain_to_sanafe`. It returns the mapped `Network` and a required manifest. Attach input and observer Processes only after exporting.

The adapter supports binary Dense input, zero weight exponent, and 1–8 configured weight bits. It applies the selected Lava Dense clipping and truncation rules before storing effective integer weights. Positive and negative row sums must independently fit signed16, guaranteeing every binary fan-in combination fits the IF interface. Nonzero initial Dense buffers, branching, recurrent graphs, virtual ports, already compiled Processes, fractional values, and unsupported models are rejected. Explicit neuron initial voltages are preserved.

The exporter validates shared resource budgets before network construction. These checks currently cover the supported chain adapter. Direct use of the general SANA-FE Network API does not automatically enforce this Python resource profile.

`sanafe.lava_conv_integer` adds a restricted fixed IF/Conv/IF export for small spatial tests. It uses Lava's effective Conv weights and expands the spatial connections into mapped synapses. This is a numerical graph translation, not a claim that a physical Loihi 2 Conv program uses the same memory layout. The tested path excludes grouped and dilated convolutions, pooling, residual connections, and a trained classifier.

An opt-in zero-based `[valid_start, valid_stop)` window gates each fixed IF layer. Outside its window, the neuron consumes arriving input, preserves voltage, omits bias integration, and emits no spike. This candidate retimes a buffered feedforward chain to match original QCFS across nonzero-bias test cases. Hardware realization of that gate and its compiler schedule remain unverified.

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

The workload report gives each numbered step's operations on every occupied core and bits on directed endpoint and XY mesh links. It also reports the busiest core and link. DendOps count active IF state updates under the candidate valid-update windows. SynOps count nonzero effective weights reached by an emitted binary message. Dense SynMem entries include stored zero weights. The `dense_entries_per_read` assumption converts each incoming axon's dense entries to an assumed number of reads, with rounding per axon. The packet-width assumption turns message counts into link bits. Neither conversion is a verified Intel packing or packet format.

`max_affine_runtime` evaluates the five-term maximum in Timcheck *et al.* [2], Equation (1), on each step. It adds the resulting step bounds for a whole run. The model is separate from SANA-FE's inherited Loihi 1 `sim_time`. Its coefficients have no defaults; the module rejects missing, nonfinite, or nonpositive rates. Published Loihi 2 microbenchmarks characterize simple programs on specified hardware and software revisions [2]. Applying their rates to this custom IF program requires validation. The current output is a conditional lower bound, not calibrated Loihi 2 latency. It excludes host input and output transfer, physical instruction costs outside the five terms, and energy.

The trace analyzer supports the restricted unbranched binary dense chain and its one-step buffered update convention. It checks spike/message correspondence, mapped source and destination cores, dense fanout, and simulation update counts. It counts directed core-to-router and router-to-core endpoint traffic separately from XY router-to-router traffic. Its heaviest-link term takes the maximum over both, consistent with the scope of Timcheck *et al.* [2]. Same-core connections bypass the NoC in this candidate, which is an assumption. Two physical meshes are aggregated into one effective link, without modeling fabric assignment. The analyzer does not model general multicast, graded spikes, or the full Loihi 2 packet protocol. `rescale_binary_packet_bits()` changes an explicit packet-width scenario without rerunning the numerical simulation; it does not convert bits to BookSim flits.

## BookSim network timing controls

The optional `timing_model="cycle"` path uses the modified BookSim library. Its settings can be supplied under `architecture.attributes.booksim` in an architecture YAML file:

```yaml
architecture:
  attributes:
    width: 8
    height: 4
    link_buffer_size: 16
    booksim:
      subnets: 2
      packet_size: 1
      clock_period: 1.0e-9
      num_vcs: 1
      vc_buf_size: 8
      use_noc_latency: false
```

All `booksim` fields are optional and retain the previous values when omitted. `width` and `height` set the concentrated mesh dimensions; its four endpoint ports per router remain fixed by this BookSim fork. The 8 × 4 router grid is documented for a single Loihi 2 chip [2]; the candidate's tile address mapping is still an assumption. `Architecture.configuration()["booksim"]` records the effective settings.

`packet_size` is **flits per spike packet**, and `vc_buf_size` is flits per virtual-channel buffer. `clock_period` is seconds per simulated BookSim cycle. These parameters define a research network model and have no calibrated Loihi 2 values. In particular, `packet_size` does not specify packet bits and is independent of the `packet_bits` assumption supplied to `analyze_binary_chain`. BookSim's `channel_width` setting does not automatically convert payload bits to extra flits. The `link_buffer_size` field belongs to SANA-FE's other network timing model; it does not set BookSim's virtual-channel buffer size. Cycle scheduling releases BookSim state after each numbered step so repeated calls can execute independently.
