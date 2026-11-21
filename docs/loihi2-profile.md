# Loihi 2 candidate profile

`sanafe.loihi2.load_loihi2_candidate()` returns an architectural candidate with 128 cores in an 8 × 4 mesh of four-core tiles. It selects `integrate_fire_int24`, `accumulator_int`, and `current_based_int` with an external pre-soma buffer. Its cost coefficients start from the bundled, Nahuku-fitted Loihi 1 configuration. Four latencies and the barrier table are scaled by the factors Intel states for Loihi 2 over Loihi 1. The message receive latency is derived from a published Loihi 2 link bandwidth, and every energy, hop and message-out coefficient is inherited unchanged (see [Cost derivation](#cost-derivation)). They are scaled and derived estimates, not Loihi 2 measurements or calibrated predictions.

`candidate_profile()` returns a serializable specification with parameter provenance, profile version 1, and numerical profile identifier `qcfs-if-int24-binary-v1`. The restricted numerical contract of `integrate_fire_int24` uses signed 16-bit effective weight transport, input, and bias (the gated unit below takes a signed 24-bit input), signed 32-bit accumulation with overflow rejection, and signed 24-bit voltage clipping before a `>=` threshold comparison. Effective weight transport is not a statement of physical Loihi 2 weight precision. Documented native stored synapses have at most 8-bit weights [2]. `audit_unit_scale_signed8_weights()` flags effective software values outside a direct signed8 unit-scale scenario; it does not test exponent selection, parallel synapses, or physical packing. Each valid update emits at most one binary event and subtracts the threshold once. Connections have one logical update of delay. These are candidate software semantics, not a verified Loihi 2 neuron program or a complete implementation of Loihi 2 capabilities.

## Gated soma unit (`loihi_if_gated`)

Each core lists a second soma unit, `loihi_if_gated` (model `integrate_fire_int24_gated`, numerical profile `qcfs-if-int24-gated-v1`), with the same costs as `loihi_lif`. A neuron selects it with `soma_hw_name`. It integrates every arriving input on every update, adds `bias_at_start` once at `window_start`, and compares its threshold (`>=`, subtractive reset) only on updates `window_start + j * stride` before `window_stop`, adding `bias_per_check` at each. Its input current per update is signed 24-bit, not the 16 bits of `integrate_fire_int24`, because a convolution's fan-in in one step can exceed 16 bits; deployed Loihi 2 networks accumulate in 24 bits (Brehove et al., ICONS 2026, Sec. 3.1; Abreu et al., arXiv:2503.18002, Sec. 3). The voltage saturates at signed 24 bits, as in the other integer soma, without a signal unless the neuron sets `strict` (see [Graded messages](#graded-messages-the-graded-source-and-the-held-current)); a comparison with an unbounded reference detects it as a state mismatch. The program is realizable in the documented Loihi 2 microcode classes but is a modeled program, not a verified Intel one.

## LIF soma unit (`loihi_cuba_lif`)

Each core lists a third soma unit, `loihi_cuba_lif` (model `cuba_lif_int24`, numerical profile `lava-lif-int24-v1`), with the same inherited and scaled costs as `loihi_lif`. A neuron selects it with `soma_hw_name`. It follows the rule of the bit-accurate LIF of Lava, with integer current and voltage decay, a bias mantissa and exponent, and an optional periodic reset. It exists for published LIF networks. The thesis networks run on the two IF units and never on this one. No cost of this unit is tuned.

`tests/python/test_cuba_lif_lava.py` runs this unit and Lava's `LIF` or `LIFReset` (`Loihi1SimCfg(select_tag='fixed_pt')`, Lava fork at `74840cb8`, whose LIF file is unchanged since `e6329c27`) on the same inputs and requires every spike and every voltage to be equal at every update, with no tolerance. It covers a grid of decays and thresholds, current wrapping, a grid of bias mantissas and exponents, the periodic reset at three phases, an initial voltage and a three-layer LIF/Dense chain with odd weights. The same file runs Lava's `LIF` under the leak-free mapping of the soft-reset IF (`du = 4095`, `dv = 0`, `vth = theta - 1`, `initial_v = 32 theta`, bias exponent 6) and compares it with the soft-reset IF soma on every voltage before the first spike and every spike up to and including it. The domain of that mapping is theta at most 131,072 (the 17-bit `vth` register holds `theta - 1`), a bias in [-4096, 4095] (the 13-bit bias mantissa) and a membrane magnitude of at most 131,071 (the LIF clips 64 V at plus or minus 2^23 - 1, while the IF clips V at 2^23). `tests/python/test_integer_models.py` runs the same mapping against the SANA-FE LIF soma alone. `sanafe.lava_lif` holds the helpers (`lava_lif_trace`, `sanafe_lif_trace`, `lif_chain_to_sanafe`), and its module docstring states the alignment of a Lava run with a SANA-FE run.

### Input range

Lava's fixed-point LIF declares a signed 16-bit input port, but its fixed-point `Dense` and `RingBuffer` send signed 32-bit activations, and `subthr_dynamics` shifts whatever arrives by 6 bits without a range check. The shifted activation is added to the decayed current, and the sum is wrapped once by 2^24 into (-2^23, 2^23]. A probe of `PyLifModelBitAcc` with `dv = 0` and `vth = 131071` recorded the values below. The current `u` was read with a `Monitor`.

| Input per update | `du` | Current `u` after each update | Voltage `v` after each update |
| --- | --- | --- | --- |
| 40,000 three times | 4095 | 2,560,000, 2,560,000, 2,560,000 | 2,560,000, 5,120,000, 7,680,000 |
| -40,000 three times | 4095 | -2,560,000 (each update) | -2,560,000, -5,120,000, -7,680,000 |
| 40,000 three times | 0 | 2,560,000, 5,119,375, 7,678,125 | 2,560,000, 7,679,375, 0 (spike) |
| -40,000 three times | 0 | -2,560,000, -5,119,375, -7,678,125 | -2,560,000, -7,679,375, -8,388,607 (clipped) |

Lava therefore integrates an activation of 40,000 exactly, with no 16-bit limit. One update from rest (`du = 4095`) locates the 24-bit boundary of the current after the shift, at |a| = 2^17.

| Activation `a` | 64 a | Current `u` | Reading |
| --- | --- | --- | --- |
| 131,071 | 8,388,544 | 8,388,544 | exact |
| 131,072 | 8,388,608 | 8,388,608 | exact, `v` clips to 8,388,607 and spikes |
| 131,073 | 8,388,672 | -8,388,544 | wrapped by 2^24 |
| -131,071 | -8,388,544 | -8,388,544 | exact |
| -131,072 | -8,388,608 | 8,388,608 | wrapped by 2^24 |
| -131,073 | -8,388,672 | 8,388,544 | wrapped by 2^24 |
| 262,144 | 16,777,216 | 0 | wrapped by 2^24 |
| 262,176 | 16,779,264 | 2,048 | wrapped by 2^24 |
| 262,177 | 16,779,328 | 2,112 | wrapped by 2^24 |
| 33,554,432 (2^25) | 2^31 | -2,130,706,432 | 32-bit shift overflow, current outside 24 bits |

The single wrap keeps the current in (-2^23, 2^23] only while the sum stays in (-3 * 2^23, 3 * 2^23]. The decay keeps at most 4095/4096 of the current, because Lava adds a fixed offset of 1 to `du`, so the decayed current lies in [-8,386,559, 8,386,560]. The wrap is therefore a 24-bit wrap from every reachable current and every decay exactly when |a| is at most 2^18 + 32 = 262,176. The bound is tight on both sides. From a current of 8,388,608 with `du = 0`, an input of 262,176 holds the current at 8,388,608, while 262,177 moves it to 8,388,672 and then 8,388,735, outside 24 bits. From -8,388,544, an input of -262,176 holds it at -8,388,544, while -262,177 moves it to -8,388,608 and then -8,388,672. Past the bound, a sustained input drives Lava's current away from 24 bits, and from 2^25 the 32-bit shift itself overflows.

The neuron accepts every activation in [-262,176, 262,176], whether external (`currents`), synaptic, or their sum in one update, and reproduces Lava's wrap over that whole range. It refuses anything beyond it with a `ValueError`. The comparison with Lava covers the range with random activations drawn over all of it, together with its two ends, ±2^17 and ±40,000. A source that sent 16-bit data to Lava's LIF would overflow in its 6-bit shift for |a| of 512 or more. Lava's own fixed-point `Dense` and `RingBuffer` send 32-bit data, so this does not arise in a Lava network.

## Graded messages, the graded source and the held current

A message may carry a signed 32-bit value. The synapse stage multiplies the stored weight by it, and the accumulator of the destination core sums the products of a step. The product of a signed 16-bit weight and a 32-bit value is below 2^47, so the product and the sum are exact. The accumulator stops when a sum leaves signed 32 bits, with an `OverflowError` that names no group, and it refuses a single product beyond that range with a `ValueError`. A binary message carries no value and takes the path it took before graded messages existed. The value is not written to the message trace, so the `spikes` field of a message still counts the synapses it reaches.

Three attributes extend the soft-reset IF neuron of ANN-to-SNN conversion, as `integrate_fire_int24_gated` implements it, for graded messages. Only `graded_values` produces a value.

| Attribute | Type | Default | Meaning |
| --- | --- | --- | --- |
| `graded_values` | list of signed 32-bit integers | absent | Makes the neuron a graded source. Entry `u` is the value it sends at its update `u`, and it sends zero after the last entry. |
| `hold_input` | bool or 0/1 | false | Adds the input of each update to a current register that drives the membrane at every update before `window_stop`. |
| `strict` | bool or 0/1 | false | Stops the run when an input, the held current or the membrane leaves signed 24 bits, and records the extremes of the input and the membrane. |

The three are attributes of the existing unit, so `arch/loihi2.yaml`, `sanafe/examples/loihi2.yaml`, the candidate profile and the architecture fingerprint are unchanged, and every record keeps its profile. A graded source costs what any neuron of the gated unit costs.

### Timing and fan-out

A value sent at timestep t reaches the membrane at t + 1, as every spike does. A zero value sends nothing. No message and no synaptic event is produced, and the destination sees no input. A graded spike produces one message for each destination core that stores at least one of its synapses. Every one of them carries the value, and each synapse it reaches is one synaptic event. The messages of an image may therefore exceed its number of nonzero values.

### The graded source

A graded source keeps its own cursor over `graded_values` and integrates nothing. It ignores its threshold, window, biases and currents, and its voltage stays at the initial voltage. It takes no synaptic input, and a nonzero input raises a `ValueError` whose message begins "A graded source takes no synaptic input" and names no group. It does not take `hold_input`, and setting the attribute on a source raises a `ValueError`. An empty list makes a source that sends nothing. A value of `graded_values` beyond signed 32 bits is refused when the attribute is set, with a `RuntimeError`. A reset restarts the cursor. A source in strict mode is accepted and reports no trace.

Any soma model that returns a current together with a fired status now sends that current as the value of its spike, where it was ignored before. The value must be a nonzero integer within signed 32 bits. No soma model of the repository other than the graded source returns one. A message gained a member, so a plugin must be built against the headers of the fork it is loaded into.

### The held current

With `hold_input` the neuron adds the input of an update to a register, and the register drives the membrane at every update before `window_stop`. This includes the updates before `window_start`. The bias at the start of the window and the biases of the checks are added as before, and the threshold is compared only at the checks. From `window_stop` on the register is neither updated nor added, and input that arrives then is added to the membrane once. The register is signed 24 bits and saturates there unless the neuron is strict. A reset clears it. The neuron reports it as the trace `held_current`, its value after each update.

A value sent once and held equals the same value sent at every timestep only when it arrives at `window_start`. A value sent at timestep t arrives at t + 1, so the source sends from one step before the window starts. A value that arrives earlier also drives the updates before the window.

### The strict mode

A strict neuron checks three quantities of each update against signed 24 bits. These are the input (synaptic plus external), the held current when the neuron holds its input (checked only for updates before `window_stop`), and the membrane after the drive (the input, or the held current of a neuron with `hold_input`) and the biases are added and before the threshold is subtracted. Only the total of an update is checked, and no partial sum of it. The first quantity outside the range stops the run with an `OverflowError` whose message begins with the group and the neuron and holds `strict mode:`, the quantity and its value. The accumulator raises its own `OverflowError` above 32 bits, without a group name, so a caller that wants strict stops matches on `strict mode:`. Lava's fixed-point `Conv` wraps the sum of one projection at 24 bits, while the strict mode checks the summed input of an update. The two coincide when a neuron receives one projection, as in every layer of a plain chain, and differ where two projections meet, as at a residual shortcut. No test drives Lava past the wrap. Without `strict`, an input outside signed 24 bits raises a `ValueError` ("IF signed 24-bit input current must be a finite integer ...") that names no group, while the membrane and the held current saturate. Strict turns the input case into the `OverflowError` that names the group and the neuron. After a stop the simulation is discarded or reset before reuse.

A strict neuron that is not a graded source records four extremes since its last reset, `input_min`, `input_max`, `voltage_min` and `voltage_max`. The input and the membrane of every update that completes are recorded, including updates in which no message arrived, so the smallest input of a neuron that receives only positive input reads 0. The initial voltage is not recorded on its own, and an update that stops records nothing. The recorded membrane is the value before the threshold subtraction. The `neuron_trace` of a run is a dictionary keyed by trace name, and each entry is a list with one flat list per step, ordered by group name and then by neuron, that holds the neurons that are strict and not sources. The held current is reported the same way for the neurons that hold their input. The dictionary is empty when no neuron reports a trace.

### Where the soma may run

A graded source, a strict neuron and a neuron with `hold_input` are each refused when the time-step buffer of their core is inside the soma unit or before the axon-out unit, because the soma then runs on the message path. The first step of the run raises a `ValueError` that names the group and the neuron. These three positions are accepted. Only the position before the soma unit, the position of the candidate profile, sums every input of a step. Inside the dendrite unit the soma receives nothing. Before the dendrite unit only the last synaptic output a neuron receives in a step reaches the soma, because each synapse overwrites the buffer, so two graded sources into one neuron give the value of one of them where the candidate's position gives their sum.

With the buffer inside the dendrite unit, the integer accumulator of the candidate delivers no input to the soma, for binary and graded messages alike. The update of the soma at step t + 1 calls the accumulator first, and the accumulator clears the sum stored at step t when it sees a new timestep. This holds without graded messages, and a short run on such an architecture shows every potential at zero.

### Counting the input delivery

`sanafe.loihi2_counts.input_delivery_counts(result, source_groups)` counts the messages and synaptic events of the delivery of an image apart from the spiking layers. Every message of the message trace that is not a placeholder counts once, and its `spikes` count as synaptic events. A message whose source group is in `source_groups` counts under `input`, with one entry per step, and every other message under `layers`. A string is taken as one group name, and a name that does not occur gives zeros, as it should for an image of zeros, which sends nothing. The delivery is counted once per image and never inside the synaptic operations of the spiking layers. A result without a message trace raises a `ValueError`.

### The cost assumption

No cost depends on the width of a value. A graded message is charged as a message and each of its synaptic events as a synaptic event, because no published cost separates them. The receive latency of the profile was derived from a 32-bit message, which holds any value a graded message carries here. Every time or energy of a graded network is therefore stated with the assumption that a graded message costs what a message costs.

### The order of a neuron's messages

The destination cores of a neuron are ordered by global core id (`CoreIdLess` in `sim_get_post_synaptic_cores`, `src/chip.cpp`). Core ids are given in creation order, tile by tile, so the order is ascending (tile, core offset) for the architecture files of the repository. A neuron's messages, their identifiers and their generation delays follow that order. Every field of a run, the modeled time included, therefore repeats between constructions of a chip. The upstream code allocated in ascending core id until commit a851812 of 2024, which introduced a set ordered by memory address. Cores of one tile lie in one vector, so within a tile the address order equals the offset order. A network inside one tile is unchanged bit for bit. Cores of different tiles lie in separate allocations, so the address order differed between constructions for a network across tiles. Results made before this change with networks across tiles carry modeled times that differ between constructions in their last digits. Graded messages are created in the same loop over the axons of a neuron and take the same order.

### Checks

`tests/python/test_graded_messages.py` checks the product, signs, the zero value, the fan-out to two cores, the source, the refused buffer positions, the strict stops and extremes, the held current at both edges of its window, the equality of the two deliveries, the two binary pins (one tile, and across tiles on every field of the run), the ascending order of a neuron's messages over eight cores on eight tiles, five constructions of that chip that give one run, and `input_delivery_counts`. `tests/python/test_graded_lava.py` runs Lava's fixed-point `Dense` and `Conv` with message bits set, each with a binary control run that differs, and Lava's `LIF` with `du = -1`, and requires every voltage and the held current to equal those of the simulator. It is listed in `tests/python/thesis_tree.py` with Lava and its `lava.proc.qcfs` extension as prerequisites, so it fails with a message naming each one that is missing, and a checkout of the fork alone leaves it out with `pytest -m "not thesis"`. The runs used the local Lava checkout at `74840cb8`, the output of `git -C Lava rev-parse --short HEAD`.

## Cost derivation

Decided 2026-09-30. The candidate's costs are the fitted Loihi 1 coefficients of `arch/loihi.yaml` (see that file's header for their Nahuku provenance), with the following changes and nothing else. Each coefficient carries one of three statuses in the provenance table, `scaled`, `derived` or `inherited`. `sanafe.loihi2.candidate_cost_provenance()` returns the same table programmatically and `test_loihi2_profile` checks the YAML against it.

| Coefficient | Loihi 1 fitted | Factor | Candidate | Basis |
| --- | --- | --- | --- | --- |
| `latency_access_neuron`, `latency_update_neuron` | 6.0 ns, 3.7 ns | ÷ 2 | 3.0 ns, 1.85 ns | "2x for simple neuron state updates" [1] |
| `latency_process_spike` | 3.8 ns | ÷ 5 | 0.76 ns | "5x for synaptic operations" [1] |
| `latency_spike_out` | 30 ns | ÷ 10 | 3.0 ns | "10x for spike generation" [1] |
| `latency_sync` table, 1 / 2 / 4 / 29 tiles | 0.6 / 1.0 / 1.4 / 1.8 µs | ÷ 9 | 66.7 / 111 / 156 / 200 ns | chip-wide entry set to the stated "minimum chip-wide time steps under 200ns" [1] |
| `latency_message_in` | 16 ns | derived | 3.536 ns | 32 bit at the effective link bandwidth of 9.05e9 bit/s [2], alternative reading 8.42e9 bit/s gives 3.800 ns |
| every `energy_*` coefficient (`energy_message_in` included), hop and message-out coefficient, `link_buffer_size` | as Loihi 1 | 1 | unchanged | no public Loihi 2 datum |

The receive latency of the `axon_in` unit is the one derived coefficient. Timcheck, Pierro and Shrestha report an effective Loihi 2 link bandwidth of 1.053 GiB/s in Figure 4(d) [2], read here as 9.05e9 bit/s, with the decimal reading 1.053 GB/s (8.42e9 bit/s) as the alternative. One 32-bit message then occupies a link for 32 / 9.05e9 = 3.536 ns (alternative 3.800 ns). The constant `MESSAGE_IN_DERIVATION` in `sanafe/loihi2.py` holds the three inputs, and `candidate_cost_provenance()` returns both results with the source. The derivation rests on an assumption. The time a destination core spends receiving one message is taken to be the time one 32-bit message occupies a link at the measured effective bandwidth. This identifies a link measurement with a per-message cost of the simulator's receive stage. It is an assumption and not a measurement of that stage. It was compared with one published deployment only, PilotNet LIF, the deployment whose time prompted the change, so the comparison is not a held-out test. There the simulator gives 154.1 µs per timestep against a measured 150.4 to 152.9 µs, under the layout taken from the published per-core plots. The deployment excludes the first chip's 16 ns and does not identify the value. Any receive cost from about 0.5 to 4.9 ns agrees within 5 percent with all three published runs.

Intel's footnote to the factors states that they are "based on comparisons between barrier synchronization time, synaptic update time, neuron update time, and neuron spike times between Loihi 1 and 2", with Loihi 1 measured on Nahuku-32 silicon and Loihi 2 measured on N3B1 silicon and pre-silicon circuit simulation [1]. The mapping of those four quantities onto SANA-FE's soma, synapse and synchronization attributes is ours. The derivation has five limits. It compounds the Loihi 1 fit error (within 24.3% on latency, TCAD 2025) with the uncertainty of vendor-stated ratios. The 200 ns floor is an upper bound of "under 200ns". NoC hop latencies and the message-out latency are inherited, because Intel's "4x faster" signaling figure concerns chip-to-chip links [1] and not the on-chip mesh. The receive latency inherits the uncertainty of reading a bandwidth off a figure and the assumption above. Energy has no scaling datum at all, so candidate energies remain Loihi 1 energies on Intel 4 silicon and must not be read as Loihi 2 energy. Runtime rates measured on Loihi 2 exist only as Figure 4 of Timcheck et al. [2]. The rates of Figures 4(a) to 4(c), read off or measured, belong in `loihi2_runtime.py`. The link bandwidth of Figure 4(d) is the one exception, since this profile derives its receive latency from it, so it is held in `MESSAGE_IN_DERIVATION` in `sanafe/loihi2.py`, and `test_loihi2_profile` checks that every other place that states it agrees.

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
| “32-bit instruction set” | Do not use this as a description of the neurocores. Management CPU ISA, neuron microcode, and the documented 32-bit graded spike payload are separate properties [1], [4]. | Binary output events, and the graded messages of the graded source; no management CPU or neuron ISA emulation. |

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

[3] SLAM Lab, “SANA-FE,” bundled Loihi configuration. [Pinned source](https://github.com/SLAM-Lab/SANA-FE/blob/93926ec8019206c1c6e6709448ac4c67f46d57db/sanafe/examples/loihi.yaml). Source for all inherited cost coefficients and for the Loihi 1 base values of the scaled ones. The file's original header attributed them to Davies et al. (2018), and only the tile-hop latencies come from that source. The rest were fitted upstream on Intel's Nahuku Loihi 1 board (Boyle et al., TCAD 2025, Sec. VI), as the header of `arch/loihi.yaml` records.

[4] A. Rao Mangalore, Intel Labs, Loihi presentation, slide “The Latest Loihi chip: Loihi 2.” [Intel-authored slides hosted by TUM](https://www.tum-venture-labs.de/media/images/Labs/Quantum/Events/HWfAI_presentations/HWfAI_Intel_Labs_Ashish_Rao_Mangalore.pdf). The slide identifies six microprocessor cores and asynchronous x86 and RISC-V. Processor ISA claims should identify the hardware revision rather than treating every Loihi 2 revision as all-x86.

Costs that depend on the width of a graded value, instruction execution, packed synaptic-memory traffic, detailed multicast, physical input/output transfer, and Loihi 2 cost calibration are outside this profile. Using the candidate with SANA-FE's scheduler does not execute Intel compiler output or Intel hardware.

## Integer Lava export

The optional Lava integration requires the `QCFSIFFixed` Process extension. Export a fresh, directly connected one-dimensional IF/Dense chain through `sanafe.lava.qcfs_fixed_chain_to_sanafe`. It returns the mapped `Network` and a required manifest. Attach input and observer Processes only after exporting.

The adapter supports binary Dense input, zero weight exponent, and 1–8 configured weight bits. It applies the selected Lava Dense clipping and truncation rules before storing effective integer weights. Positive and negative row sums must independently fit signed16, guaranteeing every binary fan-in combination fits the IF interface. Nonzero initial Dense buffers, branching, recurrent graphs, virtual ports, already compiled Processes, fractional values, and unsupported models are rejected. Explicit neuron initial voltages are preserved.

The exporter validates shared resource budgets before network construction. These checks currently cover the supported chain adapter. Direct use of the general SANA-FE Network API does not automatically enforce this Python resource profile.

`sanafe.lava_conv_integer` adds a restricted fixed IF/Conv/IF export for small spatial tests. It uses Lava's effective Conv weights and expands the spatial connections into mapped synapses. This is a numerical graph translation, not a claim that a physical Loihi 2 Conv program uses the same memory layout. The tested path excludes grouped and dilated convolutions, pooling, residual connections, and a trained classifier.

An opt-in zero-based `[valid_start, valid_stop)` window gates each fixed IF layer. Outside its window, the neuron consumes arriving input, preserves voltage, omits bias integration, and emits no spike. This candidate retimes a buffered feedforward chain to match original QCFS across nonzero-bias test cases. Hardware realization of that gate and its compiler schedule remain unverified.

`Architecture.configuration()` provides a complete configuration snapshot for provenance. The candidate adapter verifies its fingerprint, including topology, buffering, pipeline parameters, synchronization, and costs, against the loader's configuration. Modified architectures are rejected rather than receiving incorrect provenance. The manifest records the fingerprint, numerical model classes, effective weights, state, currents, placement, and actual allocation assumptions. Save the configuration snapshot alongside experiment results.

The integer pipeline carries bounded integers through SANA-FE's existing double-valued pipeline interface. All transported signed32 values are exactly representable. Neuron integration uses widened integer arithmetic, with explicit clipping at the voltage boundary. A binary message carries no value. Graded messages are described in the section "Graded messages, the graded source and the held current" above.

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
