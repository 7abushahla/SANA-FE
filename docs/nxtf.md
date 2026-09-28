# Public NxTF partition import

`sanafe.nxtf` consumes a JSON export of the public NxTF partition compiler. TensorFlow and NxSDK are not dependencies of this importer.

```python
import json
import sanafe
from sanafe.nxtf import map_compiled_partitions

manifest = json.load(open('compiled.json'))
network, architecture, report = map_compiled_partitions(
    manifest, input_biases={'input': [128, 64, 192]})
chip = sanafe.SpikingChip(architecture)
chip.load(network)
result = chip.sim(16, potential_trace=True, spike_trace=True)
```

The public compiler export is available in [the maintained model-source fork](https://github.com/7abushahla/nxsdk-models), under `nxsdk_modules_ncl.dnn.src.offline.compile_model`. It executes the original partition optimizer, layer compilers, and public synapse encoder. The JSON retains raw compartment, synapse, and axon structures plus a decoded graph. Its provenance is declarative, not an authenticated compiler certificate. The importer validates the decoded supported configuration; it does not recompile and compare every raw IR field.

## Supported execution

The `paired_compartment_if` model is a bounded, source-derived Loihi 1 candidate. It preserves separate dendrite and soma compartments, local compartment order, padded slots, and explicit negative recurrent reset synapses. A soma compares its adjacent dendrite voltage strictly above the threshold. Its reset spike reaches the dendrite on the next update. Threshold and synaptic weight scaling use the mapped integer units. If encoded reset feedback differs from the threshold, the report records that difference.

Supported configurations use zero voltage leak, fully decayed current, zero refractory interval, zero initial voltage, eight-bit shared-sign weights, exponent −6 through 7, and no explicit axonal delay. Encodings are `sparse`, `dense1`, and `runlength`; `dense2` filler connectivity is currently rejected. Signed 24-bit overflow raises an exception. Overflow, stochastic rounding, arbitrary stack programs, learning, and general Loihi neuron behavior are not emulated.

Input overrides replace input-compartment bias mantissas in logical neuron order, preserving their compiled exponent. They hold that digital drive for the complete run. They do not model host transfer or per-image register writes. Signed input encoders require explicit treatment and are not established by the small unsigned fixtures.

`placements` optionally maps every compiled partition ID to `[tile, core]` on one chip. Each partition occupies a distinct core. The default assigns cores sequentially. These are SANA-FE placements, not NxSDK placements. Mapping preserves compiler compartment order regardless of dictionary insertion order.

`potential_trace` follows SANA-FE's sorted group-name order, then local neuron offset. The execution report's `nodes` follows manifest partition order. Match columns by `(partition, cx)` when comparing these representations. Spike records carry `group_name` and `neuron_offset` explicitly.

## Evidence limits

This backend does not run the restricted NxSDK synapse-memory compiler, map board registers, or execute an Intel binary. It does not reproduce packed-memory reads or axon multicast behavior exactly. The original Loihi architecture's timing and energy coefficients are inherited; the additional compartment model has not been calibrated. Padded dummy slots occupy mapped positions but use the backend's idle accounting.

The pass-stack interpretation and update recurrence are tested against a separately implemented reference. Those tests establish software consistency. Silicon and NxSDK trace equivalence remain unverified. The existing `leaky_integrate_fire` model is unchanged.
