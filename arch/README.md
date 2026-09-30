## Architecture Description

The architecture description format is also based on the YAML file format.

Different architectures are defined using a hierarchical description.
This tool models neuromorphic designs with several assumptions, in order to
simplify the tool.

1) The chip is time-step based. A time-step is a small discrete amount of time.
    This is as opposed to a purely event driven simulation e.g. ROSS.
2) The neural cores adhere to some common design patterns

At the top level, the description begins with the "architecture" keyword. Any
other top-level sections will be ignored. This defines anything at the chip
level, including the NoC interconnect.

A chip contains one or more network tiles, representing some shared network
resources e.g., a router. Each `tile` contains one or more cores, where a core
performs computation. Each neuromorphic `core` contains a fixed spike processing
hardware pipeline. It is assumed that tiles and cores are all parallel processing
elements.

Each core is assumed to have a neuromorphic pipeline which processes the updates
for one or more neurons. The pipeline is a fixed sequence of niche hardware
units. Those hardware units could contain digital logic, analog circuits or
even novel devices.

The pipeline contains the following units:

* The input axon unit receive spike packets from the network and generate
  synaptic addresses for memory lookups.

* The synaptic unit looks up connectivity for incoming spikes and updates any
  relevant synaptic currents.

* The dendritic unit combines currents based some internal structure and a set
  of operations.

* The soma unit updates membrane potentials based on the dendritic current and
  neuron model. If the firing criteria is met, it generates a spike for that
  neuron.

* The output axon unit send spikes from the soma out to the network to go to
  other cores' pipelines.

For an example, see `arch/loihi.yaml`. There are a nested series of
keywords, where keywords define required hardware units. Each block must be
contain a `name` keyword, which may optionally specify the number of instances.
Units are duplicated the number specified in the range, for example:

    # Define 8 cores, 0 through 7
    -name: neuromorphic_core[0..7]

Units must also have both an `attributes` section and the next hardware units
in the hierarchy. The attributes section will generate one or more parameters
that are passed to be parsed by the simulator and the relevant hardware models
implemented either internally (`models.cpp`) or externally (plugins).


## Loihi 2 candidate

`loihi2.yaml` provides the integer IF candidate architecture. Its packaged copy
is `sanafe/examples/loihi2.yaml`; keep both identical. The Python helper
`sanafe.loihi2.load_loihi2_candidate()` reads the packaged copy through the
standard architecture loader. See [the profile documentation](../docs/loihi2-profile.md)
for sources, resource checks, and unsupported hardware capabilities.
Its latencies are the fitted Loihi 1 values scaled by Intel's stated Loihi 2
factors; its energies are inherited from Loihi 1. Neither is a calibrated
Loihi 2 estimate (see the profile documentation's cost derivation).

## TrueNorth, documented costs

`truenorth_documented.yaml` is the shipped `truenorth.yaml` with the two
published figures that exist: 26 pJ per synaptic event (Merolla et al.,
Science 2014, a total-energy operating point) and the 1 ms tick as a fixed
synchronization cost. Every other coefficient stays zero. The shipped file
is unchanged and remains the functional-only profile.

## Speck, preview only

`speck.yaml` exists so the Studio can draw SynSense Speck and show its catalog
card: one star router with nine cores whose neuron ceilings follow the
published per-core memory table (Richter et al. 2024, Yao et al. 2024). Every
cost is zero, the unit models are stand-ins, and no session can be started on
it: Speck has no global time-step, and the event engine that would run it is
planned, not built.
