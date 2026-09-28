Restricted Lava bridge
======================

``sanafe.lava.qcfs_chain_to_sanafe`` exports a fresh, explicitly connected
chain of one-dimensional ``QCFSIF`` Processes and plain ``Dense`` connections.
It requires NumPy and a Lava installation that provides
``lava.proc.qcfs.QCFSIF``. It does not compile arbitrary Lava ProcessModels.
The older shape-based conversion helpers remain unchanged.

The caller supplies the numeric first-layer current for each update, a target
architecture, and one ``(tile_index, core_index)`` placement per layer.
First-layer input ports must be unconnected at export. Other ports must form
exactly the supplied chain. Attach sources and observers after export when
running the same graph in Lava.

.. code-block:: python

   from sanafe.lava import qcfs_chain_to_sanafe

   # first.s_out -> dense.s_in; dense.a_out -> second.a_in
   network = qcfs_chain_to_sanafe(
       [first, second], [dense], currents, architecture,
       placements=[(0, 0), (1, 0)])

The architecture must use ``integrate_fire_float32`` as its default soma and
an external pre-soma timestep buffer, an ``accumulator`` dendrite, and a
``current_based`` synapse. The bridge checks core capacity and
copies weights and initial state without additional threshold scaling.
Every Dense connection introduces one neural update of propagation delay.
Readout and drain steps must be specified by the caller.

Float32 IF reference model
--------------------------

This model is a numerical reference, not an implementation of a processor's
instruction set. It integrates signed input without leak, fires when voltage
is greater than or equal to threshold, and subtracts the threshold once.
At most one spike is emitted per update. There is no voltage clamp or
refractory period. Each neuron retains its own state.

Attributes are ``threshold`` (positive, default 1), ``initial_voltage``
(default 0), ``bias`` (default 0), and optional ``currents`` (external current
per update, followed by zero). ``reset()`` restores initial voltage and the
external-current cursor. SANA-FE's global timestep and accounting counters
retain their existing reset semantics.

Arithmetic casts the accumulated synaptic current to float32, adds external
current and bias, then integrates voltage. The default accumulator itself
uses double precision. Exact agreement with a framework's float32 matrix
reduction therefore requires suitable representable values or an additional
accumulator model. The bridge does not establish general bit equivalence.

Timing boundary
---------------

The current array is a functional stimulus. Its host transfer, input encoding,
and first-layer affine computation have no modeled cost. Model costs must be
supplied by the architecture. Reusing the bundled Loihi architecture's costs
with this neuron is an exploratory hybrid model, not a calibrated Loihi 2
configuration or a validated implementation of this neuron on Loihi 1.
