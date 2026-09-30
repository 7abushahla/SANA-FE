# SANA-FE Studio

The Studio engine advances a SANA-FE network one numbered update at a time and
turns each update into an `UpdateRecord`. A local server and browser page show
the chip, its placement, the packets, each core's pipeline, and the modeled
performance as updates arrive. The engine is usable on its own from a script
or notebook.

```python
from sanafe.studio.engine import SanafeFiles, Session

session = Session(SanafeFiles(), {
    'arch_yaml': 'arch/example_chip.yaml',
    'net_file': 'snn/example.net',
    'horizon': 5,
}, store_dir='studio-runs')
session.step(2)                      # two updates
records = session.run_to_horizon(    # the remaining three, unless stopped
    stop_when=lambda r: 'burst' if r.counts['fired'] > 4 else None)
print(session.state, session.stop_reason)
print(records[-1].core_finish, records[-1].barrier)
```

## What a record contains

Every field has a provenance mark in `sanafe.studio.engine.PROVENANCE`. R means
recorded by SANA-FE. D means derived exactly from recorded values. X means
reconstructed.

- Chip energy by unit type, step time, and counts are recorded.
- Each message's delays and timestamps are recorded. Its router-by-router
  `path` is reconstructed with the x-then-y rule in `src/schedule.cpp`,
  because the detailed timing model records no per-router events.
- `fired` lists only neurons created with `log_spikes`, because SANA-FE's
  spike trace records only those. `counts['fired']` counts every neuron. A
  core's `core_counts[...]['fired']` is `None` unless all of its neurons log
  spikes, so a partial count is never shown as exact.
- A core's finish time is the latest send timestamp among its records in the
  update. The barrier is the step time after the last core finish or message.
- Per-core and per-unit energy is recorded. The session simulates on a copy of
  the architecture YAML with `log_energy` enabled. The copy differs from the
  source only in those flags, and the manifest stores both fingerprints. Unit
  logging is skipped for cores with more than eight synapse, dendrite, and soma
  unit instances, such as the bundled Loihi 1 file.

## Connectivity, budgets, and references

After loading, a session summarizes the mapped network once
(`session.connectivity`): synapses between groups, links between cores with
their synapse and axon counts, the neuron ranges on each core, and each
group's threshold when it is uniform. `session.neuron_detail(group, offset)`
returns one neuron's core, attributes, fan-in, and fan-out. All of it is read
from the network SANA-FE was given (R).

When the architecture is the Loihi 2 candidate (same configuration
fingerprint), the session checks every core with
`sanafe.loihi2.validate_core_budgets`: at most 8,192 neurons, 128 KiB of
synapses, and 192 KiB in total under the assumed byte layout in
`sanafe.loihi2.Allocation`. A core over budget fails the build with the
reason. Other architectures are limited only by their own YAML.

A workload may attach a reference checker. `check(record)` returns
`{'status': 'match'}`, `{'status': 'unchecked', 'reason'}`, or
`{'status': 'mismatch'}` naming the reference, neuron, quantity, expected and
actual values. The thesis QCFS workload compares every update with Lava at
the same update and with SpikingJelly shifted by one update per layer.

## Debugging

**Breakpoints** are JSON conditions checked after every update, never code:

| kind | fields | stops after the update in which |
| --- | --- | --- |
| `neuron_fires` | `neuron` (a neuron that logs spikes) | the neuron fired |
| `core_sends` | `core`, `more_than` | the core sent more than that many packets |
| `step_time` | `more_than` (seconds) | the modeled step time exceeded the limit |
| `update` | `equals` | the update number equals the value |
| `reference_mismatch` | none | the workload's reference check reported a mismatch |

`session.set_breakpoints([...])` (or `PUT /api/sessions/{id}/breakpoints`)
replaces the list; it takes effect from the next update, even during a run.
A hit leaves the session `stopped` with a reason such as
`breakpoint b1: layer_2.4 fires`. A Python `stop_when` callable still works.

**Placement editing.** `Session(..., core_map={'0.0': '31.3', '31.3': '0.0'})`
moves every neuron of each key core to its value core before loading. The
map is relative to the workload's own placement. Neurons are re-mapped in
their original order, because SANA-FE places neurons within a core in
mapping order. On the page, drag a used core onto another core to swap
them, then press "Rebuild with edits". The reference check must still pass,
and spike trains must not change; time, energy, and hops may.

**Saved runs.** Each build writes a trace store, by default under
`~/.sanafe-studio/runs` (`--store-dir` moves it, `--no-store` disables it).
`list_runs`, `load_run`, and `compare_runs` read them. A comparison reports
whether the two spike trains are identical (or the first update and neuron
where they differ), and per update the step time, energy, hops, messages,
occupied cores, the latest core finish, and the busiest core's packets.
Runs of different networks are reported as not comparable.

**Export.** `export_plot(records, kind)` renders a `sanafe.viz` plot of a
saved run as SVG: `raster`, `potential`, `energy`, `throughput`, or
`latency`. Titles state that the values are modeled, not measured.

## Trace levels and chip scale

`full` keeps every message of every update. `aggregate` keeps, per update,
the chip counts and energy, each core's counts, finish time, and energy,
the packets over each mesh link (following the reconstructed x-then-y route,
so marked X), and the membranes and spikes of watched neurons only. The
worker still holds every update's full membranes and spikes as arrays: the
reference check reads them, `session.core_state(core, update)` serves one
core's neurons at any past update, and `neuron_detail` returns a neuron's
whole history, so a watch added late still shows every update. A workload
may declare `default_trace_level`; the rail can override it.

The stage 5 probe (ResNet-20, T = 2, 109 cores, 286,720 neurons) measured
0.1 to 2.4 s per `chip.sim(1)`, but 1.5 to 9.4 s to build a full record and
6 to 12 MB of JSON per update, against a 5 s and live-playback budget. The
simulator was not the bottleneck, so the aggregate level was added instead
of C++ counters. On the same network the slowest update (simulation, record,
and reference check) takes 0.3 s at T = 2 and 1.3 s at T = 8, records are
about 30 kB, and the test process's memory high-water mark, over both runs
in turn, is 2.3 GB.

The aggregate level keeps only watched neurons' spikes, so Compare does not
claim spike-train identity for aggregate runs (it still compares time,
energy, hops, and per-core load), and raster and membrane exports of an
aggregate run show the watched neurons only. A `neuron_fires` breakpoint
still reads every neuron's spikes in the worker.

At the aggregate level the chip draws link width by packets per link and
animates a sample of packets: the first message between each pair of cores
(at most 256 per update), with its recorded times. The timeline keeps core
finish bars and the barrier, Messages lists packets per link, and the core
view fetches its membranes for the shown update.

In the "slow motion around packets" clock each packet stays on screen for at
least 4% of its update. ResNet-20 flights last about 10 ns in updates of
about 80 µs, so at true proportions they would never be visible. The
"modeled time" clock keeps true proportions.

## Limits

A fault inside SANA-FE moves the session to `faulted`, and only `reset()`
clears it. Meshes with more than 256 tiles (TrueNorth's 64 × 64) draw the
occupied tiles and a one-tile margin; the mini-map shows every tile as a
dot. The aggregate level keeps no per-message timing, so packets are
not animated and the timeline has no message rows.

## Running the Studio

```bash
.venv/bin/python -m sanafe.studio --port 8765
```

Open `http://127.0.0.1:8765/`. The server listens on this machine only. Add
workloads with `--workload NAME=module:Class` and `--path DIR`. Add
`--store-dir DIR` to save every run as a trace store. The thesis launcher
registers the QCFS compact network and QCFS ResNet-20:

```bash
SANA-FE/.venv/bin/python SANA-FE-thesis/studio_workloads/launch.py
```

Each session runs in its own worker process. A crash in SANA-FE faults that
session only; the page says so and offers a new session.
The address carries the session id (`#session=...`), so reloading the page
rejoins the session and catches up on updates it missed. The server accepts
only JSON POSTs with a loopback Host header, so other web pages cannot
drive it.

## What the page shows

- **Transport bar.** Step, Run n updates, Run to horizon, Pause, and Reset.
  A pause stops the run at the next update boundary. A pause sent while
  nothing runs is ignored.
- **Session rail.** A platform dropdown comes first: the catalog in
  `sanafe.platforms` (Intel Loihi 1, the Intel Loihi 2 candidate, IBM
  TrueNorth functional, IBM TrueNorth with documented costs). The workload
  list shows only workloads that list the chosen platform, plus the file
  workload, which shows "architecture from file". A server started before
  the catalog existed has no platform list: the page disables the dropdown,
  lists every workload, and says to restart the server (page files are read
  from disk, the server's Python is not). The form below comes from
  the workload's parameter schema. `T` and the image index are typed
  numbers. Changing them and pressing Rebuild starts a fresh session.
- **Demo workload.** `random-snn` runs on every platform: a seeded random
  feed-forward network in the shape of the TCAD 2025 randomized benchmark,
  with a bias-driven first group, the connection percentage, packed or
  spread placement, and each platform's own soma model. It has no
  reference and no readout.
- **Chip.** Tiles hold a router and their cores. Cores take the color of
  the group with most neurons on them. Packets move on their recorded send
  and receive times along the reconstructed x-then-y route. Yellow dots mark
  cores still processing neurons. Purple outlines mark cores processing a
  message.
- **Clock.** "Modeled time" keeps true proportions, so packets appear only
  briefly. "Slow motion around packets" slows the clock only while packets
  fly. Both keep the recorded order.
- **Pause, Stop, Reset.** The simulation runs on the server and the page
  animates the updates it has received, often long after the server has
  finished. Pause (or the space bar) freezes the animation where it is and,
  if the simulation is still computing, holds it at the next update
  boundary; Resume continues both. Stop ends the run: the simulation halts
  at the next boundary, the view freezes on the current frame, and nothing
  resumes. Run or Step then continues from the halted update, and every
  computed update stays on the scrubber. Reset works at any time; mid-run it
  halts first, then rebuilds at update 0.
- **Playback.** Wall-clock seconds spent on each update, from 6 s down to
  0.15 s, plus "instant", which draws each update once and moves on without
  animating it. The hint beside the control gives the playback time for the
  whole horizon, which is what makes a 20-update ResNet-20 session feel
  slower than a 5-update compact one at the same setting. Playback speed
  never changes the simulation, only how long you watch it; an update that
  has not been computed yet is still waited for.
- **Timeline.** One row per occupied core. Bars show neuron processing,
  purple marks show message processing, lines join send to receive, and the
  gray band is the barrier.
- **Live performance and messages.** Energy per update by unit, step time,
  message counts, and the current update's recorded messages.
- **Zoom and mini-map.** Click a tile to open it: each core shows its
  pipeline stages (axon in, synapse, dendrite, the update-boundary buffer,
  soma, axon out) with counters at the playhead. Click a core panel, or
  double-click a core on the chip, to open the core view: the pipeline, the
  axon_in and axon_out tables, and one cell per neuron filled by its
  membrane divided by its threshold. A red cell fired in this update. Cores
  with more than 1,024 neurons show a heatmap. The breadcrumb and the
  mini-map lead back to the chip.
- **Network mode.** Groups top to bottom in dependency order, one per row,
  with the host operations above and below. Consecutive groups are joined
  by an arrow; skip connections (ResNet shortcuts) arc through lanes on the
  left, labelled outside the lanes. Each box keeps its full name, and the
  cores it is mapped to sit beside it. Clicking a core opens it.
- **Pipeline.** For workloads that declare one (both QCFS workloads): the
  input image and the host current it becomes, a space-time grid of every
  IF site's firing rate (D) on either hardware updates (default; gated
  cells outside each site's validity window are hatched, and a strip shows
  each update's step time R and barrier share D) or algorithm timesteps (the
  same run re-indexed, D), and the host readout accumulated over the output
  window with the reference check. Both bundled checkpoints are untrained
  fixtures; the view says so and the class it shows is not a result.
- **Inspector.** Numbers for the chip, a tile, a core, a group, a neuron, or
  a message, each with its R, D, or X mark. A neuron shows its attributes,
  fan-in and fan-out, and the reference values at this update. On the
  candidate, a core shows its assumed storage against the budgets.
- **Neuron watch.** The membrane of each watched neuron per update, with the
  Lava and SpikingJelly traces dashed when the workload provides them.
- **Messages.** Filter by core or neuron. Clicking a message draws its
  reconstructed route (X) on the chip.
- **Platform.** The card for the session's platform. Its first line says
  nothing on it is a hardware measurement. Then come the title, vendor and
  generation, the execution rule, a summary, the structure read from the
  loaded architecture (mesh, tiles, cores, neurons per core, buffer
  position, synchronization), the units of the reference core with the
  model each runs and the attributes that model accepts, a costs table with
  every coefficient's value, status and source, the validation statement,
  the not-modeled list, and references. Status words use the mark colors:
  `fitted` and `documented` in green, `scaled` and `inherited` in orange,
  `none` in gray. In the inspector every modeled time and energy shows the
  platform's word after its mark, as "R · fitted", "R · scaled",
  "R · inherited", "R · documented" or "R · none". A workload that
  declares a platform but simulates a modified architecture is badged
  "(modified architecture)" and its card says the costs table describes
  the catalog profile, not what ran. A workload that brings its own file
  gets the card and badge of the catalog platform that file matches
  exactly, if any. A platform whose card cannot be built stays in the
  dropdown as "(unavailable)" with the reason under the form. A **preview
  platform** (SynSense Speck) has a card and a picture but no engine: choosing
  it draws the chip from the catalog's block list and core table in place of
  the chip view, disables Start and the view buttons, shows the preview note
  under the form, and marks every cost "planned". The Platform tab shows its
  card with the published per-core memory table and the packaged file. The
  Architecture view (the loaded YAML and its differences from a bundled
  baseline) sits under the card.
- **Debugger rail.** Breakpoints (add by kind, enable or disable, remove;
  the one that stopped the run is highlighted), watches, and the placement
  core map with pending edits.
- **Saved runs.** A dock tab with every stored run and its export links. It
  lives in the dock because the list grows with every run.
- **Layout.** The center view sets the height of the main row, so the dock
  starts directly below it. The rail and inspector keep that height and
  scroll inside it. Long names wrap inside their pane, and no view scrolls
  sideways.
- **Neuron actions.** Watch, Break when it fires, and Highlight connections
  (fan-in cores in blue, fan-out cores in green on the chip).
- **Compare.** Two saved runs: spike-train identity, totals, a step-time
  chart, and a per-update table.
- **Reference pill.** match, mismatch, unchecked, or none for the displayed
  update. A mismatch also appears in the phase banner. Reference values
  carry their own mark, `ref`: they come from Lava or SpikingJelly, not from
  SANA-FE.

## Browser checklist

Run through this list in a real browser after changes to the web files.
The headless smoke test (`tests/js/studio_smoke.cjs`) covers the same path
without a display. jsdom does no layout, so it cannot see overflow or
overlap; `SANA-FE-thesis/studio_workloads/smoke_layout.cjs` measures that in
headless Chrome for every view and tab of both QCFS workloads.

1. Start the thesis launcher and open the page. The badge reads "Loihi 2
   candidate · latencies scaled from Loihi 1 by Intel-stated factors · energy
   inherited from Loihi 1 · not hardware".
2. Start `qcfs-compact` with `T = 3`. The used cores are colored, and a
   legend under the chip names each layer and its core count. One dashed
   arrow runs from the host box to a dashed outline around IF0's cores. The
   horizon reads 5.
3. Press Step. Packets move from IF0 to IF1. The phase banner changes to
   "barrier" before the update ends.
4. Switch the clock to "modeled time". Packets now flash briefly, and the
   timeline arrows are nearly vertical.
5. Press Run to horizon. The state reads finished at update 5.
6. Open Live performance. Five stacked bars appear. Open Messages. The
   table lists the current update's messages.
7. Set the horizon override to 3000 and press Rebuild session. Press Run to
   horizon, then Pause. The state reads paused below update 3000; updates
   after the drain show no firing. Press "skip to latest" instead of waiting
   for playback to catch up.
8. Click IF1's core. The inspector shows its neurons, finish time, packets,
   and energy with R and D marks. Click the background to return to the
   chip summary.
9. Drag the update slider back to update 2. The chip and timeline show that
   update. Press "skip to latest" to return.
10. Press Run to horizon on the long session from item 7 and reload the
    page while it runs. The run continues, and the page catches up without
    duplicate or missing updates.
11. Start `qcfs-compact` with `T = 3`, placement `packed`, and 16 neurons
    per core. Eleven cores are colored. Run to horizon. The reference pill
    reads "reference: match" at every update (scrub back to check).
12. Click tile 0. Four core panels show their pipeline counters. Scrub
    within an update with "replay update": the counters rise as packets
    arrive and the soma switches from "updating" to "done".
13. Open core 0.0. Its axon_in names the host's constant current, and its
    axon_out lists IF1's cores. The mini-map highlights tile 0; click it to
    return.
14. Switch to Network. Three groups sit between the two host operations,
    labeled with their synapse counts. Click an IF2 core chip.
15. Click an IF2 neuron and press "Watch this neuron". The Neuron watch tab
    shows the SANA-FE membrane with the Lava and SpikingJelly traces dashed.
16. Open Architecture and choose `loihi` as the baseline. The table lists
    the candidate's differences, including 8192 neurons per core and the
    removed alternative units.
17. Reset, add the breakpoint "update equals 3", and press Run to horizon.
    The state reads "stopped: breakpoint b1: update 3" and the breakpoint is
    highlighted. Remove it and run to the end.
18. Open an IF2 core, click a neuron, and press "Break when it fires". Reset
    and run: the run stops at the neuron's first spike. Press "Highlight
    connections": its IF1 source cores turn blue on the chip.
19. Drag an IF1 core onto an empty core. The Placement section shows the
    pending swap. Press "Rebuild with edits" and run: the reference pill
    stays "match".
20. Open Compare. A is the edited run and B the previous run of the same
    length. The summary reads "spike trains identical"; the hops differ.
21. In the Saved runs tab, open the raster export of a run. The SVG shows the
    spikes, titled as modeled, not measured.
22. Start `qcfs-resnet20` with `T = 2` (the build takes about 30 s). 109
    cores are colored, the chip note says "Aggregate trace level", and the
    session info names the reference source (saved route traces with Lava,
    or computed at build).
23. Run to horizon. Twenty updates play; links thicken where traffic is
    heavy, and the reference pill reads "match" at every update.
24. Scrub back to update 5 and open core 0.0. Its 3,072 neurons show as a
    1,024-bin heatmap of that update's membranes.
25. Rebuild with `T = 8` (a larger, user-entered horizon). The run plays
    26 updates with the reference pill at "match".
26. With the ResNet-20 session still loaded, read the playback hint: at 3 s
    an update it prices the horizon at about a minute. Set playback to
    "instant". The remaining updates draw one after another with no
    animation, and the hint reads "no animation". Set it back to 1.2 s and
    press Replay; packets move again.
27. Run ResNet-20 at 3 s per update and press space after a second. The
    frame and the computed count both hold, and the button reads Resume.
    Press it: both continue to update 20. Press Pause again while it is
    still animating (the server has finished): the frame holds. Press
    Stop: the state no longer offers Resume. Reset, Run, and press Stop
    mid-compute: the state reads "stopped at update N".
28. Start `qcfs-resnet20` (T = 2), run to horizon, open Pipeline. Nineteen
    rows show a diagonal wavefront over 20 columns with hatched gated cells;
    the output panel shows class scores, a prediction, and "match". Switch
    to algorithm timesteps: two aligned columns. Click a cell: the inspector
    names the site and update, and the scrubber moves there.
29. Choose platform "IBM TrueNorth (functional)": `qcfs-compact` and
    `qcfs-resnet20` disappear from the workload list; `random-snn` and
    `sanafe-files` remain. Start `random-snn` with 8 neurons per group and a
    horizon of 3, run to horizon. The badge reads "IBM TrueNorth (functional)
    · no cost model: energy and time are zero · not hardware", the chip shows
    the occupied window of the 64 × 64 mesh, the timeline has bars of zero
    length, and the inspector's energy reads "0.00 pJ" with "R · none"
    beside it.
30. Open Platform. The card lists Structure (4,096 tiles, 256 neurons per
    core), Units (core_soma runs `truenorth`), Costs with every status
    "none", Validation, Not modeled and References; the Architecture file
    sits below. Choose "IBM TrueNorth (documented costs)" and rebuild: the
    step time reads "1.000 ms" and the synapse energy row reads
    "documented". The mini-map in a core view shows the used tiles in blue.
31. Choose "Intel Loihi 1", start `random-snn` with the defaults, open
    Platform: the hop latency rows read "documented" and the soma rows
    "fitted"; the inspector's step time shows "R · fitted". Type a horizon,
    switch to the Loihi 2 candidate: `random-snn` stays selected and the
    typed value stays. Reload the page: the platform and workload dropdowns
    come back as they were.
32. Choose "SynSense Speck (preview)". The center shows the block diagram:
    DVS, pre-processing, the star NoC with nine colored cores labelled with
    their memory, and the readout; the badge reads "SynSense Speck (preview)
    · preview: no engine, no costs · not hardware"; Start and the view
    buttons are disabled and the note under the form says no engine exists.
    Open Platform: the card lists the nine-core memory table, every cost
    "planned", and the packaged `speck.yaml`. Choose another platform: the
    chip view and Start come back.

## Coverage of the Streamlit workbench

Every feature below has a Studio home, and `SANA-FE-thesis/virtual_loihi_ui.py`
was retired at the end of stage 5.

| Streamlit feature | Studio | Stage |
| --- | --- | --- |
| Run a mapped model: T menu, image, named or custom three-core placement | Session rail: typed T and image, whole-layer presets, chunked packed or spread placement, and placement by drag | 3 and 4 |
| Mesh at one update with recorded packet endpoints | Chip view, packets on recorded times along the X route | 2 |
| Animated machine | Chip with both clocks | 2 |
| Performance figure | Live performance | 2 |
| Per-core activity | Timeline rows and the core inspector | 2 |
| Three paths, one aligned computation: per-neuron agreement | Reference pill, per-update check of every neuron, reference traces in neuron watch | 3 |
| Three paths, one aligned computation: the layer spike raster | Raster export from a saved run (SANA-FE; the references agree by the per-update check) | 4 |
| Network graph | Network mode | 3 |
| Membrane of one neuron | Neuron watch | 3 |
| Mapped cores and assumed resources | Core inspector on the candidate | 3 |
| Recorded messages at this update | Messages | 2 |
| Bundled Loihi 1 against the candidate | Platform tab, Architecture view, diff against `loihi` | 3 |
| Larger ResNet-20 mapping (occupancy) | `qcfs-resnet20` at chip scale: 109 used cores, per-core budgets in the inspector | 5 |
| Plots from `sanafe.viz` | Export from a saved run | 4 |
