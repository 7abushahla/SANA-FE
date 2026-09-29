# SANA-FE Studio engine

The Studio engine advances a SANA-FE network one numbered update at a time and
turns each update into an `UpdateRecord`. The server and browser views arrive
in stage 2. The engine is usable on its own from a script or notebook.

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

## Limits in this stage

Only the `full` trace level exists. Breakpoints are Python callables passed as
`stop_when`; declarative breakpoints arrive with the server. A fault inside
SANA-FE moves the session to `faulted`, and only `reset()` clears it.

## Running the Studio

```bash
.venv/bin/python -m sanafe.studio --port 8765
```

Open `http://127.0.0.1:8765/`. The server listens on this machine only. Add
workloads with `--workload NAME=module:Class` and `--path DIR`. Add
`--store-dir DIR` to save every run as a trace store. The thesis launcher
registers the QCFS compact network:

```bash
SANA-FE/.venv/bin/python SANA-FE-thesis/studio_workloads/launch.py
```

Each session runs in its own worker process. A crash in SANA-FE faults that
session only; the page says so and offers a new session.

## What the page shows

- **Transport bar.** Step, Run n updates, Run to horizon, Pause, and Reset.
  A pause stops the run at the next update boundary. A pause sent while
  nothing runs is ignored.
- **Session rail.** The form comes from the workload's parameter schema.
  `T` and the image index are typed numbers. Changing them and pressing
  Rebuild starts a fresh session.
- **Chip.** Tiles hold a router and their cores. Cores take the color of
  the group with most neurons on them. Packets move on their recorded send
  and receive times along the reconstructed x-then-y route. Yellow dots mark
  cores still processing neurons. Purple outlines mark cores processing a
  message.
- **Clock.** "Modeled time" keeps true proportions, so packets appear only
  briefly. "Slow motion around packets" slows the clock only while packets
  fly. Both keep the recorded order.
- **Timeline.** One row per occupied core. Bars show neuron processing,
  purple marks show message processing, lines join send to receive, and the
  gray band is the barrier.
- **Live performance and messages.** Energy per update by unit, step time,
  message counts, and the current update's recorded messages.
- **Inspector.** Numbers for the chip or a clicked core, each with its R, D,
  or X mark.

## Browser checklist

Run through this list in a real browser after changes to the web files.
The headless smoke test (`tests/js/studio_smoke.cjs`) covers the same path
without a display.

1. Start the thesis launcher and open the page. The badge reads "Loihi 2
   candidate · costs inherited from Loihi 1 · not hardware".
2. Start `qcfs-compact` with `T = 3`. Three cores are colored, and the host
   box shows a dashed line into IF0's core. The horizon reads 5.
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
