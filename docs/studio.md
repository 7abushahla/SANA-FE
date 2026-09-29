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
