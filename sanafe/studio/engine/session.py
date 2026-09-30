"""A SANA-FE chip advanced one numbered update at a time."""
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import subprocess
import tempfile
import threading
import uuid

import numpy as np
import sanafe
from sanafe.loihi2 import (architecture_fingerprint, load_loihi2_candidate,
                           validate_core_budgets)

from .breakpoints import Breakpoints
from .connectivity import Connectivity
from .instrument import instrument_arch_yaml
from .layout import ChipLayout
from .records import RecordCache, build_aggregate_record, build_update_record, neuron_map
from .store import TraceStore
from .workload import ParameterSpec, resolve_parameters

TRACE_LEVELS = ('full', 'aggregate')
TRACES = dict(spike_trace=True, potential_trace=True, message_trace=True,
              perf_trace=True)


class SessionState(str, Enum):
    IDLE = 'idle'
    RUNNING = 'running'
    PAUSED = 'paused'
    STOPPED = 'stopped'
    FAULTED = 'faulted'
    FINISHED = 'finished'


class SessionFault(RuntimeError):
    """SANA-FE raised during an update; the session must be reset."""


def _revision():
    root = Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, check=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


class Session:
    def __init__(self, workload, parameters=None, *, trace_level=None,
                 horizon=None, store_dir=None, core_map=None):
        if trace_level is None:
            trace_level = getattr(workload, 'default_trace_level', 'full')
        if trace_level not in TRACE_LEVELS:
            raise ValueError(f"trace level {trace_level!r} is not available; "
                             f"choose one of {', '.join(TRACE_LEVELS)}")
        if horizon is not None:
            ParameterSpec('horizon', 'int', minimum=1).validate(horizon)
        self.workload = workload
        self.parameters = resolve_parameters(workload, parameters or {})
        self.trace_level = trace_level
        self._horizon_override = horizon
        if core_map is not None and not isinstance(core_map, dict):
            raise ValueError('core_map: expected an object mapping core to core')
        self.core_map = dict(core_map or {})
        self._store_dir = Path(store_dir) if store_dir is not None else None
        self._pause = threading.Event()
        self._scratch = None
        self.breakpoints = Breakpoints()
        self.watched = []
        try:
            self._build()
        except BaseException:
            self.close()
            raise

    def _build(self):
        self._pause.clear()
        if self._scratch is not None:
            self._scratch.cleanup()
        self._scratch = tempfile.TemporaryDirectory(prefix='sanafe-studio-')
        self.built = self.workload.build(dict(self.parameters))
        self.horizon = self._horizon_override or self.built.horizon
        # The text simulated, kept in case the file changes on disk later.
        self.architecture_text = Path(self.built.arch_yaml).read_text()
        instrumented = Path(self._scratch.name) / 'architecture.yaml'
        self.instrumentation = instrument_arch_yaml(self.built.arch_yaml, instrumented)
        self._sim_arch = sanafe.load_arch(str(instrumented))
        if self.core_map:
            self._apply_core_map()
        self.chip = sanafe.SpikingChip(self._sim_arch)
        self.chip.load(self.built.network)
        self.layout = ChipLayout.from_chip(self.chip, self._sim_arch)
        self.neurons = neuron_map(self.chip)
        self.connectivity = Connectivity.from_network(self.built.network, self.neurons)
        self._candidate = (architecture_fingerprint(self.built.arch) ==
                           architecture_fingerprint(load_loihi2_candidate()))
        self.core_budgets = None
        if self._candidate:
            # The budgets are Loihi 2 figures, so only the candidate is held to them.
            self.core_budgets = {f'{stat["tile"]}.{stat["core"]}': stat for stat in
                                 validate_core_budgets(self.connectivity.core_stats())}
        self.records = []
        self._cache = self.lookup = RecordCache(self.neurons, self.layout)
        bind = getattr(self.built.reference, 'bind', None)
        if bind is not None:  # a checker may need the trace order of this chip
            bind(self)
        bind = getattr(self.built.readout, 'bind', None)
        if bind is not None:
            bind(self)
        self._history = []  # aggregate level: (membranes, fired indices) per update
        self.watched = [key for key in self.watched if key in self._cache.logged_pos]
        # Keep the breakpoints that still apply after a rebuild; say which did not.
        self.breakpoints, self.breakpoint_warnings = Breakpoints.compile_valid(
            self.breakpoints.specs, self)
        self.state = SessionState.IDLE
        self.stop_reason = None
        self.fault = None
        self.store = None
        if self._store_dir is not None:
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            name = f'{self.workload.name}-{stamp}-{uuid.uuid4().hex[:6]}'
            self.store = TraceStore.create(self._store_dir / name, self.manifest())

    def _apply_core_map(self):
        """Move every neuron of each key core to its value core.

        map_to_core also sets SANA-FE's mapping order, which decides placement
        order inside a core, so every neuron is re-mapped in its old order.
        """
        arch = self.built.arch

        def core(key):
            try:
                tile, offset = (int(part) for part in key.split('.'))
                arch.tiles[tile].cores[offset]
                if tile < 0 or offset < 0:
                    raise IndexError
            except (AttributeError, ValueError, IndexError, TypeError):
                raise ValueError(f'core_map: unknown core {key!r}') from None
            return tile, offset

        moves = {core(key): core(value) for key, value in self.core_map.items()}
        neurons = [neuron for group in self.built.network.groups.values()
                   for neuron in group if neuron.mapped_core is not None]
        for neuron in sorted(neurons, key=lambda n: n.mapping_order):
            tile, offset = moves.get(tuple(neuron.mapped_core), tuple(neuron.mapped_core))
            neuron.map_to_core(arch.tiles[tile].cores[offset])

    @property
    def update(self):
        return len(self.records)

    def manifest(self):
        return {
            'workload': self.workload.name,
            'parameters': _jsonable(self.parameters),
            'architecture_yaml': str(self.built.arch_yaml),
            'architecture_sha256': architecture_fingerprint(self.built.arch),
            'simulated_architecture_sha256': architecture_fingerprint(self._sim_arch),
            'instrumentation': self.instrumentation,
            'sanafe_revision': _revision(),
            'timing_model': 'detailed',
            'trace_level': self.trace_level,
            'horizon': self.horizon,
            'core_map': dict(self.core_map),
            'network': [{'name': g['name'], 'size': g['size']}
                        for g in self._group_sizes()],
            'metadata': _jsonable(self.built.metadata),
            'created': datetime.now(timezone.utc).isoformat(),
        }

    def _group_sizes(self):
        sizes = {}
        for neuron in self.neurons:
            sizes[neuron.group] = sizes.get(neuron.group, 0) + 1
        return [{'name': name, 'size': size} for name, size in sizes.items()]

    def network_summary(self):
        """Groups in SANA-FE trace order with their per-core neuron counts."""
        groups = {}
        for neuron in self.neurons:
            entry = groups.setdefault(neuron.group,
                                      {'name': neuron.group, 'size': 0, 'cores': {}})
            entry['size'] += 1
            entry['cores'][neuron.core] = entry['cores'].get(neuron.core, 0) + 1
        connectivity = self.connectivity
        return {'groups': list(groups.values()),
                'occupied': sorted({neuron.core for neuron in self.neurons}),
                'connections': connectivity.group_edges,
                'core_links': connectivity.core_links,
                'core_neurons': connectivity.core_neurons,
                'group_attributes': connectivity.group_attributes,
                'core_budgets': self.core_budgets}

    def watch(self, keys):
        """Choose the neurons whose membranes and spikes every record carries."""
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            raise ValueError('watches: expected a list of "group.offset" names')
        for key in keys:
            group, _, offset = key.rpartition('.')
            if not offset.isdigit() or (group, int(offset)) not in self._cache.index:
                raise ValueError(f'watch: no neuron {key!r}')
            if key not in self._cache.logged_pos:
                raise ValueError(f'watch: {key} does not log its potential, so its '
                                 'membrane is not recorded')
        self.watched = list(dict.fromkeys(keys))
        return self.watched

    def _membranes(self, update):
        """(membrane vector in logged order, fired neuron indices) after an update."""
        if not isinstance(update, int) or not 1 <= update <= self.update:
            raise ValueError(f'update must be between 1 and {self.update}, got {update!r}')
        if self.trace_level == 'aggregate':
            return self._history[update - 1]
        record = self.records[update - 1]
        values = np.array([record.potentials[self._cache.keys[i]] for i in self._cache.logged])
        fired = np.array([self._cache.index[tuple(a)] for a in record.fired], dtype=np.int32)
        return values, fired

    def core_state(self, core, update):
        """Membranes and firing of one core's neurons after one update."""
        if core not in self.connectivity.core_neurons:
            raise KeyError(f'no neurons on core {core!r}')
        values, fired = self._membranes(update)
        fired = set(fired.tolist())
        keys = [f'{group}.{offset}' for group, first, last in
                self.connectivity.core_neurons[core] for offset in range(first, last + 1)]
        position = self._cache.logged_pos
        return {'update': update, 'core': core, 'neurons': keys,
                'potentials': [float(values[position[k]]) if k in position else None
                               for k in keys],
                'fired': sorted(k for k in keys
                                if self._cache.index[(k.rpartition('.')[0],
                                                      int(k.rpartition('.')[2]))] in fired)}

    def neuron_detail(self, group, offset):
        """One neuron's placement, attributes, connections, history, and references."""
        detail = self.connectivity.neuron(group, offset)
        key = f'{group}.{int(offset)}'
        index = self._cache.index[(group, int(offset))]
        position = self._cache.logged_pos.get(key)
        potential, fired = [], []
        for update in range(1, self.update + 1):
            values, spikes = self._membranes(update)
            potential.append(float(values[position]) if position is not None else None)
            fired.append(bool(index in spikes))
        detail['history'] = {'potential': potential, 'fired': fired}
        series = getattr(self.built.reference, 'series', None)
        if series is not None:
            detail['reference'] = series(f'{group}.{int(offset)}')
        return detail

    def badge(self):
        """The provenance line every view must show for this architecture."""
        if self._candidate:
            return 'Loihi 2 candidate · costs inherited from Loihi 1 · not hardware'
        return (f'{Path(self.built.arch_yaml).name} · modeled costs from this file · '
                'not measurements')

    def describe(self):
        """Summary a viewer needs before the first update."""
        return {'layout': self.layout.to_dict(), 'network': self.network_summary(),
                'horizon': self.horizon, 'badge': self.badge(),
                'manifest': self.manifest(), 'metadata': _jsonable(self.built.metadata),
                'state': self.state.value, 'update': self.update,
                'core_map': dict(self.core_map),
                'breakpoints': list(self.breakpoints.specs),
                'breakpoint_warnings': list(self.breakpoint_warnings),
                'trace_level': self.trace_level, 'watched': list(self.watched),
                'run': self.store.directory.name if self.store is not None else None}

    def carry_breakpoints(self, specs):
        """Adopt another session's breakpoints; return the reasons for any dropped."""
        self.breakpoints, self.breakpoint_warnings = Breakpoints.compile_valid(specs, self)
        return self.breakpoint_warnings

    def set_breakpoints(self, specs):
        """Replace the breakpoint list; takes effect from the next update.

        Thread-safe: the new list is validated first and then swapped in with
        one assignment, so a running update sees either the old or new list.
        """
        self.breakpoints = Breakpoints.compile(specs, self)
        self.breakpoint_warnings = []
        return self.breakpoints.specs

    def pause(self):
        """Stop the current or next run at an update boundary. Thread-safe.

        A pause requested before a run starts stops that run before its first
        update. The run that honors it consumes it; a run that ends, stops, or
        faults discards it.
        """
        self._pause.set()

    def cancel_pause(self):
        """Discard a pending pause that no run has honored."""
        self._pause.clear()

    def step(self, n=1, stop_when=None):
        ParameterSpec('n', 'int', minimum=1).validate(n)
        return self._advance(n, stop_when)

    def run_to_horizon(self, stop_when=None):
        return self._advance(max(self.horizon - self.update, 0), stop_when)

    def _advance(self, count, stop_when):
        if self.state is SessionState.FAULTED:
            raise SessionFault(f'reset() is required after a fault: {self.fault}')
        self.stop_reason = None
        self.state = SessionState.RUNNING
        produced = []
        for _ in range(count):
            if self._pause.is_set():
                self._pause.clear()
                self.state = SessionState.PAUSED
                return produced
            # Once chip.sim returns, the chip has advanced. Any later failure
            # must fault the session, or update numbers would shift silently.
            try:
                result = self._simulate()
                if self.trace_level == 'aggregate':
                    record = build_aggregate_record(self.update + 1, result, self.layout,
                                                    self._cache, list(self.watched))
                else:
                    record = build_update_record(self.update + 1, result, self.layout,
                                                 self.neurons, self._cache)
                if self.built.reference is not None:
                    record.reference = self.built.reference.check(record)
                if self.built.readout is not None:
                    record.readout = self.built.readout.decode(record)
                if self.store is not None:
                    self.store.append(record)
            except Exception as error:
                self._set_fault(error)
            if self.trace_level == 'aggregate':
                state = record.state  # the full arrays stay here, never serialized
                self._history.append((state['potentials'],
                                      np.flatnonzero(state['fired']).astype(np.int32)))
            self.records.append(record)
            produced.append(record)
            try:
                reason = stop_when(record) if stop_when is not None else None
            except Exception as error:
                self._set_fault(error)
            reason = reason or self.breakpoints.check(record)  # may read record.state
            record.__dict__.pop('state', None)
            if reason:
                self._pause.clear()
                self.state = SessionState.STOPPED
                self.stop_reason = reason
                return produced
        self._pause.clear()
        self.state = (SessionState.FINISHED if self.update >= self.horizon
                      else SessionState.PAUSED)
        return produced

    def _set_fault(self, error):
        self._pause.clear()
        self.state = SessionState.FAULTED
        self.fault = f'{type(error).__name__}: {error}'
        raise SessionFault(self.fault) from error

    def _simulate(self):
        """One numbered update with every trace; the single call into SANA-FE."""
        return self.chip.sim(1, **TRACES)

    def reset(self):
        """Rebuild the chip from the workload. Required after a fault."""
        self._build()

    def close(self):
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None
