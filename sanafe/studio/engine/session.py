"""A SANA-FE chip advanced one numbered update at a time."""
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import subprocess
import tempfile
import threading
import uuid

import sanafe
from sanafe.loihi2 import (architecture_fingerprint, load_loihi2_candidate,
                           validate_core_budgets)

from .connectivity import Connectivity
from .instrument import instrument_arch_yaml
from .layout import ChipLayout
from .records import build_update_record, neuron_map
from .store import TraceStore
from .workload import ParameterSpec, resolve_parameters

TRACE_LEVELS = ('full',)
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
    def __init__(self, workload, parameters=None, *, trace_level='full',
                 horizon=None, store_dir=None):
        if trace_level not in TRACE_LEVELS:
            raise ValueError(f"trace level {trace_level!r} is not available; "
                             "'aggregate' arrives in stage 5, use 'full'")
        if horizon is not None:
            ParameterSpec('horizon', 'int', minimum=1).validate(horizon)
        self.workload = workload
        self.parameters = resolve_parameters(workload, parameters or {})
        self.trace_level = trace_level
        self._horizon_override = horizon
        self._store_dir = Path(store_dir) if store_dir is not None else None
        self._pause = threading.Event()
        self._scratch = None
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
        instrumented = Path(self._scratch.name) / 'architecture.yaml'
        self.instrumentation = instrument_arch_yaml(self.built.arch_yaml, instrumented)
        self._sim_arch = sanafe.load_arch(str(instrumented))
        self.chip = sanafe.SpikingChip(self._sim_arch)
        self.chip.load(self.built.network)
        self.layout = ChipLayout.from_chip(self.chip, self._sim_arch)
        self.neurons = neuron_map(self.chip)
        self.connectivity = Connectivity.from_network(self.built.network, self.neurons)
        self._candidate = (architecture_fingerprint(self.built.arch) ==
                           architecture_fingerprint(load_loihi2_candidate()))
        if self._candidate:
            # The budgets are Loihi 2 figures, so only the candidate is held to them.
            validate_core_budgets(self.connectivity.core_stats())
        self.records = []
        self.state = SessionState.IDLE
        self.stop_reason = None
        self.fault = None
        self.store = None
        if self._store_dir is not None:
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            name = f'{self.workload.name}-{stamp}-{uuid.uuid4().hex[:6]}'
            self.store = TraceStore.create(self._store_dir / name, self.manifest())

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
            'metadata': _jsonable(self.built.metadata),
            'created': datetime.now(timezone.utc).isoformat(),
        }

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
                'group_attributes': connectivity.group_attributes}

    def neuron_detail(self, group, offset):
        """One neuron's placement, attributes, connections, and reference traces."""
        detail = self.connectivity.neuron(group, offset)
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
                'state': self.state.value, 'update': self.update}

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
                record = build_update_record(self.update + 1, result, self.layout,
                                             self.neurons)
                if self.built.reference is not None:
                    record.reference = self.built.reference.check(record)
                if self.store is not None:
                    self.store.append(record)
            except Exception as error:
                self._set_fault(error)
            self.records.append(record)
            produced.append(record)
            try:
                reason = stop_when(record) if stop_when is not None else None
            except Exception as error:
                self._set_fault(error)
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
