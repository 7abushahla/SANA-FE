"""A SANA-FE chip advanced one numbered update at a time."""
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import subprocess
import tempfile
import threading
import uuid

import sanafe
from sanafe.loihi2 import architecture_fingerprint

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
        self._build()

    def _build(self):
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
        self.layout = ChipLayout.from_chip(self.chip)
        self.neurons = neuron_map(self.chip)
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

    def pause(self):
        """Stop at the next update boundary. Safe to call from another thread."""
        self._pause.set()

    def step(self, n=1):
        ParameterSpec('n', 'int', minimum=1).validate(n)
        return self._advance(n, None)

    def run_to_horizon(self, stop_when=None):
        return self._advance(max(self.horizon - self.update, 0), stop_when)

    def _advance(self, count, stop_when):
        if self.state is SessionState.FAULTED:
            raise SessionFault(f'reset() is required after a fault: {self.fault}')
        self._pause.clear()
        self.stop_reason = None
        self.state = SessionState.RUNNING
        produced = []
        for _ in range(count):
            if self._pause.is_set():
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
                self.state = SessionState.STOPPED
                self.stop_reason = reason
                return produced
        if self._pause.is_set() or self.update < self.horizon:
            self.state = SessionState.PAUSED
        else:
            self.state = SessionState.FINISHED
        return produced

    def _set_fault(self, error):
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
