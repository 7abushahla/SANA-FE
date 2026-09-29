"""One Studio session per process, driven by plain-dict messages over a Pipe.

Commands from the parent: {'op': 'step', 'n': k}, {'op': 'run'},
{'op': 'pause'}, {'op': 'reset'}, {'op': 'close'}. Events from the worker:
'ready', 'error', 'update', and 'state'. The parent's reader thread adds
'exited' when the process ends, whatever the reason.
"""
from dataclasses import dataclass
import importlib
import multiprocessing
import queue
import sys
import threading


@dataclass(frozen=True)
class WorkloadRef:
    """A workload class named as ``module:Class``, plus import paths it needs."""

    target: str
    sys_path: tuple = ()

    def load(self):
        module_name, _, class_name = self.target.partition(':')
        if not module_name or not class_name:
            raise ValueError(f'workload target must be module:Class, got {self.target!r}')
        for entry in reversed(self.sys_path):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        return getattr(importlib.import_module(module_name), class_name)()


def _state(session, state=None):
    return {'type': 'state', 'state': state or session.state.value,
            'update': session.update, 'horizon': session.horizon,
            'reason': session.stop_reason, 'fault': session.fault}


def run_worker(conn, ref, parameters, options):
    """Process entry point: build the session, then serve commands."""
    from ..engine import Session, SessionFault

    try:
        session = Session(ref.load(), parameters,
                          trace_level=options.get('trace_level', 'full'),
                          horizon=options.get('horizon'),
                          store_dir=options.get('store_dir'))
    except Exception as error:
        conn.send({'type': 'error', 'message': f'{type(error).__name__}: {error}'})
        conn.close()
        return
    conn.send({'type': 'ready', **session.describe()})

    commands = queue.Queue()
    control = threading.Lock()
    pending = [0]  # step and run commands queued or executing

    def read():
        while True:
            try:
                message = conn.recv()
            except (EOFError, OSError):
                commands.put({'op': 'close'})
                return
            op = message.get('op')
            with control:
                if op == 'pause':
                    # Only a queued or running command can be paused; a pause
                    # while idle would otherwise stop the next run at once.
                    if pending[0]:
                        session.pause()
                    continue
                if op in ('step', 'run'):
                    pending[0] += 1
            commands.put(message)
            if op == 'close':
                return

    threading.Thread(target=read, daemon=True).start()

    def stream(record):
        conn.send({'type': 'update', 'record': record.to_dict()})

    while True:
        command = commands.get()
        op = command.get('op')
        if op == 'close':
            break
        if op == 'reset':
            try:
                session.reset()
            except Exception as error:
                conn.send({'type': 'error', 'message': f'{type(error).__name__}: {error}'})
                continue
            conn.send({'type': 'ready', **session.describe()})
            continue
        if op not in ('step', 'run'):
            conn.send({'type': 'error', 'message': f'unknown command {op!r}'})
            continue
        conn.send(_state(session, 'running'))
        try:
            if op == 'step':
                session.step(command.get('n', 1), stop_when=stream)
            else:
                session.run_to_horizon(stop_when=stream)
        except SessionFault:
            pass  # the state event below carries the fault
        except Exception as error:
            conn.send({'type': 'error', 'message': f'{type(error).__name__}: {error}'})
        with control:
            pending[0] -= 1
            if not pending[0]:
                session.cancel_pause()
        conn.send(_state(session))
    session.close()
    conn.close()


class WorkerHandle:
    """Parent side of one worker process."""

    def __init__(self, ref, parameters, options, on_message):
        context = multiprocessing.get_context('spawn')
        self._conn, child = context.Pipe()
        self.process = context.Process(target=run_worker,
                                       args=(child, ref, parameters, options),
                                       daemon=True)
        self.process.start()
        child.close()  # so the parent sees EOF when the worker dies
        self._on_message = on_message
        self._send_lock = threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while True:
            try:
                message = self._conn.recv()
            except (EOFError, OSError):
                self.process.join(timeout=5)
                self._on_message({'type': 'exited', 'exitcode': self.process.exitcode})
                return
            self._on_message(message)

    def send(self, message):
        with self._send_lock:
            self._conn.send(message)

    def close(self, timeout=5.0):
        try:
            self.send({'op': 'close'})
        except (OSError, EOFError):
            pass
        self.process.join(timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout)
