"""One Studio session per process, driven by plain-dict messages over a Pipe.

Commands from the parent: {'op': 'step', 'n': k}, {'op': 'run'},
{'op': 'pause'}, {'op': 'reset'}, {'op': 'close'}, {'op': 'query', 'id',
'what', ...}, and {'op': 'breakpoints', 'id', 'specs'}. Events from the
worker: 'ready', 'error', 'update', 'state', and 'reply'. Queries and
breakpoint changes are answered at once, even during a run: queries read
only what the build produced, and a new breakpoint list is swapped in
whole, taking effect from the next update. The parent's reader thread adds
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


def _answer(session, message):
    """Reply to one query from build-time data; never touches the chip."""
    reply = {'type': 'reply', 'id': message.get('id')}
    try:
        if message.get('op') == 'breakpoints':
            reply['data'] = session.set_breakpoints(message.get('specs'))
        elif message.get('what') == 'neuron':
            reply['data'] = session.neuron_detail(message['group'], int(message['offset']))
        else:
            raise ValueError(f'unknown query {message.get("what")!r}')
    except (KeyError, IndexError) as error:
        reply.update(code='not_found', error=str(error).strip("'\""))
    except (TypeError, ValueError) as error:
        reply.update(code='bad_request', error=str(error))
    except Exception as error:  # a query must never end the reader thread
        reply.update(code='failed', error=f'{type(error).__name__}: {error}')
    return reply


def run_worker(conn, ref, parameters, options):
    """Process entry point: build the session, then serve commands."""
    from ..engine import Session, SessionFault

    # A spawned child inherits a fixed start method. Libraries such as Lava
    # call set_start_method at import, which then raises. Clear it, as in a
    # fresh interpreter; the Studio itself always uses an explicit context.
    multiprocessing.set_start_method(None, force=True)
    send_lock = threading.Lock()

    def send(message):  # the main loop and the reader thread both send
        with send_lock:
            conn.send(message)

    try:
        session = Session(ref.load(), parameters,
                          trace_level=options.get('trace_level', 'full'),
                          horizon=options.get('horizon'),
                          store_dir=options.get('store_dir'),
                          core_map=options.get('core_map'))
    except Exception as error:
        send({'type': 'error', 'message': f'{type(error).__name__}: {error}'})
        conn.close()
        return
    send({'type': 'ready', **session.describe(),
          'architecture_text': session.architecture_text})

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
            if op in ('query', 'breakpoints'):
                send(_answer(session, message))
                continue
            with control:
                if op == 'pause':
                    # Only a queued or running command can be paused; a pause
                    # while idle would otherwise stop the next run at once.
                    if pending[0]:
                        session.pause()
                    continue
                if op in ('step', 'run'):
                    pending[0] += 1
                if op == 'close' and pending[0]:
                    session.pause()  # stop a running command at the next boundary
            commands.put(message)
            if op == 'close':
                return

    threading.Thread(target=read, daemon=True).start()

    def stream(record):
        send({'type': 'update', 'record': record.to_dict()})

    while True:
        command = commands.get()
        op = command.get('op')
        if op == 'close':
            break
        if op == 'reset':
            try:
                session.reset()
            except Exception as error:
                send({'type': 'error', 'message': f'{type(error).__name__}: {error}'})
                continue
            send({'type': 'ready', **session.describe(),
                  'architecture_text': session.architecture_text})
            continue
        if op not in ('step', 'run'):
            send({'type': 'error', 'message': f'unknown command {op!r}'})
            continue
        send(_state(session, 'running'))
        try:
            if op == 'step':
                session.step(command.get('n', 1), stop_when=stream)
            else:
                session.run_to_horizon(stop_when=stream)
        except SessionFault:
            pass  # the state event below carries the fault
        except Exception as error:
            send({'type': 'error', 'message': f'{type(error).__name__}: {error}'})
        with control:
            pending[0] -= 1
            if not pending[0]:
                session.cancel_pause()
        send(_state(session))
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
