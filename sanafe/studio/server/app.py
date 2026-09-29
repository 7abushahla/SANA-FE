"""Local HTTP commands and WebSocket events for Studio sessions.

Commands are plain HTTP and return at once. Results stream over one
WebSocket per session. Each session's worker is a separate process, so a
native crash faults one session and leaves the server running.
"""
import asyncio
import contextlib
import json
import threading
import uuid
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from ..engine import ParameterSpec, resolve_parameters, to_strict_json
from .worker import WorkerHandle

WEB = Path(__file__).resolve().parents[1] / 'web'


def _json(data, status=200):
    return JSONResponse(to_strict_json(data), status_code=status)


def _dumps(message):
    return json.dumps(to_strict_json(message), allow_nan=False)


def _spec(spec):
    return {'name': spec.name, 'kind': spec.kind, 'default': spec.default,
            'minimum': spec.minimum, 'maximum': spec.maximum,
            'choices': list(spec.choices), 'help': spec.help}


async def _body(request):
    try:
        data = await request.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


class ManagedSession:
    """Server-side view of one worker: its records, state, and subscribers."""

    def __init__(self, manager, session_id, workload, ref, parameters, options):
        self.manager = manager
        self.id = session_id
        self.workload = workload
        self.parameters = parameters
        self.records = []
        self.ready = None
        self.error = None
        self.alive = True
        self.closing = False
        self.state = {'type': 'state', 'state': 'starting', 'update': 0, 'alive': True}
        self.subscribers = set()
        self.lock = threading.Lock()
        self.ready_event = threading.Event()
        self.handle = WorkerHandle(ref, parameters, options, self._on_message)

    def _snapshot_locked(self):
        ready = {key: value for key, value in (self.ready or {}).items()
                 if key not in ('type', 'state', 'update')}
        return {'id': self.id, 'workload': self.workload, 'parameters': self.parameters,
                **ready, 'state': self.state, 'update': len(self.records)}

    def snapshot(self):
        with self.lock:
            return self._snapshot_locked()

    def _on_message(self, message):  # worker reader thread
        kind = message.get('type')
        with self.lock:
            if kind == 'ready':
                self.ready = message
                self.records = []
                self.state = {'type': 'state', 'state': message['state'], 'update': 0,
                              'horizon': message['horizon'], 'alive': True}
                self.ready_event.set()
                message = {'type': 'ready', **self._snapshot_locked()}
            elif kind == 'error' and self.ready is None:
                self.error = message['message']
                self.ready_event.set()
            elif kind == 'update':
                self.records.append(message['record'])
            elif kind == 'state':
                self.state = {**message, 'alive': True}
                message = self.state
            elif kind == 'exited':
                self.alive = False
                if self.closing:
                    return
                if self.ready is None and self.error is None:
                    self.error = (f'worker exited with code {message["exitcode"]} '
                                  'before the session was ready')
                self.ready_event.set()
                self.state = {'type': 'state', 'state': 'faulted',
                              'update': len(self.records),
                              'horizon': (self.ready or {}).get('horizon'),
                              'fault': f'worker exited with code {message["exitcode"]}',
                              'reason': None, 'alive': False}
                message = self.state
        self.manager.broadcast(self, message)


class SessionManager:
    def __init__(self, registry, store_dir=None, build_timeout=300.0):
        self.registry = dict(registry)
        self.store_dir = store_dir
        self.build_timeout = build_timeout
        self.sessions = {}
        self.loop = None
        self._workloads = {}

    def workload(self, name):
        if name not in self._workloads:
            self._workloads[name] = self.registry[name].load()
        return self._workloads[name]

    def broadcast(self, session, message):
        if self.loop is None:
            return
        for subscriber in list(session.subscribers):
            self.loop.call_soon_threadsafe(subscriber.put_nowait, message)

    def close(self, session_id):
        session = self.sessions.pop(session_id, None)
        if session is not None:
            session.closing = True
            session.handle.close()

    def close_all(self):
        for session_id in list(self.sessions):
            self.close(session_id)


def create_app(registry, store_dir=None, build_timeout=300.0):
    manager = SessionManager(registry, store_dir, build_timeout)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        manager.loop = asyncio.get_running_loop()
        yield
        manager.close_all()

    def lookup(request):
        return manager.sessions.get(request.path_params['session_id'])

    async def list_workloads(request):
        return _json([{'name': name,
                       'parameters': [_spec(s) for s in manager.workload(name).parameters()]}
                      for name in manager.registry])

    async def create_session(request):
        body = await _body(request)
        name = body.get('workload')
        if name not in manager.registry:
            return _json({'error': f'unknown workload {name!r}'}, 404)
        parameters = body.get('parameters') or {}
        horizon = body.get('horizon')
        trace_level = body.get('trace_level', 'full')
        try:
            if not isinstance(parameters, dict):
                raise ValueError('parameters: expected an object')
            resolve_parameters(manager.workload(name), parameters)
            if horizon is not None:
                ParameterSpec('horizon', 'int', minimum=1).validate(horizon)
            if trace_level != 'full':
                raise ValueError(f"trace level {trace_level!r} is not available; "
                                 "'aggregate' arrives in stage 5, use 'full'")
        except ValueError as error:
            return _json({'error': str(error)}, 400)
        session_id = uuid.uuid4().hex[:12]
        options = {'trace_level': trace_level, 'horizon': horizon,
                   'store_dir': str(manager.store_dir) if manager.store_dir else None}
        session = ManagedSession(manager, session_id, name, manager.registry[name],
                                 parameters, options)
        manager.sessions[session_id] = session
        ready = await asyncio.to_thread(session.ready_event.wait, manager.build_timeout)
        if session.error or not ready:
            manager.close(session_id)
            if session.error:
                return _json({'error': session.error}, 400)
            return _json({'error': f'session build exceeded {manager.build_timeout:.0f} s'}, 504)
        return _json(session.snapshot(), 201)

    async def get_session(request):
        session = lookup(request)
        if session is None:
            return _json({'error': 'no such session'}, 404)
        return _json(session.snapshot())

    async def delete_session(request):
        if lookup(request) is None:
            return _json({'error': 'no such session'}, 404)
        manager.close(request.path_params['session_id'])
        return Response(status_code=204)

    async def command(request, op, **extra):
        session = lookup(request)
        if session is None:
            return _json({'error': 'no such session'}, 404)
        if not session.alive:
            return _json({'error': f'session worker has exited '
                                   f'({session.state.get("fault")}); start a new session'}, 409)
        if op != 'pause' and session.state.get('state') == 'running':
            return _json({'error': 'session is running; pause it first'}, 409)
        session.handle.send({'op': op, **extra})
        return _json({'accepted': op}, 202)

    async def step(request):
        n = (await _body(request)).get('n', 1)
        try:
            ParameterSpec('n', 'int', minimum=1).validate(n)
        except ValueError as error:
            return _json({'error': str(error)}, 400)
        return await command(request, 'step', n=n)

    async def run(request):
        return await command(request, 'run')

    async def pause(request):
        return await command(request, 'pause')

    async def reset(request):
        return await command(request, 'reset')

    async def updates(request):
        session = lookup(request)
        if session is None:
            return _json({'error': 'no such session'}, 404)
        try:
            start = int(request.query_params.get('from', '0'))
        except ValueError:
            start = -1
        if start < 0:
            return _json({'error': 'from: expected a non-negative integer'}, 400)
        with session.lock:
            records = session.records[start:]
        return _json({'from': start, 'updates': records})

    async def events(websocket):
        await websocket.accept()
        session = manager.sessions.get(websocket.path_params['session_id'])
        if session is None:
            await websocket.send_text(_dumps({'type': 'error', 'message': 'no such session'}))
            await websocket.close()
            return
        subscriber = asyncio.Queue()
        session.subscribers.add(subscriber)

        async def until_disconnect():
            # The client sends nothing, so receiving only ever returns the
            # disconnect. Without it, a closed socket stays open to uvicorn
            # and blocks graceful shutdown.
            while (await websocket.receive())['type'] != 'websocket.disconnect':
                pass

        watcher = asyncio.create_task(until_disconnect())
        try:
            with session.lock:
                hello = {'type': 'hello', 'update': len(session.records),
                         'state': session.state}
            await websocket.send_text(_dumps(hello))
            while True:
                getter = asyncio.create_task(subscriber.get())
                done, _ = await asyncio.wait({getter, watcher},
                                             return_when=asyncio.FIRST_COMPLETED)
                if watcher in done:
                    getter.cancel()
                    break
                await websocket.send_text(_dumps(getter.result()))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            watcher.cancel()
            session.subscribers.discard(subscriber)

    routes = [
        Route('/api/workloads', list_workloads),
        Route('/api/sessions', create_session, methods=['POST']),
        Route('/api/sessions/{session_id}', get_session, methods=['GET']),
        Route('/api/sessions/{session_id}', delete_session, methods=['DELETE']),
        Route('/api/sessions/{session_id}/step', step, methods=['POST']),
        Route('/api/sessions/{session_id}/run', run, methods=['POST']),
        Route('/api/sessions/{session_id}/pause', pause, methods=['POST']),
        Route('/api/sessions/{session_id}/reset', reset, methods=['POST']),
        Route('/api/sessions/{session_id}/updates', updates),
        WebSocketRoute('/ws/sessions/{session_id}', events),
        Mount('/', StaticFiles(directory=WEB, html=True, check_dir=False)),
    ]
    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.manager = manager
    return app
