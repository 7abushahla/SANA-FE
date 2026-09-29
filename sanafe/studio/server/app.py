"""Local HTTP commands and WebSocket events for Studio sessions.

Commands are plain HTTP and return at once. Results stream over one
WebSocket per session. Each session's worker is a separate process, so a
native crash faults one session and leaves the server running.
"""
import asyncio
import concurrent.futures
import contextlib
import itertools
import json
import threading
import uuid
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from ..engine import (ParameterSpec, bundled_architectures, diff_texts,
                      resolve_parameters, to_strict_json)
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
    """The JSON object a POST carries, or None when it is not one."""
    try:
        data = await request.json()
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


BAD_BODY = {'error': 'body: expected a JSON object'}


class JsonOnlyPosts:
    """Reject POSTs that are not application/json.

    A page on another site can send text or form POSTs without a CORS
    preflight, and this server never grants one. Requiring JSON therefore
    keeps other sites from driving local sessions.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http' and scope['method'] == 'POST':
            content_type = dict(scope['headers']).get(b'content-type', b'')
            if content_type.split(b';')[0].strip().lower() != b'application/json':
                response = JSONResponse({'error': 'POST requests must be application/json'},
                                        status_code=415)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


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
        self.replies = {}
        self.query_ids = itertools.count(1)
        self.handle = WorkerHandle(ref, parameters, options, self._on_message)

    def query(self, message):
        """Send a query to the worker; the future resolves with its reply."""
        future = concurrent.futures.Future()
        query_id = next(self.query_ids)
        with self.lock:
            self.replies[query_id] = future
        self.handle.send({**message, 'op': 'query', 'id': query_id})
        return query_id, future

    def forget(self, query_id):
        with self.lock:
            self.replies.pop(query_id, None)

    def _snapshot_locked(self):
        ready = {key: value for key, value in (self.ready or {}).items()
                 if key not in ('type', 'state', 'update', 'architecture_text')}
        return {'id': self.id, 'workload': self.workload, 'parameters': self.parameters,
                **ready, 'state': self.state, 'update': len(self.records)}

    def snapshot(self):
        with self.lock:
            return self._snapshot_locked()

    def _on_message(self, message):  # worker reader thread
        kind = message.get('type')
        if kind == 'reply':
            with self.lock:
                future = self.replies.pop(message.get('id'), None)
            if future is not None and not future.done():  # a timed-out query is cancelled
                future.set_result(message)
            return
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
                for future in self.replies.values():
                    if not future.done():
                        future.set_result({'type': 'reply', 'code': 'exited',
                                           'error': 'session worker has exited'})
                self.replies.clear()
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


def create_app(registry, store_dir=None, build_timeout=300.0,
               allowed_hosts=('127.0.0.1', 'localhost')):
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
        if body is None:
            return _json(BAD_BODY, 400)
        name = body.get('workload')
        if name not in manager.registry:
            return _json({'error': f'unknown workload {name!r}'}, 404)
        parameters = body.get('parameters') or {}
        horizon = body.get('horizon')
        trace_level = body.get('trace_level', 'full')
        core_map = body.get('core_map') or None
        try:
            if not isinstance(parameters, dict):
                raise ValueError('parameters: expected an object')
            if core_map is not None and (not isinstance(core_map, dict) or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in core_map.items())):
                raise ValueError('core_map: expected an object mapping "tile.core" to "tile.core"')
            resolve_parameters(manager.workload(name), parameters)
            if horizon is not None:
                ParameterSpec('horizon', 'int', minimum=1).validate(horizon)
            if trace_level != 'full':
                raise ValueError(f"trace level {trace_level!r} is not available; "
                                 "'aggregate' arrives in stage 5, use 'full'")
        except ValueError as error:
            return _json({'error': str(error)}, 400)
        session_id = uuid.uuid4().hex[:12]
        options = {'trace_level': trace_level, 'horizon': horizon, 'core_map': core_map,
                   'store_dir': str(manager.store_dir) if manager.store_dir else None}
        session = ManagedSession(manager, session_id, name, manager.registry[name],
                                 parameters, options)
        manager.sessions[session_id] = session
        ready = await asyncio.to_thread(session.ready_event.wait, manager.build_timeout)
        if session.error or not ready:
            await asyncio.to_thread(manager.close, session_id)
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
        # Closing waits for the worker to exit; keep the event loop free.
        await asyncio.to_thread(manager.close, request.path_params['session_id'])
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
        body = await _body(request)
        if body is None:
            return _json(BAD_BODY, 400)
        n = body.get('n', 1)
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

    async def neuron(request):
        session = lookup(request)
        if session is None:
            return _json({'error': 'no such session'}, 404)
        if not session.alive:
            return _json({'error': 'session worker has exited; start a new session'}, 409)
        try:
            offset = int(request.path_params['offset'])
        except ValueError:
            return _json({'error': 'offset: expected an integer'}, 404)
        query_id, future = session.query({'what': 'neuron',
                                          'group': request.path_params['group'],
                                          'offset': offset})
        try:
            reply = await asyncio.wait_for(asyncio.wrap_future(future), 10)
        except asyncio.TimeoutError:
            session.forget(query_id)
            return _json({'error': 'the session did not answer within 10 s'}, 504)
        status = {'not_found': 404, 'bad_request': 400, 'exited': 409}
        if 'error' in reply:
            return _json({'error': reply['error']}, status.get(reply.get('code'), 500))
        return _json(reply['data'])

    async def architectures(request):
        return _json(sorted(bundled_architectures()))

    async def architecture(request):
        session = lookup(request)
        if session is None:
            return _json({'error': 'no such session'}, 404)
        with session.lock:
            ready = session.ready or {}
        if 'architecture_text' not in ready:
            return _json({'error': 'the session is not ready'}, 409)
        # Serve the text that was simulated, not whatever the file holds now.
        name = Path((ready.get('manifest') or {}).get('architecture_yaml', 'architecture.yaml')).name
        text = ready['architecture_text']
        baseline = request.query_params.get('baseline')
        diff = None
        if baseline:
            bundled = bundled_architectures()
            if baseline not in bundled:
                return _json({'error': f'unknown baseline {baseline!r}; '
                                       f'choose one of {", ".join(sorted(bundled))}'}, 404)
            diff = await asyncio.to_thread(diff_texts, name, text, bundled[baseline].name,
                                           bundled[baseline].read_text())
        return _json({'name': name, 'text': text, 'diff': diff})

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
        Route('/api/sessions/{session_id}/neurons/{group}/{offset}', neuron),
        Route('/api/sessions/{session_id}/architecture', architecture),
        Route('/api/architectures', architectures),
        WebSocketRoute('/ws/sessions/{session_id}', events),
        Mount('/', StaticFiles(directory=WEB, html=True, check_dir=False)),
    ]
    # The Host allow-list stops DNS-rebinding pages from reaching the API.
    middleware = [Middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts)),
                  Middleware(JsonOnlyPosts)]
    app = Starlette(routes=routes, lifespan=lifespan, middleware=middleware)
    app.state.manager = manager
    return app
