"""HTTP commands, WebSocket events, reconnect, and fault isolation."""
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import unittest
import urllib.request

from websockets.sync.client import connect as ws_connect

from starlette.testclient import TestClient

from sanafe.studio.server.app import create_app
from sanafe.studio.server.worker import WorkloadRef

TESTS = str(Path(__file__).resolve().parent)
REGISTRY = {
    'sanafe-files': WorkloadRef('sanafe.studio.engine.workload:SanafeFiles'),
    'test-chain': WorkloadRef('studio_helpers:ChainWorkload', (TESTS,)),
    'test-crash': WorkloadRef('studio_helpers:CrashOnSecondUpdate', (TESTS,)),
    'test-slow': WorkloadRef('studio_helpers:SlowFirstUpdate', (TESTS,)),
}


def until_settled(socket):
    seen = []
    while True:
        message = socket.receive_json()
        seen.append(message)
        if message['type'] == 'state' and message['state'] != 'running':
            return seen


class TestServer(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(REGISTRY, allowed_hosts=('testserver',)))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def create(self, workload='test-chain', **parameters):
        response = self.client.post('/api/sessions',
                                    json={'workload': workload, 'parameters': parameters})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def test_workloads_list_parameter_schemas(self):
        by_name = {w['name']: w for w in self.client.get('/api/workloads').json()}
        self.assertEqual(set(by_name), set(REGISTRY))
        steps = {p['name']: p for p in by_name['test-chain']['parameters']}['steps']
        self.assertEqual((steps['kind'], steps['default'], steps['minimum']), ('int', 6, 1))

    def test_create_session_describes_chip(self):
        session = self.create(placement='far')
        self.assertEqual((session['layout']['width'], session['layout']['height']), (8, 4))
        self.assertEqual(session['network']['occupied'], ['0.0', '16.0', '31.0'])
        self.assertEqual(session['horizon'], 6)
        self.assertEqual(session['badge'],
                         'Loihi 2 candidate · costs inherited from Loihi 1 · not hardware')
        self.assertEqual(session['state']['state'], 'idle')
        fetched = self.client.get(f"/api/sessions/{session['id']}").json()
        self.assertEqual(fetched['id'], session['id'])

    def test_invalid_requests(self):
        bad = self.client.post('/api/sessions',
                               json={'workload': 'test-chain', 'parameters': {'steps': 0}})
        self.assertEqual(bad.status_code, 400)
        self.assertIn('steps: must be at least 1', bad.json()['error'])
        self.assertEqual(self.client.post('/api/sessions',
                                          json={'workload': 'nope'}).status_code, 404)
        sid = self.create()['id']
        for n in (0, '3'):
            response = self.client.post(f'/api/sessions/{sid}/step', json={'n': n})
            self.assertEqual(response.status_code, 400)
            self.assertIn('n: ', response.json()['error'])
        self.assertEqual(self.client.get(f'/api/sessions/{sid}/updates?from=-1').status_code, 400)
        self.assertEqual(self.client.post('/api/sessions/unknown/run', json={}).status_code, 404)

    def test_streamed_updates_equal_resent_updates(self):
        sid = self.create()['id']
        with self.client.websocket_connect(f'/ws/sessions/{sid}') as socket:
            hello = socket.receive_json()
            self.assertEqual((hello['type'], hello['update']), ('hello', 0))
            self.assertEqual(self.client.post(f'/api/sessions/{sid}/step',
                                              json={'n': 2}).status_code, 202)
            seen = until_settled(socket)
            self.assertEqual((seen[-1]['state'], seen[-1]['update']), ('paused', 2))
            self.assertTrue(seen[-1]['alive'])
            self.client.post(f'/api/sessions/{sid}/run', json={})
            seen += until_settled(socket)
            self.assertEqual((seen[-1]['state'], seen[-1]['update']), ('finished', 6))
        streamed = [m['record'] for m in seen if m['type'] == 'update']
        resent = self.client.get(f'/api/sessions/{sid}/updates?from=0').json()['updates']
        self.assertEqual(streamed, resent)
        self.assertEqual([r['update'] for r in resent], [1, 2, 3, 4, 5, 6])
        with self.client.websocket_connect(f'/ws/sessions/{sid}') as socket:  # reconnect
            hello = socket.receive_json()
            self.assertEqual((hello['update'], hello['state']['state']), (6, 'finished'))
        tail = self.client.get(f'/api/sessions/{sid}/updates?from=4').json()['updates']
        self.assertEqual(tail, resent[4:])

    def test_reset_broadcasts_ready(self):
        sid = self.create()['id']
        with self.client.websocket_connect(f'/ws/sessions/{sid}') as socket:
            socket.receive_json()
            self.client.post(f'/api/sessions/{sid}/step', json={'n': 1})
            until_settled(socket)
            self.assertEqual(self.client.post(f'/api/sessions/{sid}/reset', json={}).status_code, 202)
            while True:
                message = socket.receive_json()
                if message['type'] == 'ready':
                    break
        self.assertEqual(message['state']['state'], 'idle')
        self.assertEqual(message['update'], 0)
        self.assertEqual(self.client.get(f'/api/sessions/{sid}/updates?from=0').json()['updates'], [])

    def test_crashed_worker_leaves_server_running(self):
        crashed = self.create('test-crash')['id']
        with self.client.websocket_connect(f'/ws/sessions/{crashed}') as socket:
            socket.receive_json()
            self.client.post(f'/api/sessions/{crashed}/run', json={})
            seen = until_settled(socket)
        self.assertEqual(seen[-1]['state'], 'faulted')
        self.assertIn('worker exited with code 7', seen[-1]['fault'])
        self.assertFalse(seen[-1]['alive'])
        self.assertEqual(self.client.post(f'/api/sessions/{crashed}/step', json={}).status_code, 409)
        self.assertEqual(self.client.get('/api/workloads').status_code, 200)
        other = self.create()['id']
        self.assertEqual(self.client.post(f'/api/sessions/{other}/step',
                                          json={'n': 1}).status_code, 202)

    def test_static_page_served(self):
        page = self.client.get('/')
        self.assertEqual(page.status_code, 200)
        self.assertIn('<title>SANA-FE Studio</title>', page.text)
        import re
        scripts = re.findall(r'<script src="js/(\w+)\.js">', page.text)
        self.assertTrue({'zoom', 'network', 'watch', 'arch', 'main'} <= set(scripts), scripts)
        for script in scripts:
            with self.subTest(script=script):
                self.assertEqual(self.client.get(f'/js/{script}.js').status_code, 200)
        self.assertEqual(self.client.get('/app.css').status_code, 200)

    def test_closing_a_busy_session_does_not_block_the_server(self):
        import threading
        sid = self.create('test-slow')['id']
        with self.client.websocket_connect(f'/ws/sessions/{sid}') as socket:
            socket.receive_json()
            self.client.post(f'/api/sessions/{sid}/run', json={})
            while socket.receive_json().get('state') != 'running':
                pass
        deleter = threading.Thread(target=self.client.delete, args=(f'/api/sessions/{sid}',))
        deleter.start()
        time.sleep(0.3)
        started = time.monotonic()
        self.assertEqual(self.client.get('/api/workloads').status_code, 200)
        self.assertLess(time.monotonic() - started, 1.0)
        deleter.join(timeout=30)

    def test_rejects_cross_site_requests(self):
        # A foreign page can send text or form POSTs without a CORS preflight,
        # and a DNS-rebinding page arrives with a foreign Host header.
        plain = self.client.post('/api/sessions', content=json.dumps({'workload': 'test-chain'}),
                                 headers={'Content-Type': 'text/plain'})
        self.assertEqual(plain.status_code, 415)
        self.assertIn('application/json', plain.json()['error'])
        foreign = self.client.get('/api/workloads', headers={'host': 'attacker.example'})
        self.assertEqual(foreign.status_code, 400)
        sid = self.create()['id']
        broken = self.client.post(f'/api/sessions/{sid}/step', content='{not json',
                                  headers={'Content-Type': 'application/json'})
        self.assertEqual(broken.status_code, 400)
        self.assertIn('body: expected a JSON object', broken.json()['error'])

    def test_neuron_endpoint(self):
        sid = self.create()['id']
        detail = self.client.get(f'/api/sessions/{sid}/neurons/layer_1/2')
        self.assertEqual(detail.status_code, 200, detail.text)
        self.assertEqual((detail.json()['core'], detail.json()['fan_in_total']), ('16.0', 8))
        self.assertEqual(self.client.get(f'/api/sessions/{sid}/neurons/layer_9/0').status_code, 404)
        self.assertEqual(self.client.get(f'/api/sessions/{sid}/neurons/layer_1/x').status_code, 404)
        self.assertEqual(self.client.get('/api/sessions/nope/neurons/layer_1/2').status_code, 404)

    def test_architecture_endpoint(self):
        names = self.client.get('/api/architectures').json()
        self.assertIn('loihi', names)
        sid = self.create()['id']
        plain = self.client.get(f'/api/sessions/{sid}/architecture').json()
        self.assertIn('loihi2_candidate', plain['text'])
        self.assertIsNone(plain['diff'])
        diff = self.client.get(f'/api/sessions/{sid}/architecture?baseline=loihi').json()['diff']
        self.assertIn('architecture.name', [row['path'] for row in diff['rows']])
        self.assertEqual(self.client.get(
            f'/api/sessions/{sid}/architecture?baseline=nope').status_code, 404)

    def test_architecture_is_the_simulated_snapshot(self):
        import shutil
        import tempfile
        from importlib.resources import files
        scratch = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, scratch)
        arch = scratch / 'chip.yaml'
        shutil.copy(str(files('sanafe.examples') / 'example_chip.yaml'), arch)
        repo = Path(TESTS).parents[1]
        response = self.client.post('/api/sessions', json={'workload': 'sanafe-files', 'parameters': {
            'arch_yaml': str(arch), 'net_file': str(repo / 'snn' / 'example.net')}})
        self.assertEqual(response.status_code, 201, response.text)
        sid = response.json()['id']
        original = arch.read_text()
        edited = original.replace('  name: demo\n', '  name: edited_after_build\n', 1)
        self.assertNotEqual(edited, original)
        arch.write_text(edited)
        served = self.client.get(f'/api/sessions/{sid}/architecture?baseline=example_chip').json()
        self.assertEqual(served['text'], original)
        self.assertEqual(served['diff']['rows'], [])

    def test_late_reply_to_an_abandoned_query_is_ignored(self):
        import concurrent.futures
        sid = self.create()['id']
        session = self.client.app.state.manager.sessions[sid]
        abandoned = concurrent.futures.Future()
        abandoned.cancel()
        session.replies[999] = abandoned
        session._on_message({'type': 'reply', 'id': 999, 'data': {}})  # must not raise
        self.assertEqual(self.client.get(f'/api/sessions/{sid}/neurons/layer_1/2').status_code, 200)

    def test_create_with_a_core_map(self):
        response = self.client.post('/api/sessions', json={
            'workload': 'test-chain', 'core_map': {'16.0': '5.1', '5.1': '16.0'}})
        self.assertEqual(response.status_code, 201, response.text)
        groups = {g['name']: g['cores'] for g in response.json()['network']['groups']}
        self.assertEqual(groups['layer_1'], {'5.1': 4})
        for bad in ({'16.0': '99.9'}, ['16.0'], {'16.0': 3}):
            with self.subTest(core_map=bad):
                rejected = self.client.post('/api/sessions', json={'workload': 'test-chain',
                                                                   'core_map': bad})
                self.assertEqual(rejected.status_code, 400, rejected.text)
                self.assertIn('core_map', rejected.json()['error'])

    def test_breakpoints_round_trip_and_stop(self):
        sid = self.create()['id']
        url = f'/api/sessions/{sid}/breakpoints'
        accepted = self.client.put(url, json={'breakpoints': [
            {'id': 'b', 'kind': 'update', 'equals': 3}]})
        self.assertEqual(accepted.status_code, 200, accepted.text)
        self.assertEqual(accepted.json()['breakpoints'][0]['enabled'], True)
        rejected = self.client.put(url, json={'breakpoints': [{'id': 'x', 'kind': 'weather'}]})
        self.assertEqual(rejected.status_code, 400)
        self.assertIn('unknown kind', rejected.json()['error'])
        self.assertEqual(self.client.put(url, json=['no']).status_code, 400)
        self.assertEqual(self.client.get(f'/api/sessions/{sid}').json()['breakpoints'][0]['id'], 'b')
        with self.client.websocket_connect(f'/ws/sessions/{sid}') as socket:
            socket.receive_json()
            self.client.post(f'/api/sessions/{sid}/run', json={})
            final = until_settled(socket)[-1]
        self.assertEqual((final['state'], final['update'], final['reason']),
                         ('stopped', 3, 'breakpoint b: update 3'))

    def test_create_carries_breakpoints(self):
        response = self.client.post('/api/sessions', json={
            'workload': 'test-chain', 'breakpoints': [
                {'id': 'b1', 'kind': 'update', 'equals': 2},
                {'id': 'b2', 'kind': 'neuron_fires', 'neuron': 'layer_9.0'}]})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual([b['id'] for b in response.json()['breakpoints']], ['b1'])
        [warning] = response.json()['breakpoint_warnings']
        self.assertIn('b2', warning)
        self.assertIn('no neuron', warning)
        self.assertEqual(self.client.post('/api/sessions', json={
            'workload': 'test-chain', 'breakpoints': 'b1'}).status_code, 400)

    def test_delete(self):
        sid = self.create()['id']
        self.assertEqual(self.client.delete(f'/api/sessions/{sid}').status_code, 204)
        self.assertEqual(self.client.get(f'/api/sessions/{sid}').status_code, 404)


class TestRunsServer(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.store = tempfile.mkdtemp()
        self.client = TestClient(create_app(REGISTRY, store_dir=self.store,
                                            allowed_hosts=('testserver',)))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def run_chain(self, placement):
        response = self.client.post('/api/sessions', json={
            'workload': 'test-chain', 'parameters': {'placement': placement}})
        sid = response.json()['id']
        with self.client.websocket_connect(f'/ws/sessions/{sid}') as socket:
            socket.receive_json()
            self.client.post(f'/api/sessions/{sid}/run', json={})
            until_settled(socket)
        return response.json()['run']

    def test_runs_compare_and_export(self):
        far, near = self.run_chain('far'), self.run_chain('near')
        runs = self.client.get('/api/runs').json()
        self.assertEqual([r['id'] for r in runs], [near, far])
        self.assertEqual(runs[0]['updates'], 6)
        compared = self.client.post('/api/compare', json={'a': far, 'b': near})
        self.assertEqual(compared.status_code, 200, compared.text)
        self.assertTrue(compared.json()['identical_spikes'])
        self.assertEqual(self.client.post('/api/compare', json={'a': far, 'b': 'nope'}).status_code, 404)
        self.assertEqual(self.client.post('/api/compare', json={'a': '../x', 'b': far}).status_code, 400)
        svg = self.client.get(f'/api/runs/{far}/plots/raster.svg')
        self.assertEqual(svg.status_code, 200)
        self.assertEqual(svg.headers['content-type'], 'image/svg+xml')
        self.assertEqual(self.client.get(f'/api/runs/{far}/plots/weather.svg').status_code, 404)
        self.assertEqual(self.client.get('/api/runs/nope/plots/raster.svg').status_code, 404)

    def test_export_of_an_empty_run_is_409(self):
        response = self.client.post('/api/sessions', json={'workload': 'test-chain'})
        run = response.json()['run']
        empty = self.client.get(f'/api/runs/{run}/plots/energy.svg')
        self.assertEqual(empty.status_code, 409)
        self.assertIn('no updates', empty.json()['error'])


class TestServerProcess(unittest.TestCase):
    """Behavior only a real uvicorn process shows (the test client hides it)."""

    def test_shutdown_after_a_client_disconnects(self):
        # Found by the headless smoke test: a handler that never notices a
        # disconnect keeps the connection open, so SIGTERM never finishes.
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        base = f'http://127.0.0.1:{port}'
        process = subprocess.Popen(
            [sys.executable, '-m', 'sanafe.studio', '--port', str(port), '--no-store', '--path', TESTS,
             '--workload', 'test-chain=studio_helpers:ChainWorkload'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 60
            while True:
                try:
                    urllib.request.urlopen(base + '/api/workloads', timeout=2).read()
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.2)
            request = urllib.request.Request(
                base + '/api/sessions', method='POST',
                data=json.dumps({'workload': 'test-chain', 'parameters': {}}).encode(),
                headers={'Content-Type': 'application/json'})
            sid = json.loads(urllib.request.urlopen(request, timeout=120).read())['id']
            with ws_connect(f'ws://127.0.0.1:{port}/ws/sessions/{sid}') as client:
                self.assertEqual(json.loads(client.recv(timeout=10))['type'], 'hello')
            time.sleep(0.5)
            process.send_signal(signal.SIGTERM)
            # uvicorn finishes its graceful shutdown, then re-raises SIGTERM,
            # so the status is 0 or -SIGTERM. Hanging here was the bug.
            self.assertIn(process.wait(timeout=20), (0, -signal.SIGTERM))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


if __name__ == '__main__':
    unittest.main()
