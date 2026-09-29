"""HTTP commands, WebSocket events, reconnect, and fault isolation."""
from pathlib import Path
import unittest

from starlette.testclient import TestClient

from sanafe.studio.server.app import create_app
from sanafe.studio.server.worker import WorkloadRef

TESTS = str(Path(__file__).resolve().parent)
REGISTRY = {
    'sanafe-files': WorkloadRef('sanafe.studio.engine.workload:SanafeFiles'),
    'test-chain': WorkloadRef('studio_helpers:ChainWorkload', (TESTS,)),
    'test-crash': WorkloadRef('studio_helpers:CrashOnSecondUpdate', (TESTS,)),
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
        self.client = TestClient(create_app(REGISTRY))
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
        self.assertEqual(self.client.post('/api/sessions/unknown/run').status_code, 404)

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
            self.client.post(f'/api/sessions/{sid}/run')
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
            self.assertEqual(self.client.post(f'/api/sessions/{sid}/reset').status_code, 202)
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
            self.client.post(f'/api/sessions/{crashed}/run')
            seen = until_settled(socket)
        self.assertEqual(seen[-1]['state'], 'faulted')
        self.assertIn('worker exited with code 7', seen[-1]['fault'])
        self.assertFalse(seen[-1]['alive'])
        self.assertEqual(self.client.post(f'/api/sessions/{crashed}/step').status_code, 409)
        self.assertEqual(self.client.get('/api/workloads').status_code, 200)
        other = self.create()['id']
        self.assertEqual(self.client.post(f'/api/sessions/{other}/step',
                                          json={'n': 1}).status_code, 202)

    def test_delete(self):
        sid = self.create()['id']
        self.assertEqual(self.client.delete(f'/api/sessions/{sid}').status_code, 204)
        self.assertEqual(self.client.get(f'/api/sessions/{sid}').status_code, 404)


if __name__ == '__main__':
    unittest.main()
