"""Worker processes: one session per process, driven over a Pipe."""
from pathlib import Path
import queue
import unittest

from sanafe.studio.server.worker import WorkerHandle, WorkloadRef

TESTS = str(Path(__file__).resolve().parent)
CHAIN = WorkloadRef('studio_helpers:ChainWorkload', (TESTS,))
CRASH = WorkloadRef('studio_helpers:CrashOnSecondUpdate', (TESTS,))


class Collector:
    def __init__(self):
        self.messages = queue.Queue()

    def __call__(self, message):
        self.messages.put(message)

    def next(self, timeout=120):
        return self.messages.get(timeout=timeout)

    def until(self, predicate, timeout=120):
        seen = []
        while True:
            message = self.next(timeout)
            seen.append(message)
            if predicate(message):
                return seen


def settled(message):
    return message['type'] == 'state' and message['state'] != 'running'


class TestWorker(unittest.TestCase):
    def start(self, ref, parameters, **options):
        events = Collector()
        handle = WorkerHandle(ref, parameters, options, events)
        self.addCleanup(handle.close)
        return handle, events

    def test_ready_then_step_streams_updates(self):
        handle, events = self.start(CHAIN, {'placement': 'far'})
        ready = events.next()
        self.assertEqual(ready['type'], 'ready')
        self.assertEqual(ready['horizon'], 6)
        self.assertEqual(ready['network']['occupied'], ['0.0', '16.0', '31.0'])
        handle.send({'op': 'step', 'n': 2})
        seen = events.until(settled)
        self.assertEqual(seen[0]['state'], 'running')
        self.assertEqual([m['record']['update'] for m in seen if m['type'] == 'update'], [1, 2])
        self.assertEqual((seen[-1]['state'], seen[-1]['update']), ('paused', 2))

    def test_run_to_horizon_and_reset(self):
        handle, events = self.start(CHAIN, {})
        events.next()
        handle.send({'op': 'run'})
        final = events.until(settled)[-1]
        self.assertEqual((final['state'], final['update']), ('finished', 6))
        handle.send({'op': 'reset'})
        ready = events.until(lambda m: m['type'] == 'ready')[-1]
        self.assertEqual((ready['state'], ready['update']), ('idle', 0))

    def test_pause_interrupts_a_run(self):
        handle, events = self.start(CHAIN, {'steps': 2000})
        events.next()
        handle.send({'op': 'run'})
        events.until(lambda m: m['type'] == 'update')
        handle.send({'op': 'pause'})
        final = events.until(settled)[-1]
        self.assertEqual(final['state'], 'paused')
        self.assertLess(final['update'], 2000)

    def test_pause_right_after_run_is_honored(self):
        handle, events = self.start(CHAIN, {'steps': 2000})
        events.next()
        handle.send({'op': 'run'})
        handle.send({'op': 'pause'})
        final = events.until(settled)[-1]
        self.assertEqual(final['state'], 'paused')
        self.assertLess(final['update'], 2000)

    def test_pause_while_idle_is_ignored(self):
        handle, events = self.start(CHAIN, {})
        events.next()
        handle.send({'op': 'pause'})
        handle.send({'op': 'step', 'n': 2})
        final = events.until(settled)[-1]
        self.assertEqual((final['state'], final['update']), ('paused', 2))

    def test_build_error_is_reported(self):
        _, events = self.start(CHAIN, {'steps': 0})
        message = events.next()
        self.assertEqual(message['type'], 'error')
        self.assertIn('steps: must be at least 1', message['message'])
        self.assertEqual(events.next()['type'], 'exited')

    def test_crash_reports_exit_code(self):
        handle, events = self.start(CRASH, {})
        events.next()
        handle.send({'op': 'run'})
        seen = events.until(lambda m: m['type'] == 'exited')
        self.assertEqual([m['record']['update'] for m in seen if m['type'] == 'update'], [1])
        self.assertEqual(seen[-1]['exitcode'], 7)

    def test_close_during_a_run_is_prompt(self):
        handle, events = self.start(CHAIN, {'steps': 200000})
        events.next()
        handle.send({'op': 'run'})
        events.until(lambda m: m['type'] == 'update')
        import time
        started = time.monotonic()
        handle.close()
        self.assertLess(time.monotonic() - started, 2.0)

    def test_workload_setting_start_method_at_import_loads(self):
        # Lava calls multiprocessing.set_start_method('fork') at import; a
        # spawned worker already has a start method, so that used to raise.
        _, events = self.start(WorkloadRef('studio_forking_workload:ForkingChain', (TESTS,)), {})
        message = events.next()
        self.assertEqual(message['type'], 'ready', message)

    def test_query_answers_during_a_run(self):
        import time
        handle, events = self.start(WorkloadRef('studio_helpers:SlowFirstUpdate', (TESTS,)), {})
        events.next()
        handle.send({'op': 'run'})
        events.until(lambda m: m['type'] == 'state' and m['state'] == 'running')
        started = time.monotonic()
        handle.send({'op': 'query', 'id': 7, 'what': 'neuron', 'group': 'layer_1',
                     'offset': 2})
        reply = events.until(lambda m: m['type'] == 'reply')[-1]
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertEqual((reply['id'], reply['data']['fan_in_total']), (7, 8))
        handle.send({'op': 'query', 'id': 8, 'what': 'neuron', 'group': 'layer_9',
                     'offset': 0})
        missing = events.until(lambda m: m['type'] == 'reply')[-1]
        self.assertEqual((missing['id'], missing['code']), (8, 'not_found'))
        handle.send({'op': 'query', 'id': 9, 'what': 'weather'})
        unknown = events.until(lambda m: m['type'] == 'reply')[-1]
        self.assertEqual((unknown['id'], unknown['code']), (9, 'bad_request'))

    def test_workload_ref_rejects_bad_target(self):
        with self.assertRaisesRegex(ValueError, 'module:Class'):
            WorkloadRef('no_colon_here').load()


if __name__ == '__main__':
    unittest.main()
