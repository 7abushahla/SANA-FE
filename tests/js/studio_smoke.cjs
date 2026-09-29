/* Headless smoke test of the Studio browser UI against a live server.
 *
 * Needs Node 18 or later and jsdom, which is not a repository dependency:
 *   npm install --prefix <dir> jsdom@24
 *   NODE_PATH=<dir>/node_modules node tests/js/studio_smoke.cjs
 * Run from the SANA-FE directory. Exits nonzero on failure.
 */
const { spawn } = require('child_process');
const { JSDOM } = require('jsdom');

const PORT = 8800 + Math.floor(Math.random() * 100);
const BASE = 'http://127.0.0.1:' + PORT + '/';
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function waitFor(check, what, timeout = 120000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    const value = await check();
    if (value) return value;
    await sleep(100);
  }
  throw new Error('timed out waiting for ' + what);
}

(async () => {
  const server = spawn('.venv/bin/python', ['-m', 'sanafe.studio', '--port', String(PORT), '--path', 'tests/python',
    '--workload', 'test-chain=studio_helpers:ChainWorkload'], { stdio: ['ignore', 'pipe', 'pipe'] });
  let log = '';
  server.stdout.on('data', (d) => { log += d; });
  server.stderr.on('data', (d) => { log += d; });
  const errors = [];
  try {
    await waitFor(async () => {
      try { return (await fetch(BASE + 'api/workloads')).ok; } catch (e) { return false; }
    }, 'server', 60000);
    const dom = await JSDOM.fromURL(BASE, {
      runScripts: 'dangerously', resources: 'usable', pretendToBeVisual: true,
      beforeParse(window) {
        window.fetch = (input, init) => fetch(new URL(input, window.location.href), init);
        window.addEventListener('error', (event) => errors.push(event.message));
      },
    });
    const { document, Event, MouseEvent } = dom.window;
    const $ = (id) => document.getElementById(id);
    const text = (id) => $(id).textContent;

    await waitFor(() => [...$('workload').options].some((o) => o.value === 'test-chain'), 'workload list');
    $('workload').value = 'test-chain';
    $('workload').dispatchEvent(new Event('change'));
    $('speed').value = '1200';
    $('speed').dispatchEvent(new Event('change'));
    $('btnStart').click();
    await waitFor(() => text('uHorizon') === '6', 'session ready');
    const used = document.querySelectorAll('#chip rect.core.used').length;
    if (used !== 3) throw new Error('expected 3 occupied cores, found ' + used);
    if (!/Loihi 2 candidate/.test(text('badge'))) throw new Error('badge missing: ' + text('badge'));

    $('btnRun').click();
    await waitFor(() => text('state') === 'finished' && text('uNum') === '6', 'run to finish');
    await waitFor(() => /^update 6 ·/.test(text('clockText')), 'playback of update 6', 90000);
    const rows = document.querySelectorAll('#dock text.rowlabel').length;
    if (rows !== 3) throw new Error('expected 3 timeline rows, found ' + rows);

    document.querySelector('#tabs button[data-tab="perf"]').click();
    await waitFor(() => document.querySelectorAll('#dock rect.bar').length === 24, '24 energy bars (6 updates x 4 units)');
    document.querySelector('#tabs button[data-tab="messages"]').click();
    await waitFor(() => document.querySelectorAll('#dock tr').length > 1, 'messages table');
    document.querySelector('#chip rect.core.used').dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await waitFor(() => /core 0\.0/.test(text('insTitle')), 'inspector core selection');

    $('params').querySelector('[data-param="steps"]').value = '3000';
    $('btnStart').click();
    await waitFor(() => text('uHorizon') === '3000', 'rebuilt session');
    $('btnRun').click();
    await waitFor(() => text('state') === 'running', 'running');
    $('btnPause').click();
    await waitFor(() => text('state') === 'paused', 'paused');

    if (errors.length) throw new Error('page errors: ' + errors.join('; '));
    console.log('studio smoke: OK (3 cores, 6 updates played, 24 bars, messages, inspector, pause)');
    dom.window.close();
  } catch (error) {
    console.error('studio smoke: FAIL: ' + error.message);
    console.error(log.slice(-3000));
    process.exitCode = 1;
  } finally {
    // The page's WebSocket reconnect timer would keep Node alive, so exit
    // explicitly once the server has had a moment to shut its workers down.
    server.kill('SIGTERM');
    setTimeout(() => process.exit(process.exitCode || 0), 1500);
  }
})();
