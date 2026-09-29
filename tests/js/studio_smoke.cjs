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
    const note = $('chipNote');
    if (!note || !note.querySelector('.m.X') || !/reconstructed/.test(note.textContent)) {
      throw new Error('chip must mark the reconstructed packet route with X');
    }

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
    if (!/assumed total bytes/.test(text('inspector'))) throw new Error('candidate core budget missing from the inspector');

    // Zoom chip > tile > core, with the mini-map, and keep it across a new update.
    const click = (node) => node.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    const shown = (id) => $(id).style.display !== 'none';
    click(document.querySelector('#chip rect.tile[data-tile="0"]'));
    await waitFor(() => /Tile 0/.test(text('crumb')) && document.querySelectorAll('#zoom .corepanel').length === 4, 'tile view');
    if (!shown('minimap') || !document.querySelector('#minimap rect.tile')) throw new Error('mini-map missing in tile view');
    click(document.querySelector('#zoom .corepanel[data-core="0.0"]'));
    await waitFor(() => /Core 0/.test(text('crumb')) && document.querySelectorAll('#zoom .ncell').length === 8, 'core view with 8 neurons');
    if (document.querySelectorAll('#zoom .unit').length < 5 || !document.querySelector('#zoom .buf')) throw new Error('pipeline strip incomplete');
    await waitFor(() => /16\.0/.test($('axonOut').textContent), 'axon_out lists core 16.0');
    // The test chain declares no host input group, so layer_0 has no incoming connections.
    if (!/no incoming connections/.test($('axonIn').textContent)) throw new Error('axon_in: ' + $('axonIn').textContent);
    $('btnStep').click();
    await waitFor(() => text('uNum') === '7', 'step past the horizon');
    await waitFor(() => /^update 7 ·/.test(text('clockText')), 'update 7 shown', 90000);
    if (!/Core 0/.test(text('crumb')) || document.querySelectorAll('#zoom .ncell').length !== 8) throw new Error('zoom lost after a new update');
    click(document.querySelector('#minimap'));
    await waitFor(() => shown('chip') && !shown('minimap') && text('crumb') === 'Chip', 'back to the chip');

    // Network mode: three groups, two labeled connections, core chips link to the core view.
    click($('mNet'));
    await waitFor(() => document.querySelectorAll('#network g.group').length === 3, 'network groups');
    const labels = [...document.querySelectorAll('#network text.edgelabel')].map((n) => n.textContent).join(' | ');
    if (!/32 synapses/.test(labels) || !/8 synapses/.test(labels)) throw new Error('edge labels: ' + labels);
    click(document.querySelector('#network [data-core="16.0"]'));
    await waitFor(() => shown('zoom') && /Tile 16/.test(text('crumb')) && /Core 0/.test(text('crumb')), 'network chip opens the core');

    // Neuron inspection and watch, messages filter, Architecture diff, reference status.
    await waitFor(() => document.querySelectorAll('#zoom .ncell').length === 4, 'layer_1 cells');
    click(document.querySelector('#zoom .ncell[data-neuron="layer_1.2"]'));
    await waitFor(() => /neuron layer_1\.2/.test(text('insTitle')) && /fan-in8\n/.test(text("inspector")),
      'neuron inspector').catch((e) => { throw new Error(e.message + ': ' + text('insTitle') + ' / ' + text('inspector').slice(0, 300)); });
    $('btnWatch').click();
    document.querySelector('#tabs button[data-tab="watch"]').click();
    await waitFor(() => document.querySelectorAll('#dock svg.watchplot').length === 1, 'one watch plot');
    document.querySelector('#tabs button[data-tab="messages"]').click();
    await waitFor(() => $('msgFilter'), 'messages filter');
    const allRows = document.querySelectorAll('#dock tr.msg').length;
    $('msgFilter').value = '→ 31.0';
    $('msgFilter').dispatchEvent(new Event('input'));
    await waitFor(() => {
      const rows = [...document.querySelectorAll('#dock tr.msg')];
      return rows.length > 0 && rows.length < allRows && rows.every((r) => /31\.0/.test(r.textContent));
    }, 'filtered messages');
    click(document.querySelector('#dock tr.msg'));
    await waitFor(() => document.querySelector('#chip .route'), 'selected message route drawn');
    document.querySelector('#tabs button[data-tab="arch"]').click();
    await waitFor(() => $('archBaseline') && [...$('archBaseline').options].some((o) => o.value === 'loihi'), 'baseline list');
    $('archBaseline').value = 'loihi';
    $('archBaseline').dispatchEvent(new Event('change'));
    await waitFor(() => document.querySelectorAll('#dock tr.change').length > 0, 'architecture diff rows');
    if (!/loihi2_candidate/.test($('archText').textContent)) throw new Error('YAML text missing');
    if (text('refPill') !== 'reference: none') throw new Error('reference pill: ' + text('refPill'));

    $('params').querySelector('[data-param="steps"]').value = '3000';
    $('btnStart').click();
    await waitFor(() => text('uHorizon') === '3000', 'rebuilt session');
    $('btnRun').click();
    await waitFor(() => text('state') === 'running', 'running');
    $('btnPause').click();
    await waitFor(() => text('state') === 'paused', 'paused');

    // Reload after a pause: the page must resume the same session.
    const sid = /#session=(\w+)/.exec(dom.window.location.hash);
    if (!sid) throw new Error('session id missing from the URL: ' + dom.window.location.href);
    const pausedAt = Number(text('uNum'));
    const serverUpdate = async () => (await (await fetch(BASE + 'api/sessions/' + sid[1])).json()).update;
    await waitFor(async () => (await serverUpdate()) === pausedAt || Number(text('uNum')) === (await serverUpdate()), 'paused count to settle');
    const url = dom.window.location.href;
    dom.window.close();
    const open = () => JSDOM.fromURL(url, {
      runScripts: 'dangerously', resources: 'usable', pretendToBeVisual: true,
      beforeParse(window) {
        window.fetch = (input, init) => fetch(new URL(input, window.location.href), init);
        window.addEventListener('error', (event) => errors.push(event.message));
      },
    });
    let page = await open();
    let q = (id) => page.window.document.getElementById(id);
    await waitFor(() => q('uHorizon').textContent === '3000', 'reloaded session');
    await waitFor(async () => Number(q('uNum').textContent) === (await serverUpdate()), 'catch-up after reload');
    // Reload in the middle of a run, then pause: every update exactly once.
    q('btnRun').click();
    await waitFor(() => q('state').textContent === 'running', 'running again');
    await sleep(700);
    page.window.close();
    page = await open();
    q = (id) => page.window.document.getElementById(id);
    await waitFor(() => q('state').textContent === 'running', 'reloaded mid-run');
    q('btnPause').click();
    await waitFor(() => q('state').textContent === 'paused', 'paused after reload');
    await waitFor(async () => Number(q('uNum').textContent) === (await serverUpdate()), 'no missing or duplicate updates');
    if (errors.length) throw new Error('page errors: ' + errors.join('; '));
    console.log('studio smoke: OK (3 cores, 6 updates played, 24 bars, messages, inspector, zoom and mini-map, network mode, neuron watch, message route, architecture diff, pause, X mark, reload resume)');
    page.window.close();
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
