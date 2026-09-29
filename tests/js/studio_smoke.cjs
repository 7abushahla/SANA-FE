/* Headless smoke test of the Studio browser UI against a live server.
 *
 * Needs Node 18 or later and jsdom, which is not a repository dependency:
 *   npm install --prefix <dir> jsdom@24
 *   NODE_PATH=<dir>/node_modules node tests/js/studio_smoke.cjs
 * Run from the SANA-FE directory. Exits nonzero on failure.
 */
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { JSDOM } = require('jsdom');

const STORE = fs.mkdtempSync(path.join(os.tmpdir(), 'studio-smoke-runs-'));

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
    '--workload', 'test-chain=studio_helpers:ChainWorkload', '--store-dir', STORE], { stdio: ['ignore', 'pipe', 'pipe'] });
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
    // Group names live in a legend under the chip, not on top of the mesh.
    const legend = [...document.querySelectorAll('#chip .chiplegend .item')].map((n) => n.textContent);
    if (legend.join('|') !== 'layer_0 · 1 core|layer_1 · 1 core|layer_2 · 1 core') throw new Error('legend: ' + legend.join('|'));
    if (document.querySelector('#chip text.grouplabel')) throw new Error('group labels must not be drawn over the mesh');
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
    const markOf = (label) => {
      const cell = [...document.querySelectorAll('#inspector .kv i')].find((i) => i.textContent.trim() === label);
      return cell ? cell.nextElementSibling.nextElementSibling.className : 'missing';
    };
    click(document.querySelector('#network g.group[data-group="layer_1"] rect'));
    await waitFor(() => /group layer_1/.test(text('insTitle')), 'group inspector');
    if (markOf('fired this update') !== 'm D') throw new Error('group fired count must be marked D: ' + markOf('fired this update'));
    click(document.querySelector('#network [data-core="16.0"]'));
    await waitFor(() => shown('zoom') && /Tile 16/.test(text('crumb')) && /Core 0/.test(text('crumb')), 'network chip opens the core');

    // Neuron inspection and watch, messages filter, Architecture diff, reference status.
    await waitFor(() => document.querySelectorAll('#zoom .ncell').length === 4, 'layer_1 cells');
    click(document.querySelector('#zoom .ncell[data-neuron="layer_1.2"]'));
    await waitFor(() => /neuron layer_1\.2/.test(text('insTitle')) && /fan-in8\n/.test(text("inspector")),
      'neuron inspector').catch((e) => { throw new Error(e.message + ': ' + text('insTitle') + ' / ' + text('inspector').slice(0, 300)); });
    $('btnHighlight').click();
    await waitFor(() => shown('chip') && document.querySelector('#chip rect.hl-in[data-core="0.0"]') &&
      document.querySelector('#chip rect.hl-out[data-core="31.0"]'), 'connections highlighted on the chip');
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

    // Stage 4: breakpoints, neuron actions, placement drag, saved runs, compare, export.
    const sid0 = /#session=(\w+)/.exec(dom.window.location.hash)[1];
    const earlier = (await (await fetch(BASE + 'api/sessions/' + sid0 + '/updates?from=0')).json()).updates;
    const firesAt = earlier.find((r) => r.fired.some((f) => f[0] === 'layer_2' && f[1] === 0)).update;
    $('btnReset').click();
    await waitFor(() => text('uNum') === '0' && text('state') === 'idle', 'reset to update 0');
    $('bpKind').value = 'update';
    $('bpKind').dispatchEvent(new Event('change'));
    $('bpValue').value = '3';
    $('bpAdd').click();
    await waitFor(() => document.querySelectorAll('#bpList li.bp').length === 1, 'breakpoint listed');
    $('btnRun').click();
    await waitFor(() => /^stopped: breakpoint .*update 3$/.test(text('state')) && text('uNum') === '3', 'stopped at update 3');
    if (!document.querySelector('#bpList li.bp.hit')) throw new Error('the hit breakpoint is not highlighted');
    click(document.querySelector('#bpList li.bp button.bpDel'));
    await waitFor(() => document.querySelectorAll('#bpList li.bp').length === 0, 'breakpoint removed');
    $('btnRun').click();
    await waitFor(() => text('state') === 'finished' && text('uNum') === '6', 'full run after removing it');
    $('btnReset').click();
    await waitFor(() => text('uNum') === '0' && text('state') === 'idle', 'second reset');
    click($('mNet'));
    await waitFor(() => document.querySelector('#network [data-core="31.0"]'), 'network chips');
    click(document.querySelector('#network [data-core="31.0"]'));
    await waitFor(() => document.querySelector('#zoom .ncell[data-neuron="layer_2.0"]'), 'IF2 core view');
    click(document.querySelector('#zoom .ncell[data-neuron="layer_2.0"]'));
    await waitFor(() => $('btnBreakFire'), 'neuron actions');
    $('btnBreakFire').click();
    await waitFor(() => /layer_2\.0 fires/.test($('bpList').textContent), 'fire breakpoint listed');
    $('btnRun').click();
    await waitFor(() => /^stopped: breakpoint .*layer_2\.0 fires$/.test(text('state')), 'stopped when layer_2.0 fired');
    const stoppedAt = (await (await fetch(BASE + 'api/sessions/' + sid0)).json()).update;
    if (stoppedAt !== firesAt) throw new Error('fire breakpoint stopped at ' + stoppedAt + ', expected ' + firesAt);
    await waitFor(() => text('uNum') === String(firesAt), 'page shows the stopping update');
    click(document.querySelector('#bpList li.bp button.bpDel'));
    await waitFor(() => document.querySelectorAll('#bpList li.bp').length === 0, 'fire breakpoint removed');

    click($('mChip'));
    await waitFor(() => shown('chip'), 'chip view for dragging');
    const at = (key) => document.querySelector('#chip rect.core[data-core="' + key + '"]');
    at('16.0').dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
    at('5.1').dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
    await waitFor(() => /16\.0/.test(text('editRail')) && /5\.1/.test(text('editRail')) && !$('btnApplyEdits').disabled, 'pending placement edit');
    // The chip previews the pending swap, so a second drag acts on what is shown.
    await waitFor(() => /used/.test(at('5.1').getAttribute('class')) && /empty/.test(at('16.0').getAttribute('class')) &&
      /pending/.test(at('5.1').getAttribute('class')), 'chip previews the pending edit');
    at('5.1').dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
    at('7.0').dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
    await waitFor(() => /used/.test(at('7.0').getAttribute('class')) && /empty/.test(at('5.1').getAttribute('class')), 'second drag moves the shown core');
    if (!/16\.0 → 7\.0/.test(text('editRail')) || /5\.1/.test(text('editRail'))) throw new Error('core map after two drags: ' + text('editRail'));
    at('7.0').dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
    at('5.1').dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
    await waitFor(() => /16\.0 → 5\.1/.test(text('editRail')), 'back to the single swap');
    $('btnApplyEdits').click();
    await waitFor(() => at('5.1') && /used/.test(at('5.1').getAttribute('class')) && !/pending/.test(at('5.1').getAttribute('class')) &&
      /empty/.test(at('16.0').getAttribute('class')) && $('btnApplyEdits').disabled && text('uNum') === '0', 'rebuilt with the swap');
    $('btnRun').click();
    await waitFor(() => text('state') === 'finished' && text('uNum') === '6', 'edited placement runs');
    await waitFor(() => document.querySelectorAll('#runsRail li.run').length >= 4, 'saved runs listed');
    document.querySelector('#tabs button[data-tab="compare"]').click();
    await waitFor(() => $('btnCompare'), 'compare controls');
    $('btnCompare').click();
    await waitFor(() => $('cmpSummary') && /spike trains identical/.test(text('cmpSummary')), 'placements compare identical')
      .catch((e) => { throw new Error(e.message + ': ' + ($('cmpSummary') ? text('cmpSummary') : 'no summary; result: ' + ($('cmpResult') ? text('cmpResult') : 'none') + '; dock: ' + text('dock').slice(0, 200))); });
    if (document.querySelectorAll('#dock tr.cmprow').length !== 6) throw new Error('compare table should have 6 rows');
    const edited = document.querySelector('#runsRail li.run.current');
    if (!edited || !/edited placement/.test(edited.textContent) || !/6 updates/.test(edited.textContent)) {
      throw new Error('the current run should be the edited one with 6 updates: ' + (edited ? edited.textContent : 'none'));
    }
    const link = edited.querySelector('a.export[data-kind="raster"]');
    const exported = await fetch(new URL(link.getAttribute('href'), BASE));
    const exportedBody = await exported.text();
    if (exported.status !== 200 || !/svg/.test(exportedBody)) throw new Error('raster export failed: ' + exported.status + ' ' + exportedBody.slice(0, 300) + ' ' + link.getAttribute('href'));

    $('params').querySelector('[data-param="steps"]').value = '3000';
    $('btnStart').click();
    await waitFor(() => text('uHorizon') === '3000', 'rebuilt session');
    document.querySelector('#tabs button[data-tab="watch"]').click();
    await waitFor(() => document.querySelectorAll('#dock svg.watchplot').length === 1, 'watch kept across a rebuild');
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
    // Aggregate trace level: link heat, link table, on-demand core state, late watch.
    q('horizon').value = '';
    q('params').querySelector('[data-param="steps"]').value = '6';
    q('traceLevel').value = 'aggregate';
    q('btnStart').click();
    await waitFor(() => q('uHorizon').textContent === '6' && q('uNum').textContent === '0', 'aggregate session');
    if (!/aggregate/i.test(q('chipNote').textContent)) throw new Error('chip note must say aggregate: ' + q('chipNote').textContent);
    q('btnRun').click();
    await waitFor(() => q('state').textContent === 'finished' && q('uNum').textContent === '6', 'aggregate run');
    await waitFor(() => page.window.document.querySelectorAll('#chip line.link.heat').length > 0, 'link heat on the chip');
    q('btnReplay').click();  // sample packets move on their recorded times
    await waitFor(() => page.window.document.querySelector('#chip circle.packet'), 'sample packets in flight', 30000);
    if (!/sample/.test(q('chipNote').textContent)) throw new Error('chip note must say packets are a sample');
    page.window.document.querySelector('#tabs button[data-tab="messages"]').click();
    await waitFor(() => page.window.document.querySelectorAll('#linkTable tr.linkrow').length > 0, 'per-link table');
    const pageClick = (node) => node.dispatchEvent(new page.window.MouseEvent('click', { bubbles: true }));
    const heated = [...page.window.document.querySelectorAll('#chip line.link.heat')];
    if (!heated.every((l) => l.querySelectorAll('title').length === 1)) throw new Error('link heat titles accumulate');
    pageClick(page.window.document.querySelector('#chip rect.tile[data-tile="0"]'));
    await waitFor(() => /tile 0/.test(q('insTitle').textContent), 'aggregate tile inspector');
    const tileRow = (label) => {
      const cell = [...page.window.document.querySelectorAll('#inspector .kv i')].find((i) => i.textContent.trim() === label);
      return cell ? cell.nextElementSibling.textContent : 'missing';
    };
    if (!(Number(tileRow('packets out')) > 0)) throw new Error('aggregate tile packets out: ' + tileRow('packets out'));
    pageClick(q('mNet'));
    // The drag edit is still applied, so find layer_1's core (4 neurons) by its chip.
    const layer1Chip = () => [...page.window.document.querySelectorAll('#network rect.corechip')]
      .find((r) => / 4 neurons/.test(r.textContent));
    await waitFor(layer1Chip, 'aggregate network chips');
    pageClick(layer1Chip());
    await waitFor(() => {
      const cells = [...page.window.document.querySelectorAll('#zoom .ncell i')];
      return cells.length === 4 && cells.some((i) => i.style.height && i.style.height !== '0%');
    }, 'core state fetched for the core view');
    await waitFor(() => /not kept/.test(q('axonIn').textContent), 'axon tables say per-link counts are not kept');
    pageClick(page.window.document.querySelector('#zoom .ncell[data-neuron="layer_1.2"]'));
    await waitFor(() => q('btnWatch'), 'aggregate neuron inspector');
    q('btnWatch').click();
    page.window.document.querySelector('#tabs button[data-tab="watch"]').click();
    await waitFor(() => {
      const line = page.window.document.querySelector('#dock svg.watchplot polyline.trace.t0');
      return line && line.getAttribute('points').trim().split(/\s+/).length === 6;
    }, 'late watch shows all 6 updates');
    if (errors.length) throw new Error('page errors: ' + errors.join('; '));
    console.log('studio smoke: OK (3 cores, 6 updates played, 24 bars, messages, inspector, zoom and mini-map, network mode, neuron watch, message route, architecture diff, breakpoints, neuron actions, placement drag, saved runs, compare, export, pause, X mark, reload resume, aggregate level)');
    page.window.close();
  } catch (error) {
    console.error('studio smoke: FAIL: ' + error.message);
    console.error(log.slice(-3000));
    process.exitCode = 1;
  } finally {
    // The page's WebSocket reconnect timer would keep Node alive, so exit
    // explicitly once the server has had a moment to shut its workers down.
    server.kill('SIGTERM');
    setTimeout(() => {
      fs.rmSync(STORE, { recursive: true, force: true });
      process.exit(process.exitCode || 0);
    }, 1500);
  }
})();
