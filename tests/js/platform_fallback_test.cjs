/* The session rail against two servers: one that predates the platform
 * catalog (no /api/platforms route, workloads without platform lists) and a
 * current one. Loads the real page and scripts with a stubbed fetch.
 *   NODE_PATH=<dir>/node_modules node tests/js/platform_fallback_test.cjs
 */
const fs = require('fs');
const assert = require('assert');
const { JSDOM } = require('jsdom');

const WEB = 'sanafe/studio/web/';
const html = fs.readFileSync(WEB + 'index.html', 'utf8');
const scripts = [...html.matchAll(/<script src="([^"]+)"><\/script>/g)].map((m) => m[1]);

const param = { name: 'horizon', kind: 'int', default: 3, minimum: 1, maximum: null, choices: [], help: '' };

function page(routes) {
  const dom = new JSDOM(html.replace(/<script[^>]*><\/script>/g, ''), {
    runScripts: 'outside-only', pretendToBeVisual: true, url: 'http://127.0.0.1:1/' });
  const { window } = dom;
  window.fetch = async (url) => {
    const path = new URL(url, 'http://127.0.0.1:1/').pathname;
    const reply = routes[path];
    if (reply === undefined) return { ok: false, status: 404, statusText: 'Not Found', text: async () => 'Not Found' };
    return { ok: true, status: 200, statusText: 'OK', text: async () => JSON.stringify(reply) };
  };
  for (const src of scripts) window.eval(fs.readFileSync(WEB + src, 'utf8'));
  return window;
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 50));
const options = (window, id) => [...window.document.getElementById(id).options].map((o) => o.value);

(async () => {
  // A server started before the catalog: every workload stays usable.
  const old = page({ '/api/workloads': [
    { name: 'sanafe-files', parameters: [param] },
    { name: 'qcfs-compact', parameters: [param] }], '/api/runs': [] });
  await settle();
  const $ = (id) => old.document.getElementById(id);
  assert.deepStrictEqual(options(old, 'workload'), ['sanafe-files', 'qcfs-compact']);
  assert.strictEqual($('workload').value, 'qcfs-compact');
  assert.deepStrictEqual(options(old, 'platform'), []);
  assert.ok($('platform').disabled, 'platform dropdown disabled without a catalog');
  assert.match($('formError').textContent, /started before the platform catalog.*Restart it/);
  assert.strictEqual($('platformHint').textContent, '');
  assert.ok($('params').querySelector('[data-param="horizon"]'), 'parameter form rendered');

  // A current server: the dropdown filters the workloads.
  const now = page({
    '/api/workloads': [
      { name: 'qcfs-compact', platforms: ['loihi2'], parameters: [param] },
      { name: 'random-snn', platforms: ['loihi', 'loihi2'], parameters: [param] },
      { name: 'sanafe-files', platforms: [], parameters: [param] }],
    '/api/platforms': [{ id: 'loihi', title: 'Intel Loihi 1' }, { id: 'loihi2', title: 'Intel Loihi 2 candidate' }],
    '/api/runs': [] });
  await settle();
  const q = (id) => now.document.getElementById(id);
  assert.deepStrictEqual(options(now, 'platform'), ['loihi', 'loihi2']);
  assert.strictEqual(q('platform').value, 'loihi2');
  assert.ok(!q('platform').disabled);
  assert.strictEqual(q('formError').textContent, '');
  assert.deepStrictEqual(options(now, 'workload'), ['qcfs-compact', 'random-snn', 'sanafe-files']);
  q('platform').value = 'loihi';
  q('platform').dispatchEvent(new now.Event('change'));
  assert.deepStrictEqual(options(now, 'workload'), ['random-snn', 'sanafe-files']);
  assert.strictEqual(q('workload').value, 'random-snn');
  // Switching platform keeps what the user typed when the workload stays.
  q('params').querySelector('[data-param="horizon"]').value = '9';
  q('platform').value = 'loihi2';
  q('platform').dispatchEvent(new now.Event('change'));
  assert.strictEqual(q('workload').value, 'random-snn');
  assert.strictEqual(q('params').querySelector('[data-param="horizon"]').value, '9');
  q('workload').value = 'sanafe-files';
  q('workload').dispatchEvent(new now.Event('change'));
  assert.strictEqual(q('platformHint').textContent, 'architecture from file');

  // A preview platform: the chip picture from the catalog, no session, Start disabled.
  const preview = page({
    '/api/workloads': [{ name: 'random-snn', platforms: ['loihi'], parameters: [param] },
      { name: 'sanafe-files', platforms: [], parameters: [param] }],
    '/api/platforms': [{ id: 'loihi', title: 'Intel Loihi 1' },
      { id: 'speck', title: 'SynSense Speck (preview)', preview: true, badge: 'SynSense Speck (preview) · no engine · not hardware',
        preview_note: 'Speck preview: no execution engine yet.', not_hardware: 'n', vendor: 'SynSense', generation: 'g', yaml: 'speck.yaml',
        summary: 's', execution: 'event', time_rule: 'r', time_word: 'planned', energy_word: 'planned', validation: 'v',
        not_modeled: [], references: [], units: [], costs: [],
        structure: { width: 1, height: 1, tiles: 1, cores: 9, cores_per_tile: 9, max_neurons_per_core: 65536, buffer_position: 'b', sync_model: 'fixed', sync_table: { 0: 0 }, link_buffer_size: 1 },
        preview_layout: { blocks: [{ id: 'dvs', label: 'DVS 128 × 128' }, { id: 'preprocess', label: 'pre-processing' }, { id: 'noc', label: 'star NoC' }, { id: 'readout', label: 'readout' }],
          cores: Array.from({ length: 9 }, (_, i) => ({ id: i, kernel_words: 16384, neuron_words: 65536, leak_words: 1024 })) } }],
    '/api/runs': [] });
  await settle();
  const v = (id) => preview.document.getElementById(id);
  v('platform').value = 'speck';
  v('platform').dispatchEvent(new preview.Event('change'));
  await settle();
  assert.ok(v('btnStart').disabled, 'Start disabled on a preview platform');
  assert.match(v('platformHint').textContent, /no execution engine/);
  assert.strictEqual(v('badge').textContent, 'SynSense Speck (preview) · no engine · not hardware');
  assert.notStrictEqual(v('preview').style.display, 'none');
  assert.strictEqual(v('chip').style.display, 'none');
  assert.strictEqual(preview.document.querySelectorAll('#preview svg rect.core').length, 9);
  assert.match(v('preview').textContent, /DVS 128 × 128/);
  assert.match(v('preview').textContent, /no execution engine/);
  preview.document.querySelector('#tabs button[data-tab="platform"]').click();
  await settle();
  assert.match(v('dock').textContent, /SynSense Speck \(preview\)/);
  assert.match(v('dock').textContent, /Cores/);
  v('platform').value = 'loihi';
  v('platform').dispatchEvent(new preview.Event('change'));
  await settle();
  assert.ok(!v('btnStart').disabled, 'Start enabled again');
  assert.strictEqual(v('preview').style.display, 'none');
  assert.notStrictEqual(v('chip').style.display, 'none');
  assert.strictEqual(v('platformHint').textContent, '');

  // A platform whose card failed is listed but cannot be chosen, and the reason is shown.
  const broken = page({
    '/api/workloads': [{ name: 'random-snn', platforms: ['loihi', 'truenorth'], parameters: [param] }],
    '/api/platforms': [{ id: 'truenorth', title: 'IBM TrueNorth (functional)', error: "truenorth: cost attribute 'x' has no provenance entry" },
      { id: 'loihi', title: 'Intel Loihi 1' }],
    '/api/runs': [] });
  await settle();
  const b = (id) => broken.document.getElementById(id);
  const tn = [...b('platform').options].find((o) => o.value === 'truenorth');
  assert.ok(tn.disabled, 'broken platform option disabled');
  assert.match(tn.textContent, /unavailable/);
  assert.strictEqual(b('platform').value, 'loihi');
  assert.match(b('formError').textContent, /no provenance entry/);
  console.log('platform fallback test: OK');
  process.exit(0);
})().catch((error) => { console.error('platform fallback test: FAIL: ' + error.message); process.exit(1); });
