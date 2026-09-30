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
  q('workload').value = 'sanafe-files';
  q('workload').dispatchEvent(new now.Event('change'));
  assert.strictEqual(q('platformHint').textContent, 'architecture from file');
  console.log('platform fallback test: OK');
  process.exit(0);
})().catch((error) => { console.error('platform fallback test: FAIL: ' + error.message); process.exit(1); });
