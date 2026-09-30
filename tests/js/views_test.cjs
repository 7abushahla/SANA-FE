/* Unit tests of view details rendered in jsdom: time formatting, provenance
 * words after the mark in the inspector, the Platform card's first line and
 * warnings, and a selected route whose tiles are outside the drawn window.
 *   NODE_PATH=<dir>/node_modules node tests/js/views_test.cjs
 */
const assert = require('assert');
const fs = require('fs');
const { JSDOM } = require('jsdom');

const dom = new JSDOM('<!DOCTYPE html><body></body>', { runScripts: 'outside-only', pretendToBeVisual: true });
for (const file of ['util.js', 'inspector.js', 'platform.js', 'chip.js']) {
  dom.window.eval(fs.readFileSync('sanafe/studio/web/js/' + file, 'utf8'));
}
const S = dom.window.Studio;
const document = dom.window.document;
S.arch = { render() {} };  // the YAML view fetches from the server; not under test here

// Times: nanoseconds to seconds, never "1000.000 µs".
assert.strictEqual(S.fmtTime(150.4e-9), '150.4 ns');
assert.strictEqual(S.fmtTime(438.867e-6), '438.867 µs');
assert.strictEqual(S.fmtTime(1e-3), '1.000 ms');
assert.strictEqual(S.fmtTime(2.5), '2.500 s');
assert.strictEqual(S.fmtTime(0), '0.0 ns');

// Inspector: the platform's word follows the mark; values stay bare.
const session = { network: { occupied: ['0.0'], groups: [{ name: 'g', size: 8, cores: { '0.0': 8 } }], core_budgets: null },
  platform: { time_word: 'documented', energy_word: 'documented' } };
const record = { update: 2, step_time: 1e-3, last_activity: 0, barrier: 1e-3,
  counts: { messages: 1, hops: 0, fired: 1 }, energy: { total: 2.6e-11 }, reference: null,
  core_counts: { '0.0': { fired: 1, packets_in: 0, packets_out: 1, spikes_in: 0 } },
  core_energy: { '0.0': { total: 2.6e-11, units: { core_synapses: 2.6e-11 }, axon: null } },
  core_finish: { '0.0': 1e-3 }, tile_network_energy: {},
  messages: [{ mid: 7, src: '0.0', dst: '0.0', src_neuron: 'g.1', hops: 0, path: [], spikes: 1,
    generation_delay: 0, network_delay: 0, blocking_delay: 0, processing_delay: 0, send: 0, receive: 0, processed: 0 }],
  provenance: { step_time: 'R', last_activity: 'D', barrier: 'R', counts: 'R', energy: 'R', core_finish: 'R',
    core_counts: 'R', 'core_energy.total': 'R', 'core_energy.units': 'R', 'core_energy.axon': 'R' } };
const title = document.createElement('div');
const box = document.createElement('div');
const cellText = (label) => {
  const items = [...box.querySelectorAll('.kv > *')];
  const at = items.findIndex((el) => el.tagName === 'I' && el.textContent === label);
  assert.ok(at >= 0, 'row ' + label + ' missing: ' + box.textContent);
  return [items[at + 1].textContent, items[at + 2].textContent];
};
S.inspector.render(title, box, { kind: 'chip' }, record, session);
assert.deepStrictEqual(cellText('modeled step time'), ['1.000 ms', 'R · documented']);
assert.deepStrictEqual(cellText('barrier'), ['1.000 ms (100%)', 'R · documented']);
assert.deepStrictEqual(cellText('last core or message activity'), ['0.0 ns', 'D · documented']);
assert.deepStrictEqual(cellText('energy'), ['0.026 nJ', 'R · documented']);
assert.deepStrictEqual(cellText('messages'), ['1', 'R']);
S.inspector.render(title, box, { kind: 'core', key: '0.0' }, record, session);
assert.deepStrictEqual(cellText('energy'), ['0.026 nJ', 'R · documented']);
assert.deepStrictEqual(cellText('neuron processing ends'), ['1.000 ms', 'R · documented']);
S.inspector.render(title, box, { kind: 'message', mid: 7 }, record, session);
assert.deepStrictEqual(cellText('generation delay'), ['0.0 ns', 'R · documented']);
S.inspector.render(title, box, { kind: 'chip' }, record, Object.assign({}, session, { platform: null }));
assert.deepStrictEqual(cellText('modeled step time'), ['1.000 ms', 'R']);

// Platform card: the not-hardware line comes first; mismatches and file matches are said.
const card = { id: 'truenorth_documented', title: 'IBM TrueNorth (documented costs)', vendor: 'IBM', generation: '2014',
  yaml: 'truenorth_documented.yaml', not_hardware: 'Nothing on this card is a hardware measurement.', summary: 's',
  execution: 'timestep', time_rule: 'r', structure: { width: 64, height: 64, tiles: 4096, cores: 4096, cores_per_tile: 1,
    max_neurons_per_core: 256, buffer_position: 'before the soma unit', sync_model: 'fixed', sync_table: { 0: 1e-3 }, link_buffer_size: 1 },
  units: [], costs: [], validation: 'v', not_modeled: [], references: [], time_word: 'documented', energy_word: 'documented',
  architecture_matches: false, selected: true };
const dock = document.createElement('div');
S.platform.render(dock, { id: 's1', platform: card });
const node = dock.querySelector('#platformCard');
assert.ok(node.firstElementChild.classList.contains('nothw'), 'first line: ' + node.firstElementChild.outerHTML);
assert.match(node.querySelector('.modified').textContent, /modified/);
assert.match(node.textContent, /fixed · fixed: 1\.000 ms/);
S.platform.render(dock, { id: 's2', platform: Object.assign({}, card, { architecture_matches: true, selected: false }) });
assert.ok(!dock.querySelector('.modified'));
assert.match(dock.querySelector('.matchnote').textContent, /identical to this catalog platform/);

// Chip: a selected route whose tiles lie outside the drawn window must not throw.
const holder = document.createElement('div');
document.body.appendChild(holder);
const chip = new S.Chip(holder, () => {}, () => {}, {});
const tiles = [];
for (let x = 0; x < 20; x++) for (let y = 0; y < 20; y++) tiles.push({ id: x * 20 + y, x: x, y: y, cores: [{ key: (x * 20 + y) + '.0' }] });
chip.build({ width: 20, height: 20, tiles: tiles }, { groups: [{ name: 'g', size: 8, cores: { '0.0': 8 } }] }, {});
// The window note is page text above the picture, never SVG text that a
// two-tile window is too narrow to hold.
assert.match(holder.querySelector('.chipwindow').textContent, /Showing tiles x 0–1, y 0–1 of a 20 × 20 mesh/);
assert.ok(![...holder.querySelectorAll('svg text')].some((n) => /howing tiles/.test(n.textContent)), 'note must not be SVG text');
chip.route = 5;
const far = { update: 1, step_time: 1e-6, provenance: {}, core_finish: {},
  messages: [{ mid: 5, src: '399.0', dst: '398.0', path: [399, 398], send: 0, receive: 1e-9, processed: 2e-9 }] };
chip.render(far, 5e-7);
console.log('views test: OK');
