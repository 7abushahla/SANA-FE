/* Unit test of the Pipeline view's honesty rules, rendered in jsdom:
 * a run that ends before the output window says so, cells past the horizon
 * are marked beyond it, and a zero barrier draws no barrier. Run from the
 * SANA-FE directory:
 *   NODE_PATH=<dir>/node_modules node tests/js/pipeline_test.cjs
 */
const assert = require('assert');
const fs = require('fs');
const { JSDOM } = require('jsdom');

const dom = new JSDOM('<!DOCTYPE html><body></body>', { runScripts: 'outside-only' });
for (const file of ['util.js', 'pipeline.js']) dom.window.eval(fs.readFileSync('sanafe/studio/web/js/' + file, 'utf8'));
const S = dom.window.Studio;
const document = dom.window.document;

const pipeline = {
  T: 2, output_group: 'b', classes: ['x', 'y'], fixture: false,
  rows: [{ group: 'a', depth: 0, window: [0, 2] }, { group: 'b', depth: 1, window: [1, 3] }],
  input: { index: 0, label: 0, pixels: Buffer.alloc(3072).toString('base64'), drive: 'drive', currents: 4, group: 'a' },
};
const session = (horizon) => ({ horizon, metadata: { pipeline },
  network: { groups: [{ name: 'a', size: 4, cores: {} }, { name: 'b', size: 2, cores: {} }] } });
const record = { update: 1, step_time: 1e-6, barrier: 0, group_fired: { a: 2, b: 0 }, reference: null,
  readout: { step: null, scores: null, cumulative: null, predicted: null, reference: { status: 'waiting' }, quantity: 'spike counts' } };
const handlers = { pick() {}, open() {}, redraw() {} };
const render = (horizon, axis) => {
  const box = document.createElement('div');
  document.body.appendChild(box);
  S.pipeline.axis = axis;
  S.pipeline.render(box, session(horizon), [record], 0, handlers);
  return box;
};

// The run ends (horizon 1) before b's window (updates 2-3): no readout, and say so.
let box = render(1, 'updates');
const output = box.querySelector('#pOutput').textContent;
assert.match(output, /ends at update 1, before the output window \(updates 2–3\)/, output);
assert.doesNotMatch(output, /first fires/, 'the window opening is not a claim that the site fires');
// A zero barrier draws no red segment.
assert.strictEqual(box.querySelector('.pbar b').style.height, '0%', 'no barrier, no red bar');

// Timestep axis: b at t=0 is update 2, beyond a horizon of 1, not "not computed yet".
box = render(1, 'steps');
const beyond = box.querySelector('td.pcell[data-group="b"][data-update="2"]');
assert.ok(beyond.classList.contains('beyond'), 'cells past the horizon are marked beyond it');
assert.match(beyond.title, /beyond the horizon/);

// A horizon that reaches the window: the panel says when it opens.
box = render(3, 'updates');
assert.match(box.querySelector('#pOutput').textContent, /output window opens at update 2/);

console.log('pipeline test: OK');
