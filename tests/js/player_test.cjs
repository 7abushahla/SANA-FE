/* Unit test of the player's ordering and de-duplication, without a browser.
 * A reconnecting page receives catch-up records that race the WebSocket, so
 * updates arrive twice and out of order. Run from the SANA-FE directory:
 *   node tests/js/player_test.cjs
 */
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');

const context = { Studio: {}, requestAnimationFrame: () => 0 };
context.window = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync('sanafe/studio/web/js/player.js', 'utf8'), context);

const record = (update) => ({ update: update, step_time: 1e-6, messages: [] });
const updates = (player) => Array.from(player.records, (r) => r.update);  // main-realm array

// Live stream, then a WebSocket update that races ahead of the catch-up fetch.
const live = new context.Studio.Player(() => {});
[1, 2, 3].forEach((u) => live.add(record(u)));
live.add(record(6));
[3, 4, 5, 6].forEach((u) => live.add(record(u)));
assert.deepStrictEqual(updates(live), [1, 2, 3, 4, 5, 6]);
assert.strictEqual(live.current().update, 1, 'playback keeps its place');

// The displayed update stays displayed when earlier records are inserted.
const viewer = new context.Studio.Player(() => {});
[1, 2, 6].forEach((u) => viewer.add(record(u)));
viewer.show(2);
[4, 5, 3].forEach((u) => viewer.add(record(u)));
assert.deepStrictEqual(updates(viewer), [1, 2, 3, 4, 5, 6]);
assert.strictEqual(viewer.current().update, 6, 'the shown record moves with the sort');

console.log('player test: OK');
