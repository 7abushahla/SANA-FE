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
vm.runInContext(fs.readFileSync('sanafe/studio/web/js/util.js', 'utf8'), context);
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

// Pause freezes the playhead: time stands still and arriving updates wait.
const paused = new context.Studio.Player(() => {});
paused.clock = 'modeled';
paused.duration = 1000;  // 1e-6 s of modeled time per second of playback
[1, 2].forEach((u) => paused.add(record(u)));
paused.tick(0); paused.tick(100);
const before = paused.t;
assert.ok(before > 0 && paused.busy(), 'playing before the pause');
paused.pause();
paused.tick(200); paused.tick(300);
paused.add(record(3));
assert.strictEqual(paused.t, before, 'the playhead does not move while paused');
assert.strictEqual(paused.current().update, 1, 'no advance while paused');
assert.ok(paused.paused && paused.busy(), 'a paused player still has work to show');
paused.resume();
paused.tick(400); paused.tick(500);
assert.ok(paused.t > before && paused.current().update === 1, 'resume continues from the same modeled time');
for (let now = 600; now <= 1800; now += 50) paused.tick(now);  // frames are capped at 100 ms
assert.strictEqual(paused.current().update, 2, 'and then moves on to the next update');

// Stop freezes on the current frame for good: no resume, no following.
const stopped = new context.Studio.Player(() => {});
[1, 2, 3].forEach((u) => stopped.add(record(u)));
stopped.tick(0); stopped.tick(100);
const frozen = stopped.t;
stopped.stop();
stopped.tick(200); stopped.tick(300);
stopped.add(record(4));
assert.strictEqual(stopped.t, frozen, 'stop freezes the playhead');
assert.strictEqual(stopped.current().update, 1, 'stop keeps the current update on screen');
assert.ok(!stopped.busy() && !stopped.paused && !stopped.follow, 'nothing left to resume after a stop');
assert.deepStrictEqual(updates(stopped), [1, 2, 3, 4], 'every computed update stays reachable');

console.log('player test: OK');
