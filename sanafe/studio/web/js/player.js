/* Plays received updates in order on the chosen clock.
   'modeled' keeps true proportions; 'slow' slows only while packets fly. */
(function () {
  const S = window.Studio;
  const SLOW_SHARE = 0.6;  // slow motion: share of the playback spent on packets in flight

  /* Modeled time during which at least one packet is in flight. Chip-scale
     updates spend under 1% of their time that way, so a fixed slowdown makes
     packets flicker past; the share below keeps them visible at any scale. */
  function flightTime(record) {
    const spans = S.flying(record).map((m) => [m.send, m.receive]).sort((a, b) => a[0] - b[0]);
    let total = 0, start = null, end = null;
    for (const [from, to] of spans) {
      if (start === null) { start = from; end = to; }
      else if (from > end) { total += end - start; start = from; end = to; }
      else if (to > end) { end = to; }
    }
    if (start !== null) total += end - start;
    return Math.min(total, record.step_time);
  }

  function Player(onFrame) {
    this.onFrame = onFrame;
    this.clock = 'slow';
    this.duration = 3000;
    this.last = null;
    this.flight = new WeakMap();
    this.reset();
    const tick = (now) => { this.tick(now); requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  }

  Player.prototype.reset = function () {
    this.records = [];
    this.seen = new Set();
    this.index = -1;
    this.t = 0;
    this.playing = false;
    this.follow = true;
    this.dirty = true;
  };

  /* Records may arrive twice or out of order (WebSocket racing a catch-up
     fetch). Keep each update once, sorted. */
  Player.prototype.add = function (record) {
    if (this.seen.has(record.update)) return;
    this.seen.add(record.update);
    this.records.push(record);
    if (this.records.length > 1 && record.update < this.records[this.records.length - 2].update) {
      const shown = this.current();
      this.records.sort((a, b) => a.update - b.update);
      if (shown) this.index = this.records.indexOf(shown);
    }
    if (this.follow && !this.playing) this.advance();
    this.dirty = true;
  };

  Player.prototype.advance = function () {
    if (this.index < this.records.length - 1) {
      this.index += 1;
      this.t = 0;
      this.playing = true;
    } else {
      this.playing = false;
    }
  };

  Player.prototype.current = function () { return this.records[this.index] || null; };

  Player.prototype.show = function (index) {
    if (!this.records[index]) return;
    this.follow = false;
    this.index = index;
    this.t = this.records[index].step_time;
    this.playing = false;
    this.dirty = true;
  };

  Player.prototype.replay = function () {
    if (!this.current()) return;
    this.t = 0;
    this.playing = true;
    this.dirty = true;
  };

  Player.prototype.latest = function () {
    if (!this.records.length) return;
    this.follow = true;
    this.index = this.records.length - 1;
    this.t = this.current().step_time;
    this.playing = false;
    this.dirty = true;
  };

  /* Modeled seconds per millisecond of playback at the playhead. */
  Player.prototype.rate = function (record, inFlight) {
    if (this.duration <= 0) return Infinity;  // instant: no animation
    const modeled = record.step_time / this.duration;
    if (this.clock !== 'slow') return modeled;
    if (!this.flight.has(record)) this.flight.set(record, flightTime(record));
    const flight = this.flight.get(record);
    const rest = record.step_time - flight;
    if (flight <= 0 || rest <= 0) return modeled;  // nothing flying, or flight fills the update
    return inFlight ? flight / (this.duration * SLOW_SHARE)
                    : rest / (this.duration * (1 - SLOW_SHARE));
  };

  Player.prototype.tick = function (now) {
    const dt = this.last === null ? 0 : Math.min(100, now - this.last);
    this.last = now;
    const record = this.current();
    if (this.playing && record) {
      if (this.duration <= 0) {
        this.t = record.step_time;  // instant: skip straight to the end of the update
      } else {
        const inFlight = S.flying(record).some((m) => this.t >= m.send && this.t <= S.flightEnd(m, record, this.clock));
        this.t += dt * this.rate(record, inFlight);
      }
      if (this.t >= record.step_time) {
        this.t = record.step_time;
        if (this.follow && this.index < this.records.length - 1) this.advance();
        else this.playing = false;
      }
      this.dirty = true;
    }
    if (this.dirty) {
      this.dirty = false;
      this.onFrame();
    }
  };

  S.Player = Player;
})();
