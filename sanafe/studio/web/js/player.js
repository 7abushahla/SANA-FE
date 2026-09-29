/* Plays received updates in order on the chosen clock.
   'modeled' keeps true proportions; 'slow' slows only while packets fly. */
(function () {
  const S = window.Studio;
  const SLOW_FACTOR = 10;

  function Player(onFrame) {
    this.onFrame = onFrame;
    this.clock = 'slow';
    this.duration = 3000;
    this.last = null;
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

  Player.prototype.tick = function (now) {
    const dt = this.last === null ? 0 : Math.min(100, now - this.last);
    this.last = now;
    const record = this.current();
    if (this.playing && record) {
      const inFlight = record.messages.some((m) => this.t >= m.send && this.t <= m.receive);
      let rate = record.step_time / this.duration;
      if (this.clock === 'slow' && inFlight) rate /= SLOW_FACTOR;
      this.t += dt * rate;
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
