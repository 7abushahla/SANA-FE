/* Tile and core views: the pipeline at the playhead, neurons, and axon tables.
   Counters are derived from the update's recorded messages and finish times. */
(function () {
  const S = window.Studio;
  const MAX_CELLS = 1024;

  function counters(record, key, t) {
    const c = { pin: 0, sin: 0, pout: 0, fired: 0, busy: false, done: true, mapped: false };
    if (!record) return c;
    if (S.isAggregate(record)) {  // no messages kept: totals for the whole update
      const totals = record.core_counts[key];
      if (totals) { c.pin = totals.packets_in; c.sin = totals.spikes_in; c.pout = totals.packets_out; }
    }
    for (const m of record.messages) {
      if (m.dst === key && m.receive <= t) { c.pin += 1; c.sin += m.spikes; }
      if (m.src === key && m.send <= t) c.pout += 1;
    }
    const finish = record.core_finish[key];
    c.mapped = finish !== undefined;
    c.busy = c.mapped && t < finish;
    c.done = !c.busy;
    const counts = record.core_counts[key];
    c.fired = counts ? counts.fired : 0;
    return c;
  }

  /* SANA-FE order: axon in, synapse, dendrite, soma, axon out, with the
     update-boundary buffer before the unit the architecture names. */
  function stages(core) {
    const rank = { synapse: 1, dendrite: 2, soma: 3 };
    const units = core.pipeline.slice().sort((a, b) => (rank[a.role.split('+')[0]] || 9) - (rank[b.role.split('+')[0]] || 9));
    const out = [{ role: 'axon_in', name: core.axon_in.join(', ') || 'axon in' }];
    let buffered = false;
    for (const unit of units) {
      if (!buffered && core.buffer && unit.role.split('+').indexOf(core.buffer) >= 0) { out.push({ role: 'buffer' }); buffered = true; }
      out.push(unit);
    }
    if (!buffered && core.buffer === 'axon_out') out.push({ role: 'buffer' });
    out.push({ role: 'axon_out', name: core.axon_out.join(', ') || 'axon out' });
    return out;
  }

  function stageText(stage, c) {
    const roles = stage.role.split('+');
    if (stage.role === 'axon_in') return c.pin + ' packets in';
    if (stage.role === 'axon_out') return c.pout + ' packets out';
    if (roles.indexOf('soma') >= 0) {
      if (!c.mapped) return 'idle';
      if (c.busy) return 'updating…';
      return 'done · ' + (c.fired === null ? '?' : c.fired) + ' fired';
    }
    if (roles.indexOf('synapse') >= 0) return c.sin + ' spikes in';
    return 'accumulate';
  }

  function buildStrip(parent, core) {
    const strip = S.html(parent, 'div', { class: 'pipe' });
    const cells = [];
    stages(core).forEach((stage, index) => {
      if (index) S.html(strip, 'span', { class: 'arr' }, '→');
      if (stage.role === 'buffer') {
        const buf = S.html(strip, 'div', { class: 'buf', title: 'values wait here for the next update' });
        buf.innerHTML = 'buffer<br><small>update boundary</small>';
        return;
      }
      const unit = S.html(strip, 'div', { class: 'unit ' + stage.role.split('+')[0] });
      S.html(unit, 'b', {}, stage.role.replace('_', ' '));
      S.html(unit, 'i', {}, stage.name);
      cells.push({ stage: stage, node: S.html(unit, 'span', {}, '') });
    });
    return cells;
  }

  function Zoom(container, handlers) {
    this.container = container;
    this.handlers = handlers;  // openCore(key), selectNeuron(key)
    this.level = null;
  }

  Zoom.prototype.show = function (ctx, level, key) {
    this.ctx = ctx;
    this.level = level;
    this.key = key;
    this.container.innerHTML = '';
    if (level === 'tile') this.buildTile(ctx.layout.tiles[key]);
    else this.buildCore(key);
  };

  Zoom.prototype.buildTile = function (tile) {
    const grid = S.html(this.container, 'div', { class: 'tilegrid' });
    this.panels = tile.cores.map((core) => {
      const panel = S.html(grid, 'div', { class: 'corepanel', 'data-core': core.key });
      const groups = this.ctx.network.groups.filter((g) => g.cores[core.key]).map((g) => g.name + ' (' + g.cores[core.key] + ')');
      S.html(panel, 'div', { class: 'title' }, 'core ' + core.key + ' · ' + (groups.length ? groups.join(', ') : 'empty'));
      panel.addEventListener('click', () => this.handlers.openCore(core.key));
      if (!groups.length) {
        S.html(panel, 'div', { class: 'small muted' }, 'No neurons mapped. It still takes part in the barrier.');
        return { key: core.key, cells: [] };
      }
      const cells = buildStrip(panel, core);
      const note = S.html(panel, 'div', { class: 'small muted' });
      note.innerHTML = 'counters at the playhead ' + S.mark('D') + ' · click for the full core view';
      return { key: core.key, cells: cells };
    });
  };

  Zoom.prototype.buildCore = function (key) {
    const ctx = this.ctx;
    const core = ctx.layoutCore(key);
    const ranges = ctx.network.core_neurons[key] || [];
    const box = this.container;
    const head = S.html(box, 'div', { class: 'title' });
    head.innerHTML = 'pipeline this update, SANA-FE order · counters at the playhead ' + S.mark('D') +
      ' (whole-update totals at the aggregate trace level)';
    this.panels = [{ key: key, cells: buildStrip(box, core) }];
    const cols = S.html(box, 'div', { class: 'corecols' });
    const left = S.html(cols, 'div', { class: 'pane' });
    S.html(left, 'div', { class: 'title' }, 'axon_in (from)');
    this.axonIn = S.html(left, 'div', { id: 'axonIn' });
    const middle = S.html(cols, 'div', { class: 'pane' });
    this.neuronTitle = S.html(middle, 'div', { class: 'title' });
    this.grid = S.html(middle, 'div', { class: 'ngrid' });
    const right = S.html(cols, 'div', { class: 'pane' });
    S.html(right, 'div', { class: 'title' }, 'axon_out (to)');
    this.axonOut = S.html(right, 'div', { id: 'axonOut' });

    const neurons = [];
    for (const [group, first, last] of ranges) for (let o = first; o <= last; o++) neurons.push(group + '.' + o);
    this.neurons = neurons;
    const bins = neurons.length > MAX_CELLS ? MAX_CELLS : neurons.length;
    this.binned = neurons.length > MAX_CELLS;
    this.cells = [];
    for (let i = 0; i < bins; i++) {
      const from = Math.floor(i * neurons.length / bins);
      const to = Math.floor((i + 1) * neurons.length / bins);
      const members = neurons.slice(from, to);
      const cell = S.html(this.grid, 'div', { class: 'ncell', 'data-neuron': members[0], title: this.binned ? members[0] + ' … ' + members[members.length - 1] : members[0] });
      const fill = S.html(cell, 'i');
      cell.addEventListener('click', () => this.handlers.selectNeuron(members[0]));
      this.cells.push({ members: members, cell: cell, fill: fill });
    }
    if (!neurons.length) this.grid.textContent = 'Empty core. No neurons, synapses, or routes are mapped here.';
    const where = ranges.map((r) => r[0] + ' ' + r[1] + '–' + r[2]).join(', ');
    this.neuronTitle.innerHTML = S.escape('neurons ' + (where || 'none') + (this.binned ? ' · heatmap, mean per bin' : '')) +
      ' · fill = membrane at the end of the update ÷ threshold ' + S.mark('R') + ' · red = fired';

    this.inLinks = ctx.network.core_links.filter((l) => l.dst === key);
    this.outLinks = ctx.network.core_links.filter((l) => l.src === key);
    const hostGroup = ctx.metadata && ctx.metadata.host_input_group;
    this.hostDriven = !!hostGroup && ranges.some((r) => r[0] === hostGroup);
  };

  function linkRows(node, links, record, t, direction, key, owners) {
    const rows = links.map((link) => {
      const other = direction === 'in' ? link.src : link.dst;
      let now = 0, total = 0;
      if (record) {
        for (const m of record.messages) {
          const match = direction === 'in' ? (m.src === other && m.dst === key) : (m.src === key && m.dst === other);
          if (!match) continue;
          total += 1;
          if (direction === 'in' ? m.receive <= t : m.send <= t) now += 1;
        }
      }
      return '<tr><td>' + S.escape(other) + ' <span class="muted">' + S.escape(owners(other)) + '</span></td><td>' +
        link.synapses + ' syn · ' + link.axons + ' axons</td><td><b>' + now + '</b>/' + total + '</td></tr>';
    });
    return rows.join('');
  }

  Zoom.prototype.update = function (record, t) {
    for (const panel of this.panels || []) {
      const c = counters(record, panel.key, t);
      for (const cell of panel.cells) cell.node.textContent = stageText(cell.stage, c);
    }
    if (this.level !== 'core') return;
    const key = this.key;
    const owners = (core) => this.ctx.network.groups.filter((g) => g.cores[core]).map((g) => g.name).join(', ');
    if (S.isAggregate(record)) {  // packets between a pair of cores are not kept
      const summary = (links, label) => links.length
        ? '<table><tr><th>' + label + '</th><th>mapped</th></tr>' + links.map((link) => {
          const other = label === 'source core' ? link.src : link.dst;
          return '<tr><td>' + S.escape(other) + ' <span class="muted">' + S.escape(owners(other)) + '</span></td><td>' +
            link.synapses + ' syn · ' + link.axons + ' axons</td></tr>';
        }).join('') + '</table>' : '';
      const note = '<div class="small muted">connections ' + S.mark('R') + ' · packets per core pair: not kept (aggregate); ' +
        'this core\'s totals are in the pipeline strip ' + S.mark('D') + '</div>';
      this.axonIn.innerHTML = (this.hostDriven ? '<div class="small">host: constant current each update (not packets, not simulated)</div>' : '') +
        (summary(this.inLinks, 'source core') || (this.hostDriven ? '' : 'no incoming connections')) + note;
      this.axonOut.innerHTML = (summary(this.outLinks, 'destination core') || 'no outgoing connections') + note;
    }
    let inside = '';
    if (this.hostDriven) inside += '<tr><td colspan="3">host: constant current each update (not packets, not simulated)</td></tr>';
    if (!S.isAggregate(record)) {
      inside += linkRows(this.axonIn, this.inLinks, record, t, 'in', key, owners);
      this.axonIn.innerHTML = inside ? '<table><tr><th>source core</th><th>mapped</th><th>packets</th></tr>' + inside + '</table>' +
        '<div class="small muted">connections ' + S.mark('R') + ' · packets arrived by the playhead ' + S.mark('D') + '</div>' : 'no incoming connections';
      const outside = linkRows(this.axonOut, this.outLinks, record, t, 'out', key, owners);
      this.axonOut.innerHTML = outside ? '<table><tr><th>destination core</th><th>mapped</th><th>packets</th></tr>' + outside + '</table>' +
        '<div class="small muted">connections ' + S.mark('R') + ' · packets sent by the playhead ' + S.mark('D') + '</div>' : 'no outgoing connections; spikes are read from the trace';
    }

    const c = counters(record, key, t);
    let potentials = record ? record.potentials : {};
    let fired = new Set(record ? record.fired.map((f) => f[0] + '.' + f[1]) : []);
    if (S.isAggregate(record)) {
      // Membranes of this core are fetched from the worker for the shown update.
      const state = this.handlers.coreState(key, record.update);
      potentials = {};
      fired = new Set(state ? state.fired : []);
      if (state) state.neurons.forEach((n, i) => { if (state.potentials[i] !== null) potentials[n] = state.potentials[i]; });
    }
    const thresholds = this.ctx.network.group_attributes || {};
    let scale = 0;
    if (record) for (const n of this.neurons) scale = Math.max(scale, Math.abs(potentials[n] || 0));
    for (const cell of this.cells) {
      let sum = 0, count = 0, spiked = false;
      for (const n of cell.members) {
        const v = potentials[n];
        if (typeof v === 'number') {
          const group = n.slice(0, n.lastIndexOf('.'));
          const theta = thresholds[group] && thresholds[group].threshold;
          sum += theta ? v / theta : (scale ? Math.abs(v) / scale : 0);
          count += 1;
        }
        if (fired.has(n)) spiked = true;
      }
      const level = count ? Math.max(0, Math.min(1, sum / count)) : 0;
      cell.fill.style.height = Math.round(level * 100) + '%';
      const loading = S.isAggregate(record) && !this.handlers.coreState(key, record.update);
      cell.cell.className = 'ncell' + (spiked && c.done ? ' fired' : '') + (count || loading ? '' : ' unlogged');
    }
  };

  S.Zoom = Zoom;
})();
