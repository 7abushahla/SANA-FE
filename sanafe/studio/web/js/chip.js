/* Tile-box chip: tiles hold a router and core slots; packets move on the
   recorded send and receive times along the reconstructed x-then-y route. */
(function () {
  const S = window.Studio;
  const PITCH = 100;
  const MARGIN = 60;

  function linkKey(a, b) { return a < b ? a + '-' + b : b + '-' + a; }

  function along(points, fraction) {
    const lengths = [];
    let total = 0;
    for (let i = 1; i < points.length; i++) {
      const length = Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
      lengths.push(length);
      total += length;
    }
    let remaining = Math.max(0, Math.min(1, fraction)) * total;
    for (let i = 0; i < lengths.length; i++) {
      if (remaining <= lengths[i]) {
        const g = lengths[i] ? remaining / lengths[i] : 0;
        return [points[i][0] + g * (points[i + 1][0] - points[i][0]),
                points[i][1] + g * (points[i + 1][1] - points[i][1])];
      }
      remaining -= lengths[i];
    }
    return points[points.length - 1];
  }

  function coreOffsets(count) {
    if (count === 1) return [[-18, -18]];
    const cols = Math.ceil(Math.sqrt(count));
    const rows = Math.ceil(count / cols);
    const offsets = [];
    for (let i = 0; i < count; i++) {
      const col = i % cols;
      const row = Math.floor(i / cols);
      offsets.push([(col - (cols - 1) / 2) * 36, (row - (rows - 1) / 2) * 36]);
    }
    return offsets;
  }

  /* options.mini draws a small, label-free copy for the mini-map; it takes
     no clicks of its own. onZoom receives { tile } or { core }. */
  function Chip(container, onSelect, onZoom, options) {
    this.container = container;
    this.onSelect = onSelect || function () {};
    this.onZoom = onZoom || function () {};
    this.mini = !!(options && options.mini);
    this.onDrag = (options && options.onDrag) || null;
    this.dragFrom = null;
    this.highlight = null;  // { in: [cores], out: [cores] }
    this.selected = null;
    this.focusTile = null;
    this.route = null;
    this.overlay = null;
    this.owner = {};
  }

  Chip.prototype.build = function (layout, network, metadata) {
    this.container.innerHTML = '';
    this.layout = layout;
    const hostWidth = !this.mini && metadata && metadata.host_operations ? 150 : 0;
    const width = hostWidth + 2 * MARGIN + layout.width * PITCH;
    const height = 2 * MARGIN + layout.height * PITCH;
    const svg = S.svg(this.container, 'svg', { viewBox: '0 0 ' + width + ' ' + height, role: 'img',
      'aria-label': this.mini ? 'chip mini-map' : 'chip mesh', class: this.mini ? 'mini' : '' });
    const left = hostWidth + MARGIN + PITCH / 2;
    const top = MARGIN + PITCH / 2;

    this.owner = {};
    this.groupsOf = {};
    network.groups.forEach((group, index) => {
      for (const core in group.cores) {
        const count = group.cores[core];
        if (!this.owner[core] || this.owner[core].count < count) {
          this.owner[core] = { group: group.name, count: count, color: S.PALETTE[index % S.PALETTE.length] };
        }
        (this.groupsOf[core] = this.groupsOf[core] || []).push(group.name + ' (' + count + ')');
      }
    });

    S.svg(svg, 'rect', { x: hostWidth + MARGIN / 2, y: MARGIN / 2, width: layout.width * PITCH + MARGIN, height: layout.height * PITCH + MARGIN, rx: 12, class: 'chipframe' });
    if (!this.mini) {
      svg.addEventListener('click', (event) => {
        const target = event.target;
        const core = target.getAttribute && target.getAttribute('data-core');
        const tile = target.getAttribute && target.getAttribute('data-tile');
        if (core) this.onSelect({ kind: 'core', key: core });
        else if (tile !== null && tile !== undefined) this.onZoom({ tile: Number(tile) });
        else this.onSelect({ kind: 'chip' });
      });
      // Press on a used core and release on another core: a placement swap.
      svg.addEventListener('mousedown', (event) => {
        const target = event.target;
        const cls = target.getAttribute && target.getAttribute('class');
        this.dragFrom = cls === 'core used' ? target.getAttribute('data-core') : null;
      });
      svg.addEventListener('mouseup', (event) => {
        const to = event.target.getAttribute && event.target.getAttribute('data-core');
        if (this.onDrag && this.dragFrom && to && to !== this.dragFrom) this.onDrag(this.dragFrom, to);
        this.dragFrom = null;
      });
      svg.addEventListener('dblclick', (event) => {
        const core = event.target.getAttribute && event.target.getAttribute('data-core');
        if (core) this.onZoom({ core: core });
      });
    }

    this.tileCenter = {};
    for (const tile of layout.tiles) {
      const cx = left + tile.x * PITCH;
      const cy = top + (layout.height - 1 - tile.y) * PITCH;
      this.tileCenter[tile.id] = [cx, cy];
      const box = S.svg(svg, 'rect', { x: cx - 44, y: cy - 44, width: 88, height: 88, rx: 8, class: 'tile', 'data-tile': tile.id });
      if (!this.mini) {
        S.svg(box, 'title', {}, 'tile ' + tile.id + ' · click to zoom in');
        S.svg(svg, 'text', { x: cx - 38, y: cy - 33, class: 'tilelabel' }, 'tile ' + tile.id);
      }
    }

    this.links = {};
    for (const tile of layout.tiles) {
      const neighbors = [];
      if (tile.x + 1 < layout.width) neighbors.push((tile.x + 1) * layout.height + tile.y);
      if (tile.y + 1 < layout.height) neighbors.push(tile.x * layout.height + tile.y + 1);
      for (const other of neighbors) {
        if (other >= layout.tiles.length) continue;
        const a = this.tileCenter[tile.id];
        const b = this.tileCenter[other];
        this.links[linkKey(tile.id, other)] = S.svg(svg, 'line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], class: 'link' });
      }
    }

    this.corePos = {};
    for (const tile of layout.tiles) {
      const center = this.tileCenter[tile.id];
      const offsets = coreOffsets(tile.cores.length);
      tile.cores.forEach((core, index) => {
        const x = center[0] + offsets[index][0];
        const y = center[1] + offsets[index][1];
        this.corePos[core.key] = [x, y];
        S.svg(svg, 'line', { x1: center[0], y1: center[1], x2: x, y2: y, class: 'spoke' });
        const owner = this.owner[core.key];
        const rect = S.svg(svg, 'rect', { x: x - 11, y: y - 11, width: 22, height: 22, rx: 3, class: owner ? 'core used' : 'core empty', 'data-core': core.key });
        if (owner) rect.setAttribute('fill', owner.color);
        const where = 'tile ' + tile.id + ', core ' + core.key.split('.')[1];
        S.svg(rect, 'title', {}, where + (owner ? ' · ' + this.groupsOf[core.key].join(', ') : ' · empty') +
          ' · click to inspect, double-click to open');
      });
      S.svg(svg, 'circle', { cx: center[0], cy: center[1], r: 6.5, class: 'router' });
    }

    network.groups.forEach((group, index) => {
      const cores = Object.keys(group.cores);
      if (this.mini || !cores.length || !this.corePos[cores[0]]) return;
      const at = this.corePos[cores[0]];
      S.svg(svg, 'text', { x: at[0], y: at[1] - 20, 'text-anchor': 'middle', class: 'grouplabel', fill: S.PALETTE[index % S.PALETTE.length] },
        group.name + ' · ' + cores.length + (cores.length === 1 ? ' core' : ' cores'));
    });

    if (hostWidth) this.drawHost(svg, metadata, network, height);
    this.overlay = S.svg(svg, 'g', {});
  };

  Chip.prototype.drawHost = function (svg, metadata, network, height) {
    const y = height / 2;
    S.svg(svg, 'rect', { x: 10, y: y - 50, width: 130, height: 100, rx: 8, class: 'host' });
    S.svg(svg, 'text', { x: 75, y: y - 30, 'text-anchor': 'middle', class: 'hosttext', 'font-weight': 700 }, 'HOST');
    S.svg(svg, 'text', { x: 75, y: y - 17, 'text-anchor': 'middle', class: 'hosttext' }, '(not simulated)');
    metadata.host_operations.forEach((line, i) => S.svg(svg, 'text', { x: 18, y: y + 2 + i * 14, class: 'hosttext' }, line));
    const group = network.groups.find((g) => g.name === metadata.host_input_group);
    if (!group) return;
    for (const core in group.cores) {
      const at = this.corePos[core];
      S.svg(svg, 'path', { d: 'M140,' + y + ' L' + (at[0] - 12) + ',' + at[1], class: 'hostlink' });
    }
    S.svg(svg, 'text', { x: 146, y: y + 16, class: 'hosttext', fill: '#2368a0' }, 'constant current (not NoC)');
  };

  Chip.prototype.render = function (record, t) {
    if (!this.overlay) return;
    const overlay = this.overlay;
    while (overlay.firstChild) overlay.removeChild(overlay.firstChild);
    for (const key in this.links) this.links[key].setAttribute('class', 'link');
    if (record) {
      for (const message of record.messages) {
        if (t >= message.send && t <= message.receive) {
          const points = [this.corePos[message.src]]
            .concat(message.path.map((tile) => this.tileCenter[tile]), [this.corePos[message.dst]]);
          const span = message.receive - message.send;
          const at = along(points, span > 0 ? (t - message.send) / span : 1);
          S.svg(overlay, 'circle', { cx: at[0], cy: at[1], r: 4.5, class: 'packet' });
          for (let i = 1; i < message.path.length; i++) {
            const link = this.links[linkKey(message.path[i - 1], message.path[i])];
            if (link) link.setAttribute('class', 'link hot');
          }
        } else if (t > message.receive && t <= message.processed && this.corePos[message.dst]) {
          const at = this.corePos[message.dst];
          S.svg(overlay, 'rect', { x: at[0] - 14, y: at[1] - 14, width: 28, height: 28, rx: 5, class: 'receiving' });
        }
      }
      for (const core in record.core_finish) {
        if (t < record.core_finish[core] && this.corePos[core]) {
          const at = this.corePos[core];
          S.svg(overlay, 'circle', { cx: at[0] + 9, cy: at[1] - 9, r: 3, class: 'busy' });
        }
      }
    }
    if (this.route && record) {
      const message = record.messages.find((m) => m.mid === this.route);
      if (message) {
        const points = [this.corePos[message.src]]
          .concat(message.path.map((tile) => this.tileCenter[tile]), [this.corePos[message.dst]]);
        S.svg(overlay, 'polyline', { points: points.map((p) => p.join(',')).join(' '), class: 'route' });
        const end = points[points.length - 1];
        S.svg(overlay, 'text', { x: end[0] + 14, y: end[1] - 12, class: 'routemark' }, 'X route');
      }
    }
    if (this.highlight) {
      for (const [side, keys] of [['in', this.highlight.in], ['out', this.highlight.out]]) {
        for (const key of keys) {
          const at = this.corePos[key];
          if (at) S.svg(overlay, 'rect', { x: at[0] - 15, y: at[1] - 15, width: 30, height: 30, rx: 6, class: 'hl-' + side, 'data-core': key });
        }
      }
    }
    if (this.focusTile !== null && this.tileCenter[this.focusTile]) {
      const at = this.tileCenter[this.focusTile];
      S.svg(overlay, 'rect', { x: at[0] - 48, y: at[1] - 48, width: 96, height: 96, rx: 10, class: 'focus' });
    }
    if (this.selected && this.corePos[this.selected]) {
      const at = this.corePos[this.selected];
      S.svg(overlay, 'rect', { x: at[0] - 16, y: at[1] - 16, width: 32, height: 32, rx: 6, class: 'selected' });
    }
  };

  S.Chip = Chip;
})();
