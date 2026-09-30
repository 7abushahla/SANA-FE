/* Network mode: groups, their connections, host operations, and each
   group's cores (SANA-FE Fig. 3 style). Everything here is recorded (R).
   Groups run top to bottom in dependency order, one per row, the way deep
   networks are usually drawn: consecutive groups are joined by a short
   arrow, and skip connections (ResNet shortcuts) arc through side lanes. */
(function () {
  const S = window.Studio;
  const BOX_W = 250, GAP_V = 34, TOP = 16, CHIP_W = 34, CHIP_STEP = 37, LANE = 22;
  const HOST_CHARS = 36;  // 10px host text inside a 250-wide box

  /* Groups in dependency order; ties and cycles keep trace order. */
  function order(groups, edges) {
    const names = groups.map((g) => g.name);
    const incoming = {};
    names.forEach((n) => { incoming[n] = 0; });
    edges.forEach((e) => { if (e.src !== e.dst) incoming[e.dst] += 1; });
    const out = [];
    const ready = names.filter((n) => !incoming[n]);
    const seen = new Set();
    while (ready.length) {
      const n = ready.shift();
      if (seen.has(n)) continue;
      seen.add(n);
      out.push(n);
      edges.filter((e) => e.src === n && e.dst !== n).forEach((e) => {
        incoming[e.dst] -= 1;
        if (!incoming[e.dst]) ready.push(e.dst);
      });
    }
    return out.concat(names.filter((n) => !seen.has(n)));
  }

  /* Lanes for skip arcs: a span takes the first lane free by its start. */
  function lanes(spans) {
    const ends = [];
    return spans.map(([a, b]) => {
      const lo = Math.min(a, b), hi = Math.max(a, b);
      let lane = ends.findIndex((end) => end <= lo);
      if (lane < 0) { lane = ends.length; ends.push(hi); } else ends[lane] = hi;
      return lane;
    });
  }

  S.network = {
    render(container, session, handlers) {
      container.innerHTML = '';
      const network = session.network;
      const metadata = session.metadata || {};
      const hostOps = metadata.host_operations || [];
      const names = order(network.groups, network.connections);
      const rows = [];
      if (hostOps.length && metadata.host_input_group) rows.push({ host: hostOps[0] });
      names.forEach((name) => rows.push({ group: network.groups.find((g) => g.name === name) }));
      hostOps.slice(metadata.host_input_group ? 1 : 0).forEach((op) => rows.push({ host: op }));
      const index = {};
      rows.forEach((row, i) => { if (row.group) index[row.group.name] = i; });

      // Names that do not fit on one line wrap at their dots; boxes share one height.
      const nameLines = (name) => (S.textWidth(name, 14) * 1.1 <= BOX_W - 16 ? [name]
        : S.wrap(name.replace(/\./g, '. '), 22).map((line) => line.replace(/\. /g, '.')));
      const tallest = Math.max(1, ...rows.filter((r) => r.group).map((r) => nameLines(r.group.name).length));
      const BOX_H = 30 + 16 * tallest;
      const pitch = BOX_H + GAP_V;

      const skips = network.connections.filter((e) => index[e.src] !== undefined && index[e.dst] !== undefined &&
        index[e.dst] !== index[e.src] + 1);
      const laneOf = lanes(skips.map((e) => [index[e.src], index[e.dst]]));
      const laneCount = Math.max(0, ...laneOf.map((l) => l + 1));
      const LEFT = 96 + laneCount * LANE;  // skip arcs and their labels live left of the boxes
      const chipsX = LEFT + BOX_W + 18;
      const width = Math.max(container.clientWidth || 1000, chipsX + 4 * CHIP_STEP + 10);
      const perLine = Math.max(4, Math.floor((width - chipsX - 10) / CHIP_STEP));
      const chipLines = Math.max(1, Math.floor((BOX_H + 4) / 24));
      const height = TOP * 2 + rows.length * pitch - GAP_V;
      const svg = S.svg(container, 'svg', { viewBox: '0 0 ' + width + ' ' + height, class: 'netview' });
      const defs = S.svg(svg, 'defs', {});
      const marker = S.svg(defs, 'marker', { id: 'netArrow', viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: 'auto' });
      S.svg(marker, 'path', { d: 'M0,0 L8,4 L0,8 z', fill: '#607888' });
      const top = (i) => TOP + i * pitch;
      const mid = (i) => top(i) + BOX_H / 2;
      const cx = LEFT + BOX_W / 2;

      rows.forEach((row, i) => {
        const y = top(i);
        if (row.host) {
          const g = S.svg(svg, 'g', { class: 'host' });
          S.svg(g, 'rect', { x: LEFT, y: y, width: BOX_W, height: BOX_H, rx: 8, class: 'host' });
          const lines = S.wrap(row.host, HOST_CHARS);
          const first = y + BOX_H / 2 - (lines.length * 13) / 2 + 4;
          S.svg(g, 'text', { x: cx, y: first, 'text-anchor': 'middle', class: 'hosttext', 'font-weight': 700 }, 'HOST (not simulated)');
          lines.forEach((line, n) => S.svg(g, 'text', { x: cx, y: first + 15 + 13 * n, 'text-anchor': 'middle', class: 'hosttext' }, line));
          return;
        }
        const group = row.group;
        const color = S.PALETTE[network.groups.indexOf(group) % S.PALETTE.length];
        const g = S.svg(svg, 'g', { class: 'group', 'data-group': group.name });
        const box = S.svg(g, 'rect', { x: LEFT, y: y, width: BOX_W, height: BOX_H, rx: 8, fill: color, class: 'groupbox' });
        S.svg(box, 'title', {}, 'click to inspect ' + group.name);
        const lines = nameLines(group.name);
        const nameTop = y + BOX_H / 2 - (lines.length * 16 + 14) / 2 + 13;
        lines.forEach((line, n) => S.svg(g, 'text', { x: cx, y: nameTop + 16 * n, 'text-anchor': 'middle', class: 'groupname' }, line));
        const theta = (network.group_attributes[group.name] || {}).threshold;
        S.svg(g, 'text', { x: cx, y: nameTop + 16 * lines.length + 2, 'text-anchor': 'middle', class: 'groupmeta' },
          group.size + ' neurons' + (theta ? ' · θ ' + theta : ''));
        g.addEventListener('click', () => handlers.selectGroup(group.name));
        // Core chips beside the box; the last slot says how many more there are.
        const cores = Object.keys(group.cores);
        const room = perLine * chipLines;
        const shown = cores.length > room ? room - 1 : cores.length;
        const chipTop = y + BOX_H / 2 - (Math.min(chipLines, Math.ceil((shown + (cores.length > room ? 1 : 0)) / perLine)) * 24 - 5) / 2;
        cores.slice(0, shown).forEach((core, j) => {
          const x = chipsX + (j % perLine) * CHIP_STEP;
          const cy = chipTop + Math.floor(j / perLine) * 24;
          const chip = S.svg(svg, 'rect', { x: x, y: cy, width: CHIP_W, height: 19, rx: 3, class: 'corechip', stroke: color, 'data-core': core });
          S.svg(chip, 'title', {}, 'core ' + core + ' · ' + group.cores[core] + ' neurons · click to open');
          S.svg(svg, 'text', { x: x + CHIP_W / 2, y: cy + 13, 'text-anchor': 'middle', class: 'corechiptext', fill: color, 'pointer-events': 'none' }, core);
          chip.addEventListener('click', () => handlers.openCore(core));
        });
        if (cores.length > room) {
          const x = chipsX + (shown % perLine) * CHIP_STEP, cy = chipTop + Math.floor(shown / perLine) * 24;
          S.svg(svg, 'text', { x: x + 2, y: cy + 13, class: 'hosttext' }, '+' + (cores.length - shown) + ' more');
        }
      });

      // Consecutive groups: a short arrow down the middle, labelled beside it.
      network.connections.forEach((edge) => {
        const a = index[edge.src], b = index[edge.dst];
        if (a === undefined || b === undefined || b !== a + 1) return;
        S.svg(svg, 'line', { x1: cx, y1: top(a) + BOX_H, x2: cx, y2: top(b) - 1, class: 'edge', 'marker-end': 'url(#netArrow)' });
        S.svg(svg, 'text', { x: cx + 8, y: top(a) + BOX_H + GAP_V / 2 + 4, class: 'edgelabel' }, edge.synapses + ' synapses');
      });
      // Skips: arcs through their own lanes, labelled at the arc's outer edge.
      skips.forEach((edge, k) => {
        const a = index[edge.src], b = index[edge.dst];
        const x = LEFT - 14 - laneOf[k] * LANE;
        const y1 = mid(a), y2 = mid(b);
        S.svg(svg, 'path', { d: 'M' + LEFT + ',' + y1 + ' C' + x + ',' + y1 + ' ' + x + ',' + y1 + ' ' + x + ',' + (y1 + Math.sign(y2 - y1) * 14) +
          ' V' + (y2 - Math.sign(y2 - y1) * 14) + ' C' + x + ',' + y2 + ' ' + x + ',' + y2 + ' ' + (LEFT - 1) + ',' + y2,
          class: 'edge', 'marker-end': 'url(#netArrow)' });
        // Labels sit outside the outermost lane so no arc runs through them.
        S.svg(svg, 'text', { x: LEFT - 19 - (laneCount - 1) * LANE, y: (y1 + y2) / 2 + 4 + (laneOf[k] % 2) * 12, 'text-anchor': 'end', class: 'edgelabel' },
          edge.synapses + ' syn.');
      });
      // Host links: dashed, to the group the host drives or reads.
      rows.forEach((row, i) => {
        if (!row.host) return;
        const neighbor = i === 0 ? i + 1 : i - 1;
        if (!rows[neighbor] || !rows[neighbor].group) return;
        const [a, b] = i === 0 ? [i, neighbor] : [neighbor, i];
        S.svg(svg, 'line', { x1: cx, y1: top(a) + BOX_H, x2: cx, y2: top(b) - 1, class: 'hostlink' });
        S.svg(svg, 'text', { x: cx + 8, y: top(a) + BOX_H + GAP_V / 2 + 4, class: 'linklabel' }, i === 0 ? 'constant current' : 'spike counts');
      });
      S.html(container, 'div', { class: 'small muted' },
        'Groups in dependency order, top to bottom; arcs on the left are skip connections. Chips beside each box are the cores ' +
        'it is mapped to. Host operations run outside the simulated chip.');

      // Re-flow when the pane width changes the number of chips per line.
      if (container._netObserver) container._netObserver.disconnect();
      if (typeof ResizeObserver === 'function') {
        container._netObserver = new ResizeObserver(() => {
          const w = Math.max(container.clientWidth || 1000, chipsX + 4 * CHIP_STEP + 10);
          if (Math.abs(w - width) > 20 && container.isConnected) this.render(container, session, handlers);
        });
        container._netObserver.observe(container);
      }
    },
  };
})();
