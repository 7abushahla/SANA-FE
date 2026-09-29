/* Network mode: groups, their connections, host operations, and each
   group's cores (SANA-FE Fig. 3 style). Everything here is recorded (R). */
(function () {
  const S = window.Studio;
  const BOX_W = 150, GAP = 70, BOX_H = 84, TOP = 70, CHIPS = 12;
  const HOST_CHARS = 22;  // 10px host text inside a 150-wide box

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

  S.network = {
    render(container, session, handlers) {
      container.innerHTML = '';
      const network = session.network;
      const metadata = session.metadata || {};
      const hostOps = metadata.host_operations || [];
      const names = order(network.groups, network.connections);
      const columns = [];
      if (hostOps.length && metadata.host_input_group) columns.push({ host: hostOps[0] });
      names.forEach((name) => columns.push({ group: network.groups.find((g) => g.name === name) }));
      hostOps.slice(metadata.host_input_group ? 1 : 0).forEach((op) => columns.push({ host: op }));
      const width = 40 + columns.length * (BOX_W + GAP);
      const height = TOP + BOX_H + 150;
      const svg = S.svg(container, 'svg', { viewBox: '0 0 ' + width + ' ' + height, class: 'netview' });
      const x = {};
      columns.forEach((column, i) => {
        const left = 20 + i * (BOX_W + GAP);
        if (column.host) {
          const g = S.svg(svg, 'g', { class: 'host' });
          S.svg(g, 'rect', { x: left, y: TOP, width: BOX_W, height: BOX_H, rx: 8, class: 'host' });
          // Title plus the operation wrapped to the box, centered vertically.
          const lines = S.wrap(column.host, HOST_CHARS);
          const first = TOP + BOX_H / 2 - (lines.length * 13) / 2 + 4;
          S.svg(g, 'text', { x: left + BOX_W / 2, y: first, 'text-anchor': 'middle', class: 'hosttext', 'font-weight': 700 }, 'HOST (not simulated)');
          lines.forEach((line, n) => S.svg(g, 'text', { x: left + BOX_W / 2, y: first + 15 + 13 * n, 'text-anchor': 'middle', class: 'hosttext' }, line));
          column.left = left;
          return;
        }
        const group = column.group;
        const index = network.groups.indexOf(group);
        const color = S.PALETTE[index % S.PALETTE.length];
        x[group.name] = left;
        const g = S.svg(svg, 'g', { class: 'group', 'data-group': group.name });
        const box = S.svg(g, 'rect', { x: left, y: TOP, width: BOX_W, height: BOX_H, rx: 8, fill: color, class: 'groupbox' });
        S.svg(box, 'title', {}, 'click to inspect ' + group.name);
        S.svg(g, 'text', { x: left + BOX_W / 2, y: TOP + 34, 'text-anchor': 'middle', class: 'groupname' }, group.name);
        const theta = (network.group_attributes[group.name] || {}).threshold;
        S.svg(g, 'text', { x: left + BOX_W / 2, y: TOP + 52, 'text-anchor': 'middle', class: 'groupmeta' },
          group.size + ' neurons' + (theta ? ' · θ ' + theta : ''));
        g.addEventListener('click', () => handlers.selectGroup(group.name));
        const cores = Object.keys(group.cores);
        cores.slice(0, CHIPS).forEach((core, j) => {
          const cx = left + (j % 4) * 37;
          const cy = TOP + BOX_H + 12 + Math.floor(j / 4) * 24;
          const chip = S.svg(svg, 'rect', { x: cx, y: cy, width: 34, height: 19, rx: 3, class: 'corechip', stroke: color, 'data-core': core });
          S.svg(chip, 'title', {}, 'core ' + core + ' · ' + group.cores[core] + ' neurons · click to open');
          S.svg(svg, 'text', { x: cx + 17, y: cy + 13, 'text-anchor': 'middle', class: 'corechiptext', fill: color, 'pointer-events': 'none' }, core);
          chip.addEventListener('click', () => handlers.openCore(core));
        });
        if (cores.length > CHIPS) {
          S.svg(svg, 'text', { x: left, y: TOP + BOX_H + 12 + 3 * 24 + 14, class: 'hosttext' }, '+' + (cores.length - CHIPS) + ' more cores');
        }
      });

      const mid = TOP + BOX_H / 2;
      network.connections.forEach((edge, k) => {
        if (x[edge.src] === undefined || x[edge.dst] === undefined) return;
        const from = x[edge.src] + BOX_W;
        const to = x[edge.dst];
        let label;
        if (to > from && to - from <= GAP + 1) {
          S.svg(svg, 'line', { x1: from, y1: mid, x2: to, y2: mid, class: 'edge' });
          label = [(from + to) / 2, mid - 8];
        } else {
          const lift = 30 + (k % 3) * 12;
          S.svg(svg, 'path', { d: 'M' + (x[edge.src] + BOX_W / 2) + ',' + TOP + ' C' + (x[edge.src] + BOX_W / 2) + ',' + (TOP - lift) + ' ' +
            (to + BOX_W / 2) + ',' + (TOP - lift) + ' ' + (to + BOX_W / 2) + ',' + TOP, class: 'edge' });
          label = [(x[edge.src] + to + BOX_W) / 2, TOP - lift + 6];
        }
        S.svg(svg, 'text', { x: label[0], y: label[1], 'text-anchor': 'middle', class: 'edgelabel' }, edge.synapses + ' synapses');
      });
      columns.forEach((column, i) => {
        if (!column.host) return;
        const neighbor = i === 0 ? columns[1] : columns[i - 1];
        if (!neighbor || !neighbor.group) return;
        const a = i === 0 ? column.left + BOX_W : x[neighbor.group.name] + BOX_W;
        const b = i === 0 ? x[neighbor.group.name] : column.left;
        S.svg(svg, 'line', { x1: a, y1: mid, x2: b, y2: mid, class: 'hostlink' });
        S.svg(svg, 'text', { x: (a + b) / 2, y: mid + 16, 'text-anchor': 'middle', class: 'hosttext' }, i === 0 ? 'constant current' : 'spike counts');
      });
      S.html(container, 'div', { class: 'small muted' },
        'Boxes are neuron groups in dependency order; chips under each box are the cores it is mapped to. ' +
        'Host operations run outside the simulated chip.');
    },
  };
})();
