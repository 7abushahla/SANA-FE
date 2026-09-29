/* Live performance charts over every received update. */
(function () {
  const S = window.Studio;
  const UNITS = [['synapse', '#306f9d'], ['dendrite', '#7eb4cd'], ['soma', '#d4944c'], ['network', '#a179b6']];
  const LEFT = 50, RIGHT = 550, TOP = 16, BOTTOM = 176;

  function chart(container, title) {
    const box = S.html(container, 'div');
    S.html(box, 'div', { class: 'title' }, title);
    return S.svg(box, 'svg', { viewBox: '0 0 560 200' });
  }

  function axes(svg, maxLabel, records) {
    S.svg(svg, 'line', { x1: LEFT, y1: BOTTOM, x2: RIGHT, y2: BOTTOM, stroke: '#9fb3c0' });
    S.svg(svg, 'line', { x1: LEFT, y1: TOP, x2: LEFT, y2: BOTTOM, stroke: '#9fb3c0' });
    S.svg(svg, 'text', { x: LEFT + 4, y: TOP + 10, class: 'axis' }, 'max ' + maxLabel);
    S.svg(svg, 'text', { x: LEFT, y: BOTTOM + 14, class: 'axis' }, 'update ' + records[0].update);
    S.svg(svg, 'text', { x: RIGHT, y: BOTTOM + 14, 'text-anchor': 'end', class: 'axis' }, 'update ' + records[records.length - 1].update);
  }

  function energy(svg, records, current) {
    const max = Math.max.apply(null, records.map((r) => r.energy.total).concat([1e-30]));
    const slot = (RIGHT - LEFT) / records.length;
    const width = Math.max(1, slot * 0.72);
    records.forEach((record, index) => {
      let y = BOTTOM;
      const x = LEFT + index * slot + (slot - width) / 2;
      for (const [unit, color] of UNITS) {
        const h = Math.max(0, (record.energy[unit] / max) * (BOTTOM - TOP));
        y -= h;
        S.svg(svg, 'rect', { x: x, y: y, width: width, height: h, fill: color, class: record.update === current ? 'bar cur' : 'bar' });
      }
    });
    axes(svg, S.fmtEnergy(max), records);
    UNITS.forEach(([unit, color], i) => {
      S.svg(svg, 'rect', { x: 300 + i * 62, y: 2, width: 8, height: 8, fill: color });
      S.svg(svg, 'text', { x: 311 + i * 62, y: 10, class: 'axis' }, unit);
    });
  }

  function times(svg, records, current) {
    const maxTime = Math.max.apply(null, records.map((r) => r.step_time).concat([1e-30]));
    const maxMessages = Math.max.apply(null, records.map((r) => r.counts.messages).concat([1]));
    const slot = (RIGHT - LEFT) / records.length;
    const px = (i) => LEFT + (i + 0.5) * slot;
    function line(values, max, color) {
      const points = values.map((v, i) => px(i) + ',' + (BOTTOM - (v / max) * (BOTTOM - TOP))).join(' ');
      S.svg(svg, 'polyline', { points: points, fill: 'none', stroke: color, 'stroke-width': 2 });
    }
    line(records.map((r) => r.step_time), maxTime, '#c9473a');
    line(records.map((r) => r.counts.messages), maxMessages, '#16846f');
    const index = records.findIndex((r) => r.update === current);
    if (index >= 0) S.svg(svg, 'line', { x1: px(index), y1: TOP, x2: px(index), y2: BOTTOM, stroke: '#c9473a', 'stroke-dasharray': '3 3' });
    axes(svg, S.fmtTime(maxTime) + ' · ' + maxMessages + ' messages', records);
    S.svg(svg, 'text', { x: 300, y: 10, class: 'axis', fill: '#c9473a' }, '▬ step time');
    S.svg(svg, 'text', { x: 390, y: 10, class: 'axis', fill: '#16846f' }, '▬ messages');
  }

  S.perf = {
    render(container, records, current) {
      container.innerHTML = '';
      if (!records.length) { container.textContent = 'No updates yet.'; return; }
      const grid = S.html(container, 'div', { class: 'perfgrid' });
      energy(chart(grid, 'Modeled energy per update, stacked by unit'), records, current);
      times(chart(grid, 'Modeled step time and messages per update'), records, current);
    },
  };
})();
