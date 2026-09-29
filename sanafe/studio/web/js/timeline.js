/* One update as per-core rows on the modeled clock (SANA-FE Fig. 5 style). */
(function () {
  const S = window.Studio;

  function byCore(a, b) {
    const p = a.split('.').map(Number);
    const q = b.split('.').map(Number);
    return p[0] - q[0] || p[1] - q[1];
  }

  S.timeline = {
    render(container, record, t, owner) {
      container.innerHTML = '';
      if (!record) { container.textContent = 'No update yet. Step or run the session.'; return; }
      const cores = Object.keys(record.core_finish).sort(byCore);
      const left = 140, width = 1030, top = 26, rowHeight = 24;
      const height = top + cores.length * rowHeight + 34;
      const bottom = height - 30;
      const svg = S.svg(container, 'svg', { viewBox: '0 0 ' + (left + width + 20) + ' ' + height, class: 'timeline' });
      const step = record.step_time > 0 ? record.step_time : 1e-12;
      const x = (value) => left + Math.max(0, Math.min(1, value / step)) * width;

      S.svg(svg, 'rect', { x: x(record.last_activity), y: 4, width: Math.max(0, x(step) - x(record.last_activity)), height: bottom - 4, class: 'barrier' });
      S.svg(svg, 'text', { x: x(record.last_activity) + 6, y: 17, class: 'axis' }, 'barrier ' + S.fmtTime(record.barrier));

      const row = {};
      cores.forEach((core, index) => {
        const y = top + index * rowHeight;
        row[core] = y;
        const who = owner && owner[core];
        S.svg(svg, 'text', { x: 4, y: y + 11, class: 'rowlabel' }, core + (who ? ' · ' + who.group : ''));
        const bar = S.svg(svg, 'rect', { x: x(0), y: y, width: Math.max(1, x(record.core_finish[core]) - x(0)), height: 8, class: 'nproc' });
        bar.setAttribute('fill', who ? who.color : '#607888');
      });

      for (const message of record.messages) {
        if (row[message.dst] !== undefined) {
          S.svg(svg, 'rect', { x: x(message.receive), y: row[message.dst] + 10, width: Math.max(2, x(message.processed) - x(message.receive)), height: 6, class: 'mproc' });
        }
        if (row[message.src] !== undefined && row[message.dst] !== undefined) {
          S.svg(svg, 'line', { x1: x(message.send), y1: row[message.src] + 4, x2: x(message.receive), y2: row[message.dst] + 13, class: message.blocking_delay > 0 ? 'arrow blocked' : 'arrow' });
        }
      }

      if (S.isAggregate(record)) {
        S.svg(svg, 'text', { x: left, y: 17, class: 'axis' }, 'aggregate trace level: ' + record.counts.messages +
          ' messages counted, not kept, so no message rows');
      }
      S.svg(svg, 'line', { x1: x(t), y1: 2, x2: x(t), y2: bottom, class: 'playhead' });
      S.svg(svg, 'text', { x: left, y: height - 10, class: 'axis' }, '0');
      S.svg(svg, 'text', { x: left + width, y: height - 10, 'text-anchor': 'end', class: 'axis' }, S.fmtTime(step) + ' modeled');
      S.svg(svg, 'text', { x: left + 60, y: height - 10, class: 'axis' },
        'bars: neuron processing · purple: message processing · lines: send to receive (thick red: blocked) · red line: playhead');
    },
  };
})();
