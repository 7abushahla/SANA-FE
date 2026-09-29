/* Neuron watch: membrane per update as simulated (R), with the reference
   traces of the same neuron when the workload has them. */
(function () {
  const S = window.Studio;
  const W = 900, H = 150, LEFT = 60, TOP = 16, BOTTOM = 26;
  const REF_STYLE = { lava: '6 4', spikingjelly: '2 3' };

  function plot(container, key, records, detail, current, onRemove) {
    const head = S.html(container, 'div', { class: 'watchhead' });
    head.innerHTML = '<b>' + S.escape(key) + '</b> · membrane after each update: SANA-FE ' + S.mark('R') +
      (detail && detail.reference ? ' · reference executions ' + S.mark('ref') + ' dashed' : '') + ' · red ticks: SANA-FE spikes';
    const remove = S.html(head, 'button', { class: 'linkbtn', 'data-remove': key }, 'remove');
    remove.addEventListener('click', () => onRemove(key));
    const series = { 'SANA-FE': records.map((r) => (key in r.potentials ? r.potentials[key] : null)) };
    const references = (detail && detail.reference) || {};
    for (const name in references) series[name] = references[name].potential;
    const count = Math.max(1, ...Object.values(series).map((s) => s.length));
    const values = [].concat(...Object.values(series)).filter((v) => typeof v === 'number');
    const theta = detail && detail.attributes && detail.attributes.threshold;
    if (typeof theta === 'number') values.push(theta);
    const low = Math.min(0, ...values), high = Math.max(1, ...values);
    const x = (i) => LEFT + (count > 1 ? i / (count - 1) : 0.5) * (W - LEFT - 20);
    const y = (v) => TOP + (1 - (v - low) / (high - low || 1)) * (H - TOP - BOTTOM);
    const svg = S.svg(container, 'svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'watchplot' });
    S.svg(svg, 'line', { x1: LEFT, y1: y(0), x2: W - 20, y2: y(0), class: 'zero' });
    if (typeof theta === 'number') {
      S.svg(svg, 'line', { x1: LEFT, y1: y(theta), x2: W - 20, y2: y(theta), class: 'threshold' });
      S.svg(svg, 'text', { x: 4, y: y(theta) + 4, class: 'axis' }, 'θ ' + theta);
    }
    Object.keys(series).forEach((name, index) => {
      const points = series[name].map((v, i) => (typeof v === 'number' ? x(i) + ',' + y(v) : null));
      let run = [];
      const flush = () => {
        if (run.length) S.svg(svg, 'polyline', { points: run.join(' '), class: 'trace t' + index, 'stroke-dasharray': REF_STYLE[name] || '' });
        run = [];
      };
      points.forEach((p) => { if (p) run.push(p); else flush(); });
      flush();
      S.svg(svg, 'text', { x: W - 200, y: TOP + 12 * index, class: 'legendtext t' + index }, (REF_STYLE[name] ? '- - ' : '— ') + name);
    });
    records.forEach((r, i) => {
      const group = key.slice(0, key.lastIndexOf('.')), offset = Number(key.slice(key.lastIndexOf('.') + 1));
      if (r.fired.some((f) => f[0] === group && f[1] === offset)) {
        S.svg(svg, 'line', { x1: x(i), y1: H - BOTTOM + 2, x2: x(i), y2: H - BOTTOM + 10, class: 'spiketick' });
      }
    });
    if (current !== null) S.svg(svg, 'line', { x1: x(current), y1: TOP - 4, x2: x(current), y2: H - BOTTOM, class: 'playhead' });
    for (let i = 0; i < count; i += Math.max(1, Math.ceil(count / 12))) {
      S.svg(svg, 'text', { x: x(i), y: H - 4, 'text-anchor': 'middle', class: 'axis' }, String(i + 1));
    }
  }

  S.watch = {
    render(container, watches, records, details, currentIndex, onRemove) {
      container.innerHTML = '';
      if (!watches.length) {
        container.textContent = 'No watched neurons. Open a core, click a neuron, then press "Watch this neuron".';
        return;
      }
      for (const key of watches) plot(container, key, records, details[key], currentIndex, onRemove);
      S.html(container, 'div', { class: 'small muted' },
        'x axis: update number. SpikingJelly has no connection delay, so its trace is shown on the update axis ' +
        'shifted by the neuron\'s layer depth, where the workload provides that alignment.');
    },
  };
})();
