/* Pipeline mode: the input, every site's firing rate per update (or per
   algorithm step), the barrier per update, and the host readout. */
(function () {
  const S = window.Studio;

  function shade(rate) {
    return 'rgba(35,104,160,' + (0.12 + 0.88 * Math.min(1, rate / 0.5)).toFixed(2) + ')';
  }

  function drawImage(canvas, pixels) {
    const bytes = atob(pixels);
    const context = canvas.getContext && canvas.getContext('2d');
    if (!context) return;  // headless tests have no canvas
    const image = context.createImageData(32, 32);
    for (let i = 0; i < 1024; i++) {
      for (let c = 0; c < 3; c++) image.data[4 * i + c] = bytes.charCodeAt(3 * i + c);
      image.data[4 * i + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }

  /* The record for update u (1-based), if it has arrived. */
  function at(records, u) { return records.find((r) => r.update === u) || null; }

  S.pipeline = {
    axis: 'updates',

    render(container, session, records, index, handlers) {
      const spec = (session.metadata || {}).pipeline;
      container.innerHTML = '';
      if (!spec) { container.textContent = 'This workload declares no pipeline.'; return; }
      const sizes = {};
      for (const g of session.network.groups) sizes[g.name] = g.size;
      const shown = records[index] || null;
      const playUpdate = shown ? shown.update : null;
      if (spec.fixture) {
        S.html(container, 'div', { class: 'pwarn' }, 'Untrained fixture checkpoint: the scores and class below show the ' +
          'readout working, not a classification result.');
      }

      // 1. Input
      const input = S.html(container, 'div', { class: 'ppanel' });
      S.html(input, 'div', { class: 'ptitle' }, '1 · Input (host, not simulated)');
      const row = S.html(input, 'div', { class: 'pinput' });
      const canvas = S.html(row, 'canvas', { width: 32, height: 32 });
      drawImage(canvas, spec.input.pixels);
      const classes = spec.classes || [];
      S.html(row, 'div', {}, '');
      row.lastChild.innerHTML = 'test image ' + spec.input.index + ' · true label <b>' + S.escape(classes[spec.input.label] || spec.input.label) +
        '</b><br>' + S.escape(spec.input.drive) + ' → <b>' + spec.input.currents.toLocaleString() + ' constant currents</b> into <code>' +
        S.escape(spec.input.group) + '</code><br>injected on updates 1–' + spec.T + ', then 0 · direct coding, not spikes';

      // 2. Space-time grid
      const grid = S.html(container, 'div', { class: 'ppanel' });
      S.html(grid, 'div', { class: 'ptitle' }, '2 · Space-time grid');
      const axis = S.html(grid, 'div', { class: 'paxis' }, 'time axis: ');
      for (const [key, label] of [['updates', 'hardware updates'], ['steps', 'algorithm timesteps']]) {
        const b = S.html(axis, 'button', { 'data-axis': key, class: S.pipeline.axis === key ? 'on' : '' }, label);
        b.addEventListener('click', () => { S.pipeline.axis = key; handlers.redraw(); });
      }
      const byUpdates = S.pipeline.axis === 'updates';
      const horizon = session.horizon;
      const columns = byUpdates ? Array.from({ length: horizon }, (_, i) => i + 1) : Array.from({ length: spec.T }, (_, t) => t);
      const scroll = S.html(grid, 'div', { class: 'pscroll' });
      const table = S.html(scroll, 'table', { class: 'pgrid' });
      // The label column fits the longest site name; the time columns share the rest.
      const labelWidth = Math.ceil(Math.max(90, ...spec.rows.map((r) => S.textWidth(r.group, 11)), S.textWidth('step time / barrier', 11))) + 14;
      const cols = S.html(table, 'colgroup', {});
      S.html(cols, 'col', {}).style.width = labelWidth + 'px';
      const head = S.html(table, 'tr', {});
      S.html(head, 'th', {}, '');
      for (const c of columns) {
        const u = byUpdates ? c : null;
        S.html(head, 'th', { class: 'pu' + (u !== null && u === playUpdate ? ' play' : '') }, byUpdates ? String(c) : 't=' + c);
      }
      for (const site of spec.rows) {
        const tr = S.html(table, 'tr', { class: 'prow' });
        S.html(tr, 'th', { title: 'window: updates ' + (site.window[0] + 1) + '–' + site.window[1] }, site.group + ' ');
        for (const c of columns) {
          const u = byUpdates ? c : site.depth + c + 1;
          const inside = u - 1 >= site.window[0] && u - 1 < site.window[1];
          const record = at(records, u);
          const count = record && record.group_fired ? record.group_fired[site.group] : null;
          const cls = ['pcell'];
          let title;
          if (!inside) { cls.push('gated'); title = site.group + ' at update ' + u + ': gated (outside its window)'; }
          else if (!record) { cls.push('future'); title = 'update ' + u + ' not computed yet'; }
          else title = site.group + ' at update ' + u + ': ' + (count === null ? 'spikes not logged' : count + ' / ' + sizes[site.group] + ' spikes');
          if (u === playUpdate) cls.push('play');
          const td = S.html(tr, 'td', { class: cls.join(' '), title: title, 'data-group': site.group, 'data-update': u });
          if (inside && record && count !== null) td.style.background = shade(count / sizes[site.group]);
          td.addEventListener('click', () => handlers.pick(site.group, u));
          td.addEventListener('dblclick', () => handlers.open(site.group));
        }
      }
      if (byUpdates) {
        const most = Math.max(...records.map((r) => r.step_time), 1e-12);
        const bars = S.html(table, 'tr', {});
        S.html(bars, 'th', { title: 'modeled step time R; red: barrier wait D' }, 'step time / barrier');
        const refs = S.html(table, 'tr', {});
        S.html(refs, 'th', {}, 'reference');
        for (const u of columns) {
          const r = at(records, u);
          const cell = S.html(bars, 'td', {});
          if (r) {
            const bar = S.html(cell, 'div', { class: 'pbar', title: 'update ' + u + ': ' + S.fmtTime(r.step_time) + ', barrier ' + S.fmtTime(r.barrier) });
            S.html(bar, 'i', {}).style.height = (100 * (r.step_time - r.barrier) / most).toFixed(0) + '%';
            S.html(bar, 'b', {}).style.height = Math.max(3, 100 * r.barrier / most).toFixed(0) + '%';
          }
          const status = r && r.reference ? r.reference.status : null;
          S.html(refs, 'td', { class: 'pref ' + (status || '') }, status === 'match' ? '✓' : status === 'mismatch' ? '✗' : '–');
        }
      }
      S.html(grid, 'div', { class: 'small muted' }, byUpdates
        ? 'Shade: firing rate (spikes ÷ neurons, D). Hatched: gated, outside the site\'s validity window. Red outline: the shown update.'
        : 'Each site at step t is read from update depth + t + 1: the same run re-indexed (D), so latency and gating are hidden.');

      // 3. Output
      const out = S.html(container, 'div', { class: 'ppanel', id: 'pOutput' });
      S.html(out, 'div', { class: 'ptitle' }, '3 · Output: readout on the host');
      const readout = shown && shown.readout;
      if (!readout || !readout.cumulative) {
        S.html(out, 'div', { class: 'muted' }, 'Waiting for the output window: ' + spec.output_group + ' first fires at update ' +
          (spec.rows.find((r) => r.group === spec.output_group).window[0] + 1) + '.');
        return;
      }
      const values = readout.cumulative;
      const lo = Math.min(0, ...values), hi = Math.max(...values);
      values.forEach((v, i) => {
        const line = S.html(out, 'div', { class: 'lrow' + (i === readout.predicted ? ' win' : '') });
        S.html(line, 'span', {}, classes[i] || String(i));
        const bar = S.html(line, 'div', { class: 'lbar' });
        S.html(bar, 'i', {}).style.width = (hi > lo ? 100 * (v - lo) / (hi - lo) : 0).toFixed(0) + '%';
        S.html(line, 'em', {}, Number.isInteger(v) ? String(v) : v.toFixed(3));
      });
      const ref = readout.reference || {};
      const refText = ref.status === 'match' ? 'reference ' + readout.quantity + ': match' + (ref.max_abs_diff ? ' (max diff ' + ref.max_abs_diff.toExponential(1) + ')' : '')
        : ref.status === 'mismatch' ? 'reference ' + readout.quantity + ': MISMATCH (max diff ' + ref.max_abs_diff + ')' : 'reference: waiting';
      S.html(out, 'div', { class: 'small' }, 'prediction ' + (classes[readout.predicted] || readout.predicted) + ' · true label ' +
        (classes[spec.input.label] || spec.input.label) + ' · ' + readout.quantity + ' summed over steps so far (D) · ' + refText + ' (ref)');
    },
  };
})();
