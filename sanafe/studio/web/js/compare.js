/* Compare two saved runs: spike-train identity and per-update differences. */
(function () {
  const S = window.Studio;

  function label(run) {
    const params = Object.keys(run.parameters || {}).map((k) => k + '=' + run.parameters[k]).join(' ');
    return run.workload + ' · ' + params + (Object.keys(run.core_map || {}).length ? ' · edited' : '') +
      ' · ' + run.updates + ' updates · ' + (run.created || '').slice(11, 19);
  }

  /* A: this session's run. B: the newest other run of the same workload
     and length, else the newest other run. */
  function defaults(runs, current) {
    const a = runs.find((r) => r.id === current) || runs[0];
    const others = runs.filter((r) => r !== a);
    const b = others.find((r) => a && r.workload === a.workload && r.updates === a.updates) || others[0];
    return [a, b];
  }

  function chart(parent, rows) {
    const W = 520, H = 130, L = 50;
    const svg = S.svg(parent, 'svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'cmpchart' });
    const values = rows.flatMap((r) => [r.a.step_time, r.b.step_time]);
    const high = Math.max(...values, 1e-12);
    const x = (i) => L + (rows.length > 1 ? i / (rows.length - 1) : 0.5) * (W - L - 10);
    const y = (v) => 10 + (1 - v / high) * (H - 30);
    ['a', 'b'].forEach((side, k) => {
      S.svg(svg, 'polyline', { points: rows.map((r, i) => x(i) + ',' + y(r[side].step_time)).join(' '), class: 'trace t' + k });
      S.svg(svg, 'text', { x: W - 60, y: 14 + 12 * k, class: 'legendtext t' + k }, 'run ' + side.toUpperCase());
    });
    S.svg(svg, 'text', { x: 2, y: 14, class: 'axis' }, S.fmtTime(high));
    S.svg(svg, 'text', { x: L, y: H - 4, class: 'axis' }, 'modeled step time per update');
  }

  S.compare = {
    render(container, runs, current) {
      const key = runs.map((r) => r.id + ':' + r.updates).join(',') + '|' + current;
      if (container.dataset.compare === key) return;
      container.dataset.compare = key;
      container.innerHTML = '';
      if (runs.length < 2) { container.textContent = 'Compare needs two saved runs. Run the session, change the placement, and run again.'; return; }
      const bar = S.html(container, 'div', { class: 'msgbar' });
      const [a, b] = defaults(runs, current);
      const pick = (id, chosen) => {
        S.html(bar, 'span', {}, id === 'cmpA' ? 'A' : 'B');
        const select = S.html(bar, 'select', { id: id });
        for (const run of runs) S.html(select, 'option', { value: run.id }, label(run));
        if (chosen) select.value = chosen.id;
        return select;
      };
      const selectA = pick('cmpA', a);
      const selectB = pick('cmpB', b);
      const go = S.html(bar, 'button', { id: 'btnCompare' }, 'Compare');
      const result = S.html(container, 'div', { id: 'cmpResult' });
      go.addEventListener('click', async () => {
        result.innerHTML = '';
        let data;
        try {
          data = await S.api.compare(selectA.value, selectB.value);
        } catch (error) {
          result.textContent = error.message;
          return;
        }
        let summary;
        if (!data.comparable) summary = 'not comparable: the runs simulate different networks';
        else if (data.identical_spikes) summary = 'spike trains identical over ' + data.updates_a + ' updates';
        else if (data.first_difference.neuron === null) summary = 'spike trains agree for ' + data.updates.length + ' updates, then run ' + data.first_difference.in.toUpperCase() + ' continues';
        else summary = 'spike trains differ at update ' + data.first_difference.update + ': ' + data.first_difference.neuron + ' fired only in run ' + data.first_difference.in.toUpperCase();
        const line = S.html(result, 'div', { id: 'cmpSummary', class: data.identical_spikes ? 'cmpok' : 'cmpbad' });
        line.innerHTML = S.escape(summary) + ' ' + S.mark('D');
        const t = data.totals;
        S.html(result, 'div', { class: 'small muted' }, 'Totals A / B: modeled time ' + S.fmtTime(t.a.step_time) + ' / ' + S.fmtTime(t.b.step_time) +
          ' · energy ' + S.fmtEnergy(t.a.energy) + ' / ' + S.fmtEnergy(t.b.energy) + ' · hops ' + t.a.hops + ' / ' + t.b.hops +
          '. Modeled by SANA-FE with this architecture\'s costs, not measured.');
        chart(result, data.updates);
        const table = S.html(result, 'table');
        const head = S.html(table, 'tr');
        ['update', 'step time A', 'B', 'energy A', 'B', 'hops A', 'B', 'messages A', 'B', 'busiest core packets A', 'B']
          .forEach((h) => S.html(head, 'th', {}, h));
        for (const row of data.updates) {
          const tr = S.html(table, 'tr', { class: 'cmprow' });
          [row.update, S.fmtTime(row.a.step_time), S.fmtTime(row.b.step_time), S.fmtEnergy(row.a.energy), S.fmtEnergy(row.b.energy),
           row.a.hops, row.b.hops, row.a.messages, row.b.messages, row.a.busiest_core_packets, row.b.busiest_core_packets]
            .forEach((v) => S.html(tr, 'td', {}, String(v)));
        }
        S.html(result, 'div', { class: 'small muted' }, 'Step time, energy, hops, and messages are recorded (R); busiest core packets is derived (D).');
      });
    },
  };
})();
