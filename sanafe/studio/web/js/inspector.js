/* Numbers for the selected chip or core, each with its provenance mark. */
(function () {
  const S = window.Studio;

  function rows(items) {
    return '<div class="kv">' + items.map((item) =>
      '<i>' + S.escape(item[0]) + '</i><span>' + S.escape(String(item[1])) + '</span>' +
      (item[2] ? S.mark(item[2]) : '<span></span>')).join('') + '</div>';
  }

  S.inspector = {
    render(titleNode, box, selection, record, session) {
      if (!session || !record) {
        titleNode.textContent = 'Inspector';
        box.textContent = session ? 'No update yet.' : 'Start a session, then click a core.';
        return;
      }
      const p = record.provenance;
      if (selection.kind === 'core') {
        const key = selection.key;
        titleNode.textContent = 'Inspector · core ' + key;
        const counts = record.core_counts[key];
        const energy = record.core_energy[key];
        const groups = session.network.groups.filter((g) => g.cores[key]).map((g) => g.name + ' (' + g.cores[key] + ')');
        let fired = '0';
        if (counts) fired = counts.fired === null ? 'unknown (neurons without log_spikes)' : counts.fired;
        const items = [
          ['neurons', groups.length ? groups.join(', ') : 'none mapped', 'R'],
          ['neuron processing ends', key in record.core_finish ? S.fmtTime(record.core_finish[key]) : 'no records', p.core_finish],
          ['fired this update', fired, p.core_counts],
          ['packets in / out', counts ? counts.packets_in + ' / ' + counts.packets_out : '0 / 0', p.core_counts],
          ['spikes in', counts ? counts.spikes_in : 0, p.core_counts],
          ['energy', energy ? S.fmtEnergy(energy.total) : 'not recorded', energy ? p['core_energy.total'] : ''],
        ];
        if (energy) {
          for (const unit in energy.units) items.push(['  ' + unit, S.fmtEnergy(energy.units[unit]), p['core_energy.units']]);
          if (energy.axon !== null) items.push(['  axon in and out', S.fmtEnergy(energy.axon), p['core_energy.axon']]);
        }
        box.innerHTML = rows(items);
        return;
      }
      titleNode.textContent = 'Inspector · chip';
      const share = record.step_time > 0 ? ' (' + Math.round(100 * record.barrier / record.step_time) + '%)' : '';
      box.innerHTML = rows([
        ['update', record.update, ''],
        ['modeled step time', S.fmtTime(record.step_time), p.step_time],
        ['last core or message activity', S.fmtTime(record.last_activity), p.last_activity],
        ['barrier', S.fmtTime(record.barrier) + share, p.barrier],
        ['messages', record.counts.messages, p.counts],
        ['hops', record.counts.hops, p.counts],
        ['fired (all neurons)', record.counts.fired, p.counts],
        ['energy', S.fmtEnergy(record.energy.total), p.energy],
        ['occupied cores', session.network.occupied.length, ''],
      ]);
    },
  };
})();
