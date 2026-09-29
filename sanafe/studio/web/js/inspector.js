/* Numbers for the selection (chip, tile, core, group, neuron, or message),
   each with its provenance mark. */
(function () {
  const S = window.Studio;

  function rows(items) {
    return '<div class="kv">' + items.map((item) =>
      '<i>' + S.escape(item[0]) + '</i><span>' + S.escape(String(item[1])) + '</span>' +
      (item[2] ? S.mark(item[2]) : '<span></span>')).join('') + '</div>';
  }

  function referenceText(reference) {
    if (!reference) return 'none for this workload';
    if (reference.status === 'match') return 'match (' + reference.references.join(', ') + ')';
    if (reference.status === 'unchecked') return 'unchecked: ' + reference.reason;
    return 'mismatch: ' + reference.neuron + ' ' + reference.quantity + ' vs ' + reference.reference +
      ' (expected ' + reference.expected + ', got ' + reference.actual + '; ' + reference.mismatches + ' differ)';
  }

  function edges(list, total) {
    const shown = list.slice(0, 8).map((e) => e.neuron + ' @' + e.core + ' w=' + e.weight).join('\n');
    return total + (total ? '\n' + shown + (total > 8 ? '\n…' : '') : '');
  }

  const render = {
    chip(record, session) {
      const p = record.provenance;
      const share = record.step_time > 0 ? ' (' + Math.round(100 * record.barrier / record.step_time) + '%)' : '';
      return ['Inspector · chip', rows([
        ['update', record.update, ''],
        ['modeled step time', S.fmtTime(record.step_time), p.step_time],
        ['last core or message activity', S.fmtTime(record.last_activity), p.last_activity],
        ['barrier', S.fmtTime(record.barrier) + share, p.barrier],
        ['messages', record.counts.messages, p.counts],
        ['hops', record.counts.hops, p.counts],
        ['fired (all neurons)', record.counts.fired, p.counts],
        ['energy', S.fmtEnergy(record.energy.total), p.energy],
        ['occupied cores', session.network.occupied.length, ''],
        ['reference check', referenceText(record.reference), record.reference ? 'D' : ''],
      ])];
    },

    tile(record, session, selection) {
      const tile = session.layout.tiles[selection.tile];
      const keys = tile.cores.map((c) => c.key);
      const mapped = keys.filter((k) => session.network.occupied.indexOf(k) >= 0).length;
      let out = record.messages.filter((m) => keys.indexOf(m.src) >= 0).length;
      let inn = record.messages.filter((m) => keys.indexOf(m.dst) >= 0).length;
      if (S.isAggregate(record)) {  // no messages kept: sum the per-core counts
        out = 0; inn = 0;
        for (const key of keys) {
          const counts = record.core_counts[key];
          if (counts) { out += counts.packets_out; inn += counts.packets_in; }
        }
      }
      let through = record.messages.filter((m) => m.path.indexOf(tile.id) >= 0).length;
      if (S.isAggregate(record)) {  // packets entering or leaving this router on mesh links
        through = 0;
        for (const hop in record.links) if (hop.split('>').map(Number).indexOf(tile.id) >= 0) through += record.links[hop];
      }
      return ['Inspector · tile ' + tile.id, rows([
        ['position (x, y)', tile.x + ', ' + tile.y, 'R'],
        ['mapped cores', mapped + ' / ' + keys.length, 'R'],
        ['packets out', out, 'D'],
        ['packets in', inn, 'D'],
        ['messages routed through its router', through, 'X'],
        ['network energy', record.tile_network_energy[tile.id] === undefined ? 'not recorded' :
          S.fmtEnergy(record.tile_network_energy[tile.id]), record.tile_network_energy[tile.id] === undefined ? '' : record.provenance.tile_network_energy],
      ])];
    },

    core(record, session, selection) {
      const p = record.provenance;
      const key = selection.key;
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
      const budget = session.network.core_budgets && session.network.core_budgets[key];
      if (budget) {
        items.push(['stored synapses · outgoing neurons', budget.edges + ' · ' + budget.outgoing_neurons, 'R']);
        items.push(['assumed synapse bytes', budget.assumed_synapse_bytes + ' of 131072', 'D']);
        items.push(['assumed total bytes', budget.assumed_total_bytes + ' of 196608', 'D']);
        items.push(['  (assumed candidate layout, not Intel packing)', '', '']);
      }
      return ['Inspector · core ' + key, rows(items)];
    },

    group(record, session, selection) {
      const group = session.network.groups.find((g) => g.name === selection.name);
      const theta = (session.network.group_attributes[group.name] || {}).threshold;
      const inn = session.network.connections.filter((c) => c.dst === group.name);
      const out = session.network.connections.filter((c) => c.src === group.name);
      const logged = (session.network.group_attributes[group.name] || {}).log_spikes && !S.isAggregate(record);
      const fired = logged ? record.fired.filter((f) => f[0] === group.name).length
        : (S.isAggregate(record) ? 'not kept (aggregate; see core counts)' : 'unknown (neurons without log_spikes)');
      return ['Inspector · group ' + group.name, rows([
        ['neurons', group.size, 'R'],
        ['cores', Object.keys(group.cores).length, 'R'],
        ['threshold', theta === null || theta === undefined ? 'not uniform' : theta, 'R'],
        ['inputs', inn.map((c) => c.src + ' (' + c.synapses + ')').join(', ') || 'none', 'R'],
        ['outputs', out.map((c) => c.dst + ' (' + c.synapses + ')').join(', ') || 'none', 'R'],
        ['fired this update', fired, logged ? 'D' : ''],
      ])];
    },

    neuron(record, session, selection, detail) {
      const key = selection.key;
      const title = 'Inspector · neuron ' + key;
      if (!detail) return [title, 'Loading…'];
      if (detail.error) return [title, S.escape(detail.error)];
      const [group, offset] = [detail.group, detail.offset];
      const items = [['core', detail.core, 'R']];
      for (const name in detail.attributes) {
        const value = detail.attributes[name];
        items.push(['  ' + name, Array.isArray(value) ? value.slice(0, 6).join(', ') + (value.length > 6 ? ' …' : '') : value, 'R']);
      }
      if (!record) {
        items.push(['this update', 'no update yet', '']);
      } else {
        const past = detail.history && detail.history.potential.length >= record.update;
        const inRecord = key in record.potentials;
        // Aggregate records carry only watched neurons; older updates come from the
        // history fetched with the detail, which is refreshed when a run settles.
        const pending = S.isAggregate(record) && !inRecord && !past;
        let v = inRecord ? record.potentials[key] : (past ? detail.history.potential[record.update - 1] : undefined);
        if (v === null) v = undefined;
        const fired = record.fired.some((f) => f[0] === group && f[1] === offset) ||
          (past && detail.history.fired[record.update - 1]);
        if (pending) {
          items.push(['membrane after update ' + record.update, 'loading (refreshes when the run settles)', '']);
          items.push(['fired this update', 'loading', '']);
        } else {
          items.push(['membrane after update ' + record.update, v === undefined ? 'not logged' : v, v === undefined ? '' : record.provenance.potentials]);
          items.push(['fired this update', detail.log_spikes ? (fired ? 'yes' : 'no') : 'not logged', detail.log_spikes ? record.provenance.fired : '']);
        }
      }
      if (detail.reference && record) {
        // Reference executions are not SANA-FE records: they carry their own mark.
        for (const name in detail.reference) {
          const value = detail.reference[name].potential[record.update - 1];
          const label = name + ' membrane' + (name === 'spikingjelly' ? ', aligned by layer depth' : '');
          items.push([label, value === null || value === undefined ? 'no value at this update' : value, 'ref']);
        }
      }
      items.push(['fan-in', edges(detail.fan_in, detail.fan_in_total), 'R']);
      items.push(['fan-out', edges(detail.fan_out, detail.fan_out_total), 'R']);
      const actions = '<div class="actions">' + (detail.log_potential
        ? '<button id="btnWatch" class="watchbtn">＋ Watch this neuron</button>'
        : '<div class="hint">Not watchable: this neuron does not log its membrane.</div>') +
        (detail.log_spikes ? '<button id="btnBreakFire" class="watchbtn">⏸ Break when it fires</button>' : '') +
        '<button id="btnHighlight" class="watchbtn">◎ Highlight connections</button></div>';
      return [title, rows(items) + actions];
    },

    message(record, session, selection) {
      const m = record.messages.find((item) => item.mid === selection.mid);
      if (!m) return ['Inspector · message', 'Not in this update.'];
      return ['Inspector · message ' + m.mid, rows([
        ['source neuron', m.src_neuron, 'R'],
        ['source → destination core', m.src + ' → ' + m.dst, 'R'],
        ['hops', m.hops, 'R'],
        ['route (tiles)', m.path.join(' → '), 'X'],
        ['spikes', m.spikes, 'R'],
        ['generation delay', S.fmtTime(m.generation_delay), 'R'],
        ['network delay', S.fmtTime(m.network_delay), 'R'],
        ['blocking delay', S.fmtTime(m.blocking_delay), 'R'],
        ['processing delay', S.fmtTime(m.processing_delay), 'R'],
        ['send · receive · processed', S.fmtTime(m.send) + ' · ' + S.fmtTime(m.receive) + ' · ' + S.fmtTime(m.processed), 'R'],
      ])];
    },
  };

  S.inspector = {
    referenceText: referenceText,

    render(titleNode, box, selection, record, session, detail) {
      if (!session || (!record && selection.kind !== 'neuron')) {
        titleNode.textContent = 'Inspector';
        box.textContent = session ? 'No update yet.' : 'Start a session, then click a core.';
        return;
      }
      const view = render[selection.kind] || render.chip;
      const [title, html] = view(record, session, selection, detail);
      titleNode.textContent = title;
      box.innerHTML = html;
    },
  };
})();
