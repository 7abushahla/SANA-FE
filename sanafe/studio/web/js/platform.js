/* The Platform card: what the chosen platform is, what SANA-FE models of it,
   and where every cost coefficient comes from. The Architecture view (YAML and
   diff) sits under the card. */
(function () {
  const S = window.Studio;

  function fmt(attribute, value) {
    if (value && typeof value === 'object') {
      return Object.keys(value).map((k) => (k === '0' ? 'fixed' : k + ' tiles') + ': ' + S.fmtTime(value[k])).join(', ');
    }
    if (attribute === 'link_buffer_size') return String(value);
    return attribute.startsWith('energy') ? S.fmtEnergy(value) : S.fmtTime(value);
  }

  function table(parent, headers, rows) {
    const t = S.html(parent, 'table', { class: 'cardtable' });
    const head = S.html(t, 'tr');
    headers.forEach((h) => S.html(head, 'th', {}, h));
    for (const row of rows) {
      const tr = S.html(t, 'tr');
      row.forEach((cell) => {
        const td = S.html(tr, 'td');
        if (cell && typeof cell === 'object' && cell.status) {
          S.html(td, 'span', { class: 'status ' + cell.status }, cell.status + (cell.factor ? ' ÷ ' + cell.factor : ''));
        } else {
          td.textContent = cell === null || cell === undefined ? '' : String(cell);
        }
      });
    }
    return t;
  }

  S.platform = {
    render(container, session) {
      if (container.dataset.session === session.id && container.dataset.view === 'platform') return;
      container.dataset.session = session.id;
      container.dataset.view = 'platform';
      container.innerHTML = '';
      const card = S.html(container, 'div', { id: 'platformCard', class: 'platformcard' });
      const p = session.platform;
      if (!p) {
        S.html(card, 'div', { class: 'small muted' },
          'This workload brings its own architecture file; no catalog platform applies. The file is shown below.');
      } else {
        S.html(card, 'div', { class: 'small nothw' }, p.not_hardware);
        S.html(card, 'h4', {}, p.title);
        S.html(card, 'div', { class: 'small muted' }, p.vendor + ' · ' + p.generation + ' · ' + p.yaml);
        if (p.architecture_matches === false) {
          S.html(card, 'div', { class: 'small modified' }, 'This session simulates a modified architecture, not the ' +
            'packaged ' + p.yaml + '. Costs come from the workload\'s own file; the costs table below describes the ' +
            'catalog profile, not what was simulated.');
        }
        if (p.selected === false) {
          S.html(card, 'div', { class: 'small matchnote' }, 'The workload\'s own architecture file is identical to this ' +
            'catalog platform, so this card applies.');
        }
        S.html(card, 'p', {}, p.summary);
        S.html(card, 'p', {}, 'Execution: ' + p.execution + '. ' + p.time_rule);
        S.html(card, 'h5', {}, 'Structure');
        const s = p.structure;
        table(card, ['property', 'value'], [
          ['mesh', s.width + ' × ' + s.height + ' tiles'], ['tiles', s.tiles], ['cores', s.cores],
          ['cores per tile', s.cores_per_tile], ['max neurons per core', s.max_neurons_per_core],
          ['update-boundary buffer', s.buffer_position],
          ['synchronization', s.sync_model + ' · ' + fmt('latency_sync', s.sync_table)],
          ['link buffer', s.link_buffer_size]]);
        S.html(card, 'h5', {}, 'Units and models (reference core)');
        table(card, ['role', 'unit', 'model', 'accepted attributes'], p.units.map((u) => [
          u.role, u.name + (u.instances > 1 ? ' × ' + u.instances : ''),
          u.model + (u.plugin ? ' (' + u.plugin + ')' : ''),
          u.attributes.map((a) => a.name + (a.help ? ': ' + a.help : '')).join('\n')]));
        S.html(card, 'h5', {}, 'Costs');
        S.html(card, 'div', { class: 'small muted' },
          'Modeled times are marked "' + p.time_word + '" and energies "' + p.energy_word + '" throughout the page.');
        table(card, ['unit', 'coefficient', 'value', 'status', 'source and note'], p.costs.map((c) => [
          c.unit, c.attribute, fmt(c.attribute, c.value), { status: c.status, factor: c.factor },
          c.source + (c.note ? '. ' + c.note : '')]));
        S.html(card, 'h5', {}, 'Validation');
        S.html(card, 'p', {}, p.validation);
        S.html(card, 'h5', {}, 'Not modeled');
        const ul = S.html(card, 'ul');
        p.not_modeled.forEach((item) => S.html(ul, 'li', {}, item));
        S.html(card, 'h5', {}, 'References');
        const ol = S.html(card, 'ol');
        p.references.forEach((item) => S.html(ol, 'li', {}, item));
      }
      S.html(container, 'h5', {}, 'Architecture file');
      const arch = S.html(container, 'div', { id: 'archView' });
      S.arch.render(arch, session);
    },
  };
})();
