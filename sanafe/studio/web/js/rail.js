/* Left-rail debugger sections: breakpoints, watches, placement edits, saved runs. */
(function () {
  const S = window.Studio;
  const FIELDS = {
    neuron_fires: ['neuron'], core_sends: ['core', 'value'], step_time: ['value'],
    update: ['value'], reference_mismatch: [],
  };
  const VALUE_LABEL = { core_sends: 'more than (packets)', step_time: 'more than (ns)', update: 'equals (update)' };

  function describe(spec) {
    if (spec.kind === 'neuron_fires') return spec.neuron + ' fires';
    if (spec.kind === 'core_sends') return 'core ' + spec.core + ' sends > ' + spec.more_than;
    if (spec.kind === 'step_time') return 'step time > ' + (spec.more_than * 1e9).toFixed(1) + ' ns';
    if (spec.kind === 'update') return 'update ' + spec.equals;
    return 'reference mismatch';
  }

  S.rail = {
    describe: describe,

    /* The kind select shows only the fields that kind needs. */
    bindBreakpointForm(form, onAdd) {
      const kind = form.querySelector('#bpKind');
      const show = () => {
        const needed = FIELDS[kind.value];
        form.querySelector('#bpNeuronRow').style.display = needed.indexOf('neuron') >= 0 ? '' : 'none';
        form.querySelector('#bpCoreRow').style.display = needed.indexOf('core') >= 0 ? '' : 'none';
        form.querySelector('#bpValueRow').style.display = needed.indexOf('value') >= 0 ? '' : 'none';
        form.querySelector('#bpValueLabel').textContent = VALUE_LABEL[kind.value] || '';
      };
      kind.addEventListener('change', show);
      show();
      form.querySelector('#bpAdd').addEventListener('click', () => {
        const spec = { kind: kind.value };
        const value = form.querySelector('#bpValue').value.trim();
        if (kind.value === 'neuron_fires') spec.neuron = form.querySelector('#bpNeuron').value.trim();
        if (kind.value === 'core_sends') { spec.core = form.querySelector('#bpCore').value.trim(); spec.more_than = Number(value); }
        if (kind.value === 'step_time') spec.more_than = Number(value) * 1e-9;
        if (kind.value === 'update') spec.equals = Number(value);
        onAdd(spec);
      });
    },

    renderBreakpoints(list, specs, hitId, handlers) {
      list.innerHTML = '';
      if (!specs.length) { S.html(list, 'li', { class: 'muted' }, 'none'); return; }
      for (const spec of specs) {
        const item = S.html(list, 'li', { class: 'bp' + (spec.id === hitId ? ' hit' : ''), 'data-id': spec.id });
        const box = S.html(item, 'input', { type: 'checkbox', class: 'bpOn' });
        box.checked = spec.enabled !== false;
        box.addEventListener('change', () => handlers.toggle(spec.id, box.checked));
        S.html(item, 'span', {}, describe(spec));
        const remove = S.html(item, 'button', { class: 'bpDel', title: 'remove' }, '×');
        remove.addEventListener('click', () => handlers.remove(spec.id));
      }
    },

    renderWatches(list, watches, onSelect) {
      list.innerHTML = '';
      if (!watches.length) { S.html(list, 'li', { class: 'muted' }, 'none: open a neuron and press Watch'); return; }
      for (const key of watches) {
        const item = S.html(list, 'li', { class: 'watch', 'data-key': key }, key);
        item.addEventListener('click', () => onSelect(key));
      }
    },

    /* pending and applied are full core maps; an edit is where they differ. */
    renderEdits(box, pending, applied, handlers) {
      box.innerHTML = '';
      const changed = JSON.stringify(pending) !== JSON.stringify(applied);
      const moves = Object.keys(pending).sort().map((from) => from + ' → ' + pending[from]);
      S.html(box, 'div', { class: 'small' }, moves.length ? 'core map: ' + moves.join(', ') : 'workload placement');
      if (!changed) {
        S.html(box, 'div', { class: 'hint' }, 'Drag a used core onto another core to swap them.');
      } else {
        S.html(box, 'div', { class: 'hint' }, 'Pending: rebuild to apply. Spike trains must not change; time and energy may.');
      }
      const row = S.html(box, 'div', { class: 'row' });
      const apply = S.html(row, 'button', { id: 'btnApplyEdits' }, 'Rebuild with edits');
      apply.disabled = !changed;
      apply.addEventListener('click', handlers.apply);
      const discard = S.html(row, 'button', { id: 'btnDiscardEdits', class: 'plain' }, 'Discard');
      discard.disabled = !changed;
      discard.addEventListener('click', handlers.discard);
      if (Object.keys(applied).length) {
        const reset = S.html(row, 'button', { id: 'btnResetPlacement', class: 'plain' }, 'Workload placement');
        reset.addEventListener('click', handlers.reset);
      }
    },

    renderRuns(list, runs, currentRun) {
      list.innerHTML = '';
      if (!runs.length) { S.html(list, 'li', { class: 'muted' }, 'none saved (server started with --no-store?)'); return; }
      for (const run of runs.slice(0, 12)) {
        const item = S.html(list, 'li', { class: 'run' + (run.id === currentRun ? ' current' : ''), 'data-run': run.id });
        const params = Object.keys(run.parameters || {}).map((k) => k + '=' + run.parameters[k]).join(' ');
        const edits = Object.keys(run.core_map || {}).length ? ' · edited placement' : '';
        S.html(item, 'div', {}, (run.id === currentRun ? '● ' : '') + run.workload + ' · ' + run.updates + ' updates');
        S.html(item, 'div', { class: 'hint' }, params + edits + ' · ' + (run.created || '').slice(11, 19));
        const links = S.html(item, 'div', { class: 'hint' }, 'export: ');
        for (const kind of ['raster', 'potential', 'energy', 'throughput', 'latency']) {
          S.html(links, 'a', { class: 'export', 'data-kind': kind, target: '_blank',
            href: '/api/runs/' + encodeURIComponent(run.id) + '/plots/' + kind + '.svg' }, kind);
          links.appendChild(document.createTextNode(' '));
        }
      }
    },
  };
})();
