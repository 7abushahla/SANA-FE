/* The recorded messages of the displayed update, filterable and linked to
   the chip: clicking a row draws that message's reconstructed route. */
(function () {
  const S = window.Studio;
  const LIMIT = 200;

  function rowText(m) { return m.src_neuron + ' ' + m.src + ' → ' + m.dst; }

  S.messages = {
    filter: '',

    render(container, record, selectedMid, onPick) {
      if (!record) { container.innerHTML = ''; container.textContent = 'No update yet.'; return; }
      let input = container.querySelector('#msgFilter');
      if (!input) {
        container.innerHTML = '';
        const bar = S.html(container, 'div', { class: 'msgbar' });
        input = S.html(bar, 'input', { id: 'msgFilter', type: 'search', placeholder: 'filter: core, neuron, or "→ 16.0"' });
        input.value = S.messages.filter;
        input.addEventListener('input', () => { S.messages.filter = input.value; S.messages.rows(container, container._record, container._selected, container._pick); });
        S.html(bar, 'span', { class: 'small muted', id: 'msgCount' });
        S.html(container, 'div', { id: 'msgRows' });
      }
      container._record = record;
      container._selected = selectedMid;
      container._pick = onPick;
      S.messages.rows(container, record, selectedMid, onPick);
    },

    rows(container, record, selectedMid, onPick) {
      const needle = S.messages.filter.trim().toLowerCase();
      const matching = record.messages.filter((m) => !needle || rowText(m).toLowerCase().indexOf(needle) >= 0);
      const shown = matching.slice(0, LIMIT);
      container.querySelector('#msgCount').textContent = 'Update ' + record.update + ': ' + matching.length + ' of ' +
        record.messages.length + ' messages' + (matching.length > LIMIT ? ', first ' + LIMIT + ' shown' : '') +
        '. Times are recorded modeled times within the update. Click a row to draw its route on the chip.';
      const box = container.querySelector('#msgRows');
      box.innerHTML = '';
      const table = S.html(box, 'table');
      const head = S.html(table, 'tr');
      for (const label of ['source neuron', 'route', 'hops', 'spikes', 'send', 'receive', 'processed', 'network delay', 'blocking']) {
        S.html(head, 'th', {}, label);
      }
      for (const m of shown) {
        const row = S.html(table, 'tr', { class: 'msg' + (m.mid === selectedMid ? ' on' : ''), 'data-mid': m.mid });
        [m.src_neuron, m.src + ' → ' + m.dst, m.hops, m.spikes, S.fmtTime(m.send), S.fmtTime(m.receive),
         S.fmtTime(m.processed), S.fmtTime(m.network_delay), S.fmtTime(m.blocking_delay)]
          .forEach((value) => S.html(row, 'td', {}, String(value)));
        row.addEventListener('click', () => onPick(m.mid));
      }
    },
  };
})();
