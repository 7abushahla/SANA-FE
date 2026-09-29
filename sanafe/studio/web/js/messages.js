/* The recorded messages of the displayed update. */
(function () {
  const S = window.Studio;
  const LIMIT = 200;

  S.messages = {
    render(container, record) {
      container.innerHTML = '';
      if (!record) { container.textContent = 'No update yet.'; return; }
      const shown = record.messages.slice(0, LIMIT);
      S.html(container, 'div', { class: 'small muted' },
        'Update ' + record.update + ': ' + record.messages.length + ' messages' +
        (record.messages.length > LIMIT ? ', first ' + LIMIT + ' shown' : '') +
        '. Times are recorded modeled times within the update.');
      const table = S.html(container, 'table');
      const head = S.html(table, 'tr');
      for (const label of ['source neuron', 'route', 'hops', 'spikes', 'send', 'receive', 'processed', 'network delay', 'blocking']) {
        S.html(head, 'th', {}, label);
      }
      for (const m of shown) {
        const row = S.html(table, 'tr');
        [m.src_neuron, m.src + ' → ' + m.dst, m.hops, m.spikes, S.fmtTime(m.send), S.fmtTime(m.receive),
         S.fmtTime(m.processed), S.fmtTime(m.network_delay), S.fmtTime(m.blocking_delay)]
          .forEach((value) => S.html(row, 'td', {}, String(value)));
      }
    },
  };
})();
