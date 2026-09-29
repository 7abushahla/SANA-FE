/* The loaded architecture YAML and its differences from a bundled baseline. */
(function () {
  const S = window.Studio;

  function show(value) {
    if (value === null || value === undefined) return '—';
    if (typeof value === 'object') return JSON.stringify(value).slice(0, 160);
    return String(value);
  }

  S.arch = {
    async render(container, session) {
      if (container.dataset.session === session.id) return;
      container.dataset.session = session.id;
      container.innerHTML = '';
      const bar = S.html(container, 'div', { class: 'msgbar' });
      S.html(bar, 'span', {}, 'compare with ');
      const select = S.html(bar, 'select', { id: 'archBaseline' });
      S.html(select, 'option', { value: '' }, '(no baseline)');
      S.html(container, 'div', { class: 'small muted' },
        'File contents and differences; costs are this file\'s modeled coefficients, not measurements.');
      const diffBox = S.html(container, 'div', { id: 'archDiff' });
      const text = S.html(container, 'pre', { id: 'archText', class: 'yaml' });
      try {
        const [names, loaded] = await Promise.all([S.api.architectures(), S.api.architecture(session.id)]);
        names.forEach((name) => S.html(select, 'option', { value: name }, name));
        text.textContent = loaded.text;
      } catch (error) {
        diffBox.textContent = error.message;
      }
      select.addEventListener('change', async () => {
        diffBox.innerHTML = '';
        if (!select.value) return;
        try {
          const result = await S.api.architecture(session.id, select.value);
          const diff = result.diff;
          S.html(diffBox, 'div', { class: 'small' }, diff.rows.length + ' differences: ' + diff.loaded + ' against ' + diff.baseline);
          const table = S.html(diffBox, 'table', { class: 'difftable' });
          const head = S.html(table, 'tr');
          ['path', diff.baseline, diff.loaded, 'change'].forEach((label) => S.html(head, 'th', {}, label));
          for (const row of diff.rows) {
            const tr = S.html(table, 'tr', { class: 'change ' + row.change });
            [row.path, show(row.baseline), show(row.loaded), row.change].forEach((value) => S.html(tr, 'td', {}, value));
          }
        } catch (error) {
          diffBox.textContent = error.message;
        }
      });
    },
  };
})();
