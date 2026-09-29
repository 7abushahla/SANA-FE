/* Parameter forms generated from a workload's schema. */
(function () {
  const S = window.Studio;

  S.params = {
    render(container, specs) {
      container.innerHTML = '';
      for (const spec of specs) {
        const label = S.html(container, 'label', {}, spec.name + ' ');
        let input;
        if (spec.kind === 'choice') {
          input = document.createElement('select');
          for (const choice of spec.choices) {
            const option = S.html(input, 'option', { value: choice }, String(choice));
            if (choice === spec.default) option.selected = true;
          }
        } else {
          input = document.createElement('input');
          if (spec.kind === 'int') {
            input.type = 'number';
            input.step = '1';
            if (spec.minimum !== null) input.min = spec.minimum;
            if (spec.maximum !== null) input.max = spec.maximum;
          } else {
            input.type = 'text';
            input.placeholder = 'path to file';
          }
          if (spec.default !== null) input.value = spec.default;
        }
        input.setAttribute('data-param', spec.name);
        input.setAttribute('data-kind', spec.kind);
        label.appendChild(input);
        if (spec.help) S.html(label, 'div', { class: 'hint' }, spec.help);
      }
    },

    /* Blank fields are left out so the server applies defaults. Anything that
       is not a whole number is sent as typed, and the server names the error. */
    read(container) {
      const values = {};
      for (const input of container.querySelectorAll('[data-param]')) {
        const raw = input.value.trim();
        if (raw === '') continue;
        const isInt = input.getAttribute('data-kind') === 'int' && /^-?\d+$/.test(raw);
        values[input.getAttribute('data-param')] = isInt ? Number(raw) : raw;
      }
      return values;
    },
  };
})();
