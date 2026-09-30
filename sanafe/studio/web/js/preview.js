/* A platform that has a card but no engine (Speck): the chip drawn from the
   catalog's block list and core table. Nothing here comes from a simulation. */
(function () {
  const S = window.Studio;

  S.preview = {
    render(container, card) {
      container.innerHTML = '';
      S.html(container, 'div', { class: 'small pnote' }, card.preview_note);
      const layout = card.preview_layout || { blocks: [], cores: [] };
      const W = 900, H = 400;
      const svg = S.svg(container, 'svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': card.title + ' block diagram' });
      const defs = S.svg(svg, 'defs');
      const marker = S.svg(defs, 'marker', { id: 'parrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto' });
      S.svg(marker, 'path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: '#607888' });
      S.svg(svg, 'rect', { x: 8, y: 8, width: W - 16, height: H - 16, rx: 12, class: 'chipframe' });
      S.svg(svg, 'text', { x: 24, y: 30, class: 'plabel' }, card.title);
      S.svg(svg, 'text', { x: 24, y: 46, class: 'psub' }, card.vendor + ' · ' + card.generation);

      const byId = {};
      layout.blocks.forEach((b) => { byId[b.id] = b; });
      const block = (id, x, y, w, h) => {
        const b = byId[id] || { label: id, sub: '' };
        S.svg(svg, 'rect', { x: x, y: y, width: w, height: h, rx: 8, class: 'pblock', 'data-block': id });
        S.svg(svg, 'text', { x: x + w / 2, y: y + h / 2 - 4, class: 'plabel', 'text-anchor': 'middle' }, b.label);
        S.svg(svg, 'text', { x: x + w / 2, y: y + h / 2 + 12, class: 'psub', 'text-anchor': 'middle' }, b.sub || '');
        return { x: x, y: y, w: w, h: h };
      };
      const arrow = (a, b) => S.svg(svg, 'line', { x1: a.x + a.w, y1: a.y + a.h / 2, x2: b.x, y2: b.y + b.h / 2, class: 'parrow', 'marker-end': 'url(#parrow)' });

      const dvs = block('dvs', 30, 170, 130, 60);
      const pre = block('preprocess', 190, 170, 150, 60);
      const noc = { x: 380, y: 70, w: 320, h: 260 };
      S.svg(svg, 'rect', { x: noc.x, y: noc.y, width: noc.w, height: noc.h, rx: 12, class: 'tile' });
      const nb = byId.noc || { label: 'NoC', sub: '' };
      S.svg(svg, 'text', { x: noc.x + noc.w / 2, y: noc.y + 18, class: 'plabel', 'text-anchor': 'middle' }, nb.label);
      S.svg(svg, 'text', { x: noc.x + noc.w / 2, y: noc.y + 32, class: 'psub', 'text-anchor': 'middle' }, nb.sub || '');
      const readout = block('readout', 740, 170, 130, 60);
      arrow(dvs, pre);
      arrow(pre, noc);
      arrow(noc, readout);

      // The star: one router in the middle, the cores around it.
      const cx = noc.x + noc.w / 2, cy = noc.y + noc.h / 2 + 20;
      const cores = layout.cores || [];
      const rx = 122, ry = 82;
      cores.forEach((core, i) => {
        const angle = -Math.PI / 2 + (2 * Math.PI * i) / Math.max(1, cores.length);
        const x = cx + rx * Math.cos(angle), y = cy + ry * Math.sin(angle);
        S.svg(svg, 'line', { x1: cx, y1: cy, x2: x, y2: y, class: 'spoke' });
        const rect = S.svg(svg, 'rect', { x: x - 42, y: y - 16, width: 84, height: 32, rx: 4, class: 'core used', 'data-core': String(core.id) });
        rect.setAttribute('fill', S.PALETTE[i % S.PALETTE.length]);
        S.svg(rect, 'title', {}, (core.label || 'core ' + core.id) + ' · ' + (core.memory || ''));
        S.svg(svg, 'text', { x: x, y: y - 3, class: 'pcore', 'text-anchor': 'middle' }, core.label || 'core ' + core.id);
        S.svg(svg, 'text', { x: x, y: y + 9, class: 'pcoresub', 'text-anchor': 'middle' }, core.memory || '');
        if (core.memory && core.memory.length > 22) svg.lastChild.setAttribute('textLength', 78);
      });
      S.svg(svg, 'circle', { cx: cx, cy: cy, r: 9, class: 'router' });
      S.svg(svg, 'text', { x: 24, y: H - 22, class: 'psub' },
        'Structure from the published architecture; the cores show their neuron and kernel memory. No engine runs this platform yet.');
    },
  };
})();
