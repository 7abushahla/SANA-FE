/* Shared helpers on the window.Studio namespace. */
(function () {
  const S = (window.Studio = window.Studio || {});
  const NS = 'http://www.w3.org/2000/svg';

  S.PALETTE = ['#2368a0', '#c77622', '#16846f', '#8659a8', '#b8445a', '#5a7d1a', '#3b8ea5', '#a0522d'];

  S.svg = function (parent, tag, attrs, text) {
    const node = document.createElementNS(NS, tag);
    for (const key in attrs || {}) node.setAttribute(key, attrs[key]);
    if (text !== undefined) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  };

  S.html = function (parent, tag, attrs, text) {
    const node = document.createElement(tag);
    for (const key in attrs || {}) node.setAttribute(key, attrs[key]);
    if (text !== undefined) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  };

  S.fmtTime = function (seconds) {
    if (typeof seconds !== 'number' || !isFinite(seconds)) return String(seconds);
    const ns = seconds * 1e9;
    if (Math.abs(ns) < 1e3) return ns.toFixed(1) + ' ns';
    if (Math.abs(ns) < 1e6) return (ns / 1e3).toFixed(3) + ' µs';
    if (Math.abs(ns) < 1e9) return (ns / 1e6).toFixed(3) + ' ms';
    return (ns / 1e9).toFixed(3) + ' s';
  };

  S.fmtEnergy = function (joules) {
    if (typeof joules !== 'number' || !isFinite(joules)) return String(joules);
    const nj = joules * 1e9;
    if (Math.abs(nj) >= 1000) return (nj / 1000).toFixed(3) + ' µJ';
    if (Math.abs(nj) >= 0.01) return nj.toFixed(3) + ' nJ';
    return (nj * 1000).toFixed(2) + ' pJ';
  };

  /* Aggregate records keep counts per core and link, not individual messages. */
  S.isAggregate = function (record) {
    return !!record && record.provenance && record.provenance.messages === 'not kept (aggregate)';
  };

  /* In slow motion a packet stays on screen for at least this share of its
     update, even if its recorded flight is far shorter (ResNet-20: about
     10 ns of an 83 µs update). The modeled-time clock keeps true proportions. */
  S.MIN_FLIGHT = 0.04;
  S.flightEnd = function (message, record, clock) {
    if (clock !== 'slow') return message.receive;
    return Math.max(message.receive, message.send + S.MIN_FLIGHT * record.step_time);
  };

  /* Messages to animate: all of them, or the aggregate level's sample. */
  S.flying = function (record) {
    if (!record) return [];
    return S.isAggregate(record) ? (record.sample || []) : record.messages;
  };

  S.escape = function (text) {
    return String(text).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  };

  /* Rendered width of SVG text at a font size, for sizing label gutters. A
     hidden SVG text in the page font measures it; where text is not laid out
     (headless tests) an estimate stands in. */
  let measure = null;
  const measured = new Map();  // each measurement forces a layout: measure a label once
  S.textWidth = function (text, size) {
    const key = size + '|' + text;
    if (!measured.has(key)) measured.set(key, S.measureText(String(text), size));
    return measured.get(key);
  };
  S.measureText = function (text, size) {
    if (measure === null) {
      const ns = 'http://www.w3.org/2000/svg';
      const svg = document.createElementNS(ns, 'svg');
      svg.setAttribute('style', 'position:absolute;width:0;height:0;visibility:hidden');
      const node = document.createElementNS(ns, 'text');
      svg.appendChild(node);
      document.body.appendChild(svg);
      measure = typeof node.getComputedTextLength === 'function' ? node : false;
    }
    if (!measure) return String(text).length * size * 0.6;
    measure.setAttribute('font-size', size);
    measure.textContent = String(text);
    return measure.getComputedTextLength();
  };

  /* SVG text does not wrap: break at spaces into lines of at most `chars`. */
  S.wrap = function (text, chars) {
    const lines = [];
    let line = '';
    for (const word of String(text).split(/\s+/)) {
      if (line && (line + ' ' + word).length > chars) { lines.push(line); line = word; }
      else line = line ? line + ' ' + word : word;
    }
    if (line) lines.push(line);
    return lines;
  };

  /* Escaped text with break opportunities inside identifiers: after '_' and
     before a named segment, so conv2_x.0.shortcut_relay wraps between words. */
  S.breakable = function (text) {
    return S.escape(text).replace(/_/g, '_<wbr>').replace(/\.(?=[A-Za-z])/g, '.<wbr>');
  };

  S.mark = function (letter) {
    return '<span class="m ' + letter + '">' + letter + '</span>';
  };

  /* Undo the server's {"$float": "inf"} tags for non-finite values. */
  S.decodeNonFinite = function decode(value) {
    if (Array.isArray(value)) return value.map(decode);
    if (value && typeof value === 'object') {
      const keys = Object.keys(value);
      if (keys.length === 1 && keys[0] === '$float') {
        return { inf: Infinity, '-inf': -Infinity, nan: NaN }[value.$float];
      }
      const out = {};
      for (const key of keys) out[key] = decode(value[key]);
      return out;
    }
    return value;
  };
})();
