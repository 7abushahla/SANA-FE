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
    return Math.abs(ns) < 1000 ? ns.toFixed(1) + ' ns' : (ns / 1000).toFixed(3) + ' µs';
  };

  S.fmtEnergy = function (joules) {
    if (typeof joules !== 'number' || !isFinite(joules)) return String(joules);
    const nj = joules * 1e9;
    if (Math.abs(nj) >= 1000) return (nj / 1000).toFixed(3) + ' µJ';
    if (Math.abs(nj) >= 0.01) return nj.toFixed(3) + ' nJ';
    return (nj * 1000).toFixed(2) + ' pJ';
  };

  S.escape = function (text) {
    return String(text).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
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
