/* HTTP commands and the per-session WebSocket. */
(function () {
  const S = window.Studio;

  async function request(method, url, body) {
    const options = { method: method, headers: {} };
    if (body !== undefined) {
      options.headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(body);
    }
    const response = await fetch(url, options);
    const text = await response.text();
    const data = text ? S.decodeNonFinite(JSON.parse(text)) : null;
    if (!response.ok) throw new Error((data && data.error) || response.status + ' ' + response.statusText);
    return data;
  }

  S.api = {
    workloads: () => request('GET', '/api/workloads'),
    create: (body) => request('POST', '/api/sessions', body),
    get: (id) => request('GET', '/api/sessions/' + id),
    remove: (id) => request('DELETE', '/api/sessions/' + id),
    step: (id, n) => request('POST', '/api/sessions/' + id + '/step', { n: n }),
    run: (id) => request('POST', '/api/sessions/' + id + '/run', {}),
    pause: (id) => request('POST', '/api/sessions/' + id + '/pause', {}),
    reset: (id) => request('POST', '/api/sessions/' + id + '/reset', {}),
    updates: (id, from) => request('GET', '/api/sessions/' + id + '/updates?from=' + from),
    neuron: (id, key) => {
      const cut = key.lastIndexOf('.');
      return request('GET', '/api/sessions/' + id + '/neurons/' + encodeURIComponent(key.slice(0, cut)) + '/' + key.slice(cut + 1));
    },
    architectures: () => request('GET', '/api/architectures'),
    platforms: () => request('GET', '/api/platforms'),
    setBreakpoints: (id, specs) => request('PUT', '/api/sessions/' + id + '/breakpoints', { breakpoints: specs }),
    runs: () => request('GET', '/api/runs'),
    setWatches: (id, keys) => request('PUT', '/api/sessions/' + id + '/watches', { watches: keys }),
    coreState: (id, core, update) => request('GET', '/api/sessions/' + id + '/cores/' + core + '?update=' + update),
    compare: (a, b) => request('POST', '/api/compare', { a: a, b: b }),
    architecture: (id, baseline) => request('GET', '/api/sessions/' + id + '/architecture' +
      (baseline ? '?baseline=' + encodeURIComponent(baseline) : '')),

    /* Reconnects after a drop; every connection starts with a 'hello'. */
    connect(id, onMessage) {
      let closed = false;
      let socket = null;
      function open() {
        const scheme = location.protocol === 'https:' ? 'wss://' : 'ws://';
        socket = new WebSocket(scheme + location.host + '/ws/sessions/' + id);
        socket.onmessage = (event) => onMessage(S.decodeNonFinite(JSON.parse(event.data)));
        socket.onclose = () => { if (!closed) setTimeout(open, 1000); };
      }
      open();
      return { close() { closed = true; if (socket) socket.close(); } };
    },
  };
})();
