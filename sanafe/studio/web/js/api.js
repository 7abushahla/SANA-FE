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
