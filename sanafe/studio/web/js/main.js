/* Page controller: session lifecycle, events, views, selection, and rendering. */
(function () {
  const S = window.Studio;
  const $ = (id) => document.getElementById(id);
  const app = {
    workloads: [], session: null, socket: null, selection: { kind: 'chip' }, tab: 'timeline',
    renderKey: null, alive: true, view: { mode: 'chip', level: 'chip', tile: null, core: null },
    details: {}, detailVersion: 0, watches: [],
    breakpoints: [], hitId: null, runs: [], applied: {}, pending: {}, coreStates: {},
  };
  const CHIP_NOTE = $('chipNote').innerHTML;

  const chip = new S.Chip($('chip'), select, zoomTo, { onDrag: dragCore });
  const mini = new S.Chip($('minimap'), null, null, { mini: true });
  const zoom = new S.Zoom($('zoom'), { openCore: openCore, selectNeuron: (key) => select({ kind: 'neuron', key: key }),
    coreState: coreState });

  /* Aggregate sessions keep a core's membranes in the worker; fetch them per
     shown update and redraw when they arrive. */
  function coreState(core, update) {
    const key = core + '@' + update;
    if (key in app.coreStates) return app.coreStates[key];
    app.coreStates[key] = null;
    const sessionId = app.session.id;
    S.api.coreState(sessionId, core, update).then((state) => {
      if (app.session && app.session.id === sessionId) { app.coreStates[key] = state; redraw(); }
    }).catch(() => { app.coreStates[key] = { failed: true, neurons: [], potentials: [], fired: [] }; });
    return null;
  }

  /* The worker validates watches; on refusal the list goes back to what it
     accepted, so one bad key cannot block every later sync. */
  function syncWatches(previous) {
    if (!app.session) return;
    S.api.setWatches(app.session.id, app.watches).then((reply) => {
      app.watches = reply.watches;
      renderRail();
    }).catch((error) => {
      $('formError').textContent = error.message;
      app.watches = previous || [];
      renderRail();
      redraw();
    });
  }
  const player = new S.Player(frame);

  function context() {
    const session = app.session;
    return {
      layout: session.layout, network: session.network, metadata: session.metadata || {},
      layoutCore(key) { const [tile, offset] = key.split('.').map(Number); return session.layout.tiles[tile].cores[offset]; },
    };
  }

  function redraw() {
    app.renderKey = null;
    player.dirty = true;
  }

  /* ---- debugger rail: breakpoints, watches, placement edits, saved runs ---- */
  /* The chip shows the pending placement, so every drag acts on what is shown. */
  function previewPlacement() {
    if (!app.session) return;
    const network = app.session.network;
    const same = JSON.stringify(app.pending) === JSON.stringify(app.applied);
    let shown = network;
    const moved = new Set();
    if (!same) {
      const where = (key) => {
        const origin = Object.keys(app.applied).find((x) => app.applied[x] === key);
        const original = origin !== undefined ? origin : key;
        return app.pending[original] !== undefined ? app.pending[original] : original;
      };
      shown = Object.assign({}, network, {
        groups: network.groups.map((group) => {
          const cores = {};
          for (const key in group.cores) {
            const to = where(key);
            if (to !== key) moved.add(to);
            cores[to] = (cores[to] || 0) + group.cores[key];
          }
          return Object.assign({}, group, { cores: cores });
        }),
      });
    }
    chip.build(app.session.layout, shown, app.session.metadata || {}, moved);
    $('chipNote').classList.toggle('previewing', !same);
    redraw();
  }

  function renderRail() {
    S.rail.renderBreakpoints($('bpList'), app.breakpoints, app.hitId, { toggle: toggleBreakpoint, remove: removeBreakpoint });
    S.rail.renderWatches($('watchList'), app.watches, (key) => select({ kind: 'neuron', key: key }));
    S.rail.renderEdits($('editRail'), app.pending, app.applied, {
      apply: () => startSession(app.pending),
      discard: () => { app.pending = Object.assign({}, app.applied); renderRail(); previewPlacement(); },
      reset: () => { app.pending = {}; renderRail(); previewPlacement(); },
    });
    S.rail.renderRuns($('runsRail'), app.runs, app.session && app.session.run);
  }

  async function saveBreakpoints(specs) {
    if (!app.session) return;
    $('formError').textContent = '';
    try {
      const reply = await S.api.setBreakpoints(app.session.id, specs);
      app.breakpoints = reply.breakpoints;
    } catch (error) {
      $('formError').textContent = error.message;
    }
    renderRail();
  }

  function addBreakpoint(spec) {
    const used = app.breakpoints.map((b) => Number(b.id.slice(1)) || 0);
    spec.id = 'b' + (Math.max(0, ...used) + 1);
    saveBreakpoints(app.breakpoints.concat([spec]));
  }

  function toggleBreakpoint(id, enabled) {
    saveBreakpoints(app.breakpoints.map((b) => (b.id === id ? Object.assign({}, b, { enabled: enabled }) : b)));
  }

  function removeBreakpoint(id) {
    if (app.hitId === id) app.hitId = null;
    saveBreakpoints(app.breakpoints.filter((b) => b.id !== id));
  }

  async function refreshRuns() {
    try {
      app.runs = await S.api.runs();
    } catch (error) {
      app.runs = [];
    }
    renderRail();
    if (app.tab === 'compare') redraw();
  }

  /* Swap the contents of two displayed cores. The core map is kept relative
     to the workload's own placement: original core -> displayed core. */
  function dragCore(from, to) {
    const map = Object.assign({}, app.pending);
    const origin = (shown) => {
      const moved = Object.keys(map).find((key) => map[key] === shown);
      if (moved !== undefined) return moved;
      return map[shown] === undefined ? shown : null;  // null: nothing is shown there
    };
    const a = origin(from);
    const b = origin(to);
    if (a === null) return;
    map[a] = to;
    if (b !== null && b !== a) map[b] = from;
    // Only cores that hold neurons need an entry; empty ones have nothing to move.
    const holding = new Set(app.session.network.occupied.map((key) => {
      const origin = Object.keys(app.applied).find((x) => app.applied[x] === key);
      return origin !== undefined ? origin : key;
    }));
    for (const key of Object.keys(map)) if (map[key] === key || !holding.has(key)) delete map[key];
    app.pending = map;
    renderRail();
    previewPlacement();
  }

  /* ---- selection ---- */
  function select(selection) {
    chip.highlight = null;
    app.selection = selection;
    chip.selected = selection.kind === 'core' ? selection.key : null;
    chip.route = selection.kind === 'message' ? selection.mid : null;
    if (selection.kind === 'neuron') loadDetail(selection.key);
    redraw();
  }

  async function loadDetail(key) {
    if (!app.session || key in app.details) return;
    const sessionId = app.session.id;
    app.details[key] = null;
    let detail;
    try {
      detail = await S.api.neuron(sessionId, key);
    } catch (error) {
      detail = { error: error.message };
    }
    if (!app.session || app.session.id !== sessionId) return;
    app.details[key] = detail;
    app.detailVersion += 1;
    redraw();
  }

  function addWatch(key) {
    const previous = app.watches.slice();
    if (app.watches.indexOf(key) < 0) app.watches.push(key);
    syncWatches(previous);
    delete app.details[key];  // refetch, so its history is current
    loadDetail(key);
    renderRail();
    redraw();
  }

  function removeWatch(key) {
    const previous = app.watches.slice();
    app.watches = app.watches.filter((k) => k !== key);
    syncWatches(previous);
    renderRail();
    redraw();
  }

  /* Outline the cores this neuron hears from (blue) and talks to (green). */
  function highlightConnections(key) {
    const detail = app.details[key];
    if (!detail || detail.error) return;
    setView({ level: 'chip' });
    chip.highlight = { in: detail.fan_in_cores, out: detail.fan_out_cores };  // complete, not capped
    redraw();
  }

  /* ---- views: chip > tile > core, and network ---- */
  function setView(view) {
    app.view = Object.assign({ mode: 'chip', level: 'chip', tile: null, core: null }, view);
    const v = app.view;
    const zoomed = v.mode === 'chip' && v.level !== 'chip';
    $('chip').style.display = v.mode === 'chip' && v.level === 'chip' ? '' : 'none';
    $('zoom').style.display = zoomed ? '' : 'none';
    $('minimap').style.display = zoomed ? '' : 'none';
    $('network').style.display = v.mode === 'net' ? '' : 'none';
    $('chipNote').style.display = v.mode === 'chip' && v.level === 'chip' ? '' : 'none';
    $('mChip').className = v.mode === 'chip' ? 'on' : '';
    $('mNet').className = v.mode === 'net' ? 'on' : '';
    mini.focusTile = zoomed ? v.tile : null;
    if (app.session) {
      if (zoomed) zoom.show(context(), v.level, v.level === 'tile' ? v.tile : v.core);
      if (v.mode === 'net') S.network.render($('network'), app.session, { openCore: openCore, selectGroup: (name) => select({ kind: 'group', name: name }) });
    }
    crumb();
    redraw();
  }

  function crumb() {
    const node = $('crumb');
    node.innerHTML = '';
    const v = app.view;
    if (v.mode === 'net') { node.textContent = 'Network: groups, connections, and their cores'; return; }
    const link = (label, view) => {
      const a = S.html(node, 'a', { href: '#' }, label);
      a.addEventListener('click', (event) => { event.preventDefault(); setView(view); });
    };
    if (v.level === 'chip') { node.textContent = 'Chip'; return; }
    link('Chip', { level: 'chip' });
    node.appendChild(document.createTextNode(' › '));
    if (v.level === 'tile') { S.html(node, 'b', {}, 'Tile ' + v.tile); return; }
    link('Tile ' + v.tile, { level: 'tile', tile: v.tile });
    node.appendChild(document.createTextNode(' › '));
    S.html(node, 'b', {}, 'Core ' + v.core.split('.')[1]);
  }

  function zoomTo(target) {
    if (target.core) { openCore(target.core); return; }
    setView({ level: 'tile', tile: target.tile });
    select({ kind: 'tile', tile: target.tile });
  }

  /* A route spans the mesh, so picking a message shows the whole chip. */
  function pickMessage(mid) {
    if (app.view.mode !== 'chip' || app.view.level !== 'chip') setView({ level: 'chip' });
    select({ kind: 'message', mid: mid });
  }

  function openCore(key) {
    setView({ level: 'core', tile: Number(key.split('.')[0]), core: key });
    select({ kind: 'core', key: key });
  }

  /* ---- rendering ---- */
  function referencePill(record) {
    const pill = $('refPill');
    const status = !record ? '–' : (record.reference ? record.reference.status : 'none');
    pill.textContent = 'reference: ' + status;
    pill.className = 'refpill ' + (record && record.reference ? record.reference.status : 'none');
  }

  function frame() {
    const record = player.current();
    const v = app.view;
    if (v.mode === 'chip' && v.level === 'chip') chip.render(record, player.t);
    if (v.mode === 'chip' && v.level !== 'chip') {
      mini.render(record, player.t);
      zoom.update(record, player.t);
    }
    $('clockText').textContent = record
      ? 'update ' + record.update + ' · modeled time ' + S.fmtTime(player.t) + ' of ' + S.fmtTime(record.step_time)
      : '–';
    const phase = $('phase');
    if (record && record.reference && record.reference.status === 'mismatch') {
      phase.textContent = 'reference mismatch at update ' + record.update + ': ' + record.reference.neuron + ' ' +
        record.reference.quantity + ' (' + record.reference.reference + ')';
      phase.className = 'phase mismatch';
    } else if (record) {
      const barrier = player.t >= record.last_activity;
      phase.textContent = barrier ? 'barrier: waiting for the update to close' : 'processing and spike distribution (overlapping)';
      phase.className = barrier ? 'phase barrier' : 'phase';
    } else {
      phase.textContent = '';
      phase.className = 'phase';
    }
    referencePill(record);
    if (app.tab === 'timeline') S.timeline.render($('dock'), record, player.t, chip.owner);
    const sel = app.selection;
    const key = [player.index, player.records.length, app.tab, sel.kind, sel.key, sel.tile, sel.name, sel.mid,
      !!app.session, app.detailVersion, app.watches.join(','), v.mode, v.level, v.core].join('|');
    if (key === app.renderKey) return;
    app.renderKey = key;
    if (app.tab === 'perf') S.perf.render($('dock'), player.records, record ? record.update : null);
    if (app.tab === 'messages') S.messages.render($('dock'), record, sel.kind === 'message' ? sel.mid : null, pickMessage);
    if (app.tab === 'watch') S.watch.render($('dock'), app.watches, player.records, app.details, player.index >= 0 ? player.index : null, removeWatch);
    if (app.tab === 'arch' && app.session) S.arch.render($('dock'), app.session);
    S.inspector.render($('insTitle'), $('inspector'), sel, record, app.session, sel.kind === 'neuron' ? app.details[sel.key] : null);
    const watch = $('btnWatch');
    if (watch) watch.addEventListener('click', () => addWatch(sel.key));
    const breakFire = $('btnBreakFire');
    if (breakFire) breakFire.addEventListener('click', () => addBreakpoint({ kind: 'neuron_fires', neuron: sel.key }));
    const highlight = $('btnHighlight');
    if (highlight) highlight.addEventListener('click', () => highlightConnections(sel.key));
    if (app.tab === 'compare') S.compare.render($('dock'), app.runs, app.session && app.session.run);
    $('uNum').textContent = player.records.length;
    const scrub = $('scrub');
    scrub.max = Math.max(1, player.records.length);
    scrub.value = player.index + 1;
    scrub.disabled = player.records.length === 0;
    $('follow').checked = player.follow;
    $('btnReplay').disabled = !record;
    $('btnLatest').disabled = !player.records.length;
  }

  function describeState(info) {
    if (info.state === 'stopped') return 'stopped: ' + info.reason;
    if (info.state === 'faulted') {
      return 'faulted: ' + (info.fault || 'unknown') + (info.alive === false ? ' · start a new session' : ' · reset to continue');
    }
    return info.state;
  }

  function setState(info) {
    app.alive = info.alive !== false;
    const hit = info.state === 'stopped' && /^breakpoint ([^:]+):/.exec(info.reason || '');
    const hitId = hit ? hit[1] : null;
    if (hitId !== app.hitId) { app.hitId = hitId; renderRail(); }
    if (['finished', 'stopped', 'paused'].indexOf(info.state) >= 0) {
      refreshRuns();
      // A settled run: refresh the histories of watched and selected neurons.
      const keys = app.watches.slice();
      if (app.selection.kind === 'neuron' && keys.indexOf(app.selection.key) < 0) keys.push(app.selection.key);
      for (const key of keys) { delete app.details[key]; loadDetail(key); }
    }
    const node = $('state');
    node.textContent = describeState(info);
    node.className = 'state ' + info.state;
    const usable = !!app.session && app.alive;
    const running = info.state === 'running' || info.state === 'starting';
    const blocked = !usable || running || info.state === 'faulted';
    $('btnStep').disabled = blocked;
    $('btnRunN').disabled = blocked;
    $('btnRun').disabled = blocked;
    $('btnPause').disabled = !usable || !running;
    $('btnReset').disabled = !usable || running;
    if (typeof info.horizon === 'number') $('uHorizon').textContent = info.horizon;
  }

  function applyReady(ready) {
    const same = app.session && app.session.id === ready.id;
    app.session = Object.assign({}, app.session, ready);
    // A rebuild keeps the watches whose neurons still exist.
    const sizes = {};
    for (const group of ready.network.groups) sizes[group.name] = group.size;
    app.watches = app.watches.filter((key) => {
      const cut = key.lastIndexOf('.');
      return Number(key.slice(cut + 1)) < (sizes[key.slice(0, cut)] || 0);
    });
    app.details = {};
    app.coreStates = {};
    app.detailVersion += 1;
    $('chipNote').innerHTML = ready.trace_level === 'aggregate'
      ? 'Aggregate trace level. Moving packets are a sample: the first message between each pair of cores, sent at its recorded time ' +
        S.mark('R') + '. In slow motion each stays visible for at least 4% of the update (flights are nanoseconds in updates of tens of microseconds). ' +
        'Link width shows all packets per mesh link over the update. Both follow the reconstructed x-then-y route ' +
        S.mark('X') + '. Core finish times ' + S.mark('D') + ' animate as usual.'
      : CHIP_NOTE;
    chip.selected = null;
    chip.route = null;
    app.selection = { kind: 'chip' };
    chip.build(ready.layout, ready.network, ready.metadata || {});
    mini.build(ready.layout, ready.network, ready.metadata || {});
    $('badge').textContent = ready.badge;
    const params = (ready.manifest && ready.manifest.parameters) || {};
    $('sessionName').textContent = app.session.workload + ' · ' + Object.keys(params).map((k) => k + '=' + params[k]).join(' ');
    const metadata = ready.metadata || {};
    const rule = metadata.horizon_rule;
    $('sessionInfo').textContent = 'Horizon: ' + ready.horizon + ' updates' + (rule ? ' (' + rule + ')' : '') +
      '. Run length is set by Step, Run n, or Run to horizon.' +
      (metadata.reference ? ' Reference: ' + metadata.reference + ' (built in ' + metadata.reference_seconds + ' s).' : '');
    $('uHorizon').textContent = ready.horizon;
    setState(typeof ready.state === 'object' ? ready.state : { state: ready.state, horizon: ready.horizon });
    const dock = $('dock');
    delete dock.dataset.session;
    delete dock.dataset.compare;
    app.breakpoints = ready.breakpoints || [];
    app.applied = Object.assign({}, ready.core_map || {});
    app.pending = Object.assign({}, app.applied);
    app.hitId = null;
    const warnings = ready.breakpoint_warnings || [];
    if (warnings.length) $('formError').textContent = 'Breakpoints not carried over: ' + warnings.join('; ');
    renderRail();
    refreshRuns();
    for (const key of app.watches) loadDetail(key);
    if (app.watches.length) syncWatches([]);
    setView(same ? app.view : { level: 'chip' });
  }

  async function catchUp() {
    if (!app.session) return;
    try {
      const data = await S.api.updates(app.session.id, player.records.length);
      data.updates.forEach((record) => player.add(record));
    } catch (error) {
      $('formError').textContent = error.message;
    }
  }

  async function onEvent(message) {
    if (message.type === 'hello') {
      setState(message.state);
      await catchUp();
    } else if (message.type === 'ready') {
      player.reset();
      applyReady(message);
    } else if (message.type === 'update') {
      player.add(message.record);
    } else if (message.type === 'state') {
      setState(message);
    } else if (message.type === 'error') {
      $('formError').textContent = message.message;
    }
  }

  /* Start keeps the current placement edits while the workload is unchanged. */
  function start() {
    const same = app.session && app.session.workload === $('workload').value;
    return startSession(same ? app.applied : {});
  }

  async function startSession(coreMap) {
    $('formError').textContent = '';
    const button = $('btnStart');
    const label = button.textContent;
    button.disabled = true;
    button.textContent = 'Building…';
    const body = { workload: $('workload').value, parameters: S.params.read($('params')) };
    const horizon = $('horizon').value.trim();
    if (horizon) body.horizon = /^\d+$/.test(horizon) ? Number(horizon) : horizon;
    if (coreMap && Object.keys(coreMap).length) body.core_map = coreMap;
    if ($('traceLevel').value) body.trace_level = $('traceLevel').value;
    if (app.breakpoints.length) body.breakpoints = app.breakpoints;
    try {
      const created = await S.api.create(body);
      const previous = app.session;
      if (app.socket) app.socket.close();
      if (previous) S.api.remove(previous.id).catch(() => {});
      open(created);
    } catch (error) {
      $('formError').textContent = error.message;
      button.textContent = label;
    } finally {
      button.disabled = false;
    }
  }

  /* Show a session and follow it. The id goes in the URL so a reload resumes
     the same session; the WebSocket hello then triggers the catch-up fetch. */
  function open(snapshot) {
    app.session = null;  // a different session: applyReady keeps only matching watches
    player.reset();
    applyReady(snapshot);
    history.replaceState(null, '', '#session=' + snapshot.id);
    app.socket = S.api.connect(snapshot.id, onEvent);
    $('btnStart').textContent = 'Rebuild session';
  }

  async function resume() {
    const match = /^#session=(\w+)$/.exec(location.hash);
    if (!match) return;
    try {
      const snapshot = await S.api.get(match[1]);
      $('workload').value = snapshot.workload;
      renderParams();
      const values = (snapshot.manifest && snapshot.manifest.parameters) || {};
      for (const input of $('params').querySelectorAll('[data-param]')) {
        const name = input.getAttribute('data-param');
        if (name in values) input.value = values[name];
      }
      open(snapshot);
    } catch (error) {
      history.replaceState(null, '', location.pathname);
      $('formError').textContent = 'The session in the address no longer exists. Start a new one.';
    }
  }

  async function command(action) {
    if (!app.session) return;
    try {
      await action(app.session.id);
    } catch (error) {
      $('formError').textContent = error.message;
    }
  }

  function renderParams() {
    const workload = app.workloads.find((w) => w.name === $('workload').value);
    S.params.render($('params'), workload ? workload.parameters : []);
  }

  async function init() {
    try {
      app.workloads = await S.api.workloads();
    } catch (error) {
      $('formError').textContent = error.message;
      return;
    }
    for (const workload of app.workloads) S.html($('workload'), 'option', { value: workload.name }, workload.name);
    const preferred = app.workloads.find((w) => w.name !== 'sanafe-files');
    if (preferred) $('workload').value = preferred.name;
    renderParams();
    await resume();
  }

  $('workload').addEventListener('change', renderParams);
  $('btnStart').addEventListener('click', start);
  $('btnStep').addEventListener('click', () => command((id) => S.api.step(id, 1)));
  $('btnRunN').addEventListener('click', () => {
    const raw = $('runN').value.trim();
    command((id) => S.api.step(id, /^\d+$/.test(raw) ? Number(raw) : raw));
  });
  $('btnRun').addEventListener('click', () => command(S.api.run));
  $('btnPause').addEventListener('click', () => command(S.api.pause));
  $('btnReset').addEventListener('click', () => command(S.api.reset));
  $('clock').addEventListener('change', () => {
    player.clock = $('clock').value;
    chip.clock = mini.clock = $('clock').value;
    redraw();
  });
  $('speed').addEventListener('change', () => { player.duration = Number($('speed').value); });
  $('scrub').addEventListener('input', () => player.show(Number($('scrub').value) - 1));
  $('follow').addEventListener('change', () => { if ($('follow').checked) player.latest(); else player.follow = false; });
  $('btnReplay').addEventListener('click', () => player.replay());
  $('btnLatest').addEventListener('click', () => player.latest());
  $('mChip').addEventListener('click', () => setView({ mode: 'chip', level: 'chip' }));
  $('mNet').addEventListener('click', () => setView({ mode: 'net' }));
  $('minimap').addEventListener('click', () => { setView({ level: 'chip' }); select({ kind: 'chip' }); });
  for (const tab of document.querySelectorAll('#tabs button')) {
    tab.addEventListener('click', () => {
      for (const other of document.querySelectorAll('#tabs button')) other.className = '';
      tab.className = 'on';
      app.tab = tab.getAttribute('data-tab');
      const dock = $('dock');
      dock.innerHTML = '';
      delete dock.dataset.session;
      delete dock.dataset.compare;
      redraw();
    });
  }
  S.rail.bindBreakpointForm($('bpForm'), addBreakpoint);
  renderRail();
  init();
})();
