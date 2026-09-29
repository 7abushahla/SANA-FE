/* Page controller: session lifecycle, events, and rendering. */
(function () {
  const S = window.Studio;
  const $ = (id) => document.getElementById(id);
  const app = { workloads: [], session: null, socket: null, selection: { kind: 'chip' }, tab: 'timeline', renderKey: null, alive: true };

  const chip = new S.Chip($('chip'), (selection) => {
    app.selection = selection;
    chip.selected = selection.kind === 'core' ? selection.key : null;
    app.renderKey = null;
    player.dirty = true;
  });
  const player = new S.Player(frame);

  function frame() {
    const record = player.current();
    chip.render(record, player.t);
    $('clockText').textContent = record
      ? 'update ' + record.update + ' · modeled time ' + S.fmtTime(player.t) + ' of ' + S.fmtTime(record.step_time)
      : '–';
    const phase = $('phase');
    if (record) {
      const barrier = player.t >= record.last_activity;
      phase.textContent = barrier ? 'barrier: waiting for the update to close' : 'processing and spike distribution (overlapping)';
      phase.className = barrier ? 'phase barrier' : 'phase';
    } else {
      phase.textContent = '';
      phase.className = 'phase';
    }
    if (app.tab === 'timeline') S.timeline.render($('dock'), record, player.t, chip.owner);
    const key = [player.index, player.records.length, app.tab, app.selection.kind, app.selection.key, !!app.session].join('|');
    if (key === app.renderKey) return;
    app.renderKey = key;
    if (app.tab === 'perf') S.perf.render($('dock'), player.records, record ? record.update : null);
    if (app.tab === 'messages') S.messages.render($('dock'), record);
    S.inspector.render($('insTitle'), $('inspector'), app.selection, record, app.session);
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
    app.session = Object.assign({}, app.session, ready);
    chip.selected = null;
    app.selection = { kind: 'chip' };
    chip.build(ready.layout, ready.network, ready.metadata || {});
    $('badge').textContent = ready.badge;
    const params = (ready.manifest && ready.manifest.parameters) || {};
    $('sessionName').textContent = app.session.workload + ' · ' + Object.keys(params).map((k) => k + '=' + params[k]).join(' ');
    const rule = ready.metadata && ready.metadata.horizon_rule;
    $('sessionInfo').textContent = 'Horizon: ' + ready.horizon + ' updates' + (rule ? ' (' + rule + ')' : '') +
      '. Run length is set by Step, Run n, or Run to horizon.';
    $('uHorizon').textContent = ready.horizon;
    setState(typeof ready.state === 'object' ? ready.state : { state: ready.state, horizon: ready.horizon });
    app.renderKey = null;
    player.dirty = true;
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

  async function start() {
    $('formError').textContent = '';
    const button = $('btnStart');
    const label = button.textContent;
    button.disabled = true;
    button.textContent = 'Building…';
    const body = { workload: $('workload').value, parameters: S.params.read($('params')) };
    const horizon = $('horizon').value.trim();
    if (horizon) body.horizon = /^\d+$/.test(horizon) ? Number(horizon) : horizon;
    try {
      const created = await S.api.create(body);
      const previous = app.session;
      if (app.socket) app.socket.close();
      if (previous) S.api.remove(previous.id).catch(() => {});
      app.session = null;
      player.reset();
      applyReady(created);
      app.socket = S.api.connect(created.id, onEvent);
      button.textContent = 'Rebuild session';
    } catch (error) {
      $('formError').textContent = error.message;
      button.textContent = label;
    } finally {
      button.disabled = false;
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
  $('clock').addEventListener('change', () => { player.clock = $('clock').value; });
  $('speed').addEventListener('change', () => { player.duration = Number($('speed').value); });
  $('scrub').addEventListener('input', () => player.show(Number($('scrub').value) - 1));
  $('follow').addEventListener('change', () => { if ($('follow').checked) player.latest(); else player.follow = false; });
  $('btnReplay').addEventListener('click', () => player.replay());
  $('btnLatest').addEventListener('click', () => player.latest());
  for (const tab of document.querySelectorAll('#tabs button')) {
    tab.addEventListener('click', () => {
      for (const other of document.querySelectorAll('#tabs button')) other.className = '';
      tab.className = 'on';
      app.tab = tab.getAttribute('data-tab');
      app.renderKey = null;
      player.dirty = true;
    });
  }
  init();
})();
