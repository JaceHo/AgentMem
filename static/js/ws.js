/* ═══════════════════════════════════════════════════════════════════════════
   AgentMem — WebSocket: real-time connection with polling fallback
   ═══════════════════════════════════════════════════════════════════════════ */

let wsRetries = 0, pollTimer = null;

function connectWs() {
  if (wsRetries >= S.WS_MAX) { startPoll(); return }
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  try { S.ws = new WebSocket(proto + '://' + location.host + '/ws') } catch { wsRetries++; setTimeout(connectWs, 3000); return }
  S.ws.onopen = () => {
    wsRetries = 0; stopPoll(); updateWs(true);
    addLog('info', 'WebSocket connected');
    S.ws.send(JSON.stringify({ type: 'subscribe' }));
    $('live-cnt').style.display = 'flex';
  };
  S.ws.onclose = () => {
    updateWs(false); addLog('dim', 'WS disconnected — retry 3s');
    $('live-cnt').style.display = 'none';
    wsRetries++; setTimeout(connectWs, 3000);
  };
  S.ws.onerror = () => {};
  S.ws.onmessage = ev => { try { handleWs(JSON.parse(ev.data)) } catch {} };
}

function startPoll() {
  if (pollTimer) return;
  $('ws-badge').className = 'badge badge-ws poll'; $('ws-badge').textContent = 'POLL';
  addLog('warn', 'Falling back to polling (10s)');
  pollTimer = setInterval(async () => {
    const h = await api('/health'); if (h) handleWs({ event: 'health', data: h });
    const s = await api('/stats'); if (s) handleWs({ event: 'stats', data: s });
  }, 10000);
}

function stopPoll() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null } }

function updateWs(ok) {
  $('ws-badge').className = 'badge badge-ws' + (ok ? '' : ' off');
  $('ws-badge').textContent = ok ? 'LIVE' : 'OFF';
}

function handleWs(msg) {
  S.liveEvents++; S.lastEventTs = Date.now();
  const ev = msg.event || msg.type;

  if (ev === 'connected') {
    $('ver').textContent = 'v' + (msg.version || '?');
  } else if (ev === 'health' || ev === 'stats') {
    if (S.tab === 'dashboard') window.loadDash?.();
  } else if (ev === 'store' || ev === 'remember' || ev === 'forget' || ev === 'consolidate') {
    addLog('info', '[ws] ' + ev + ': ' + (msg.uid || msg.key || '').slice(0, 16));
    if (S.tab === 'dashboard') window.loadDash?.();
    if (S.tab === 'audit') window.loadAudit?.();
    if (S.tab === 'activity') window.loadAct?.();
    if (S.tab === 'graph') window.loadGraph?.();
    if (S.tab === 'memories') window.loadMem?.(true);
  } else if (ev === 'session_start' || ev === 'session_end') {
    addLog('info', '[ws] ' + ev + ': ' + (msg.session_id || '').slice(0, 16));
    if (S.tab === 'sessions') window.loadSess?.();
  }
}
