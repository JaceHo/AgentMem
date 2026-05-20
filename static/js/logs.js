/* ═══════════════════════════════════════════════════════════════════════════
   AgentMem — Logs: live log viewer
   ═══════════════════════════════════════════════════════════════════════════ */

function initLogs() {
  const box = $('lbox');
  if (!box) return;
  box.innerHTML = S.logs.map(l => {
    const cls = l.level === 'error' ? 'error' : l.level === 'warn' ? 'warn' : l.level === 'dim' ? 'dim' : 'info';
    return `<div class="ll ${cls}">${l.ts} ${esc(l.msg)}</div>`;
  }).join('');
  box.scrollTop = box.scrollHeight;
}

async function loadLogs() {
  if (S._logsLoaded) { initLogs(); return; }
  S._logsLoaded = true;
  const data = await api('/logs/recent?limit=500');
  const entries = Array.isArray(data) ? data : (data?.logs || []);
  entries.forEach(e => {
    const ts = e.ts || e.time || fmtT(Date.now());
    const level = e.color || (e.level || 'info').toLowerCase();
    const msg = e.msg || e.message || '';
    S.logs.push({ level, msg, ts });
  });
  if (S.logs.length > 500) S.logs.splice(0, S.logs.length - 500);
  const ct = $('log-ct');
  if (ct) ct.textContent = S.logs.length;
  initLogs();
}

function clearLogs() {
  S.logs = [];
  $('log-ct').textContent = '0';
  $('lbox').innerHTML = '';
}

// SSE log stream (if available)
function connectLogSSE() {
  try {
    const es = new EventSource('/logs/stream');
    es.onmessage = ev => {
      try {
        const d = JSON.parse(ev.data);
        addLog(d.level || 'info', d.msg || d.message || ev.data);
      } catch { addLog('info', ev.data) }
    };
    es.onerror = () => {}; // silent reconnect
  } catch {}
}

document.addEventListener('DOMContentLoaded', () => {
  // When switching to logs tab, render
  const origSwitch = window.switchTab;
  // Init SSE
  connectLogSSE();
});

window.initLogs = initLogs;
window.loadLogs = loadLogs;
window.clearLogs = clearLogs;
