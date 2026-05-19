/* ═══════════════════════════════════════════════════════════════════════════
   AgentMem — Logs: live log viewer
   ═══════════════════════════════════════════════════════════════════════════ */

function initLogs() {
  const box = $('lbox');
  if (!box) return;
  // Render existing logs
  box.innerHTML = S.logs.map(l => {
    const cls = l.level === 'error' ? 'error' : l.level === 'warn' ? 'warn' : l.level === 'dim' ? 'dim' : 'info';
    return `<div class="ll ${cls}">${l.ts} ${esc(l.msg)}</div>`;
  }).join('');
  box.scrollTop = box.scrollHeight;
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
window.clearLogs = clearLogs;
