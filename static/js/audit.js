/* ═══════════════════════════════════════════════════════════════════════════
   AgentMem — Audit: mutation audit log
   ═══════════════════════════════════════════════════════════════════════════ */

async function loadAudit() {
  const data = await api('/observations?limit=100');
  const items = data?.observations || [];
  if (!items.length) { $('audit-list').innerHTML = '<div class="empty">No audit entries</div>'; return }

  $('audit-list').innerHTML = items.slice(0, 50).map(it => {
    const action = it.action || it.type || 'store';
    return `<div class="ae">
      <span class="aets">${fmtD(it.timestamp || it.ts)}</span>
      <span class="aeact ${action}">${esc(action)}</span>
      <span style="color:var(--text);font-size:12px">${esc((it.content || it.detail || '').slice(0, 120))}</span>
    </div>`;
  }).join('');
}

window.loadAudit = loadAudit;
