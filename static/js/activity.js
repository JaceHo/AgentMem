/* ═══════════════════════════════════════════════════════════════════════════
   AgentMem — Activity: real-time activity feed
   ═══════════════════════════════════════════════════════════════════════════ */

async function loadAct() {
  const data = await api('/observations?limit=50');
  const items = data?.observations || [];
  if (!items.length) { $('act-list').innerHTML = '<div class="empty">No recent activity</div>'; return }

  $('act-list').innerHTML = items.map(it => {
    const action = it.action || it.type || 'store';
    const cat = it.category || 'general';
    return `<div class="mc lo">
      <div class="mh">
        <span class="aeact ${action}">${esc(action)}</span>
        <span class="tag ${tagC(cat)}">${esc(cat)}</span>
        <span class="mi">${fmtR(it.timestamp || it.ts)}</span>
      </div>
      <div class="mt">${esc((it.content || '').slice(0, 150))}</div>
    </div>`;
  }).join('');
}

window.loadAct = loadAct;
