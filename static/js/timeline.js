/* ═══════════════════════════════════════════════════════════════════════════
   AgentMem — Timeline: chronological memory view
   ═══════════════════════════════════════════════════════════════════════════ */

async function loadTL() {
  const q = $('tlq')?.value || 'recent activity';
  const sid = $('tl-sess')?.value || '';
  const data = await post('/timeline', { query: q, limit: 40, session_id: sid || undefined });
  const items = (data?.timeline || []).sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));
  if (!items.length) { $('tllist').innerHTML = '<div class="empty">No timeline entries</div>'; return }

  $('tllist').innerHTML = '<div class="tlc"><div class="tlln"></div>' + items.map(it => {
    const cat = it.category || 'general', cls = cat === 'episode' ? 'ep' : cat === 'fact' ? 'ft' : cat === 'procedure' ? 'pr' : '';
    return `<div class="tli ${cls}"><div class="tlt">${fmtD(it.timestamp)}</div><div class="tlc2">${esc((it.content || '').slice(0, 250))}</div></div>`;
  }).join('') + '</div>';
}

window.loadTL = loadTL;
