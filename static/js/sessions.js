/* ═══════════════════════════════════════════════════════════════════════════
   AgentMem — Sessions: session list and detail
   ═══════════════════════════════════════════════════════════════════════════ */

async function loadSess() {
  const data = await api('/sessions?limit=30');
  S.sessL = 1;
  const list = data?.sessions || [];

  // Populate session selectors in other tabs
  const opts = '<option value="">All sessions</option>' + list.map(s =>
    `<option value="${esc(s.session_id || s.id || '')}">${esc((s.project || s.session_id || '').slice(0, 30))}</option>`
  ).join('');
  if ($('tl-sess')) $('tl-sess').innerHTML = opts;
  if ($('rp-sess')) $('rp-sess').innerHTML = '<option value="">Select session...</option>' + list.map(s =>
    `<option value="${esc(s.session_id || s.id || '')}">${esc((s.project || s.session_id || '').slice(0, 30))}</option>`
  ).join('');

  if (!list.length) { $('slist').innerHTML = '<div class="empty">No sessions</div>'; return }

  $('slist').innerHTML = list.map(s => {
    const st = s.status || 'unknown';
    const stC = st === 'active' ? 'var(--green)' : st === 'ended' ? 'var(--text3)' : 'var(--yellow)';
    const sid = s.session_id || s.id || '';
    return `<div class="scard" onclick="showSessDetail('${esc(sid)}')">
      <div class="sh">
        <span class="si">${esc((s.project || sid).slice(0, 40))}</span>
        <span class="sm" style="color:${stC}">${esc(st)}</span>
      </div>
      <div class="sb">Obs: ${s.observation_count || s.observationCount || 0} · ${fmtR(s.started_at || s.created_at)}</div>
    </div>`;
  }).join('');
}

async function showSessDetail(sid) {
  if (!sid) return;
  const data = await api('/session/' + encodeURIComponent(sid));
  if (!data) { openModal('<h3>Session</h3><div class="empty">Not found</div><div class="ma"><button class="btn" onclick="closeModal()">Close</button></div>'); return }

  const s = data.session || data;
  let html = `<h3>Session Detail</h3>
    <div class="dr"><span class="dk">ID</span><span class="dv"><code style="color:var(--accent)">${esc(sid)}</code></span></div>
    <div class="dr"><span class="dk">Status</span><span class="dv">${esc(s.status || '—')}</span></div>
    <div class="dr"><span class="dk">Project</span><span class="dv">${esc(s.project || '—')}</span></div>
    <div class="dr"><span class="dk">Started</span><span class="dv">${fmtD(s.started_at || s.created_at)}</span></div>
    <div class="dr"><span class="dk">Observations</span><span class="dv">${s.observation_count || 0}</span></div>`;
  if (s.summary) html += `<div style="margin-top:10px;padding:10px;background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius);font-size:12px;word-break:break-word;max-height:200px;overflow-y:auto">${esc(s.summary)}</div>`;
  html += `<div class="ma"><button class="btn" onclick="closeModal()">Close</button></div>`;
  openModal(html);
}

window.loadSess = loadSess;
window.showSessDetail = showSessDetail;
