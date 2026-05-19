/* ═══════════════════════════════════════════════════════════════════════════
   AgentMem — Memories: browse, search, detail, forget
   ═══════════════════════════════════════════════════════════════════════════ */

async function loadMem(refresh) {
  const q = refresh ? '' : $('mq')?.value || '';
  const tier = $('mtier')?.value || 'semantic';
  const cat = $('mcat')?.value || '';
  let data;

  if (q) {
    data = await post('/smart-search', { query: q, limit: 30, category: cat || undefined });
    if (data?.results) data.memories = data.results;
  } else {
    data = await api('/' + tier + '?limit=50');
    // Normalize response from various endpoints
    const raw = data?.facts || data?.episodes || data?.procedures || data?.memories || data?.results || [];
    data = data || {};
    data.memories = raw.map(m => ({ content: m.content || m.text || m.task || '', category: m.category || (tier === 'procedural' ? 'procedure' : tier === 'episodes' ? 'episode' : 'fact'), score: m.score || m.importance, uid: m.uid, id: m.id, timestamp: m.timestamp || m.ts }));
  }

  S.memL = 1;
  const list = data?.memories || data?.results || [];
  if (!list.length) { $('mlist').innerHTML = '<div class="empty">No memories found</div>'; return }

  $('mlist').innerHTML = list.map(m => {
    const imp = m.importance || m.score || .5, ic = impC(imp), cat2 = m.category || 'general';
    return `<div class="mc ${ic}" onclick="showMemDetail(this)" data-uid="${esc(m.uid || m.id || '')}" data-json="${esc(JSON.stringify(m).slice(0, 500))}">
      <div class="mh">
        <span class="tag ${tagC(cat2)}">${esc(cat2)}</span>
        <span class="mi">${fmtR(m.timestamp || m.ts || Date.now())}</span>
        <div class="score-bar"><div class="score-fill" style="width:${Math.round(imp * 100)}%;background:${imp >= .7 ? 'var(--red)' : imp >= .4 ? 'var(--yellow)' : 'var(--green)'}"></div></div>
      </div>
      <div class="mt">${esc((m.content || m.text || '').slice(0, 200))}</div>
      ${m.tags ? `<div class="mm">${(m.tags || []).slice(0, 5).map(t => `<span class="tag tag-general">${esc(t)}</span>`).join('')}</div>` : ''}
    </div>`;
  }).join('');
}

function showMemDetail(el) {
  const uid = el.dataset.uid; if (!uid) return;
  let m = {};
  try { m = JSON.parse(el.dataset.json || '{}') } catch {}
  const imp = m.importance || m.score || 0;
  let html = `<h3>Memory Detail</h3>
    <div class="dr"><span class="dk">UID</span><span class="dv"><code style="color:var(--accent)">${esc(uid)}</code></span></div>
    <div class="dr"><span class="dk">Category</span><span class="dv"><span class="tag ${tagC(m.category || 'general')}">${esc(m.category || '—')}</span></span></div>
    <div class="dr"><span class="dk">Importance</span><span class="dv">${(imp * 100).toFixed(0)}%</span></div>
    <div class="dr"><span class="dk">Time</span><span class="dv">${fmtD(m.timestamp || m.ts)}</span></div>
    <div style="margin-top:10px;padding:10px;background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius);font-size:12px;word-break:break-word;max-height:200px;overflow-y:auto">${esc(m.content || m.text || '—')}</div>
    <div class="ma">
      <button class="btn dan" onclick="forgetMem('${esc(uid)}')">Forget</button>
      <button class="btn" onclick="closeModal()">Close</button>
    </div>`;
  openModal(html);
}

async function forgetMem(uid) {
  if (!uid) return;
  if (!confirm('Forget this memory?')) return;
  const r = await post('/forget', { key: uid });
  addLog('info', 'Forgot: ' + uid);
  closeModal(); loadMem(true);
}

window.loadMem = loadMem;
window.showMemDetail = showMemDetail;
window.forgetMem = forgetMem;
