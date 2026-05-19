/* ═══════════════════════════════════════════════════════════════════════════
   AgentMem — Procedures: procedural memory browser
   ═══════════════════════════════════════════════════════════════════════════ */

async function loadProcs() {
  const q = $('pq')?.value || '';
  let data;
  if (q) {
    data = await post('/recall-procedures', { query: q, limit: 30 });
  } else {
    data = await api('/procedural?limit=50');
  }

  S.procL = 1;
  const list = data?.procedures || data?.results || [];
  if (!list.length) { $('plist').innerHTML = '<div class="empty">No procedures found</div>'; return }

  $('plist').innerHTML = list.map(p => {
    const task = p.task || p.content || p.name || '';
    const steps = p.steps || p.content_steps || [];
    return `<div class="mc md" onclick="showProcDetail(this)" data-json="${esc(JSON.stringify(p).slice(0, 800))}">
      <div class="mh">
        <span class="tag tag-procedure">procedure</span>
        <span class="mi">${fmtR(p.timestamp || p.ts || Date.now())}</span>
      </div>
      <div class="mt">${esc(task.slice(0, 200))}</div>
      ${steps.length ? `<div class="mm">${steps.length} steps</div>` : ''}
    </div>`;
  }).join('');
}

function showProcDetail(el) {
  let p = {};
  try { p = JSON.parse(el.dataset.json || '{}') } catch {}
  const steps = p.steps || p.content_steps || [];
  let html = `<h3>Procedure Detail</h3>
    <div class="dr"><span class="dk">Task</span><span class="dv">${esc(p.task || p.name || '—')}</span></div>
    <div class="dr"><span class="dk">UID</span><span class="dv"><code style="color:var(--accent)">${esc(p.uid || p.id || '—')}</code></span></div>`;
  if (steps.length) {
    html += `<div style="margin-top:8px"><strong style="font-size:10px;color:var(--text3);text-transform:uppercase;letter-spacing:.1em">Steps</strong>`;
    steps.forEach((s, i) => { html += `<div style="padding:4px 0 4px 12px;border-left:2px solid var(--orange);margin:4px 0;font-size:12px">${i + 1}. ${esc(typeof s === 'string' ? s : JSON.stringify(s))}</div>` });
    html += '</div>';
  }
  html += `<div class="ma"><button class="btn" onclick="closeModal()">Close</button></div>`;
  openModal(html);
}

window.loadProcs = loadProcs;
