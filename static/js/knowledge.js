/* AgentMem — Knowledge base console (progressive disclosure: status → index → search) */

async function loadKB() {
  const [st, idx] = await Promise.all([api('/kb/status'), api('/kb/index')]);
  const box = $('kb-status');
  if (box) {
    if (!st) {
      box.innerHTML = '<div class="empty">KB status unavailable</div>';
    } else {
      box.innerHTML = `<div style="font-size:12px;line-height:1.7">
        <div><b>Docs</b> ${st.docs ?? 0} · <b>Chunks</b> ${st.chunks ?? 0}</div>
        <div><b>BM25</b> ${st.bm25_indexed ?? st.bm25_ready ?? 0} indexed · ready=${st.bm25_ready ?? '—'}</div>
        <div><b>Embed</b> ${esc(st.embed_sig || '?')} · dims=${st.dims ?? '?'}</div>
        <div><b>Dims OK</b> ${st.dims_verified ? 'yes' : 'no'} · stale=${st.stale_embed ?? 0}</div>
      </div>`;
    }
  }
  const ib = $('kb-index');
  if (ib) {
    const docs = idx?.docs || [];
    if (!docs.length) {
      ib.innerHTML = '<div class="empty">No documents — POST /kb/ingest</div>';
    } else {
      ib.innerHTML = docs.slice(0, 80).map(d =>
        `<div style="padding:6px 0;border-bottom:1px solid var(--border);font-size:11px">
          <div style="font-weight:700">${esc(d.title || d.id)}</div>
          <div style="color:var(--text3)">${esc((d.description || '').slice(0, 140))}</div>
          <div style="color:var(--text3);font-size:10px">${esc(d.id)} · ${d.chunks || 0} chunks · ${esc(d.library || '')}</div>
        </div>`
      ).join('');
    }
  }
}

async function kbSearch() {
  const q = $('kbq')?.value?.trim() || '';
  const hits = $('kb-hits');
  if (!hits) return;
  if (!q) { hits.innerHTML = '<div class="empty">Enter a query</div>'; return; }
  hits.innerHTML = '<div class="empty">Searching…</div>';
  const data = await post('/kb/search', { query: q, limit: 12 });
  const chunks = data?.chunks || [];
  if (!chunks.length) {
    hits.innerHTML = '<div class="empty">No KB hits</div>';
    return;
  }
  hits.innerHTML = chunks.map(c =>
    `<div style="padding:8px 0;border-bottom:1px solid var(--border);font-size:11px">
      <div style="font-weight:700">${esc(c.title || c.doc)} <span style="color:var(--text3);font-weight:400">${esc(c.section || '')}</span></div>
      <div style="white-space:pre-wrap;color:var(--text2)">${esc((c.text || '').slice(0, 420))}</div>
      <div style="color:var(--text3);font-size:10px">${esc(c.path || '')}</div>
    </div>`
  ).join('');
}

window.loadKB = loadKB;
window.kbSearch = kbSearch;
