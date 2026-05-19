/* ═══════════════════════════════════════════════════════════════════════════
   AgentMem — Search: hybrid BM25 + vector + graph search
   ═══════════════════════════════════════════════════════════════════════════ */

async function doSearch() {
  const q = $('sq')?.value || '';
  if (!q) return;
  const mode = $('smode')?.value || 'hybrid';
  const limit = parseInt($('slimit')?.value) || 20;

  let data;
  if (mode === 'hybrid') {
    data = await post('/smart-search', { query: q, limit });
  } else if (mode === 'vector') {
    data = await post('/recall', { query: q, limit });
  } else if (mode === 'bm25') {
    data = await post('/search', { query: q, limit });
  } else if (mode === 'graph') {
    data = await post('/graph/recall', { entity: q, limit });
  }

  const results = data?.results || data?.facts || data?.memories || [];
  S.searchL = 1;

  if (!results.length) { $('sresults').innerHTML = '<div class="empty">No results found</div>'; return }

  // Show search mode info
  const sources = data?.sources || {};
  let srcInfo = '';
  if (sources.bm25 || sources.vector || sources.graph) {
    srcInfo = `<div style="font-size:10px;color:var(--text3);margin-bottom:8px">Sources: ${[
      sources.bm25 ? `BM25(${sources.bm25})` : '',
      sources.vector ? `Vector(${sources.vector})` : '',
      sources.graph ? `Graph(${sources.graph})` : ''
    ].filter(Boolean).join(' + ')}</div>`;
  }

  $('sresults').innerHTML = srcInfo + results.map(r => {
    const imp = r.importance || r.score || 0;
    const cat = r.category || r.type || 'general';
    const content = r.content || r.text || '';
    // Highlight query terms
    const highlighted = content.replace(new RegExp(`(${q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi'), '<mark>$1</mark>');
    return `<div class="mc ${impC(imp)}" onclick="window.showMemDetail?.(this)" data-uid="${esc(r.uid || r.id || '')}" data-json="${esc(JSON.stringify(r).slice(0, 500))}">
      <div class="mh">
        <span class="tag ${tagC(cat)}">${esc(cat)}</span>
        <span class="mi">${fmtR(r.timestamp || r.ts)}</span>
        <div class="score-bar"><div class="score-fill" style="width:${Math.round(imp * 100)}%;background:${imp >= .7 ? 'var(--red)' : imp >= .4 ? 'var(--yellow)' : 'var(--green)'}"></div></div>
      </div>
      <div class="mt">${highlighted.slice(0, 300)}</div>
    </div>`;
  }).join('');
}

window.doSearch = doSearch;
