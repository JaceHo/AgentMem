/* ═══════════════════════════════════════════════════════════════════════════
   AgentMem — Graph: interactive knowledge graph with force layout
   ═══════════════════════════════════════════════════════════════════════════ */

let G = { nodes: [], edges: [], loaded: false, zoom: 1, panX: 0, panY: 0, drag: null, hover: null, filters: {} };

async function loadGraph() {
  const q = $('gq')?.value || '';
  const [stats, facts] = await Promise.all([
    api('/graph/stats'),
    q ? post('/smart-search', { query: q, limit: 50 }) : api('/memories?limit=80')
  ]);
  if (!stats) return;

  const nc = stats.total_nodes || stats.nodes || 0, ec = stats.total_edges || stats.edges || 0;
  $('ginfo').textContent = `${nc} nodes · ${ec} edges`;
  G.nodes = []; G.edges = []; G.loaded = true;

  // Build category nodes
  const cats = ['fact', 'episode', 'procedure', 'preference', 'rule', 'identity'];
  const catColors = { fact: '#26c6da', episode: '#ba68c8', procedure: '#ff9800', preference: '#4caf50', rule: '#ef5350', identity: '#26c6da' };
  cats.forEach(c => { G.nodes.push({ id: c, label: c, color: catColors[c] || '#4fc3f7', type: 'category', r: 22 }) });

  // Build entity nodes from memories
  const memList = facts?.memories || facts?.results || [];
  const entityMap = new Map();
  memList.forEach(m => {
    const content = m.content || m.text || '';
    const cat = m.category || 'general';
    // Extract entities: capitalized phrases
    const matches = content.match(/(?:^|\s)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)/g);
    if (matches) matches.forEach(e => {
      const name = e.trim();
      if (name.length >= 3 && name.length <= 40 && !entityMap.has(name) && entityMap.size < 80) {
        entityMap.set(name, cat);
      }
    });
  });

  let idx = 0;
  entityMap.forEach((cat, name) => {
    const nodeId = 'e' + idx;
    G.nodes.push({ id: nodeId, label: name, color: '#4fc3f7', type: 'entity', r: 10 });
    const target = cats.includes(cat) ? cat : 'fact';
    G.edges.push({ source: target, target: nodeId });
    idx++;
  });

  // Force layout
  const canvas = $('gcanvas');
  const W = canvas.width, H = canvas.height;
  G.nodes.forEach(n => { n.x = W / 2 + (Math.random() - .5) * 400; n.y = H / 2 + (Math.random() - .5) * 400; n.vx = 0; n.vy = 0 });

  for (let iter = 0; iter < 100; iter++) {
    // Repulsion
    G.nodes.forEach(a => { G.nodes.forEach(b => { if (a === b) return; const dx = b.x - a.x, dy = b.y - a.y, d = Math.max(1, Math.sqrt(dx * dx + dy * dy)), f = 1200 / (d * d); b.vx += dx / d * f; b.vy += dy / d * f }) });
    // Attraction along edges
    G.edges.forEach(e => { const a = G.nodes.find(n => n.id === e.source), b = G.nodes.find(n => n.id === e.target); if (!a || !b) return; const dx = b.x - a.x, dy = b.y - a.y, d = Math.max(1, Math.sqrt(dx * dx + dy * dy)), f = (d - 100) * .04; b.vx += dx / d * f; b.vy += dy / d * f; a.vx -= dx / d * f; a.vy -= dy / d * f });
    // Center gravity
    G.nodes.forEach(n => { n.vx += (W / 2 - n.x) * .001; n.vy += (H / 2 - n.y) * .001 });
    // Damping + bounds
    G.nodes.forEach(n => { n.vx *= .88; n.vy *= .88; n.x += n.vx; n.y += n.vy; n.x = Math.max(30, Math.min(W - 30, n.x)); n.y = Math.max(30, Math.min(H - 30, n.y)) });
  }

  renderG(); renderGLegend(); renderGSidebar();
}

function renderG() {
  const c = $('gcanvas'), ctx = c.getContext('2d'), W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  ctx.save(); ctx.translate(G.panX, G.panY); ctx.scale(G.zoom, G.zoom);

  // Grid
  ctx.strokeStyle = isDark() ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.04)'; ctx.lineWidth = .5;
  for (let x = 0; x < W; x += 24) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke() }
  for (let y = 0; y < H; y += 24) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke() }

  // Edges
  ctx.lineWidth = 1;
  G.edges.forEach(e => {
    const a = G.nodes.find(n => n.id === e.source), b = G.nodes.find(n => n.id === e.target);
    if (!a || !b || a.hidden || b.hidden) return;
    const isH = G.hover && (G.hover.id === a.id || G.hover.id === b.id);
    ctx.strokeStyle = isH ? 'rgba(79,195,247,0.5)' : 'rgba(79,195,247,0.12)';
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  });

  // Nodes
  G.nodes.forEach(n => {
    if (n.hidden) return;
    const isH = G.hover && G.hover.id === n.id;
    // Glow
    if (isH || n.type === 'category') {
      ctx.save(); ctx.shadowColor = n.color; ctx.shadowBlur = isH ? 24 : 10;
      ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = n.color + '33'; ctx.fill(); ctx.restore();
    }
    // Circle
    ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
    ctx.fillStyle = isH ? n.color : n.color + 'aa'; ctx.fill();
    ctx.strokeStyle = isH ? '#fff' : 'rgba(255,255,255,0.1)'; ctx.lineWidth = isH ? 2 : 1; ctx.stroke();
    // Label
    ctx.fillStyle = isDark() ? '#d4dce8' : '#212529';
    ctx.font = (n.type === 'category' ? 'bold 11px' : '9px') + ' sans-serif';
    ctx.textAlign = 'center'; ctx.fillText(n.label.slice(0, 18), n.x, n.y + n.r + 13);
  });

  ctx.restore();
}

function renderGLegend() {
  $('glegend').innerHTML = [
    { l: 'Fact', c: '#26c6da' }, { l: 'Episode', c: '#ba68c8' }, { l: 'Procedure', c: '#ff9800' },
    { l: 'Preference', c: '#4caf50' }, { l: 'Rule', c: '#ef5350' }, { l: 'Entity', c: '#4fc3f7' }
  ].map(c => `<div class="legi"><div class="legd" style="background:${c.c}"></div>${c.l}</div>`).join('');
}

function renderGSidebar() {
  const filters = ['fact', 'episode', 'procedure', 'preference', 'rule', 'identity'];
  $('g-filters').innerHTML = filters.map(c =>
    `<div style="margin-bottom:4px"><label style="font-size:11px;display:flex;align-items:center;gap:4px;cursor:pointer">
      <input type="checkbox" checked onchange="toggleGFilter('${c}',this.checked)"> ${c}</label></div>`
  ).join('');
  $('g-sidebar-info').innerHTML = `<div style="font-size:11px;color:var(--text2)">${G.nodes.length} nodes<br>${G.edges.length} edges</div>`;
}

function toggleGFilter(cat, on) { G.nodes.forEach(n => { if (n.type === 'category' && n.id === cat) n.hidden = !on }) }
function toggleGSidebar() { $('g-sidebar').classList.toggle('open') }
function gZoom(f) { G.zoom *= f; G.zoom = Math.max(.2, Math.min(5, G.zoom)) }
function gReset() { G.zoom = 1; G.panX = 0; G.panY = 0 }

// Canvas interactions
document.addEventListener('DOMContentLoaded', () => {
  const gc = $('gcanvas'); if (!gc) return;
  gc.addEventListener('wheel', e => { e.preventDefault(); G.zoom *= e.deltaY < 0 ? 1.1 : .9; G.zoom = Math.max(.2, Math.min(5, G.zoom)) }, { passive: false });
  gc.addEventListener('mousedown', e => {
    const r = gc.getBoundingClientRect(), mx = (e.clientX - r.left - G.panX) / G.zoom, my = (e.clientY - r.top - G.panY) / G.zoom;
    G.nodes.forEach(n => { if (Math.hypot(n.x - mx, n.y - my) < n.r) G.drag = n });
    if (!G.drag) G._dragStart = { x: e.clientX - G.panX, y: e.clientY - G.panY };
  });
  gc.addEventListener('mousemove', e => {
    const r = gc.getBoundingClientRect(), mx = (e.clientX - r.left - G.panX) / G.zoom, my = (e.clientY - r.top - G.panY) / G.zoom;
    G.hover = null; G.nodes.forEach(n => { if (!n.hidden && Math.hypot(n.x - mx, n.y - my) < n.r) G.hover = n });
    if (G.drag) { G.drag.x = mx; G.drag.y = my }
    if (G._dragStart) { G.panX = e.clientX - G._dragStart.x; G.panY = e.clientY - G._dragStart.y }
    gc.style.cursor = G.hover ? 'pointer' : G.drag ? 'grabbing' : 'grab';
  });
  gc.addEventListener('mouseup', () => { G.drag = null; G._dragStart = null });
  gc.addEventListener('click', () => { if (G.hover && G.hover.type === 'entity') { switchTab('search'); $('sq').value = G.hover.label; window.doSearch?.() } });
  // Auto-resize canvas
  const resizeCanvas = () => { const wrap = gc.parentElement; if (wrap) { gc.width = wrap.clientWidth; gc.height = Math.max(400, window.innerHeight - 200) } };
  window.addEventListener('resize', () => { resizeCanvas(); if (G.loaded) renderG() });
  resizeCanvas();
});

window.loadGraph = loadGraph;
window.toggleGFilter = toggleGFilter;
window.toggleGSidebar = toggleGSidebar;
window.gZoom = gZoom;
window.gReset = gReset;
