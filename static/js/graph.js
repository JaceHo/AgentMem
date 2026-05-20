/* ═══════════════════════════════════════════════════════════════════════════
   AgentMem — Graph: real knowledge graph from mem:graph:* Redis keys
   ═══════════════════════════════════════════════════════════════════════════ */

let G = {
  nodes: [], edges: [], loaded: false,
  zoom: 1, panX: 0, panY: 0,
  drag: null, hover: null, selected: null,
  raf: null, running: false,
  dpr: window.devicePixelRatio || 1,
};

const NODE_COLOR = '#4fc3f7';
const NODE_COLOR_HOT = '#ff9800';    // high-connection nodes
const NODE_COLOR_SEL = '#ba68c8';    // selected
const EDGE_COLOR = 'rgba(79,195,247,0.15)';
const EDGE_COLOR_HOT = 'rgba(79,195,247,0.55)';

async function loadGraph() {
  resizeGCanvas(); // canvas may have been 0×0 while pane was hidden
  const q = $('gq')?.value?.trim() || '';
  const [stats, graphData] = await Promise.all([
    api('/graph/stats'),
    q ? api(`/graph/${encodeURIComponent(q)}`) : api('/graph/nodes?limit=60'),
  ]);

  if (!stats) return;
  const nc = stats.total_nodes || stats.nodes || 0;
  const ec = stats.total_edges || stats.edges || 0;
  $('ginfo').textContent = `${nc} nodes · ${ec} edges total`;

  // Build node + edge lists
  let rawNodes = [], rawEdges = [];
  if (q && graphData?.neighbors) {
    // Single-entity expand mode
    rawNodes.push({ id: q, label: q.replace(/_/g, ' '), connections: graphData.count });
    graphData.neighbors.forEach(n => {
      rawNodes.push({ id: n.entity, label: n.entity.replace(/_/g, ' '), connections: n.connection_count });
      rawEdges.push({ source: q, target: n.entity, type: n.edge_type || 'related_to' });
    });
  } else {
    rawNodes = graphData?.nodes || [];
    rawEdges = graphData?.edges || [];
  }

  if (!rawNodes.length) {
    $('ginfo').textContent = 'No graph data yet.';
    return;
  }

  // Max connections for sizing
  const maxConn = Math.max(1, ...rawNodes.map(n => n.connections || 1));

  // Build node objects with layout positions
  const canvas = $('gcanvas');
  const W = canvas.width / G.dpr, H = canvas.height / G.dpr;
  const nodeById = new Map();

  G.nodes = rawNodes.map(n => {
    const conn = n.connections || 1;
    const r = 7 + Math.sqrt(conn / maxConn) * 14;  // 7–21px radius
    const node = {
      id: n.id, label: n.label || n.id,
      connections: conn, r,
      color: conn > maxConn * 0.5 ? NODE_COLOR_HOT : NODE_COLOR,
      x: W / 2 + (Math.random() - 0.5) * Math.min(W, H) * 0.7,
      y: H / 2 + (Math.random() - 0.5) * Math.min(W, H) * 0.7,
      vx: 0, vy: 0,
    };
    nodeById.set(n.id, node);
    return node;
  });

  // Only keep edges where both endpoints exist
  G.edges = rawEdges
    .filter(e => nodeById.has(e.source) && nodeById.has(e.target))
    .map(e => ({ source: e.source, target: e.target, type: e.type || 'related_to' }));

  G.loaded = true;
  G.selected = null;

  renderGSidebar();
  startSimulation();
}

// ── Physics simulation (continuous RAF) ───────────────────────────────────────

function simStep() {
  const nodes = G.nodes, edges = G.edges;
  const canvas = $('gcanvas'); if (!canvas) return;
  const W = canvas.width / G.dpr, H = canvas.height / G.dpr;

  // Repulsion between all pairs
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j];
      const dx = b.x - a.x, dy = b.y - a.y;
      const d2 = dx * dx + dy * dy || 1;
      const d = Math.sqrt(d2);
      const f = 800 / d2;
      const fx = (dx / d) * f, fy = (dy / d) * f;
      a.vx -= fx; a.vy -= fy;
      b.vx += fx; b.vy += fy;
    }
  }

  // Spring attraction along edges
  const nodeById = new Map(nodes.map(n => [n.id, n]));
  edges.forEach(e => {
    const a = nodeById.get(e.source), b = nodeById.get(e.target);
    if (!a || !b) return;
    const dx = b.x - a.x, dy = b.y - a.y;
    const d = Math.sqrt(dx * dx + dy * dy) || 1;
    const restLen = 120;
    const f = (d - restLen) * 0.03;
    const fx = (dx / d) * f, fy = (dy / d) * f;
    a.vx += fx; a.vy += fy;
    b.vx -= fx; b.vy -= fy;
  });

  // Gravity toward center
  nodes.forEach(n => {
    n.vx += (W / 2 - n.x) * 0.002;
    n.vy += (H / 2 - n.y) * 0.002;
  });

  // Damping + position update + bounds
  nodes.forEach(n => {
    if (G.drag === n) return;
    n.vx *= 0.85; n.vy *= 0.85;
    n.x += n.vx; n.y += n.vy;
    n.x = Math.max(n.r + 4, Math.min(W - n.r - 4, n.x));
    n.y = Math.max(n.r + 4, Math.min(H - n.r - 4, n.y));
  });
}

function startSimulation() {
  if (G.raf) cancelAnimationFrame(G.raf);
  G.running = true;
  let ticks = 0;
  function frame() {
    simStep();
    renderG();
    ticks++;
    // Slow down after settling (still redraws for hover/drag)
    if (ticks < 200 || G.drag || G.hover) {
      G.raf = requestAnimationFrame(frame);
    } else {
      G.raf = requestAnimationFrame(idleFrame);
    }
  }
  function idleFrame() {
    renderG();
    G.raf = requestAnimationFrame(idleFrame);
  }
  G.raf = requestAnimationFrame(frame);
}

// ── Rendering ─────────────────────────────────────────────────────────────────

function renderG() {
  const canvas = $('gcanvas'); if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const dpr = G.dpr;

  ctx.clearRect(0, 0, W, H);
  ctx.save();
  ctx.translate(G.panX * dpr, G.panY * dpr);
  ctx.scale(G.zoom * dpr, G.zoom * dpr);

  const nodeById = new Map(G.nodes.map(n => [n.id, n]));
  const selId = G.selected?.id;
  const hovId = G.hover?.id;

  // Edges
  G.edges.forEach(e => {
    const a = nodeById.get(e.source), b = nodeById.get(e.target);
    if (!a || !b) return;
    const hot = hovId === a.id || hovId === b.id || selId === a.id || selId === b.id;
    ctx.strokeStyle = hot ? EDGE_COLOR_HOT : EDGE_COLOR;
    ctx.lineWidth = hot ? 1.5 : 0.8;
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  });

  // Nodes
  G.nodes.forEach(n => {
    const isSel = n.id === selId;
    const isHov = n.id === hovId;
    const color = isSel ? NODE_COLOR_SEL : n.color;

    // Glow for selected/hovered/hot nodes
    if (isSel || isHov || n.connections > 5) {
      ctx.save();
      ctx.shadowColor = color;
      ctx.shadowBlur = isSel ? 20 : isHov ? 14 : 6;
      ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = color + '22'; ctx.fill();
      ctx.restore();
    }

    // Circle
    ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
    ctx.fillStyle = (isHov || isSel) ? color : color + 'bb';
    ctx.fill();
    ctx.strokeStyle = (isHov || isSel) ? '#fff' : 'rgba(255,255,255,0.15)';
    ctx.lineWidth = (isHov || isSel) ? 1.5 : 0.8;
    ctx.stroke();

    // Label — show always for large nodes, only on hover for small ones
    if (isHov || isSel || n.r >= 14) {
      ctx.fillStyle = isDark() ? '#d4dce8' : '#1a1a2e';
      ctx.font = `${n.r >= 14 ? 'bold ' : ''}${Math.max(8, Math.min(11, n.r))}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText(n.label.slice(0, 20), n.x, n.y + n.r + 11);
    }
  });

  ctx.restore();
}

// ── Sidebar ───────────────────────────────────────────────────────────────────

function renderGSidebar() {
  $('g-sidebar-info').innerHTML =
    `<div style="font-size:11px;color:var(--text2)">${G.nodes.length} nodes shown<br>${G.edges.length} edges shown</div>`;
  $('g-filters').innerHTML =
    `<div style="font-size:11px;color:var(--text2);line-height:1.6">
      Click node to select<br>
      Double-click to expand<br>
      Drag to reposition<br>
      Scroll to zoom
    </div>`;
}

function showNodePanel(node) {
  $('g-sidebar-info').innerHTML = `
    <div style="font-size:12px;border:1px solid var(--border);border-radius:6px;padding:10px;margin-top:8px;background:var(--bg2)">
      <div style="font-weight:700;color:var(--text);margin-bottom:6px;word-break:break-word">${esc(node.label)}</div>
      <div style="color:var(--text2);font-size:11px">${node.connections} connections</div>
      <button class="btn" style="margin-top:8px;width:100%;font-size:11px" onclick="expandNode('${esc(node.id)}')">Expand neighbors</button>
      <button class="btn" style="margin-top:4px;width:100%;font-size:11px" onclick="searchNode('${esc(node.id)}')">Search memories</button>
    </div>`;
}

async function expandNode(entityId) {
  const data = await api(`/graph/${encodeURIComponent(entityId)}`);
  if (!data?.neighbors?.length) return;

  const nodeById = new Map(G.nodes.map(n => [n.id, n]));
  const existing = G.nodes.find(n => n.id === entityId);
  const cx = existing?.x ?? $('gcanvas').width / G.dpr / 2;
  const cy = existing?.y ?? $('gcanvas').height / G.dpr / 2;

  data.neighbors.forEach(nb => {
    if (!nodeById.has(nb.entity)) {
      const r = 7 + Math.sqrt(Math.min(nb.connection_count, 20) / 20) * 14;
      G.nodes.push({
        id: nb.entity, label: nb.entity.replace(/_/g, ' '),
        connections: nb.connection_count, r,
        color: nb.connection_count > 10 ? NODE_COLOR_HOT : NODE_COLOR,
        x: cx + (Math.random() - 0.5) * 120,
        y: cy + (Math.random() - 0.5) * 120,
        vx: 0, vy: 0,
      });
    }
    const pair = [entityId, nb.entity].sort().join('|');
    if (!G.edges.some(e => [e.source, e.target].sort().join('|') === pair)) {
      G.edges.push({ source: entityId, target: nb.entity, type: nb.edge_type || 'related_to' });
    }
  });

  renderGSidebar();
  // Re-warm simulation
  if (G.raf) cancelAnimationFrame(G.raf);
  startSimulation();
}

function searchNode(entityId) {
  switchTab('search');
  $('sq').value = entityId.replace(/_/g, ' ');
  window.doSearch?.();
}

// ── Canvas setup & interactions ───────────────────────────────────────────────

function resizeGCanvas() {
  const gc = $('gcanvas'); if (!gc) return;
  const wrap = gc.parentElement;
  if (!wrap) return;
  const W = wrap.clientWidth;
  if (!W) return; // pane still hidden — skip
  const H = Math.max(400, window.innerHeight - 200);
  gc.style.width = W + 'px';
  gc.style.height = H + 'px';
  gc.width = W * G.dpr;
  gc.height = H * G.dpr;
}

function setupGCanvas() {
  const gc = $('gcanvas'); if (!gc) return;

  G.dpr = window.devicePixelRatio || 1;

  resizeGCanvas();
  window.addEventListener('resize', () => { resizeGCanvas(); if (G.loaded) renderG(); });

  // Wheel zoom
  gc.addEventListener('wheel', e => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    G.zoom = Math.max(0.15, Math.min(6, G.zoom * factor));
    renderG();
  }, { passive: false });

  // Mouse interactions
  gc.addEventListener('mousedown', e => {
    const { mx, my } = gcCoords(gc, e);
    G.drag = G.nodes.find(n => Math.hypot(n.x - mx, n.y - my) < n.r + 3) || null;
    if (!G.drag) G._pan = { sx: e.clientX - G.panX, sy: e.clientY - G.panY };
  });

  gc.addEventListener('mousemove', e => {
    const { mx, my } = gcCoords(gc, e);
    const prev = G.hover;
    G.hover = G.nodes.find(n => Math.hypot(n.x - mx, n.y - my) < n.r + 3) || null;
    if (G.drag) { G.drag.x = mx; G.drag.y = my; G.drag.vx = 0; G.drag.vy = 0; }
    if (G._pan) { G.panX = e.clientX - G._pan.sx; G.panY = e.clientY - G._pan.sy; }
    gc.style.cursor = G.hover ? 'pointer' : (G.drag || G._pan) ? 'grabbing' : 'grab';
    showTooltip(gc, e, G.hover);
  });

  gc.addEventListener('mouseup', e => {
    if (G.drag && !G._pan) {
      G.selected = G.drag;
      showNodePanel(G.drag);
    }
    G.drag = null; G._pan = null;
  });

  gc.addEventListener('dblclick', e => {
    const { mx, my } = gcCoords(gc, e);
    const node = G.nodes.find(n => Math.hypot(n.x - mx, n.y - my) < n.r + 3);
    if (node) expandNode(node.id);
  });
}

function gcCoords(gc, e) {
  const r = gc.getBoundingClientRect();
  return {
    mx: (e.clientX - r.left - G.panX) / G.zoom,
    my: (e.clientY - r.top - G.panY) / G.zoom,
  };
}

function showTooltip(gc, e, node) {
  let tt = document.getElementById('g-tooltip');
  if (!tt) {
    tt = document.createElement('div');
    tt.id = 'g-tooltip';
    tt.style.cssText = 'position:absolute;pointer-events:none;background:var(--bg2,#1e2535);border:1px solid var(--border,#2d3a50);border-radius:6px;padding:7px 10px;font-size:11px;max-width:200px;z-index:999;transition:opacity 0.1s';
    gc.parentElement.style.position = 'relative';
    gc.parentElement.appendChild(tt);
  }
  if (!node) { tt.style.opacity = '0'; return; }
  tt.innerHTML = `<div style="font-weight:700;color:var(--text,#d4dce8)">${esc(node.label)}</div><div style="color:var(--text2,#8899bb);margin-top:2px">${node.connections} connections</div><div style="color:var(--text3,#566382);margin-top:2px;font-size:10px">dblclick to expand</div>`;
  const r = gc.getBoundingClientRect();
  tt.style.left = (e.clientX - r.left + 14) + 'px';
  tt.style.top = (e.clientY - r.top + 14) + 'px';
  tt.style.opacity = '1';
}

function gZoom(f) { G.zoom = Math.max(0.15, Math.min(6, G.zoom * f)); renderG(); }
function gReset() { G.zoom = 1; G.panX = 0; G.panY = 0; renderG(); }
function toggleGSidebar() { $('g-sidebar').classList.toggle('open'); }

document.addEventListener('DOMContentLoaded', setupGCanvas);

window.loadGraph = loadGraph;
window.expandNode = expandNode;
window.searchNode = searchNode;
window.toggleGFilter = () => {};  // no-op, kept for compat
window.toggleGSidebar = toggleGSidebar;
window.gZoom = gZoom;
window.gReset = gReset;
