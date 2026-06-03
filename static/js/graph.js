/* ═══════════════════════════════════════════════════════════════════════════
   AgentMem — Graph: Knowledge Graph Visualization
   Cinematic load · Rich presentation · Friendly interaction
   ═══════════════════════════════════════════════════════════════════════════ */

let G = {
  nodes: [], edges: [], loaded: false,
  zoom: 1, panX: 0, panY: 0,
  drag: null, hover: null, selected: null,
  raf: null, running: false,
  dpr: window.devicePixelRatio || 1,
  // Animation state
  entranceProgress: 0,
  entranceStart: 0,
  settled: false,
  tickCount: 0,
  // Interaction state
  _pan: null,
  _lastClick: 0,
  _searchHighlight: null,
  // Edge particles
  _particles: [],
  // Time for animations
  _time: 0,
  // Hovered connected nodes/edges
  _hoverConnected: null,
};

// ── Color palette ────────────────────────────────────────────────────────────

const PALETTE = {
  hub:     { fill: '#ff6b6b', glow: 'rgba(255,107,107,0.5)', light: '#ff9b9b' },
  core:    { fill: '#ffd93d', glow: 'rgba(255,217,61,0.4)', light: '#ffe87a' },
  normal:  { fill: '#6bcb77', glow: 'rgba(107,203,119,0.4)', light: '#9de0a5' },
  leaf:    { fill: '#4d96ff', glow: 'rgba(77,150,255,0.4)', light: '#85b5ff' },
  selected:{ fill: '#c084fc', glow: 'rgba(192,132,252,0.6)', light: '#dbb4ff' },
  edgeTypes: {
    related_to:    { color: 'rgba(77,150,255,0.15)',  flow: 'rgba(77,150,255,0.5)' },
    depends_on:    { color: 'rgba(255,107,107,0.25)', flow: 'rgba(255,107,107,0.7)' },
    part_of:       { color: 'rgba(107,203,119,0.25)', flow: 'rgba(107,203,119,0.7)' },
    influences:    { color: 'rgba(255,217,61,0.25)',  flow: 'rgba(255,217,61,0.7)' },
    contradicts:   { color: 'rgba(239,68,68,0.25)',   flow: 'rgba(239,68,68,0.7)' },
    supports:      { color: 'rgba(52,211,153,0.25)',  flow: 'rgba(52,211,153,0.7)' },
  },
  edgeHighlight: 'rgba(192,132,252,0.7)',
  edgeHighlightFlow: 'rgba(192,132,252,0.9)',
};

function nodeColor(n, maxConn) {
  if (G.selected?.id === n.id) return PALETTE.selected;
  const ratio = (n.connections || 1) / maxConn;
  if (ratio > 0.6) return PALETTE.hub;
  if (ratio > 0.3) return PALETTE.core;
  if (ratio > 0.1) return PALETTE.normal;
  return PALETTE.leaf;
}

function edgeColor(type, hot) {
  if (hot) return PALETTE.edgeHighlight;
  return (PALETTE.edgeTypes[type] || PALETTE.edgeTypes.related_to).color;
}

function edgeFlowColor(type, hot) {
  if (hot) return PALETTE.edgeHighlightFlow;
  return (PALETTE.edgeTypes[type] || PALETTE.edgeTypes.related_to).flow;
}

// ── Data loading ─────────────────────────────────────────────────────────────

async function loadGraph() {
  const q = $('gq')?.value?.trim() || '';
  const [stats, graphData] = await Promise.all([
    api('/graph/stats'),
    q ? api(`/graph/${encodeURIComponent(q)}`) : api('/graph/nodes?limit=80'),
  ]);

  if (!stats) return;
  const nc = stats.total_nodes || stats.nodes || 0;
  const ec = stats.total_edges || stats.edges || 0;
  $('ginfo').textContent = `${nc} entities · ${ec} relations`;

  let rawNodes = [], rawEdges = [];
  if (q && graphData?.neighbors) {
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
    $('ginfo').textContent = 'No graph data yet — store some memories first.';
    return;
  }

  const maxConn = Math.max(1, ...rawNodes.map(n => n.connections || 1));
  const canvas = $('gcanvas');
  const W = canvas.width / G.dpr, H = canvas.height / G.dpr;

  // Sort nodes by connections (hubs first) for staggered entrance
  rawNodes.sort((a, b) => (b.connections || 1) - (a.connections || 1));

  const nodeById = new Map();
  G.nodes = rawNodes.map((n, i) => {
    const conn = n.connections || 1;
    const r = 8 + Math.sqrt(conn / maxConn) * 22;  // 8–30px radius
    const color = nodeColor({ ...n, connections: conn, id: n.id }, maxConn);
    const node = {
      id: n.id, label: n.label || n.id,
      connections: conn, r, color,
      x: W / 2, y: H / 2,   // start from center
      tx: W / 2 + (Math.random() - 0.5) * Math.min(W, H) * 0.6,
      ty: H / 2 + (Math.random() - 0.5) * Math.min(W, H) * 0.6,
      vx: 0, vy: 0,
      // Entrance animation
      entranceDelay: i * 20,  // stagger ms
      opacity: 0,
      scale: 0,
      // Pulse animation
      pulsePhase: Math.random() * Math.PI * 2,
    };
    nodeById.set(n.id, node);
    return node;
  });

  G.edges = rawEdges
    .filter(e => nodeById.has(e.source) && nodeById.has(e.target))
    .map(e => ({ source: e.source, target: e.target, type: e.type || 'related_to' }));

  G.loaded = true;
  G.selected = null;
  G.hover = null;
  G._searchHighlight = q || null;
  G._particles = [];

  // Start entrance animation
  G.entranceStart = performance.now();
  G.entranceProgress = 0;
  G.settled = false;
  G.tickCount = 0;

  renderGSidebar();
  startSimulation();
}

// ── Physics simulation ───────────────────────────────────────────────────────

function simStep() {
  const nodes = G.nodes, edges = G.edges;
  const canvas = $('gcanvas'); if (!canvas) return;
  const W = canvas.width / G.dpr, H = canvas.height / G.dpr;
  const now = performance.now();
  G._time = now;

  // Entrance animation — nodes fly out from center with elastic ease
  if (G.entranceProgress < 1) {
    G.entranceProgress = Math.min(1, (now - G.entranceStart) / 2200);
    nodes.forEach(n => {
      const t = Math.max(0, Math.min(1, (G.entranceProgress * 2200 - n.entranceDelay) / 700));
      // Elastic ease-out
      const ease = t === 0 ? 0 : t === 1 ? 1 : 1 - Math.pow(2, -10 * t) * Math.cos((t * 10 - 0.75) * (2 * Math.PI) / 3);
      n.opacity = Math.min(1, t * 2);  // fade in faster
      n.scale = ease;
      if (t < 1) {
        n.x = W / 2 + (n.tx - W / 2) * ease;
        n.y = H / 2 + (n.ty - H / 2) * ease;
      }
    });
  }

  // Repulsion between all pairs
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j];
      const dx = b.x - a.x, dy = b.y - a.y;
      const d2 = dx * dx + dy * dy || 1;
      const d = Math.sqrt(d2);
      const minDist = a.r + b.r + 30;
      const f = d < minDist ? 3000 / d2 : 800 / d2;
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
    const restLen = 120 + (a.r + b.r);
    const f = (d - restLen) * 0.02;
    const fx = (dx / d) * f, fy = (dy / d) * f;
    a.vx += fx; a.vy += fy;
    b.vx -= fx; b.vy -= fy;
  });

  // Gravity toward center
  nodes.forEach(n => {
    n.vx += (W / 2 - n.x) * 0.004;
    n.vy += (H / 2 - n.y) * 0.004;
  });

  // Damping + position update + bounds
  nodes.forEach(n => {
    if (G.drag === n) return;
    n.vx *= 0.85; n.vy *= 0.85;
    n.x += n.vx; n.y += n.vy;
    const pad = n.r + 10;
    n.x = Math.max(pad, Math.min(W - pad, n.x));
    n.y = Math.max(pad, Math.min(H - pad, n.y));
  });

  // Update edge particles
  updateParticles(nodes, edges, nodeById);
}

function updateParticles(nodes, edges, nodeById) {
  // Spawn particles on hovered/selected edges
  const hovId = G.hover?.id;
  const selId = G.selected?.id;
  const activeId = hovId || selId;

  if (activeId && Math.random() < 0.3) {
    const activeEdges = edges.filter(e => e.source === activeId || e.target === activeId);
    if (activeEdges.length > 0) {
      const e = activeEdges[Math.floor(Math.random() * activeEdges.length)];
      const a = nodeById.get(e.source), b = nodeById.get(e.target);
      if (a && b) {
        G._particles.push({
          source: e.source, target: e.target, type: e.type,
          t: 0, speed: 0.008 + Math.random() * 0.006,
          size: 1.5 + Math.random() * 1.5,
        });
      }
    }
  }

  // Also spawn ambient particles on random edges occasionally
  if (Math.random() < 0.02 && edges.length > 0) {
    const e = edges[Math.floor(Math.random() * edges.length)];
    G._particles.push({
      source: e.source, target: e.target, type: e.type,
      t: 0, speed: 0.004 + Math.random() * 0.004,
      size: 1 + Math.random(),
    });
  }

  // Update and remove dead particles
  G._particles = G._particles.filter(p => {
    p.t += p.speed;
    return p.t < 1;
  });

  // Cap particle count
  if (G._particles.length > 80) {
    G._particles = G._particles.slice(-80);
  }
}

function startSimulation() {
  if (G.raf) cancelAnimationFrame(G.raf);
  G.running = true;
  G.tickCount = 0;

  function frame() {
    simStep();
    renderG();
    G.tickCount++;
    if (G.tickCount < 400 || G.drag || G.hover || G.entranceProgress < 1) {
      G.raf = requestAnimationFrame(frame);
    } else {
      G.settled = true;
      G.raf = requestAnimationFrame(idleFrame);
    }
  }
  function idleFrame() {
    G._time = performance.now();
    // Keep updating particles even when settled
    const nodeById = new Map(G.nodes.map(n => [n.id, n]));
    updateParticles(G.nodes, G.edges, nodeById);
    renderG();
    G.raf = requestAnimationFrame(idleFrame);
  }
  G.raf = requestAnimationFrame(frame);
}

// ── Rendering ────────────────────────────────────────────────────────────────

function renderG() {
  const canvas = $('gcanvas'); if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const dpr = G.dpr;

  ctx.clearRect(0, 0, W, H);

  // Background with subtle radial gradient
  drawBackground(ctx, W, H, dpr);

  // Background grid (subtle dot grid)
  drawDotGrid(ctx, W, H, dpr);

  ctx.save();
  ctx.translate(G.panX * dpr, G.panY * dpr);
  ctx.scale(G.zoom * dpr, G.zoom * dpr);

  const nodeById = new Map(G.nodes.map(n => [n.id, n]));
  const selId = G.selected?.id;
  const hovId = G.hover?.id;
  const activeId = hovId || selId;

  // Find connected nodes/edges for hover highlighting
  G._hoverConnected = null;
  if (activeId) {
    const connectedNodes = new Set([activeId]);
    const connectedEdges = new Set();
    G.edges.forEach((e, i) => {
      if (e.source === activeId || e.target === activeId) {
        connectedNodes.add(e.source);
        connectedNodes.add(e.target);
        connectedEdges.add(i);
      }
    });
    G._hoverConnected = { nodes: connectedNodes, edges: connectedEdges };
  }

  // Edges — curved lines with arrows and flow particles
  G.edges.forEach((e, idx) => {
    const a = nodeById.get(e.source), b = nodeById.get(e.target);
    if (!a || !b) return;
    const alpha = Math.min(a.opacity, b.opacity);
    if (alpha < 0.05) return;

    const isHot = activeId && (e.source === activeId || e.target === activeId);
    const isDimmed = activeId && !isHot;

    ctx.globalAlpha = alpha * (isDimmed ? 0.15 : isHot ? 1 : 0.5);
    drawCurvedEdge(ctx, a, b, e.type, isHot);
    ctx.globalAlpha = 1;
  });

  // Edge particles
  G._particles.forEach(p => {
    const a = nodeById.get(p.source), b = nodeById.get(p.target);
    if (!a || !b) return;
    const alpha = Math.min(a.opacity, b.opacity);
    if (alpha < 0.1) return;

    const isDimmed = activeId && p.source !== activeId && p.target !== activeId;
    if (isDimmed) return;

    // Position along curve
    const dx = b.x - a.x, dy = b.y - a.y;
    const d = Math.sqrt(dx * dx + dy * dy) || 1;
    const nx = -dy / d, ny = dx / d;
    const curveOffset = d * 0.08;
    const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
    const cx = mx + nx * curveOffset, cy = my + ny * curveOffset;

    const t = p.t;
    const px = (1 - t) * (1 - t) * a.x + 2 * (1 - t) * t * cx + t * t * b.x;
    const py = (1 - t) * (1 - t) * a.y + 2 * (1 - t) * t * cy + t * t * b.y;

    const flowC = edgeFlowColor(p.type, activeId && (p.source === activeId || p.target === activeId));
    ctx.globalAlpha = alpha * (1 - Math.abs(t - 0.5) * 1.2);
    ctx.fillStyle = flowC;
    ctx.shadowColor = flowC;
    ctx.shadowBlur = 6;
    ctx.beginPath();
    ctx.arc(px, py, p.size, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1;
  });

  // Nodes — draw in reverse order (hubs last so they're on top)
  const sortedNodes = [...G.nodes].sort((a, b) => a.connections - b.connections);
  sortedNodes.forEach(n => {
    if (n.opacity < 0.02) return;
    const isSel = n.id === selId;
    const isHov = n.id === hovId;
    const isSearchMatch = G._searchHighlight && n.label.toLowerCase().includes(G._searchHighlight.toLowerCase());
    const isConnected = G._hoverConnected?.nodes.has(n.id);
    const isDimmed = activeId && !isConnected;
    const color = isSel ? PALETTE.selected : n.color;

    ctx.globalAlpha = n.opacity * (isDimmed ? 0.2 : 1);

    // Pulsing glow for hubs and selected
    const pulse = Math.sin(G._time * 0.002 + n.pulsePhase) * 0.15 + 0.85;
    const r = n.r * n.scale;

    // Outer glow ring
    if (isSel || isHov || isSearchMatch || n.connections > 5) {
      ctx.save();
      const glowSize = isSel ? 28 : isHov ? 22 : isSearchMatch ? 18 : 12;
      const glowAlpha = isSel ? 0.5 : isHov ? 0.4 : isSearchMatch ? 0.35 : 0.2 * pulse;
      ctx.shadowColor = color.glow || color;
      ctx.shadowBlur = glowSize * pulse;
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 2, 0, Math.PI * 2);
      ctx.fillStyle = (color.glow || color).replace(/[\d.]+\)$/, glowAlpha + ')');
      ctx.fill();
      ctx.restore();
    }

    // Selection ring
    if (isSel) {
      ctx.save();
      ctx.strokeStyle = color.light || color.fill || color;
      ctx.lineWidth = 2.5;
      ctx.setLineDash([4, 4]);
      ctx.lineDashOffset = -G._time * 0.03;
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 6, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }

    // Search match ring
    if (isSearchMatch && !isSel) {
      ctx.save();
      ctx.strokeStyle = '#ffd93d';
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      ctx.lineDashOffset = -G._time * 0.02;
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }

    // Node circle with gradient
    const grad = ctx.createRadialGradient(n.x - r * 0.3, n.y - r * 0.3, 0, n.x, n.y, r);
    const baseColor = typeof color === 'object' ? color.fill : color;
    const lightColor = typeof color === 'object' ? color.light : lighten(baseColor, 50);
    grad.addColorStop(0, lightColor);
    grad.addColorStop(0.7, baseColor);
    grad.addColorStop(1, darken(baseColor, 20));
    ctx.beginPath();
    ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();

    // Inner highlight (glass effect)
    const innerGrad = ctx.createRadialGradient(n.x - r * 0.2, n.y - r * 0.35, 0, n.x, n.y, r);
    innerGrad.addColorStop(0, 'rgba(255,255,255,0.25)');
    innerGrad.addColorStop(0.5, 'rgba(255,255,255,0.05)');
    innerGrad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = innerGrad;
    ctx.fill();

    // Border
    ctx.strokeStyle = (isHov || isSel) ? 'rgba(255,255,255,0.7)' : 'rgba(255,255,255,0.15)';
    ctx.lineWidth = (isHov || isSel) ? 2.5 : 0.8;
    ctx.stroke();

    // Connection count badge for hubs
    if (n.connections > 3 && r >= 10) {
      const badgeR = Math.max(7, r * 0.35);
      const bx = n.x + r * 0.65, by = n.y - r * 0.65;
      // Badge background
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.beginPath();
      ctx.arc(bx, by, badgeR + 1, 0, Math.PI * 2);
      ctx.fill();
      // Badge fill
      ctx.fillStyle = baseColor;
      ctx.beginPath();
      ctx.arc(bx, by, badgeR, 0, Math.PI * 2);
      ctx.fill();
      // Badge text
      ctx.fillStyle = '#fff';
      ctx.font = `bold ${Math.max(7, badgeR)}px -apple-system,system-ui,sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(n.connections, bx, by);
    }

    // Label with background pill
    if (isHov || isSel || r >= 10 || isSearchMatch) {
      const fontSize = Math.max(9, Math.min(13, r * 0.6));
      ctx.font = `${(isHov || isSel || isSearchMatch) ? 'bold ' : ''}${fontSize}px -apple-system,system-ui,sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      const text = n.label.length > 28 ? n.label.slice(0, 26) + '…' : n.label;
      const textW = ctx.measureText(text).width;
      const labelY = n.y + r + 6;
      const padX = 5, padY = 2;

      // Label background pill
      ctx.fillStyle = isDark() ? 'rgba(17,24,32,0.85)' : 'rgba(255,255,255,0.85)';
      ctx.beginPath();
      const pillW = textW + padX * 2, pillH = fontSize + padY * 2;
      const pillX = n.x - pillW / 2, pillY = labelY - padY;
      roundRect(ctx, pillX, pillY, pillW, pillH, 4);
      ctx.fill();

      // Label border
      ctx.strokeStyle = isDark() ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
      ctx.lineWidth = 0.5;
      ctx.stroke();

      // Label text
      ctx.fillStyle = isDark() ? '#e2e8f0' : '#1a1a2e';
      ctx.fillText(text, n.x, labelY);
    }

    ctx.globalAlpha = 1;
  });

  ctx.restore();

  // Minimap
  drawMinimap(ctx, W, H, dpr);
}

function drawBackground(ctx, W, H, dpr) {
  // Radial gradient background
  const cx = W / 2, cy = H / 2;
  const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(W, H) * 0.7);
  if (isDark()) {
    grad.addColorStop(0, '#141c26');
    grad.addColorStop(1, '#0a0f16');
  } else {
    grad.addColorStop(0, '#f8f9fb');
    grad.addColorStop(1, '#eef0f4');
  }
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);
}

function drawDotGrid(ctx, W, H, dpr) {
  const gridSize = 30 * G.zoom;
  if (gridSize < 8) return;
  const ox = (G.panX * dpr) % gridSize;
  const oy = (G.panY * dpr) % gridSize;
  const dotAlpha = isDark() ? 0.08 : 0.1;
  ctx.fillStyle = isDark() ? `rgba(255,255,255,${dotAlpha})` : `rgba(0,0,0,${dotAlpha})`;
  for (let x = ox; x < W; x += gridSize) {
    for (let y = oy; y < H; y += gridSize) {
      ctx.beginPath();
      ctx.arc(x, y, 0.8, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function drawCurvedEdge(ctx, a, b, type, hot) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const d = Math.sqrt(dx * dx + dy * dy) || 1;

  // Control point for curve
  const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
  const nx = -dy / d, ny = dx / d;
  const curveOffset = d * 0.1;
  const cx = mx + nx * curveOffset, cy = my + ny * curveOffset;

  ctx.strokeStyle = edgeColor(type, hot);
  ctx.lineWidth = hot ? 2.5 : 0.8;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.quadraticCurveTo(cx, cy, b.x, b.y);
  ctx.stroke();

  // Arrow at target
  if (hot || G.zoom > 0.6) {
    const arrowLen = hot ? 10 : 6;
    const t = 0.88;
    const px = (1 - t) * (1 - t) * a.x + 2 * (1 - t) * t * cx + t * t * b.x;
    const py = (1 - t) * (1 - t) * a.y + 2 * (1 - t) * t * cy + t * t * b.y;
    const angle = Math.atan2(b.y - py, b.x - px);
    ctx.fillStyle = edgeColor(type, hot);
    ctx.beginPath();
    ctx.moveTo(b.x - Math.cos(angle - 0.25) * arrowLen, b.y - Math.sin(angle - 0.25) * arrowLen);
    ctx.lineTo(b.x, b.y);
    ctx.lineTo(b.x - Math.cos(angle + 0.25) * arrowLen, b.y - Math.sin(angle + 0.25) * arrowLen);
    ctx.fill();
  }

  // Edge type label on hover
  if (hot && G.zoom > 0.5) {
    const labelX = cx, labelY = cy;
    const fontSize = 9;
    ctx.font = `${fontSize}px -apple-system,system-ui,sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const typeLabel = type.replace(/_/g, ' ');
    const tw = ctx.measureText(typeLabel).width;
    // Label background
    ctx.fillStyle = isDark() ? 'rgba(17,24,32,0.9)' : 'rgba(255,255,255,0.9)';
    roundRect(ctx, labelX - tw / 2 - 4, labelY - fontSize / 2 - 2, tw + 8, fontSize + 4, 3);
    ctx.fill();
    // Label text
    ctx.fillStyle = edgeFlowColor(type, true);
    ctx.fillText(typeLabel, labelX, labelY);
  }
}

function drawMinimap(ctx, W, H, dpr) {
  if (!G.loaded || G.nodes.length < 2) return;
  const mmW = 150, mmH = 100, mmPad = 12;
  const mmX = W / dpr - mmW - mmPad;
  const mmY = mmPad;

  ctx.save();
  // Background with blur effect
  ctx.globalAlpha = 0.8;
  ctx.fillStyle = isDark() ? 'rgba(17,24,32,0.9)' : 'rgba(255,255,255,0.9)';
  ctx.strokeStyle = isDark() ? 'rgba(42,53,69,0.6)' : 'rgba(173,181,189,0.6)';
  ctx.lineWidth = 1;
  roundRect(ctx, mmX, mmY, mmW, mmH, 6);
  ctx.fill();
  ctx.stroke();

  // Compute bounds
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  G.nodes.forEach(n => {
    if (n.x < minX) minX = n.x; if (n.x > maxX) maxX = n.x;
    if (n.y < minY) minY = n.y; if (n.y > maxY) maxY = n.y;
  });
  const rangeX = (maxX - minX) || 1, rangeY = (maxY - minY) || 1;
  const scale = Math.min((mmW - 14) / rangeX, (mmH - 14) / rangeY);

  // Draw edges as faint lines
  const nodeById = new Map(G.nodes.map(n => [n.id, n]));
  ctx.strokeStyle = isDark() ? 'rgba(77,150,255,0.15)' : 'rgba(77,150,255,0.2)';
  ctx.lineWidth = 0.5;
  G.edges.forEach(e => {
    const a = nodeById.get(e.source), b = nodeById.get(e.target);
    if (!a || !b) return;
    ctx.beginPath();
    ctx.moveTo(mmX + 7 + (a.x - minX) * scale, mmY + 7 + (a.y - minY) * scale);
    ctx.lineTo(mmX + 7 + (b.x - minX) * scale, mmY + 7 + (b.y - minY) * scale);
    ctx.stroke();
  });

  // Draw nodes as dots
  G.nodes.forEach(n => {
    const nx = mmX + 7 + (n.x - minX) * scale;
    const ny = mmY + 7 + (n.y - minY) * scale;
    ctx.fillStyle = typeof n.color === 'object' ? n.color.fill : n.color;
    ctx.globalAlpha = n.opacity * 0.9;
    ctx.beginPath();
    ctx.arc(nx, ny, Math.max(1.2, n.r * scale * 0.5), 0, Math.PI * 2);
    ctx.fill();
  });

  // Viewport indicator
  const canvas = $('gcanvas');
  const vw = canvas.width / G.dpr, vh = canvas.height / G.dpr;
  const vpX = mmX + 7 + ((-G.panX / G.zoom) - minX) * scale;
  const vpY = mmY + 7 + ((-G.panY / G.zoom) - minY) * scale;
  const vpW = (vw / G.zoom) * scale;
  const vpH = (vh / G.zoom) * scale;
  ctx.globalAlpha = 0.5;
  ctx.strokeStyle = '#4fc3f7';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(vpX, vpY, vpW, vpH);
  ctx.fillStyle = 'rgba(79,195,247,0.05)';
  ctx.fillRect(vpX, vpY, vpW, vpH);

  ctx.restore();
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function lighten(hex, pct) {
  const num = parseInt(hex.replace('#', ''), 16);
  const r = Math.min(255, ((num >> 16) & 0xff) + pct);
  const g = Math.min(255, ((num >> 8) & 0xff) + pct);
  const b = Math.min(255, (num & 0xff) + pct);
  return `rgb(${r},${g},${b})`;
}

function darken(hex, pct) {
  const num = parseInt(hex.replace('#', ''), 16);
  const r = Math.max(0, ((num >> 16) & 0xff) - pct);
  const g = Math.max(0, ((num >> 8) & 0xff) - pct);
  const b = Math.max(0, (num & 0xff) - pct);
  return `rgb(${r},${g},${b})`;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

// ── Sidebar ──────────────────────────────────────────────────────────────────

function renderGSidebar() {
  const hubCount = G.nodes.filter(n => n.connections > 5).length;
  const edgeTypes = {};
  G.edges.forEach(e => { edgeTypes[e.type] = (edgeTypes[e.type] || 0) + 1; });

  $('g-sidebar-info').innerHTML = `
    <div style="font-size:11px;color:var(--text2);line-height:1.8">
      <div style="display:flex;justify-content:space-between"><span>Nodes</span><span style="color:var(--text);font-weight:700">${G.nodes.length}</span></div>
      <div style="display:flex;justify-content:space-between"><span>Edges</span><span style="color:var(--text);font-weight:700">${G.edges.length}</span></div>
      <div style="display:flex;justify-content:space-between"><span>Hubs</span><span style="color:var(--text);font-weight:700">${hubCount}</span></div>
    </div>
    ${Object.keys(edgeTypes).length > 0 ? `
    <div style="margin-top:10px">
      <div style="font-weight:700;color:var(--text2);text-transform:uppercase;letter-spacing:.08em;font-size:9px;margin-bottom:6px">Edge Types</div>
      <div style="font-size:10px;color:var(--text2);line-height:1.8">
        ${Object.entries(edgeTypes).map(([type, count]) => `
          <div style="display:flex;justify-content:space-between">
            <span>${type.replace(/_/g, ' ')}</span>
            <span style="color:var(--text);font-weight:700">${count}</span>
          </div>
        `).join('')}
      </div>
    </div>` : ''}`;

  $('g-filters').innerHTML = `
    <div style="font-size:10px;color:var(--text3);line-height:1.8">
      <div style="margin-bottom:6px;font-weight:700;color:var(--text2);text-transform:uppercase;letter-spacing:.08em;font-size:9px">Controls</div>
      <div><span class="kbd">Click</span> Select node</div>
      <div><span class="kbd">Dbl-click</span> Expand neighbors</div>
      <div><span class="kbd">Drag</span> Reposition</div>
      <div><span class="kbd">Scroll</span> Zoom to cursor</div>
      <div><span class="kbd">Drag bg</span> Pan canvas</div>
      <div><span class="kbd">Esc</span> Deselect</div>
    </div>
    <div style="margin-top:10px">
      <div style="font-weight:700;color:var(--text2);text-transform:uppercase;letter-spacing:.08em;font-size:9px;margin-bottom:6px">Legend</div>
      <div style="display:flex;flex-direction:column;gap:4px;font-size:10px">
        <div style="display:flex;align-items:center;gap:6px"><span style="width:10px;height:10px;border-radius:50%;background:#ff6b6b;display:inline-block;box-shadow:0 0 6px rgba(255,107,107,0.4)"></span> Hub (>60%)</div>
        <div style="display:flex;align-items:center;gap:6px"><span style="width:10px;height:10px;border-radius:50%;background:#ffd93d;display:inline-block;box-shadow:0 0 6px rgba(255,217,61,0.3)"></span> Core (30-60%)</div>
        <div style="display:flex;align-items:center;gap:6px"><span style="width:10px;height:10px;border-radius:50%;background:#6bcb77;display:inline-block;box-shadow:0 0 6px rgba(107,203,119,0.3)"></span> Normal (10-30%)</div>
        <div style="display:flex;align-items:center;gap:6px"><span style="width:10px;height:10px;border-radius:50%;background:#4d96ff;display:inline-block;box-shadow:0 0 6px rgba(77,150,255,0.3)"></span> Leaf (<10%)</div>
      </div>
    </div>`;
}

function showNodePanel(node) {
  $('g-sidebar').classList.add('open');
  const neighbors = G.edges.filter(e => e.source === node.id || e.target === node.id);
  const neighborNodes = neighbors.map(e => {
    const nid = e.source === node.id ? e.target : e.source;
    return G.nodes.find(n => n.id === nid);
  }).filter(Boolean);

  const edgeTypeMap = {};
  neighbors.forEach(e => {
    const type = e.type || 'related_to';
    edgeTypeMap[type] = (edgeTypeMap[type] || 0) + 1;
  });

  $('g-sidebar-info').innerHTML = `
    <div style="border:1px solid var(--border);border-radius:8px;padding:12px;background:var(--bg3);margin-top:8px">
      <div style="font-weight:800;color:var(--text);font-size:14px;word-break:break-word;margin-bottom:6px">${esc(node.label)}</div>
      <div style="display:flex;gap:16px;margin-bottom:10px">
        <div style="text-align:center">
          <div style="font-size:22px;font-weight:900;color:${typeof node.color === 'object' ? node.color.fill : node.color}">${node.connections}</div>
          <div style="font-size:9px;color:var(--text3)">connections</div>
        </div>
        <div style="text-align:center">
          <div style="font-size:22px;font-weight:900;color:var(--accent)">${neighbors.length}</div>
          <div style="font-size:9px;color:var(--text3)">edges shown</div>
        </div>
      </div>
      ${Object.keys(edgeTypeMap).length > 0 ? `
        <div style="font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:.08em;font-weight:700;margin-bottom:4px">Relationships</div>
        <div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:8px">
          ${Object.entries(edgeTypeMap).map(([type, count]) => `
            <span style="font-size:9px;padding:1px 5px;border-radius:4px;background:var(--bg);border:1px solid var(--border);color:var(--text2)">${type.replace(/_/g,' ')} (${count})</span>
          `).join('')}
        </div>
      ` : ''}
      ${neighborNodes.length > 0 ? `
        <div style="font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:.08em;font-weight:700;margin-bottom:4px">Connected to</div>
        <div style="max-height:140px;overflow-y:auto;font-size:10px">
          ${neighborNodes.slice(0, 20).map(nn => `
            <div style="padding:3px 0;color:var(--text2);cursor:pointer;display:flex;justify-content:space-between;align-items:center" onclick="selectNodeById('${esc(nn.id)}')">
              <span>${esc(nn.label)}</span>
              <span style="font-size:9px;color:var(--text3)">${nn.connections}</span>
            </div>
          `).join('')}
          ${neighborNodes.length > 20 ? `<div style="color:var(--text3);font-size:9px">+${neighborNodes.length - 20} more</div>` : ''}
        </div>
      ` : ''}
      <div style="display:flex;gap:4px;margin-top:10px">
        <button class="btn sm pri" style="flex:1" onclick="expandNode('${esc(node.id)}')">Expand</button>
        <button class="btn sm" style="flex:1" onclick="searchNode('${esc(node.id)}')">Search</button>
      </div>
    </div>`;
}

function selectNodeById(id) {
  const node = G.nodes.find(n => n.id === id);
  if (node) {
    G.selected = node;
    showNodePanel(node);
    // Smooth pan to center on node
    const canvas = $('gcanvas');
    const W = canvas.width / G.dpr, H = canvas.height / G.dpr;
    const targetPanX = W / 2 - node.x * G.zoom;
    const targetPanY = H / 2 - node.y * G.zoom;
    animatePan(targetPanX, targetPanY);
  }
}

function animatePan(targetX, targetY) {
  const startX = G.panX, startY = G.panY;
  const startTime = performance.now();
  const duration = 400;

  function step() {
    const t = Math.min(1, (performance.now() - startTime) / duration);
    const ease = 1 - Math.pow(1 - t, 3); // ease-out cubic
    G.panX = startX + (targetX - startX) * ease;
    G.panY = startY + (targetY - startY) * ease;
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── Expand & Search ──────────────────────────────────────────────────────────

async function expandNode(entityId) {
  const data = await api(`/graph/${encodeURIComponent(entityId)}`);
  if (!data?.neighbors?.length) return;

  const nodeById = new Map(G.nodes.map(n => [n.id, n]));
  const existing = G.nodes.find(n => n.id === entityId);
  const cx = existing?.x ?? $('gcanvas').width / G.dpr / 2;
  const cy = existing?.y ?? $('gcanvas').height / G.dpr / 2;
  const maxConn = Math.max(1, ...G.nodes.map(n => n.connections || 1));

  let added = 0;
  data.neighbors.forEach(nb => {
    if (!nodeById.has(nb.entity)) {
      const conn = nb.connection_count || 1;
      const r = 8 + Math.sqrt(conn / Math.max(maxConn, conn)) * 22;
      const color = nodeColor({ id: nb.entity, connections: conn }, Math.max(maxConn, conn));
      const angle = Math.random() * Math.PI * 2;
      const dist = 80 + Math.random() * 80;
      G.nodes.push({
        id: nb.entity, label: nb.entity.replace(/_/g, ' '),
        connections: conn, r, color,
        x: cx, y: cy,  // start from parent
        tx: cx + Math.cos(angle) * dist,
        ty: cy + Math.sin(angle) * dist,
        vx: 0, vy: 0,
        entranceDelay: added * 30,
        opacity: 0, scale: 0,
        pulsePhase: Math.random() * Math.PI * 2,
      });
      nodeById.set(nb.entity, G.nodes[G.nodes.length - 1]);
      added++;
    }
    const pair = [entityId, nb.entity].sort().join('|');
    if (!G.edges.some(e => [e.source, e.target].sort().join('|') === pair)) {
      G.edges.push({ source: entityId, target: nb.entity, type: nb.edge_type || 'related_to' });
    }
  });

  // Animate new nodes
  const now = performance.now();
  G.nodes.forEach(n => {
    if (n.opacity < 1) {
      n.entranceDelay = (now - G.entranceStart) + n.entranceDelay;
    }
  });
  G.entranceProgress = 0;
  G.entranceStart = now - (G.nodes.filter(n => n.opacity >= 1).length * 20);

  renderGSidebar();
  if (G.selected) showNodePanel(G.selected);
  if (G.raf) cancelAnimationFrame(G.raf);
  G.settled = false;
  G.tickCount = 0;
  startSimulation();
}

function searchNode(entityId) {
  switchTab('search');
  $('sq').value = entityId.replace(/_/g, ' ');
  window.doSearch?.();
}

// ── Canvas setup & interactions ──────────────────────────────────────────────

function setupGCanvas() {
  const gc = $('gcanvas'); if (!gc) return;
  G.dpr = window.devicePixelRatio || 1;

  function resizeCanvas() {
    const wrap = gc.parentElement;
    if (!wrap) return;
    const W = wrap.clientWidth;
    const H = Math.max(550, window.innerHeight - 140);
    gc.style.width = W + 'px';
    gc.style.height = H + 'px';
    gc.width = W * G.dpr;
    gc.height = H * G.dpr;
  }
  resizeCanvas();
  window.addEventListener('resize', () => { resizeCanvas(); if (G.loaded) renderG(); });

  // Wheel zoom — zoom toward cursor position
  gc.addEventListener('wheel', e => {
    e.preventDefault();
    const rect = gc.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.08 : 0.92;
    const newZoom = Math.max(0.1, Math.min(8, G.zoom * factor));
    G.panX = mx - (mx - G.panX) * (newZoom / G.zoom);
    G.panY = my - (my - G.panY) * (newZoom / G.zoom);
    G.zoom = newZoom;
  }, { passive: false });

  // Mouse interactions
  gc.addEventListener('mousedown', e => {
    const { mx, my } = gcCoords(gc, e);
    const node = findNodeAt(mx, my);
    if (node) {
      G.drag = node;
    } else {
      G._pan = { sx: e.clientX - G.panX, sy: e.clientY - G.panY };
    }
  });

  gc.addEventListener('mousemove', e => {
    const { mx, my } = gcCoords(gc, e);
    const prev = G.hover;
    G.hover = findNodeAt(mx, my);
    if (G.drag) {
      G.drag.x = mx; G.drag.y = my;
      G.drag.vx = 0; G.drag.vy = 0;
      if (G.settled) { G.settled = false; G.tickCount = 0; if (G.raf) cancelAnimationFrame(G.raf); startSimulation(); }
    }
    if (G._pan) {
      G.panX = e.clientX - G._pan.sx;
      G.panY = e.clientY - G._pan.sy;
    }
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
    const node = findNodeAt(mx, my);
    if (node) expandNode(node.id);
  });

  // Click on background to deselect
  gc.addEventListener('click', e => {
    const { mx, my } = gcCoords(gc, e);
    if (!findNodeAt(mx, my)) {
      G.selected = null;
      $('g-sidebar').classList.remove('open');
      renderGSidebar();
    }
  });

  // Touch support
  let touchStart = null;
  gc.addEventListener('touchstart', e => {
    if (e.touches.length === 1) {
      const t = e.touches[0];
      const { mx, my } = gcCoords(gc, t);
      const node = findNodeAt(mx, my);
      if (node) {
        G.drag = node;
      } else {
        G._pan = { sx: t.clientX - G.panX, sy: t.clientY - G.panY };
      }
      touchStart = { x: t.clientX, y: t.clientY, time: Date.now() };
    }
  }, { passive: true });

  gc.addEventListener('touchmove', e => {
    if (e.touches.length === 1) {
      const t = e.touches[0];
      const { mx, my } = gcCoords(gc, t);
      if (G.drag) {
        G.drag.x = mx; G.drag.y = my; G.drag.vx = 0; G.drag.vy = 0;
      } else if (G._pan) {
        G.panX = t.clientX - G._pan.sx;
        G.panY = t.clientY - G._pan.sy;
      }
    }
  }, { passive: true });

  gc.addEventListener('touchend', e => {
    if (G.drag && touchStart && Date.now() - touchStart.time < 300) {
      G.selected = G.drag;
      showNodePanel(G.drag);
    }
    G.drag = null; G._pan = null; touchStart = null;
  });

  // Keyboard on canvas
  gc.setAttribute('tabindex', '0');
  gc.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      G.selected = null;
      $('g-sidebar').classList.remove('open');
      renderGSidebar();
    } else if (e.key === '+' || e.key === '=') {
      gZoom(1.2);
    } else if (e.key === '-') {
      gZoom(0.8);
    } else if (e.key === 'f' || e.key === 'F') {
      gFitAll();
    } else if (e.key === 'r' || e.key === 'R') {
      gReset();
    }
  });
}

function findNodeAt(mx, my) {
  // Search in reverse to find top-most node first
  for (let i = G.nodes.length - 1; i >= 0; i--) {
    const n = G.nodes[i];
    if (n.opacity > 0.3 && Math.hypot(n.x - mx, n.y - my) < n.r + 4) {
      return n;
    }
  }
  return null;
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
    tt.style.cssText = 'position:absolute;pointer-events:none;background:var(--bg2,#111820);border:1px solid var(--border2,#3a4a5e);border-radius:10px;padding:10px 14px;font-size:11px;max-width:240px;z-index:999;transition:opacity 0.12s;box-shadow:0 8px 24px rgba(0,0,0,0.4);backdrop-filter:blur(8px)';
    gc.parentElement.style.position = 'relative';
    gc.parentElement.appendChild(tt);
  }
  if (!node) { tt.style.opacity = '0'; return; }
  const color = typeof node.color === 'object' ? node.color.fill : node.color;
  const neighborCount = G.edges.filter(ed => ed.source === node.id || ed.target === node.id).length;
  tt.innerHTML = `
    <div style="font-weight:800;color:var(--text,#d4dce8);font-size:14px;margin-bottom:4px">${esc(node.label)}</div>
    <div style="display:flex;gap:12px;align-items:center">
      <span style="color:${color};font-weight:700;font-size:12px">${node.connections} connections</span>
      <span style="color:var(--text3,#5a6a7a);font-size:10px">${neighborCount} edges shown</span>
    </div>
    <div style="color:var(--text3,#5a6a7a);margin-top:6px;font-size:9px;display:flex;gap:8px">
      <span>dbl-click → expand</span>
      <span>click → select</span>
    </div>`;
  const r = gc.getBoundingClientRect();
  let left = e.clientX - r.left + 16;
  let top = e.clientY - r.top + 16;
  if (left + 240 > r.width) left = e.clientX - r.left - 250;
  if (top + 90 > r.height) top = e.clientY - r.top - 100;
  tt.style.left = left + 'px';
  tt.style.top = top + 'px';
  tt.style.opacity = '1';
}

// ── Controls ─────────────────────────────────────────────────────────────────

function gZoom(f) {
  const canvas = $('gcanvas');
  const W = canvas.width / G.dpr, H = canvas.height / G.dpr;
  const newZoom = Math.max(0.1, Math.min(8, G.zoom * f));
  G.panX = W / 2 - (W / 2 - G.panX) * (newZoom / G.zoom);
  G.panY = H / 2 - (H / 2 - G.panY) * (newZoom / G.zoom);
  G.zoom = newZoom;
}

function gReset() {
  G.zoom = 1; G.panX = 0; G.panY = 0;
  G.selected = null;
  $('g-sidebar').classList.remove('open');
  renderGSidebar();
}

function gFitAll() {
  if (!G.nodes.length) return;
  const canvas = $('gcanvas');
  const W = canvas.width / G.dpr, H = canvas.height / G.dpr;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  G.nodes.forEach(n => {
    if (n.x - n.r < minX) minX = n.x - n.r;
    if (n.x + n.r > maxX) maxX = n.x + n.r;
    if (n.y - n.r < minY) minY = n.y - n.r;
    if (n.y + n.r > maxY) maxY = n.y + n.r;
  });
  const rangeX = maxX - minX || 1, rangeY = maxY - minY || 1;
  const pad = 80;
  G.zoom = Math.min((W - pad * 2) / rangeX, (H - pad * 2) / rangeY, 2.5);
  G.panX = (W - rangeX * G.zoom) / 2 - minX * G.zoom;
  G.panY = (H - rangeY * G.zoom) / 2 - minY * G.zoom;
}

function toggleGSidebar() {
  $('g-sidebar').classList.toggle('open');
}

// ── Init ─────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', setupGCanvas);

window.loadGraph = loadGraph;
window.expandNode = expandNode;
window.searchNode = searchNode;
window.selectNodeById = selectNodeById;
window.toggleGSidebar = toggleGSidebar;
window.gZoom = gZoom;
window.gReset = gReset;
window.gFitAll = gFitAll;
