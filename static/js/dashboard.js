/* ═══════════════════════════════════════════════════════════════════════════
   AgentMem — Dashboard: stats, health, gauges, token savings
   ═══════════════════════════════════════════════════════════════════════════ */

async function loadDash() {
  const [health, stats, sess, obs, gs] = await Promise.all([
    api('/health'), api('/stats'), api('/sessions?limit=5'),
    api('/observations?limit=5'), api('/graph/stats')
  ]);
  S.health = health; S.stats = stats;

  // Health dot
  const ok = health && health.status === 'ok';
  $('dot').className = 'dot ' + (ok ? 'ok' : 'err');
  $('ver').textContent = 'v' + (health?.version || '?');
  $('emb-info').textContent = health?.embedding?.provider || '—';

  // Uptime
  const up = Math.floor((Date.now() - S.t0) / 1000);
  const upH = Math.floor(up / 3600), upM = Math.floor((up % 3600) / 60), upS = up % 60;
  $('uptime').textContent = upH > 0 ? upH + 'h ' + upM + 'm' : upM + 'm ' + upS + 's';

  // Tier grid
  if (stats) {
    const tiers = [
      { l: 'EPISODIC', v: stats.episodes || 0, s: 'Episodes', c: 'var(--purple)' },
      { l: 'SEMANTIC', v: stats.facts || 0, s: 'Facts', c: 'var(--cyan)' },
      { l: 'PROCEDURAL', v: stats.procedures || 0, s: 'Procedures', c: 'var(--orange)' },
      { l: 'CAPABILITY', v: stats.tools || 0, s: 'Tools', c: 'var(--green)' },
      { l: 'ENVIRONMENT', v: stats.env_fields || 0, s: 'Env Fields', c: 'var(--yellow)' },
      { l: 'PERSONA', v: stats.persona_fields || 0, s: 'Profile', c: 'var(--pink)' },
    ];
    $('tier-grid').innerHTML = tiers.map(t =>
      `<div class="stat"><div class="l">${t.l}</div><div class="v">${t.v}</div><div class="s">${t.s}</div></div>`
    ).join('');
  }

  // Token savings
  if (stats && sess?.sessions) {
    const totalObs = (sess.sessions || []).reduce((a, s) => a + (s.observation_count || s.observationCount || 0), 0);
    const tokenBudget = 2000, estFull = totalObs * 80, estInjected = (sess.sessions || []).length * tokenBudget;
    const savings = estFull > 0 ? Math.round((1 - estInjected / Math.max(estFull, 1)) * 100) : 0;
    const tokensSaved = Math.max(0, estFull - estInjected);
    const costCents = Math.round(tokensSaved / 1000 * 30);
    const costStr = costCents >= 100 ? '$' + (costCents / 100).toFixed(2) : costCents + 'ct';
    if (savings > 0) {
      $('token-bar').style.display = 'flex';
      $('tb-pct').textContent = savings + '%';
      $('tb-detail').textContent = `~${tokensSaved.toLocaleString()} tokens saved · ${costStr} saved/yr`;
    }
  }

  // Health info
  if (health) {
    let h = '<div style="font-size:12px">';
    h += `<div style="margin-bottom:4px"><span style="color:var(--green)">●</span> Redis: ${esc(health.redis || '—')}</div>`;
    if (health.embedding) h += `<div style="margin-bottom:4px"><span style="color:var(--cyan)">●</span> Embedding: ${esc(health.embedding.provider || '?')} (${health.embedding.dims || '?'}d)</div>`;
    if (health.bm25_available !== undefined) h += `<div style="margin-bottom:4px"><span style="color:var(--orange)">●</span> BM25: ${health.bm25_available ? 'Active' : 'Off'}</div>`;
    h += '</div>';
    $('health-info').innerHTML = h;
  }

  // Writer pipeline
  if (stats?.writer) {
    const w = stats.writer, rate = w.success_rate != null ? Math.round(w.success_rate * 100) + '%' : 'n/a';
    $('writer-info').innerHTML = `<div class="g4" style="font-size:11px">
      <div class="stat"><div class="l">Attempts</div><div class="v" style="font-size:18px">${w.attempts || 0}</div><div class="s">received</div></div>
      <div class="stat"><div class="l">Successes</div><div class="v" style="font-size:18px;color:var(--green)">${w.successes || 0}</div><div class="s">stored</div></div>
      <div class="stat"><div class="l">Skips</div><div class="v" style="font-size:18px;color:var(--yellow)">${w.skips || 0}</div><div class="s">filtered</div></div>
      <div class="stat"><div class="l">Rate</div><div class="v" style="font-size:18px">${rate}</div><div class="s">${w.avg_ms ? w.avg_ms + 'ms' : '—'}</div></div></div>`;
  }

  // Gauges
  let gaugeH = '';
  if (stats) {
    const ep = stats.episodes || 0, fa = stats.facts || 0, pr = stats.procedures || 0, total = ep + fa + pr;
    gaugeH += gauge('Episodes', total ? Math.round(ep / total * 100) : 0, 'var(--purple)', ep + ' items');
    gaugeH += gauge('Facts', total ? Math.round(fa / total * 100) : 0, 'var(--cyan)', fa + ' items');
    gaugeH += gauge('Procedures', total ? Math.round(pr / total * 100) : 0, 'var(--orange)', pr + ' items');
  }
  if (gs) { const n = gs.total_nodes || gs.nodes || 0, e = gs.total_edges || gs.edges || 0; gaugeH += gauge('Graph', Math.min(100, n / 10), 'var(--accent)', n + ' nodes, ' + e + ' edges') }
  $('gauge-area').innerHTML = gaugeH || '<div class="empty">No data</div>';

  // Recent observations
  if (obs?.observations?.length) {
    $('dash-obs').innerHTML = obs.observations.slice(0, 5).map(o =>
      `<div class="mc lo"><div class="mh"><span class="mi">${fmtR(o.timestamp)}</span></div><div class="mt">${esc((o.content || '').slice(0, 120))}</div></div>`
    ).join('');
  } else { $('dash-obs').innerHTML = '<div class="empty">No observations</div>' }

  // Hero / getting started
  if (stats && stats.episodes === 0 && stats.facts === 0) {
    $('hero-area').innerHTML = `<div class="hero"><div class="hero-tag">Getting Started</div><div class="hero-title">Seed your first memory</div>
    <div class="hero-desc">AgentMem is running but hasn't stored any memories yet. Start a session or use the API to store observations.</div>
    <pre>curl -X POST http://localhost:18800/store \\
  -H "Content-Type: application/json" \\
  -d '{"messages":[{"role":"user","content":"I prefer dark mode"}]}'</pre></div>`;
  } else { $('hero-area').innerHTML = '' }
}

window.loadDash = loadDash;
