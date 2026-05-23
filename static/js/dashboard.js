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
  const redisOk = health && health.redis && !String(health.redis).includes('error') && health.status !== 'degraded';
  $('dot').className = 'dot ' + (ok ? 'ok' : 'err');
  $('ver').textContent = 'v' + (health?.version || '?');
  $('emb-info').textContent = health?.embedding?.provider || '—';

  // Redis disconnected banner
  let banner = document.getElementById('redis-banner');
  if (!redisOk) {
    if (!banner) {
      banner = document.createElement('div');
      banner.id = 'redis-banner';
      banner.style.cssText = 'background:rgba(255,80,80,.12);border:1px solid rgba(255,80,80,.4);border-radius:var(--radius);padding:8px 14px;margin-bottom:10px;font-size:11px;color:#f77;display:flex;align-items:center;gap:8px';
      const pane = document.getElementById('p-dashboard');
      pane.insertBefore(banner, pane.firstChild);
    }
    banner.innerHTML = `<span style="font-size:16px">⚠</span> <span><b>Redis disconnected</b> — all memory pages will be empty. ${esc(health?.error || '')} Run <code style="font-family:var(--mono);font-size:10px">brew services start redis</code> then restart AgentMem.</span>`;
  } else if (banner) {
    banner.remove();
  }

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

// ── Setup & Agent Config panel ────────────────────────────────────────────────

const AGENTS = [
  {
    id: 'claude',   label: 'Claude Code',    transport: 'hooks + HTTP',  color: 'var(--orange)',
    setup: 'agentmem.sh setup',
    note: 'Full lifecycle hooks (auto-recall + auto-store). Best integration.',
    cfgPath: '~/.claude.json  +  ~/.claude/settings.json',
    snippet: '{"mcpServers":{"agentmem":{"type":"http","url":"http://localhost:18800/mcp"}}}',
  },
  {
    id: 'cursor',   label: 'Cursor',         transport: 'HTTP',          color: 'var(--accent)',
    setup: 'agentmem.sh setup --agent cursor',
    note: 'HTTP MCP. Tools available in Agent mode.',
    cfgPath: '~/.cursor/mcp.json',
    snippet: '{"mcpServers":{"agentmem":{"url":"http://localhost:18800/mcp"}}}',
  },
  {
    id: 'windsurf', label: 'Windsurf',       transport: 'SSE',           color: 'var(--cyan)',
    setup: 'agentmem.sh setup --agent windsurf',
    note: 'SSE MCP. User-level config only.',
    cfgPath: '~/.codeium/windsurf/mcp_config.json',
    snippet: '{"mcpServers":{"agentmem":{"serverUrl":"http://localhost:18800/mcp/sse"}}}',
  },
  {
    id: 'copilot',  label: 'GitHub Copilot', transport: 'HTTP',          color: 'var(--green)',
    setup: 'agentmem.sh setup --agent copilot',
    note: 'HTTP MCP via VS Code. Note: key is "servers" not "mcpServers".',
    cfgPath: '.vscode/mcp.json',
    snippet: '{"servers":{"agentmem":{"type":"http","url":"http://localhost:18800/mcp"}}}',
  },
  {
    id: 'zed',      label: 'Zed',            transport: 'SSE',           color: 'var(--purple)',
    setup: 'agentmem.sh setup --agent zed',
    note: 'SSE MCP. Key is "context_servers" inside settings.json.',
    cfgPath: '~/.config/zed/settings.json',
    snippet: '{"context_servers":{"agentmem":{"url":"http://localhost:18800/mcp/sse"}}}',
  },
  {
    id: 'continue', label: 'Continue.dev',   transport: 'stdio',         color: 'var(--yellow)',
    setup: 'agentmem.sh setup --agent continue',
    note: 'stdio MCP. YAML format.',
    cfgPath: '~/.continue/config.yaml',
    snippet: 'mcpServers:\n  - name: agentmem\n    command: python\n    args: ["/path/to/agentmem/mcp_server.py"]',
  },
  {
    id: 'augment',  label: 'Augment',        transport: 'SSE',           color: 'var(--pink)',
    setup: 'agentmem.sh setup --agent augment',
    note: 'SSE MCP via VS Code settings.',
    cfgPath: 'VS Code User settings.json → augment.advanced.mcpServers',
    snippet: '{"augment.advanced":{"mcpServers":{"agentmem":{"type":"sse","url":"http://localhost:18800/mcp/sse"}}}}',
  },
  {
    id: 'codex',    label: 'Codex CLI',      transport: 'stdio',         color: 'var(--text2)',
    setup: 'agentmem.sh setup --agent codex',
    note: 'stdio MCP. TOML format.',
    cfgPath: '~/.codex/config.toml',
    snippet: '[mcp_servers.agentmem]\ncommand = "python"\nargs = ["/path/to/agentmem/mcp_server.py"]',
  },
  {
    id: 'cline',    label: 'Cline',          transport: 'stdio',         color: 'var(--green)',
    setup: 'agentmem.sh setup --agent cline',
    note: 'stdio MCP. User-level only.',
    cfgPath: '~/.cline/cline_mcp_settings.json',
    snippet: '{"mcpServers":{"agentmem":{"command":"python","args":["/path/to/agentmem/mcp_server.py"]}}}',
  },
  {
    id: 'kilo',     label: 'Kilo Code',      transport: 'stdio',         color: 'var(--orange)',
    setup: 'agentmem.sh setup --agent kilo',
    note: 'stdio MCP. JSONC format.',
    cfgPath: '~/.config/kilo/kilo.jsonc',
    snippet: '{"mcpServers":{"agentmem":{"type":"stdio","command":"python","args":["/path/to/agentmem/mcp_server.py"]}}}',
  },
  {
    id: 'kiro',     label: 'Kiro',           transport: 'stdio',         color: 'var(--cyan)',
    setup: 'agentmem.sh setup --agent kiro',
    note: 'stdio MCP.',
    cfgPath: '~/.kiro/settings/mcp.json',
    snippet: '{"mcpServers":{"agentmem":{"command":"python","args":["/path/to/agentmem/mcp_server.py"]}}}',
  },
  {
    id: 'antigravity', label: 'Antigravity', transport: 'SSE',           color: 'var(--yellow)',
    setup: 'agentmem.sh setup --agent antigravity',
    note: 'SSE MCP. Google Antigravity IDE.',
    cfgPath: '~/.gemini/antigravity/mcp_config.json',
    snippet: '{"mcpServers":{"agentmem":{"serverUrl":"http://localhost:18800/mcp/sse"}}}',
  },
  {
    id: 'opencode', label: 'Opencode',       transport: 'SSE',           color: 'var(--purple)',
    setup: 'agentmem.sh setup --agent opencode',
    note: 'SSE MCP. Key is "mcp" (not "mcpServers").',
    cfgPath: '~/.config/opencode/opencode.json',
    snippet: '{"mcp":{"agentmem":{"type":"remote","url":"http://localhost:18800/mcp/sse","enabled":true}}}',
  },
  {
    id: 'aider',    label: 'Aider',          transport: '—',             color: 'var(--text3)',
    setup: null,
    note: 'No native MCP support. Use community mcpm-aider fork or connect via another agent.',
    cfgPath: '—',
    snippet: null,
  },
];

let _setupOpen = false;
let _activeAgent = 'claude';

function toggleSetup() {
  _setupOpen = !_setupOpen;
  const body = $('setup-body');
  const chev = $('setup-chevron');
  if (_setupOpen) {
    renderSetup();
    body.style.display = 'block';
    chev.style.transform = 'rotate(180deg)';
  } else {
    body.style.display = 'none';
    chev.style.transform = '';
  }
}

function renderSetup() {
  const origin = window.location.origin;
  const tabs = AGENTS.map(a =>
    `<button onclick="selectAgent('${a.id}')" id="st-${a.id}" style="
      padding:4px 10px;font-size:9px;border-radius:var(--radius);cursor:pointer;
      font-family:var(--font);font-weight:700;letter-spacing:.05em;text-transform:uppercase;
      border:1px solid ${_activeAgent === a.id ? a.color : 'var(--border)'};
      background:${_activeAgent === a.id ? a.color + '22' : 'var(--bg3)'};
      color:${_activeAgent === a.id ? a.color : 'var(--text3)'};
      transition:all .15s
    ">${a.label}</button>`
  ).join('');

  const a = AGENTS.find(x => x.id === _activeAgent) || AGENTS[0];
  const snippetHtml = a.snippet
    ? `<div style="position:relative">
        <pre style="background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);padding:10px 12px;font-family:var(--mono);font-size:10px;color:var(--accent);overflow-x:auto;white-space:pre;margin-top:6px">${esc(a.snippet)}</pre>
        <button onclick="copySetup('${a.id}')" style="position:absolute;top:6px;right:6px;padding:2px 7px;font-size:8px;background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius);color:var(--text3);cursor:pointer;font-family:var(--font);text-transform:uppercase;letter-spacing:.05em" id="copy-${a.id}">COPY</button>
       </div>`
    : `<div style="color:var(--text3);font-size:11px;font-style:italic;margin-top:6px">${esc(a.note)}</div>`;

  const setupCmd = a.setup
    ? `<div style="margin-top:10px">
        <div style="font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:.1em;font-weight:700;margin-bottom:4px">Auto-configure</div>
        <div style="position:relative">
          <pre style="background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);padding:8px 12px;font-family:var(--mono);font-size:11px;color:var(--green);overflow-x:auto">${esc(a.setup)}</pre>
          <button onclick="copyText('${esc(a.setup)}')" style="position:absolute;top:4px;right:4px;padding:2px 7px;font-size:8px;background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius);color:var(--text3);cursor:pointer;font-family:var(--font);text-transform:uppercase;letter-spacing:.05em">COPY</button>
        </div>
       </div>`
    : '';

  $('setup-body').innerHTML = `
    <div style="padding-top:10px;max-height:60vh;overflow-y:auto;padding-right:4px">
      <!-- Endpoint pills -->
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;align-items:center">
        <span style="font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:.1em;font-weight:700">MCP endpoints</span>
        ${endpointPill('HTTP', origin + '/mcp', 'var(--accent)')}
        ${endpointPill('SSE', origin + '/mcp/sse', 'var(--cyan)')}
        ${endpointPill('stdio', 'python mcp_server.py', 'var(--orange)')}
        ${endpointPill('sys-prompt', origin + '/system-prompt', 'var(--purple)')}
      </div>

      <!-- Agent tabs -->
      <div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:12px">${tabs}</div>

      <!-- Agent detail -->
      <div style="background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius);border-left:3px solid ${a.color};padding:10px 12px">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
          <span style="font-size:13px;font-weight:800;color:${a.color}">${a.label}</span>
          <span style="font-size:9px;padding:1px 6px;border-radius:4px;background:${a.color}22;color:${a.color};font-weight:700;letter-spacing:.04em;text-transform:uppercase">${a.transport}</span>
        </div>
        <div style="font-size:11px;color:var(--text2);margin-bottom:6px">${esc(a.note)}</div>
        <div style="font-size:9px;color:var(--text3);margin-bottom:2px;text-transform:uppercase;letter-spacing:.1em;font-weight:700">Config file</div>
        <div style="font-family:var(--mono);font-size:10px;color:var(--text2);margin-bottom:8px">${esc(a.cfgPath)}</div>
        <div style="font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:.1em;font-weight:700">Config snippet</div>
        ${snippetHtml}
        ${setupCmd}
      </div>

      <!-- Batch setup hint -->
      <div style="margin-top:10px;font-size:10px;color:var(--text3)">
        Configure all detected agents at once: <code style="font-family:var(--mono);color:var(--green);font-size:10px">agentmem.sh setup --agent all</code>
      </div>
    </div>`;
}

function endpointPill(label, url, color) {
  return `<span onclick="copyText('${esc(url)}')" title="${esc(url)}" style="
    display:inline-flex;align-items:center;gap:4px;padding:2px 8px;
    background:${color}18;border:1px solid ${color}44;border-radius:10px;
    font-size:9px;font-weight:700;color:${color};cursor:pointer;letter-spacing:.04em;
    text-transform:uppercase;transition:all .15s
  " onmouseover="this.style.background='${color}33'" onmouseout="this.style.background='${color}18'">${label}</span>`;
}

function selectAgent(id) {
  _activeAgent = id;
  renderSetup();
}

function copySetup(id) {
  const a = AGENTS.find(x => x.id === id);
  if (!a?.snippet) return;
  copyText(a.snippet);
  const btn = document.getElementById('copy-' + id);
  if (btn) { btn.textContent = 'COPIED'; setTimeout(() => { btn.textContent = 'COPY' }, 1500) }
}

function copyText(txt) {
  navigator.clipboard.writeText(txt).catch(() => {});
}

window.loadDash = loadDash;
window.toggleSetup = toggleSetup;
window.selectAgent = selectAgent;
window.copySetup = copySetup;
window.copyText = copyText;
