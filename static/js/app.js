/* ═══════════════════════════════════════════════════════════════════════════
   AgentMem — App Core: state, utilities, tab routing, init
   ═══════════════════════════════════════════════════════════════════════════ */

const S = {
  tab: 'dashboard', ws: null, as: true,
  logs: [], audit: [], t0: Date.now(),
  health: null, stats: null,
  memL: 0, sessL: 0, procL: 0, searchL: 0,
  liveEvents: 0, lastEventTs: 0,
  WS_MAX: 5,
};

const TABS = ['dashboard','graph','memories','timeline','sessions','procedures','search','audit','activity','profile','replay','logs','api'];

/* ── Utilities ─────────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const api = async p => { try { const r = await fetch(p); return r.ok ? r.json() : null } catch { return null } };
const post = async (p, d) => { try { const r = await fetch(p, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(d) }); return r.ok ? r.json() : null } catch { return null } };
const del = async p => { try { const r = await fetch(p, { method: 'DELETE' }); return r.ok ? r.json() : null } catch { return null } };
const esc = s => s ? s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;') : '';
const fmtT = ts => { if (!ts) return '—'; return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) };
const fmtD = ts => { if (!ts) return '—'; const d = new Date(ts); return d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' + fmtT(ts) };
const fmtR = ts => { if (!ts) return '—'; const s = Math.floor((Date.now() - new Date(ts).getTime()) / 1000); if (s < 60) return s + 's'; if (s < 3600) return Math.floor(s / 60) + 'm'; if (s < 86400) return Math.floor(s / 3600) + 'h'; return Math.floor(s / 86400) + 'd' };
const tagC = c => ({ fact: 'tag-fact', episode: 'tag-episode', procedure: 'tag-procedure', preference: 'tag-preference', rule: 'tag-rule', identity: 'tag-identity' }[c] || 'tag-general');
const impC = i => i >= 0.7 ? 'hi' : i >= 0.4 ? 'md' : 'lo';
const gauge = (label, pct, color, value) => {
  pct = Math.min(100, Math.max(0, pct));
  const c = pct > 80 ? 'var(--red)' : pct > 60 ? 'var(--yellow)' : color;
  return `<div class="gauge"><span class="gauge-label">${esc(label)}</span><div class="gauge-bar"><div class="gauge-fill" style="width:${pct}%;background:${c}"></div></div><span class="gauge-value">${esc(value)}</span></div>`;
};

/* ── Theme ─────────────────────────────────────────────────────────────── */
function isDark() { return document.documentElement.dataset.theme === 'dark' }
function applyTheme(d) { document.documentElement.dataset.theme = d ? 'dark' : 'light'; $('theme-btn').textContent = d ? 'LIGHT' : 'DARK'; localStorage.setItem('am-theme', d ? 'dark' : 'light') }
function toggleTheme() { applyTheme(!isDark()) }
const _saved = localStorage.getItem('am-theme');
if (_saved) applyTheme(_saved === 'dark'); else if (window.matchMedia('(prefers-color-scheme:light)').matches) applyTheme(false);

/* ── Tab Routing ───────────────────────────────────────────────────────── */
function normTab(t) { const n = String(t || '').replace(/^#/, '').toLowerCase(); return TABS.includes(n) ? n : 'dashboard' }
function tabFromHash() { try { return normTab(decodeURIComponent(location.hash.slice(1))) } catch { return 'dashboard' } }

function switchTab(name, skipRoute) {
  name = normTab(name);
  S.tab = name;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('on', t.dataset.t === name));
  document.querySelectorAll('.pane').forEach(p => p.classList.toggle('on', p.id === 'p-' + name));
  if (!skipRoute) location.hash = name;
  loadTab(name);
}

function loadTab(t) {
  const loaders = {
    dashboard: () => window.loadDash?.(),
    graph: () => window.loadGraph?.(),
    memories: () => window.loadMem?.(),
    timeline: () => window.loadTL?.(),
    sessions: () => window.loadSess?.(),
    procedures: () => window.loadProcs?.(),
    search: () => {},
    audit: () => window.loadAudit?.(),
    activity: () => window.loadAct?.(),
    profile: () => window.loadProfile?.(),
    replay: () => window.loadReplay?.(),
    logs: () => window.loadLogs?.(),
    api: () => window.loadAPI?.(),
  };
  (loaders[t] || (() => {}))();
}

/* ── Modal ─────────────────────────────────────────────────────────────── */
function openModal(html) { $('modal-bg').classList.add('open'); $('modal-body').innerHTML = html }
function closeModal() { $('modal-bg').classList.remove('open') }

/* ── Logging ───────────────────────────────────────────────────────────── */
function addLog(level, msg) {
  const ts = fmtT(Date.now());
  S.logs.push({ level, msg, ts });
  if (S.logs.length > 500) S.logs.shift();
  const ct = $('log-ct');
  if (ct) ct.textContent = S.logs.length;
  const box = $('lbox');
  if (box && S.tab === 'logs') {
    const cls = level === 'error' ? 'error' : level === 'warn' ? 'warn' : level === 'dim' ? 'dim' : 'info';
    box.innerHTML += `<div class="ll ${cls}">${ts} ${esc(msg)}</div>`;
    box.scrollTop = box.scrollHeight;
  }
}

/* ── Keyboard Shortcuts ────────────────────────────────────────────────── */
function showShortcuts() {
  openModal(`<h3>Keyboard Shortcuts</h3><div style="font-size:12px;line-height:2">
    <span class="kbd">/</span> Search &nbsp; <span class="kbd">g</span> Graph &nbsp; <span class="kbd">d</span> Dashboard &nbsp;
    <span class="kbd">l</span> Logs &nbsp; <span class="kbd">m</span> Memories &nbsp; <span class="kbd">t</span> Timeline &nbsp;
    <span class="kbd">?</span> Help
  </div><div class="ma"><button class="btn" onclick="closeModal()">Close</button></div>`);
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
  if (e.key === '/') { e.preventDefault(); switchTab('search'); $('sq')?.focus() }
  else if (e.key === 'g') switchTab('graph');
  else if (e.key === 'd') switchTab('dashboard');
  else if (e.key === 'l') switchTab('logs');
  else if (e.key === 'm') switchTab('memories');
  else if (e.key === 't') switchTab('timeline');
  else if (e.key === '?') showShortcuts();
  else if (e.key === 'Escape') closeModal();
});

/* ── Init ──────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  // Tab click handlers
  document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => switchTab(t.dataset.t)));
  // Hash routing
  window.addEventListener('hashchange', () => switchTab(tabFromHash(), true));
  switchTab(tabFromHash(), true);
  // Connect WebSocket
  window.connectWs?.();
  // Auto-refresh dashboard every 30s
  setInterval(() => { if (S.tab === 'dashboard') window.loadDash?.() }, 30000);
});
