/* ═══════════════════════════════════════════════════════════════════════════
   AgentMem — API Reference: interactive endpoint documentation
   ═══════════════════════════════════════════════════════════════════════════ */

const API_ENDPOINTS = [
  { method: 'POST', path: '/recall', desc: 'Before-agent-start hook (includes tool context + graph)', body: '{"query":"string","limit":10}' },
  { method: 'POST', path: '/store', desc: 'Agent-end hook (async, non-blocking)', body: '{"messages":[{"role":"user","content":"..."}]}' },
  { method: 'POST', path: '/smart-search', desc: 'Hybrid BM25 + vector + graph search', body: '{"query":"string","limit":20}' },
  { method: 'POST', path: '/search', desc: 'BM25 keyword search', body: '{"query":"string","limit":20}' },
  { method: 'POST', path: '/remember', desc: 'Explicitly store a memory', body: '{"content":"string","category":"fact"}' },
  { method: 'POST', path: '/forget', desc: 'Delete a memory by key', body: '{"key":"uid"}' },
  { method: 'POST', path: '/timeline', desc: 'Chronological memory timeline', body: '{"query":"string","limit":30}' },
  { method: 'GET', path: '/memories', desc: 'List all memories (paginated)' },
  { method: 'GET', path: '/memories/{id}', desc: 'Get single memory by ID' },
  { method: 'GET', path: '/semantic', desc: 'List semantic/fact memories' },
  { method: 'GET', path: '/procedural', desc: 'List procedural memories' },
  { method: 'GET', path: '/sessions', desc: 'List sessions' },
  { method: 'GET', path: '/session/{id}', desc: 'Get session detail' },
  { method: 'POST', path: '/session/compress', desc: 'Promote Tier 1 session to Tier 2 long-term' },
  { method: 'POST', path: '/session/compact', desc: 'Mid-session compress Tier 1 KV if > threshold' },
  { method: 'POST', path: '/session/start', desc: 'Start a new session' },
  { method: 'POST', path: '/session/end', desc: 'End current session' },
  { method: 'GET', path: '/profile', desc: 'Get user persona profile' },
  { method: 'GET', path: '/graph/{entity}', desc: 'Knowledge graph: entity neighbours' },
  { method: 'POST', path: '/graph/recall', desc: 'Knowledge graph: neighbourhood fact retrieval' },
  { method: 'GET', path: '/graph/stats', desc: 'Knowledge graph: node/edge counts' },
  { method: 'POST', path: '/store-procedure', desc: 'Store a workflow/how-to', body: '{"task":"string","steps":["..."]}' },
  { method: 'POST', path: '/recall-procedures', desc: 'Semantic search over procedural memory', body: '{"query":"string","limit":10}' },
  { method: 'POST', path: '/register-tools', desc: 'Register agent tool/skill index' },
  { method: 'POST', path: '/register-env', desc: 'Register current environment state' },
  { method: 'POST', path: '/recall-tools', desc: 'Semantic search over tool index' },
  { method: 'GET', path: '/capabilities', desc: 'Full capability manifest' },
  { method: 'POST', path: '/consolidate', desc: 'Trigger memory consolidation' },
  { method: 'POST', path: '/consolidate/hard-prune', desc: 'VREM soft-deleted + stale entries' },
  { method: 'POST', path: '/tool-feedback', desc: 'Record tool success/fail' },
  { method: 'POST', path: '/procedure-feedback', desc: 'Record procedure success/fail' },
  { method: 'GET', path: '/health', desc: 'Service health check' },
  { method: 'GET', path: '/stats', desc: 'Service statistics' },
  { method: 'GET', path: '/config', desc: 'Service configuration' },
  { method: 'GET', path: '/export', desc: 'Export all memories' },
  { method: 'POST', path: '/import', desc: 'Import memories from JSON' },
  { method: 'GET', path: '/observations', desc: 'List recent observations' },
  { method: 'POST', path: '/observe', desc: 'Store an observation directly' },
  { method: 'POST', path: '/context', desc: 'Get token-budget context for injection' },
  { method: 'POST', path: '/answer', desc: 'Answer a question using memory' },
];

function loadAPI() {
  const filter = $('api-filter')?.value?.toLowerCase() || '';
  const filtered = filter ? API_ENDPOINTS.filter(e => e.path.toLowerCase().includes(filter) || e.desc.toLowerCase().includes(filter)) : API_ENDPOINTS;

  $('api-list').innerHTML = filtered.map((e, i) => `
    <div class="ep">
      <div class="eh" onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display==='none'?'block':'none'">
        <span class="meth ${e.method}">${e.method}</span>
        <span class="epp">${esc(e.path)}</span>
        <span class="epd">${esc(e.desc.slice(0, 60))}</span>
      </div>
      <div style="display:none;padding:10px;font-size:12px">
        <div style="color:var(--text2);margin-bottom:6px">${esc(e.desc)}</div>
        ${e.body ? `<div style="margin-top:4px"><strong style="font-size:9px;color:var(--text3);text-transform:uppercase">Body</strong><pre style="margin-top:4px;padding:8px;background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);font-family:var(--mono);font-size:11px;overflow-x:auto">${esc(e.body)}</pre></div>` : ''}
        <div style="margin-top:6px">
          <strong style="font-size:9px;color:var(--text3);text-transform:uppercase">Try it</strong>
          <div style="display:flex;gap:6px;margin-top:4px">
            <button class="btn sm pri" onclick="apiTry('${e.method}','${esc(e.path)}')">Send</button>
            <button class="btn sm" onclick="apiCopy('curl -X ${e.method} http://localhost:18800${e.path}${e.body ? " -H \"Content-Type: application/json\" -d '" + e.body.replace(/'/g, "\\'") + "'" : ''}')">Copy cURL</button>
          </div>
          <pre id="api-result-${i}" style="margin-top:6px;padding:8px;background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);font-family:var(--mono);font-size:10px;max-height:200px;overflow-y:auto;display:none"></pre>
        </div>
      </div>
    </div>
  `).join('');
}

async function apiTry(method, path) {
  const idx = path.replace(/[^a-z]/gi, '');
  let result;
  if (method === 'GET') result = await api(path);
  else result = await post(path, {});
  const el = document.querySelector(`[id^="api-result"]`);
  if (el) { el.style.display = 'block'; el.textContent = JSON.stringify(result, null, 2) }
}

function apiCopy(text) {
  navigator.clipboard?.writeText(text);
  addLog('info', 'Copied to clipboard');
}

window.loadAPI = loadAPI;
window.apiTry = apiTry;
window.apiCopy = apiCopy;
