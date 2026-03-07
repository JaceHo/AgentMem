# AgentMem

**Local persistent memory for any AI agent framework — MCP · LangChain · LangGraph · CrewAI · AutoGen · Claude API**

[![Version](https://img.shields.io/badge/version-0.9.1-blue)](.) [![SimpleMem](https://img.shields.io/badge/algorithm-SimpleMem%20SOTA-green)](https://arxiv.org/abs/2601.02553) [![Redis 8](https://img.shields.io/badge/backend-Redis%208%20vectorset-red)](https://redis.io) [![License](https://img.shields.io/badge/license-MIT-yellow)](.)

One service. Every framework. Memories that persist across every session — without any cloud API.

```
Recall P50 (warm): 1.7ms  │  Store: async, instant  │  SimpleMem F1: 43.24 (SOTA)
```

---

## What It Does

AgentMem gives any AI agent **long-term memory** with zero cloud dependency:

- **Before each conversation** — relevant facts, episodes, and procedures are retrieved and prepended to the agent's context
- **After each conversation** — the session is compressed and stored permanently
- **Across sessions** — memories consolidate, decay, and prune automatically, so quality stays high without manual maintenance

Every memory is a **lossless fact** — pronoun-free, absolute-timestamped, independently understandable — following the SimpleMem research standard (SOTA on LoCoMo benchmark, 43.24 F1).

---

## Quick Start

### 1. Start the service
```bash
# Prerequisites: Python 3.12+, Redis 8+
git clone <your-repo>
cd agentmem
python3 -m venv venv && venv/bin/pip install -r requirements.txt
bash start.sh
# Service running at http://localhost:18800
# Dashboard at http://localhost:18800/
```

### 2. Store and recall (raw HTTP — works from any language)
```bash
# Store a conversation turn
curl -s -X POST http://localhost:18800/store \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I prefer bun over npm always"},
                   {"role":"assistant","content":"Noted, will use bun."}],
       "session_id":"demo"}'

# Recall before next turn
curl -s -X POST http://localhost:18800/recall \
  -H "Content-Type: application/json" \
  -d '{"query":"what package manager should I use?","session_id":"demo"}'
# → {"prependContext": "<cross_session_memory>\n[Facts]\nThe user prefers bun ...\n</cross_session_memory>", "latency_ms": 2}
```

### 3. Optional: add a ZAI API key for LLM-powered extraction
```bash
echo "ZAI_API_KEY=your-key" > .env
```
Without a key: fast regex-only extraction. With a key: full SimpleMem lossless restatement via GLM-4-flash (~200ms async, non-blocking).

---

## Framework Integrations

### Claude API (direct)
```python
from adapters.claude import ClaudeMemorySession
import anthropic

session = ClaudeMemorySession(
    session_id="user-123",
    anthropic_client=anthropic.Anthropic(),
)
response = await session.chat("What tools should I use for this project?")
# Automatically: recall → build context → call Claude → store exchange
await session.end()   # promotes session to long-term memory
```

### MCP (Claude Desktop / Claude Code / Cursor / Windsurf)
```bash
python mcp_server.py   # stdio transport — 7 tools exposed
```

Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "agentmem": {
      "command": "python",
      "args": ["/path/to/agentmem/mcp_server.py"]
    }
  }
}
```
Available MCP tools: `recall_memory`, `store_memory`, `recall_tools`, `recall_procedures`, `store_procedure`, `get_stats`, `compress_session`

### LangChain
```python
from adapters.langchain import ClawMemory
from langchain.chains import ConversationChain
from langchain_anthropic import ChatAnthropic

memory = ClawMemory(session_id="user-123")
chain = ConversationChain(llm=ChatAnthropic(model="claude-sonnet-4-6"), memory=memory)
chain.predict(input="My name is Alice and I work at Acme Corp")
# Memory persists across chain instantiations, server restarts, and reboots
```

### LangGraph
```python
from adapters.langgraph import make_memory_nodes
from langgraph.graph import StateGraph

recall_node, store_node = make_memory_nodes(session_id="user-123")

graph = StateGraph(dict)
graph.add_node("recall", recall_node)    # first: inject memories into state["memory_context"]
graph.add_node("agent", your_agent_node)
graph.add_node("store", store_node)      # last: persist exchange
graph.add_edge("recall", "agent")
graph.add_edge("agent", "store")
graph.set_entry_point("recall")
```

### CrewAI
```python
from adapters.crewai import RecallMemoryTool, StoreMemoryCallback

agent = Agent(
    role="Research Analyst",
    tools=[RecallMemoryTool(session_id="user-123")],
    callbacks=[StoreMemoryCallback(session_id="user-123")],
)
```

### AutoGen
```python
from adapters.autogen import ClawMemoryHook

hook = ClawMemoryHook(session_id="user-123")
agent.register_hook("process_message_before_send", hook.process_message_before_send)
agent.register_hook("process_last_received_message", hook.process_last_received_message)
```

### OpenClaw (original integration)
```bash
cp -r plugin/ ~/.openclaw/extensions/memos-local/
```
```json
{
  "plugins": {
    "entries": {
      "memos-local-openclaw-plugin": {
        "enabled": true,
        "config": { "baseUrl": "http://127.0.0.1:18800", "memoryLimitNumber": 6 }
      }
    }
  }
}
```

---

## What Gets Remembered

Six memory tiers following the cognitive science taxonomy from [arXiv:2602.19320](https://arxiv.org/abs/2602.19320):

| Tier | Redis Key | Contents | Lifecycle |
|------|-----------|----------|-----------|
| **Working** | `mem:session:*` | Current session rolling summary | 4h TTL → promoted to long-term on `agent_end` |
| **Episodic** | `mem:episodes` | Conversation turns ("what happened") | Permanent |
| **Semantic** | `mem:facts` | Distilled lossless facts ("what is true") | Permanent; consolidates over time |
| **Procedural** | `mem:procedures` | How-to workflows, tool patterns ("how to do X") | Permanent |
| **Capability** | `mem:tools` + `mem:env` | Agent tools, environment state, self-model | Permanent |
| **Persona** | `mem:persona` | Evolving user profile | Permanent |

The **working tier → long-term promotion** is automatic: the JS plugin (or your framework adapter) calls `/session/compress` at session end, crystallising the session into permanent facts before the 4h TTL expires.

---

## Architecture

```
Any Framework / Client
    │  recall: before first message
    │  store:  after last message
    │  HTTP REST (framework adapters or raw curl)
    ▼
AgentMem Service (FastAPI) v0.9.1 @ localhost:18800
    │
    ├── POST /recall ──────────────────────────────────────────
    │   1. [opt] Intent-Aware Planning (SimpleMem §3.3)
    │      plan_queries() → 1-3 targeted queries (LLM, 4s timeout)
    │   2. Embed query (MiniLM-L12, 384-dim, LRU-cached)
    │   3. KNN search: scene-filtered facts + episodes + procedures
    │   4. Keyword lexical boost (BM25-lite overlap scoring)
    │   5. [opt] Reflection loop: check_sufficiency() → 2nd pass
    │   6. Knowledge graph expansion (entity neighbourhood)
    │   7. Token-budget packing into <cross_session_memory> tags
    │      Priority: persona > env > tools > session > facts > episodes
    │      Default budget: 1500 tokens (configurable per request)
    │   → returns prependContext string
    │
    ├── POST /store ───────────────────────────────────────────
    │   1. Entropy gate H(W_t) — discard low-info turns
    │   2. Scene detection (language + domain)
    │   3. Summarize long turns (LLM or truncate)
    │   4. Hybrid extraction:
    │      Layer 1: regex ~1ms (18 patterns)
    │      Layer 2: LLM ~200-500ms (SimpleMem §3.1 lossless restatement)
    │              Φ_coref: no pronouns  |  Φ_time: absolute ISO 8601
    │              + topic, location, importance (0.0-1.0)
    │   5. Embed + dedup + save to mem:episodes + mem:facts
    │   6. Persona update from identity/preference/rule facts
    │   7. Auto-consolidation (every 50 stores + hourly timer)
    │   → returns {"status": "queued"} instantly (async)
    │
    ├── POST /consolidate (3-phase SimpleMem §3.2)
    │   Phase 1 — Decay:  importance × 0.9 for entries > 90 days old
    │   Phase 2 — Merge:  temporal-affinity clustering (cos × e^(-λ·days))
    │                     LLM-merge → soft-delete losers (superseded_by)
    │   Phase 3 — Prune:  soft-delete entries with importance < 0.05
    │
    └── Redis 8 (localhost:6379)
         ├── mem:episodes   — episodic vectorset (HNSW)
         ├── mem:facts      — semantic facts + importance + superseded_by
         ├── mem:tools      — tool/skill capability index
         ├── mem:procedures — procedural how-to workflows
         ├── mem:graph:*    — entity relationship graph (Redis Sets)
         ├── mem:persona    — user profile Hash
         ├── mem:env        — environment state Hash
         └── mem:session:*  — working memory (String, 4h TTL)
```

---

## Advanced Recall Options

The `/recall` endpoint accepts several options to tune quality vs. speed:

```json
{
  "query": "how do I deploy this service?",
  "session_id": "user-123",
  "memory_limit_number": 8,
  "token_budget": 2000,
  "enable_planning": true,
  "enable_reflection": true,
  "include_graph": true,
  "time_from": 1738368000000,
  "time_to": 1740787199000
}
```

| Option | Default | Effect |
|--------|---------|--------|
| `token_budget` | `1500` | Max words in context output (greedy packing by tier priority) |
| `enable_planning` | `false` | LLM generates 1-3 targeted sub-queries before retrieval (~600ms extra) |
| `enable_reflection` | `false` | LLM checks if results are sufficient, triggers second pass if not |
| `include_graph` | `false` | Expands retrieval into entity knowledge-graph neighbourhood |
| `time_from` / `time_to` | `null` | Unix ms timestamp range filter applied in Redis VSIM FILTER |
| `include_tools` | `true` | Append relevant tool context when query is capability-related |

**Performance guidance:**
- Real-time agent use: defaults (1.7ms P50, no LLM calls)
- Deeper reasoning tasks: `enable_planning: true` (add ~600ms, one LLM call)
- Research/analysis agents: `enable_planning: true, enable_reflection: true` (add ~1s, two LLM calls)

---

## Memory Quality: SimpleMem Implementation

AgentMem implements all three stages of [SimpleMem](https://arxiv.org/abs/2601.02553) (UNC-Chapel Hill / UC Berkeley / UCSC, 2026), the current SOTA for long-term agent memory:

| System | F1 Score | Total Time | Token Cost |
|--------|:--------:|:----------:|:----------:|
| Full Context | 18.70% | — | ~16,910 |
| A-Mem | 32.58% | 5937s | — |
| Mem0 | 34.20% | 1934s | ~1,000 |
| **SimpleMem** | **43.24%** | **481s** | **~550** |

Cross-session benchmark (Feb 2026): **SimpleMem 48 vs Claude-Mem 29.3 (+64%)**.

### Stage 1 — Lossless Extraction (extractor.py)

Every stored fact is a **pronoun-free, absolute-timestamped** lossless restatement:

```diff
raw:    "He'll go there tomorrow at 2pm and work on the project"
stored: "Bob will attend the Acme Corp office at 2026-03-08T14:00:00 to work on Project Phoenix"
```

- **Φ_coref**: pronouns (he/she/it/they/this/他/她/它) replaced with named entities
- **Φ_time**: relative dates (tomorrow/yesterday/last week/明天) → absolute ISO 8601
- **Multi-view indexing I(m_k)**: each fact stored with semantic embedding + keywords[] + persons[] + entities[] + topic + location + importance

SimpleMem ablation: removing atomization alone drops temporal reasoning F1 from **58.62 → 25.40 (−57%)**.

### Stage 2 — Consolidation (main.py `_do_consolidate`)

Three phases run automatically (every 50 stores + hourly):
```
Phase 1 Decay:  entries > 90 days → importance × 0.9
Phase 2 Merge:  affinity = cosine × exp(−λ × |days|) ≥ 0.85
                → LLM merges cluster → losers soft-deleted (superseded_by)
Phase 3 Prune:  importance < 0.05 → soft-deleted
```

Soft-delete (never hard VREM): losers remain for audit but are filtered from recall. This matches SimpleMem's pattern and preserves history.

### Stage 3 — Intent-Aware Retrieval (retrieval_planner.py)

With `enable_planning: true`, an LLM first decomposes the query:
```
Query: "What are all the constraints for the trading system I'm building?"
→ plan_queries() generates:
  1. "trading system constraints requirements"      ← semantic
  2. "trading system technical limitations"         ← lexical variation
  3. "trading system performance budget"            ← symbolic angle
→ Parallel KNN on all 3 → merged + deduplicated results
```

With `enable_reflection: true`, a second LLM call checks sufficiency and triggers an additional retrieval pass if the first result set is incomplete.

---

## Consolidation API

```bash
# Trigger async consolidation (returns immediately)
curl -X POST http://localhost:18800/consolidate

# Sync — see full results
curl -X POST http://localhost:18800/consolidate/sync
# → {"decayed": 12, "merged": 3, "pruned": 1, "total_before": 87, "total_after": 83, "ms": 1247}
```

---

## Knowledge Graph

Entity relationships are tracked automatically. When a fact mentions "Alice" and "Redis" together, a bidirectional edge is recorded. Use this for neighbourhood-aware recall:

```bash
# Who/what is related to "alice"?
curl http://localhost:18800/graph/alice
# → {"entity": "alice", "related": [{"entity": "redis", "fact_count": 3}, ...]}

# Recall all facts in Alice's neighbourhood
curl -X POST http://localhost:18800/graph/recall \
  -d '{"entity": "alice", "k": 5}'

# Graph statistics
curl http://localhost:18800/graph/stats
# → {"node_count": 24, "edge_count": 67}
```

---

## Procedural Memory

The 4th cognitive tier — storing *how to do things*, not just facts about the world:

```bash
# Store a workflow (also auto-extracted from conversations)
curl -X POST http://localhost:18800/store-procedure \
  -d '{"task": "deploy FastAPI service to production",
       "procedure": "1. docker build -t app . 2. docker push registry/app 3. kubectl apply -f deploy.yaml",
       "tools_used": ["Bash", "Docker", "kubectl"],
       "domain": "devops"}'

# Recall by task description
curl -X POST http://localhost:18800/recall-procedures \
  -d '{"query": "how do I push a Docker image", "k": 3}'
# → [{"task": "deploy FastAPI service...", "procedure": "...", "tools_used": [...], "score": 0.91}]
```

Procedure facts are also auto-extracted from conversation text — if you say "to fix X, I used Y", it's captured.

---

## Session Lifecycle

```
Session start:  POST /recall      → inject memories into context
  ↕ turns:      POST /store       → accumulate Tier 1 rolling summary
Session end:    POST /session/compress → crystallise into Tier 2 long-term

Hourly:         _periodic_consolidate() → 3-phase decay/merge/prune
Every 50 stores: _do_consolidate()     → triggered automatically
```

Inspect current session state at any time:
```bash
curl http://localhost:18800/session/user-123
# → {"session_id": "user-123", "context": "User is Alice, working on...", "length": 412, "tier": "Tier 1 / Session KV"}
```

---

## macOS Auto-Start (LaunchAgent)

```bash
# Load (first time)
launchctl load ~/Library/LaunchAgents/ai.openclaw.memory.plist

# Restart after code changes
launchctl kickstart -k "gui/$(id -u)/ai.openclaw.memory"

# Status
launchctl list | grep openclaw.memory

# Logs
tail -f ~/.openclaw/logs/memory-stdout.log
```

KeepAlive: true — auto-restarts on crash, resumes on login. The plist sets `OMP_NUM_THREADS=1` and `TOKENIZERS_PARALLELISM=false` to prevent semaphore leaks during graceful restarts.

---

## Performance Benchmarks

Tested on Apple M4 Pro / Redis 8 / MiniLM-L12 on MPS (61 episodes, 23 facts):

| Operation | Result |
|-----------|--------|
| Recall P50 (warm, no planning) | **1.7ms** |
| Recall P90 (warm, no planning) | **2.0ms** |
| Recall cold start | **46ms** |
| Embedding (cached) | **20× faster** via LRU cache |
| Store (queued) | **instant** (async background) |
| Consolidation (no merges) | **3ms** |
| Consolidation (with LLM merges) | **~1.2s** |
| Planning mode (+plan_queries) | **+600ms** |
| Planning + reflection | **+~1s** |

---

## Full API Reference

### `POST /recall`
```json
{
  "query": "string",
  "session_id": "string",
  "memory_limit_number": 6,
  "token_budget": 1500,
  "enable_planning": false,
  "enable_reflection": false,
  "include_graph": false,
  "include_tools": true,
  "time_from": null,
  "time_to": null
}
```
Returns `{"prependContext": "...", "latency_ms": 2}`

### `POST /store`
```json
{
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "session_id": "string"
}
```
Returns `{"status": "queued"}` immediately. Processing is async and non-blocking.

### `POST /session/compress`
```json
{"session_id": "string", "wait": false}
```
Promotes Tier 1 session buffer → Tier 2 long-term memory. `wait: true` for synchronous result.
Returns `{"status": "ok", "ep_saved": 1, "facts_saved": 3}`

### `GET /session/{session_id}`
Returns `{"session_id": "...", "context": "...", "length": 312, "tier": "Tier 1 / Session KV"}`

### `POST /consolidate` / `POST /consolidate/sync`
Async / sync 3-phase consolidation.
Sync returns `{"decayed": N, "merged": N, "pruned": N, "total_before": N, "total_after": N, "ms": N}`

### `POST /register-tools`
```json
{"tools": [{"name": "Glob", "description": "...", "source": "builtin", "parameters": []}], "agent_id": "..."}
```

### `POST /recall-tools`
```json
{"query": "search the filesystem", "k": 5, "source": "mcp"}
```
Returns `{"tools": [{name, description, category, score, use_count}], "count": N}`

### `POST /recall-procedures`
```json
{"query": "how do I run tests", "k": 3}
```
Returns `{"procedures": [{task, procedure, tools_used, score}], "count": N}`

### `POST /store-procedure`
```json
{"task": "...", "procedure": "...", "tools_used": ["Bash"], "domain": "coding"}
```

### `GET /graph/{entity}` · `POST /graph/recall` · `GET /graph/stats`
Knowledge graph endpoints. See [Knowledge Graph](#knowledge-graph) section above.

### `GET /capabilities`
Returns full capability manifest: tools, env state, agent model, stats.

### `GET /config`
```json
{
  "version": "0.9.1",
  "auto_consolidate_every": 50,
  "stores_since_last": 12,
  "periodic_interval_s": 3600,
  "consolidate_threshold": 0.85,
  "session_ttl_s": 14400
}
```

### `GET /health`
```json
{"status": "ok", "redis": "ok", "version": "0.9.1"}
```

### `GET /stats`
```json
{"episodes": 57, "facts": 17, "procedures": 8, "tools": 15, "persona_fields": 4, "env_fields": 8}
```

### `GET /logs/stream`
Server-Sent Events real-time log stream. Connect via `curl -N http://localhost:18800/logs/stream` or `EventSource`.

---

## Research Foundation

| Paper | What We Use |
|-------|------------|
| [SimpleMem (arXiv:2601.02553)](https://arxiv.org/abs/2601.02553) | Full 3-stage pipeline: lossless compression → consolidation → intent-aware retrieval. SOTA 43.24 F1. |
| [Anatomy of Agentic Memory (arXiv:2602.19320)](https://arxiv.org/abs/2602.19320) | 6-tier cognitive taxonomy (episodic/semantic/procedural/working/capability/persona) |
| [MemoryOS (arXiv:2506.06326)](https://arxiv.org/abs/2506.06326) | Heat-tiered reranking formula |
| [Redis 8 Vectorset](https://redis.io/blog/searching-1-billion-vectors-with-redis-8/) | HNSW native vectorset — no separate vector DB required |
| [Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) | Hybrid recall architecture reference |

---

## License

MIT
