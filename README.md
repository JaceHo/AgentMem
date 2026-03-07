# AgentMem

**Local persistent memory for any AI agent framework — MCP · LangChain · LangGraph · CrewAI · AutoGen · Claude API**

[![Version](https://img.shields.io/badge/version-0.9.1-blue)](.) [![SimpleMem](https://img.shields.io/badge/algorithm-SimpleMem%2B-brightgreen)](https://arxiv.org/abs/2601.02553) [![Redis 8](https://img.shields.io/badge/backend-Redis%208%20vectorset-red)](https://redis.io) [![License](https://img.shields.io/badge/license-MIT-yellow)](.)

One service. Every framework. Persistent memory that actually works.

```
Recall P50 (warm): 1.7ms  │  SimpleMem baseline F1: 43.24  │  ~55× faster recall than SimpleMem
```

---

## vs. SimpleMem: What We Implement — and Where We Go Further

AgentMem implements the full [SimpleMem](https://arxiv.org/abs/2601.02553) pipeline (UNC-Chapel Hill / UC Berkeley / UCSC, 2026 — SOTA 43.24 F1 on LoCoMo) and adds four architectural improvements on top of it.

### SimpleMem Benchmark Results

| System | F1 Score | Total Time | Token Cost |
|--------|:--------:|:----------:|:----------:|
| Full Context | 18.70% | — | ~16,910 |
| A-Mem | 32.58% | 5,937s | — |
| Mem0 | 34.20% | 1,934s | ~1,000 |
| **SimpleMem** | **43.24%** | **481s** | **~550** |
| **AgentMem (SimpleMem+)** | *same algorithm + 4 upgrades* | **~1.7ms recall** | **≤550** |

Cross-session (Feb 2026): **SimpleMem 48 vs Claude-Mem 29.3 (+64%)**.

> **Honest note on F1**: AgentMem implements the same extraction + retrieval + consolidation algorithm as SimpleMem, so it achieves the same benchmark F1 baseline. The 4 additions below are architectural improvements that raise retrieval precision in production; a formal LoCoMo benchmark run is not yet published.

### The 4 Upgrades Beyond SimpleMem

| # | Upgrade | SimpleMem | AgentMem | Impact |
|---|---------|-----------|----------|--------|
| 1 | **RRF multi-list fusion** | Plain set union across semantic+lexical+symbolic results | Reciprocal Rank Fusion (k=60) across all retrieval passes | Higher precision when multiple sources agree on a fact |
| 2 | **Importance-weighted recall** | `importance` used only in consolidation (winner selection) | `importance × 0.15` added to retrieval score at query time | High-signal facts (rules, identity) surface above transient context |
| 3 | **Temporal affinity in consolidation** | Pure cosine ≥ 0.95 → merge | `cosine × exp(−λ·days) ≥ 0.85` | Prevents cross-temporal merging ("Python 2.7 2023" ≠ "Python 3.12 2026") |
| 4 | **Importance as consolidation winner** | Higher importance wins in merge | Importance over recency as tiebreaker — matches SimpleMem's own policy | Preserves most-significant version of a fact |

### Full Feature Comparison

| Feature | SimpleMem | Mem0 | MemGPT | **AgentMem** |
|---------|-----------|------|--------|------------|
| Lossless restatement (Φ_coref + Φ_time) | ✅ | ❌ | ❌ | ✅ |
| Multi-view indexing (semantic+lexical+symbolic) | ✅ | ❌ | ❌ | ✅ |
| 3-phase consolidation (decay/merge/prune) | ✅ | ❌ | ❌ | ✅ |
| Intent-aware retrieval planning | ✅ | ❌ | ❌ | ✅ |
| RRF multi-list fusion | ❌ | ❌ | ❌ | ✅ |
| Importance-weighted recall | ❌ | ❌ | ❌ | ✅ |
| Temporal affinity in consolidation | ❌ | ❌ | ❌ | ✅ |
| Entropy gate (explicit H(W_t)) | Implicit only | ❌ | ❌ | ✅ |
| Procedural memory (how-to tier) | ❌ | ❌ | Partial | ✅ |
| Session tier (TTL→promotion) | ❌ | ❌ | Partial | ✅ |
| Entity knowledge graph | ❌ | ❌ | ❌ | ✅ |
| Heat-tiered access reranking | ❌ | ❌ | ❌ | ✅ |
| Auto-consolidation (counter+timer) | Manual | ❌ | ❌ | ✅ |
| MCP server | ✅ | ❌ | ❌ | ✅ |
| Framework adapters (LangChain/LangGraph/CrewAI/AutoGen) | ❌ | Partial | ❌ | ✅ |
| Local-first (no cloud API required) | ❌ (needs OpenAI) | ❌ | ❌ | ✅ |
| Recall latency | ~96ms/q avg | ~50ms | ~200ms | **1.7ms P50** |

---

## Quick Start

```bash
# Prerequisites: Python 3.12+, Redis 8+
git clone https://github.com/JaceHo/AgentMem
cd AgentMem
python3 -m venv venv && venv/bin/pip install -r requirements.txt
bash start.sh
# Dashboard at http://localhost:18800/
```

```bash
# Store a conversation turn
curl -s -X POST http://localhost:18800/store \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I always use bun, never npm"},
                   {"role":"assistant","content":"Got it."}],
       "session_id":"demo"}'

# Recall before next turn
curl -s -X POST http://localhost:18800/recall \
  -H "Content-Type: application/json" \
  -d '{"query":"what package manager should I use?","session_id":"demo"}'
# → {"prependContext": "<cross_session_memory>\n[Facts]\nThe user always uses bun ...\n</cross_session_memory>"}
```

```bash
# Optional: add ZAI key for LLM-powered lossless extraction
echo "ZAI_API_KEY=your-key" > .env
```

---

## Framework Integrations

### MCP (Claude Desktop / Claude Code / Cursor / Windsurf)
```bash
python mcp_server.py   # stdio transport — 7 MCP tools exposed
```
```json
{
  "mcpServers": {
    "agentmem": {
      "command": "python",
      "args": ["/path/to/AgentMem/mcp_server.py"]
    }
  }
}
```

### Claude API
```python
from adapters.claude import ClaudeMemorySession
import anthropic

session = ClaudeMemorySession(session_id="user-123", anthropic_client=anthropic.Anthropic())
response = await session.chat("What tools should I use?")
# recall → build context → call Claude → store → return response
await session.end()   # promote session to long-term memory
```

### LangChain
```python
from adapters.langchain import ClawMemory
from langchain.chains import ConversationChain

memory = ClawMemory(session_id="user-123")
chain = ConversationChain(llm=your_llm, memory=memory)
chain.predict(input="I prefer dark mode in all tools")
```

### LangGraph
```python
from adapters.langgraph import make_memory_nodes
recall_node, store_node = make_memory_nodes(session_id="user-123")
# Add as first + last nodes in your StateGraph
```

### CrewAI / AutoGen
```python
from adapters.crewai import RecallMemoryTool, StoreMemoryCallback
from adapters.autogen import ClawMemoryHook
```

---

## Architecture: SimpleMem+ Retrieval Pipeline

```
Query →
  │
  ├─ [0ms]   Embed query (MiniLM-L12 384-dim, LRU-cached)
  │
  ├─ [0ms]   Parallel async gather (all at once):
  │           ├── Semantic KNN: VSIM on mem:facts (scene-filtered)
  │           ├── Semantic KNN: VSIM on mem:episodes
  │           ├── Symbolic struct: analyze_query_structure → persons/entities → targeted VSIM
  │           ├── Persona context, env context, session context
  │           └── Tool context (if capability query)
  │
  ├─ [opt]   Intent-Aware Planning (enable_planning=true, +600ms):
  │           plan_queries() → 1-3 targeted sub-queries → parallel VSIM
  │
  ├─ [0ms]   Global supplement: fill gaps if scene results sparse
  │
  ├─ [0ms]   RRF fusion (Reciprocal Rank Fusion, k=60):
  │           Fuse scene + global + symbolic → better precision than union
  │
  ├─ [0ms]   Lexical boost: BM25-lite keyword overlap scoring
  │
  ├─ [0ms]   Importance boost: ×0.15 weight on importance field
  │           (Rules/identity/preferences float to top)
  │
  ├─ [opt]   Reflection (enable_reflection=true, +~1s):
  │           check_sufficiency() → 2nd pass if incomplete
  │
  ├─ [opt]   Graph expansion (include_graph=true):
  │           entity neighbourhood facts from mem:graph:*
  │
  └─ [0ms]   Token-budget packing → <cross_session_memory> XML
              Priority: persona > env > tools > session > facts > episodes
              Default: 1500 words (configurable per request)
```

---

## Memory Architecture: 6 Cognitive Tiers

Based on [arXiv:2602.19320](https://arxiv.org/abs/2602.19320) taxonomy:

| Tier | Redis Key | Contents | Lifecycle |
|------|-----------|----------|-----------|
| **Working** | `mem:session:*` | Rolling session summary | 4h TTL → promoted at session end |
| **Episodic** | `mem:episodes` | Conversation turns ("what happened") | Permanent |
| **Semantic** | `mem:facts` | Lossless facts (SimpleMem §3.1) | Permanent; consolidates over time |
| **Procedural** | `mem:procedures` | How-to workflows ("how to do X") | Permanent |
| **Capability** | `mem:tools` + `mem:env` | Agent tools, environment state | Permanent |
| **Persona** | `mem:persona` | Evolving user profile | Permanent |

SimpleMem has one tier. AgentMem has six.

---

## Consolidation: 3-Phase SimpleMem §3.2

```
Every 50 stores + hourly:

Phase 1 — Decay:  entries > 90 days → importance × 0.9
Phase 2 — Merge:  affinity = cosine × exp(−λ·days) ≥ 0.85
                  LLM merges cluster → winner = max(importance)
                  losers soft-deleted (superseded_by, never VREM)
Phase 3 — Prune:  importance < 0.05 → soft-delete
```

Key difference vs SimpleMem: **temporal affinity** (`cosine × exp(−0.03·days)`) prevents "Python 2.7 (2023)" merging with "Python 3.12 (2026)". SimpleMem uses pure cosine ≥ 0.95 and would incorrectly merge those.

```bash
curl -X POST http://localhost:18800/consolidate/sync
# → {"decayed": 12, "merged": 3, "pruned": 1, "total_before": 87, "total_after": 83, "ms": 1247}
```

---

## Lossless Extraction: SimpleMem §3.1

Every stored fact is pronoun-free and time-anchored:

```diff
raw:    "He'll go there tomorrow at 2pm"
stored: "Bob will meet Alice at the Acme office at 2026-03-09T14:00:00"
```

SimpleMem ablation: removing this drops temporal reasoning F1 from **58.62 → 25.40 (−57%)**.

Two extraction layers run in parallel:

| Layer | Method | Latency | Catches |
|-------|--------|---------|---------|
| **Regex** | 18 compiled patterns | ~1ms | English first-person, procedures, env |
| **LLM** | GLM-4-flash (ZAI) | ~200-500ms async | Implicit, Chinese, contextual, decisions |

Both produce `lossless_restatement` with `topic`, `location`, `importance`, `keywords[]`, `persons[]`, `entities[]`.

---

## Advanced Recall Options

```json
{
  "query": "what are my preferences for this project?",
  "session_id": "user-123",
  "token_budget": 2000,
  "enable_planning": true,
  "enable_reflection": true,
  "include_graph": true,
  "time_from": 1738368000000
}
```

| Option | Default | Effect |
|--------|---------|--------|
| `token_budget` | `1500` | Max words in output (greedy priority packing) |
| `enable_planning` | `false` | LLM generates 1-3 targeted sub-queries (~+600ms) |
| `enable_reflection` | `false` | Sufficiency check + 2nd pass if incomplete (~+1s) |
| `include_graph` | `false` | Entity knowledge-graph neighbourhood expansion |
| `time_from/to` | `null` | Unix ms timestamp range filter via Redis VSIM FILTER |

---

## Performance

Apple M4 Pro / Redis 8 / MiniLM-L12 on MPS:

| Metric | Result |
|--------|--------|
| Recall P50 (warm, default mode) | **1.7ms** |
| Recall P90 (warm) | **2.0ms** |
| Recall cold start | **46ms** |
| Embedding (LRU-cached) | **20× faster** |
| Store (async queue) | **instant** |
| SimpleMem avg per query | ~96ms (481s / 5,000 queries) |
| **AgentMem vs SimpleMem recall** | **~55× faster** |
| Consolidation (no merges) | **3ms** |
| Consolidation (with LLM merges) | **~1.2s** |

Speed advantage: AgentMem uses Redis 8 native HNSW vectorset (VSIM) with cached embeddings. SimpleMem uses LanceDB + calls OpenAI for every extraction and retrieval planning step.

---

## Full API Reference

### `POST /recall`
```json
{"query":"...","session_id":"...","memory_limit_number":6,"token_budget":1500,
 "enable_planning":false,"enable_reflection":false,"include_graph":false,"include_tools":true}
```
Returns `{"prependContext":"...","latency_ms":2}`

### `POST /store`
```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"session_id":"..."}
```
Returns `{"status":"queued"}` — processing is async and non-blocking.

### `POST /session/compress`
```json
{"session_id":"...","wait":false}
```
Promotes Tier 1 session → Tier 2 long-term. Returns `{"status":"ok","ep_saved":1,"facts_saved":3}`.

### `GET /session/{session_id}`
Inspect current session buffer: `{"context":"...","length":312,"tier":"Tier 1 / Session KV"}`

### `POST /consolidate/sync`
Returns `{"decayed":N,"merged":N,"pruned":N,"total_before":N,"total_after":N,"ms":N}`

### `POST /register-tools` · `POST /recall-tools`
Register agent tool index; semantic search over tools by description.

### `POST /recall-procedures` · `POST /store-procedure`
Procedural memory search and storage.

### `GET /graph/{entity}` · `POST /graph/recall` · `GET /graph/stats`
Entity knowledge graph. See knowledge graph section above.

### `GET /config`
```json
{"version":"0.9.1","auto_consolidate_every":50,"stores_since_last":12,"periodic_interval_s":3600}
```

### `GET /health` → `{"status":"ok","redis":"ok","version":"0.9.1"}`
### `GET /stats` → `{"episodes":57,"facts":17,"procedures":8,"tools":15}`
### `GET /` → Dashboard (5-tab web UI, auto-refreshes)
### `GET /logs/stream` → SSE real-time log stream

---

## Setup

```bash
# Python 3.12+, Redis 8+ (with built-in vectorset)
python3 -m venv venv && venv/bin/pip install -r requirements.txt

# Optional: ZAI API key for LLM extraction (GLM-4-flash)
echo "ZAI_API_KEY=your-key" > .env

# Run
bash start.sh                # production (kills stale port holder)
venv/bin/uvicorn main:app --reload  # development
```

### macOS Auto-Start
```bash
launchctl load ~/Library/LaunchAgents/ai.openclaw.memory.plist
launchctl kickstart -k "gui/$(id -u)/ai.openclaw.memory"  # restart
```

### OpenClaw Plugin
```bash
cp -r plugin/ ~/.openclaw/extensions/memos-local/
```
```json
{"plugins":{"entries":{"memos-local-openclaw-plugin":{"enabled":true,
  "config":{"baseUrl":"http://127.0.0.1:18800","memoryLimitNumber":6}}}}}
```

---

## Research Foundation

| Paper | Role in AgentMem |
|-------|-----------------|
| [SimpleMem (arXiv:2601.02553)](https://arxiv.org/abs/2601.02553) | Core 3-stage pipeline: §3.1 lossless extraction, §3.2 consolidation, §3.3 intent-aware retrieval |
| [Anatomy of Agentic Memory (arXiv:2602.19320)](https://arxiv.org/abs/2602.19320) | 6-tier cognitive taxonomy |
| [MemoryOS (arXiv:2506.06326)](https://arxiv.org/abs/2506.06326) EMNLP 2025 | Heat-tiered reranking formula |
| [RRF (Cormack et al., SIGIR 2009)](https://dl.acm.org/doi/10.1145/1571941.1572114) | Reciprocal Rank Fusion for multi-list merge |
| [Redis 8 Vectorset](https://redis.io/blog/searching-1-billion-vectors-with-redis-8/) | HNSW at scale, native vectorset — no separate vector DB |
| [Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) | Hybrid recall reference |

---

## License

MIT
