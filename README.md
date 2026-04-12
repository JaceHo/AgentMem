# AgentMem

**Local, free, research-grade persistent memory for AI agents.**
Works with Claude Code, OpenClaw, LangChain, LangGraph, CrewAI, AutoGen — or any MCP client.

[![Version](https://img.shields.io/badge/version-1.1.0-blue)](.) [![A-MAC](https://img.shields.io/badge/algorithm-A--MAC%20%2B%20wRRF%20%2B%20BM25-brightgreen)](https://arxiv.org/abs/2603.04549) [![Redis 8](https://img.shields.io/badge/backend-Redis%208%20HNSW-red)](https://redis.io) [![License](https://img.shields.io/badge/license-MIT-yellow)](.)

```
LLM-F1: 76.34%†  │  P50 recall latency: 15ms  │  AIC: 97.9%  │  Cost: $0/month
Episodes: 1,334  │  Facts: 2,450  │  Procedures: 28,664  │  Tools: 84
```

---

## vs. SuperMemory

> If you're paying for SuperMemory (or Mem0) to give your AI agent memory — this is the free, local, more powerful alternative.

| | SuperMemory | Mem0 | **AgentMem** |
|---|:---:|:---:|:---:|
| **Price** | $29–$99/month | $0.002–0.01/op | **$0 forever** |
| **Your data stays on your machine** | ❌ cloud | ❌ cloud | ✅ **fully local** |
| **Works offline** | ❌ | ❌ | ✅ |
| **Claude Code integration** | ❌ | ❌ | ✅ **native hooks** |
| **Recall latency** | ~200ms (API RTT) | ~50ms | **15ms P50** |
| **Memory tiers** | 1 (search index) | 1 | **6 cognitive tiers** |
| **Auto-extracts facts from conversation** | ❌ | ✅ | ✅ |
| **Lossless restatement (pronoun-free, ISO timestamps)** | ❌ | ❌ | ✅ |
| **Semantic triple extraction (s,p,o)** | ❌ | ❌ | ✅ v1.1 |
| **BM25 hybrid exact-term search** | ❌ | ❌ | ✅ v1.1 |
| **MemAgent overwrite session compaction** | ❌ | ❌ | ✅ v1.1 |
| **Deduplication / consolidation** | ❌ | ❌ | ✅ 3-phase SimpleMem |
| **Knowledge graph** | ❌ | ❌ | ✅ |
| **Tool reliability tracking** | ❌ | ❌ | ✅ |
| **Procedural / how-to memory** | ❌ | ❌ | ✅ |
| **Cross-session continuity** | basic | basic | ✅ pinned handoff |
| **Session-level context** | ❌ | ❌ | ✅ Tier 1 rolling KV |
| **CJK / Chinese support** | ❌ | ❌ | ✅ |
| **MCP server** | ✅ | ❌ | ✅ |
| **LangChain / LangGraph / CrewAI / AutoGen** | ❌ | Partial | ✅ |
| **Benchmark F1** | — | 34.20% | **76.34%†** |

**The short version:** SuperMemory is a great bookmark/knowledge-search tool for humans. AgentMem is built from scratch for AI agents that need to remember — architecture, costs, and privacy included.

---

## vs. Everything Else

| | Mem0 | SimpleMem | claude-mem | MEMORY.md | **AgentMem** |
|---|:---:|:---:|:---:|:---:|:---:|
| **Local / offline** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Cost** | $0.002–0.01/op | needs OpenAI | free | free | **free** |
| **Recall latency** | ~50ms | ~96ms | varies | 0ms (static) | **15ms P50** |
| **Scales to 1,000s of memories** | ✅ | ✅ | ✅ | ❌ truncates | ✅ |
| **Auto-extracts facts** | ✅ | ✅ | ✅ | ❌ manual | ✅ |
| **Lossless restatement (Φ_coref + Φ_time)** | ❌ | ✅ | ❌ | ❌ | ✅ |
| **Semantic triple (s,p,o) extraction** | ❌ | ❌ | ❌ | ❌ | ✅ v1.1 |
| **BM25 hybrid 4-input wRRF** | ❌ | ❌ | ❌ | ❌ | ✅ v1.1 |
| **MemAgent overwrite compaction** | ❌ | ❌ | ❌ | ❌ | ✅ v1.1 |
| **A-MEM memory evolution** | ❌ | ❌ | ❌ | ❌ | ✅ v1.1 |
| **3-phase consolidation** | ❌ | ✅ | ❌ | ❌ | ✅ |
| **A-MAC 5-factor admission gate** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Dynamic weighted RRF** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **6-tier cognitive memory** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Knowledge graph** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Causal episode chaining** | ❌ | ❌ | ✅ | ❌ | ✅ |
| **Cross-session handoff (pinned summary)** | ❌ | ❌ | ❌ | ❌ | ✅ v1.0 |
| **Hard-delete pruning (capacity management)** | ❌ | ❌ | ❌ | ❌ | ✅ v1.0 |
| **Auto graph expansion on entity queries** | ❌ | ❌ | ❌ | ❌ | ✅ v1.0 |
| **Batch MCP tools** | ❌ | ❌ | ❌ | ❌ | ✅ v1.0 |
| **ToolMem per-tool reliability** | ❌ | ❌ | ❌ | ❌ | ✅ v0.9.5 |
| **AutoTool next-tool suggestions (TIG)** | ❌ | ❌ | ❌ | ❌ | ✅ v0.9.5 |
| **AWO meta-tool synthesis** | ❌ | ❌ | ❌ | ❌ | ✅ v0.9.6 |
| **Secret redaction from session KV** | ❌ | ❌ | ❌ | ❌ | ✅ v1.1 |
| **Claude Code hooks** | ❌ | ❌ | ✅ | ✅ built-in | ✅ |
| **CJK / Chinese** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **MCP server** | ❌ | ✅ | ✅ | ❌ | ✅ |
| **LangChain / LangGraph / CrewAI / AutoGen** | Partial | ❌ | ❌ | ❌ | ✅ |
| **Benchmark F1** | 34.20% | 43.24% | ❌ | ❌ | **76.34%†** |

---

## Benchmark

**Published baselines — real LoCoMo dataset (500 convs, GPT-4 answer extractor):**

| System | LLM-F1 | Source |
|--------|:------:|--------|
| Full Context | 18.70% | [SimpleMem paper](https://arxiv.org/abs/2601.02553) |
| A-Mem | 32.58% | [SimpleMem paper](https://arxiv.org/abs/2601.02553) |
| Mem0 | 34.20% | [SimpleMem paper](https://arxiv.org/abs/2601.02553) |
| SimpleMem | 43.24% | [SimpleMem paper](https://arxiv.org/abs/2601.02553) |
| AriadneMem | **46.30%** | [AriadneMem paper](https://arxiv.org/abs/2603.03290) — SOTA |
| Memori | 81.95% | [Memori paper](https://arxiv.org/abs/2603.19935) — on LoCoMo |

**AgentMem — internal bench† (8 convs, 47 Q/A, flush-between-sessions):**

| System | LLM-F1 | Context-F1 | AIC | Recall@1 |
|--------|:------:|:----------:|:---:|:--------:|
| No Memory | 0% | 0% | 0% | 0% |
| Full Context oracle | — | 36.96% | 100% | 100% |
| **AgentMem v1.1** | **76.34%†** | **34.59%** | **97.9%** | **100%** |

> **†** Not directly comparable to published numbers. Different dataset (8 vs 500 convs), different answer extractor (Kimi-K2.5 vs GPT-4), and no official LoCoMo test split. Context-F1 and AIC are dataset-agnostic retrieval quality metrics. Run `scripts/bench-f1.py --flush-between-sessions` to reproduce.

---

## Claude Code — One Command

```bash
git clone https://github.com/JaceHo/AgentMem
cd AgentMem
python3 -m venv venv && venv/bin/pip install -r requirements.txt
bash agentmem.sh setup   # installs service + all 5 hooks
```

Open a new Claude Code session. Done — every prompt gets cross-session memory.

**What you get on every prompt:**
```xml
<cross_session_memory>
## User Profile
- rules: always use bun for JavaScript; always deploy with Docker Compose

## Last Session Summary          ← v1.0: always bridged, even on cold start
Completed Redis migration for aiserv gateway. Reduced cloud timeouts to 60s.
Fixed cascade fallback to use health matrix only (score > 0).

## Available Tools (Relevant)
- **Bash** [system/builtin]: Execute shell commands ⟨reliable (24/26✓)⟩
- *Frequent next tools*: edit, bash, glob          ← AutoTool TIG hint

## Relevant Skills
- **Debug systematically**: 1. Reproduce, 2. Isolate, 3. Fix root cause...

## Long-Term Memory (Facts)
1. [rule] Jace always uses AgentMem for OpenClaw memory.
   triple: jace → always uses → AgentMem for OpenClaw memory  ← v1.1
2. [preference] Jace prefers bun over npm as JavaScript package manager.

## Recent Relevant Episodes
1. [decision] Decided to always use bun instead of npm for new projects.
</cross_session_memory>
```

**5 hooks, zero config after install:**

| Hook | Trigger | Action |
|------|---------|--------|
| `register-env.sh` | Session start | Registers OS / git / cwd / model |
| `register-tools.sh` | Session start | Registers built-in tools + MCP tools from settings |
| `recall.sh` | Every prompt | Injects memory via `additionalContext` (~15ms) |
| `compact.sh` | After every tool use | Session compact (MemAgent overwrite) + ToolMem feedback |
| `store.sh` | Session end | Store + compress + TIG recording + AWO meta-tool synthesis |

---

## How it works — 6 tiers, zero conflict

```
┌──────────────────────────────────────────────────────────────────┐
│  Tier 0 — LLM Context Window  (framework manages this)           │
│  AgentMem injects: <cross_session_memory>…</…>                   │
│                    ~1,064 tokens vs ~16,910 raw (94%↓)           │
├──────────────────────────────────────────────────────────────────┤
│  Tier 1 — Session KV  (Redis, 4h TTL)                            │
│  rolling summary · MemAgent overwrite compaction (v1.1)          │
│  auto-compacted when >3K chars · secrets redacted on write       │
│  Pinned to mem:pinned:session_summary at session end ← v1.0      │
├──────────────────────────────────────────────────────────────────┤
│  Tier 2 — Episodic  mem:episodes                                  │
│  typed turns (decision/procedure/discovery/feature/change…)       │
│  causal chain: prev_episode_id ↔ next_episode_id                 │
│  hard-prune: stale (>180d, unaccessed) → VREM daily ← v1.0      │
├──────────────────────────────────────────────────────────────────┤
│  Tier 3 — Semantic  mem:facts                                     │
│  lossless facts · pronoun-free (Φ_coref) · ISO timestamps        │
│  semantic triple: (subject, predicate, object) ← v1.1 Memori    │
│  dual-layer link: source_episode_id → narrative context          │
│  A-MEM evolution: near-dups enrich keywords + topic ← v1.1      │
│  hard-prune: superseded (>7d) → VREM daily ← v1.0               │
├──────────────────────────────────────────────────────────────────┤
│  Tier 4 — Procedural  mem:procedures                             │
│  how-to workflows · AWO meta-tools · MACLA Beta scoring           │
│  mem:proc_by_tool reverse index                                   │
├──────────────────────────────────────────────────────────────────┤
│  Tier 5 — Capability + Persona  mem:tools · mem:env · mem:persona│
│  ToolMem reliability (success/fail counts per tool)               │
│  AutoTool TIG: transition graph → next-tool hints                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Retrieval Pipeline

```
Query →
  ├─ Embed (MiniLM-L12 384-dim, LRU-cached)
  ├─ Parallel async gather (all fire simultaneously):
  │   ├── VSIM on mem:facts       (scene-filtered + global supplement)
  │   ├── VSIM on mem:episodes
  │   ├── VSIM on mem:procedures  (MACLA Beta re-ranked)
  │   ├── BM25Okapi on in-memory corpus  ← v1.1 (4th wRRF input)
  │   ├── Symbolic: named entities → targeted VSIM
  │   ├── Persona · env · session context
  │   ├── Pinned last-session summary              ← v1.0 (always injected)
  │   └── Tool context + TIG next-tool hints
  ├─ Auto graph expansion when query has entities  ← v1.0 (auto_graph=True)
  ├─ Dynamic Weighted RRF (wRRF, arXiv:2511.18194) — 4 inputs:
  │   entity+temporal → weights [0.8, 0.8, 1.4, 1.2]
  │   entity only     → weights [0.9, 0.9, 1.2, 1.1]
  │   temporal only   → weights [0.9, 0.9, 1.2, 0.8]
  │   semantic only   → weights [1.0, 1.0, 0.6, 0.9]
  ├─ Importance boost: score += 0.15 × importance
  └─ Token-budget greedy packing → <cross_session_memory>
     Priority: persona > last_session > env > tools > skills > session > facts > episodes

Store → A-MAC 5-factor gate (arXiv:2603.04549):
  F1 semantic_novelty (0.25)  F2 entity_novelty (0.15)
  F3 factual_confidence (0.20)  F4 temporal_signal (0.10)
  F5 content_type_prior (0.30)  ← DOMINANT
  → extract → embed → triple extract (v1.1) → save → evolve similar (v1.1) → consolidate

Capacity management (v1.0 — daily):
  VREM superseded facts older than 7 days
  VREM episodes older than 180 days with zero recalls
```

---

## Consolidation — 3-Phase (SimpleMem §3.2)

Runs every 50 stores + hourly:

```
Phase 1 — Decay:  age > 90 days → importance × 0.9
Phase 2 — Merge:  affinity = cosine × exp(−λ·days) ≥ 0.85 → LLM merges cluster
          (temporal factor prevents "Python 2.7 (2023)" merging "Python 3.12 (2026)")
Phase 3 — Prune:  importance < 0.05 → soft-delete (superseded_by="pruned")

Hard-prune (daily):  VREM all soft-deleted entries older than 7 days
                     VREM all unaccessed episodes older than 180 days
```

Category floors prevent critical facts ever reaching prune:
`identity/rule ≥ 0.80 · preference ≥ 0.75 · decision ≥ 0.60 · general ≥ 0.30`

---

## v1.1 — What's New

| Feature | Paper | Detail |
|---------|-------|--------|
| **Semantic triple extraction** | [Memori §2.2](https://arxiv.org/abs/2603.19935) | Extractor LLM now outputs `(subject, predicate, object)` triples alongside the lossless restatement. Stored as `triple_s/p/o` fields in Redis; `triple_str` feeds BM25 keyword corpus. Shown in recall prepend for precision context. |
| **Real BM25 hybrid search** | [Memori §2.3](https://arxiv.org/abs/2603.19935) | `_BM25Index` class (rank_bm25) maintains an in-memory BM25Okapi corpus over all stored facts + triple strings. Populated at startup, updated on every store. Used as **4th wRRF input** alongside vector/symbolic/global passes — catches exact names, flags, and identifiers that embedding search misses. |
| **A-MEM memory evolution** | [A-MEM §3](https://arxiv.org/abs/2502.12110) | Near-duplicate facts (cosine 0.80–0.95) trigger async `_evolve_similar_fact()`: merges new keywords into the existing fact and updates its topic if the new topic is more specific. Keeps memory self-improving without hard-deleting older facts. |
| **Dual-layer episode linking** | [Memori §2.2](https://arxiv.org/abs/2603.19935) | Each fact now stores `source_episode_id` linking to its originating episode, enabling narrative-context retrieval from fact hits. |
| **MemAgent overwrite compaction** | [MemAgent §3.1](https://arxiv.org/abs/2507.02259) | `_accumulate_session()` now calls `overwrite_update()` instead of naive concatenate + summarize. The LLM sees `<memory>old</memory> + <new_conversation>new_turn</new_conversation>` and selectively retains critical facts — matching MemAgent's constant-window design. `core/summarizer.overwrite_update()` uses `max_tokens=400` (vs 80 for `summarize()`), producing complete ~900-char output without mid-sentence truncation. |
| **Chunked compact for long sessions** | [MemAgent §3.1](https://arxiv.org/abs/2507.02259) | `/session/compact` for contexts >2400 chars iterates `overwrite_update()` over 1200-char chunks. Single-chunk path splits head+tail and calls `overwrite_update()` for proper token budget. O(N) total complexity. |
| **Secret redaction in Tier 1 KV** | Security fix | `_accumulate_session()` calls `_redact_secrets()` before writing to Redis. API keys (`sk-…`), GitHub tokens, and `password=…` patterns are replaced with `[REDACTED]` — preventing leakage via `prependContext` in future prompts. |

---

## v1.0 — What's New

| Feature | Detail |
|---------|--------|
| **Session handoff bridge** | Last session summary pinned at `mem:pinned:session_summary`, always injected at next session start — regardless of query similarity |
| **Hard-delete pruning** | Daily `VREM` of superseded facts (>7 days) and unaccessed episodes (>180 days). Keeps HNSW index fresh and search quality high |
| **Auto graph expansion** | Knowledge-graph neighbourhood auto-fires when the query contains named entities (`auto_graph=True` default). No need to set `include_graph=True` |
| **Batch MCP tools** | `batch_recall_memory` and `batch_store_memory` — parallel multi-query/multi-turn in one round-trip |
| **Role-based LLM routing** | Extractor, summarizer, and retrieval planner now use `/v1/role/nlp` for live model selection. Auto-excludes failed models and retries |
| `/consolidate/hard-prune` | Manual trigger endpoint for the daily prune pass |

---

## Quick Start

```bash
# Clone + install
git clone https://github.com/JaceHo/AgentMem
cd AgentMem
python3 -m venv venv && venv/bin/pip install -r requirements.txt

# Claude Code (service + all hooks in one shot)
bash agentmem.sh setup

# Or standalone
bash agentmem.sh start
```

```bash
# Store a turn
curl -s -X POST http://localhost:18800/store \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I always use bun, never npm"},
                   {"role":"assistant","content":"Got it."}],
       "session_id":"demo"}'

# Recall before next turn
curl -s -X POST http://localhost:18800/recall \
  -H "Content-Type: application/json" \
  -d '{"query":"what package manager?","session_id":"demo"}'
```

```bash
# Service management
bash agentmem.sh status    # check health
bash agentmem.sh logs      # tail logs
bash agentmem.sh restart   # reload after code changes (--force to skip connection check)

# Run test suite
uv run python scripts/test_api.py
```

---

## Bootstrap from History

```bash
uv run python scripts/digest-claudecode.py   # ingest past Claude Code sessions
uv run python scripts/digest-openclaw.py     # ingest past OpenClaw sessions
uv run python scripts/digest-metaclaw.py --skills-dir /path/to/MetaClaw/memory_data/skills
curl -X POST http://localhost:18800/consolidate/sync   # merge duplicates
curl -X POST http://localhost:18800/proc-backfill-index
```

---

## OpenClaw Integration

```json
{
  "plugins": {
    "entries": {
      "memos-local-openclaw-plugin": {
        "enabled": true,
        "config": { "baseUrl": "http://127.0.0.1:18800", "memoryLimitNumber": 8 }
      }
    },
    "load": { "paths": ["/path/to/AgentMem/plugin"] }
  }
}
```

Full parity with Claude Code hooks: register env → register tools → recall → tool feedback → compact → store + TIG + AWO.

---

## Framework Adapters

```python
# Claude API
from adapters.claude import ClaudeMemorySession
session = ClaudeMemorySession(session_id="user-123", anthropic_client=client)
response = await session.chat("What tools should I use?")
await session.end()

# LangChain
from adapters.langchain import ClawMemory
memory = ClawMemory(session_id="user-123")
chain = ConversationChain(llm=llm, memory=memory)

# LangGraph
from adapters.langgraph import make_memory_nodes
recall_node, store_node = make_memory_nodes(session_id="user-123")

# MCP (Claude Desktop / Cursor / Windsurf / any MCP client)
# python mcp_server.py — 9 tools exposed via stdio
# including batch_recall_memory and batch_store_memory (v1.0)
```

---

## API Reference

| Endpoint | Description |
|----------|-------------|
| `POST /recall` | Before-prompt — returns `prependContext` + `latency_ms` |
| `POST /store` | After-session — async, returns `{"status":"queued"}` |
| `POST /session/compress` | Promote Tier 1 → Tier 2 + pin session summary (async) |
| `POST /session/compact` | Mid-session MemAgent overwrite compress if >threshold chars |
| `GET  /session/{id}` | Inspect current Tier 1 session context + length |
| `POST /consolidate/sync` | Run 3-phase consolidation now |
| `POST /consolidate/hard-prune` | Trigger daily VREM pass on demand |
| `POST /register-tools` | Register agent tool index into mem:tools |
| `POST /recall-tools` | Semantic search over tools |
| `POST /store-procedure` | Save a how-to workflow / MetaClaw skill |
| `POST /recall-procedures` | Search procedural memory |
| `POST /tool-feedback` | Record success/fail for a tool (ToolMem) |
| `POST /record-tool-sequence` | Record tool sequence into TIG |
| `GET  /tool-graph/{name}` | TIG outgoing transitions from a tool |
| `POST /tool-graph/detect-meta-tools` | AWO: synthesize meta-tool procedures from TIG chains |
| `POST /procedure-feedback` | Record procedure success/fail for MACLA Beta scoring |
| `GET  /tool-procedures/{name}` | Reverse index: procedures that use a given tool |
| `GET  /graph/{entity}` | Knowledge graph neighbours |
| `POST /answer` | LLM extracts short answer from recalled context |
| `GET  /stats` | Counts across all memory tiers |
| `GET  /health` | `{"status":"ok","redis":"ok","version":"1.0.0"}` |
| `GET  /` | Web dashboard (live SSE logs) |

---

## Advanced Recall

```json
{
  "query": "...", "session_id": "...",
  "token_budget": 2000,
  "include_procedures": true,
  "include_tools": true,
  "auto_graph": true,
  "enable_planning": true,
  "enable_reflection": true,
  "time_from": 1738368000000
}
```

| Option | Default | Effect |
|--------|---------|--------|
| `token_budget` | 1500 | Max words in output |
| `auto_graph` | **true** | Auto graph expansion when query has named entities (v1.0) |
| `include_graph` | false | Force graph expansion regardless of query |
| `include_procedures` | false | Inject top skills (MACLA Beta scored) |
| `include_tools` | false | Inject matched tools with reliability + TIG hints |
| `enable_planning` | false | LLM generates sub-queries (+600ms) |
| `enable_reflection` | false | Sufficiency check + 2nd retrieval pass (+1s) |
| `time_from/to` | null | Unix ms timestamp range filter |

---

## Research Foundation

| Paper | Role |
|-------|------|
| [SimpleMem arXiv:2601.02553](https://arxiv.org/abs/2601.02553) | Core pipeline: lossless extraction §3.1, consolidation §3.2, intent-aware retrieval §3.3 |
| [AriadneMem arXiv:2603.03290](https://arxiv.org/abs/2603.03290) | Graph bridge discovery — SOTA 46.30% on real LoCoMo |
| [A-MAC arXiv:2603.04549](https://arxiv.org/abs/2603.04549) | 5-factor admission gate + category importance floors |
| [Memori arXiv:2603.19935](https://arxiv.org/abs/2603.19935) | **v1.1** Semantic triple extraction, BM25 hybrid search, dual-layer linking — 81.95% LoCoMo |
| [A-MEM arXiv:2502.12110](https://arxiv.org/abs/2502.12110) | **v1.1** Zettelkasten memory evolution — near-duplicate enrichment |
| [MemAgent arXiv:2507.02259](https://arxiv.org/abs/2507.02259) | **v1.1** Overwrite strategy for session compaction — O(1) per chunk, O(N) total |
| [wRRF arXiv:2511.18194](https://arxiv.org/abs/2511.18194) | Dynamic weighted RRF per query type — now 4-input (vector + symbolic + global + BM25) |
| [MAGMA arXiv:2601.03236](https://arxiv.org/abs/2601.03236) | Multi-graph memory — inspired knowledge graph tier |
| [ToolMem arXiv:2510.06664](https://arxiv.org/abs/2510.06664) | Per-tool success/fail tracking + reliability hints |
| [AutoTool TIG arXiv:2511.14650](https://arxiv.org/abs/2511.14650) | Tool Inertia Graph → next-tool suggestions |
| [MACLA arXiv:2512.18950](https://arxiv.org/abs/2512.18950) | Beta posterior scoring for procedure recall |
| [AWO arXiv:2601.22037](https://arxiv.org/abs/2601.22037) | Autonomous Workflow Optimization — meta-tool synthesis |
| [Anatomy of Agentic Memory arXiv:2602.19320](https://arxiv.org/abs/2602.19320) | 6-tier cognitive taxonomy |
| [MemoryOS arXiv:2506.06326](https://arxiv.org/abs/2506.06326) | Heat-tiered reranking |
| [claude-mem](https://github.com/thedotmack/claude-mem) | Episode taxonomy, causal chaining, Endless Mode |
| [MetaClaw](https://github.com/aiming-lab/MetaClaw) | 36 behavioral skills + SkillEvolver |
| [Redis 8 Vectorset](https://redis.io/blog/searching-1-billion-vectors-with-redis-8/) | Native HNSW — no separate vector DB |

---

## License

MIT
