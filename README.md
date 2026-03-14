# AgentMem

**Local persistent memory for Claude Code, OpenClaw, and any AI agent framework.**

[![Version](https://img.shields.io/badge/version-0.9.3-blue)](.) [![A-MAC](https://img.shields.io/badge/algorithm-A--MAC%20%2B%20wRRF-brightgreen)](https://arxiv.org/abs/2603.04549) [![Redis 8](https://img.shields.io/badge/backend-Redis%208%20HNSW-red)](https://redis.io) [![License](https://img.shields.io/badge/license-MIT-yellow)](.)

```
LLM-F1: 64.70%†  │  Context-F1: 32.64%  │  AIC: 97.9%  │  Recall P50: 1.7ms  │  $0/month
```

---

## How it works — two layers, zero conflict

Your AI framework (Claude Code / OpenClaw) manages the **LLM context window**. AgentMem manages everything **below** it:

```
┌──────────────────────────────────────────────────────────────┐
│  Tier 0 — LLM Context Window  (framework manages this)       │
│  Claude Code / OpenClaw:  sliding window · summary · pointer  │
│  AgentMem injects:        <cross_session_memory>…</…>        │
│                           ~1,064 tokens vs ~16,910 raw (94%↓) │
├──────────────────────────────────────────────────────────────┤
│  Tier 1 — Session KV  (Redis, 4h TTL)                        │
│  rolling summary · auto-compacted when >3K chars (v0.9.3)    │
├──────────────────────────────────────────────────────────────┤
│  Tier 2 — Episodic  mem:episodes                             │
│  conversation turns · typed (decision/procedure/discovery…)   │
│  causal chain: prev_episode_id ↔ next_episode_id (v0.9.3)    │
├──────────────────────────────────────────────────────────────┤
│  Tier 3 — Semantic  mem:facts                                 │
│  lossless facts · pronoun-free · ISO-timestamped             │
├──────────────────────────────────────────────────────────────┤
│  Tier 4 — Procedural  mem:procedures                         │
│  how-to workflows · tool recipes                              │
├──────────────────────────────────────────────────────────────┤
│  Tier 5 — Capability + Persona  mem:tools · mem:env          │
│  tool index · environment state · evolving user profile       │
└──────────────────────────────────────────────────────────────┘
```

The framework prunes **Tier 0** (context window) — AgentMem is never touched. AgentMem's `before_agent_start` / `UserPromptSubmit` hook re-injects the **most relevant** fragments every turn.

---

## Benchmark Results

> ⚠️ AgentMem's 64.70% is on an **internal bench** (8 convs, 47 Q/A, GLM-4-flash). SimpleMem/AriadneMem scores are on the harder real LoCoMo dataset (1,986 Q, 200-400 turn convs). Not directly comparable — our real-dataset score will be lower. Run `bench-f1.py` to reproduce.

| System | LLM-F1 | Dataset | Notes |
|--------|:------:|---------|-------|
| Full Context | 18.70% | real LoCoMo | No compression baseline |
| Mem0 | 34.20% | real LoCoMo | Cloud API |
| SimpleMem | 43.24% | real LoCoMo | SOTA research baseline |
| AriadneMem | 46.30% | real LoCoMo | **Published SOTA** |
| **AgentMem** | **64.70%†** | internal† | Local · GLM-4-flash · 97.9% AIC |

---

## vs. Everything

| | Mem0 | SimpleMem | claude-mem | MEMORY.md | **AgentMem** |
|---|:---:|:---:|:---:|:---:|:---:|
| **Local / offline** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Cost** | $0.002–0.01/op | needs OpenAI | free | free | **free** |
| **Recall latency** | ~50ms | ~96ms | varies | 0ms (static) | **1.7ms P50** |
| **Scales to 1,000s of memories** | ✅ | ✅ | ✅ | ❌ (truncates) | ✅ |
| **Auto-extracts facts** | ✅ | ✅ | ✅ | ❌ manual | ✅ |
| **Lossless restatement (Φ_coref + Φ_time)** | ❌ | ✅ | ❌ | ❌ | ✅ |
| **3-phase consolidation** | ❌ | ✅ | ❌ | ❌ | ✅ |
| **A-MAC 5-factor admission gate** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Dynamic weighted RRF (wRRF)** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **6-tier cognitive memory** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Knowledge graph** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Episode type taxonomy** | ❌ | ❌ | ✅ | ❌ | ✅ |
| **Causal episode chaining** | ❌ | ❌ | ✅ | ❌ | ✅ |
| **Mid-session compact (O(N) context)** | ❌ | ❌ | ✅ Endless | ❌ | ✅ |
| **Claude Code hooks** | ❌ | ❌ | ✅ | ✅ built-in | ✅ |
| **CJK / Chinese** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **MCP server** | ❌ | ✅ | ✅ | ❌ | ✅ |
| **LangChain/LangGraph/CrewAI/AutoGen** | Partial | ❌ | ❌ | ❌ | ✅ |
| **Benchmarked F1** | 34.20% | 43.24% | ❌ | ❌ | 64.70%† |

> claude-mem inspired v0.9.3's episode taxonomy, causal chaining, and mid-session compact. What AgentMem adds: research-grade retrieval (wRRF + A-MAC + consolidation), 6 memory tiers, knowledge graph, and benchmark-verified quality.

---

## Claude Code — One Command

```bash
git clone https://github.com/JaceHo/AgentMem
cd AgentMem
python3 -m venv venv && venv/bin/pip install -r requirements.txt
bash setup-claude.sh   # installs service + all 4 hooks
```

Open a new Claude Code session. Done — every prompt now gets cross-session memory.

**What you get on every prompt:**
```xml
<cross_session_memory>
## User Profile
- rules: always use bun for JavaScript; always deploy with Docker Compose
- preferences: prefers Python for scripting, Rust for performance

## Long-Term Memory (Facts)
1. [rule] Jace always uses AgentMem for OpenClaw memory.
2. [preference] Jace prefers bun over npm as JavaScript package manager.

## Recent Relevant Episodes
1. [decision] user: I decided to always use bun instead of npm...
2. [procedure] user: to fix Redis HNSW connection, run start.sh first...
</cross_session_memory>
```

**4 hooks, zero config after install:**

| Hook | Trigger | Action |
|------|---------|--------|
| `register-env.sh` | Session start | Registers OS / git / cwd / model |
| `recall.sh` | Every prompt | Injects memory via `additionalContext` (~330ms) |
| `compact.sh` | After every tool use | Compresses session KV when >3,000 chars (fire-and-forget) |
| `store.sh` | Session end | Persists transcript → Redis long-term (async) |

---

## OpenClaw — Easy Setup

```bash
# Plugin already at /path/to/AgentMem/plugin — add to openclaw.json:
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

Plugin v0.4.0: recalls on `before_agent_start` · compacts + stores on `agent_end`.

---

## Quick Start

```bash
# Start service (or use setup-claude.sh for Claude Code)
bash start.sh
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
# Optional: ZAI key for LLM-powered lossless extraction
echo "ZAI_API_KEY=your-key" > .env
```

---

## Bootstrap from Existing History

```bash
# Ingest all past Claude Code sessions
python3 digest-claudecode.py          # incremental
python3 digest-claudecode.py --reset  # re-ingest everything

# Ingest all past OpenClaw sessions
python3 digest-openclaw.py

# Merge semantic duplicates after bulk ingest
curl -X POST http://localhost:18800/consolidate/sync
```

---

## Architecture — Retrieval Pipeline

```
Query →
  ├─ Embed (MiniLM-L12 384-dim, LRU-cached)
  ├─ Parallel async gather:
  │   ├── VSIM on mem:facts  (scene-filtered)
  │   ├── VSIM on mem:episodes
  │   ├── Symbolic pass: named entities → targeted VSIM
  │   ├── Persona · env · session context
  │   └── Tool context (if capability query)
  ├─ Dynamic Weighted RRF (wRRF, arXiv:2511.18194):
  │   entity+temporal → weights [0.8, 0.8, 1.4]
  │   semantic only   → weights [1.0, 1.0, 0.6]
  ├─ BM25-lite keyword boost
  ├─ Importance boost: score += 0.15 × importance
  └─ Token-budget greedy packing → <cross_session_memory>
     Priority: persona > env > tools > session > facts > episodes

Store → A-MAC 5-factor gate (arXiv:2603.04549):
  F1 semantic_novelty   (w=0.25)  F2 entity_novelty    (w=0.15)
  F3 factual_confidence (w=0.20)  F4 temporal_signal   (w=0.10)
  F5 content_type_prior (w=0.30)  ← DOMINANT (A-MAC ablation)
  score ≥ 0.40 → admit → extract → embed → save
```

---

## Consolidation — 3-Phase (SimpleMem §3.2)

Runs every 50 stores + hourly:

```
Phase 1 — Decay:  age > 90 days → importance × 0.9
Phase 2 — Merge:  affinity = cosine × exp(−λ·days) ≥ 0.85 → LLM merges cluster
          (temporal affinity prevents "Python 2.7 (2023)" merging with "Python 3.12 (2026)")
Phase 3 — Prune:  importance < 0.05 → soft-delete
```

Category importance floors prevent critical facts from ever reaching the prune threshold:
`identity/rule ≥ 0.80 · preference ≥ 0.75 · decision ≥ 0.60 · general ≥ 0.30`

```bash
curl -X POST http://localhost:18800/consolidate/sync
# → {"decayed":12,"merged":3,"pruned":1,"total_before":87,"total_after":83,"ms":1247}
```

---

## Advanced Recall

```json
{
  "query": "...", "session_id": "...",
  "token_budget": 2000,
  "enable_planning": true,
  "enable_reflection": true,
  "include_graph": true,
  "time_from": 1738368000000
}
```

| Option | Default | Effect |
|--------|---------|--------|
| `token_budget` | 1500 | Max words in output |
| `enable_planning` | false | LLM generates 1-3 sub-queries (+600ms) |
| `enable_reflection` | false | Sufficiency check + 2nd pass (+1s) |
| `include_graph` | false | Knowledge-graph neighbourhood expansion |
| `time_from/to` | null | Unix ms timestamp range filter |

---

## API Reference

| Endpoint | Description |
|----------|-------------|
| `POST /recall` | Before-prompt hook — returns `prependContext` |
| `POST /store` | After-session hook — async, returns `{"status":"queued"}` |
| `POST /session/compress` | Promote Tier 1 → Tier 2 long-term |
| `POST /session/compact` | Mid-session compress Tier 1 KV if >threshold chars |
| `POST /answer` | LLM extracts short answer from recalled context (bench use) |
| `POST /consolidate/sync` | Run 3-phase consolidation now |
| `POST /register-tools` | Register agent tool index |
| `POST /recall-tools` | Semantic search over tools |
| `POST /store-procedure` | Save a how-to workflow |
| `POST /recall-procedures` | Search procedural memory |
| `GET /graph/{entity}` | Knowledge graph neighbours |
| `GET /stats` | `{"episodes":N,"facts":N,"procedures":N,"tools":N}` |
| `GET /health` | `{"status":"ok","redis":"ok","version":"0.9.3"}` |
| `GET /` | Web dashboard (5-tab, live SSE logs) |

---

## Framework Integrations

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

# MCP (Claude Desktop / Cursor / Windsurf)
# python mcp_server.py  — 7 MCP tools exposed via stdio
```

---

## Setup

```bash
# Prerequisites: Python 3.12+, Redis 8+
python3 -m venv venv && venv/bin/pip install -r requirements.txt

# Claude Code (recommended — service + hooks in one shot)
bash setup-claude.sh

# Standalone
bash start.sh

# macOS launchd auto-start
launchctl load ~/Library/LaunchAgents/ai.agent.memory.plist
```

---

## Research Foundation

| Paper | Role |
|-------|------|
| [SimpleMem arXiv:2601.02553](https://arxiv.org/abs/2601.02553) | Core pipeline: lossless extraction §3.1, consolidation §3.2, intent-aware retrieval §3.3 |
| [AriadneMem arXiv:2603.03290](https://arxiv.org/abs/2603.03290) | Graph bridge discovery — SOTA 46.30% on real LoCoMo. Inspired `include_graph` |
| [A-MAC arXiv:2603.04549](https://arxiv.org/abs/2603.04549) | 5-factor admission gate + category importance floors |
| [wRRF arXiv:2511.18194](https://arxiv.org/abs/2511.18194) | Dynamic weighted RRF per query type |
| [MAGMA arXiv:2601.03236](https://arxiv.org/abs/2601.03236) | Multi-graph memory — inspired knowledge graph tier |
| [Anatomy of Agentic Memory arXiv:2602.19320](https://arxiv.org/abs/2602.19320) | 6-tier cognitive taxonomy |
| [MemoryOS arXiv:2506.06326](https://arxiv.org/abs/2506.06326) | Heat-tiered reranking |
| [claude-mem](https://github.com/thedotmack/claude-mem) | Episode taxonomy, causal chaining, Endless Mode compact (v0.9.3) |
| [Redis 8 Vectorset](https://redis.io/blog/searching-1-billion-vectors-with-redis-8/) | Native HNSW — no separate vector DB |

---

## License

MIT
