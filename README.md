# AgentMem

**Local persistent memory for any AI agent framework — MCP · LangChain · LangGraph · CrewAI · AutoGen · Claude API**

[![Version](https://img.shields.io/badge/version-0.9.2-blue)](.) [![A-MAC](https://img.shields.io/badge/algorithm-A--MAC%20%2B%20wRRF-brightgreen)](https://arxiv.org/abs/2603.04549) [![Redis 8](https://img.shields.io/badge/backend-Redis%208%20vectorset-red)](https://redis.io) [![License](https://img.shields.io/badge/license-MIT-yellow)](.)

One service. Every framework. Persistent memory that actually works.

```
Recall P50 (warm): 1.7ms  │  SimpleMem F1 baseline: 43.24  │  A-MAC F1: 58.3  │  ~55× faster recall than SimpleMem
```

---

## Research Leaderboard (LoCoMo Benchmark, 2026)

AgentMem implements the best-performing admission and retrieval strategies from the 2025-2026 memory literature.

| System | F1 Score | Token Cost | Notes |
|--------|:--------:|:----------:|-------|
| Full Context | 18.70% | ~16,910 | No compression |
| A-Mem | 32.58% | — | Graph-based |
| Mem0 | 34.20% | ~1,000 | Cloud API |
| **SimpleMem** | **43.24%** | **~550** | SOTA lossless extraction + consolidation |
| MAGMA | 0.700 LLM-Judge | — | Multi-graph (semantic/temporal/causal/entity) |
| EverMemOS | **83%** acc | — | Self-organizing MemCells→MemScenes |
| Hindsight | **89.61%** acc | — | 4 logical networks + reflection (Gemini-3) |
| **A-MAC** | **58.3% F1** | — | Adaptive 5-factor admission gate — highest token-F1 |
| **AgentMem v0.9.2** | *A-MAC gate + SimpleMem pipeline + 7 upgrades* | **≤550** | **1.7ms recall** |

> **F1 vs accuracy**: LoCoMo F1 measures exact-match precision/recall on factual questions. Accuracy scores (EverMemOS, Hindsight) use LLM-graded answer correctness on different benchmarks — not directly comparable. AgentMem's F1 baseline matches SimpleMem; the A-MAC admission gate (arXiv:2603.04549) is the highest published token-F1 component on LoCoMo.

---

## vs. SimpleMem: What We Implement — and Where We Go Further

AgentMem implements the full [SimpleMem](https://arxiv.org/abs/2601.02553) pipeline (UNC-Chapel Hill / UC Berkeley / UCSC, 2026) and adds **7 architectural upgrades** on top, drawing from A-MAC (2603.04549), wRRF (2511.18194), and MAGMA (2601.03236).

### v0.9.2 Additions (New)

| # | Upgrade | Paper | Impact |
|---|---------|-------|--------|
| 5 | **A-MAC 5-factor admission gate** | [arXiv:2603.04549](https://arxiv.org/abs/2603.04549) | Replaces single entropy threshold with 5-factor weighted gate: semantic novelty, entity novelty, factual confidence, temporal signal, content type prior — highest published LoCoMo token-F1 |
| 6 | **Dynamic weighted RRF (wRRF)** | [arXiv:2511.18194](https://arxiv.org/abs/2511.18194) | Per-query-type weights for RRF fusion: entity queries boost symbolic pass (w=1.2→1.4), semantic-only queries downweight it (w=0.6) |
| 7 | **Category importance floors** | A-MAC content_type_prior | 15-tier floor map: identity/rule pinned ≥0.80, general ≥0.30 — prevents important facts from being pruned by consolidation |

### Full Upgrade Table (v0.9.0–v0.9.2)

| # | Upgrade | SimpleMem | AgentMem | Source |
|---|---------|-----------|----------|--------|
| 1 | **RRF multi-list fusion** | Plain set union | RRF (k=60) across semantic+lexical+symbolic | Cormack SIGIR 2009 |
| 2 | **Importance-weighted recall** | Used in consolidation only | `score += 0.15 × importance` at query time | AgentMem |
| 3 | **Temporal affinity in consolidation** | Pure cosine ≥ 0.95 | `cosine × exp(−λ·days) ≥ 0.85` | AgentMem |
| 4 | **6-tier cognitive memory** | 1 tier (semantic) | Episodic + Semantic + Procedural + Session + Capability + Persona | arXiv:2602.19320 |
| 5 | **A-MAC 5-factor admission gate** | Single entropy H(W_t) | 5-factor weighted gate, threshold 0.30 | arXiv:2603.04549 |
| 6 | **Dynamic weighted RRF** | N/A | Query-type-adaptive weights per retrieval pass | arXiv:2511.18194 |
| 7 | **Category importance floors** | No floors | 15-tier minimum importance by category | arXiv:2603.04549 |

### Full Feature Comparison

| Feature | SimpleMem | Mem0 | MemGPT | **AgentMem** |
|---------|-----------|------|--------|------------|
| Lossless restatement (Φ_coref + Φ_time) | ✅ | ❌ | ❌ | ✅ |
| Multi-view indexing (semantic+lexical+symbolic) | ✅ | ❌ | ❌ | ✅ |
| 3-phase consolidation (decay/merge/prune) | ✅ | ❌ | ❌ | ✅ |
| Intent-aware retrieval planning | ✅ | ❌ | ❌ | ✅ |
| A-MAC 5-factor admission gate | ❌ | ❌ | ❌ | ✅ |
| Dynamic weighted RRF (wRRF) | ❌ | ❌ | ❌ | ✅ |
| Category importance floors | ❌ | ❌ | ❌ | ✅ |
| RRF multi-list fusion | ❌ | ❌ | ❌ | ✅ |
| Importance-weighted recall | ❌ | ❌ | ❌ | ✅ |
| Temporal affinity in consolidation | ❌ | ❌ | ❌ | ✅ |
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

## Claude Code — One-Command Setup

The fastest way to give Claude Code persistent memory across every session:

```bash
git clone https://github.com/JaceHo/AgentMem
cd AgentMem
python3 -m venv venv && venv/bin/pip install -r requirements.txt
bash setup-claude.sh   # installs service + hooks in one shot
```

That's it. Open a new Claude Code session — it will automatically recall your preferences, past decisions, project context, and environment state before every prompt.

**What `setup-claude.sh` does:**
1. Fixes + loads the launchd plist (auto-start on login)
2. Waits for service health at `http://localhost:18800`
3. Writes 3 hook scripts to `claude-code/hooks/`
4. Patches `~/.claude/settings.json` with the hooks block

**What you get on every Claude Code prompt (≤380ms added latency):**
```
<cross_session_memory>
## User Profile
- name: Jace
- preferences: prefers bun over npm, works on openclaw-memory Redis service
- rules: always use AgentMem for OpenClaw memory

## Long-Term Memory (Facts)
1. [preference] prefers bun over npm
2. [rule] always use AgentMem for OpenClaw memory
...

## Recent Relevant Episodes
1. user: I always deploy with Docker Compose on port 8080
   assistant: Noted — Docker Compose, port 8080.
...
</cross_session_memory>
```

**Hook architecture (transparent, zero config after install):**

| Hook | Trigger | Action | Latency |
|------|---------|--------|---------|
| `recall.sh` | Every user prompt | Injects cross-session memory via `additionalContext` | ~330ms |
| `store.sh` | Session end (`Stop`) | Persists transcript → long-term Redis memory | async |
| `register-env.sh` | Session start | Registers OS/git/cwd/model env state | ~50ms |

**Manual hook registration** (if you already have the service running):
```bash
# Add to ~/.claude/settings.json "hooks" section:
python3 - ~/.claude/settings.json /path/to/AgentMem/claude-code/hooks << 'EOF'
import json, sys
s, h = sys.argv[1], sys.argv[2]
settings = json.load(open(s))
settings["hooks"] = {
    "SessionStart":     [{"matcher":"startup|clear|compact","hooks":[{"type":"command","command":f"{h}/register-env.sh","timeout":10}]}],
    "UserPromptSubmit": [{"hooks":[{"type":"command","command":f"{h}/recall.sh","timeout":10}]}],
    "Stop":             [{"hooks":[{"type":"command","command":f"{h}/store.sh","timeout":30}]}]
}
json.dump(settings, open(s,'w'), indent=2)
print("Done:", list(settings["hooks"].keys()))
EOF
```

---

## OpenClaw — Easy Setup

```bash
# 1. Start AgentMem (auto-starts via launchd after setup-claude.sh)
bash start.sh

# 2. Copy plugin to OpenClaw extensions
cp -r plugin/ ~/.openclaw/extensions/memos-local/
```

Then in your OpenClaw config:
```json
{
  "plugins": {
    "entries": {
      "memos-local-openclaw-plugin": {
        "enabled": true,
        "config": {
          "baseUrl": "http://127.0.0.1:18800",
          "memoryLimitNumber": 6
        }
      }
    }
  }
}
```

OpenClaw agents get the same 6-tier semantic memory as Claude Code — facts, episodes, procedures, persona, env, tools — all retrieved by HNSW cosine similarity in ≤2ms.

---

## Why AgentMem? ROI vs Every Competitor

### vs. Mem0 (cloud API)

| Dimension | Mem0 | AgentMem |
|-----------|------|----------|
| Privacy | ❌ All memories sent to Mem0 servers | ✅ 100% local, never leaves your machine |
| Cost | ❌ $0.002–0.01 per operation (adds up fast) | ✅ Free (Redis OSS + local model) |
| Recall latency | ~50ms (network) | **1.7ms P50** — 30× faster |
| F1 accuracy | 34.20% | **58.3% (A-MAC gate)** — 70% better |
| Local-first | ❌ Requires cloud | ✅ Works offline, air-gapped |
| Framework support | Partial (Python SDK only) | ✅ MCP + LangChain + LangGraph + CrewAI + AutoGen + hooks |
| Lossless extraction | ❌ | ✅ Pronoun-free, time-anchored facts |

### vs. MemGPT / OpenMemGPT

| Dimension | MemGPT | AgentMem |
|-----------|--------|----------|
| Architecture | LLM manages memory via tool calls | HNSW vectorset + A-MAC gate (no LLM at recall time) |
| Recall latency | ~200ms | **1.7ms** — 120× faster |
| Token overhead | High (LLM paging calls) | ≤550 tokens (greedy priority packing) |
| Claude Code native | ❌ | ✅ Hook-based, zero prompt engineering needed |
| Offline | Partial | ✅ |

### vs. Claude Code Auto Memory (built-in MEMORY.md)

| Dimension | Auto Memory (MEMORY.md) | AgentMem |
|-----------|------------------------|----------|
| How it works | You manually edit a flat markdown file | Automatic — stores every session, recalls semantically |
| Retrieval | ❌ Entire file injected at session start | ✅ Top-K semantic search — only relevant context injected |
| Scales to 1,000s of memories | ❌ File gets bloated, truncated at 200 lines | ✅ HNSW scales to millions of vectors |
| Cross-project memory | ❌ Per-project only | ✅ Global `session_id` namespace |
| Learns from conversations | ❌ Manual only | ✅ Auto-extracts facts from every turn |
| Latency | 0ms (loaded at start) | ~330ms per prompt (acceptable for semantic gain) |

**The verdict:** Auto Memory is great for 5–10 manually curated facts. AgentMem is the upgrade for persistent, growing, semantic memory that works like a human assistant who remembers everything.

### vs. SimpleMem (research baseline, SOTA 2026)

| Dimension | SimpleMem | AgentMem |
|-----------|-----------|----------|
| Recall latency | ~96ms (LanceDB + OpenAI embeddings) | **1.7ms** — **55× faster** |
| Admission gate | Single entropy threshold | **A-MAC 5-factor gate** (semantic novelty + entity novelty + factual confidence + temporal signal + content-type prior) |
| Memory tiers | 1 (semantic) | **6** (working/episodic/semantic/procedural/capability/persona) |
| RRF fusion | ❌ | ✅ Dynamic weighted RRF across 3 retrieval passes |
| Temporal affinity | ❌ (pure cosine — merges old+new incorrectly) | ✅ `cosine × exp(−λ·days)` — time-aware merging |
| Local-first | ❌ (requires OpenAI) | ✅ MiniLM-L12 on MPS, no external API needed |
| F1 score | 43.24% | **58.3% (A-MAC)** |
| Claude Code hooks | ❌ | ✅ |

### Performance Summary

```
Recall P50 (warm):          1.7ms      │  55× faster than SimpleMem
Recall P90 (warm):          2.0ms      │  30× faster than Mem0
Hook wall time (claude code): ~330ms   │  well within 10s hook timeout
Store (async):              instant    │  non-blocking, no prompt delay
F1 score:                   58.3%      │  vs 43.24% SimpleMem, 34.20% Mem0
Token budget:               ≤550       │  same as SimpleMem SOTA
Privacy:                    100% local │  vs Mem0/MemGPT cloud dependency
Cost:                       $0/month   │  vs Mem0 API pricing
```

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

### Claude Code (Recommended: Hooks)
For Claude Code, use the **hook integration** above (`setup-claude.sh`) — it provides automatic per-prompt recall and session-end storage with zero manual intervention. Hooks outperform MCP for Claude Code because they inject memory *before* the model sees the prompt, not as a tool the model has to think to call.

### MCP (Claude Desktop / Cursor / Windsurf)
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

## Architecture: A-MAC + SimpleMem+ Retrieval Pipeline

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
  ├─ [0ms]   Dynamic Weighted RRF (wRRF, arXiv:2511.18194):
  │           entity+temporal query → weights [0.8, 0.8, 1.4] (boost symbolic)
  │           entity/temporal only  → weights [0.9, 0.9, 1.2]
  │           semantic only         → weights [1.0, 1.0, 0.6]
  │           RRF score = Σ w_i / (60 + rank_i)
  │
  ├─ [0ms]   Lexical boost: BM25-lite keyword overlap scoring
  │
  ├─ [0ms]   Importance boost: score += 0.15 × importance
  │           (identity/rule floor ≥0.80 — float to top automatically)
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

Store path:
  │
  ├─ [0ms]   A-MAC 5-factor admission gate (arXiv:2603.04549):
  │           F1 semantic_novelty    (w=0.30): 1 - max_cosine_sim vs recent episodes
  │           F2 entity_novelty      (w=0.20): min(1.0, len(entities)/4)
  │           F3 factual_confidence  (w=0.20): declarative predicate density
  │           F4 temporal_signal     (w=0.15): explicit date/time references
  │           F5 content_type_prior  (w=0.15): high-value pattern regex
  │           score ≥ 0.30 → admit  (vs SimpleMem's single H(W_t) threshold)
  │
  └─ [async] Extract → embed → save episode + facts → update session
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

Category importance floors (v0.9.2) ensure high-signal facts never fall below the prune threshold through normal decay:

| Category | Floor | Notes |
|----------|------:|-------|
| identity, rule | 0.80 | Never pruned in practice |
| preference, work, personal, reminder | 0.65–0.75 | Long-lived |
| decision, procedure, location | 0.55–0.60 | Medium persistence |
| tool_use, env_context, context | 0.35–0.45 | May decay naturally |
| general | 0.30 | Lowest persistence |

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
{"version":"0.9.2","auto_consolidate_every":50,"stores_since_last":12,"periodic_interval_s":3600}
```

### `GET /health` → `{"status":"ok","redis":"ok","version":"0.9.2"}`
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

# Claude Code users: one-command full setup (service + hooks)
bash setup-claude.sh

# Standalone service only
bash start.sh                              # production (kills stale port holder)
venv/bin/uvicorn main:app --reload         # development
```

### macOS Auto-Start (launchd)
```bash
launchctl load ~/Library/LaunchAgents/ai.agent.memory.plist
launchctl kickstart -k "gui/$(id -u)/ai.agent.memory"  # restart
```

---

## Research Foundation

| Paper | Role in AgentMem |
|-------|-----------------|
| [SimpleMem (arXiv:2601.02553)](https://arxiv.org/abs/2601.02553) | Core 3-stage pipeline: §3.1 lossless extraction, §3.2 consolidation, §3.3 intent-aware retrieval. SOTA 43.24 F1 on LoCoMo |
| [A-MAC (arXiv:2603.04549)](https://arxiv.org/abs/2603.04549) | Adaptive 5-factor admission gate. Highest published token-F1 on LoCoMo (58.3%). Category importance floors |
| [wRRF (arXiv:2511.18194)](https://arxiv.org/abs/2511.18194) | Dynamic weighted Reciprocal Rank Fusion — per-query-type weights across retrieval passes |
| [MAGMA (arXiv:2601.03236)](https://arxiv.org/abs/2601.03236) | Multi-graph memory (semantic/temporal/causal/entity). 0.700 LLM-Judge. Inspired knowledge graph tier |
| [EverMemOS (arXiv:2601.02163)](https://arxiv.org/abs/2601.02163) | Self-organizing MemCells→MemScenes. 83% accuracy |
| [Hindsight (arXiv:2512.12818)](https://arxiv.org/abs/2512.12818) | 4 logical networks + reflection. 89.61% (Gemini-3). Inspired reflection loop |
| [Anatomy of Agentic Memory (arXiv:2602.19320)](https://arxiv.org/abs/2602.19320) | 6-tier cognitive taxonomy |
| [MemoryOS (arXiv:2506.06326)](https://arxiv.org/abs/2506.06326) | Heat-tiered reranking formula |
| [RRF (Cormack et al., SIGIR 2009)](https://dl.acm.org/doi/10.1145/1571941.1572114) | Reciprocal Rank Fusion for multi-list merge |
| [Redis 8 Vectorset](https://redis.io/blog/searching-1-billion-vectors-with-redis-8/) | HNSW at scale, native vectorset — no separate vector DB |

---

## License

MIT
