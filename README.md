# openclaw-memory

**Local long-term memory + agent capability registry for [OpenClaw](https://openclaw.ai) agents.**

Drop-in replacement for the `memos-cloud-openclaw-plugin` (MemTensor cloud API).
Fully local, persistent, fast — no external dependencies.

## Features

| Feature | Description |
|---------|-------------|
| **5-Tier cognitive memory** | Episodic, Semantic, Procedural, Capability, and Working memory — each modeled on distinct cognitive structures |
| **Procedural memory** *(v0.6.0)* | Stores *how to do things* (workflows, tool patterns, successful procedures) for cross-session skill retention |
| **Atomized fact extraction** | SimpleMem Stage 1: LLM produces self-contained atomic facts — coreference resolved ("he" → "Bob"), time anchored ("tomorrow" → "2025-11-16") |
| **Hybrid extraction** | Dual-layer: regex (~1ms) + LLM/GLM-4-flash (~200-500ms) for Chinese/implicit/contextual facts |
| **Entropy-aware gate** *(v0.8.0)* | SimpleMem Stage 1 ingestion filter: H(W_t) = entity_novelty + semantic_divergence. Discards low-information turns before embedding/LLM cost |
| **Keyword lexical boost** *(v0.8.0)* | SimpleMem Stage 3 lexical layer: BM25-lite keyword overlap scoring after KNN retrieval + symbolic time-range filter |
| **Structured metadata** | Facts carry `keywords[]`, `persons[]`, `entities[]` for rich recall context |
| **Adaptive retrieval depth** | Query complexity estimator C_q dynamically adjusts k — SimpleMem Stage 3 intent-aware planning (k_dyn = k_base × (1 + δ × C_q)) |
| **Memory consolidation** | Async semantic synthesis with temporal affinity (cos × e^(-λ·days)) — facts from different time periods stay separate |
| **Semantic recall** | Redis 8 native vectorset HNSW + `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) |
| **Heat-tiered ranking** | Frequently/recently accessed memories rank higher |
| **Scene isolation** | Language + domain tags filter recall to relevant context |
| **Evolving persona** | Structured user profile updated incrementally from conversations |
| **Tool/Skill Index** *(v0.5.0)* | Agent registers available tools/skills; semantic search finds relevant tools by description |
| **Environment Registry** *(v0.5.0)* | Persists OS/shell/CWD/git/MCP state across sessions |
| **Capability-Aware Recall** *(v0.5.0)* | `/recall` automatically includes tool context when query is capability-related |
| **Live Dashboard** *(v0.6.0)* | Web UI at `localhost:18800` — memory stats, tool index, env state, live log stream |
| **SSE Log Streaming** *(v0.6.0)* | Real-time log events via `GET /logs/stream` (Server-Sent Events) |

## Architecture

```
OpenClaw Gateway (Node.js)
    │  lifecycle hooks
    ▼
memos-local plugin (JS) v0.3.0  ← ~/.openclaw/extensions/memos-local/
    │  before_agent_start: recall + register-tools + register-env
    │  agent_end:          store (fire-and-forget)
    │  HTTP POST
    ▼
Memory Service (Python FastAPI) v0.8.0 @ localhost:18800
    ├── GET  /                    → Dashboard (5-tab web UI)
    ├── POST /recall              → embed query → adaptive k → KNN (scene-filtered) → heat rerank
    │                               + tool context if capability query → prependContext
    ├── POST /store               → filter → detect scene → summarize → atomized extract → embed → dedup → save
    │                               → procedure facts extracted to mem:procedures
    │                               → accumulate Tier 1 rolling session summary
    ├── POST /session/compress    → promote Tier 1 session KV → Tier 2 long-term (NEW v0.7.0)
    ├── GET  /session/{id}        → inspect current Tier 1 session state (NEW v0.7.0)
    ├── POST /register-tools      → embed tool descriptions → upsert into mem:tools
    ├── POST /register-env        → store env state in mem:env hash
    ├── POST /recall-tools        → semantic search over tool index
    ├── POST /recall-procedures   → semantic search over procedure/workflow index
    ├── POST /store-procedure     → manually store a workflow/how-to
    ├── GET  /capabilities        → full capability manifest JSON
    ├── GET  /capabilities/context → formatted context strings
    ├── GET  /logs/stream         → SSE real-time log stream (text/event-stream)
    └── POST /consolidate         → find similar facts → LLM merge → deduplicate
         │
         ▼
    Redis 8 (localhost:6379)
         ├── mem:episodes   (HNSW vectorset — episodic memory)
         ├── mem:facts      (HNSW vectorset — semantic facts + metadata)
         ├── mem:tools      (HNSW vectorset — tool/skill capability index)
         ├── mem:procedures (HNSW vectorset — procedural memory: how-to workflows)  ← NEW v0.6.0
         ├── mem:persona    (Hash — evolving user profile)
         ├── mem:env        (Hash — current environment state)
         ├── mem:agent      (Hash — agent self-model)
         └── mem:session:*  (String+TTL — working memory, 4h)
```

## Memory Tiers

Based on cognitive science taxonomy (arXiv:2602.19320 "Anatomy of Agentic Memory"):

| Tier | Redis Key | What it stores | TTL |
|------|-----------|----------------|-----|
| **Working** | `mem:session:*` | Current session context | 4h |
| **Episodic** | `mem:episodes` | Conversation turns (what happened) | Permanent |
| **Semantic** | `mem:facts` | Distilled facts (what is true) | Permanent |
| **Procedural** | `mem:procedures` | Workflows & how-tos (how to do X) | Permanent |
| **Capability** | `mem:tools` + `mem:env` + `mem:agent` | Agent tools, environment, self-model | Permanent |
| **Persona** | `mem:persona` | Evolving user profile | Permanent |

---

## What's New in v0.8.0 (SimpleMem Full Implementation)

Learned from and adapted **SimpleMem** ([arXiv:2601.02553](https://arxiv.org/abs/2601.02553), [GitHub](https://github.com/aiming-lab/SimpleMem), UNC-Chapel Hill / UC Berkeley / UCSC). SimpleMem is the current SOTA for long-term agent memory:

| System | F1 Score | Total Time | Token Cost |
|--------|:--------:|:----------:|:----------:|
| A-Mem | 32.58% | 5937s | — |
| Mem0 | 34.20% | 1934s | ~1000 |
| **SimpleMem** | **43.24%** | **481s** | **~550** |
| Full Context | 18.70% | — | ~16,910 |

Cross-session (Feb 2026): **SimpleMem 48 vs Claude-Mem 29.3 (+64%)**.

SimpleMem's core insight: **memory is metabolism, not storage**. Three pipeline stages implement "semantic lossless compression" — filtering noise at ingestion, fusing redundant fragments on-the-fly, and adapting retrieval depth to query complexity.

---

### Stage 1: Semantic Structured Compression → our `_entropy_gate()` + atomization

SimpleMem's implicit semantic density gate filters low-information windows before they enter the memory pipeline. It reformulates raw dialogue into **atomic memory units** — self-contained facts with resolved coreferences and absolute timestamps.

**What we adapted:** an explicit pre-filter gate that combines two signals:

```
H(W_t) = α × entity_novelty + β × semantic_divergence
         = 0.35 × min(1, entities/4) + 0.65 × (1 − max_cosine_sim(text, last 3 episodes))
```

- **entity_novelty**: named entities, ISO dates, filenames, acronyms — new structure = higher entropy
- **semantic_divergence**: 1 − cosine similarity against recent stored episodes — near-duplicate = lower entropy
- **Threshold**: H < 0.35 → discard before any embedding/summarize/LLM call

```
[store] entropy gate filtered (H=0.27): 'ok, sounds good!'    ← noise, discarded
[store] session=abc lang=en ep+1 facts+2 388ms                 ← high-entropy, stored
```

Atomization (coreference resolution + temporal normalization) was already implemented in our LLM extraction prompt since v0.4.0:

```diff
- "He'll go tomorrow at 2pm"         [❌ ambiguous pronouns, relative time]
+ "Bob will go 2025-11-16T14:00:00"  [✅ resolved, ISO-8601 absolute]
```

SimpleMem ablation: removing atomization alone drops temporal reasoning F1 from **58.62 → 25.40** (−57%).

SimpleMem uses three complementary index layers per memory unit:

| Layer | Type | Purpose |
|-------|------|---------|
| **Semantic** | Dense vector (1024-dim) | Conceptual similarity matching |
| **Lexical** | BM25 sparse | Exact-term matching (names, IDs) |
| **Symbolic** | Structured metadata | Hard filters (timestamp, entity type) |

Our adaptation: 384-dim MiniLM multilingual embeddings (Chinese/Japanese/English), with keyword lexical boost and Redis VSIM FILTER for symbolic constraints.

---

### Stage 2: Online Semantic Synthesis → our temporal-affinity consolidation

SimpleMem synthesizes related fragments **on-the-fly during the write phase** — memories are fused immediately, keeping the memory graph always compact. Example:

```diff
- Fragment 1: "User wants coffee"
- Fragment 2: "User prefers oat milk"
- Fragment 3: "User likes it hot"
+ Synthesized: "User prefers hot coffee with oat milk"
```

**Our adaptation:** async consolidation (deferred to `/consolidate` calls) with the key upgrade — temporal affinity score instead of raw cosine:

```
affinity = cosine_sim × exp(−λ × |days_between|)   [λ = 0.03 → 23-day half-life]
```

Two facts can be semantically identical yet represent different time periods. Without temporal decay, "user uses Python 2.7" from 2023 would incorrectly merge with "user uses Python 3.12" from 2026. The exponential decay prevents cross-temporal consolidation.

Trade-off vs. SimpleMem: our async approach has lower write latency; SimpleMem's online synthesis keeps the index always minimal. Both are valid — ours favours response speed.

SimpleMem ablation: removing consolidation drops multi-hop reasoning F1 by **−31.3%**.

---

### Stage 3: Intent-Aware Retrieval Planning → our adaptive depth + hybrid scoring

SimpleMem generates a full **retrieval plan** `{q_sem, q_lex, q_sym, d}` using LLM reasoning, then runs parallel multi-view retrieval across all three index layers.

**What we adapted:**

**Adaptive depth** (k_dyn, since v0.4.0):
```
k_dyn = ⌊k_base × (1 + δ × C_q)⌋
```
C_q ∈ [0,1] estimated from query length + signal words (what/when/compare/所有) — zero LLM cost.

**Keyword lexical boost** (BM25-lite, v0.8.0):
After KNN retrieval, stored `keywords[]` overlap with query tokens adds score:
```
score += 0.06 × kw_overlap + 0.018 × content_overlap
```
Named entities ("Bob", "Redis", "2025-11-20") surface higher even when dense vectors rank them lower.

**Symbolic time-range filter** (v0.8.0):
```json
{ "query": "what happened last February", "time_from": 1738368000000, "time_to": 1740787199000 }
```
Passed as Redis VSIM FILTER: `.ts >= N && .language == "zh"` — combined with scene filter, zero post-processing cost.

SimpleMem ablation: removing adaptive retrieval drops open-domain QA F1 by **−26.6%**.

---

### Implementation Mapping Summary

| SimpleMem Stage | SimpleMem Mechanism | Our Implementation | Since |
|----------------|--------------------|--------------------|-------|
| Stage 1 gate | Implicit semantic density gate (LLM-integrated) | Explicit `_entropy_gate()` H(W_t) ≥ 0.35 | v0.8.0 |
| Stage 1 atomize | Coreference resolution + temporal normalization | LLM extraction prompt (Φ_coref, Φ_time) | v0.4.0 |
| Stage 1 index | 3-layer: dense (1024-d) + BM25 + symbolic | Dense (384-d MiniLM) + keyword boost + VSIM FILTER | v0.8.0 |
| Stage 2 synthesis | On-the-fly during write phase | Async `/consolidate` (deferred) | v0.4.0 |
| Stage 2 affinity | Content similarity only | cos × exp(−λ·days) temporal affinity | v0.8.0 |
| Stage 3 retrieval | LLM-generated retrieval plan | Zero-cost C_q complexity estimator | v0.4.0 |
| Stage 3 hybrid | Parallel semantic + lexical + symbolic | KNN + keyword boost + time-range FILTER | v0.8.0 |

---

## What's New in v0.7.0 (Session Tier Promotion)

Implements the **压缩 (compression/promotion)** transport layer from the 2026 layered controlled architecture — the piece that prevents Tier 1 session data from silently expiring.

### Session Rolling Summary (Tier 1 accumulation)

Previously, `/store` overwrote the session KV with just the last user text. Now it **appends** each turn's summary to a rolling session buffer. When the buffer exceeds 1,200 chars, it's compressed via LLM before appending:

```
Turn 1 → "User asked about bun vs npm..."
Turn 2 → "User asked about bun vs npm...\n---\nUser set up Redis..."
Turn 3 → (buffer > 1200 chars → LLM compress) → "User uses bun, dislikes npm. Set up Redis 8 on port 6379. Working on openclaw-memory service."
```

### Session → Long-term Promotion (`/session/compress`)

At session end, the JS plugin calls `/session/compress`. This crystallises the accumulated session into **permanent long-term memory** before the 4h TTL expires:

```bash
# Trigger manually (async)
curl -X POST http://localhost:18800/session/compress \
  -d '{"session_id": "abc123", "wait": false}'

# Sync — wait and see what was promoted
curl -X POST http://localhost:18800/session/compress \
  -d '{"session_id": "abc123", "wait": true}'
# → {"status": "ok", "ep_saved": 1, "facts_saved": 3, "session_id": "abc123"}
```

What happens during promotion:
1. Accumulated Tier 1 session text → LLM summarize
2. Saved as `[Session Summary]` episode in `mem:episodes` (Tier 2)
3. Hybrid fact extraction → saved to `mem:facts` + persona updated
4. Session KV deleted (consumed into long-term memory)

### Inspect Session State

```bash
curl http://localhost:18800/session/abc123
# → {"session_id": "abc123", "context": "User uses bun...", "length": 312, "tier": "Tier 1 / Session KV"}
```

---

## What's New in v0.6.0 (Procedural Memory + Dashboard)

### Procedural Memory (`mem:procedures` vectorset)

The missing 4th cognitive tier identified in arXiv:2602.19320 — storing *how to do things* rather than facts about the world. Every successful workflow, tool pattern, or step-by-step procedure is embedded and indexed:

```bash
# Store a workflow
curl -X POST http://localhost:18800/store-procedure \
  -d '{"task": "search codebase for a class definition",
       "procedure": "Use Glob tool with pattern **/*.py, then Grep for class ClassName",
       "tools_used": ["Glob", "Grep"]}'

# Recall: semantic search by task description
curl -X POST http://localhost:18800/recall-procedures \
  -d '{"query": "find where a function is defined in the repo", "k": 3}'
# → [{"task": "search codebase for a class definition",
#      "procedure": "Use Glob tool...", "tools_used": ["Glob", "Grep"], "score": 0.91}]
```

Procedures are also auto-extracted from conversations: when the extractor detects `procedure` category facts ("to fix X, I used Y"), they're saved directly to `mem:procedures`.

New extraction patterns:
- `procedure` — "To fix this bug, use `git stash` then rebase"
- `tool_use` — "Used Bash tool to run npm install"
- `capability_gained` — "Installed Docker MCP server"
- `env_change` — "Switched to Python 3.12 environment"
- `env_context` — "Current project directory is /code/app on branch main"

### Live Dashboard

Open `http://localhost:18800/` for the web UI — no setup required:

```
┌─────────────────────────────────────────────────────┐
│  openclaw-memory v0.6.0          ● Connected         │
│                                                       │
│  [ Overview ] [ Tools ] [ Env ] [ Logs ] [ API ]     │
│                                                       │
│  EPISODIC    SEMANTIC   PROCED.   CAPABILITY          │
│    57           17        8         15                │
│                                                       │
│  ENVIRONMENT  PERSONA                                  │
│       8          4                                    │
└─────────────────────────────────────────────────────┘
```

Tabs:
- **Overview** — health dot, Redis status, memory tier counters, persona display (auto-refresh every 8s)
- **Tools** — searchable tool index grouped by category with use_count badges
- **Environment** — `mem:env` state hash, agent self-model, active MCPs/plugins
- **Logs** — real-time log stream via SSE; level filter (DEBUG/INFO/WARNING/ERROR); text search; error badge
- **API Reference** — all endpoints with live "Try →" buttons

### SSE Log Streaming

```bash
# Stream logs in real time (curl — ctrl+C to stop)
curl -N http://localhost:18800/logs/stream

# Each event:
# data: {"ts": 1741305123.4, "event": "store", "level": "INFO", "msg": "[store] saved 3 facts"}
```

---

## What's New in v0.5.0 (Agent Capability Registry)

Agents are no longer amnesiac about their own powers. Three memory structures give every session persistent self-awareness:

### Tool/Skill Index (`mem:tools` vectorset)

On `before_agent_start`, the plugin sends the full tool list to `/register-tools`. Each tool description is embedded into 384-dim space — enabling **semantic capability search**:

```bash
curl -X POST http://localhost:18800/recall-tools \
  -d '{"query": "search the filesystem for files matching a pattern", "k": 5}'
# → [{"name": "Glob", "description": "...", "score": 0.94, "category": "file"}]
```

Tool categories auto-detected: `file`, `system`, `search`, `code`, `ai`, `data`, `git`, `web`, `memory`, `skill`, `mcp`

### Environment Registry (`mem:env` hash)

Persists OS/shell/CWD/git-branch/active-MCPs across sessions. Automatically populated from agent context:

```bash
curl -X POST http://localhost:18800/register-env \
  -d '{"os": "darwin", "shell": "zsh", "cwd": "/Users/jace/code", "git_branch": "main",
       "active_mcps": ["filesystem", "github"], "agent_model": "claude-sonnet-4-6"}'
```

### Capability-Aware Recall

When a query is detected as capability-related ("what tools do you have?", "can you search files?"), `/recall` automatically includes the most relevant tools in `prependContext`:

```
## Available Tools (Relevant)
- **Glob** [file/builtin]: Fast file pattern matching — supports **/*.js etc.
- **Grep** [search/builtin]: Search file content with regex
- **Bash** [system/builtin]: Execute shell commands
```

---

## What's New in v0.4.0 (SimpleMem-Inspired)

### Atomized Fact Extraction

Inspired by [SimpleMem](https://github.com/aiming-lab/SimpleMem) (SOTA on LoCoMo benchmark, 43.24 F1), facts are extracted as **atomic entries** — self-contained sentences that can be understood without surrounding context:

| Before (v0.3.0) | After (v0.4.0) |
|------------------|-----------------|
| `I like bun` | `用户偏好使用 bun 作为 JavaScript 包管理器，明确拒绝 npm` |
| `my name is Jace` | `用户名为 Jace` |
| `always use GLM-4-flash` | `用户设定规则：轻量级 LLM 任务永远优先使用 GLM-4-flash` |

Techniques applied:
- **Coreference resolution (Φ_coref)**: No pronouns — "he" → "Bob", "I" → "用户"
- **Temporal anchoring (Φ_time)**: "yesterday" → "2026-02-28"
- **Structured metadata**: Each fact stores `keywords[]`, `persons[]`, `entities[]`

### Adaptive Retrieval Depth

```
k_dyn = k_base × (1 + δ × C_q)
```

Where `C_q` is estimated from query features (word count, signal words) at zero cost. Simple queries get fewer results; complex queries get deeper retrieval.

### Memory Consolidation

```bash
curl -X POST http://localhost:18800/consolidate/sync
# → {"merged": 1, "removed": 1, "total_before": 18, "total_after": 17, "ms": 1267}
```

---

## Heat Scoring

Each recalled memory bumps its `access_count`. Re-ranking formula:

```
heat  = exp(-days_old / 30) × (1 + log(1 + access_count) × 0.25)
score = cosine_similarity × heat
```

New memories start at `heat ≈ 1.0`. Old, never-accessed memories decay to `~0.05`.

## Scene Isolation

On every recall, query language and domain are detected (regex + keyword matching).
A Redis VSIM FILTER expression narrows results to the same scene first.
Global search supplements if scene results are sparse.

Detected domains: `trading`, `coding`, `devops`, `ai`, `finance`, `capability`, `general`

## Hybrid Fact Extraction

Two-layer extraction runs on every `/store` call (async, non-blocking):

| Layer | Method | Latency | Catches |
|-------|--------|---------|---------|
| **LLM** | GLM-4-flash (ZAI) | ~1-5s | Chinese, implicit, contextual, decisions, third-person. **Atomized output.** |
| **Regex** | 18 compiled patterns | ~1ms | English first-person + procedural + capability patterns (fallback) |

Categories: `identity`, `work`, `preference`, `location`, `personal`, `reminder`, `rule`, `decision`, `context`, `credential`, `tool_use`, `capability_gained`, `env_change`, `env_context`, `procedure`

---

## Setup

### Requirements
- Python 3.12+
- Redis 8+ (community edition, `vectorset` module built-in)

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env`:
```bash
# Optional: ZAI API key for LLM summarization + extraction (GLM-4-flash)
ZAI_API_KEY=your-key-here
```

Both summarization and LLM extraction degrade gracefully without a key (summarizer truncates to 180 chars; extractor falls back to regex-only).

### Running

```bash
# Development (auto-reload)
venv/bin/uvicorn main:app --host 127.0.0.1 --port 18800 --reload

# Production — start.sh kills any stale port holder before starting
bash start.sh
```

### macOS LaunchAgent (auto-start on login)

```bash
# Load (first time)
launchctl load ~/Library/LaunchAgents/ai.openclaw.memory.plist

# Restart (kill old process + restart)
launchctl kickstart -k "gui/$(id -u)/ai.openclaw.memory"

# Stop
launchctl unload ~/Library/LaunchAgents/ai.openclaw.memory.plist

# Check status
launchctl print "gui/$(id -u)/ai.openclaw.memory"
```

The LaunchAgent uses `start.sh` as the entry point, which:
1. Kills any stale process on port 18800 with `kill -9`
2. Waits up to 8 seconds for the port to free
3. Starts uvicorn with `--timeout-graceful-shutdown 0` for instant clean shutdown

Environment variables set by plist: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` (prevents semaphore leaks on restart), `TOKENIZERS_PARALLELISM=false`, `HF_HUB_OFFLINE=1`.

### OpenClaw Plugin

Install the JS plugin shim:
```bash
cp -r plugin/ ~/.openclaw/extensions/memos-local/
```

Add to `~/.openclaw/openclaw.json`:
```json
{
  "plugins": {
    "entries": {
      "memos-local-openclaw-plugin": {
        "enabled": true,
        "config": { "baseUrl": "http://127.0.0.1:18800", "memoryLimitNumber": 6 }
      }
    },
    "allow": ["memos-local-openclaw-plugin"]
  }
}
```

---

## API Reference

### `GET /`
Dashboard web UI (5-tab interface). No auth required.

### `POST /recall`
```json
{ "query": "user's current message", "session_id": "abc", "memory_limit_number": 6, "include_tools": true }
```
Returns `{ "prependContext": "...", "latency_ms": 45 }`

Retrieval depth adapts to query complexity. When `include_tools: true` (default) and query is capability-related, relevant tools are appended to `prependContext`.

### `POST /store`
```json
{ "messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}], "session_id": "abc" }
```
Returns `{ "status": "queued" }` (async background processing)

Facts with category `procedure` are automatically also stored to `mem:procedures`.
Each turn's summary is appended to the Tier 1 rolling session buffer.

### `POST /session/compress` *(v0.7.0)*
```json
{ "session_id": "abc", "wait": false }
```
Promotes Tier 1 accumulated session context → Tier 2 long-term memory. Set `wait: true` for synchronous result.
Returns `{ "status": "ok", "ep_saved": 1, "facts_saved": 3, "session_id": "abc" }`

Called automatically by the JS plugin on `agent_end`.

### `GET /session/{session_id}` *(v0.7.0)*
Inspect the current Tier 1 session buffer.
Returns `{ "session_id": "abc", "context": "...", "length": 312, "tier": "Tier 1 / Session KV" }`

### `POST /consolidate`
Trigger async memory consolidation. Returns `{ "status": "consolidation_queued" }`

### `POST /consolidate/sync`
Synchronous consolidation. Returns `{ "merged": N, "removed": N, "total_before": N, "total_after": N, "ms": N }`

### `POST /register-tools` *(v0.5.0)*
```json
{
  "tools": [
    {"name": "Glob", "description": "Fast file pattern matching", "source": "builtin", "parameters": ["pattern", "path"]},
    {"name": "web_search", "description": "Search the web", "source": "mcp", "parameters": ["query"]}
  ],
  "agent_id": "session-abc"
}
```
Returns `{ "status": "queued", "tool_count": 2 }` (async, non-blocking)

### `POST /register-env` *(v0.5.0)*
```json
{
  "os": "darwin", "shell": "zsh", "cwd": "/Users/jace/code",
  "git_branch": "main", "active_mcps": ["filesystem", "github"],
  "agent_model": "claude-sonnet-4-6", "session_id": "abc"
}
```
Returns `{ "status": "ok", "fields": ["os", "shell", ...] }`

### `POST /recall-tools` *(v0.5.0)*
```json
{ "query": "search the web for information", "k": 5, "source": "mcp" }
```
Returns `{ "tools": [{name, description, category, score, use_count}], "count": 3, "latency_ms": 2 }`

### `POST /recall-procedures` *(v0.6.0)*
```json
{ "query": "how do I search for a file in the codebase", "k": 3 }
```
Returns `{ "procedures": [{task, procedure, tools_used, score}], "count": 2, "latency_ms": 3 }`

### `POST /store-procedure` *(v0.6.0)*
```json
{
  "task": "search codebase for class definition",
  "procedure": "Use Glob with **/*.py, then Grep for 'class ClassName'",
  "tools_used": ["Glob", "Grep"],
  "domain": "coding"
}
```
Returns `{ "status": "ok", "uid": "01JNXX..." }`

### `GET /capabilities` *(v0.5.0)*
Returns full capability manifest:
```json
{
  "tools": [...],
  "env": {"os": "darwin", "cwd": "...", "active_mcps": [...]},
  "agent": {...},
  "stats": {"tool_count": 15, "categories": ["file", "system", "search"], "mcp_count": 2}
}
```

### `GET /capabilities/context` *(v0.5.0)*
Returns formatted context strings (for debugging what the agent sees):
```json
{
  "persona_context": "## User Profile\n- name: Jace\n...",
  "env_context": "## Agent Environment\n- os: darwin\n...",
  "tool_index_by_category": {"file": ["Glob", "Grep", "Read"], "system": ["Bash"]},
  "tool_count": 15
}
```

### `GET /logs/stream` *(v0.6.0)*
Server-Sent Events log stream. Connect with `EventSource('/logs/stream')` or `curl -N`.

Each event: `data: {"ts": 1741305123.4, "event": "store", "level": "INFO", "color": "#4ade80", "name": "mem", "msg": "..."}`

### `GET /health`
```json
{ "status": "ok", "redis": "ok", "version": "0.7.0" }
```

### `GET /stats`
```json
{ "episodes": 57, "facts": 17, "procedures": 8, "tools": 15, "persona_fields": 4, "env_fields": 8 }
```

---

## Benchmark

Tested with 61 episodes + 23 facts on Apple M4 Pro / Redis 8 / MiniLM-L12 on MPS:

| Metric | Result |
|--------|--------|
| Recall P50 (warm) | **1.7ms** |
| Recall P90 (warm) | **2.0ms** |
| Recall cold | **46ms** |
| Embedding cache speedup | **20x** |
| Store (queue) | instant (async) |
| Consolidation (no merges) | **3ms** |
| Consolidation (with merges) | **~1.2s** |

---

## Research Basis

| Paper | Contribution |
|-------|-------------|
| [arXiv:2602.19320](https://arxiv.org/abs/2602.19320) "Anatomy of Agentic Memory" | 4-structure taxonomy (episodic/semantic/procedural/working); identifies procedural as the missing tier in most agent systems |
| [SimpleMem](https://github.com/aiming-lab/SimpleMem) ([arXiv:2601.02553](https://arxiv.org/abs/2601.02553)) | 3-stage pipeline: semantic compression → online synthesis → intent-aware retrieval. LoCoMo SOTA 43.24 F1, ~550 tokens/query (30× fewer than full-context), +64% over Claude-Mem on cross-session benchmark. We adapted: entropy gate H(W_t), temporal affinity cos×e^(-λ·days), keyword lexical boost, time-range symbolic filter |
| [MemGPT](https://arxiv.org/abs/2310.08560) | Virtual context paging concept |
| [MemoryOS](https://arxiv.org/abs/2506.06326) | Heat-tiered promotion (EMNLP 2025 Oral) |
| [Mem0](https://arxiv.org/abs/2504.19413) | Hybrid recall architecture |
| [MIRIX](https://arxiv.org/abs/2507.07957) | Multi-type memory taxonomy |
| [Redis vectorset](https://redis.io/blog/searching-1-billion-vectors-with-redis-8/) | HNSW at scale, native Redis 8 |

---

## License

MIT
