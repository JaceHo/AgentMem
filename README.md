# openclaw-memory

**Local long-term memory service for [OpenClaw](https://openclaw.ai) agents.**

Drop-in replacement for the `memos-cloud-openclaw-plugin` (MemTensor cloud API).
Fully local, persistent, fast — no external dependencies.

## Features

| Feature | Description |
|---------|-------------|
| **Atomized fact extraction** | LLM produces self-contained facts with coreference resolution (no pronouns) and temporal anchoring (absolute timestamps) |
| **Hybrid extraction** | Dual-layer: regex (~1ms) + LLM/GLM-4-flash (~200-500ms) for Chinese/implicit/contextual facts |
| **Structured metadata** | Facts carry `keywords[]`, `persons[]`, `entities[]` for rich recall context |
| **Adaptive retrieval depth** | Query complexity estimator dynamically adjusts k (simple queries get fewer results, complex ones get more) |
| **Memory consolidation** | Merges semantically similar facts (cosine > 0.85) via LLM to reduce redundancy |
| **Semantic recall** | Redis 8 native vectorset HNSW + `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) |
| **Heat-tiered ranking** | Frequently/recently accessed memories rank higher |
| **Scene isolation** | Language + domain tags filter recall to relevant context |
| **Evolving persona** | Structured user profile updated incrementally from conversations |
| **Summarize-then-embed** | Long turns compressed via GLM-4-flash before embedding |
| **Deduplication** | Near-identical memories (cosine > 0.95) skipped on store |
| **Fully local** | No cloud dependency; Redis AOF for crash-safe persistence |

## Architecture

```
OpenClaw Gateway (Node.js)
    │  lifecycle hooks
    ▼
memos-local plugin (JS)   ← ~/.openclaw/extensions/memos-local/
    │  HTTP POST
    ▼
Memory Service (Python FastAPI) @ localhost:18800
    ├── /recall        → embed query → adaptive k → KNN (scene-filtered) → heat rerank → prependContext
    ├── /store         → filter → detect scene → summarize → atomized extract → embed → dedup → save
    └── /consolidate   → find similar facts → LLM merge → deduplicate
         │
         ▼
    Redis 8 (localhost:6379)
         ├── mem:episodes  (HNSW vectorset — episodic memory)
         ├── mem:facts     (HNSW vectorset — semantic facts + metadata)
         ├── mem:persona   (Hash — evolving user profile)
         └── mem:session:* (String+TTL — working memory)
```

## Memory Tiers

| Tier | Store | Scope | TTL |
|------|-------|-------|-----|
| Working | Redis String | Current session | 4h |
| Episodic | Redis vectorset HNSW | All sessions | Permanent |
| Semantic | Redis vectorset HNSW | Distilled facts | Permanent |
| Persona | Redis Hash | User profile | Permanent |

## What's New in v0.4.0 (SimpleMem-Inspired)

### Atomized Fact Extraction

Inspired by [SimpleMem](https://github.com/aiming-lab/SimpleMem) (SOTA on LoCoMo benchmark, 43.24 F1), facts are now extracted as **atomic entries** — self-contained sentences that can be understood without surrounding context:

| Before (v0.3.0) | After (v0.4.0) |
|------------------|-----------------|
| `I like bun` | `用户偏好使用 bun 作为 JavaScript 包管理器，明确拒绝 npm` |
| `my name is Jace` | `用户名为 Jace` |
| `always use GLM-4-flash` | `用户设定规则：轻量级 LLM 任务永远优先使用 GLM-4-flash` |

Key techniques from SimpleMem applied:
- **Coreference resolution (Φ_coref)**: No pronouns — "he" → "Bob", "I" → "用户"
- **Temporal anchoring (Φ_time)**: "yesterday" → "2026-02-28"
- **Structured metadata**: Each fact stores `keywords[]`, `persons[]`, `entities[]`

### Adaptive Retrieval Depth

Instead of fixed k, retrieval depth adapts to query complexity:

```
k_dyn = k_base × (1 + δ × C_q)
```

Where `C_q` is estimated from query features (word count, signal words like "what/when/compare/所有") at zero cost — no LLM call needed. Simple queries like "my name" get fewer results; complex queries like "compare all the APIs I've used for image editing" get deeper retrieval.

### Memory Consolidation

New endpoint merges semantically similar facts to prevent recall slot waste:

```bash
# Async (fire-and-forget)
curl -X POST http://localhost:18800/consolidate

# Sync (returns results)
curl -X POST http://localhost:18800/consolidate/sync
# → {"merged": 1, "removed": 1, "total_before": 18, "total_after": 17, "ms": 1267}
```

Uses LLM (GLM-4-flash) to produce a merged restatement preserving all distinct details. Collects keywords/persons/entities from all cluster members.

## Heat Scoring

Each recalled memory bumps its `access_count`. Re-ranking formula:

```
heat   = exp(-days_old / 30) × (1 + log(1 + access_count) × 0.25)
score  = cosine_similarity × heat
```

New memories start at `heat ≈ 1.0`. Old, never-accessed memories decay to `~0.05`.

## Scene Isolation

On every recall, the query language and domain are detected (regex + keyword matching).
A Redis VSIM FILTER expression narrows results to the same scene first.
Global search supplements if scene results are sparse.

Detected domains: `trading`, `coding`, `devops`, `ai`, `finance`, `general`

## Hybrid Fact Extraction

Two-layer extraction runs on every `/store` call (async, non-blocking):

| Layer | Method | Latency | Catches |
|-------|--------|---------|---------|
| **Regex** | 9 compiled patterns | ~1ms | English first-person: "My name is...", "I prefer..." |
| **LLM** | GLM-4-flash (ZAI) | ~200-500ms | Chinese, implicit, contextual, decisions, third-person |

Categories: `identity`, `work`, `preference`, `location`, `personal`, `reminder`, `rule`, `decision`, `context`, `credential`

LLM extraction skips `credential` for safety. Results are merged and deduped (substring match) before embedding and storing.

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
# Optional: ZAI API key for LLM summarization (GLM-4-flash)
ZAI_API_KEY=your-key-here

# Optional: MemOS API key (if keeping memos-cloud as fallback)
MEMOS_API_KEY=your-key-here
```

Both summarization and LLM extraction degrade gracefully without a key (summarizer truncates to 180 chars; extractor falls back to regex-only).

### Running

```bash
# Development
venv/bin/uvicorn main:app --host 127.0.0.1 --port 18800 --reload

# Production (LaunchAgent on macOS)
launchctl load ~/Library/LaunchAgents/ai.openclaw.memory.plist
```

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

## API

### `POST /recall`
```json
{ "query": "user's current message", "session_id": "abc", "memory_limit_number": 6 }
```
Returns `{ "prependContext": "...", "latency_ms": 45 }`

Retrieval depth adapts to query complexity. Returned facts include metadata annotations (persons, entities) when available.

### `POST /store`
```json
{ "messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}], "session_id": "abc" }
```
Returns `{ "status": "queued" }` (async background processing)

Facts are extracted as atomized entries with `keywords[]`, `persons[]`, `entities[]`.

### `POST /consolidate`
Trigger async memory consolidation (fire-and-forget).
Returns `{ "status": "consolidation_queued" }`

### `POST /consolidate/sync`
Synchronous consolidation — waits and returns results.
Returns `{ "merged": N, "removed": N, "total_before": N, "total_after": N, "ms": N }`

### `GET /health`
```json
{ "status": "ok", "redis": "ok", "version": "0.4.0" }
```

### `GET /stats`
```json
{ "episodes": 57, "facts": 17, "persona_fields": 4 }
```

## Latency Targets

| Operation | P50 | P95 |
|-----------|-----|-----|
| Recall (cache hit) | ~8ms | ~15ms |
| Recall (cold) | ~70ms | ~100ms |
| Store (async, regex-only) | non-blocking | non-blocking |
| Store (async, hybrid) | non-blocking (~1.2s background) | non-blocking |
| Consolidate (sync) | ~1-3s | ~5s |

## Research Basis

| Paper | Contribution |
|-------|-------------|
| [SimpleMem](https://github.com/aiming-lab/SimpleMem) | Atomized extraction, adaptive retrieval depth, consolidation (LoCoMo SOTA) |
| [MemGPT](https://arxiv.org/abs/2310.08560) | Virtual context paging concept |
| [MemoryOS](https://arxiv.org/abs/2506.06326) | Heat-tiered promotion (EMNLP 2025 Oral) |
| [Mem0](https://arxiv.org/abs/2504.19413) | Hybrid recall architecture |
| [MIRIX](https://arxiv.org/abs/2507.07957) | Multi-type memory taxonomy |
| [Redis vectorset](https://redis.io/blog/searching-1-billion-vectors-with-redis-8/) | HNSW at scale |

## License

MIT
