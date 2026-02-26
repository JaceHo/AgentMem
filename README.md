# openclaw-memory

**Local long-term memory service for [OpenClaw](https://openclaw.ai) agents.**

Drop-in replacement for the `memos-cloud-openclaw-plugin` (MemTensor cloud API).
Fully local, persistent, fast — no external dependencies.

## Features

| Feature | Description |
|---------|-------------|
| **Hybrid fact extraction** | Dual-layer: regex (~1ms) + LLM/GLM-4-flash (~200-500ms) for Chinese/implicit/contextual facts |
| **Semantic recall** | Redis 8 native vectorset HNSW + `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) |
| **Heat-tiered ranking** | Frequently/recently accessed memories rank higher |
| **Scene isolation** | Language + domain tags filter recall to relevant context |
| **Evolving persona** | Structured user profile updated incrementally from conversations |
| **Summarize-then-embed** | Long turns compressed via GLM-4-flash before embedding |
| **Deduplication** | Near-identical memories (cosine dist < 0.05) skipped on store |
| **Fully local** | No cloud dependency; Redis AOF for crash-safe persistence |

## Architecture

```
OpenClaw Gateway (Node.js)
    │  lifecycle hooks
    ▼
memos-local plugin (JS)   ← ~/.openclaw/extensions/memos-local/
    │  HTTP POST
    ▼
Memory Service (Python FastAPI) @ localhost:18790
    ├── /recall  → embed query → KNN (scene-filtered) → heat rerank → prependContext
    └── /store   → filter → detect scene → summarize → hybrid extract → embed → dedup → save
         │
         ▼
    Redis 8 (localhost:6379)
         ├── mem:episodes  (HNSW vectorset — episodic memory)
         ├── mem:facts     (HNSW vectorset — semantic facts)
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

## Hybrid Fact Extraction (v0.3.0)

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
venv/bin/uvicorn main:app --host 127.0.0.1 --port 18790 --reload

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
        "config": { "baseUrl": "http://127.0.0.1:18790", "memoryLimitNumber": 6 }
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

### `POST /store`
```json
{ "messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}], "session_id": "abc" }
```
Returns `{ "status": "queued" }` (async background processing)

### `GET /health`
```json
{ "status": "ok", "redis": "ok", "version": "0.3.0" }
```

### `GET /stats`
```json
{ "episodes": 42, "facts": 18, "persona_fields": 4 }
```

## Latency Targets

| Operation | P50 | P95 |
|-----------|-----|-----|
| Recall (cache hit) | ~8ms | ~15ms |
| Recall (cold) | ~70ms | ~100ms |
| Store (async, regex-only) | non-blocking | non-blocking |
| Store (async, hybrid) | non-blocking (~1.2s background) | non-blocking |

## Research Basis

| Paper | Contribution |
|-------|-------------|
| [MemGPT](https://arxiv.org/abs/2310.08560) | Virtual context paging concept |
| [MemoryOS](https://arxiv.org/abs/2506.06326) | Heat-tiered promotion (EMNLP 2025 Oral) |
| [Mem0](https://arxiv.org/abs/2504.19413) | Hybrid recall architecture |
| [MIRIX](https://arxiv.org/abs/2507.07957) | Multi-type memory taxonomy |
| [Redis vectorset](https://redis.io/blog/searching-1-billion-vectors-with-redis-8/) | HNSW at scale |

## License

MIT
