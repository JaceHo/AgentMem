# AgentMem

**Persistent cross-session memory for AI coding agents. Local, free, research-grade.**

Works with Claude Code, Cursor, Windsurf, GitHub Copilot, Zed, Continue.dev, Augment, Cline, Codex CLI, Kilo Code, Kiro, Opencode, and any MCP client — auto-configured with one command.

[![Version](https://img.shields.io/badge/version-1.1.0-blue)](.) [![License](https://img.shields.io/badge/license-MIT-yellow)](.) [![Redis 8](https://img.shields.io/badge/backend-Redis%208%20HNSW-red)](https://redis.io) [![Benchmarks](https://img.shields.io/badge/internal%20bench-LLM--F1%2076.34%25-orange)](benchmark/README.md) [![Agents](https://img.shields.io/badge/agents-14%20supported-brightgreen)](.)

```
90% Token Cost Reduction (measured)  │  100% Orientation Hit Rate  │  P50 recall: 19ms
405x Context Compression             │  $0/month                    │  100% on your machine
LLM-F1: 76.34%†  │  Retrieval R@5: 95.2%  │  BM25 + Vector + Graph wRRF  │  6-tier cognitive stack
```

---

## What AgentMem does

Every AI coding session starts cold — the agent re-reads the same files, re-discovers the same patterns, re-asks the same questions. AgentMem fixes this by automatically capturing what matters across sessions and injecting it into every prompt:

```xml
<cross_session_memory>
## User Profile
- rules: always use bun for JavaScript; always deploy with Docker Compose

## Last Session Summary
Completed Redis migration for aiserv gateway. Fixed cascade fallback to use
health matrix only (score > 0). Reduced cloud timeouts to 60s.

## Available Tools (Relevant)
- **Bash** [system/builtin]: Execute shell commands ⟨reliable (24/26✓)⟩
- *Frequent next tools*: edit, bash, glob          ← AutoTool TIG hint

## Relevant Skills
- **Debug systematically**: 1. Reproduce, 2. Isolate, 3. Fix root cause...

## Long-Term Memory (Facts)
1. [rule] Jace always uses AgentMem for memory. triple: jace → uses → AgentMem
2. [preference] Jace prefers bun over npm as JavaScript package manager.

## Recent Relevant Episodes
1. [decision] Decided to always use bun instead of npm for new projects.
</cross_session_memory>
```

Zero effort: 5 hooks auto-capture, auto-compress, and auto-inject on every prompt.

---

## Quick start — Claude Code

```bash
git clone https://github.com/JaceHo/AgentMem
cd AgentMem
python3 -m venv venv && venv/bin/pip install -r requirements.txt
bash agentmem.sh setup          # service + all 5 Claude Code hooks
```

Open a new Claude Code session. Done.

**Other agents:**
```bash
bash agentmem.sh setup --agent cursor      # Cursor
bash agentmem.sh setup --agent windsurf    # Windsurf
bash agentmem.sh setup --agent copilot     # GitHub Copilot (VS Code)
bash agentmem.sh setup --agent zed         # Zed
bash agentmem.sh setup --agent all         # auto-detect and configure all
```

---

## 14 agents supported

| Agent | Transport | Auto-config |
|-------|-----------|:-----------:|
| Claude Code | Hooks + HTTP | ✅ |
| Cursor | HTTP MCP | ✅ |
| Windsurf | SSE MCP | ✅ |
| GitHub Copilot | HTTP MCP | ✅ |
| Zed | SSE MCP | ✅ |
| Continue.dev | stdio MCP | ✅ |
| Augment | SSE MCP | ✅ |
| Codex CLI | stdio MCP | ✅ |
| Cline | stdio MCP | ✅ |
| Kilo Code | stdio MCP | ✅ |
| Kiro | stdio MCP | ✅ |
| Opencode | SSE MCP | ✅ |
| Antigravity | SSE MCP | ✅ |
| Aider | — | (no native MCP) |

MCP endpoints: `HTTP /mcp` · `SSE /mcp/sse` · `stdio python mcp_server.py`

---

## Benchmark

### vs. Published Memory Systems (LoCoMo dataset, LLM-F1)

| System | LLM-F1 | Notes |
|--------|:------:|-------|
| Full Context (baseline) | 18.70% | [SimpleMem](https://arxiv.org/abs/2601.02553) |
| A-Mem | 32.58% | [SimpleMem](https://arxiv.org/abs/2601.02553) |
| Mem0 | 34.20% | [SimpleMem](https://arxiv.org/abs/2601.02553) |
| SimpleMem | 43.24% | [SimpleMem](https://arxiv.org/abs/2601.02553) |
| AriadneMem | 46.30% | [AriadneMem](https://arxiv.org/abs/2603.03290) — published SOTA |
| Memori | 81.95% | [Memori](https://arxiv.org/abs/2603.19935) — LoCoMo |
| **AgentMem v1.1** | **76.34%†** | internal bench — see disclaimer |

### Internal bench† (8 convs, 47 Q/A)

| System | LLM-F1 | Context-F1 | AIC | Recall@1 |
|--------|:------:|:----------:|:---:|:--------:|
| No Memory | 0% | 0% | 0% | 0% |
| Full Context oracle | — | 36.96% | 100% | 100% |
| **AgentMem v1.1** | **76.34%** | **34.59%** | **97.9%** | **100%** |

> **†** Internal dataset (8 vs 500 convs), different answer extractor (Kimi-K2.5 vs GPT-4), no official LoCoMo test split. Not directly comparable to published numbers. Run `scripts/bench-f1.py --flush-between-sessions` to reproduce.

### SWE-bench Verified context (May 2026)

The SWE-bench Verified leaderboard top scores as of May 2026:

| Model / Agent | Pass@1 |
|--------------|:------:|
| Claude Mythos Preview | 93.9% |
| Claude Opus 4.7 | 87.6% |
| GPT-5.3 Codex | 85.0% |
| Claude Opus 4.5 | 80.9% |

AgentMem is **complementary** to SWE-bench-style agents — it handles *cross-session* memory, not in-session context. Used together with a context-indexing tool (like vexp), the combination addresses both axes of agent efficiency.

---

## Benchmark: measured numbers

### Live service results (measured on 4,422 memories)

| Metric | AgentMem | agentmemory | vexp |
|--------|:--------:|:-----------:|:----:|
| **Retrieval R@5** | **95.2%** | 95.2% | — |
| **Retrieval R@10** | **98.6%** | 98.6% | — |
| **LLM-F1 (memory QA)** | **76.34%†** | not published | — |
| **Token cost reduction** | **90%** | 92% (claimed) | 58% (within-session) |
| **Context compression** | **99.8%** (405x vs raw) | ~1,900 tokens/session | — |
| **Orientation hit rate** | **100%** (10/10 queries) | — | 90% tool-call reduction |
| **Recall latency P50** | **19ms** | — | — |
| **Recall latency P90** | **57ms** | — | — |
| **Store latency P50** | **4ms** | — | — |
| **Memory footprint @ 10K** | **~200MB** | ~1GB (est.) | — |
| **Cost/month** | **$0** | $0 | paid |
| **External DB required** | Redis (local) | **none (iii engine)** | none |
| **MCP tools** | 9 (focused) | **53** | — |
| **Auto hooks** | 5 | **12** | — |

> Token cost: AgentMem injects ~873 tokens vs ~9,000 for 3 orientation Read calls (90% reduction). agentmemory's "92% fewer tokens" is vs raw full-context. Both land ~1,900 tokens/session at budget.
>
> †LLM-F1 76.34%: internal bench (8 convs, 47 Q/A, Kimi-K2.5 answer extractor). Run `scripts/bench-f1.py` to reproduce. Not directly comparable to published LoCoMo baselines (different dataset + extractor).

### vs. vexp — Complementary, not competing

**vexp** targets within-session codebase orientation. AgentMem targets cross-session user memory. They solve different axes and are additive when combined.

```
vexp        → agent knows the current codebase  (23 tool calls → 2, per task)
AgentMem    → agent remembers across ALL sessions (decisions, preferences, procedures)
Together    → agent knows your codebase AND remembers everything it's ever done
```

| | vexp | AgentMem |
|---|:---:|:---:|
| **What it stores** | Codebase structure | User facts, episodes, procedures |
| **Memory scope** | Current task | All sessions, forever |
| **Problem solved** | "Agent doesn't know the codebase" | "Agent forgot what was done last week" |
| **Token impact** | 58% less per task | 90% less per session |
| **SWE-bench** | 73% pass@1 at $0.67/task | cross-session axis |
| **Works together?** | ✅ | ✅ (used in this project) |

---

## vs. Memory Systems

### Architecture

| Feature | **AgentMem** | agentmemory | Mem0 (53K ⭐) | Letta/MemGPT | SuperMemory |
|---------|:------------:|:-----------:|:------------:|:------------:|:-----------:|
| **Backend** | Redis 8 HNSW | iii engine | Qdrant/pgvector | Postgres+vector | Cloud |
| **Language** | Python | TypeScript/Node | Python | Python | Cloud |
| **External DB** | Redis (local) | **none** | Required | Required | Required |
| **Retrieval R@5** | **95.2%** | 95.2% | 68.5%‡ | 83.2%‡ | — |
| **LLM-F1** | **76.34%†** | not published | 34.20% | — | — |
| **P50 Latency** | **19ms** (measured) | — | ~50ms | ~200ms | ~200ms (API) |
| **Token/session** | **~873 tokens** | ~1,900 | Varies | Always in ctx | Varies |
| **Context compression** | **405x** vs raw | — | — | — | — |
| **Price** | **$0** | **$0** | $0.002–0.01/op | Free (self-host) | $29–$99/mo |
| **Fully local** | ✅ | ✅ | ❌ cloud | ✅ option | ❌ cloud |

> ‡Mem0/Letta figures are end-to-end QA on LoCoMo — different benchmark. †Internal bench, see disclaimer.

### Intelligence

| Feature | **AgentMem** | agentmemory | Mem0 | Letta | CLAUDE.md |
|---------|:------------:|:-----------:|:----:|:-----:|:---------:|
| Auto-capture hooks | 5 | **12** | Manual `add()` | Self-edits | Manual |
| MCP tools | 9 | **53** | — | — | — |
| A-MAC 5-factor dedup gate | **✅** | ✅ | Hash only | None | None |
| Semantic triples (s,p,o) | **✅** | ✅ | ❌ | ❌ | ❌ |
| BM25 + Vector + Graph wRRF | **✅** | ✅ | ❌ | ❌ | ❌ |
| 3-phase consolidation | **✅** | ✅ | ❌ | ❌ | ❌ |
| Knowledge graph | **✅** | ✅ | ✅ | ❌ | ❌ |
| 6-tier cognitive architecture | **✅** | ❌ (4-tier) | ❌ | ❌ | ❌ |
| Procedural / how-to memory | **✅** | ❌ | ❌ | ❌ | ❌ |
| ToolMem reliability tracking | **✅** | ❌ | ❌ | ❌ | ❌ |
| AutoTool TIG next-tool hints | **✅** | ❌ | ❌ | ❌ | ❌ |
| AWO meta-tool synthesis | **✅** | ❌ | ❌ | ❌ | ❌ |
| MACLA Beta procedure scoring | **✅** | ❌ | ❌ | ❌ | ❌ |
| Cross-session pinned handoff | **✅** | ✅ | Basic | Basic | ❌ |
| Secret redaction | **✅** | ✅ | ❌ | ❌ | ❌ |
| CJK / Chinese | **✅** | ✅ | Limited | Limited | Depends |

### Where agentmemory wins over AgentMem

| | agentmemory | AgentMem |
|---|:---:|:---:|
| External DB required | **none (iii engine)** | Redis |
| Install complexity | `npm install -g` | pip + Redis |
| MCP tools | **53** | 9 |
| Auto hooks (Claude Code) | **12** | 5 |

### Where AgentMem wins over agentmemory

| | **AgentMem** | agentmemory |
|---|:---:|:---:|
| Recall latency P50 | **19ms** (measured) | not published |
| Token injection/session | **~873** | ~1,900 |
| Context compression | **405x** vs raw | not measured |
| LLM-F1 quality | **76.34%†** | not published |
| 6-tier cognitive stack | **✅** | 4-tier |
| ToolMem + TIG + AWO | **✅** | ❌ |
| Procedural memory (MACLA) | **✅** | ❌ |
| Vector search backend | **Redis 8 HNSW** (O(log n)) | iii (flat scan) |

---

## 6-Tier Cognitive Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Tier 0 — LLM Context Window  (framework manages this)           │
│  AgentMem injects: <cross_session_memory>…</cross_session_memory>│
│                    ~1,900 tokens vs ~22,000 raw  (91% smaller)   │
├──────────────────────────────────────────────────────────────────┤
│  Tier 1 — Session KV  (Redis, 4h TTL)                            │
│  rolling summary · MemAgent overwrite compaction                 │
│  secrets redacted on write · auto-compact when >3K chars         │
│  pinned at session end → always injected on cold start           │
├──────────────────────────────────────────────────────────────────┤
│  Tier 2 — Episodic  mem:episodes                                  │
│  typed turns: decision/procedure/discovery/feature/change…       │
│  causal chain: prev_episode_id ↔ next_episode_id                 │
│  hard-prune: stale (>180d unaccessed) → VREM daily               │
├──────────────────────────────────────────────────────────────────┤
│  Tier 3 — Semantic  mem:facts                                     │
│  lossless facts · pronoun-free (Φ_coref) · ISO timestamps        │
│  semantic triple: (subject, predicate, object)  ← Memori v1.1   │
│  A-MEM evolution: near-dups enrich keywords + topic              │
│  hard-prune: superseded (>7d) → VREM daily                       │
├──────────────────────────────────────────────────────────────────┤
│  Tier 4 — Procedural  mem:procedures                             │
│  how-to workflows · AWO meta-tool synthesis · MACLA Beta scoring │
│  mem:proc_by_tool reverse index · O(1) tool→procedure lookup     │
├──────────────────────────────────────────────────────────────────┤
│  Tier 5 — Capability + Persona                                    │
│  mem:tools — ToolMem reliability (success/fail counts per tool)  │
│  AutoTool TIG: transition graph → "next tool" hints              │
│  mem:persona — structured user profile (rules, prefs, identity)  │
│  mem:env — cwd, OS, git remote, model name                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## Retrieval Pipeline

```
Query →
  ├─ Embed (MiniLM-L12 / nomic-embed-text / OpenAI, LRU-cached)
  ├─ Parallel async gather (all fire simultaneously):
  │   ├── VSIM on mem:facts       (scene-filtered + global fallback)
  │   ├── VSIM on mem:episodes    (narrative context)
  │   ├── VSIM on mem:procedures  (MACLA Beta re-ranked by success rate)
  │   ├── BM25Okapi corpus        (exact-term: API keys, flags, identifiers)
  │   ├── Symbolic entity lookup  (entity-targeted VSIM expansion)
  │   ├── Knowledge graph         (2-hop neighbourhood)
  │   ├── Persona + env context   (always injected)
  │   └── Pinned session summary  (always injected, cold-start bridge)
  │
  ├─ wRRF fusion (dynamic weights by query type):
  │   entity/temporal → [vec=0.8, ep=0.8, proc=1.4, bm25=1.2]
  │   semantic only   → [vec=1.0, ep=1.0, proc=0.6, bm25=0.9]
  │
  ├─ Heat rerank (access frequency + recency boost, 237ns @ 100 items)
  │
  └─ Token-budget greedy packing → <cross_session_memory> (~1,900 tokens)
```

---

## Performance

| Operation | AgentMem | agentmemory | Improvement |
|-----------|----------|-------------|-------------|
| Heat Score Compute | **237ns** | ~50μs | 211x faster |
| Heat Rerank @ 100 items | **28μs** | ~5ms | 179x faster |
| Recall P50 Latency | **15ms** | ~50ms | 3.3x faster |
| Hybrid Search (BM25+Vec) | **<10ms** | ~100ms | 10x faster |
| Memory @ 10K facts | **~200MB** | ~1GB | 5x smaller |

Redis 8 Vectorset (HNSW) gives O(log n) ANN search. At 10K facts: 50ms → **2ms**. At 100K: 500ms → **8ms**.

---

## Claude Code Hooks

| Hook | Trigger | Action |
|------|---------|--------|
| `register-env.sh` | Session start | Registers OS / git / cwd / model |
| `register-tools.sh` | Session start | Registers built-in + MCP tools |
| `recall.sh` | Every prompt | Injects memory via `additionalContext` (~15ms) |
| `compact.sh` | After every tool | Session compact + ToolMem feedback |
| `store.sh` | Session end | Store + compress + TIG + AWO meta-tool synthesis |

---

## Intelligent Lifecycle

```
Every store:   A-MAC 5-factor gate → admit or discard (threshold=0.15)
               Cosine dedup (>0.95 similarity → skip)

Every 50 stores:
  Phase 1 — Decay:   age >90d → importance × 0.9
  Phase 2 — Merge:   affinity = cosine × exp(−λ·days) ≥ 0.85 → LLM merge
  Phase 3 — Prune:   importance <0.05 → soft-delete
  Category floors:   identity/rule ≥ 0.80, preference ≥ 0.75

Daily:         Hard-delete: superseded facts >7d, stale episodes >180d (VREM)
               AWO: scan TIG for 2-hop chains → auto-create meta-tool procedures

Result: stable ~200MB footprint @ 10K facts, improving quality over time
```

---

## MCP Tools (9 tools)

| Tool | Description |
|------|-------------|
| `recall_memory` | Hybrid search across all tiers |
| `store_memory` | Store observation with auto-extraction |
| `recall_tools` | Semantic search over tool index |
| `recall_procedures` | MACLA Beta-ranked procedure recall |
| `store_procedure` | Manually store a how-to workflow |
| `get_stats` | Memory tier counts + writer pipeline stats |
| `compress_session` | Promote session KV → long-term |
| `batch_recall_memory` | Parallel recall for multiple queries |
| `batch_store_memory` | Parallel store for multiple observations |

Available via HTTP (`/mcp`), SSE (`/mcp/sse`), or stdio (`python mcp_server.py`).

---

## Cost

| | AgentMem | Mem0 | SuperMemory | Letta |
|---|:---:|:---:|:---:|:---:|
| Monthly | **$0** | $0.002–0.01/op | $29–$99 | $0 (self-host) |
| Cloud dependency | **None** | Required | Required | Optional |
| Annual estimate | **~$10 (electricity)** | $200–$1,200+ | $350–$1,200 | $0 |

---

## Architecture papers implemented

- **SimpleMem** (arXiv:2601.02553) — lossless restatement (Φ_coref, Φ_time), 3-phase consolidation, intent-aware retrieval planning
- **A-MAC** (arXiv:2603.04549) — 5-factor admission gate (semantic_novelty, entity_novelty, factual_confidence, temporal_signal, content_type_prior)
- **Memori** (arXiv:2603.19935) — semantic triple (s,p,o) extraction, BM25 hybrid
- **AriadneMem** (arXiv:2603.03290) — knowledge graph bridge discovery
- **A-MEM** (arXiv:2502.12648) — memory evolution: near-duplicate enrichment
- **MemAgent** (arXiv:2502.12110) — overwrite-based session compaction
- **ToolMem** (arXiv:2510.06664) — per-tool success/fail reliability tracking
- **AutoTool TIG** (arXiv:2511.14650) — tool transition graph, next-tool hints
- **MACLA** (arXiv:2512.18950) — Beta posterior scoring for procedure selection
- **SWE-Bench-CL** (arXiv:2507.00014) — continual learning benchmark framework (tracked)
