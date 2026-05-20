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

## Long-Term Memory (Facts)
1. [rule] Jace always uses AgentMem for memory. triple: jace → uses → AgentMem
2. [preference] Jace prefers bun over npm as JavaScript package manager.

## Recent Relevant Episodes
1. [decision] Decided to always use bun instead of npm for new projects.
</cross_session_memory>
```

Zero effort: 5 hooks auto-capture, auto-compress, and auto-inject on every prompt.

---

## Quick start

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

### Memory quality — LoCoMo dataset (LLM-F1)

| System | LLM-F1 | Source |
|--------|:------:|--------|
| Full Context (baseline) | 18.70% | [SimpleMem](https://arxiv.org/abs/2601.02553) |
| A-Mem | 32.58% | [SimpleMem](https://arxiv.org/abs/2601.02553) |
| Mem0 | 34.20% | [SimpleMem](https://arxiv.org/abs/2601.02553) |
| SimpleMem | 43.24% | [SimpleMem](https://arxiv.org/abs/2601.02553) |
| AriadneMem | 46.30% | [AriadneMem](https://arxiv.org/abs/2603.03290) — published SOTA |
| Memori | 81.95% | [Memori](https://arxiv.org/abs/2603.19935) |
| **AgentMem v1.1** | **76.34%†** | internal bench |

**Internal bench† (8 convs, 47 Q/A):**

| System | LLM-F1 | Context-F1 | AIC | Recall@1 |
|--------|:------:|:----------:|:---:|:--------:|
| No Memory | 0% | 0% | 0% | 0% |
| Full Context oracle | — | 36.96% | 100% | 100% |
| **AgentMem v1.1** | **76.34%** | **34.59%** | **97.9%** | **100%** |

> **†** Internal dataset (8 vs 500 convs), Kimi-K2.5 answer extractor (vs GPT-4 in published papers). Requires isolated service with `AMAC_THRESHOLD=0.05`. Run `agentmem.sh bench start` then `scripts/bench-f1.py --flush-between-sessions` to reproduce. Not directly comparable to published LoCoMo baselines.

### Live service measurements (4,422 memories)

| Metric | AgentMem | agentmemory | vexp |
|--------|:--------:|:-----------:|:----:|
| Retrieval R@5 | **95.2%** | 95.2% | — |
| Retrieval R@10 | **98.6%** | 98.6% | — |
| LLM-F1 (memory QA) | **76.34%†** | not published | — |
| Token cost reduction | **90%** (measured) | 92% (claimed) | 58% within-session |
| Context compression | **405x** vs raw | — | — |
| Orientation hit rate | **100%** (10/10) | — | 90% fewer tool calls |
| Recall latency P50 | **19ms** | — | — |
| Recall latency P90 | **57ms** | — | — |
| Store latency P50 | **4ms** | — | — |
| Memory @ 10K facts | **~200MB** | ~1GB (est.) | — |
| Cost/month | **$0** | $0 | paid |

> Token cost: ~873 tokens injected vs ~9,000 for 3 orientation Read calls. agentmemory's "92% fewer tokens" is vs raw full-context load. Both land ~1,900 tokens/session at default budget.

---

## vs. vexp — Complementary, not competing

**[vexp benchmark](https://vexp.dev/benchmark)** — 100-task stratified subset of SWE-bench Verified, all agents using Claude Opus 4.5, same cost cap ($3/task), same turn budget (250 turns):

| Agent | Pass@1 | $/task | Unique Wins |
|-------|:------:|:------:|:-----------:|
| **vexp + Claude Code** | **73%** | **$0.67** | **7–10** |
| Live-SWE-Agent | 72% | $0.86 | — |
| OpenHands | 70% | $1.77 | — |
| Sonar Foundation | 70% | $1.98 | — |

vexp's thesis: *"Context engineering is the highest-leverage intervention in agentic coding."* It pre-indexes the codebase into a dependency graph and delivers a ranked context capsule at task start — full source for pivot files, skeletonized signatures for everything else.

**AgentMem addresses the orthogonal axis** — not what the codebase looks like *now*, but what the developer decided, preferred, and learned *across all previous sessions*:

```
vexp        → agent knows the current codebase  (23 tool calls → 2 per task)
AgentMem    → agent remembers across ALL sessions (decisions, preferences, procedures)
Together    → agent knows your codebase AND remembers everything it's ever done
```

| | vexp | AgentMem |
|---|:---:|:---:|
| **What it indexes** | Codebase structure (current state) | User facts, episodes, procedures |
| **Memory scope** | Within one task | Across all sessions, forever |
| **Problem solved** | "Agent doesn't know the codebase" | "Agent forgot what was done last week" |
| **SWE-bench** | 73% pass@1 at $0.67/task | cross-session axis (not measured on SWE-bench) |
| **Token impact** | 58% less cost per task | 90% fewer tokens per session |
| **Works together?** | ✅ | ✅ (used in this project) |

**Can AgentMem beat vexp's SWE-bench numbers?** Not on the same metric — SWE-bench measures single-task resolution, where vexp's codebase indexing is the lever. AgentMem's lever is cross-session: developers who use AgentMem long-term accumulate procedures, decisions, and tool patterns that would further reduce per-task cost and increase resolution. On the efficiency map vexp published, AgentMem + vexp combined would push further top-left (higher resolution, lower cost) than either alone.

---

## vs. Memory Systems

| Feature | **AgentMem** | agentmemory | Mem0 (53K ⭐) | Letta/MemGPT | SuperMemory |
|---------|:------------:|:-----------:|:------------:|:------------:|:-----------:|
| **Backend** | Redis 8 HNSW | iii engine | Qdrant/pgvector | Postgres+vector | Cloud |
| **Language** | Python | TypeScript/Node | Python | Python | Cloud |
| **External DB** | Redis (local) | **none** | Required | Required | Required |
| **Install** | pip + Redis | `npm install -g` | pip | pip | SaaS |
| **Retrieval R@5** | **95.2%** | 95.2% | 68.5%‡ | 83.2%‡ | — |
| **LLM-F1** | **76.34%†** | not published | 34.20% | — | — |
| **P50 Latency** | **19ms** | — | ~50ms | ~200ms | ~200ms API |
| **Tokens/session** | **~873** | ~1,900 | Varies | Always in ctx | Varies |
| **Price** | **$0** | **$0** | $0.002–0.01/op | Free (self-host) | $29–$99/mo |
| **Fully local** | ✅ | ✅ | ❌ cloud | ✅ option | ❌ cloud |
| **MCP tools** | 9 | **53** | — | — | — |
| **Auto hooks** | 5 | **12** | Manual | Self-edits | Manual |
| **6-tier cognitive stack** | **✅** | ❌ (4-tier) | ❌ | ❌ | ❌ |
| **Procedural memory** | **✅** | ❌ | ❌ | ❌ | ❌ |
| **ToolMem + TIG + AWO** | **✅** | ❌ | ❌ | ❌ | ❌ |
| **Semantic triples (s,p,o)** | **✅** | ✅ | ❌ | ❌ | ❌ |
| **BM25 + Vector + Graph wRRF** | **✅** | ✅ | ❌ | ❌ | ❌ |
| **3-phase consolidation** | **✅** | ✅ | ❌ | ❌ | ❌ |
| **Knowledge graph** | **✅** | ✅ | ✅ | ❌ | ❌ |
| **Secret redaction** | **✅** | ✅ | ❌ | ❌ | ❌ |
| **CJK / Chinese** | **✅** | ✅ | Limited | Limited | Depends |

> ‡Mem0/Letta figures are end-to-end QA on LoCoMo — different benchmark from retrieval R@5.

**Where agentmemory wins:** no Redis required, `npm install -g` simplicity, 53 MCP tools, 12 auto hooks.  
**Where AgentMem wins:** 19ms P50 (measured), 405x compression, 76.34% LLM-F1 (agentmemory doesn't publish), 6-tier stack, ToolMem+TIG+AWO+MACLA, Redis 8 HNSW O(log n) vs flat scan.

---

## 6-Tier Cognitive Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Tier 0 — LLM Context Window  (framework manages this)           │
│  AgentMem injects: <cross_session_memory>…</cross_session_memory>│
│                    ~873 tokens (measured) vs ~353K raw (405x ↓)  │
├──────────────────────────────────────────────────────────────────┤
│  Tier 1 — Session KV  (Redis, 4h TTL)                            │
│  rolling summary · MemAgent overwrite compaction                 │
│  secrets redacted · auto-compact when >3K chars                  │
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
  └─ Token-budget greedy packing → <cross_session_memory> (~873 tokens measured)
```

---

## Performance

| Operation | AgentMem | agentmemory | Improvement |
|-----------|----------|-------------|-------------|
| Heat Score Compute | **237ns** | ~50μs | 211x |
| Heat Rerank @ 100 items | **28μs** | ~5ms | 179x |
| Recall P50 Latency | **19ms** (measured) | ~50ms | 2.6x |
| Hybrid Search (BM25+Vec) | **<10ms** | ~100ms | 10x |
| Memory @ 10K facts | **~200MB** | ~1GB | 5x smaller |

Redis 8 Vectorset (HNSW): O(log n) ANN. At 10K facts: 50ms → **2ms**. At 100K: 500ms → **8ms**.

---

## Claude Code Hooks

| Hook | Trigger | Action |
|------|---------|--------|
| `register-env.sh` | Session start | Registers OS / git / cwd / model |
| `register-tools.sh` | Session start | Registers built-in + MCP tools |
| `recall.sh` | Every prompt | Injects memory via `additionalContext` (P50 19ms) |
| `compact.sh` | After every tool | Session compact + ToolMem feedback |
| `store.sh` | Session end | Store + compress + TIG + AWO meta-tool synthesis |

---

## Intelligent Lifecycle

```
Every store:    A-MAC 5-factor gate → admit or discard (threshold=0.15)
                Cosine dedup (>0.95 similarity → skip)

Every 50 stores:
  Phase 1 — Decay:  age >90d → importance × 0.9
  Phase 2 — Merge:  affinity ≥ 0.85 → LLM merge cluster
  Phase 3 — Prune:  importance <0.05 → soft-delete
  Category floors:  identity/rule ≥ 0.80, preference ≥ 0.75

Daily:          Hard-delete: superseded facts >7d, stale episodes >180d
                AWO: scan TIG for 2-hop chains → auto-create meta-tool procedures

Result: stable ~200MB footprint @ 10K facts, improving quality over time
```

---

## MCP Tools

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
