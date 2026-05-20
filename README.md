# AgentMem

**Research-grade persistent memory for AI coding agents. Local. Free. Faster.**

Works with Claude Code, Cursor, Windsurf, GitHub Copilot, Zed, Continue.dev, Augment, Cline, Codex CLI, Kilo Code, Kiro, Opencode, and any MCP client.

[![Version](https://img.shields.io/badge/version-1.1.0-blue)](.) [![License](https://img.shields.io/badge/license-MIT-yellow)](.) [![Redis 8 HNSW](https://img.shields.io/badge/backend-Redis%208%20HNSW-red)](https://redis.io) [![LLM-F1](https://img.shields.io/badge/LLM--F1-76.34%25%E2%80%A0-orange)](benchmark/README.md) [![Agents](https://img.shields.io/badge/agents-14%20supported-brightgreen)](.)

```
LLM-F1: 76.34%†   │  Retrieval R@5: 95.2%   │  P50 recall: 19ms
~873 tokens/session │  405x context compression │  $0/month, 100% local
6-tier cognitive stack  │  9 research papers  │  7 hooks  │  HyDE + session diversity
```

---

## What AgentMem does

Every AI coding session starts cold — the agent re-reads the same files, re-discovers the same patterns, re-asks the same questions. AgentMem fixes this. Five hooks auto-capture what matters, compress it across 6 memory tiers, and inject the right context (~873 tokens) into every prompt:

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

---

## Quick start — Claude Code

```bash
git clone https://github.com/JaceHo/AgentMem
cd AgentMem
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash agentmem.sh setup          # service + all 7 Claude Code hooks
```

Open a new Claude Code session. Done.

**Other agents:**
```bash
bash agentmem.sh setup --agent cursor      # Cursor
bash agentmem.sh setup --agent windsurf    # Windsurf
bash agentmem.sh setup --agent copilot     # GitHub Copilot (VS Code)
bash agentmem.sh setup --agent zed         # Zed
bash agentmem.sh setup --agent all         # auto-detect + configure all
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

## Benchmarks

### Quality (LoCoMo dataset, LLM-F1)

| System | LLM-F1 | Notes |
|--------|:------:|-------|
| Full Context (baseline) | 18.70% | [SimpleMem](https://arxiv.org/abs/2601.02553) |
| A-Mem | 32.58% | [SimpleMem](https://arxiv.org/abs/2601.02553) |
| Mem0 | 34.20% | [SimpleMem](https://arxiv.org/abs/2601.02553) |
| SimpleMem | 43.24% | [SimpleMem](https://arxiv.org/abs/2601.02553) |
| AriadneMem | 46.30% | [AriadneMem](https://arxiv.org/abs/2603.03290) — published SOTA |
| Memori | 81.95% | [Memori](https://arxiv.org/abs/2603.19935) |
| **AgentMem v1.1** | **76.34%†** | internal bench on LLM-compressed memories — see disclaimer |
| agentmemory (rohitg00) | not published | — |

> **†** Internal dataset (8 convs, 47 Q/A), answer extractor: Kimi-K2.5. Not directly comparable to LoCoMo baselines (different dataset + extractor). Run `agentmem.sh bench start` then `scripts/bench-f1.py --flush-between-sessions` to reproduce.

> **Note on agentmemory's 95.2% R@5 claim:** their benchmark (`benchmark/longmemeval-bench.ts`) stores each session as a single raw text blob and retrieves it — it does not test retrieval through their LLM compression pipeline that real users experience. A benchmark run against compressed memories would show significantly lower numbers. AgentMem's benchmarks run against the same representations that live deployments use.

### Speed (measured on live service, 4,422 memories)

| Operation | AgentMem | agentmemory |
|-----------|:--------:|:-----------:|
| Recall P50 | **19ms** | not published |
| Recall P90 | **57ms** | not published |
| Hybrid search (BM25+Vec) | **<10ms** | ~100ms est. |
| Heat rerank @ 100 items | **28μs** | ~5ms est. |
| Store P50 | **4ms** | not published |

Redis 8 HNSW gives O(log n) ANN search — at 10K facts: 50ms → **2ms**; at 100K: 500ms → **8ms**.

### Token efficiency

| | AgentMem | agentmemory | CLAUDE.md |
|---|:---:|:---:|:---:|
| Injected tokens/session | **~873** | ~1,900 | 22K+ |
| Context compression | **405x** vs raw | — | none |
| Token cost reduction | **90%** | 92% (vs full ctx) | none |
| Annual API cost est. | **~$10** | ~$10 | impossible (>ctx window) |

> agentmemory's "92% fewer tokens" is measured vs raw full-context paste. Both land ~$10/yr — but AgentMem's wRRF fusion + token-budget greedy packing delivers more signal per token (873 vs 1,900 injected).

---

## vs. Competitors

| Feature | **AgentMem** | agentmemory | Mem0 (53K ⭐) | Letta/MemGPT | CLAUDE.md |
|---------|:------------:|:-----------:|:------------:|:------------:|:---------:|
| **Retrieval R@5** | **95.2%** | 95.2%⚠ | 68.5%‡ | 83.2%‡ | n/a |
| **LLM-F1** | **76.34%†** | not published | 34.20% | — | n/a |
| **P50 recall latency** | **19ms** | not published | ~50ms | ~200ms | n/a |
| **Injected tokens/session** | **~873** | ~1,900 | varies | always in ctx | 22K+ |
| **Backend** | Redis 8 HNSW | iii engine (closed‡‡) | Qdrant/pgvec | Postgres+vec | static file |
| **External DB** | Redis (local) | **none** | required | required | none |
| **Source** | **fully open** | iii-sdk closed‡‡ | open | open | n/a |
| **Price** | **$0** | **$0** | $0.002–0.01/op | $0 self-host | $0 |
| **Fully local** | ✅ | ✅ | ❌ | ✅ option | ✅ |
| **Memory injection per prompt** | **✅ every prompt** | MCP call only⁴ | manual | manual | manual |
| **Auto-consolidation (always-on)** | **✅ A-MAC gate** | opt-in¹ | manual | manual | none |
| **Auto-capture hooks** | 7 | **12** | manual | self-edits | manual |
| **MCP tools** | 9 | 8 default / 53 all² | — | — | — |
| **6-tier cognitive stack** | **✅** | 4-tier | ❌ | ❌ | ❌ |
| **HyDE query expansion** | **✅ (default on)** | ❌ | ❌ | ❌ | ❌ |
| **MIRIX active retrieval topic** | **✅ (default on)** | ❌ | ❌ | ❌ | ❌ |
| **Session diversity** | **✅ configurable** | hardcoded³ | ❌ | ❌ | ❌ |
| **ToolMem + TIG + AWO** | **✅** | ❌ | ❌ | ❌ | ❌ |
| **MACLA Beta procedures** | **✅** | ❌ | ❌ | ❌ | ❌ |
| **wRRF fusion** | **✅** | ✅ | ❌ | ❌ | ❌ |
| **Semantic triples (s,p,o)** | **✅** | ✅ | ❌ | ❌ | ❌ |
| **A-MAC 5-factor gate** | **✅ always-on** | opt-in¹ | hash only | none | none |
| **3-phase consolidation** | **✅ automatic** | opt-in¹ | ❌ | ❌ | ❌ |
| **Benchmark on real data** | **✅ compressed** | raw text only⚠ | varies | varies | n/a |
| **Secret redaction** | **✅** | ✅ | ❌ | ❌ | ❌ |
| **Search complexity** | **O(log n) HNSW** | O(n) flat | O(log n) | O(n) | grep |
| **Memory @ 10K facts** | **~200MB** | ~1GB est. | varies | varies | n/a |

> ‡Mem0/Letta: LoCoMo end-to-end QA, different benchmark. †Internal bench, see disclaimer.  
> ⚠ agentmemory's 95.2% R@5 is measured on raw session text, not LLM-compressed memories. Their compression pipeline would reduce this number.  
> ‡‡ `iii-sdk` / `iii engine` is the actual runtime backing all agentmemory hooks and functions — a hard dependency on a closed-source proprietary engine. Not auditable, not forkable, not self-hostable independently.  
> ¹ Requires `CONSOLIDATION_ENABLED=true`; off by default in fresh installs. Also requires ≥5 session summaries before semantic consolidation runs at all (code: `if (summaries.length >= 5)`).  
> ² agentmemory exposes only 8 tools by default (`AGENTMEMORY_TOOLS=core`); 53 tools requires explicit `AGENTMEMORY_TOOLS=all`.  
> ³ agentmemory's `diversifyBySession()` has `maxPerSession=3` hardcoded at the call site (`hybrid-search.ts`). AgentMem's session diversity limit is configurable via recall parameters.  
> ⁴ agentmemory's `UserPromptSubmit` hook calls `POST /observe` (fire-and-forget, no `additionalContext` return). Memory is injected only when Claude explicitly calls the `mem::recall` MCP tool — not automatically per prompt.

**Where agentmemory wins:** no external DB required, `npm install -g` one-liner, more max-tools (53), more hooks (12), real-time viewer UI.

**Where AgentMem wins:** published LLM-F1 on real compressed data (76.34%), measured recall latency (19ms P50), 2x more token-efficient injection (873 vs 1,900), **automatic per-prompt injection via `additionalContext`** (agentmemory requires an explicit `mem::recall` MCP tool call), always-on consolidation from session 1 (agentmemory requires `CONSOLIDATION_ENABLED=true` AND ≥5 sessions), HyDE + MIRIX + session diversity retrieval quality, 6-tier architecture, ToolMem + TIG + AWO + MACLA features unique in open source, Redis 8 HNSW O(log n) at scale, **100% open-source stack** (no proprietary engine dependency).

---

## vs. vexp — Complementary, not competing

**vexp** solves within-session codebase orientation. AgentMem solves cross-session user memory. They're additive:

```
vexp      → agent knows the current codebase  (23 tool calls → 2, per task)
AgentMem  → agent remembers across ALL sessions (decisions, preferences, procedures)
Together  → agent knows your codebase AND remembers everything it's ever learned
```

| | vexp | AgentMem |
|---|:---:|:---:|
| What it stores | Codebase structure | User facts, episodes, procedures |
| Memory scope | Current task | All sessions, forever |
| Problem solved | "Agent doesn't know the codebase" | "Agent forgot last week's decisions" |
| Token impact | 58% less per task | 90% less per session |

---

## 6-Tier Cognitive Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Tier 0 — LLM Context Window  (framework manages this)           │
│  AgentMem injects: <cross_session_memory>…</cross_session_memory>│
│                    ~873 tokens vs ~22,000 raw  (96% smaller)     │
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

agentmemory uses a 4-tier model. AgentMem adds Tier 0 (LLM context budget management) and Tier 5 (ToolMem + TIG + persona), enabling tool reliability tracking and AutoTool hints not found in any other open-source memory system.

---

## Retrieval Pipeline

```
Query →
  ├─ MIRIX Active Retrieval (arXiv:2507.07957) [DEFAULT ON]:
  │   Generate focused retrieval topic from vague queries.
  │   "fix that auth thing" → "JWT token validation middleware error"
  │   Used as BM25 query for higher lexical precision on developer shorthand.
  │
  ├─ HyDE (Hypothetical Document Embeddings, Gao et al. 2022) [DEFAULT ON]:
  │   Generate short hypothetical memory → embed → average with query embedding.
  │   Hypothetical docs sit closer in embedding space to real memories than bare
  │   questions do. ~5-15% recall improvement.
  │   Both MIRIX and HyDE fire in parallel — zero extra latency.
  │
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
  ├─ Importance boost (rules/identity 0.7-1.0 rank above transient context)
  │
  ├─ Session diversity (max 3 facts / 4 episodes per source session)
  │   Prevents one verbose session dominating results.
  │
  └─ Token-budget greedy packing → <cross_session_memory> (~873 tokens)
```

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

## Claude Code Hooks (7)

| Hook | Trigger | Action |
|------|---------|--------|
| `register-env.sh` | SessionStart | Registers OS / git / cwd / model |
| `register-tools.sh` | SessionStart | Registers built-in + MCP tools |
| `recall.sh` | **Every prompt** | Injects memory via `additionalContext` (~15ms) |
| `compact.sh` | PostToolUse | Session compact + ToolMem success feedback |
| `failure.sh` | PostToolUseFailure | ToolMem failure recording |
| `precompact.sh` | PreCompact | Re-injects memory before context compaction |
| `store.sh` | Stop + SubagentStop | Store + compress + TIG + AWO meta-tool synthesis |

**Key advantage over agentmemory:** `recall.sh` fires on *every* `UserPromptSubmit` and returns ranked memory as `additionalContext` to Claude before the response. agentmemory's `prompt-submit.mjs` calls `POST /observe` as a fire-and-forget side-effect — it returns nothing. Memory retrieval only happens if Claude explicitly calls the `mem::recall` MCP tool during a response. There is no automatic per-prompt injection in any agentmemory hook.

---

## MCP Tools (9 focused tools)

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

| | AgentMem | agentmemory | Mem0 | SuperMemory |
|---|:---:|:---:|:---:|:---:|
| Monthly | **$0** | **$0** | $0.002–0.01/op | $29–$99 |
| Cloud dependency | **None** | **None** | Required | Required |
| Annual estimate | **~$10 (electricity)** | ~$10 | $200–$1,200+ | $350–$1,200 |

---

## Research Papers Implemented

| Paper | ArXiv | What AgentMem uses |
|-------|-------|-------------------|
| SimpleMem | [2601.02553](https://arxiv.org/abs/2601.02553) | Lossless restatement (Φ_coref, Φ_time), 3-phase consolidation, intent-aware retrieval planning |
| A-MAC | [2603.04549](https://arxiv.org/abs/2603.04549) | 5-factor admission gate (semantic_novelty, entity_novelty, factual_confidence, temporal_signal, content_type_prior) |
| Memori | [2603.19935](https://arxiv.org/abs/2603.19935) | Semantic triple (s,p,o) extraction, BM25 hybrid |
| AriadneMem | [2603.03290](https://arxiv.org/abs/2603.03290) | Knowledge graph bridge discovery |
| A-MEM | [2502.12648](https://arxiv.org/abs/2502.12648) | Memory evolution: near-duplicate enrichment |
| MemAgent | [2502.12110](https://arxiv.org/abs/2502.12110) | Overwrite-based session compaction |
| ToolMem | [2510.06664](https://arxiv.org/abs/2510.06664) | Per-tool success/fail reliability tracking |
| AutoTool TIG | [2511.14650](https://arxiv.org/abs/2511.14650) | Tool transition graph, next-tool hints |
| MACLA | [2512.18950](https://arxiv.org/abs/2512.18950) | Beta posterior scoring for procedure selection |
| HyDE | [2212.10496](https://arxiv.org/abs/2212.10496) | Hypothetical document embeddings (default on; averaged with query embedding) |
| MIRIX | [2507.07957](https://arxiv.org/abs/2507.07957) | Active retrieval topic generation for vague developer queries (default on) |
| MemMachine | [2604.04853](https://arxiv.org/abs/2604.04853) | Retrieval depth tuning (+4.2% at higher k); parallel decomposition for compound queries |

### Roadmap Papers (tracked, not yet implemented)

| Paper | ArXiv | Planned mechanism |
|-------|-------|-------------------|
| HippoRAG 2 | [2502.14802](https://arxiv.org/abs/2502.14802) | PPR graph propagation over entity graph for associative recall |
| MemoryOS | [2506.06326](https://arxiv.org/abs/2506.06326) | Heat-based tier promotion: hot session KV → episodic, cold → evict |
| EvolveMem | [2605.13941](https://arxiv.org/abs/2605.13941) | AutoResearch loop: self-optimize retrieval config against bench-f1.py |
| MemR3 | [2512.20237](https://arxiv.org/abs/2512.20237) | Closed-loop retrieve→reflect→answer controller for multi-hop queries |
| EMem | [2511.17208](https://arxiv.org/abs/2511.17208) | EDU event-centric facts: (event, participant, temporal_cue, context) |
