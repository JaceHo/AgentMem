"""
OpenClaw Local Memory Service  v0.9.1
======================================
Replaces memos-cloud-openclaw-plugin with fully local, persistent memory.

2026 Layered Controlled Architecture (Tier 0-2):
  Tier 0 — Working memory    : in-context window (LLM prompt)
  Tier 1 — Session KV        : rolling accumulated session summary (Redis String, 4h TTL)
  Tier 2 — Long-term vectors : episodic + semantic + procedural (Redis HNSW, permanent)
  +       — Capability layer : tool/env/agent self-model (Redis Hash + vectorset)

Features (v0.9.1 — SimpleMem arXiv:2601.02553 full implementation):
  1. Heat-tiered recall        — frequently/recently accessed memories rank higher
  2. Scene isolation           — language + domain tags filter recall by context
  3. Evolving persona          — structured user profile updated from extracted facts
  4. Summarize-then-embed      — long turns compressed before MiniLM embedding
  5. Hybrid extraction         — regex (instant) + LLM (GLM-4-flash, async) for facts
  6. Agent capability registry — tool/skill/env/agent self-model (v0.6.0)
  7. Procedural memory         — 4th cognitive tier: how-to workflows (v0.6.0)
  8. Session promotion         — Tier 1→2 compression at session end (v0.7.0)
  9. SimpleMem Sec 3.1         — lossless restatement: Φ_coref (no pronouns) +
                                 Φ_time (absolute ISO 8601), topic/location/importance (v0.9.1)
 10. SimpleMem Sec 3.2         — 3-phase consolidation: Decay (×0.9/90d) → Merge
                                 (soft-delete via superseded_by) → Prune (<0.05) (v0.9.1)
 11. SimpleMem Sec 3.3         — Intent-Aware Retrieval Planning: plan_queries() +
                                 check_sufficiency() reflection loop (v0.9.1)
 12. Token-budget context      — greedy word-count packing into <cross_session_memory>
                                 tags, configurable per request (v0.9.1)
 13. Multi-framework adapters  — LangChain, LangGraph, CrewAI, AutoGen, Claude API (v0.9.0)
 14. MCP server                — mcp_server.py for Claude Desktop/Code/Cursor (v0.9.0)
 15. Knowledge graph           — entity relationship graph in Redis Sets (v0.9.0)
 16. Auto-consolidation        — counter-triggered (every 50 stores) + hourly timer (v0.9.0)

Endpoints:
  POST /recall              — before_agent_start hook (includes tool context + graph)
  POST /store               — agent_end hook (async, non-blocking)
  POST /session/compress    — promote Tier 1 session → Tier 2 long-term
  GET  /session/{id}        — inspect current Tier 1 session state
  POST /register-tools      — register agent's tool/skill index
  POST /register-env        — register current environment state
  POST /recall-tools        — semantic search over tool index
  POST /recall-procedures   — semantic search over procedural memory
  POST /store-procedure     — manually store a workflow/how-to
  GET  /capabilities        — full capability manifest
  GET  /graph/{entity}      — knowledge graph: entity neighbours + fact counts
  POST /graph/recall        — knowledge graph: neighbourhood fact retrieval
  GET  /graph/stats         — knowledge graph: node/edge counts
  GET  /config              — service configuration + auto-consolidation state
  GET  /health
  GET  /stats
"""

import asyncio
import json
import logging
import math
import re
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import capability as cap_mod
import embedder
import extractor
import graph as graph_mod
import heat as heat_mod
import log_sse
import persona as persona_mod
import retrieval_planner
import scene as scene_mod
import store as mem_store
import summarizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [mem] %(message)s")
log = logging.getLogger("mem")

# Attach SSE log handler — broadcasts every log record to dashboard clients
_sse_handler = log_sse.LogSSEHandler(level=logging.INFO)
_sse_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
logging.getLogger().addHandler(_sse_handler)

_redis = None

# ── Auto-consolidation counters ────────────────────────────────────────────────
_stores_since_consolidation: int = 0
_AUTO_CONSOLIDATE_EVERY: int = 50   # trigger after every N stored facts

# ── System-injected content filter ────────────────────────────────────────────
_SKIP_PREFIXES = (
    "## Long-Term Memory",
    "## Recent Relevant Episodes",
    "## Current Session Context",
    "## User Profile",
    "[cron:",
)

def _is_injected(text: str) -> bool:
    t = text.strip()
    return any(t.startswith(p) for p in _SKIP_PREFIXES)


# ── Trivial message filter ─────────────────────────────────────────────────────
_TRIVIAL_MIN_CHARS = 15  # user text shorter than this → skip episode store
_TRIVIAL_PATTERNS = re.compile(
    r"^(hi+|hello+|hey+|yo+|ok+|okay|sure|thanks?|thx|bye+|good\s*(morning|night|day)|"
    r"好的|嗯|哦|呵|哈|谢谢|再见|早|晚安|你好|您好|ji+|test+|ping)[\s!?.。！？]*$",
    re.IGNORECASE,
)

def _is_trivial(user_text: str) -> bool:
    t = user_text.strip()
    if len(t) < _TRIVIAL_MIN_CHARS:
        return True
    return bool(_TRIVIAL_PATTERNS.match(t))


# ── Embedding cache ────────────────────────────────────────────────────────────
@lru_cache(maxsize=1024)
def _cached_encode(text: str) -> bytes:
    return embedder.encode(text).tobytes()

def _encode(text: str) -> np.ndarray:
    return np.frombuffer(_cached_encode(text), dtype=np.float32).copy()


# ── Lifespan ──────────────────────────────────────────────────────────────────
async def _periodic_consolidate() -> None:
    """Background task: consolidate memory every hour if there are facts to merge."""
    while True:
        await asyncio.sleep(3600)
        try:
            card = await _redis.execute_command("VCARD", mem_store.FACT_KEY)
            if int(card or 0) > 5:
                await _do_consolidate()
        except Exception as e:
            log.warning(f"[periodic_consolidate] error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis
    log.info("Loading embedding model…")
    embedder.get_model()
    log.info("Connecting to Redis…")
    _redis = await mem_store.get_client()
    log.info("Ensuring vectorset indexes…")
    await mem_store.ensure_indexes(_redis)
    asyncio.create_task(_periodic_consolidate())
    log.info("Memory service v0.9.1 ready (SimpleMem-aligned)")
    yield
    await _redis.aclose()


app = FastAPI(title="AgentMem — Local Agent Memory Service", version="0.9.1", lifespan=lifespan)

# ── Static files + SSE log router ─────────────────────────────────────────────
import os as _os
_static_dir = _os.path.join(_os.path.dirname(__file__), "static")
if _os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

app.include_router(log_sse.log_sse_router)


@app.get("/", include_in_schema=False)
async def dashboard():
    """Serve the memory service dashboard."""
    idx = _os.path.join(_static_dir, "index.html")
    if _os.path.exists(idx):
        return FileResponse(idx)
    return {"message": "AgentMem v0.9.1", "docs": "/docs"}


# ── Request / Response models ──────────────────────────────────────────────────
class RecallRequest(BaseModel):
    query: str
    session_id: str = ""
    memory_limit_number: int = 6
    include_tools: bool = True          # Whether to append relevant tool context
    include_graph: bool = False         # Expand knowledge graph neighbours (v0.9.0)
    # SimpleMem Section 3.3: Intent-Aware Retrieval Planning (v0.9.1)
    enable_planning: bool = False       # LLM-based query planning (adds ~300-600ms)
    enable_reflection: bool = False     # LLM sufficiency check + second-pass retrieval
    token_budget: int = 1500            # Max tokens in context output (SimpleMem budget)
    # SimpleMem symbolic filter: restrict recall to a time window (Unix ms)
    time_from: int | None = None        # e.g. 1740000000000 (ms)
    time_to:   int | None = None        # e.g. 1742687999000 (ms)

class Message(BaseModel):
    role: str
    content: Any

class StoreRequest(BaseModel):
    messages: list[Message]
    session_id: str = ""

# ── Capability request models ──────────────────────────────────────────────────
class ToolDefinition(BaseModel):
    name: str
    description: str
    category: str = ""
    source: str = "builtin"             # builtin | mcp | plugin | skill
    parameters: list[str] = []

class RegisterToolsRequest(BaseModel):
    tools: list[ToolDefinition]
    agent_id: str = ""                  # optional agent identifier
    replace_all: bool = False           # if True, clear existing tools first

class EnvState(BaseModel):
    os: str = ""
    os_version: str = ""
    shell: str = ""
    cwd: str = ""
    git_repo: str = ""
    git_branch: str = ""
    runtime: str = ""                   # e.g. "python3.12", "node20"
    active_mcps: list[str] = []
    active_plugins: list[str] = []
    active_skills: list[str] = []
    agent_model: str = ""               # e.g. "claude-sonnet-4-6"
    agent_version: str = ""
    session_id: str = ""
    extra: dict = {}                    # any extra key/value pairs

class RecallToolsRequest(BaseModel):
    query: str
    k: int = 5
    category: str = ""                  # optional filter by tool category
    source: str = ""                    # optional filter by source

class StoreProcedureRequest(BaseModel):
    task: str                           # what kind of task (used as embedding key)
    procedure: str                      # the step-by-step procedure
    tools_used: list[str] = []          # tools/skills involved
    domain: str = ""
    session_id: str = ""


# ── Helpers ────────────────────────────────────────────────────────────────────
def _msg_text(m: Message) -> str:
    if isinstance(m.content, str):
        return m.content.strip()
    if isinstance(m.content, list):
        return " ".join(b.get("text","") for b in m.content if isinstance(b, dict)).strip()
    return ""

def _messages_to_text(messages: list[Message]) -> str:
    return "\n".join(
        f"{m.role}: {_msg_text(m)}" for m in messages if _msg_text(m)
    )

def _estimate_tokens(text: str) -> int:
    """Fast word-count token estimate (SimpleMem Section 3.3 budget packing)."""
    return len(text.split())


def _format_prepend(
    facts: list[dict],
    episodes: list[dict],
    session_ctx: str | None,
    persona_ctx: str,
    env_ctx: str = "",
    tool_ctx: str = "",
    token_budget: int = 1500,
) -> str:
    """
    Build the context string injected into the agent's system prompt.

    SimpleMem Section 3.3 token-budget packing:
    Priority order: persona → env → tools → session_ctx → facts → episodes
    Greedy packing: stop adding items when token_budget is exhausted.
    Output wrapped in <cross_session_memory> XML tags (SimpleMem convention).
    """
    sections: list[str] = []
    tokens_used = 0

    def _add(text: str) -> bool:
        nonlocal tokens_used
        cost = _estimate_tokens(text)
        if tokens_used + cost > token_budget:
            return False
        sections.append(text)
        tokens_used += cost
        return True

    # Priority 1: persona (user profile — most stable context)
    if persona_ctx:
        _add(persona_ctx)
        sections.append("")

    # Priority 2: environment context
    if env_ctx:
        _add(env_ctx)
        sections.append("")

    # Priority 3: tool context
    if tool_ctx:
        _add(tool_ctx)
        sections.append("")

    # Priority 4: session context (Tier 1 rolling summary)
    if session_ctx:
        _add("## Current Session Context")
        _add(session_ctx)
        sections.append("")

    # Priority 5: facts (SimpleMem lossless_restatement entries)
    if facts:
        sections.append("## Long-Term Memory (Facts)")
        for i, f in enumerate(facts, 1):
            tag = f.get("category") or f.get("domain") or "?"
            attrs = f.get("attrs", {})
            meta_parts = []
            if p := attrs.get("persons"):
                meta_parts.append(f"who:{','.join(p)}")
            if e := attrs.get("entities"):
                meta_parts.append(f"re:{','.join(e)}")
            if t := attrs.get("topic"):
                meta_parts.append(f"topic:{t}")
            meta = f" ({'; '.join(meta_parts)})" if meta_parts else ""
            line = f"{i}. [{tag}]{meta} {f['content']}"
            if not _add(line):
                break   # budget exhausted
        sections.append("")

    # Priority 6: episodes (lower priority than facts)
    if episodes:
        sections.append("## Recent Relevant Episodes")
        for i, e in enumerate(episodes, 1):
            line = f"{i}. {e['content'][:300]}"
            if not _add(line):
                break
        sections.append("")

    body = "\n".join(sections).strip()
    if not body:
        return ""

    # SimpleMem XML wrapper — clearly delineates injected memory from user content
    return (
        "<cross_session_memory>\n"
        "The following is relevant context from long-term memory.\n"
        "Use it to inform your responses but do not repeat it verbatim.\n\n"
        + body
        + "\n</cross_session_memory>"
    )


# ── Query complexity estimator (SimpleMem-inspired, zero-cost) ────────────────
_COMPLEX_SIGNALS = re.compile(
    r'\b(what|when|where|who|how|why|which|compare|difference|between|all|every|list'
    r'|什么|怎么|哪|为什么|比较|区别|所有|每个|列出)\b', re.I
)

def _estimate_depth(query: str, base_limit: int) -> tuple[int, int]:
    """
    Adaptive retrieval depth: k_dyn = k_base × (1 + δ × C_q).
    Returns (n_facts, n_episodes). C_q estimated from query features.
    """
    words = query.split()
    n_words = len(words)
    n_signals = len(_COMPLEX_SIGNALS.findall(query))
    # Simple complexity score: 0.0 (trivial) to 1.0 (complex)
    c_q = min(1.0, (n_words / 30) + (n_signals * 0.2))
    delta = 0.8   # scaling factor

    k_base_facts = max(1, base_limit // 2)
    k_base_eps   = base_limit

    n_facts    = max(1, int(k_base_facts * (1.0 + delta * c_q)))
    n_episodes = max(2, int(k_base_eps   * (1.0 + delta * c_q * 0.5)))

    return n_facts, n_episodes


# ── SimpleMem: Entropy-Aware Gate (Phase 1) ───────────────────────────────────

_ENTITY_RE = re.compile(
    r'\b[A-Z][a-z]{1,}\b'              # Capitalized words (names, places)
    r'|\b\d{4}[-/]\d{2}[-/]\d{2}\b'   # ISO dates
    r'|\b[A-Z]{2,}\b'                  # Acronyms (API, MCP, ...)
    r'|\b\w+\.(py|js|ts|sh|json|md)\b' # File references
)

async def _entropy_gate(user_text: str) -> bool:
    """
    SimpleMem Phase 1: entropy-aware ingestion filter.
    H(W_t) = α × entity_novelty + β × semantic_divergence
    Returns True  → high entropy, worth storing.
    Returns False → low entropy, discard (saves embedding + Redis write).

    entity_novelty:    min(1.0, unique_entity_count / 4)
    semantic_divergence: 1 - max_cosine_sim against last 3 stored episodes
    Threshold: 0.35 (from SimpleMem paper)
    """
    # α, β weights — semantic divergence is the stronger signal
    alpha, beta = 0.35, 0.65

    # Entity novelty: count distinct named entities / file refs / dates
    entities = set(_ENTITY_RE.findall(user_text))
    entity_novelty = min(1.0, len(entities) / 4)

    # Semantic divergence: compare to recent stored episodes
    emb = _encode(user_text[:300])
    recent = await mem_store.knn_search(_redis, mem_store.EPISODE_KEY, emb, k=3)
    if recent:
        max_sim = max(r.get("score", 0.0) for r in recent)
        semantic_divergence = 1.0 - max_sim
    else:
        semantic_divergence = 1.0   # nothing stored yet → fully novel

    h = alpha * entity_novelty + beta * semantic_divergence
    if h < 0.35:
        log.info(f"[store] entropy gate filtered (H={h:.2f}): {user_text[:50]!r}")
        return False
    return True


# ── SimpleMem: Keyword Lexical Boost (Phase 3 hybrid scoring) ─────────────────

def _tokenize_query(query: str) -> set[str]:
    """Extract meaningful tokens from query for BM25-lite keyword matching."""
    # Strip stopwords, keep tokens ≥ 3 chars
    STOP = {"the", "and", "for", "are", "you", "can", "how", "what",
            "when", "where", "who", "why", "this", "that", "was", "did",
            "的", "了", "是", "在", "有", "我", "你", "他", "她"}
    tokens = set(re.findall(r'\b\w{3,}\b', query.lower()))
    return tokens - STOP


def _keyword_boost(items: list[dict], query: str, boost: float = 0.06) -> list[dict]:
    """
    SimpleMem Phase 3: lexical layer boost.
    For each retrieved item, add `boost` × (matching_kw_count) to score.
    Enables exact-name matches (Bob, 2025-11-20, Redis) to surface higher.
    """
    if not items:
        return items
    query_tokens = _tokenize_query(query)
    if not query_tokens:
        return items

    boosted = []
    for item in items:
        attrs    = item.get("attrs", {})
        keywords = [kw.lower() for kw in (attrs.get("keywords") or [])]
        content  = item.get("content", "").lower()

        # Keyword overlap with stored keywords array
        kw_overlap  = len(query_tokens & set(keywords))
        # Content term overlap (lighter weight)
        ct_overlap  = sum(1 for t in query_tokens if t in content)
        overlap_score = kw_overlap + ct_overlap * 0.3

        new_item = dict(item)
        new_item["score"] = item.get("score", 0.0) + boost * overlap_score
        boosted.append(new_item)

    boosted.sort(key=lambda x: x["score"], reverse=True)
    return boosted


# ── Background store ──────────────────────────────────────────────────────────
async def _do_store(messages: list[Message], session_id: str) -> None:
    t0 = time.time()
    try:
        # 1. Filter system-injected and empty messages
        clean = [m for m in messages if _msg_text(m) and not _is_injected(_msg_text(m))]
        if not clean:
            return
        clean = clean[-4:]   # last 2 real turns

        # 2. Detect scene from user messages
        user_text_raw = " ".join(_msg_text(m) for m in clean if m.role == "user")
        if not user_text_raw:
            return

        # 2b. Skip trivial exchanges (greetings, single words) — prevents feedback loops
        if _is_trivial(user_text_raw):
            log.info(f"[store] skipped trivial exchange: {user_text_raw[:40]!r}")
            return

        # 2c. SimpleMem entropy gate: H(W_t) = α×entity_novelty + β×semantic_divergence
        #     Discards low-entropy inputs before heavy processing (summarize/embed/LLM).
        #     Threshold 0.35 from SimpleMem paper (UNC/UCB/UCSC).
        if not await _entropy_gate(user_text_raw):
            return  # low-entropy input, skip (not worth storing)

        sc = scene_mod.detect(user_text_raw)
        lang, domain = sc["language"], sc["domain"]

        # 3. Summarize long turn before embedding
        turn_text = _messages_to_text(clean)
        summary = await summarizer.summarize(turn_text)

        # 4. Embed + dedup-check + save episode
        ep_emb = _encode(summary[:500])
        similar = await mem_store.knn_search(_redis, mem_store.EPISODE_KEY, ep_emb, k=1)
        # VSIM returns similarity (1=identical). Dedup if score > 0.95.
        is_dup = bool(similar and similar[0].get("score", 0.0) > 0.95)
        ep_saved = 0
        if not is_dup:
            await mem_store.save_episode(
                _redis, session_id, turn_text[:2000], ep_emb, lang, domain
            )
            ep_saved = 1

        # 5. Hybrid fact extraction (regex + LLM) → dedup → save → update persona
        raw_msgs = [m.model_dump() for m in clean if m.role == "user"]
        facts = await extractor.extract_hybrid(raw_msgs, turn_text)
        fact_saved = 0
        for fact in facts:
            f_emb = _encode(fact.content)
            existing = await mem_store.knn_search(_redis, mem_store.FACT_KEY, f_emb, k=1)
            if existing and existing[0].get("score", 0.0) > 0.95:
                continue
            await mem_store.save_fact(
                _redis, fact.content, fact.category, fact.confidence, f_emb, lang, domain,
                keywords=fact.keywords, persons=fact.persons, entities=fact.entities,
                importance=fact.importance, topic=fact.topic, location=fact.location,
            )
            await persona_mod.update(_redis, fact.category, fact.content)
            fact_saved += 1

        # 6. Extract and store procedures (the 4th cognitive tier)
        proc_facts = [f for f in facts if f.category == "procedure"]
        for pf in proc_facts:
            p_emb = _encode(pf.content)
            existing = await mem_store.knn_search(_redis, mem_store.PROC_KEY, p_emb, k=1)
            if not existing or existing[0].get("score", 0.0) < 0.90:
                # Extract tools mentioned in the procedure fact
                tools_in_proc = [
                    w for w in pf.content.lower().split()
                    if len(w) > 3 and w[0].isupper() or w in
                    ("bash","glob","grep","read","edit","write","search","fetch")
                ]
                await mem_store.save_procedure(
                    _redis, task=pf.content, procedure=pf.content,
                    embedding=p_emb, tools_used=tools_in_proc[:5],
                    domain=domain, language=lang,
                )

        # 7. Record knowledge graph edges from persons + entities co-occurrences
        for fact in facts:
            if fact.persons or fact.entities:
                all_ents = (fact.persons or []) + (fact.entities or [])
                if len(all_ents) >= 2:
                    await graph_mod.record_entities(_redis, all_ents, [])

        # 8. Accumulate rolling session summary (Tier 1 — layered controlled architecture)
        #    Append this turn's summary instead of overwriting, so Tier 1 grows across
        #    the whole session rather than holding only the last turn.
        await _accumulate_session(session_id, summary)

        # 9. Auto-consolidation: trigger after every _AUTO_CONSOLIDATE_EVERY stored facts
        global _stores_since_consolidation
        _stores_since_consolidation += fact_saved
        if _stores_since_consolidation >= _AUTO_CONSOLIDATE_EVERY:
            _stores_since_consolidation = 0
            asyncio.create_task(_do_consolidate())
            log.info("[store] auto-consolidation triggered")

        ms = int((time.time() - t0) * 1000)
        log.info(f"[store] session={session_id} lang={lang} domain={domain} "
                 f"ep+{ep_saved} facts+{fact_saved} {ms}ms")

    except Exception as e:
        log.warning(f"[store] background error: {e}", exc_info=True)


# ── Session helpers (Tier 1 — rolling accumulation + promote) ─────────────────

async def _accumulate_session(session_id: str, turn_summary: str) -> None:
    """
    Tier 1 rolling session summary.
    Append each turn's summary; LLM-compress when the accumulated text exceeds
    1 200 chars so the KV value stays readable and token-efficient.
    """
    if not session_id:
        return
    existing = await mem_store.get_session_context(_redis, session_id) or ""
    separator = "\n---\n" if existing else ""
    combined  = f"{existing}{separator}{turn_summary}"
    if len(combined) > 1200:
        combined = await summarizer.summarize(combined)
    await mem_store.set_session_context(_redis, session_id, combined[:1500])


async def _do_compress_session(session_id: str) -> dict:
    """
    Promote Tier 1 session context → Tier 2 long-term vector memory.
    Called by /session/compress (explicit) or the JS plugin on agent_end.

    Steps:
      1. Read accumulated session summary from Redis KV
      2. Summarize the whole session via LLM
      3. Save as a labelled episode in mem:episodes
      4. Run hybrid fact extraction → save facts + update persona
      5. Delete session KV (consumed, now lives in long-term memory)
    """
    if not session_id:
        return {"status": "skipped", "reason": "no session_id"}

    ctx = await mem_store.get_session_context(_redis, session_id)
    if not ctx or len(ctx) < 80:
        return {"status": "skipped", "reason": "session too short to promote"}

    sc = scene_mod.detect(ctx[:500])
    lang, domain = sc["language"], sc["domain"]

    # Summarize the entire accumulated session context
    session_summary = await summarizer.summarize(ctx)

    # Save as a session-summary episode (labelled so recall can distinguish)
    ep_emb  = _encode(session_summary[:500])
    similar = await mem_store.knn_search(_redis, mem_store.EPISODE_KEY, ep_emb, k=1)
    ep_saved = 0
    if not similar or similar[0].get("score", 0.0) < 0.95:
        await mem_store.save_episode(
            _redis, session_id,
            f"[Session Summary] {session_summary[:2000]}",
            ep_emb, lang, domain,
        )
        ep_saved = 1

    # Hybrid fact extraction on the full session text
    facts = await extractor.extract_hybrid(
        [{"role": "user", "content": ctx}], ctx
    )
    fact_saved = 0
    for fact in facts:
        f_emb    = _encode(fact.content)
        existing = await mem_store.knn_search(_redis, mem_store.FACT_KEY, f_emb, k=1)
        if existing and existing[0].get("score", 0.0) > 0.95:
            continue
        await mem_store.save_fact(
            _redis, fact.content, fact.category, fact.confidence,
            f_emb, lang, domain,
            keywords=fact.keywords, persons=fact.persons, entities=fact.entities,
            importance=fact.importance, topic=fact.topic, location=fact.location,
        )
        await persona_mod.update(_redis, fact.category, fact.content)
        fact_saved += 1

    # Delete Tier 1 KV — session has been crystallised into long-term memory
    await _redis.delete(f"{mem_store.SESSION_PRE}{session_id}:ctx")

    log.info(f"[compress] session={session_id} promoted → ep+{ep_saved} facts+{fact_saved}")
    return {
        "status":     "ok",
        "session_id": session_id,
        "ep_saved":   ep_saved,
        "facts_saved":fact_saved,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    try:
        await _redis.ping()
        return {"status": "ok", "redis": "ok", "version": "0.9.1"}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.get("/stats")
async def stats():
    try:
        counts = await mem_store.vcard(_redis)
        persona_raw = await _redis.hgetall("mem:persona")
        counts["persona_fields"] = len(persona_raw)

        # v0.6.0+: capability + procedural stats
        tool_card = await _redis.execute_command("VCARD", cap_mod.TOOL_KEY)
        counts["tools"] = max(0, int(tool_card or 0) - 1)

        proc_card = await _redis.execute_command("VCARD", mem_store.PROC_KEY)
        counts["procedures"] = max(0, int(proc_card or 0) - 1)

        env_raw = await _redis.hgetall(cap_mod.ENV_KEY)
        counts["env_fields"] = len(env_raw)

        return counts
    except Exception as e:
        return {"error": str(e)}


@app.post("/recall")
async def recall(req: RecallRequest):
    t0 = time.time()

    # Detect scene of incoming query
    query_scene = scene_mod.detect(req.query)
    q_lang, q_domain = query_scene["language"], query_scene["domain"]
    emb = _encode(req.query)

    # Adaptive retrieval depth (SimpleMem-inspired C_q estimator)
    n_facts, n_episodes = _estimate_depth(req.query, req.memory_limit_number)

    # Build filter expressions for scene isolation
    lang_filter   = f'.language == "{q_lang}"'
    domain_filter = f'.domain == "{q_domain}"' if q_domain != "general" else None

    # SimpleMem symbolic filter: time-range hard constraint
    # Combines with lang_filter as a compound VSIM FILTER expression
    def _build_filter(base_filter: str) -> str:
        parts = [base_filter] if base_filter else []
        if req.time_from is not None:
            parts.append(f".ts >= {req.time_from}")
        if req.time_to is not None:
            parts.append(f".ts <= {req.time_to}")
        return " && ".join(parts) if parts else ""

    lang_filter_sym = _build_filter(lang_filter)

    # Recall strategy: scene-filtered first, supplement with global if sparse
    async def _fetch(vset: str, k: int, fexpr=None):
        return await mem_store.knn_search(
            _redis, vset, emb, k, filter_expr=fexpr, bump_heat=True
        )

    # Detect if this is a capability query (needs tool context)
    is_cap_query = query_scene.get("is_capability_query", False)

    # Parallel: persona + env + session ctx + scene-filtered facts + episodes
    # + tool recall if capability query or include_tools=True
    gather_tasks = [
        persona_mod.get_context(_redis),
        cap_mod.get_env_context(_redis),
        mem_store.get_session_context(_redis, req.session_id),
        _fetch(mem_store.FACT_KEY,    n_facts,    lang_filter_sym or None),
        _fetch(mem_store.EPISODE_KEY, n_episodes, lang_filter_sym or None),
    ]
    if req.include_tools and is_cap_query:
        gather_tasks.append(cap_mod.recall_tools(_redis, emb, k=6))
    else:
        gather_tasks.append(asyncio.sleep(0))

    results = await asyncio.gather(*gather_tasks)
    persona_ctx = results[0]
    env_ctx     = results[1]
    session_ctx = results[2]
    facts_scene = results[3]
    eps_scene   = results[4]
    tools_raw   = results[5] if isinstance(results[5], list) else []

    # SimpleMem Section 3.3 — Intent-Aware Planning (opt-in)
    # If enable_planning=True, run LLM query planning to generate targeted queries,
    # then re-fetch for each query and merge. Adds ~300-600ms but improves recall
    # for complex multi-part questions.
    if req.enable_planning:
        targeted_queries = await retrieval_planner.plan_queries(req.query)
        if len(targeted_queries) > 1:
            # Run targeted queries in parallel (exclude original already fetched above)
            extra_queries = [q for q in targeted_queries if q != req.query]
            extra_tasks = [
                mem_store.knn_search(_redis, mem_store.FACT_KEY, _encode(q), n_facts,
                                     filter_expr=lang_filter_sym or None, bump_heat=True)
                for q in extra_queries
            ]
            extra_results = await asyncio.gather(*extra_tasks)
            seen_extra = {f["content"] for f in facts_scene}
            for batch in extra_results:
                for item in (batch if isinstance(batch, list) else []):
                    if item["content"] not in seen_extra:
                        facts_scene.append(item)
                        seen_extra.add(item["content"])
            log.info(f"[recall] planning: {len(targeted_queries)} queries, "
                     f"total facts after planning: {len(facts_scene)}")

    # Supplement with global search if scene results are sparse
    supplement_tasks = []
    if len(facts_scene) < n_facts:
        supplement_tasks.append(_fetch(mem_store.FACT_KEY, n_facts - len(facts_scene)))
    else:
        supplement_tasks.append(asyncio.sleep(0))

    if len(eps_scene) < max(2, n_episodes // 2):
        supplement_tasks.append(
            _fetch(mem_store.EPISODE_KEY, n_episodes - len(eps_scene))
        )
    else:
        supplement_tasks.append(asyncio.sleep(0))

    supp_results = await asyncio.gather(*supplement_tasks)
    facts_global = supp_results[0] if isinstance(supp_results[0], list) else []
    eps_global   = supp_results[1] if isinstance(supp_results[1], list) else []

    # Merge, dedup by content, heat-rerank
    def _merge(primary, supplement, limit):
        seen = {i["content"] for i in primary}
        merged = list(primary) + [i for i in supplement if i["content"] not in seen]
        ranked = heat_mod.heat_rerank(merged)
        return ranked[:limit]

    facts    = _merge(facts_scene,   facts_global,  n_facts)
    episodes = _merge(eps_scene,     eps_global,    n_episodes)

    # SimpleMem Phase 3: lexical keyword boost (BM25-lite layer)
    # Boosts facts whose stored keywords overlap with query tokens.
    facts    = _keyword_boost(facts,    req.query)
    episodes = _keyword_boost(episodes, req.query, boost=0.03)  # lighter for episodes

    # SimpleMem reflection loop (opt-in): check if results sufficient, if not re-fetch
    if req.enable_reflection and facts:
        is_sufficient = await retrieval_planner.check_sufficiency(req.query, facts)
        if not is_sufficient:
            # One additional targeted pass with a different angle
            extra_q = await retrieval_planner.plan_queries(
                f"Find additional information about: {req.query}"
            )
            if extra_q:
                reflection_emb = _encode(extra_q[0])
                reflection_facts = await mem_store.knn_search(
                    _redis, mem_store.FACT_KEY, reflection_emb,
                    k=max(3, n_facts // 2), bump_heat=True
                )
                seen_reflect = {f["content"] for f in facts}
                for rf in reflection_facts:
                    if rf["content"] not in seen_reflect:
                        facts.append(rf)
                        seen_reflect.add(rf["content"])
                facts = facts[:n_facts + 3]
                log.info(f"[recall] reflection pass: added {len(reflection_facts)} candidates")

    # Knowledge graph expansion (v0.9.0): pull facts from graph neighbourhood
    if req.include_graph and facts:
        graph_ents: list[str] = []
        for f in facts[:3]:
            attrs = f.get("attrs", {})
            graph_ents += attrs.get("persons", []) + attrs.get("entities", [])
        graph_ents = list(dict.fromkeys(graph_ents))[:5]

        if graph_ents:
            graph_tasks = [
                graph_mod.entity_recall(_redis, ent, emb, k=2)
                for ent in graph_ents
            ]
            graph_results = await asyncio.gather(*graph_tasks)
            seen_contents = {f["content"] for f in facts}
            for batch in graph_results:
                for gf in batch:
                    if gf["content"] not in seen_contents:
                        facts.append(gf)
                        seen_contents.add(gf["content"])
            facts = facts[:n_facts + 3]

    # Format tool context (only include env if it's non-trivial)
    tool_ctx = cap_mod.format_tool_context(tools_raw) if tools_raw else ""
    env_ctx_formatted = env_ctx if env_ctx and is_cap_query else ""

    # SimpleMem Section 3.3 token-budget injection with XML wrapper
    prepend = _format_prepend(
        facts, episodes, session_ctx, persona_ctx,
        env_ctx=env_ctx_formatted, tool_ctx=tool_ctx,
        token_budget=req.token_budget,
    )
    ms = int((time.time() - t0) * 1000)

    planning_info = f" planning={req.enable_planning}" if req.enable_planning else ""
    log.info(f"[recall] session={req.session_id} lang={q_lang} domain={q_domain} "
             f"facts={len(facts)} ep={len(episodes)} tools={len(tools_raw)}"
             f"{planning_info} {ms}ms")

    return {"prependContext": prepend if prepend else None, "latency_ms": ms}


@app.post("/store")
async def store_memory(req: StoreRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(_do_store, req.messages, req.session_id)
    return {"status": "queued"}


# ── Session Tier-1 endpoints (v0.7.0) ─────────────────────────────────────────

class CompressSessionRequest(BaseModel):
    session_id: str
    wait: bool = False   # True → sync (returns results); False → background

@app.post("/session/compress")
async def compress_session(req: CompressSessionRequest, background_tasks: BackgroundTasks):
    """
    Promote Tier 1 session KV → Tier 2 long-term vector memory.

    Call this from the agent_end hook when a session is finishing.
    The accumulated session summary is compressed, saved as an episode,
    and fact-extracted — then the session KV is deleted.

    Set wait=true to block until promotion completes (returns full result).
    """
    if req.wait:
        result = await _do_compress_session(req.session_id)
        return result
    background_tasks.add_task(_do_compress_session, req.session_id)
    return {"status": "queued", "session_id": req.session_id}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """
    Inspect current Tier 1 session context for a given session_id.
    Useful for debugging what the agent has accumulated in this session.
    """
    ctx = await mem_store.get_session_context(_redis, session_id)
    return {
        "session_id": session_id,
        "context":    ctx,
        "length":     len(ctx) if ctx else 0,
        "tier":       "Tier 1 / Session KV",
    }


# ── Capability endpoints (v0.6.0) ─────────────────────────────────────────────

@app.post("/register-tools")
async def register_tools(req: RegisterToolsRequest, background_tasks: BackgroundTasks):
    """
    Register the agent's available tools/skills into mem:tools vectorset.
    Tools are embedded for semantic recall ("find tools that can X").
    Call this from before_agent_start with the full tool list.
    """
    if not req.tools:
        return {"status": "ok", "registered": 0}

    async def _do_register():
        count = 0
        for tool in req.tools:
            emb = _encode(f"{tool.name}: {tool.description}")
            await cap_mod.register_tool(
                _redis,
                name=tool.name,
                description=tool.description,
                embedding=emb,
                category=tool.category,
                source=tool.source,
                parameters=tool.parameters or [],
                agent_id=req.agent_id,
            )
            count += 1
        log.info(f"[tools] registered {count} tools (agent={req.agent_id})")

    background_tasks.add_task(_do_register)
    return {"status": "queued", "tool_count": len(req.tools)}


@app.post("/register-env")
async def register_env(req: EnvState):
    """
    Store current environment state in mem:env hash.
    Call this from before_agent_start with OS/shell/cwd/git/MCP info.
    """
    env_data: dict = {}
    if req.os:           env_data["os"]            = req.os
    if req.os_version:   env_data["os_version"]    = req.os_version
    if req.shell:        env_data["shell"]          = req.shell
    if req.cwd:          env_data["cwd"]            = req.cwd
    if req.git_repo:     env_data["git_repo"]       = req.git_repo
    if req.git_branch:   env_data["git_branch"]     = req.git_branch
    if req.runtime:      env_data["runtime"]        = req.runtime
    if req.agent_model:  env_data["agent_model"]    = req.agent_model
    if req.agent_version:env_data["agent_version"]  = req.agent_version
    if req.session_id:   env_data["session_id"]     = req.session_id
    if req.active_mcps:  env_data["active_mcps"]    = req.active_mcps
    if req.active_plugins: env_data["active_plugins"] = req.active_plugins
    if req.active_skills:  env_data["active_skills"]  = req.active_skills
    if req.extra:
        for k, v in req.extra.items():
            env_data[k] = str(v)

    await cap_mod.set_env(_redis, env_data)
    log.info(f"[env] registered env: {list(env_data.keys())}")
    return {"status": "ok", "fields": list(env_data.keys())}


@app.post("/recall-tools")
async def recall_tools_endpoint(req: RecallToolsRequest):
    """
    Semantic search over registered tools.
    Query: natural language description of needed capability.
    Returns: ranked list of matching tools with scores.

    Example: {"query": "search the filesystem for files"} →
             [{"name": "Glob", "description": "...", "score": 0.92}]
    """
    t0 = time.time()
    emb = _encode(req.query)
    tools = await cap_mod.recall_tools(
        _redis, emb,
        k=req.k,
        category_filter=req.category or None,
        source_filter=req.source or None,
    )
    ms = int((time.time() - t0) * 1000)
    return {"tools": tools, "count": len(tools), "latency_ms": ms}


@app.get("/capabilities")
async def get_capabilities():
    """
    Return the full agent capability manifest:
    - tools: all registered tools with metadata
    - env: current environment state
    - agent: agent self-model
    - stats: summary counts by category
    """
    return await cap_mod.get_capability_summary(_redis)


@app.post("/recall-procedures")
async def recall_procedures(req: RecallToolsRequest):
    """
    Semantic search over procedural memory (4th cognitive tier).
    Query with a task description → get back how-to procedures.

    Example: {"query": "how to search files by pattern", "k": 3}
    → [{"task": "find Python files recursively", "procedure": "Use Glob with **/*.py pattern"}]
    """
    t0 = time.time()
    emb = _encode(req.query)
    blob = emb.astype("float32").tobytes()
    cmd = ["VSIM", mem_store.PROC_KEY, "FP32", blob,
           "COUNT", req.k + 1, "WITHSCORES", "WITHATTRIBS"]
    try:
        results = await _redis.execute_command(*cmd)
    except Exception:
        return {"procedures": [], "latency_ms": 0}

    procs = []
    i = 0
    while i + 2 < len(results):
        elem  = results[i]
        score = results[i+1]
        raw   = results[i+2]
        i += 3
        elem_str = elem.decode() if isinstance(elem, bytes) else elem
        if elem_str == "__seed__":
            continue
        try:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
            continue
        if attrs.get("_seed") or not attrs.get("task"):
            continue
        procs.append({
            "task":          attrs.get("task", ""),
            "procedure":     attrs.get("procedure", ""),
            "tools_used":    attrs.get("tools_used", []),
            "domain":        attrs.get("domain", ""),
            "success_count": attrs.get("success_count", 1),
            "score":         float(score) if score else 0.0,
        })

    ms = int((time.time() - t0) * 1000)
    return {"procedures": procs, "count": len(procs), "latency_ms": ms}


@app.post("/store-procedure")
async def store_procedure(req: StoreProcedureRequest, background_tasks: BackgroundTasks):
    """
    Explicitly store a procedural memory (agent workflow / how-to pattern).
    The task description is embedded for semantic retrieval.
    Agents or the JS plugin can call this after completing a successful task.
    """
    async def _do_store_proc():
        emb = _encode(req.task)
        sc = scene_mod.detect(req.task)
        existing = await mem_store.knn_search(_redis, mem_store.PROC_KEY, emb, k=1)
        if existing and existing[0].get("score", 0.0) > 0.90:
            # Bump success count on near-duplicate
            elem = existing[0].get("_element")
            if elem:
                attrs = dict(existing[0].get("attrs", {}))
                attrs["success_count"] = attrs.get("success_count", 1) + 1
                try:
                    await _redis.execute_command(
                        "VSETATTR", mem_store.PROC_KEY, elem, json.dumps(attrs)
                    )
                except Exception:
                    pass
            return
        await mem_store.save_procedure(
            _redis,
            task=req.task,
            procedure=req.procedure,
            embedding=emb,
            tools_used=req.tools_used,
            domain=req.domain or sc["domain"],
            language=sc["language"],
        )
        log.info(f"[proc] stored procedure: {req.task[:60]}")

    background_tasks.add_task(_do_store_proc)
    return {"status": "queued"}


@app.get("/capabilities/context")
async def get_capability_context():
    """
    Return environment context as formatted string (same as what /recall injects).
    Useful for debugging what an agent sees about its environment.
    """
    env_ctx   = await cap_mod.get_env_context(_redis)
    persona   = await persona_mod.get_context(_redis)
    agent     = await cap_mod.get_agent(_redis)
    all_tools = await cap_mod.list_all_tools(_redis)

    # Format tools by category
    by_cat: dict[str, list] = {}
    for t in all_tools:
        cat = t.get("category", "general")
        by_cat.setdefault(cat, []).append(t["name"])

    tool_lines = []
    for cat, names in sorted(by_cat.items()):
        tool_lines.append(f"  [{cat}]: {', '.join(names)}")

    return {
        "persona_context":     persona,
        "env_context":         env_ctx,
        "agent_self_model":    agent,
        "tool_index_by_category": by_cat,
        "tool_count":          len(all_tools),
    }


# ── Consolidation (SimpleMem-inspired recursive merging) ─────────────────────

async def _do_consolidate(
    similarity_threshold: float = 0.85,
    temporal_lambda: float = 0.03,      # temporal decay: λ=0.03 → 23-day half-life
) -> dict:
    """
    SimpleMem 3-phase consolidation (Section 3.2):
      Phase 1 — Decay:  importance × 0.9 for entries older than 90 days
      Phase 2 — Merge:  cluster near-duplicates (cosine×temporal ≥ threshold),
                        LLM-merge into keeper, soft-delete losers via superseded_by
      Phase 3 — Prune:  soft-delete any active entry with importance < 0.05
    """
    t0 = time.time()
    now_ms = int(time.time() * 1000)
    NINETY_DAYS_MS = 90 * 86_400_000

    card = await _redis.execute_command("VCARD", mem_store.FACT_KEY)
    if not card or int(card) <= 1:
        return {"merged": 0, "decayed": 0, "pruned": 0, "ms": 0}

    try:
        seed = np.zeros(mem_store.DIMS, dtype=np.float32)
        results = await _redis.execute_command(
            "VSIM", mem_store.FACT_KEY, "FP32", seed.tobytes(),
            "COUNT", min(int(card), 500), "WITHSCORES", "WITHATTRIBS"
        )
    except Exception as e:
        log.warning(f"[consolidate] failed to scan facts: {e}")
        return {"error": str(e)}

    # Parse only active (non-superseded) facts
    all_facts = []
    i = 0
    while i + 2 < len(results):
        elem = results[i]
        score = results[i + 1]
        raw = results[i + 2]
        i += 3
        elem_str = elem.decode() if isinstance(elem, bytes) else elem
        if elem_str == "__seed__":
            continue
        try:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
            continue
        if attrs.get("_seed") or not attrs.get("content"):
            continue
        if attrs.get("superseded_by"):   # skip already-superseded entries
            continue
        all_facts.append({"element": elem_str, "attrs": attrs})

    if len(all_facts) < 2:
        return {"merged": 0, "decayed": 0, "pruned": 0, "total": len(all_facts), "ms": 0}

    # ── Phase 1: Decay ─────────────────────────────────────────────────────────
    # Entries older than 90 days lose 10% importance each consolidation run.
    decayed_count = 0
    for fact in all_facts:
        ts = fact["attrs"].get("ts", now_ms)
        if (now_ms - ts) > NINETY_DAYS_MS:
            old_imp = fact["attrs"].get("importance", 0.5)
            new_imp = round(old_imp * 0.9, 4)
            fact["attrs"]["importance"] = new_imp
            try:
                await _redis.execute_command(
                    "VSETATTR", mem_store.FACT_KEY, fact["element"],
                    json.dumps(fact["attrs"])
                )
                decayed_count += 1
            except Exception as e:
                log.debug(f"[consolidate] decay VSETATTR failed: {e}")

    # ── Phase 2: Merge ─────────────────────────────────────────────────────────
    # affinity = cosine_similarity × exp(−λ × |days_between|)
    # Cluster losers are soft-deleted (superseded_by = keeper_element), never VREM'd.
    merged_count = 0
    superseded_elements: set[str] = set()

    for fact in all_facts:
        if fact["element"] in superseded_elements:
            continue

        f_emb = _encode(fact["attrs"]["content"])
        similar = await mem_store.knn_search(
            _redis, mem_store.FACT_KEY, f_emb, k=5
        )

        cluster = [fact]
        fact_ts = fact["attrs"].get("ts", 0)

        for s in similar:
            if s["_element"] == fact["element"]:
                continue
            if s["_element"] in superseded_elements:
                continue

            cosine_sim = s.get("score", 0.0)
            s_ts = 0
            for f2 in all_facts:
                if f2["element"] == s["_element"]:
                    s_ts = f2["attrs"].get("ts", 0)
                    break

            days_between = abs(fact_ts - s_ts) / 86_400_000
            temporal_factor = math.exp(-temporal_lambda * days_between)
            affinity = cosine_sim * temporal_factor

            if affinity >= similarity_threshold:
                for f2 in all_facts:
                    if f2["element"] == s["_element"]:
                        cluster.append(f2)
                        break

        if len(cluster) < 2:
            continue

        contents = [c["attrs"]["content"] for c in cluster]
        merged_content = await _llm_merge_facts(contents)
        if not merged_content:
            continue

        cluster.sort(key=lambda c: c["attrs"].get("ts", 0), reverse=True)
        keeper = cluster[0]
        keeper_element = keeper["element"]

        new_attrs = dict(keeper["attrs"])
        new_attrs["content"] = merged_content[:500]
        new_attrs["access_count"] = max(c["attrs"].get("access_count", 0) for c in cluster)
        new_attrs["importance"] = max(c["attrs"].get("importance", 0.5) for c in cluster)
        new_attrs["consolidated_from"] = len(cluster)

        all_kw: set[str] = set()
        all_persons: set[str] = set()
        all_entities: set[str] = set()
        for c in cluster:
            all_kw.update(c["attrs"].get("keywords", []))
            all_persons.update(c["attrs"].get("persons", []))
            all_entities.update(c["attrs"].get("entities", []))
        if all_kw:
            new_attrs["keywords"] = list(all_kw)[:10]
        if all_persons:
            new_attrs["persons"] = list(all_persons)[:5]
        if all_entities:
            new_attrs["entities"] = list(all_entities)[:5]

        m_emb = _encode(merged_content)
        try:
            await _redis.execute_command(
                "VADD", mem_store.FACT_KEY, "FP32", m_emb.tobytes(),
                keeper_element, "SETATTR", json.dumps(new_attrs)
            )
        except Exception as e:
            log.warning(f"[consolidate] failed to update keeper: {e}")
            continue

        # Soft-delete losers: mark superseded_by = keeper_element (never VREM)
        for c in cluster[1:]:
            if c["element"] not in superseded_elements:
                await mem_store.soft_delete_fact(_redis, c["element"], keeper_element)
                superseded_elements.add(c["element"])

        merged_count += 1

    # ── Phase 3: Prune ─────────────────────────────────────────────────────────
    # Soft-delete any remaining active entry whose importance has decayed below 0.05.
    pruned_count = 0
    for fact in all_facts:
        if fact["element"] in superseded_elements:
            continue
        if fact["attrs"].get("importance", 1.0) < 0.05:
            await mem_store.soft_delete_fact(_redis, fact["element"], "pruned")
            superseded_elements.add(fact["element"])
            pruned_count += 1

    ms = int((time.time() - t0) * 1000)
    log.info(
        f"[consolidate] decayed={decayed_count} merged={merged_count} "
        f"pruned={pruned_count} {ms}ms"
    )
    return {
        "decayed": decayed_count,
        "merged": merged_count,
        "pruned": pruned_count,
        "total_before": len(all_facts),
        "total_after": len(all_facts) - len(superseded_elements),
        "ms": ms,
    }


async def _llm_merge_facts(contents: list[str]) -> str | None:
    """Use GLM-4-flash to merge multiple similar facts into one consolidated fact."""
    key = extractor._load_key()
    if not key:
        # Fallback: just pick the longest fact
        return max(contents, key=len)

    facts_text = "\n".join(f"- {c}" for c in contents)
    prompt = (
        f"以下是关于同一主题的多条记忆，请合并为一条完整、准确的事实陈述。"
        f"保留所有不同的细节，去除重复。输出仅包含合并后的一句话，不要解释。\n\n{facts_text}"
    )

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                extractor.ZAI_URL,
                json={
                    "model": "glm-4-flash",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                    "temperature": 0.1,
                },
                headers={"Authorization": f"Bearer {key}"},
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning(f"[consolidate] LLM merge failed: {e}")

    return max(contents, key=len)


@app.post("/consolidate")
async def consolidate(background_tasks: BackgroundTasks):
    """Trigger memory consolidation — merges similar facts to reduce redundancy."""
    background_tasks.add_task(_do_consolidate)
    return {"status": "consolidation_queued"}


@app.post("/consolidate/sync")
async def consolidate_sync():
    """Synchronous consolidation — returns results immediately."""
    result = await _do_consolidate()
    return result


@app.post("/admin/delete-facts-by-content")
async def admin_delete_facts_by_content(pattern: str = "Always send a report"):
    """Delete facts whose content contains the given pattern (admin only)."""
    card = await _redis.execute_command("VCARD", mem_store.FACT_KEY)
    if not card or int(card) <= 1:
        return {"deleted": 0, "message": "no facts to scan"}

    seed = np.zeros(mem_store.DIMS, dtype=np.float32)
    results = await _redis.execute_command(
        "VSIM", mem_store.FACT_KEY, "FP32", seed.tobytes(),
        "COUNT", min(int(card), 500), "WITHSCORES", "WITHATTRIBS"
    )
    deleted = 0
    i = 0
    while i + 2 < len(results):
        elem = results[i]
        raw = results[i + 2]
        i += 3
        elem_str = elem.decode() if isinstance(elem, bytes) else elem
        if elem_str == "__seed__":
            continue
        try:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
            continue
        content = attrs.get("content", "")
        if pattern.lower() in content.lower():
            await _redis.execute_command("VREM", mem_store.FACT_KEY, elem_str)
            deleted += 1
    return {"deleted": deleted, "pattern": pattern}


# ── Knowledge Graph endpoints (v0.9.0) ────────────────────────────────────────

class GraphRecallRequest(BaseModel):
    entity: str
    k: int = 5


@app.get("/graph/{entity}")
async def graph_neighbors(entity: str):
    """
    Return entities related to *entity* in the knowledge graph, with connection counts.

    Example: GET /graph/bob  →  related entities Bob co-occurs with in stored facts.
    """
    neighbors = await graph_mod.get_entity_neighbors_with_counts(_redis, entity)
    return {
        "entity":    entity,
        "neighbors": neighbors,
        "count":     len(neighbors),
    }


@app.post("/graph/recall")
async def graph_recall(req: GraphRecallRequest):
    """
    Retrieve facts from the knowledge graph neighbourhood of an entity.

    Example: {"entity": "Bob", "k": 5}
    → facts that mention Bob's related entities.
    """
    emb = _encode(req.entity)
    facts = await graph_mod.entity_recall(_redis, req.entity, emb, k=req.k)
    return {
        "entity": req.entity,
        "facts":  [{"content": f["content"], "score": f.get("score", 0.0)} for f in facts],
        "count":  len(facts),
    }


@app.get("/graph/stats")
async def graph_stats_endpoint():
    """Return knowledge graph statistics: node count and edge count."""
    return await graph_mod.graph_stats(_redis)


# ── Config endpoint (v0.9.0) ──────────────────────────────────────────────────

@app.get("/config")
async def get_config():
    """Return current service configuration including auto-consolidation settings."""
    return {
        "version":                 "0.9.1",
        "auto_consolidate_every":  _AUTO_CONSOLIDATE_EVERY,
        "stores_since_last":       _stores_since_consolidation,
        "periodic_interval_s":     3600,
        "entropy_gate_threshold":  0.35,
        "dedup_threshold":         0.95,
        "consolidate_threshold":   0.85,
        "session_ttl_s":           14400,
    }
