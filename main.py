"""
AgentMem — Local Agent Memory Service  v1.1.0
==============================================
Replaces memos-cloud-openclaw-plugin with fully local, persistent memory.

2026 Layered Controlled Architecture (Tier 0-2):
  Tier 0 — Working memory    : in-context window (LLM prompt)
  Tier 1 — Session KV        : rolling accumulated session summary (Redis String, 4h TTL)
  Tier 2 — Long-term vectors : episodic + semantic + procedural (Redis HNSW, permanent)
  +       — Capability layer : tool/env/agent self-model (Redis Hash + vectorset)

Features (v1.1.0 — Refactored with atomic counters + centralized config):
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
                                 tags; bug-fixed header accounting (v0.9.2)
 13. Multi-framework adapters  — LangChain, LangGraph, CrewAI, AutoGen, Claude API (v0.9.0)
 14. MCP server                — mcp_server.py for Claude Desktop/Code/Cursor (v0.9.0)
 15. Knowledge graph           — entity relationship graph in Redis Sets (v0.9.0)
 16. Auto-consolidation        — counter-triggered (every 50 stores) + hourly timer (v0.9.0)
 17. A-MAC admission gate      — 5-factor gate: semantic_novelty + entity_novelty +
                                 factual_confidence + temporal_signal + content_type_prior (v0.9.2)
 18. Dynamic weighted RRF      — per-query-type wRRF weights: entity/temporal/semantic (v0.9.2)
 19. Category importance floors — 15-tier floor map; identity/rule pinned ≥0.80 (v0.9.2)
 20. Episode type taxonomy    — 8-type classification per claude-mem: bugfix|feature|
                                 discovery|decision|change|procedure|preference|context (v0.9.3)
 21. Causal episode chaining  — doubly-linked prev/next episode IDs in Redis attrs,
                                 tracks temporal narrative flow across sessions (v0.9.3)
 22. Mid-session compact      — /session/compact: LLM-compress Tier 1 KV when >3000
                                 chars, PostToolUse hook keeps context O(N) (v0.9.3)
 23. Session handoff bridge  — /session/compress pins last summary; always injected
                                 at next session start for cross-session continuity (v1.0)
 24. Hard-delete pruning     — daily VREM of superseded facts (>7d) and stale
                                 episodes (>180d, unaccessed); keeps HNSW fresh (v1.0)
 25. Auto graph expansion    — graph neighbourhood auto-triggers on entity queries
                                 without explicit include_graph=True flag (v1.0)
 26. Semantic triple extraction — Memori-style (s,p,o) triples extracted alongside
                                 lossless facts; triple_str stored for BM25 coverage (v1.1)
 27. Real BM25 hybrid search  — rank_bm25 BM25Okapi over in-memory fact corpus;
                                 4th wRRF input alongside vector/symbolic/global (v1.1)
 28. A-MEM memory evolution   — near-duplicate facts (cos 0.80-0.95) trigger async
                                 keyword/topic enrichment on existing fact (v1.1)
 29. Dual-layer fact linking  — source_episode_id links each fact to source episode
                                 for Memori-style narrative context retrieval (v1.1)
 30. Atomic counters          — thread-safe state management (v1.1 refactor)
 31. Centralized config       — environment-variable tunable parameters (v1.1 refactor)

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
  POST /session/compact     — mid-session compress Tier 1 KV if > threshold (v0.9.3)
  POST /tool-feedback       — ToolMem: record success/fail per tool invocation (v0.9.5)
  POST /record-tool-sequence — AutoTool TIG: record tool transition sequence (v0.9.5)
  GET  /tool-graph/{name}   — AutoTool TIG: outgoing transitions from tool (v0.9.5)
  POST /tool-graph/detect-meta-tools — AWO: synthesize composite procedures from TIG chains (v0.9.6)
  POST /procedure-feedback  — MACLA Beta: record procedure success/fail (v0.9.5)
  GET  /tool-procedures/{name} — reverse index: procedures that use a given tool (v0.9.6)
  POST /proc-backfill-index — backfill proc_by_tool reverse index for existing procedures (v0.9.6)
  POST /consolidate/hard-prune — manual trigger: VREM soft-deleted + stale entries (v1.0)
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

import httpx
import numpy as np
from fastapi import FastAPI, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Schemas (extracted to api/schemas/) ──────────────────────────────────────
from api.schemas.memory import RecallRequest, Message, StoreRequest
from api.schemas.capability import (
    ToolDefinition, RegisterToolsRequest, EnvState,
    RecallToolsRequest, StoreProcedureRequest,
    ToolFeedbackRequest, ToolSequenceRequest, ProcedureFeedbackRequest,
)
from api.schemas.compat import (
    CompatSessionStartRequest, CompatSessionEndRequest,
    ObserveRequest, SummarizeRequest, EnrichRequest,
    ContextRequest, SessionCommitRequest,
    SearchRequest, RememberRequest, ForgetRequest,
    FileContextRequest, PatternsRequest, SmartSearchRequest,
    TimelineRequest, ClaudeBridgeSyncRequest, ImportRequest,
)
from api.schemas.consolidation import (
    CompactRequest, AnswerRequest, CompressSessionRequest,
    FeedbackRequest, CrystallizeRequest,
)
from api.schemas.graph import (
    GraphRecallRequest, TypedEdgeRequest, TraverseRequest, ReinforceRequest,
)

# ── Compat helpers (extracted to api/compat.py) ──────────────────────────────
from api.compat import compat_sid, check_auth as _compat_check_auth

# ── Search utilities (extracted to core/search.py) ───────────────────────────
from core.search import BM25Index, encode as _encode, vscan as _vscan, BM25_AVAILABLE as _BM25_AVAILABLE

# ── Import refactored modules ────────────────────────────────────────────────
from concurrency import AtomicCounter, AtomicFloat, TaskManager
from config.settings import settings
from utils.text_processing import (
    flatten_message_content,
    estimate_tokens,
    redact_secrets,
    contains_secret,
    strip_platform_noise,
    is_only_platform_noise,
    is_brief_option_reply,
    is_trivial,
    is_injected_system_content,
)

# Private aliases — used throughout main.py with underscore prefix
_is_injected = is_injected_system_content
_strip_platform_noise = strip_platform_noise
_is_only_platform_noise = is_only_platform_noise
_is_brief_option_reply = is_brief_option_reply
_contains_secret = contains_secret
_redact_secrets = redact_secrets
_is_trivial = is_trivial
_flatten_message_content = flatten_message_content
_estimate_tokens = estimate_tokens

from core import capability as cap_mod
from core import embedder
from core import extractor
from core import graph as graph_mod
from core import heat as heat_mod
from core import log_sse
from core import persona as persona_mod
from core import retrieval_planner
from core import scene as scene_mod
from core import store as mem_store
from core import summarizer

logging.basicConfig(level=getattr(logging, settings.log_level), format=settings.log_format)
log = logging.getLogger("mem")
APP_VERSION = settings.app_version

# Attach SSE log handler — broadcasts every record to dashboard clients,
# stores it in an in-memory ring buffer, and persists it to
# ~/.agentmem/logs/dashboard.jsonl so logs survive process restarts.
log_sse.init_persistence()
_sse_handler = log_sse.LogSSEHandler(level=logging.INFO)
# Bare %(message)s — the SSE payload already carries `ts`, `level`, and `name`,
# and the dashboard formats them itself. This avoids double-stamped output.
_sse_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(_sse_handler)

_redis = None

# ── Task manager for fire-and-forget background work ─────────────────────────
_task_manager = TaskManager(max_concurrent=settings.bg_task_limit)
_AUTO_CONSOLIDATE_EVERY = settings.auto_consolidate_every

def _spawn(coro, name: str = "bg") -> None:
    """Fire-and-forget a coroutine with TaskManager supervision."""
    _task_manager.spawn(coro, name=name)

# ── Thread-safe auto-consolidation counter (replaces bare int) ────────────────
_stores_since_consolidation = AtomicCounter()

# ── Thread-safe hard-prune scheduler counter ─────────────────────────────────
_periodic_prune_counter = AtomicCounter()

# ── Thread-safe store observability counters (replaces bare int/float) ───────
_store_attempts = AtomicCounter()
_store_successes = AtomicCounter()
_store_skips = AtomicCounter()
_store_errors = AtomicCounter()
_store_latency_sum_ms = AtomicFloat()

# ── BM25 + search utilities now in core/search.py ────────────────────────────
_bm25_index = BM25Index()

# ── Session handoff: pinned last-session summary ───────────────────────────────
# Key stores the most recent session summary so the NEXT session can always
# retrieve it regardless of query similarity (cross-session continuity bridge).
_PINNED_SESSION_KEY = "mem:pinned:session_summary"


# ── Lifespan ──────────────────────────────────────────────────────────────────
async def _periodic_consolidate() -> None:
    """Background task: consolidate memory every hour; hard-prune every 24 hours."""
    global _periodic_prune_counter
    while True:
        await asyncio.sleep(3600)
        try:
            card = await _redis.execute_command("VCARD", mem_store.FACT_KEY)
            if int(card or 0) > 5:
                await _do_consolidate()
        except Exception as e:
            log.warning(f"[periodic_consolidate] error: {e}")
        prune_count = await _periodic_prune_counter.increment()
        if prune_count >= 24:
            await _periodic_prune_counter.reset()
            try:
                await _do_hard_prune()
            except Exception as e:
                log.warning(f"[periodic_consolidate] hard-prune error: {e}")


async def _populate_bm25_from_redis(r) -> None:
    """Load all existing facts from Redis into the in-memory BM25 corpus on startup."""
    if not _BM25_AVAILABLE:
        return
    try:
        scanned = await _vscan(r, mem_store.FACT_KEY, max_count=5000)
        items = [
            {"_element": item["element"], "content": item["attrs"]["content"], "attrs": item["attrs"]}
            for item in scanned
            if item["attrs"].get("content") and not item["attrs"].get("superseded_by")
        ]
        await _bm25_index.populate_from_items(items)
        log.info(f"[bm25] populated corpus from Redis: {len(items)} facts")
    except Exception as e:
        log.warning(f"[bm25] startup population failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis
    log.info("Loading embedding model…")
    embedder._get_provider()  # warm up embedding model
    log.info("Connecting to Redis…")
    _redis = await mem_store.get_client()

    # Dimension guard: if existing vectorset has different dims than current
    # provider, force local (384-dim) to avoid corrupting vector search.
    try:
        info = await _redis.execute_command("VINFO", mem_store.FACT_KEY)
        if info:
            # VINFO returns flat list: [key, val, key, val, ...]
            info_dict = dict(zip(info[::2], info[1::2]))
            existing_dims = int(info_dict.get(b"vector-dim", info_dict.get("vector-dim", 0)))
            if existing_dims and existing_dims != embedder.DIMS:
                log.warning("Dimension mismatch: existing vectors=%d, provider=%s dims=%d — forcing local provider",
                            existing_dims, embedder._get_provider().name, embedder.DIMS)
                embedder._reset_provider("local")
                mem_store.DIMS = embedder.DIMS
    except Exception:
        pass  # no existing vectorset, safe to use any provider

    log.info("Ensuring vectorset indexes…")
    await mem_store.ensure_indexes(_redis)
    _spawn(_periodic_consolidate(), "consolidate")
    _spawn(_auto_crystallize(), "crystallize")  # Auto-crystallize sessions every 6h
    # Populate BM25 corpus from existing Redis facts (non-blocking, best-effort)
    _spawn(_populate_bm25_from_redis(_redis), "bm25-populate")
    log.info("AgentMem v%s ready (session-handoff+hard-prune+auto-graph+batch-MCP+compat+auto-crystallize)", APP_VERSION)
    yield
    # Cancel tracked background tasks before closing Redis
    await _task_manager.shutdown(timeout=settings.bg_task_shutdown_timeout)
    await mem_store.close_pool()


app = FastAPI(title="AgentMem — Local Agent Memory Service", version=APP_VERSION, lifespan=lifespan)


@app.middleware("http")
async def redis_guard(request: Request, call_next):
    """Return 503 if Redis is not connected (prevents NoneType crashes)."""
    if _redis is None and request.url.path not in ("/health", "/docs", "/openapi.json", "/redoc"):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content={"detail": "Redis not connected"})
    return await call_next(request)

# ── Static files + SSE log router ─────────────────────────────────────────────
import os as _os
_static_dir = _os.path.join(_os.path.dirname(__file__), "static")
if _os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")
    # Mount css/js subdirs at root so <link href="/css/..."> and <script src="/js/..."> work
    _css_dir = _os.path.join(_static_dir, "css")
    _js_dir = _os.path.join(_static_dir, "js")
    if _os.path.isdir(_css_dir):
        app.mount("/css", StaticFiles(directory=_css_dir), name="css")
    if _os.path.isdir(_js_dir):
        app.mount("/js", StaticFiles(directory=_js_dir), name="js")

app.include_router(log_sse.log_sse_router)
from api.routes import memory_router
app.include_router(memory_router)

# ── WebSocket for real-time dashboard updates ─────────────────────────────────
_ws_clients: set[WebSocket] = set()


async def _ws_broadcast(event: str, data: dict) -> None:
    """Push an event to all connected WebSocket clients."""
    if not _ws_clients:
        return
    msg = json.dumps({"event": event, **data, "ts": time.time()})
    dead = set()
    for ws in _ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        # Send initial state
        await ws.send_text(json.dumps({
            "event": "connected",
            "version": APP_VERSION,
            "ts": time.time(),
        }))
        while True:
            # Keep alive — client pings or we just wait for disconnect
            data = await ws.receive_text()
            # Client can request specific data
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await ws.send_text(json.dumps({"event": "pong", "ts": time.time()}))
                elif msg.get("type") == "subscribe":
                    pass  # all events broadcast by default
            except (json.JSONDecodeError, KeyError):
                pass
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


@app.get("/", include_in_schema=False)
async def dashboard():
    """Serve the memory service dashboard."""
    idx = _os.path.join(_static_dir, "index.html")
    if _os.path.exists(idx):
        return FileResponse(idx)
    return {"message": f"AgentMem v{APP_VERSION}", "docs": "/docs"}


# ── Request / Response models ──────────────────────────────────────────────────
# All Pydantic request models are now in api/schemas/ — imported above.

# ── Helpers ────────────────────────────────────────────────────────────────────

def _msg_text(m: Message) -> str:
    raw = _flatten_message_content(m.content)
    return _strip_platform_noise(raw)

def _messages_to_text(messages: list[Message]) -> str:
    return "\n".join(
        f"{m.role}: {_msg_text(m)}" for m in messages if _msg_text(m)
    )


def _format_prepend(
    facts: list[dict],
    episodes: list[dict],
    session_ctx: str | None,
    persona_ctx: str,
    env_ctx: str = "",
    tool_ctx: str = "",
    proc_ctx: list[dict] | None = None,
    token_budget: int = 1500,
    last_session_summary: str | None = None,
    crystal_digests: list[dict] | None = None,
) -> str:
    """
    Build the context string injected into the agent's system prompt.

    SimpleMem Section 3.3 token-budget packing:
    Priority order: persona → last_session_summary → crystallized digests → env → tools → skills → session_ctx → facts → episodes
    Greedy packing: stop adding items when token_budget is exhausted.
    Output wrapped in <cross_session_memory> XML tags (SimpleMem convention).
    """
    sections: list[str] = []
    tokens_used = 0

    def _add(text: str) -> bool:
        nonlocal tokens_used
        text = _redact_secrets(text)
        cost = _estimate_tokens(text)
        if tokens_used + cost > token_budget:
            return False
        sections.append(text)
        tokens_used += cost
        return True

    # Priority 1: persona (user profile — most stable context)
    if persona_ctx and _add(persona_ctx):
        _add("")

    # Priority 1.5: last session summary (cross-session continuity bridge)
    # Always injected when present so the agent knows what was accomplished
    # in the previous session, regardless of current query similarity.
    if last_session_summary and _add("## Last Session Summary"):
        _add(last_session_summary[:600])
        _add("")

    # Priority 1.7: crystallized digests (lessons learned from past sessions)
    # Auto-generated summaries of completed work chains with key findings
    if crystal_digests and _add("## Lessons Learned"):
        for digest in crystal_digests[:3]:  # top 3 digests
            summary = digest.get("summary", "")
            fact_count = digest.get("fact_count", 0)
            entities = digest.get("entities", [])
            
            # Format: summary + key stats
            digest_text = f"- **Completed Session** ({fact_count} facts): {summary}"
            if not _add(digest_text):
                break
            
            # Add key entities if space allows
            if entities and len(entities) <= 5:
                entity_str = ", ".join(entities[:5])
                if not _add(f"  Key entities: {entity_str}"):
                    break
        
        _add("")

    # Priority 2: environment context
    if env_ctx and _add(env_ctx):
        _add("")

    # Priority 3: tool context
    if tool_ctx and _add(tool_ctx):
        _add("")

    # Priority 3.5: procedural skills (MetaClaw-style behavioral guidelines)
    if proc_ctx and _add("## Relevant Skills"):
        for p in proc_ctx:
            task      = p.get("task", "")
            procedure = p.get("procedure", "")
            # task may be "[skill:name] description" — strip the prefix tag for display
            display = task.split("] ", 1)[-1] if task.startswith("[skill:") else task
            # Include first 200 chars of procedure body as a hint
            body_hint = procedure[:200].replace("\n", " ").strip()
            line = f"- **{display}**" + (f": {body_hint}" if body_hint else "")
            if not _add(line):
                break
        _add("")

    # Priority 4: session context (Tier 1 rolling summary)
    if session_ctx and _add("## Current Session Context"):
        _add(session_ctx)
        _add("")

    # Priority 5: facts (SimpleMem lossless_restatement + Memori triple entries)
    # ALL appends go through _add() so headers count toward budget too.
    if facts and _add("## Long-Term Memory (Facts)"):
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
            # Show compact triple form when available (more precise, fewer tokens)
            triple_str = attrs.get("triple_str", "")
            if triple_str:
                meta_parts.append(f"triple:{triple_str}")
            meta = f" ({'; '.join(meta_parts)})" if meta_parts else ""
            line = f"{i}. [{tag}]{meta} {f['content']}"
            if not _add(line):
                break   # budget exhausted
        _add("")

    # Priority 6: episodes (lower priority than facts)
    if episodes and _add("## Recent Relevant Episodes"):
        for i, e in enumerate(episodes, 1):
            attrs = e.get("attrs", {})
            ep_type = attrs.get("ep_type", "")
            type_tag = f"[{ep_type}] " if ep_type and ep_type != "general" else ""
            line = f"{i}. {type_tag}{e['content'][:300]}"
            if not _add(line):
                break
        _add("")

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


# ── A-MAC 5-Factor Admission Gate (arXiv:2603.04549) ─────────────────────────
#
# A-MAC (Adaptive Memory Admission Control, March 2026) achieves F1=0.583 on
# a LoCoMo *admission classification* task (binary: should this memory be stored?
# precision=0.417, recall=0.972). NOTE: this 0.583 is NOT the same metric as
# SimpleMem's 43.24 QA token-F1 — they measure different things on different
# task definitions and cannot be compared on the same leaderboard.
#
# Ablation reveals factor importance order:
#   type_prior > novelty > utility > confidence > recency
# We implement all 5 with zero extra LLM calls (no LLM utility scoring).
# Weight order follows A-MAC ablation (dominant factor gets highest weight):
#   F1 — semantic_novelty    (w=0.25): novelty vs. recently-stored episodes
#   F2 — entity_novelty      (w=0.15): new named entities / dates / file refs
#   F3 — factual_confidence  (w=0.20): declarative predicate density (confidence)
#   F4 — temporal_signal     (w=0.10): explicit dates/times (recency proxy)
#   F5 — content_type_prior  (w=0.30): high-value category patterns — DOMINANT
#                                       (A-MAC ablation: most influential factor)
#
# Threshold: 0.40 (between SimpleMem 0.35 and A-MAC's learned θ*=0.55;
# balances recall coverage vs. precision for a local non-LLM gate)

_ENTITY_RE = re.compile(
    r'\b[A-Z][a-z]{1,}\b'              # Capitalized words (names, places)
    r'|\b\d{4}[-/]\d{2}[-/]\d{2}\b'   # ISO dates
    r'|\b[A-Z]{2,}\b'                  # Acronyms (API, MCP, ...)
    r'|\b\w+\.(py|js|ts|sh|json|md)\b' # File references
    r'|[\u4e00-\u9fff\u3400-\u4dbf]{2,}' # CJK entity names (≥2 chars)
)
_DATE_RE  = re.compile(
    r'\b\d{4}[-/]\d{2}[-/]\d{2}\b'    # ISO date
    r'|\b(?:today|tomorrow|yesterday|monday|tuesday|wednesday|thursday|friday'
    r'|january|february|march|april|may|june|july|august|september|october'
    r'|november|december)\b'
    r'|\d{4}年\d{1,2}月(?:\d{1,2}日)?'  # Chinese date: 2026年3月7日
    r'|(?:昨天|今天|明天|上周|下周|上个月|下个月)', re.I
)
_FACT_RE  = re.compile(                # declarative fact patterns (EN + ZH)
    r'\b(?:is|are|was|were|has|have|had|will|prefer|prefers|preferred|use|uses'
    r'|used|using|like|likes|liked|work|works|worked|working|live|lives|lived'
    r'|need|needs|needed|always|never|must|should|remember|remembers|note|notes'
    r'|default|workspace|directory|repo|repository|branch|shell|tool|stack|my|our)\b'
    r'|(?:是|有|在|住|工作|喜欢|需要|记住|总是|从不|必须|应该|我的)', re.I
)
_HIGH_VALUE_RE = re.compile(           # content type prior: high-value (EN + ZH)
    r'\b(?:my name|i am|i\'m|i work|i live|i prefer|i like|i use|i always|i never'
    r'|always use|never use|uses|prefers|default shell|package manager|workspace'
    r'|directory|repo|repository|branch|project|tool|stack|deadline|remember'
    r'|important|rule:|note:|prefer|password|token|key|api|model)\b'
    r'|(?:我叫|我是|我在|我住|我喜欢|总是用|截止|记住|重要|密码|令牌)', re.I
)

async def _admission_gate(user_text: str) -> bool:
    """
    A-MAC 5-factor admission control (arXiv:2603.04549).

    Returns True  → worth storing (any of the 5 factors fires strongly).
    Returns False → low-value turn, discard to save embedding + write cost.
    """
    text = user_text[:400]

    # F1 — Semantic novelty (w=0.25)
    emb = _encode(text)
    recent = await mem_store.knn_search(_redis, mem_store.EPISODE_KEY, emb, k=3)
    if recent:
        max_sim = max(r.get("score", 0.0) for r in recent)
        semantic_novelty = 1.0 - max_sim
    else:
        semantic_novelty = 1.0  # first store → always novel

    # F2 — Entity novelty (w=0.15)
    entities = set(_ENTITY_RE.findall(text))
    entity_novelty = min(1.0, len(entities) / 4)

    # F3 — Factual confidence: declarative predicate density (w=0.20)
    # For Chinese text .split() returns 1 token for the whole string; use
    # character count ÷ avg_word_len as a proxy for mixed-language word count.
    en_words  = len(text.split())
    zh_chars  = len(re.findall(r'[\u4e00-\u9fff]', text))
    words = max(en_words, 1) + zh_chars // 2  # ~2 chars per Chinese word
    fact_hits = len(_FACT_RE.findall(text))
    factual_confidence = min(1.0, fact_hits / max(words * 0.15, 1))

    # F4 — Temporal signal: explicit date/time references (w=0.10)
    temporal_signal = min(1.0, len(_DATE_RE.findall(text)) / 2)

    # F5 — Content type prior: high-value category patterns (w=0.30, DOMINANT)
    # A-MAC ablation: type_prior is the single most influential factor.
    # Scalar 1.0/0.0 is coarse; use 0.5 for partial matches (soft categories).
    high_value_hits = len(_HIGH_VALUE_RE.findall(text))
    content_type_prior = min(1.0, high_value_hits * 0.5) if high_value_hits else 0.0

    score = (
        0.25 * semantic_novelty
        + 0.15 * entity_novelty
        + 0.20 * factual_confidence
        + 0.10 * temporal_signal
        + 0.30 * content_type_prior
    )

    _gate_threshold = float(_os.getenv("AMAC_THRESHOLD", "0.40"))
    if score < _gate_threshold:
        log.info(
            f"[store] admission gate filtered (score={score:.2f} "
            f"nov={semantic_novelty:.2f} ent={entity_novelty:.2f} "
            f"fact={factual_confidence:.2f} tmp={temporal_signal:.2f} "
            f"typ={content_type_prior:.2f}): {text[:50]!r}"
        )
        return False
    return True


# ── SimpleMem: Keyword Lexical Boost (Phase 3 hybrid scoring) ─────────────────

def _tokenize_query(query: str) -> set[str]:
    """Extract meaningful tokens from query for BM25-lite keyword matching.

    Handles both English (space-separated, ≥3 chars) and Chinese (CJK character
    sequences ≥2 chars, since Chinese has no word boundaries).
    """
    STOP = {"the", "and", "for", "are", "you", "can", "how", "what",
            "when", "where", "who", "why", "this", "that", "was", "did",
            "的", "了", "是", "在", "有", "我", "你", "他", "她", "它",
            "这", "那", "吗", "呢", "啊", "哦", "嗯"}
    # English tokens (≥3 chars)
    en_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
    # Chinese tokens: sequences of CJK chars (≥2 chars)
    zh_tokens = set(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]{2,}', query))
    return (en_tokens | zh_tokens) - STOP


def _keyword_boost(items: list[dict], query: str, boost: float = 0.06) -> list[dict]:
    """
    SimpleMem Phase 3: lexical layer boost (BM25-lite).
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

        kw_overlap    = len(query_tokens & set(keywords))
        ct_overlap    = sum(1 for t in query_tokens if t in content)
        overlap_score = kw_overlap + ct_overlap * 0.3

        new_item = dict(item)
        new_item["score"] = item.get("score", 0.0) + boost * overlap_score
        boosted.append(new_item)

    boosted.sort(key=lambda x: x["score"], reverse=True)
    return boosted


def _importance_boost(items: list[dict], weight: float = 0.15) -> list[dict]:
    """
    AgentMem advantage over SimpleMem: importance-weighted recall reranking.

    SimpleMem uses `importance` only during consolidation winner selection.
    Here we use it during retrieval so high-signal facts (rules, identity,
    preferences — importance 0.7-1.0) outrank transient context (0.3-0.5).

    weight=0.15 adds up to 0.15 to the base cosine score (0.0-1.0 range),
    enough to surface critical memories without overwhelming cosine signal.
    """
    if not items:
        return items
    boosted = []
    for item in items:
        importance = item.get("attrs", {}).get("importance", 0.5)
        new_item = dict(item)
        new_item["score"] = item.get("score", 0.0) + weight * float(importance)
        boosted.append(new_item)
    boosted.sort(key=lambda x: x["score"], reverse=True)
    return boosted


def _rrf_merge(
    lists: list[list[dict]],
    weights: list[float] | None = None,
    k: int = 60,
    limit: int = 20,
) -> list[dict]:
    """
    Dynamic Weighted Reciprocal Rank Fusion.

    Standard RRF (Cormack et al., SIGIR 2009):
      RRF(d) = Σ_i  w_i / (k + rank_i(d))   k=60 smoothing constant

    Weights are query-type adaptive (arXiv:2511.18194 wRRF for agents):
      - Default equal weights [1.0, 1.0, 1.0] for [scene, global, symbolic]
      - Caller passes adjusted weights based on query type
      - Entity/person queries: [0.8, 0.8, 1.4] → upweight symbolic pass
      - Temporal queries: [0.9, 0.9, 1.2] → upweight symbolic (has timestamps)
      - Semantic/general: [1.0, 1.0, 0.6] → downweight symbolic if no entities
    """
    if weights is None:
        weights = [1.0] * len(lists)
    if len(weights) < len(lists):
        weights = weights + [1.0] * (len(lists) - len(weights))

    rrf_scores: dict[str, float] = {}
    items_by_content: dict[str, dict] = {}

    for w, lst in zip(weights, lists):
        for rank, item in enumerate(lst, 1):
            content = item.get("content", "")
            if not content:
                continue
            rrf_scores[content] = rrf_scores.get(content, 0.0) + w / (k + rank)
            if content not in items_by_content:
                items_by_content[content] = item

    sorted_pairs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for content, rrf_score in sorted_pairs[:limit]:
        item = dict(items_by_content[content])
        item["score"] = rrf_score
        result.append(item)
    return result


# ── Episode type taxonomy (claude-mem inspired, v0.9.3) ───────────────────────

def _infer_ep_type(facts: list) -> str:
    """
    Map extracted fact categories → claude-mem episode type taxonomy.

    Priority order ensures the most specific/actionable type wins:
      decision > procedure > feature > change > discovery > preference > context > general

    This mirrors claude-mem's observation type system (bugfix|feature|refactor|
    discovery|decision|change) but uses AgentMem's existing category vocabulary.
    """
    if not facts:
        return "general"
    cat_set = {f.category for f in facts}
    if "decision" in cat_set:
        return "decision"
    if "procedure" in cat_set:
        return "procedure"
    if "capability_gained" in cat_set:
        return "feature"
    if "env_change" in cat_set:
        return "change"
    if cat_set & {"identity", "work", "personal", "location"}:
        return "discovery"
    if cat_set & {"preference", "rule", "reminder"}:
        return "preference"
    if "tool_use" in cat_set:
        return "context"
    return "general"


# ── A-MEM memory evolution: refine existing similar facts ─────────────────────
async def _evolve_similar_fact(element_id: str, new_keywords: list[str], new_topic: str | None) -> None:
    """
    A-MEM-inspired memory evolution (arXiv:2502.12110 §3).

    When a new fact is nearly-duplicate (cosine 0.80-0.95) of an existing fact,
    instead of discarding the new fact silently, we use its richer metadata
    (keywords, topic) to enrich the existing fact's attributes.

    This ensures that as the agent learns more specific context about a topic,
    previously stored facts gain better retrieval metadata — mimicking how
    human memory evolves and refines rather than just appending.
    """
    if not element_id or not new_keywords:
        return
    try:
        raw = await _redis.execute_command("VGETATTR", mem_store.FACT_KEY, element_id)
        if not raw:
            return
        attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        existing_kws = set(attrs.get("keywords") or [])
        # Merge new keywords (deduplicated, capped at 12)
        merged_kws = list(existing_kws | set(new_keywords))[:12]
        attrs["keywords"] = merged_kws
        # Update topic if new one is more specific (longer = more specific)
        if new_topic and len(new_topic) > len(attrs.get("topic") or ""):
            attrs["topic"] = new_topic[:100]
        await _redis.execute_command("VSETATTR", mem_store.FACT_KEY, element_id, json.dumps(attrs))
        log.debug(f"[evolution] enriched fact {element_id}: +{len(new_keywords)} kws")
    except Exception as e:
        log.debug(f"[evolution] skipped: {e}")


async def _check_contradiction(existing_fact: dict, new_fact) -> None:
    """
    LLM Wiki v2: Contradiction detection on store.

    When a new fact is semantically similar (0.80-0.95) to an existing one,
    check if they contradict. If so, add a 'contradicts' graph edge and
    mark the older fact as superseded with reason='contradicted'.

    Uses simple heuristic: if the same subject has opposing values
    (e.g., "X is A" vs "X is B" where A ≠ B), it's a contradiction.
    """
    element_id = existing_fact.get("_element", "")
    if not element_id:
        return

    old_content = existing_fact.get("content", "").lower()
    new_content = (new_fact.content if hasattr(new_fact, "content") else str(new_fact)).lower()

    # Extract entities from both facts for graph edge
    old_ents = existing_fact.get("attrs", {}).get("entities") or []
    new_ents = (new_fact.entities if hasattr(new_fact, "entities") else None) or []

    # Simple contradiction heuristic: same entities, different content
    # (a proper implementation would use LLM judge, but this covers common cases)
    shared_ents = set(e.lower() for e in old_ents) & set(e.lower() for e in new_ents)
    if not shared_ents:
        return

    # Check for opposing patterns: "is X" vs "is not X", "uses A" vs "uses B"
    _OPPOSITES = [
        ("is not", "is"), ("does not", "does"), ("cannot", "can"),
        ("will not", "will"), ("has no", "has"), ("no longer", ""),
    ]
    is_contradiction = False
    for neg, pos in _OPPOSITES:
        if neg in old_content and pos in new_content:
            is_contradiction = True
            break
        if neg in new_content and pos in old_content:
            is_contradiction = True
            break

    if not is_contradiction:
        return

    # Add 'contradicts' graph edges between old and new fact entities
    # (not self-edges — connect entities from the old fact to entities
    # from the new fact that are different but share the same subject)
    for old_ent in old_ents:
        for new_ent in new_ents:
            if old_ent.lower() != new_ent.lower():
                await graph_mod.add_typed_edge(
                    _redis, old_ent, new_ent, "contradicts",
                    confidence=0.9, source="contradiction_detector", bidirectional=True,
                )

    # Mark old fact as superseded
    await mem_store.soft_delete_fact(_redis, element_id, "contradicted", reason="contradicted")
    log.info(f"[contradiction] fact {element_id} superseded: contradicted by new fact")


# ── Background store ──────────────────────────────────────────────────────────
async def _do_store(messages: list[Message], session_id: str) -> None:
    global _store_attempts, _store_successes, _store_skips, _store_errors, _store_latency_sum_ms
    await _store_attempts.increment()
    t0 = time.time()
    outcome = "error"
    try:
        # 1. Filter system-injected and empty messages.
        #    _msg_text() strips <cross_session_memory> prefixes and [[platform_tags]]
        #    before returning text, so purely-noise messages collapse to "" here.
        clean = [m for m in messages if _msg_text(m) and not _is_injected(_msg_text(m))]
        if not clean:
            outcome = "skip"
            return
        clean = clean[-4:]   # last 4 messages = last 2 turns (user+assistant pairs)

        # 1b. Check if the raw (pre-strip) user messages were ONLY cross_session_memory
        #     wrappers — e.g. heartbeat sessions that contain nothing real.
        raw_user_msgs = [
            m for m in messages
            if m.role == "user" and (
                isinstance(m.content, str) and m.content.strip()
            )
        ]
        if raw_user_msgs and all(
            _is_only_platform_noise(m.content if isinstance(m.content, str) else "")
            for m in raw_user_msgs
        ):
            log.info("[store] skipped: all user messages are cross_session_memory wrappers")
            outcome = "skip"
            return

        # 2. Detect scene from user messages
        user_text_raw = " ".join(_msg_text(m) for m in clean if m.role == "user")
        if not user_text_raw:
            outcome = "skip"
            return

        # 2b. Skip trivial exchanges (greetings, single words) — prevents feedback loops
        if _is_trivial(user_text_raw):
            log.info(f"[store] skipped trivial exchange: {user_text_raw[:40]!r}")
            outcome = "skip"
            return

        # 2c. A-MAC 5-factor admission gate (arXiv:2603.04549).
        #     Discards low-value inputs before heavy processing (summarize/embed/LLM).
        #     type_prior is dominant (w=0.30); threshold 0.40.
        if not await _admission_gate(user_text_raw):
            outcome = "skip"
            return  # low-entropy input, skip (not worth storing)

        sc = scene_mod.detect(user_text_raw)
        lang, domain = sc["language"], sc["domain"]

        # 3. Summarize long turn before embedding
        turn_text = _messages_to_text(clean)
        summary = await summarizer.summarize(turn_text)

        # 4. Embed + dedup-check + save episode (with type taxonomy + causal chain)
        ep_emb = _encode(summary[:500])
        similar = await mem_store.knn_search(_redis, mem_store.EPISODE_KEY, ep_emb, k=1)
        is_dup = bool(similar and similar[0].get("score", 0.0) > 0.95)
        ep_saved = 0
        new_ep_uid = ""
        prev_ep_id = ""
        if not is_dup:
            # Causal chain: look up the previous episode for this session
            prev_ep_id = await mem_store.get_last_episode_id(_redis, session_id)
            new_ep_uid = await mem_store.save_episode(
                _redis, session_id, turn_text[:2000], ep_emb, lang, domain,
                ep_type="general",        # placeholder; updated after fact extraction below
                prev_episode_id=prev_ep_id,
            )
            ep_saved = 1

        # 5. Hybrid fact extraction (regex + LLM) → dedup → save → update persona
        # Extract from all roles: assistant messages contain rich facts too
        # (e.g. "I work at NovaPay", "I live in Oakland" are in assistant turns)
        raw_msgs = [{"role": m.role, "content": _msg_text(m)} for m in clean if _msg_text(m)]
        facts = await extractor.extract_hybrid(raw_msgs, turn_text)
        fact_saved = 0
        for fact in facts:
            if _contains_secret(fact.content):
                continue
            f_emb = _encode(fact.content)
            existing = await mem_store.knn_search(_redis, mem_store.FACT_KEY, f_emb, k=1)
            if existing and existing[0].get("score", 0.0) > 0.95:
                # A-MEM memory evolution: similar fact exists → async-update its keywords/topic
                # (arXiv:2502.12110 §3: new memories trigger context refinement on neighbors)
                if fact.keywords and existing[0].get("score", 0.0) > 0.80:
                    _spawn(_evolve_similar_fact(
                        existing[0].get("_element", ""),
                        fact.keywords,
                        fact.topic,
                    ), "evolve")
                continue
            # LLM Wiki v2: Contradiction detection — if a high-similarity fact exists
            # (0.80-0.95 range) with opposing meaning, mark old as superseded.
            if existing and existing[0].get("score", 0.0) > 0.80:
                _spawn(_check_contradiction(
                    existing[0], fact,
                ), "contradiction")
            uid = await mem_store.save_fact(
                _redis, fact.content, fact.category, fact.confidence, f_emb, lang, domain,
                keywords=fact.keywords, persons=fact.persons, entities=fact.entities,
                importance=fact.importance, topic=fact.topic, location=fact.location,
                source_episode_id=new_ep_uid or None,
                triple_s=fact.triple_s, triple_p=fact.triple_p, triple_o=fact.triple_o,
            )
            # Add to in-memory BM25 corpus for hybrid recall
            attrs = {
                "content": fact.content, "category": fact.category,
                "language": lang, "domain": domain,
                "keywords": fact.keywords or [], "importance": fact.importance,
            }
            if fact.triple_s and fact.triple_p and fact.triple_o:
                attrs["triple_str"] = f"{fact.triple_s} | {fact.triple_p} | {fact.triple_o}"
            await _bm25_index.add(uid, fact.content, attrs)
            await persona_mod.update(_redis, fact.category, fact.content)
            fact_saved += 1

        # 5b. Back-fill ep_type on newly saved episode + complete causal chain
        if new_ep_uid:
            ep_type = _infer_ep_type(facts)
            # Update ep_type attr on the episode we just saved
            try:
                raw = await _redis.execute_command("VGETATTR", mem_store.EPISODE_KEY, new_ep_uid)
                if raw:
                    ep_attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                    ep_attrs["ep_type"] = ep_type
                    await _redis.execute_command(
                        "VSETATTR", mem_store.EPISODE_KEY, new_ep_uid, json.dumps(ep_attrs)
                    )
            except Exception:
                pass  # VGETATTR not critical — ep_type stays "general"
            # Complete doubly-linked chain: tell prev episode its successor
            # (prev_ep_id was already fetched before save_episode above)
            if prev_ep_id and prev_ep_id != new_ep_uid:
                _spawn(
                    mem_store.update_episode_next_id(_redis, prev_ep_id, new_ep_uid),
                    "ep-chain",
                )
            # Advance the session's last-episode pointer
            await mem_store.set_last_episode_id(_redis, session_id, new_ep_uid)
            log.debug(f"[store] episode {new_ep_uid} type={ep_type} prev={prev_ep_id or 'none'}")

        # 6. Extract and store procedures (the 4th cognitive tier)
        proc_facts = [f for f in facts if f.category == "procedure"]
        # Known tool name whitelist — matched case-insensitively against procedure text.
        # Using a whitelist (not heuristic capitalization) avoids storing random words
        # like ['When', 'HEARTBEAT.md,'] as "tool names" (bug fixed v0.9.5).
        _KNOWN_TOOLS = {
            "bash", "glob", "grep", "read", "edit", "write", "websearch", "webfetch",
            "agent", "task", "notebook", "notebookedit", "mcp", "skill",
        }
        for pf in proc_facts:
            p_emb = _encode(pf.content)
            existing = await mem_store.knn_search(_redis, mem_store.PROC_KEY, p_emb, k=1)
            if not existing or existing[0].get("score", 0.0) < 0.90:
                # Match only known tool names mentioned in the procedure text
                content_lower = pf.content.lower()
                tools_in_proc = [t for t in _KNOWN_TOOLS if t in content_lower]
                await mem_store.save_procedure(
                    _redis, task=pf.content, procedure=pf.content,
                    embedding=p_emb, tools_used=tools_in_proc[:8],
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

        # 9. Auto-consolidation: trigger after every settings.auto_consolidate_every stored facts
        global _stores_since_consolidation
        fact_count = await _stores_since_consolidation.increment(fact_saved)
        if fact_count >= settings.auto_consolidate_every:
            await _stores_since_consolidation.reset()
            _spawn(_do_consolidate(), "consolidate")
            log.info("[store] auto-consolidation triggered")

        ms = int((time.time() - t0) * 1000)
        await _store_successes.increment()
        outcome = "success"
        log.info(f"[store] session={session_id} lang={lang} domain={domain} "
                 f"ep+{ep_saved} facts+{fact_saved} {ms}ms")

    except Exception as e:
        log.warning(f"[store] background error: {e}", exc_info=True)
    finally:
        ms = int((time.time() - t0) * 1000)
        await _store_latency_sum_ms.add(ms)
        if outcome == "skip":
            await _store_skips.increment()
        elif outcome == "error":
            await _store_errors.increment()

# ── Session helpers (Tier 1 — rolling accumulation + promote) ─────────────────

async def _accumulate_session(session_id: str, turn_summary: str) -> None:
    """
    Tier 1 rolling session summary.

    v1.1 (MemAgent §3.1): uses overwrite_update() instead of naive concatenation.
    When accumulated text exceeds 1200 chars the LLM sees existing memory +
    new turn and selectively retains critical facts — mirrors MemAgent's overwrite
    strategy (arXiv:2507.02259). Memory stays bounded at ~1200 chars regardless
    of session length; total compute is O(N) not O(N²).

    Security: secrets are redacted before storing in Tier 1 KV so they never
    appear in prependContext injected into future prompts.
    """
    if not session_id:
        return
    # Redact secrets before writing to Tier 1 KV (prevents leakage via prependContext)
    turn_summary = _redact_secrets(turn_summary)
    existing = await mem_store.get_session_context(_redis, session_id) or ""
    if not existing:
        await mem_store.set_session_context(_redis, session_id, turn_summary[:1500])
        return
    combined = f"{existing}\n---\n{turn_summary}"
    if len(combined) > 1200:
        # MemAgent-style incremental overwrite: LLM decides what to keep
        combined = await summarizer.overwrite_update(existing, turn_summary, target_chars=900)
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

    # Strip platform tags and cross_session_memory wrapping from the accumulated
    # session context before it is summarised and stored as a long-term episode.
    ctx = _strip_platform_noise(ctx)
    if not ctx or len(ctx) < 80:
        return {"status": "skipped", "reason": "session context empty after noise strip"}

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
        if _contains_secret(fact.content):
            continue
        f_emb    = _encode(fact.content)
        existing = await mem_store.knn_search(_redis, mem_store.FACT_KEY, f_emb, k=1)
        if existing and existing[0].get("score", 0.0) > 0.95:
            if fact.keywords and existing[0].get("score", 0.0) > 0.80:
                _spawn(_evolve_similar_fact(
                    existing[0].get("_element", ""),
                    fact.keywords, fact.topic,
                ))
            continue
        uid = await mem_store.save_fact(
            _redis, fact.content, fact.category, fact.confidence,
            f_emb, lang, domain,
            keywords=fact.keywords, persons=fact.persons, entities=fact.entities,
            importance=fact.importance, topic=fact.topic, location=fact.location,
            triple_s=fact.triple_s, triple_p=fact.triple_p, triple_o=fact.triple_o,
        )
        # Add to BM25 corpus
        attrs = {
            "content": fact.content, "category": fact.category,
            "language": lang, "domain": domain,
            "keywords": fact.keywords or [], "importance": fact.importance,
        }
        if fact.triple_s and fact.triple_p and fact.triple_o:
            attrs["triple_str"] = f"{fact.triple_s} | {fact.triple_p} | {fact.triple_o}"
        await _bm25_index.add(uid, fact.content, attrs)
        await persona_mod.update(_redis, fact.category, fact.content)
        fact_saved += 1

    # Delete Tier 1 KV — session has been crystallised into long-term memory
    await _redis.delete(f"{mem_store.SESSION_PRE}{session_id}:ctx")

    # Pin the session summary for the NEXT session's recall (cross-session handoff).
    # Strip platform tags before persisting — summaries must contain real content only.
    # Stored without TTL so it persists until the next compress overwrites it.
    clean_summary = _strip_platform_noise(session_summary)
    if clean_summary:
        await _redis.set(_PINNED_SESSION_KEY, clean_summary[:600].encode())

    log.info(f"[compress] session={session_id} promoted → ep+{ep_saved} facts+{fact_saved}")
    return {
        "status":     "ok",
        "session_id": session_id,
        "ep_saved":   ep_saved,
        "facts_saved":fact_saved,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/session/compact")
async def session_compact(req: CompactRequest):
    """
    Mid-session Tier 1 compression (v0.9.3 — inspired by claude-mem Endless Mode).

    Problem: Tier 1 session KV grows linearly with every turn accumulation.
    At ~1200 chars it is already LLM-compressed by _accumulate_session(), but
    across many tool uses / recall cycles the KV can still balloon to 3-5K chars.

    This endpoint re-compresses the KV on demand from a PostToolUse hook,
    keeping context complexity O(N) rather than O(N²).

    Idempotent: no-op if session KV is already below threshold_chars.
    """
    if not req.session_id:
        return {"status": "skipped", "reason": "no session_id"}

    ctx = await mem_store.get_session_context(_redis, req.session_id)
    if not ctx:
        return {"status": "skipped", "reason": "no session context"}

    size_before = len(ctx)
    if size_before <= req.threshold_chars:
        return {"status": "skipped", "reason": "below threshold",
                "size": size_before, "threshold": req.threshold_chars}

    # MemAgent-style chunked overwrite (arXiv:2507.02259 §3.1):
    # Split context into head (existing memory) + tail (new info to incorporate).
    # overwrite_update uses max_tokens=400 giving a proper ~900-char result,
    # unlike summarize()'s max_tokens=80 which truncates mid-sentence.
    # For very long contexts iterate over multiple chunks.
    _CHUNK = 1200
    if size_before > _CHUNK * 2:
        chunks = [ctx[i:i + _CHUNK] for i in range(0, len(ctx), _CHUNK)]
        memory = await summarizer.summarize(chunks[0])
        for chunk in chunks[1:]:
            memory = await summarizer.overwrite_update(memory, chunk, target_chars=1000)
        compacted = memory
    else:
        # Single chunk: split head/tail so overwrite_update has both parts in context
        mid = len(ctx) // 2
        compacted = await summarizer.overwrite_update(ctx[:mid], ctx[mid:], target_chars=1000)

    await mem_store.set_session_context(_redis, req.session_id, compacted[:1500])

    size_after = len(compacted)
    log.info(
        f"[compact] session={req.session_id} {size_before}→{size_after} chars "
        f"({round(100*(1-size_after/size_before))}% reduction)"
    )
    return {
        "status":        "ok",
        "session_id":    req.session_id,
        "size_before":   size_before,
        "size_after":    size_after,
        "reduction_pct": round(100 * (1 - size_after / size_before), 1),
    }


@app.post("/answer")
async def answer(req: AnswerRequest):
    """
    Extract a concise answer from recalled context using the LLM.

    Mirrors SimpleMem's AnswerGenerator (core/answer_generator.py):
    LLM reads the retrieved context and produces a short span answer,
    enabling token-F1 computation comparable to published LoCoMo baselines
    (which all use LLM-in-the-loop answer generation, not raw context F1).

    Returns {"answer": "<short span>"} or {"answer": ""} if LLM unavailable.
    """
    _AISERV_OAI = "http://127.0.0.1:4000/v1/chat/completions"
    _AISERV_KEY = "sk-aiserv-local-dev"

    if not req.context.strip():
        return {"answer": ""}

    def _parse_answer_raw(raw: str) -> str:
        """Parse JSON answer from LLM response, handling fences and both shapes."""
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        obj_start = raw.find("{")
        arr_start = raw.find("[")
        starts = [x for x in [obj_start, arr_start] if x != -1]
        if not starts:
            return ""
        start = min(starts)
        end_ch = "}" if raw[start] == "{" else "]"
        end = raw.rfind(end_ch)
        if end == -1:
            return ""
        parsed = json.loads(raw[start:end + 1])
        if isinstance(parsed, dict):
            return str(parsed.get("answer", ""))
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return str(parsed[0].get("answer", ""))
        return ""

    prompt = (
        f"Based only on the following memory context, answer the question concisely "
        f"using the exact phrase or name from the context (1-8 words preferred).\n\n"
        f"Context:\n{req.context[:2500]}\n\n"
        f"Question: {req.query}\n\n"
        f'Return JSON: {{"answer": "<exact phrase from context>"}}'
    )
    system_msg = ("You are a precise question-answering assistant. "
                  "Extract the exact answer phrase from the provided context. "
                  "Output valid JSON only.")

    # Two attempts: first with best QA model (qa role = stronger reasoning),
    # retry with fallback on timeout/failure.
    tried: list[str] = []
    exclude: str | None = None
    for attempt in range(2):
        model, _ = await extractor._resolve_qa_model(exclude=exclude)
        if model in tried:
            break
        tried.append(model)
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    _AISERV_OAI,
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_msg},
                            {"role": "user",   "content": prompt},
                        ],
                        "temperature": 0.0,
                        "max_tokens": 80,
                    },
                    headers={"Authorization": f"Bearer {_AISERV_KEY}"},
                )
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                answer = _parse_answer_raw(raw)
                if answer:
                    _spawn(extractor._report_quality(model, +1), "quality")
                    return {"answer": answer}
            exclude = model
        except httpx.TimeoutException:
            log.warning("[answer] %s timed out (attempt %d)", model, attempt + 1)
            _spawn(extractor._report_quality(model, -1, reason="timeout"), "quality")
            exclude = model
        except Exception as e:
            log.debug("[answer] %s failed: %s", model, e)
            _spawn(extractor._report_quality(model, -1, reason="other"), "quality")
            exclude = model
    return {"answer": ""}


@app.get("/health")
async def health():
    try:
        await _redis.ping()
        return {"status": "ok", "redis": "ok", "version": APP_VERSION,
                "embedding": embedder.get_provider_info()}
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

        # v0.9.7+: background writer observability (since process start).
        # Nested under "writer" so the dashboard can render it as its own panel
        # and the top-level stat list stays focused on memory tier counts.
        w_attempts = await _store_attempts.get()
        w_successes = await _store_successes.get()
        w_skips = await _store_skips.get()
        w_errors = await _store_errors.get()
        w_latency = await _store_latency_sum_ms.get()
        completed = w_successes + w_errors
        counts["writer"] = {
            "attempts": w_attempts,
            "successes": w_successes,
            "skips": w_skips,
            "errors": w_errors,
            "success_rate": round(w_successes / completed, 3) if completed > 0 else None,
            "avg_ms": round(w_latency / w_attempts) if w_attempts > 0 else None,
        }

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

    if _is_brief_option_reply(req.query):
        hint = (
            "<cross_session_memory>\n"
            "The user's latest message is a brief numeric option selection.\n"
            "Resolve it against the numbered options in the immediately preceding assistant message in this same conversation.\n"
            "Only ask for clarification if no numbered options were offered.\n"
            "</cross_session_memory>"
        )
        ms = int((time.time() - t0) * 1000)
        log.info(f"[recall] session={req.session_id} option-reply shortcut {ms}ms")
        return {"prependContext": hint, "latency_ms": ms}

    # Adaptive retrieval depth (SimpleMem-inspired C_q estimator)
    n_facts, n_episodes = _estimate_depth(req.query, req.memory_limit_number)

    # Build filter expressions for scene isolation
    lang_filter   = f'.language == "{q_lang}"'
    domain_filter = f'.domain == "{q_domain}"' if q_domain != "general" else None

    # SimpleMem symbolic filter: time-range hard constraint
    # Combines with lang_filter + domain_filter as a compound VSIM FILTER expression
    def _build_filter(base_filter: str | None, domain_f: str | None = None) -> str:
        parts = []
        if base_filter:
            parts.append(base_filter)
        if domain_f:
            parts.append(domain_f)
        if req.time_from is not None:
            parts.append(f".ts >= {req.time_from}")
        if req.time_to is not None:
            parts.append(f".ts <= {req.time_to}")
        return " && ".join(parts) if parts else ""

    lang_filter_sym = _build_filter(lang_filter, domain_filter)

    # Recall strategy: scene-filtered first, supplement with global if sparse
    async def _fetch(vset: str, k: int, fexpr=None):
        return await mem_store.knn_search(
            _redis, vset, emb, k, filter_expr=fexpr, bump_heat=True
        )

    # Detect if this is a capability query (needs tool context)
    is_cap_query = query_scene.get("is_capability_query", False)

    # Parallel: persona + env + session ctx + scene-filtered facts + episodes
    # + tool recall if capability query or include_tools=True
    # + symbolic structured pass (async, no extra latency via gather)
    # + pinned last-session summary (cross-session handoff bridge, index 6)
    gather_tasks = [
        persona_mod.get_context(_redis),
        cap_mod.get_env_context(_redis),
        mem_store.get_session_context(_redis, req.session_id),
        _fetch(mem_store.FACT_KEY,    n_facts,    lang_filter_sym or None),
        _fetch(mem_store.EPISODE_KEY, n_episodes, lang_filter_sym or None),
        retrieval_planner.analyze_query_structure(req.query) if (req.query.strip() and req.enable_planning) else asyncio.sleep(0),  # symbolic layer (index 5) — LLM call only when enable_planning=True
        _redis.get(_PINNED_SESSION_KEY),                       # handoff bridge (index 6)
    ]
    if req.include_tools and is_cap_query:
        gather_tasks.append(cap_mod.recall_tools(_redis, emb, k=6))
    else:
        gather_tasks.append(asyncio.sleep(0))
    # Procedure recall (index 7): fire in parallel if include_procedures=True
    # Uses raw VSIM (not knn_search) because procedures store 'task' not 'content'.
    async def _fetch_procs(embedding, k: int = 2) -> list[dict]:
        """Fetch top-k procedures with MACLA Beta posterior scoring (v0.9.5).

        Beta(success+1, fail+1) × cosine_sim reranks procedures by both
        relevance AND historical success rate (arXiv:2512.18950).
        """
        blob = embedding.astype("float32").tobytes()
        # Fetch more than needed so Beta reranking can reorder
        cmd  = ["VSIM", mem_store.PROC_KEY, "FP32", blob,
                "COUNT", max(k * 3, 6), "WITHSCORES", "WITHATTRIBS"]
        try:
            results = await _redis.execute_command(*cmd)
        except Exception:
            return []
        procs = []
        idx = 0
        while idx + 2 < len(results):
            elem = results[idx]; score = results[idx+1]; raw = results[idx+2]
            idx += 3
            elem_str = elem.decode() if isinstance(elem, bytes) else elem
            if elem_str == "__seed__":
                continue
            try:
                attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            except Exception:
                continue
            if attrs.get("_seed") or not attrs.get("task"):
                continue
            cosine = float(score) if score else 0.0
            # MACLA Beta posterior: E[Beta(α,β)] = α/(α+β) with α=success+1, β=fail+1
            success = attrs.get("success_count", 0)
            fail    = attrs.get("fail_count", 0)
            total   = success + fail
            beta_mean = (success + 1) / (total + 2) if total >= 3 else 1.0
            procs.append({
                "task":          attrs.get("task", ""),
                "procedure":     attrs.get("procedure", ""),
                "_adjusted_score": cosine * beta_mean,
                "_cosine":       cosine,
            })
        # Re-sort by Beta-weighted score
        procs.sort(key=lambda x: x["_adjusted_score"], reverse=True)
        return procs[:k]

    if req.include_procedures:
        gather_tasks.append(_fetch_procs(emb))
    else:
        gather_tasks.append(asyncio.sleep(0))
    
    # Crystallized digests (index 8): fetch recent auto-crystallized session summaries
    async def _fetch_crystallized_digests() -> list[dict]:
        """Fetch top-3 most recent crystallized session digests for lessons learned."""
        try:
            pattern = "mem:crystallized:*"
            cursor = 0
            all_keys = []

            while True:
                cursor, keys = await _redis.scan(cursor, match=pattern, count=50)
                all_keys.extend(keys)
                if cursor == 0 or len(all_keys) >= 10:
                    break

            if not all_keys:
                return []

            # Batch fetch all values via pipeline (avoids N+1 round-trips)
            pipe = _redis.pipeline(transaction=False)
            for key in all_keys:
                pipe.get(key)
            raw_values = await pipe.execute()

            digests = []
            for raw in raw_values:
                if not raw:
                    continue
                try:
                    digest = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                    digests.append(digest)
                except Exception:
                    continue

            # Sort by crystallized_at (most recent first), return top 3
            digests.sort(key=lambda d: d.get("crystallized_at", 0), reverse=True)
            return digests[:3]
        
        except Exception as e:
            log.warning(f"[recall] failed to fetch crystallized digests: {e}")
            return []
    
    gather_tasks.append(_fetch_crystallized_digests())

    results = await asyncio.gather(*gather_tasks)
    persona_ctx  = results[0]
    env_ctx      = results[1]
    session_ctx  = results[2]
    facts_scene  = results[3]
    eps_scene    = results[4]
    query_struct = results[5] if isinstance(results[5], dict) else {}
    _pinned_raw  = results[6]
    last_session_summary = (
        _pinned_raw.decode() if isinstance(_pinned_raw, bytes) else _pinned_raw
    ) if _pinned_raw else None
    tools_raw    = results[7] if isinstance(results[7], list) else []
    procs_raw    = results[8] if isinstance(results[8], list) else []
    crystal_digests = results[9] if isinstance(results[9], list) else []

    # Symbolic structured pass (AgentMem §3.3 extension):
    # If the query mentions specific persons or entities, do a targeted scan
    # for facts that explicitly contain those names — catches cases where
    # semantic similarity is low but the name is a direct match.
    sym_facts: list[dict] = []
    sym_entities = (query_struct.get("persons") or []) + (query_struct.get("entities") or [])
    if sym_entities:
        sym_tasks = []
        for ent in sym_entities[:3]:   # cap at 3 to bound latency
            sym_tasks.append(
                mem_store.knn_search(_redis, mem_store.FACT_KEY, _encode(ent), k=3,
                                     filter_expr=lang_filter_sym or None)
            )
        sym_batches = await asyncio.gather(*sym_tasks)
        seen_sym = {f["content"] for f in facts_scene}
        for batch in sym_batches:
            for item in (batch if isinstance(batch, list) else []):
                if item["content"] not in seen_sym:
                    sym_facts.append(item)
                    seen_sym.add(item["content"])
        if sym_facts:
            log.debug(f"[recall] symbolic pass: +{len(sym_facts)} facts for entities {sym_entities[:3]}")

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

    # Dynamic Weighted RRF (arXiv:2511.18194 wRRF):
    # Adjust per-source weights based on what the query structure reveals.
    # Entity/person queries → boost symbolic + BM25 passes (exact name matches).
    # Temporal queries → boost symbolic (timestamps stored there).
    # Pure semantic queries → reduce symbolic/BM25 weight (no metadata to match).
    _has_entities = bool(sym_entities)
    _has_temporal  = bool(query_struct.get("time_expression"))
    if _has_entities and _has_temporal:
        _rrf_weights = [0.8, 0.8, 1.4, 1.2]   # strong symbolic + BM25 boost
    elif _has_entities:
        _rrf_weights = [0.9, 0.9, 1.2, 1.1]   # moderate symbolic + BM25 boost
    elif _has_temporal:
        _rrf_weights = [0.9, 0.9, 1.2, 0.8]   # symbolic boost, light BM25
    else:
        _rrf_weights = [1.0, 1.0, 0.6, 0.9]   # semantic: moderate BM25, low symbolic

    # Memori BM25 pass (arXiv:2603.19935): exact-term matching over fact corpus.
    # Runs in O(1) (no I/O) — BM25 corpus is maintained in-memory and searched locally.
    # Particularly effective for entity names, dates, tool names, specific terms.
    bm25_facts = await _bm25_index.search(req.query, k=n_facts)
    if bm25_facts:
        log.debug(f"[bm25] {len(bm25_facts)} hits for: {req.query[:60]!r}")

    def _merge(primary, supplement, symbolic, limit):
        p_ranked   = heat_mod.heat_rerank(primary)
        s_ranked   = heat_mod.heat_rerank(supplement) if supplement else []
        sym_ranked = heat_mod.heat_rerank(symbolic)   if symbolic   else []
        # BM25 hits are already scored by TF-IDF; pass as-is (no heat reranking)
        fused = _rrf_merge(
            [p_ranked, s_ranked, sym_ranked, bm25_facts],
            weights=_rrf_weights,
            limit=limit * 2,
        )
        return fused[:limit]

    facts    = _merge(facts_scene, facts_global, sym_facts,  n_facts)
    episodes = _merge(eps_scene,   eps_global,   [],         n_episodes)

    # SimpleMem Phase 3: lexical keyword boost (BM25-lite additive layer)
    # Note: BM25 above handles full TF-IDF scoring; this adds a lightweight keyword
    # overlap boost on top for any residual exact-match signal not captured by BM25.
    facts    = _keyword_boost(facts,    req.query)
    episodes = _keyword_boost(episodes, req.query, boost=0.03)

    # AgentMem advantage: importance-weighted reranking.
    # High-importance facts (rules, identity, preferences) surface above transient context.
    facts    = _importance_boost(facts)
    episodes = _importance_boost(episodes, weight=0.05)  # lighter for episodes

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

    # Knowledge graph expansion (v0.9.0 + auto-trigger v1.0):
    # Fire when explicitly requested OR when query has named entities (auto_graph=True).
    # Auto-trigger catches "what did Bob tell me about Redis?" without needing include_graph.
    _auto_graph_trigger = req.auto_graph and bool(sym_entities)
    if (req.include_graph or _auto_graph_trigger) and facts:
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

    # AriadneMem Steiner tree bridge discovery (arXiv:2603.03290 §3.3):
    # Find facts that structurally bridge pairs of retrieved terminal facts.
    # Fires when include_graph=True and we have ≥2 retrieved facts.
    # Deterministic — no extra LLM call, single graph+vector pass.
    if (req.include_graph or _auto_graph_trigger) and len(facts) >= 2:
        try:
            bridge_facts = await graph_mod.find_bridge_nodes(_redis, facts, emb, k=3)
            if bridge_facts:
                seen_bridge = {f["content"] for f in facts}
                for bf in bridge_facts:
                    if bf["content"] not in seen_bridge:
                        facts.append(bf)
                        seen_bridge.add(bf["content"])
                facts = facts[:n_facts + 5]
                log.debug(f"[recall] bridge discovery: +{len(bridge_facts)} bridge nodes")
        except Exception as _be:
            log.debug(f"[recall] bridge discovery skipped: {_be}")

    # AutoTool TIG: get transition hints from most-used recalled tool (v0.9.5)
    tig_hints: list[str] = []
    if tools_raw:
        top_tool_elem = tools_raw[0].get("_element", "")
        if top_tool_elem:
            try:
                all_tig = await _redis.hgetall(mem_store.TOOL_GRAPH_KEY)
                prefix  = f"{top_tool_elem}:"
                trans: dict[str, int] = {}
                for k_b, v_b in all_tig.items():
                    key_s = k_b.decode() if isinstance(k_b, bytes) else k_b
                    if key_s.startswith(prefix):
                        target = key_s[len(prefix):]
                        trans[target] = int(v_b)
                if trans:
                    total_trans = sum(trans.values())
                    sorted_trans = sorted(trans.items(), key=lambda x: x[1], reverse=True)
                    tig_hints = [
                        f"{t} ({round(c/total_trans*100)}%)"
                        for t, c in sorted_trans[:3]
                        if c / total_trans >= 0.15  # only show ≥15% probability edges
                    ]
            except Exception:
                pass

    # Format tool context (only include env if it's non-trivial)
    tool_ctx = cap_mod.format_tool_context(tools_raw, tig_hints=tig_hints or None) if tools_raw else ""
    env_ctx_formatted = env_ctx if env_ctx and is_cap_query else ""

    # proc_ctx: already decoded by _fetch_procs (list of {task, procedure} dicts)
    proc_ctx: list[dict] = procs_raw if isinstance(procs_raw, list) else []

    # SimpleMem Section 3.3 token-budget injection with XML wrapper
    # Skip pinned session summary if this session has already accumulated context
    # (session_ctx means we're mid-session, not starting fresh).
    _handoff = last_session_summary if not session_ctx else None
    prepend = _format_prepend(
        facts, episodes, session_ctx, persona_ctx,
        env_ctx=env_ctx_formatted, tool_ctx=tool_ctx,
        proc_ctx=proc_ctx or None,
        token_budget=req.token_budget,
        last_session_summary=_handoff,
        crystal_digests=crystal_digests or None,  # Auto-crystallized lessons learned
    )
    ms = int((time.time() - t0) * 1000)

    planning_info = f" planning={req.enable_planning}" if req.enable_planning else ""
    log.info(f"[recall] session={req.session_id} lang={q_lang} domain={q_domain} "
             f"facts={len(facts)} ep={len(episodes)} tools={len(tools_raw)} procs={len(proc_ctx)}"
             f"{planning_info} {ms}ms")

    return {"prependContext": prepend if prepend else None, "latency_ms": ms}


@app.post("/store")
async def store_memory(req: StoreRequest):
    _spawn(_do_store(req.messages, req.session_id), "store")
    await _ws_broadcast("store", {"session_id": req.session_id})
    return {"status": "queued"}


# ── Session Tier-1 endpoints (v0.7.0) ─────────────────────────────────────────

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

    # Also populate the Agent Self-Model (mem:agent) with identity fields so the
    # dashboard's "Agent Self-Model" panel isn't perpetually empty. set_agent()
    # stamps last_seen automatically; we only forward what was actually provided.
    agent_data: dict = {}
    if req.agent_model:   agent_data["model"]      = req.agent_model
    if req.agent_version: agent_data["version"]    = req.agent_version
    if req.session_id:    agent_data["session_id"] = req.session_id
    if req.runtime:       agent_data["runtime"]    = req.runtime
    if agent_data:
        await cap_mod.set_agent(_redis, agent_data)

    log.info(f"[env] registered env: {list(env_data.keys())}")
    return {"status": "ok", "fields": list(env_data.keys()), "agent_fields": list(agent_data.keys())}


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


# ── ToolMem feedback endpoint (v0.9.5) ────────────────────────────────────────

@app.post("/tool-feedback")
async def tool_feedback(req: ToolFeedbackRequest):
    """
    Record success/failure for a tool invocation (ToolMem arXiv:2510.06664).

    Called from PostToolUse hook after each tool use. Updates success_count /
    fail_count on the tool's mem:tools entry and refreshes capability_summary
    once enough data is available (≥5 uses).

    Example: {"tool_name": "web_search_prime", "success": true}
    """
    elem_key = req.tool_name.lower().replace(" ", "_").replace("/", "_")[:64]
    try:
        raw = await _redis.execute_command("VGETATTR", cap_mod.TOOL_KEY, elem_key)
    except Exception:
        return {"ok": False, "reason": "redis error"}
    if not raw:
        return {"ok": False, "reason": "tool not found"}

    try:
        attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    except Exception:
        return {"ok": False, "reason": "parse error"}

    if req.success:
        attrs["success_count"] = attrs.get("success_count", 0) + 1
    else:
        attrs["fail_count"]    = attrs.get("fail_count", 0) + 1
    attrs["use_count"] = attrs.get("use_count", 0) + 1

    # Refresh capability_summary once ≥5 feedback data points exist
    total = attrs.get("success_count", 0) + attrs.get("fail_count", 0)
    if total >= 5:
        rate = attrs.get("success_count", 0) / total
        if   rate >= 0.85: quality = "reliable"
        elif rate >= 0.60: quality = "mostly reliable"
        elif rate >= 0.40: quality = "mixed"
        else:              quality = "often fails"
        attrs["capability_summary"] = f"{quality} ({attrs.get('success_count',0)}/{total}✓)"

    try:
        await _redis.execute_command("VSETATTR", cap_mod.TOOL_KEY, elem_key, json.dumps(attrs))
    except Exception as e:
        return {"ok": False, "reason": str(e)}

    log.debug(f"[tool-feedback] {elem_key} success={req.success} "
              f"s={attrs.get('success_count',0)} f={attrs.get('fail_count',0)}")
    return {
        "ok":           True,
        "tool":         elem_key,
        "success_count": attrs.get("success_count", 0),
        "fail_count":    attrs.get("fail_count", 0),
        "capability_summary": attrs.get("capability_summary", ""),
    }


# ── AutoTool TIG endpoints (v0.9.5) ───────────────────────────────────────────

@app.post("/record-tool-sequence")
async def record_tool_sequence(req: ToolSequenceRequest):
    """
    Record an ordered tool-use sequence into the Tool Inertia Graph (AutoTool arXiv:2511.14650).

    Called from Stop hook after each session. For each consecutive pair (A→B),
    increments HINCRBY on mem:tool_graph key "A:B". These transition counts
    are used at recall time to suggest likely-next tools.

    Example: {"sequence": ["Glob", "Read", "Edit", "Bash"]}
    → increments Glob:Read, Read:Edit, Edit:Bash
    """
    seq = [t.strip() for t in req.sequence if t.strip()]
    if len(seq) < 2:
        return {"ok": False, "transitions": 0}
    count = 0
    for i in range(len(seq) - 1):
        a = seq[i].lower().replace(" ", "_")[:64]
        b = seq[i + 1].lower().replace(" ", "_")[:64]
        if a == b:
            continue
        await _redis.hincrby(mem_store.TOOL_GRAPH_KEY, f"{a}:{b}", 1)
        count += 1
    log.debug(f"[tig] recorded {count} transitions for session={req.session_id}")
    return {"ok": True, "transitions": count}


@app.get("/tool-graph/{tool_name}")
async def tool_graph(tool_name: str, k: int = 5):
    """
    Return the top-k outgoing transitions from tool_name in the Tool Inertia Graph.

    Example: GET /tool-graph/glob → [{"next": "read", "count": 42, "prob": 0.72}]
    """
    elem_key = tool_name.lower().replace(" ", "_")[:64]
    try:
        all_entries = await _redis.hgetall(mem_store.TOOL_GRAPH_KEY)
    except Exception:
        return {"tool": tool_name, "transitions": []}
    prefix = f"{elem_key}:"
    trans: dict[str, int] = {}
    for k_b, v_b in all_entries.items():
        key_s = k_b.decode() if isinstance(k_b, bytes) else k_b
        if key_s.startswith(prefix):
            target = key_s[len(prefix):]
            trans[target] = int(v_b)
    total = sum(trans.values())
    sorted_t = sorted(trans.items(), key=lambda x: x[1], reverse=True)[:k]
    return {
        "tool":        tool_name,
        "transitions": [
            {"next": t, "count": c, "prob": round(c / total, 2) if total else 0}
            for t, c in sorted_t
        ],
        "total_transitions": total,
    }


# ── AWO meta-tool detection (v0.9.6) ──────────────────────────────────────────

@app.post("/tool-graph/detect-meta-tools")
async def detect_meta_tools(threshold: int = 5, background_tasks: BackgroundTasks = None):
    """
    AWO meta-tool detection (Autonomous Workflow Optimization arXiv:2601.22037).

    Scans the Tool Inertia Graph for high-frequency 2-hop chains A→B→C
    where both A:B and B:C have count ≥ threshold. Each discovered chain is
    auto-synthesized into a composite procedure in mem:procedures so future
    recall surfaces it as a recommended workflow pattern.

    Example: Glob→Read (28) + Read→Edit (23) → synthesize "Glob-Read-Edit pattern"
    Returns: {"detected": N, "chains": [...], "new_procedures": M}
    """
    try:
        all_entries = await _redis.hgetall(mem_store.TOOL_GRAPH_KEY)
    except Exception:
        return {"detected": 0, "chains": [], "new_procedures": 0}

    if not all_entries:
        return {"detected": 0, "chains": [], "new_procedures": 0}

    # Build adjacency: from_tool → {to_tool: count}
    adj: dict[str, dict[str, int]] = {}
    for k_b, v_b in all_entries.items():
        key_s = k_b.decode() if isinstance(k_b, bytes) else k_b
        val   = int(v_b) if v_b else 0
        if ":" not in key_s:
            continue
        a, b = key_s.split(":", 1)
        adj.setdefault(a, {})[b] = val

    # Find 2-hop chains: A→B (≥ threshold) + B→C (≥ threshold), A≠C
    chains: list[dict] = []
    for a, a_neighbors in adj.items():
        for b, ab_count in a_neighbors.items():
            if ab_count < threshold:
                continue
            b_neighbors = adj.get(b, {})
            for c, bc_count in b_neighbors.items():
                if bc_count < threshold or c == a:
                    continue
                chains.append({
                    "chain":    [a, b, c],
                    "ab_count": ab_count,
                    "bc_count": bc_count,
                    "strength": min(ab_count, bc_count),  # bottleneck strength
                })

    # Sort by bottleneck strength descending
    chains.sort(key=lambda x: x["strength"], reverse=True)

    if not chains:
        return {"detected": 0, "chains": [], "new_procedures": 0}

    async def _synthesize_chains(chains_to_add: list[dict]):
        new_count = 0
        for ch in chains_to_add[:20]:   # cap at 20 new meta-tools per run
            a, b, c = ch["chain"]
            task = f"[meta-tool] {a} → {b} → {c} workflow"
            procedure = (
                f"High-frequency tool chain discovered by AWO analysis "
                f"(TIG count: {a}→{b}={ch['ab_count']}, {b}→{c}={ch['bc_count']}).\n"
                f"1. Use {a}\n2. Use {b}\n3. Use {c}"
            )
            emb = _encode(task)
            # Check for near-duplicate before storing
            existing = await mem_store.knn_search(_redis, mem_store.PROC_KEY, emb, k=1)
            if existing and existing[0].get("score", 0.0) > 0.90:
                continue
            await mem_store.save_procedure(
                _redis,
                task=task,
                procedure=procedure,
                embedding=emb,
                tools_used=[a, b, c],
                domain="meta",
                language="en",
            )
            new_count += 1
        log.info(f"[awo] synthesized {new_count} meta-tool procedures from {len(chains_to_add)} chains")

    if background_tasks is not None:
        background_tasks.add_task(_synthesize_chains, chains)
        synthesis_status = "queued"
        new_procs = None
    else:
        await _synthesize_chains(chains)
        synthesis_status = "done"
        new_procs = min(len(chains), 20)

    return {
        "detected":       len(chains),
        "chains":         chains[:10],   # return top-10 for inspection
        "new_procedures": new_procs,
        "synthesis":      synthesis_status,
        "threshold":      threshold,
    }


# ── MACLA procedure feedback endpoint (v0.9.5) ────────────────────────────────

@app.post("/procedure-feedback")
async def procedure_feedback(req: ProcedureFeedbackRequest):
    """
    Record success/failure for a procedure retrieval (MACLA arXiv:2512.18950).

    Finds the best matching procedure by semantic similarity to task_prefix,
    then updates its fail_count. (success_count is already bumped by /store-procedure.)
    Used by agents after a recalled procedure was followed and succeeded/failed.

    Example: {"task_prefix": "how to search files by pattern", "success": false}
    """
    emb = _encode(req.task_prefix[:80])
    blob = emb.astype("float32").tobytes()
    try:
        results = await _redis.execute_command(
            "VSIM", mem_store.PROC_KEY, "FP32", blob,
            "COUNT", 3, "WITHSCORES", "WITHATTRIBS"
        )
    except Exception:
        return {"ok": False, "reason": "redis error"}

    best_elem = None
    best_score = 0.0
    best_attrs: dict = {}
    idx = 0
    while idx + 2 < len(results):
        elem  = results[idx]; score = results[idx+1]; raw = results[idx+2]
        idx += 3
        elem_str = elem.decode() if isinstance(elem, bytes) else elem
        if elem_str == "__seed__":
            continue
        try:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
            continue
        if attrs.get("_seed") or not attrs.get("task"):
            continue
        s = float(score) if score else 0.0
        if s > best_score:
            best_score = s; best_elem = elem_str; best_attrs = attrs

    if not best_elem or best_score < 0.50:
        return {"ok": False, "reason": "no matching procedure found", "best_score": best_score}

    if req.success:
        best_attrs["success_count"] = best_attrs.get("success_count", 0) + 1
    else:
        best_attrs["fail_count"]    = best_attrs.get("fail_count", 0) + 1

    try:
        await _redis.execute_command("VSETATTR", mem_store.PROC_KEY, best_elem, json.dumps(best_attrs))
    except Exception as e:
        return {"ok": False, "reason": str(e)}

    log.debug(f"[proc-feedback] {best_elem[:40]} success={req.success} score={best_score:.2f}")
    return {
        "ok":           True,
        "matched_task": best_attrs.get("task", "")[:60],
        "score":        best_score,
        "success_count": best_attrs.get("success_count", 0),
        "fail_count":    best_attrs.get("fail_count", 0),
    }


@app.get("/tool-procedures/{tool_name}")
async def tool_procedures(tool_name: str):
    """
    Return all procedures that use a given tool (reverse index lookup).

    Uses mem:proc_by_tool:<tool> Redis Sets populated by save_procedure()
    and the /proc-backfill-index endpoint.

    Example: GET /tool-procedures/bash
    → [{"uid": "...", "task": "...", "tools_used": [...], "success_count": N}]
    """
    uids = await mem_store.get_procs_for_tool(_redis, tool_name)
    if not uids:
        return {"tool": tool_name, "procedures": [], "count": 0}

    procs = []
    for uid in uids:
        try:
            raw = await _redis.execute_command("VGETATTR", mem_store.PROC_KEY, uid)
        except Exception:
            continue
        if not raw:
            continue
        try:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
            continue
        if not attrs.get("task"):
            continue
        procs.append({
            "uid":           uid,
            "task":          attrs.get("task", ""),
            "procedure":     attrs.get("procedure", "")[:200],
            "tools_used":    attrs.get("tools_used", []),
            "domain":        attrs.get("domain", ""),
            "success_count": attrs.get("success_count", 1),
            "fail_count":    attrs.get("fail_count", 0),
        })

    return {"tool": tool_name, "procedures": procs, "count": len(procs)}


@app.post("/proc-backfill-index")
async def proc_backfill_index(background_tasks: BackgroundTasks):
    """
    Backfill mem:proc_by_tool reverse index for all existing procedures.

    Run once after upgrading to v0.9.6 to populate the reverse index for
    procedures that were stored before link_proc_to_tools() was added.
    Idempotent — SADD is a no-op for already-present members.
    """
    async def _do_backfill():
        procs = await mem_store.scan_all_procedures(_redis)
        count = 0
        for p in procs:
            uid   = p.get("uid", "")
            tools = p.get("tools_used", [])
            if uid and tools:
                await mem_store.link_proc_to_tools(_redis, uid, tools)
                count += len(tools)
        log.info(f"[backfill] reverse-indexed {len(procs)} procedures, {count} tool→proc links")
        return len(procs), count

    background_tasks.add_task(_do_backfill)
    return {"status": "queued", "message": "backfilling proc_by_tool reverse index in background"}


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
    SimpleMem 3-phase consolidation (Section 3.2) + LLM Wiki v2 Ebbinghaus decay:
      Phase 1 — Decay:  Ebbinghaus forgetting curve — effective confidence decays
                        based on category stability and time since last confirmation.
                        importance × 0.9 for entries older than 90 days (legacy).
      Phase 2 — Merge:  cluster near-duplicates (cosine×temporal ≥ threshold),
                        LLM-merge into keeper, soft-delete losers via superseded_by
                        (reason="merged"). Merged keeper inherits source_count sum.
      Phase 3 — Prune:  soft-delete any active entry with importance < 0.05
                        (reason="pruned") or effective confidence < 0.1.
    """
    t0 = time.time()
    now_ms = int(time.time() * 1000)
    NINETY_DAYS_MS = 90 * 86_400_000

    scanned = await _vscan(_redis, mem_store.FACT_KEY, max_count=500)
    if not scanned:
        return {"merged": 0, "decayed": 0, "pruned": 0, "ms": 0}

    # Filter only active (non-superseded) facts with content
    all_facts = [
        {"element": item["element"], "attrs": item["attrs"]}
        for item in scanned
        if item["attrs"].get("content") and not item["attrs"].get("superseded_by")
    ]

    if len(all_facts) < 2:
        return {"merged": 0, "decayed": 0, "pruned": 0, "total": len(all_facts), "ms": 0}

    # ── Phase 1: Decay (Ebbinghaus + legacy importance) ─────────────────────────
    # LLM Wiki v2: Ebbinghaus forgetting curve — confidence decays based on
    # category stability and time since last confirmation.
    # Legacy: entries older than 90 days lose 10% importance each run.
    decayed_count = 0
    for fact in all_facts:
        changed = False
        ts = fact["attrs"].get("ts", now_ms)
        # Legacy importance decay
        if (now_ms - ts) > NINETY_DAYS_MS:
            old_imp = fact["attrs"].get("importance", 0.5)
            new_imp = round(old_imp * 0.9, 4)
            fact["attrs"]["importance"] = new_imp
            changed = True
        # LLM Wiki v2: compute effective confidence and store it
        eff_conf = mem_store.confidence_decay(fact["attrs"])
        if eff_conf != fact["attrs"].get("effective_confidence", -1):
            fact["attrs"]["effective_confidence"] = eff_conf
            changed = True
        if changed:
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

    # Precompute all fact embeddings in batch to avoid re-encoding inside the loop.
    # This reduces Phase 2 from O(N²) encoding cost to O(N) — each fact's embedding
    # is computed once and reused for both knn_search and keeper re-embedding.
    fact_embeddings: dict[str, np.ndarray] = {}
    for fact in all_facts:
        fact_embeddings[fact["element"]] = _encode(fact["attrs"]["content"])

    # Build a lookup dict for O(1) element → fact mapping (replaces O(N) scan)
    fact_by_element: dict[str, dict] = {f["element"]: f for f in all_facts}

    for fact in all_facts:
        if fact["element"] in superseded_elements:
            continue

        f_emb = fact_embeddings[fact["element"]]
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
            # O(1) lookup instead of O(N) scan
            s_fact = fact_by_element.get(s["_element"])
            s_ts = s_fact["attrs"].get("ts", 0) if s_fact else 0

            days_between = abs(fact_ts - s_ts) / 86_400_000
            temporal_factor = math.exp(-temporal_lambda * days_between)
            affinity = cosine_sim * temporal_factor

            if affinity >= similarity_threshold and s_fact:
                cluster.append(s_fact)

        if len(cluster) < 2:
            continue

        contents = [c["attrs"]["content"] for c in cluster]
        merged_content = await _llm_merge_facts(contents)
        if not merged_content:
            continue

        # Keeper = highest importance in cluster (aligned with SimpleMem's winner
        # selection policy: importance beats recency). Ties broken by newer timestamp.
        cluster.sort(
            key=lambda c: (c["attrs"].get("importance", 0.5), c["attrs"].get("ts", 0)),
            reverse=True
        )
        keeper = cluster[0]
        keeper_element = keeper["element"]

        new_attrs = dict(keeper["attrs"])
        new_attrs["content"] = merged_content[:500]
        new_attrs["access_count"] = max(c["attrs"].get("access_count", 0) for c in cluster)
        new_attrs["importance"] = max(c["attrs"].get("importance", 0.5) for c in cluster)
        new_attrs["consolidated_from"] = len(cluster)
        # LLM Wiki v2: inherit source_count sum and reset last_confirmed_ts
        new_attrs["source_count"] = sum(c["attrs"].get("source_count", 1) for c in cluster)
        new_attrs["last_confirmed_ts"] = int(time.time() * 1000)
        new_attrs["version"] = keeper["attrs"].get("version", 1) + 1

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
                await mem_store.soft_delete_fact(_redis, c["element"], keeper_element, reason="merged")
                superseded_elements.add(c["element"])

        merged_count += 1

    # ── Phase 3: Prune ─────────────────────────────────────────────────────────
    # Soft-delete any remaining active entry whose importance has decayed below 0.05
    # or whose effective confidence (Ebbinghaus) has fallen below 0.1.
    pruned_count = 0
    for fact in all_facts:
        if fact["element"] in superseded_elements:
            continue
        imp = fact["attrs"].get("importance", 1.0)
        eff_conf = fact["attrs"].get("effective_confidence", 1.0)
        if imp < 0.05 or eff_conf < 0.1:
            reason = "pruned" if imp < 0.05 else "confidence_expired"
            await mem_store.soft_delete_fact(_redis, fact["element"], "pruned", reason=reason)
            superseded_elements.add(fact["element"])
            pruned_count += 1

    # ── Post-consolidation: invalidate BM25 index ─────────────────────────────
    # Superseded and merged facts changed state; the in-memory BM25 corpus is now
    # stale. Force a rebuild on next search so superseded facts are excluded and
    # merged keepers reflect their new content.
    if superseded_elements:
        await _bm25_index.reset()
        _spawn(_populate_bm25_from_redis(_redis), "bm25-rebuild")

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


async def _do_hard_prune() -> dict:
    """
    Physically VREM entries that have been soft-deleted for > 7 days,
    and stale episodes (>180 days old, never accessed).

    Soft-delete (superseded_by != "") is the SimpleMem audit trail.
    Hard-delete (VREM) is the capacity management pass that actually
    reduces vectorset size and keeps HNSW index fresh.

    Returns counts of entries removed from each vectorset.
    """
    t0 = time.time()
    now_ms = int(time.time() * 1000)
    SEVEN_DAYS_MS   = 7  * 86_400_000
    SIX_MONTHS_MS   = 180 * 86_400_000

    removed_facts = 0
    removed_eps   = 0

    async def _hard_prune_vset(vset_key: str, max_scan: int = 2000) -> int:
        items = await _vscan(_redis, vset_key, max_count=max_scan)
        if not items:
            return 0

        to_remove: list[str] = []
        for item in items:
            attrs = item["attrs"]
            ts = attrs.get("ts", now_ms)
            age_ms = now_ms - ts

            # Hard-delete soft-deleted entries older than 7 days
            superseded = attrs.get("superseded_by", "")
            if superseded and age_ms > SEVEN_DAYS_MS:
                to_remove.append(item["element"])
                continue

            # Hard-delete stale episodes: 180+ days old, never recalled
            if vset_key == mem_store.EPISODE_KEY:
                if age_ms > SIX_MONTHS_MS and attrs.get("access_count", 0) == 0:
                    to_remove.append(item["element"])

        removed = 0
        for elem_str in to_remove:
            try:
                await _redis.execute_command("VREM", vset_key, elem_str)
                removed += 1
            except Exception:
                pass
        return removed

    removed_facts = await _hard_prune_vset(mem_store.FACT_KEY)
    removed_eps   = await _hard_prune_vset(mem_store.EPISODE_KEY)

    # Invalidate BM25 index after hard-prune (entries physically removed)
    if removed_facts > 0:
        await _bm25_index.reset()
        _spawn(_populate_bm25_from_redis(_redis), "bm25-rebuild")

    ms = int((time.time() - t0) * 1000)
    log.info(f"[hard_prune] removed facts={removed_facts} episodes={removed_eps} {ms}ms")
    return {"removed_facts": removed_facts, "removed_episodes": removed_eps, "ms": ms}


async def _llm_merge_facts(contents: list[str]) -> str | None:
    """Use role-based routing to merge multiple similar facts into one consolidated fact."""
    _AISERV_OAI = "http://127.0.0.1:4000/v1/chat/completions"
    _AISERV_KEY = "sk-aiserv-local-dev"

    # Use live health matrix — no hardcoded dead models.
    model, _ = await extractor._resolve_nlp_model()

    facts_text = "\n".join(f"- {c}" for c in contents)
    prompt = (
        f"以下是关于同一主题的多条记忆，请合并为一条完整、准确的事实陈述。"
        f"保留所有不同的细节，去除重复。输出仅包含合并后的一句话，不要解释。\n\n{facts_text}"
    )

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                _AISERV_OAI,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                    "temperature": 0.1,
                },
                headers={"Authorization": f"Bearer {_AISERV_KEY}"},
            )
            if resp.status_code == 200:
                _spawn(extractor._report_quality(model, +1), "quality")
                return resp.json()["choices"][0]["message"]["content"].strip()
    except httpx.TimeoutException:
        log.warning("[consolidate] %s timed out", model)
        _spawn(extractor._report_quality(model, -1, reason="timeout"), "quality")
    except Exception as e:
        log.warning(f"[consolidate] LLM merge failed: {e}")
        _spawn(extractor._report_quality(model, -1, reason="other"), "quality")

    return max(contents, key=len)


@app.post("/consolidate")
async def consolidate(background_tasks: BackgroundTasks):
    """Trigger memory consolidation — merges similar facts to reduce redundancy."""
    background_tasks.add_task(_do_consolidate)
    await _ws_broadcast("consolidate", {"status": "queued"})
    return {"status": "consolidation_queued"}


@app.post("/consolidate/sync")
async def consolidate_sync():
    """Synchronous consolidation — returns results immediately."""
    result = await _do_consolidate()
    return result


@app.post("/consolidate/hard-prune")
async def hard_prune(background_tasks: BackgroundTasks):
    """
    Physical VREM of soft-deleted entries (>7 days) and stale episodes (>180 days).
    Runs automatically every 24h; use this to trigger on demand.
    """
    background_tasks.add_task(_do_hard_prune)
    return {"status": "hard_prune_queued"}


@app.post("/admin/delete-facts-by-content")
async def admin_delete_facts_by_content(pattern: str = "Always send a report"):
    """Delete facts whose content contains the given pattern (admin only)."""
    card = await _redis.execute_command("VCARD", mem_store.FACT_KEY)
    if not card or int(card) <= 1:
        return {"deleted": 0, "message": "no facts to scan"}

    seed = np.zeros(embedder.DIMS, dtype=np.float32)
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


# ── User Feedback Endpoints (v1.2.0) ──────────────────────────────────────────
# Enable users to rate, pin, and delete memories for continuous quality improvement


@app.post("/feedback")
async def provide_feedback(req: FeedbackRequest):
    """
    User rates memory relevance (1-5 stars). Adjusts importance and confidence.
    
    - Rating 4-5: boost importance × 1.2, reinforce fact (increment source_count)
    - Rating 1-2: reduce importance × 0.7, flag for review
    - Rating 3: no change
    
    This creates a reinforcement learning loop where the system learns from
    user preferences over time.
    """
    if req.rating < 1 or req.rating > 5:
        return {"error": "rating must be 1-5"}
    
    try:
        raw = await _redis.execute_command("VGETATTR", mem_store.FACT_KEY, req.element_id)
        if not raw:
            return {"status": "not_found", "element_id": req.element_id}
        
        attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        
        # Apply feedback-based adjustments
        if req.rating >= 4:
            # Positive feedback: boost importance and reinforce
            old_importance = attrs.get("importance", 0.5)
            attrs["importance"] = min(1.0, old_importance * 1.2)
            attrs["user_rating"] = req.rating
            attrs["user_rating_ts"] = int(time.time() * 1000)
            if req.comment:
                attrs["user_comment"] = req.comment[:500]
            
            # Reinforce fact (increment source_count, reset decay)
            await mem_store.reinforce_fact(_redis, req.element_id, source=f"user_feedback_{req.rating}")
            
            # Update Redis attrs
            await _redis.execute_command(
                "VSETATTR", mem_store.FACT_KEY, req.element_id,
                json.dumps(attrs, ensure_ascii=False)
            )
            
            log.info(f"[feedback] positive rating {req.rating}/5 for {req.element_id}, "
                    f"importance {old_importance:.2f} → {attrs['importance']:.2f}")
            
            return {
                "status": "ok",
                "element_id": req.element_id,
                "new_importance": attrs["importance"],
                "action": "boosted_and_reinforced"
            }
        
        elif req.rating <= 2:
            # Negative feedback: reduce importance, flag for review
            old_importance = attrs.get("importance", 0.5)
            attrs["importance"] = max(0.05, old_importance * 0.7)
            attrs["user_rating"] = req.rating
            attrs["user_rating_ts"] = int(time.time() * 1000)
            attrs["needs_review"] = True
            if req.comment:
                attrs["user_comment"] = req.comment[:500]
            
            # Update Redis attrs
            await _redis.execute_command(
                "VSETATTR", mem_store.FACT_KEY, req.element_id,
                json.dumps(attrs, ensure_ascii=False)
            )
            
            log.info(f"[feedback] negative rating {req.rating}/5 for {req.element_id}, "
                    f"importance {old_importance:.2f} → {attrs['importance']:.2f}")
            
            return {
                "status": "ok",
                "element_id": req.element_id,
                "new_importance": attrs["importance"],
                "action": "reduced_and_flagged_for_review"
            }
        
        else:
            # Neutral rating (3): just record it
            attrs["user_rating"] = req.rating
            attrs["user_rating_ts"] = int(time.time() * 1000)
            if req.comment:
                attrs["user_comment"] = req.comment[:500]
            
            await _redis.execute_command(
                "VSETATTR", mem_store.FACT_KEY, req.element_id,
                json.dumps(attrs, ensure_ascii=False)
            )
            
            return {
                "status": "ok",
                "element_id": req.element_id,
                "action": "recorded_neutral_rating"
            }
    
    except Exception as e:
        log.error(f"[feedback] error processing feedback: {e}")
        return {"error": str(e)}


@app.post("/facts/{element_id}/pin")
async def pin_fact(element_id: str):
    """
    Pin fact permanently (importance = 1.0, never pruned).
    
    Pinned facts are excluded from consolidation pruning phase.
    Use for critical rules, identity facts, or essential procedures.
    """
    try:
        raw = await _redis.execute_command("VGETATTR", mem_store.FACT_KEY, element_id)
        if not raw:
            return {"status": "not_found", "element_id": element_id}
        
        attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        attrs["pinned"] = True
        attrs["pinned_at"] = int(time.time() * 1000)
        attrs["importance"] = 1.0  # maximum importance
        
        await _redis.execute_command(
            "VSETATTR", mem_store.FACT_KEY, element_id,
            json.dumps(attrs, ensure_ascii=False)
        )
        
        log.info(f"[pin] pinned fact {element_id}: {attrs.get('content', '')[:80]}")
        
        return {
            "status": "ok",
            "element_id": element_id,
            "action": "pinned_permanently"
        }
    
    except Exception as e:
        log.error(f"[pin] error pinning fact: {e}")
        return {"error": str(e)}


@app.delete("/facts/{element_id}")
async def delete_fact(element_id: str):
    """
    User-initiated hard delete (immediate VREM).
    
    Unlike soft-delete (superseded_by), this physically removes the fact
    from the vectorset. Use with caution — no undo.
    """
    try:
        # Verify fact exists
        raw = await _redis.execute_command("VGETATTR", mem_store.FACT_KEY, element_id)
        if not raw:
            return {"status": "not_found", "element_id": element_id}
        
        attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        content_preview = attrs.get("content", "")[:80]
        
        # Physical removal
        await _redis.execute_command("VREM", mem_store.FACT_KEY, element_id)
        
        # Invalidate BM25 index
        await _bm25_index.reset()
        _spawn(_populate_bm25_from_redis(_redis), "bm25-rebuild-after-delete")
        
        log.info(f"[delete] user-deleted fact {element_id}: {content_preview}")
        
        return {
            "status": "ok",
            "element_id": element_id,
            "action": "hard_deleted"
        }
    
    except Exception as e:
        log.error(f"[delete] error deleting fact: {e}")
        return {"error": str(e)}


@app.get("/facts/{element_id}/metadata")
async def get_fact_metadata(element_id: str):
    """
    Get full metadata for a fact including user ratings, pins, lifecycle info.
    
    Useful for debugging and understanding why a fact was stored/retrieved.
    """
    try:
        raw = await _redis.execute_command("VGETATTR", mem_store.FACT_KEY, element_id)
        if not raw:
            return {"status": "not_found", "element_id": element_id}
        
        attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        eff_conf = mem_store.confidence_decay(attrs)
        
        return {
            "element_id": element_id,
            "content": attrs.get("content", "")[:500],
            "category": attrs.get("category", ""),
            "importance": attrs.get("importance", 0.5),
            "confidence": attrs.get("confidence", 0.8),
            "effective_confidence": eff_conf,
            "source_count": attrs.get("source_count", 1),
            "access_count": attrs.get("access_count", 0),
            "created_at": attrs.get("ts", 0),
            "last_confirmed_ts": attrs.get("last_confirmed_ts", 0),
            "version": attrs.get("version", 1),
            "pinned": attrs.get("pinned", False),
            "user_rating": attrs.get("user_rating"),
            "user_comment": attrs.get("user_comment"),
            "needs_review": attrs.get("needs_review", False),
            "superseded_by": attrs.get("superseded_by", ""),
            "superseded_reason": attrs.get("superseded_reason", ""),
        }
    
    except Exception as e:
        return {"error": str(e), "element_id": element_id}


# ── Knowledge Graph endpoints (v0.9.0) ────────────────────────────────────────

@app.get("/graph/stats")
async def graph_stats_endpoint():
    """Return knowledge graph statistics: node count and edge count."""
    try:
        return await graph_mod.graph_stats(_redis)
    except Exception as e:
        log.error(f"[graph/stats] {e}")
        return {"nodes": 0, "edges": 0, "total_nodes": 0, "total_edges": 0, "error": str(e)}


@app.get("/graph/nodes")
async def graph_nodes_endpoint(limit: int = 60):
    """
    Return top N most-connected entities with their edges for graph visualization.
    Scans mem:graph:* keys, ranks by connection count, returns nodes + edges.
    """
    try:
        prefix = graph_mod.GRAPH_PREFIX
        # Collect all entity keys
        all_keys = []
        async for key in _redis.scan_iter(f"{prefix}*", count=500):
            all_keys.append(key.decode() if isinstance(key, bytes) else key)

        if not all_keys:
            return {"nodes": [], "edges": []}

        # Batch get connection counts — try scard (Set keys), fall back handled per-key
        pipe = _redis.pipeline(transaction=False)
        for k in all_keys:
            pipe.scard(k)
        scards = await pipe.execute(raise_on_error=False)

        # Sort by connection count descending, take top limit
        ranked = sorted(
            [(k, int(sc) if isinstance(sc, int) else 0) for k, sc in zip(all_keys, scards)],
            key=lambda x: x[1], reverse=True
        )[:limit]

        # Fetch neighbors for top entities
        top_keys = [k for k, _ in ranked]
        pipe = _redis.pipeline(transaction=False)
        for k in top_keys:
            pipe.smembers(k)  # legacy Set edges (slug strings)
        members_list = await pipe.execute(raise_on_error=False)

        top_slugs = {k[len(prefix):] for k in top_keys}
        nodes_out = []
        edges_out = []
        seen_edges: set[tuple] = set()

        for (k, conn_count), members in zip(ranked, members_list):
            slug = k[len(prefix):]
            label = slug.replace("_", " ")
            nodes_out.append({"id": slug, "label": label, "connections": conn_count})

            if not isinstance(members, (set, list)):
                continue
            for m in members:
                nb = m.decode() if isinstance(m, bytes) else m
                pair = (min(slug, nb), max(slug, nb))
                if pair not in seen_edges:
                    seen_edges.add(pair)
                    edges_out.append({"source": slug, "target": nb, "type": "related_to"})

        return {"nodes": nodes_out, "edges": edges_out}
    except Exception as e:
        log.error(f"[graph/nodes] {e}")
        return {"nodes": [], "edges": [], "error": str(e)}


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


# ── LLM Wiki v2 endpoints (v1.1) ─────────────────────────────────────────────

@app.post("/graph/edge")
async def add_graph_edge(req: TypedEdgeRequest):
    """
    Add a typed relationship edge between two entities.

    LLM Wiki v2: typed relationships carry semantic weight.
    Types: uses, depends_on, contradicts, caused, fixed, supersedes, related_to.
    """
    await graph_mod.add_typed_edge(
        _redis, req.source_entity, req.target_entity,
        req.relationship_type, req.confidence, req.source, req.bidirectional,
    )
    await _ws_broadcast("graph_edge", {
        "source": req.source_entity, "target": req.target_entity,
        "type": req.relationship_type, "confidence": req.confidence,
    })
    return {"status": "ok", "source": req.source_entity, "target": req.target_entity,
            "type": req.relationship_type}


@app.post("/graph/traverse")
async def graph_traverse(req: TraverseRequest):
    """
    Walk outward through typed edges for impact analysis.

    LLM Wiki v2: "Start at a node, walk outward through 'depends on' and 'uses'
    edges, and find everything downstream."
    """
    results = await graph_mod.traverse(
        _redis, req.entity, req.relationship_types,
        req.max_depth, req.max_nodes,
    )
    return {"entity": req.entity, "nodes": results, "count": len(results)}


@app.post("/facts/{element_id}/reinforce")
async def reinforce_fact_endpoint(element_id: str, req: ReinforceRequest):
    """
    Reinforce a fact — increment source_count, reset Ebbinghaus decay.

    LLM Wiki v2: "Confidence strengthens with reinforcement. Each reinforcement
    resets the forgetting curve."
    """
    ok = await mem_store.reinforce_fact(_redis, element_id, req.source or None)
    if not ok:
        return {"status": "not_found", "element_id": element_id}
    await _ws_broadcast("reinforce", {"element_id": element_id, "source": req.source})
    return {"status": "ok", "element_id": element_id}


@app.get("/facts/{element_id}/confidence")
async def get_fact_confidence(element_id: str):
    """
    Get effective confidence for a fact (Ebbinghaus decay applied).

    Returns both base confidence and effective confidence after decay.
    """
    try:
        raw = await _redis.execute_command("VGETATTR", mem_store.FACT_KEY, element_id)
        if not raw:
            return {"error": "not_found", "element_id": element_id}
        attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        eff = mem_store.confidence_decay(attrs)
        return {
            "element_id": element_id,
            "base_confidence": attrs.get("confidence", 0.8),
            "effective_confidence": eff,
            "source_count": attrs.get("source_count", 1),
            "last_confirmed_ts": attrs.get("last_confirmed_ts", 0),
            "category": attrs.get("category", ""),
            "version": attrs.get("version", 1),
            "superseded_by": attrs.get("superseded_by", ""),
            "superseded_reason": attrs.get("superseded_reason", ""),
        }
    except Exception as e:
        return {"error": str(e), "element_id": element_id}


@app.get("/lifecycle/stats")
async def lifecycle_stats():
    """
    Return lifecycle statistics: confidence distribution, supersession counts,
    version distribution, and category-level health.
    """
    card = await _redis.execute_command("VCARD", mem_store.FACT_KEY)
    if not card or int(card) <= 1:
        return {"total": 0}

    seed = np.zeros(embedder.DIMS, dtype=np.float32)
    results = await _redis.execute_command(
        "VSIM", mem_store.FACT_KEY, "FP32", seed.tobytes(),
        "COUNT", min(int(card), 500), "WITHSCORES", "WITHATTRIBS"
    )

    active = 0
    superseded = 0
    conf_buckets = {"high": 0, "medium": 0, "low": 0, "expired": 0}
    reason_counts: dict[str, int] = {}
    category_health: dict[str, dict] = {}

    i = 0
    while i + 2 < len(results):
        raw = results[i + 2]
        i += 3
        try:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
            continue
        if attrs.get("_seed") or not attrs.get("content"):
            continue

        if attrs.get("superseded_by"):
            superseded += 1
            reason = attrs.get("superseded_reason", "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            continue

        active += 1
        eff = mem_store.confidence_decay(attrs)
        if eff >= 0.7:
            conf_buckets["high"] += 1
        elif eff >= 0.4:
            conf_buckets["medium"] += 1
        elif eff >= 0.1:
            conf_buckets["low"] += 1
        else:
            conf_buckets["expired"] += 1

        cat = attrs.get("category", "general")
        if cat not in category_health:
            category_health[cat] = {"count": 0, "avg_confidence": 0.0, "total_conf": 0.0}
        category_health[cat]["count"] += 1
        category_health[cat]["total_conf"] += eff

    for cat in category_health:
        c = category_health[cat]
        c["avg_confidence"] = round(c["total_conf"] / max(1, c["count"]), 4)
        del c["total_conf"]

    return {
        "total": active + superseded,
        "active": active,
        "superseded": superseded,
        "confidence_distribution": conf_buckets,
        "supersession_reasons": reason_counts,
        "category_health": category_health,
    }


@app.post("/crystallize")
async def crystallize(req: CrystallizeRequest):
    """
    Crystallize a session — distill it into a structured digest.

    LLM Wiki v2: "Crystallization is the process of taking a completed chain of
    work and automatically distilling it into a structured digest. What was the
    question? What did we find? What entities were involved? What lessons emerged?"
    """
    scanned = await _vscan(_redis, mem_store.FACT_KEY, max_count=200)

    facts = []
    all_entities: set[str] = set()
    for item in scanned:
        attrs = item["attrs"]
        if not attrs.get("content") or attrs.get("superseded_by"):
            continue
        facts.append(attrs)
        for e in attrs.get("entities", []):
            all_entities.add(e)
        for p in attrs.get("persons", []):
            all_entities.add(p)

    # Sort by importance, take top N
    facts.sort(key=lambda a: a.get("importance", 0.5), reverse=True)
    top_facts = facts[:req.max_facts]

    # Build digest
    digest = {
        "session_id": req.session_id,
        "fact_count": len(top_facts),
        "total_facts_available": len(facts),
        "facts": [
            {
                "content": f.get("content", ""),
                "category": f.get("category", ""),
                "confidence": f.get("confidence", 0.8),
                "effective_confidence": mem_store.confidence_decay(f),
                "importance": f.get("importance", 0.5),
                "source_count": f.get("source_count", 1),
            }
            for f in top_facts
        ],
        "entities": sorted(all_entities),
        "categories": sorted(set(f.get("category", "") for f in top_facts)),
        "crystallized_at": int(time.time() * 1000),
    }

    await _ws_broadcast("crystallize", {"session_id": req.session_id, "fact_count": len(top_facts)})
    return digest


# ── Config endpoint (v0.9.0) ──────────────────────────────────────────────────

@app.get("/config")
async def get_config():
    """Return current service configuration including auto-consolidation settings."""
    return {
        "version":                 APP_VERSION,
        "auto_consolidate_every":  _AUTO_CONSOLIDATE_EVERY,
        "stores_since_last":       _stores_since_consolidation,
        "periodic_interval_s":     3600,
        "entropy_gate_threshold":  0.35,
        "dedup_threshold":         0.95,
        "consolidate_threshold":   0.85,
        "session_ttl_s":           14400,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Absorbed agentmemory scaffold endpoints (no /agentmemory prefix)
# These provide full compatibility with rohitg00/agentmemory hooks & MCP tools.
# ═══════════════════════════════════════════════════════════════════════════════

def _compat_sid(req) -> str:
    """Extract session ID from compat request (supports both camelCase and snake_case)."""
    return req.sessionId or req.session_id or f"ses_{int(time.time())}"


def _compat_check_auth(request: Request) -> dict | None:
    """Check AGENTMEMORY_SECRET bearer token if configured."""
    secret = _os.getenv("AGENTMEMORY_SECRET", "")
    if not secret:
        return None
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {secret}":
        return {"status_code": 401, "error": "unauthorized"}
    return None


async def _observe_internal(session_id: str, _project: str, _cwd: str,
                            hook_type: str, data: dict) -> dict:
    """Core observe logic shared by /observe endpoint."""
    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", "")
    tool_output = data.get("tool_output", "")

    content_parts = []
    if tool_name:
        content_parts.append(f"Tool: {tool_name}")
    if tool_input:
        inp = tool_input if isinstance(tool_input, str) else json.dumps(tool_input, ensure_ascii=False)
        content_parts.append(f"Input: {inp[:2000]}")
    if tool_output:
        out = tool_output if isinstance(tool_output, str) else json.dumps(tool_output, ensure_ascii=False)
        content_parts.append(f"Output: {out[:4000]}")

    content = "\n".join(content_parts)
    if not content.strip():
        return {"status": "ok", "action": "skipped_empty"}

    if _is_trivial(content) or _is_injected(content):
        return {"status": "ok", "action": "skipped_filter"}
    if _contains_secret(content):
        content = _redact_secrets(content)

    if not await _admission_gate(content):
        return {"status": "ok", "action": "skipped_gate"}

    emb = _encode(content)
    uid = await mem_store.save_episode(
        _redis, session_id, content, emb,
        ep_type=hook_type,
    )

    try:
        from core import extractor
        facts = await extractor.extract_facts(content, session_id)
        for fact_text, fact_attrs in facts:
            fact_emb = _encode(fact_text)
            await mem_store.save_fact(
                _redis,
                content=fact_text,
                category=fact_attrs.get("category", "fact"),
                confidence=fact_attrs.get("confidence", 0.7),
                embedding=fact_emb,
                language=fact_attrs.get("language", "en"),
                domain=fact_attrs.get("domain", "general"),
                keywords=fact_attrs.get("keywords"),
                importance=fact_attrs.get("importance", 0.5),
                source_episode_id=uid,
            )
    except Exception as e:
        log.warning("[observe] fact extraction failed: %s", e)

    return {"status": "ok", "action": "observed", "id": uid}


# ── Liveness / Feature flags ──────────────────────────────────────────────────

@app.get("/livez")
async def livez():
    return {"status": "ok", "service": "agentmem"}


@app.get("/config/flags")
async def config_flags():
    return {
        "graphExtractionEnabled": _os.getenv("GRAPH_EXTRACTION_ENABLED", "false").lower() == "true",
        "consolidationEnabled": True,
        "autoCompressEnabled": True,
        "contextInjectionEnabled": _os.getenv("AGENTMEMORY_INJECT_CONTEXT", "true").lower() == "true",
    }


# ── Automatic Crystallization Background Task ────────────────────────────────

async def _crystallize_session_inline(session_id: str, session_obj: dict, max_facts: int = 20) -> dict | None:
    """
    Inline crystallization logic (same as /crystallize endpoint but without HTTP overhead).

    Returns digest dict or None if crystallization fails.
    """
    try:
        scanned = await _vscan(_redis, mem_store.FACT_KEY, max_count=200)

        facts = []
        all_entities: set[str] = set()
        session_ts = session_obj.get("ts", 0)

        for item in scanned:
            attrs = item["attrs"]
            if not attrs.get("content") or attrs.get("superseded_by"):
                continue
            fact_ts = attrs.get("ts", 0)
            if abs(fact_ts - session_ts) < 3600000:  # within 1 hour
                facts.append(attrs)
                for e in attrs.get("entities", []):
                    all_entities.add(e)
                for p in attrs.get("persons", []):
                    all_entities.add(p)
        
        if not facts:
            return None
        
        # Sort by importance, take top N
        facts.sort(key=lambda a: a.get("importance", 0.5), reverse=True)
        top_facts = facts[:max_facts]
        
        # Generate summary using LLM (reuse existing summarizer)
        # Note: Ensure 'summarizer' is imported or available in scope
        fact_texts = [f.get("content", "") for f in top_facts[:10]]  # top 10 for summary
        summary_prompt = (
            "Summarize these key findings from a completed work session in 2-3 sentences. "
            "Focus on what was accomplished, what was learned, and any important decisions made.\n\n"
            + "\n".join(f"- {t}" for t in fact_texts)
        )
        
        summary = ""
        try:
            from core import summarizer
            summary = await summarizer.summarize(summary_prompt)
        except ImportError:
            summary = "Auto-crystallized session summary (summarizer unavailable)."
        except Exception as e:
            log.warning(f"[crystallize] summarization failed: {e}")
            summary = "Auto-crystallized session summary (summarization error)."
        
        # Build digest
        digest = {
            "session_id": session_id,
            "summary": summary[:500],
            "fact_count": len(top_facts),
            "total_facts_available": len(facts),
            "facts": [
                {
                    "content": f.get("content", ""),
                    "category": f.get("category", ""),
                    "confidence": f.get("confidence", 0.8),
                    "effective_confidence": mem_store.confidence_decay(f),
                    "importance": f.get("importance", 0.5),
                    "source_count": f.get("source_count", 1),
                }
                for f in top_facts
            ],
            "entities": sorted(all_entities)[:20],  # top 20 entities
            "categories": sorted(set(f.get("category", "") for f in top_facts)),
            "crystallized_at": int(time.time() * 1000),
            "auto_crystallized": True,
        }
        
        return digest
    
    except Exception as e:
        log.warning(f"[crystallize] inline crystallization failed for {session_id}: {e}")
        return None


async def _auto_crystallize() -> None:
    """
    Background task: automatically crystallize sessions older than 24h.
    
    Crystallization distills completed work chains into structured digests.
    Runs every 6 hours.
    """
    while True:
        await asyncio.sleep(21600)  # 6 hours
        try:
            now_ms = int(time.time() * 1000)
            twenty_four_hours_ms = 24 * 3600 * 1000

            # Phase 1: Batch-scan all session keys and crystallized keys
            session_pattern = "mem:session:*"
            cursor = 0
            all_session_keys = []

            while True:
                cursor, keys = await _redis.scan(cursor, match=session_pattern, count=100)
                all_session_keys.extend(keys)
                if cursor == 0:
                    break

            if not all_session_keys:
                continue

            # Batch check which sessions are already crystallized + get session data
            # Pipeline: exists(crystal_key) + get(session_key) per session
            pipe = _redis.pipeline(transaction=False)
            key_map = []  # track (session_key, session_id, crystal_key) for each pipeline pair
            for key in all_session_keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                session_id = key_str.replace("mem:session:", "")
                crystal_key = f"mem:crystallized:{session_id}"
                pipe.exists(crystal_key)
                pipe.get(key)
                key_map.append((key_str, session_id, crystal_key))

            pipe_results = await pipe.execute()

            # Parse pipeline results (pairs: exists, get for each session)
            candidates = []
            for idx, (key_str, session_id, crystal_key) in enumerate(key_map):
                exists = pipe_results[idx * 2]
                session_data = pipe_results[idx * 2 + 1]
                if exists:
                    continue
                if not session_data:
                    continue
                try:
                    session_obj = json.loads(session_data.decode() if isinstance(session_data, bytes) else session_data)
                except Exception:
                    continue

                ts = session_obj.get("ts", 0)
                if not ts:
                    started_at = session_obj.get("started_at", 0)
                    if started_at:
                        ts = int(started_at * 1000)
                if not ts:
                    continue
                age_ms = now_ms - ts
                if age_ms < twenty_four_hours_ms:
                    continue

                candidates.append((session_id, session_obj, ts, crystal_key))

            if not candidates:
                continue

            # Phase 2: Fetch all facts once (not per-session) and count per session
            scanned = await _vscan(_redis, mem_store.FACT_KEY, max_count=200)
            if len(scanned) <= 5:
                continue

            # Build fact timestamp list once
            fact_timestamps = [
                item["attrs"].get("ts", 0)
                for item in scanned
                if not item["attrs"].get("superseded_by")
            ]

            # Phase 3: Crystallize qualifying sessions
            crystallized_count = 0
            for session_id, session_obj, ts, crystal_key in candidates:
                # Count facts within ±1 hour of session start
                fact_count = sum(1 for ft in fact_timestamps if abs(ft - ts) < 3600000)
                if fact_count < 5:
                    continue

                log.info(f"[crystallize] auto-crystallizing session {session_id} ({fact_count} facts, age={(now_ms - ts)/3600000:.1f}h)")

                digest = await _crystallize_session_inline(session_id, session_obj, fact_count)
                if digest:
                    await _redis.setex(
                        crystal_key,
                        90 * 86400,
                        json.dumps(digest, ensure_ascii=False)
                    )
                    crystallized_count += 1
                    await _ws_broadcast("crystallize_auto", {
                        "session_id": session_id,
                        "fact_count": fact_count,
                        "digest_summary": digest.get("summary", "")[:200]
                    })

            if crystallized_count > 0:
                log.info(f"[crystallize] auto-crystallized {crystallized_count} sessions")

        except Exception as e:
            log.warning(f"[crystallize] auto-crystallization error: {e}", exc_info=True)


# ── Session lifecycle (agentmemory hooks) ─────────────────────────────────────

@app.post("/session/start")
async def compat_session_start(req: CompatSessionStartRequest, request: Request):
    auth_err = _compat_check_auth(request)
    if auth_err:
        return auth_err

    sid = _compat_sid(req)
    project = req.project or req.cwd or _os.getcwd()

    session_key = f"mem:session:{sid}"
    await _redis.set(session_key, json.dumps({
        "session_id": sid,
        "project": project,
        "started_at": time.time(),
        "observations": 0,
    }), ex=14400)

    persona_ctx = await persona_mod.get_context(_redis)

    pinned_raw = await _redis.get("mem:pinned:session_summary")
    last_summary = pinned_raw.decode() if pinned_raw else None

    context_parts = []
    if persona_ctx:
        context_parts.append(persona_ctx)
    if last_summary:
        context_parts.append(f"## Last Session Summary\n{last_summary[:600]}")

    context = "\n\n".join(context_parts) if context_parts else None

    return {"status": "ok", "sessionId": sid, "context": context}


@app.post("/session/end")
async def compat_session_end(req: CompatSessionEndRequest, request: Request,
                              background_tasks: BackgroundTasks):
    auth_err = _compat_check_auth(request)
    if auth_err:
        return auth_err

    sid = _compat_sid(req)

    async def _do_end():
        try:
            await _do_compress_session(sid)
        except Exception as e:
            log.warning("[session/end] compress failed: %s", e)

    background_tasks.add_task(_do_end)
    return {"status": "ok", "sessionId": sid}


# ── Observe (post-tool-use, prompt-submit, subagent hooks) ────────────────────

@app.post("/observe")
async def observe(req: ObserveRequest, request: Request, background_tasks: BackgroundTasks):
    auth_err = _compat_check_auth(request)
    if auth_err:
        return auth_err

    sid = _compat_sid(req)
    project = req.project or req.cwd or ""
    hook_type = req.hookType or "observe"
    data = req.data or {}

    async def _do_observe():
        try:
            await _observe_internal(sid, project, req.cwd or "", hook_type, data)
        except Exception as e:
            log.warning("[observe] background error: %s", e)

    background_tasks.add_task(_do_observe)
    return {"status": "ok"}


# ── Summarize (stop hook) ─────────────────────────────────────────────────────

@app.post("/summarize")
async def compat_summarize(req: SummarizeRequest, request: Request,
                            background_tasks: BackgroundTasks):
    auth_err = _compat_check_auth(request)
    if auth_err:
        return auth_err

    sid = _compat_sid(req)

    async def _do_summarize():
        try:
            await _do_compress_session(sid)
        except Exception as e:
            log.warning("[summarize] compress failed: %s", e)

    background_tasks.add_task(_do_summarize)
    return {"status": "ok", "sessionId": sid}


# ── Enrich (pre-tool-use hook) ────────────────────────────────────────────────

@app.post("/enrich")
async def enrich(req: EnrichRequest, request: Request):
    auth_err = _compat_check_auth(request)
    if auth_err:
        return auth_err

    files = req.files or []
    query = req.query or ""

    file_contexts = []
    for f in files[:5]:
        fname = _os.path.basename(f)
        emb = _encode(fname)
        results = await mem_store.knn_search(_redis, mem_store.FACT_KEY, emb, k=3)
        for r in results:
            attrs = r.get("attrs", {})
            if attrs.get("content"):
                file_contexts.append(attrs["content"][:200])

    if query:
        emb = _encode(query)
        results = await mem_store.knn_search(_redis, mem_store.FACT_KEY, emb, k=5)
        for r in results:
            attrs = r.get("attrs", {})
            if attrs.get("content"):
                file_contexts.append(attrs["content"][:200])

    context = "\n".join(file_contexts[:10]) if file_contexts else None
    return {"status": "ok", "context": context}


# ── Context (pre-compact hook) ────────────────────────────────────────────────

@app.post("/context")
async def compat_context(req: ContextRequest, request: Request):
    auth_err = _compat_check_auth(request)
    if auth_err:
        return auth_err

    sid = _compat_sid(req)
    query = req.query or ""
    budget = req.token_budget or 1500

    persona_ctx = await persona_mod.get_context(_redis)
    session_ctx = await mem_store.get_session_context(_redis, sid)

    pinned_raw = await _redis.get("mem:pinned:session_summary")
    last_summary = pinned_raw.decode() if pinned_raw else None

    facts = []
    episodes = []
    if query:
        emb = _encode(query)
        facts = await mem_store.knn_search(_redis, mem_store.FACT_KEY, emb, k=8)
        episodes = await mem_store.knn_search(_redis, mem_store.EPISODE_KEY, emb, k=6)

    prepend = _format_prepend(
        facts=facts,
        episodes=episodes,
        session_ctx=session_ctx,
        persona_ctx=persona_ctx or "",
        token_budget=budget,
        last_session_summary=last_summary,
    )

    return {"status": "ok", "context": prepend}


# ── Session commit (post-commit hook) ─────────────────────────────────────────

@app.post("/session/commit")
async def compat_session_commit(req: SessionCommitRequest, request: Request):
    auth_err = _compat_check_auth(request)
    if auth_err:
        return auth_err

    sid = _compat_sid(req)
    commit_key = f"mem:commit:{req.sha}"
    await _redis.hset(commit_key, mapping={
        "sha": req.sha,
        "message": req.message,
        "branch": req.branch,
        "repo": req.repo,
        "session_id": sid,
        "timestamp": str(int(time.time() * 1000)),
    })

    return {"status": "ok", "sha": req.sha}


@app.get("/session/by-commit")
async def session_by_commit(sha: str = ""):
    if not sha:
        return {"error": "sha parameter required"}

    commit_key = f"mem:commit:{sha}"
    data = await _redis.hgetall(commit_key)
    if not data:
        return {"error": "commit not found", "sha": sha}

    result = {k.decode() if isinstance(k, bytes) else k:
              v.decode() if isinstance(v, bytes) else v
              for k, v in data.items()}
    return result


@app.get("/commits")
async def commits(branch: str = "", repo: str = "", limit: int = 20):
    _, keys = await _redis.scan(match="mem:commit:*", count=limit)
    if not keys:
        return {"commits": []}

    # Batch fetch all commit data via pipeline (avoids N+1 round-trips)
    pipe = _redis.pipeline(transaction=False)
    for key in keys:
        k = key.decode() if isinstance(key, bytes) else key
        pipe.hgetall(k)
    pipe_results = await pipe.execute()

    results = []
    for data in pipe_results:
        if not data:
            continue
        entry = {dk.decode() if isinstance(dk, bytes) else dk:
                 dv.decode() if isinstance(dv, bytes) else dv
                 for dk, dv in data.items()}
        if branch and entry.get("branch") != branch:
            continue
        if repo and entry.get("repo") != repo:
            continue
        results.append(entry)
    results.sort(key=lambda x: x.get("timestamp", "0"), reverse=True)
    return {"commits": results[:limit]}


# ── Claude bridge sync ────────────────────────────────────────────────────────

@app.post("/claude-bridge/sync")
async def claude_bridge_sync(req: ClaudeBridgeSyncRequest, request: Request):
    auth_err = _compat_check_auth(request)
    if auth_err:
        return auth_err

    sid = _compat_sid(req)
    project = req.project or req.cwd or ""

    persona_ctx = await persona_mod.get_context(_redis)
    session_ctx = await mem_store.get_session_context(_redis, sid)

    pinned_raw = await _redis.get("mem:pinned:session_summary")
    last_summary = pinned_raw.decode() if pinned_raw else None

    emb = _encode(project or "project context")
    facts = await mem_store.knn_search(_redis, mem_store.FACT_KEY, emb, k=10)
    episodes = await mem_store.knn_search(_redis, mem_store.EPISODE_KEY, emb, k=6)

    prepend = _format_prepend(
        facts=facts,
        episodes=episodes,
        session_ctx=session_ctx,
        persona_ctx=persona_ctx or "",
        token_budget=2000,
        last_session_summary=last_summary,
    )

    memory_md_path = _os.path.expanduser("~/.claude/AGENTMEM_MEMORY.md")
    if prepend:
        try:
            _os.makedirs(_os.path.dirname(memory_md_path), exist_ok=True)
            with open(memory_md_path, "w") as f:
                f.write(prepend)
        except Exception as e:
            log.warning("[claude-bridge] write failed: %s", e)

    return {"status": "ok", "memoryPath": memory_md_path if prepend else None}


# ── Consolidate pipeline (session-end hook) ───────────────────────────────────

@app.post("/consolidate-pipeline")
async def compat_consolidate_pipeline(request: Request, background_tasks: BackgroundTasks):
    auth_err = _compat_check_auth(request)
    if auth_err:
        return auth_err

    async def _do_consolidate_compat():
        try:
            await _do_consolidate()
        except Exception as e:
            log.warning("[consolidate-pipeline] error: %s", e)

    background_tasks.add_task(_do_consolidate_compat)
    return {"status": "ok"}


# ── Crystals auto (session-end hook) ──────────────────────────────────────────

@app.post("/crystals/auto")
async def crystals_auto(request: Request):
    auth_err = _compat_check_auth(request)
    if auth_err:
        return auth_err
    return {"status": "ok", "crystals": []}


# ── Search (MCP memory_recall) ────────────────────────────────────────────────

@app.post("/search")
async def compat_search(req: SearchRequest, request: Request):
    auth_err = _compat_check_auth(request)
    if auth_err:
        return auth_err

    sid = _compat_sid(req)
    emb = _encode(req.query)

    facts = await mem_store.knn_search(_redis, mem_store.FACT_KEY, emb, k=req.limit)
    episodes = await mem_store.knn_search(_redis, mem_store.EPISODE_KEY, emb, k=req.limit)

    persona_ctx = await persona_mod.get_context(_redis)
    session_ctx = await mem_store.get_session_context(_redis, sid)

    pinned_raw = await _redis.get("mem:pinned:session_summary")
    last_summary = pinned_raw.decode() if pinned_raw else None

    budget = req.token_budget or 1500

    if req.format == "compact":
        results = []
        for f in facts:
            attrs = f.get("attrs", {})
            results.append({
                "content": f.get("content", "")[:200],
                "category": attrs.get("category", ""),
                "score": round(f.get("score", 0), 3),
            })
        for e in episodes:
            attrs = e.get("attrs", {})
            results.append({
                "content": e.get("content", "")[:200],
                "category": attrs.get("category", ""),
                "score": round(e.get("score", 0), 3),
            })
        return {"results": results}

    prepend = _format_prepend(
        facts=facts, episodes=episodes,
        session_ctx=session_ctx, persona_ctx=persona_ctx or "",
        token_budget=budget, last_session_summary=last_summary,
    )
    return {"context": prepend, "facts": len(facts), "episodes": len(episodes)}


# ── Remember (MCP memory_save) ────────────────────────────────────────────────

@app.post("/remember")
async def compat_remember(req: RememberRequest, request: Request):
    auth_err = _compat_check_auth(request)
    if auth_err:
        return auth_err

    content = req.content
    emb = _encode(content)

    uid = await mem_store.save_fact(
        _redis,
        content=content,
        category=req.type or "fact",
        confidence=0.8,
        embedding=emb,
        keywords=[c.strip() for c in req.concepts.split(",") if c.strip()] if req.concepts else None,
        importance=0.8,
    )

    await _ws_broadcast("remember", {"id": uid, "type": req.type or "fact"})
    return {"status": "ok", "id": uid}


# ── Forget (MCP memory_forget) ────────────────────────────────────────────────

@app.post("/forget")
async def compat_forget(req: ForgetRequest, request: Request):
    auth_err = _compat_check_auth(request)
    if auth_err:
        return auth_err

    emb = _encode(req.query)
    facts = await mem_store.knn_search(_redis, mem_store.FACT_KEY, emb, k=req.limit)

    if req.dry_run:
        return {
            "status": "dry_run",
            "would_delete": len(facts),
            "memories": [f.get("content", "")[:100] for f in facts],
        }

    deleted = 0
    for f in facts:
        elem = f.get("_element", "")
        if elem:
            await _redis.execute_command("VREM", mem_store.FACT_KEY, elem)
            deleted += 1

    await _ws_broadcast("forget", {"deleted": deleted})
    return {"status": "ok", "deleted": deleted}


# ── Compress file ─────────────────────────────────────────────────────────────

@app.post("/compress-file")
async def compress_file(filePath: str = ""):
    if not filePath or not _os.path.isfile(filePath):
        return {"error": "file not found", "path": filePath}

    try:
        with open(filePath, "r") as f:
            content = f.read()

        backup_path = filePath.replace(".md", ".original.md")
        if not _os.path.exists(backup_path):
            with open(backup_path, "w") as f:
                f.write(content)

        lines = content.split("\n")
        compressed = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("http") or stripped.startswith("```"):
                compressed.append(line)
            elif stripped and len(stripped) > 20:
                compressed.append(stripped[:120] + "…")
            elif stripped:
                compressed.append(line)

        with open(filePath, "w") as f:
            f.write("\n".join(compressed))

        return {"status": "ok", "original_lines": len(lines), "compressed_lines": len(compressed)}
    except Exception as e:
        return {"error": str(e)}


# ── Sessions list ─────────────────────────────────────────────────────────────

@app.get("/sessions")
async def compat_sessions(limit: int = 20):
    # Sessions are stored as mem:session:{id}:ctx (plain-text string, not JSON).
    # Scan for :ctx keys to enumerate known session IDs.
    _, ctx_keys = await _redis.scan(match="mem:session:*:ctx", count=500)
    if not ctx_keys:
        return {"sessions": []}

    pipe = _redis.pipeline(transaction=False)
    for key in ctx_keys:
        pipe.get(key)
    raw_values = await pipe.execute()

    results = []
    for key, raw in zip(ctx_keys, raw_values):
        k = key.decode() if isinstance(key, bytes) else key
        # Extract session ID from "mem:session:{id}:ctx"
        parts = k.split(":")
        sid = parts[2] if len(parts) >= 4 else k
        ctx_str = (raw.decode() if isinstance(raw, bytes) else raw) or ""
        results.append({
            "session_id": sid,
            "status": "ended",
            "observation_count": 0,
            "summary": ctx_str[:300] if ctx_str else None,
        })

    return {"sessions": results[:limit]}


# ── Observations list ─────────────────────────────────────────────────────────

@app.get("/observations")
async def compat_observations(sessionId: str = "", limit: int = 20):
    seed = np.zeros(embedder.DIMS, dtype=np.float32)
    card = await _redis.execute_command("VCARD", mem_store.EPISODE_KEY)
    if not card or int(card) <= 1:
        return {"observations": []}

    results_raw = await _redis.execute_command(
        "VSIM", mem_store.EPISODE_KEY, "FP32", seed.astype("float32").tobytes(),
        "COUNT", min(int(card), limit + 1), "WITHSCORES", "WITHATTRIBS"
    )

    items = []
    i = 0
    while i + 2 < len(results_raw):
        elem = results_raw[i]
        raw = results_raw[i + 2]
        i += 3
        elem_str = elem.decode() if isinstance(elem, bytes) else elem
        if elem_str == "__seed__":
            continue
        try:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
            continue
        if sessionId and attrs.get("session_id") != sessionId:
            continue
        items.append({
            "id": elem_str,
            "content": attrs.get("content", "")[:300],
            "session_id": attrs.get("session_id", ""),
            "tool_name": attrs.get("tool_name", ""),
            "timestamp": attrs.get("ts", 0),
        })

    return {"observations": items[:limit]}


# ── File context ──────────────────────────────────────────────────────────────

@app.post("/file-context")
async def file_context(req: FileContextRequest, request: Request):
    auth_err = _compat_check_auth(request)
    if auth_err:
        return auth_err

    files = [f.strip() for f in req.files.split(",") if f.strip()]
    sid = _compat_sid(req)

    all_results = []
    for f in files[:10]:
        fname = _os.path.basename(f)
        emb = _encode(fname)
        results = await mem_store.knn_search(_redis, mem_store.EPISODE_KEY, emb, k=5,
                                              filter_expr=f'.session_id != "{sid}"')
        for r in results:
            attrs = r.get("attrs", {})
            all_results.append({
                "file": f,
                "content": attrs.get("content", "")[:300],
                "session_id": attrs.get("session_id", ""),
                "score": round(r.get("score", 0), 3),
            })

    return {"results": all_results}


# ── Patterns ──────────────────────────────────────────────────────────────────

@app.post("/patterns")
async def compat_patterns(req: PatternsRequest, request: Request):
    auth_err = _compat_check_auth(request)
    if auth_err:
        return auth_err

    query = req.query or "recurring patterns"
    emb = _encode(query)
    facts = await mem_store.knn_search(_redis, mem_store.FACT_KEY, emb, k=req.limit)

    categories = {}
    for f in facts:
        cat = f.get("attrs", {}).get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    return {
        "patterns": [{"category": k, "count": v} for k, v in
                     sorted(categories.items(), key=lambda x: x[1], reverse=True)],
        "memories": [{"content": f.get("content", "")[:200],
                       "category": f.get("attrs", {}).get("category", "")}
                      for f in facts[:req.limit]],
    }


# ── Smart search ──────────────────────────────────────────────────────────────

@app.post("/smart-search")
async def smart_search(req: SmartSearchRequest, request: Request):
    auth_err = _compat_check_auth(request)
    if auth_err:
        return auth_err

    emb = _encode(req.query)

    facts = await mem_store.knn_search(_redis, mem_store.FACT_KEY, emb, k=req.limit)
    episodes = await mem_store.knn_search(_redis, mem_store.EPISODE_KEY, emb, k=req.limit)

    merged = _rrf_merge([facts, episodes], weights=[1.0, 0.8], limit=req.limit)

    return {
        "results": [{
            "content": m.get("content", "")[:300],
            "category": m.get("attrs", {}).get("category", ""),
            "score": round(m.get("score", 0), 4),
        } for m in merged],
    }


# ── Timeline ──────────────────────────────────────────────────────────────────

@app.post("/timeline")
async def compat_timeline(req: TimelineRequest, request: Request):
    auth_err = _compat_check_auth(request)
    if auth_err:
        return auth_err

    seed = np.zeros(embedder.DIMS, dtype=np.float32)
    card = await _redis.execute_command("VCARD", mem_store.EPISODE_KEY)
    if not card or int(card) <= 1:
        return {"timeline": []}

    results_raw = await _redis.execute_command(
        "VSIM", mem_store.EPISODE_KEY, "FP32", seed.astype("float32").tobytes(),
        "COUNT", min(int(card), req.limit + 1), "WITHSCORES", "WITHATTRIBS"
    )

    items = []
    i = 0
    while i + 2 < len(results_raw):
        elem = results_raw[i]
        raw = results_raw[i + 2]
        i += 3
        elem_str = elem.decode() if isinstance(elem, bytes) else elem
        if elem_str == "__seed__":
            continue
        try:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
            continue
        sid = req.sessionId or req.session_id
        if sid and attrs.get("session_id") != sid:
            continue
        items.append({
            "id": elem_str,
            "content": attrs.get("content", "")[:200],
            "timestamp": attrs.get("ts", 0),
            "session_id": attrs.get("session_id", ""),
            "category": attrs.get("category", ""),
        })

    items.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return {"timeline": items[:req.limit]}


# ── Profile ───────────────────────────────────────────────────────────────────

@app.get("/profile")
async def compat_profile():
    # Return raw persona hash fields (not the formatted markdown string)
    persona_raw = await _redis.hgetall("mem:persona")
    fields = {
        (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
        for k, v in persona_raw.items()
    }
    return {"profile": fields}


# ── Export / Import ───────────────────────────────────────────────────────────

@app.get("/export")
async def compat_export():
    export = {"version": "1.2.0", "exported_at": time.time(), "facts": [], "episodes": []}

    for key, label in [(mem_store.FACT_KEY, "facts"), (mem_store.EPISODE_KEY, "episodes")]:
        card = await _redis.execute_command("VCARD", key)
        if not card or int(card) <= 1:
            continue
        seed = np.zeros(embedder.DIMS, dtype=np.float32)
        results = await _redis.execute_command(
            "VSIM", key, "FP32", seed.astype("float32").tobytes(),
            "COUNT", min(int(card), 5000), "WITHSCORES", "WITHATTRIBS"
        )
        i = 0
        while i + 2 < len(results):
            elem = results[i]; raw = results[i + 2]; i += 3
            elem_str = elem.decode() if isinstance(elem, bytes) else elem
            if elem_str == "__seed__":
                continue
            try:
                attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            except Exception:
                continue
            export[label].append({"id": elem_str, **attrs})

    return export


@app.post("/import")
async def compat_import(req: ImportRequest, request: Request):
    auth_err = _compat_check_auth(request)
    if auth_err:
        return auth_err

    data = req.data
    imported = 0

    for item in data.get("episodes", []):
        content = item.get("content", "")
        if not content:
            continue
        emb = _encode(content)
        await mem_store.save_episode(
            _redis,
            session_id=item.get("session_id", ""),
            content=content,
            embedding=emb,
            ep_type=item.get("ep_type", item.get("category", "general")),
        )
        imported += 1

    for item in data.get("facts", []):
        content = item.get("content", "")
        if not content:
            continue
        emb = _encode(content)
        await mem_store.save_fact(
            _redis,
            content=content,
            category=item.get("category", "fact"),
            confidence=item.get("confidence", 0.7),
            embedding=emb,
            language=item.get("language", "en"),
            domain=item.get("domain", "general"),
            keywords=item.get("keywords"),
            importance=item.get("importance", 0.5),
        )
        imported += 1

    return {"status": "ok", "imported": imported}


# ── Memories list ─────────────────────────────────────────────────────────────

async def _list_memories(limit: int = 50, category: str = "") -> dict:
    """Shared logic for listing memories (used by /memories and /semantic)."""
    seed = np.zeros(embedder.DIMS, dtype=np.float32)
    card = await _redis.execute_command("VCARD", mem_store.FACT_KEY)
    if not card or int(card) <= 1:
        return {"memories": []}

    results_raw = await _redis.execute_command(
        "VSIM", mem_store.FACT_KEY, "FP32", seed.astype("float32").tobytes(),
        "COUNT", min(int(card), limit + 1), "WITHSCORES", "WITHATTRIBS"
    )

    items = []
    i = 0
    while i + 2 < len(results_raw):
        elem = results_raw[i]; raw = results_raw[i + 2]; i += 3
        elem_str = elem.decode() if isinstance(elem, bytes) else elem
        if elem_str == "__seed__":
            continue
        try:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
            continue
        if category and attrs.get("category") != category:
            continue
        items.append({"id": elem_str, **attrs})

    return {"memories": items[:limit]}


@app.get("/memories")
async def compat_memories(limit: int = 50, category: str = ""):
    return await _list_memories(limit=limit, category=category)


@app.get("/memories/{memory_id}")
async def memory_detail(memory_id: str):
    try:
        raw = await _redis.execute_command("VGETATTR", mem_store.FACT_KEY, memory_id)
        if raw:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            return {"id": memory_id, **attrs}
    except Exception:
        pass

    try:
        raw = await _redis.execute_command("VGETATTR", mem_store.EPISODE_KEY, memory_id)
        if raw:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            return {"id": memory_id, **attrs}
    except Exception:
        pass

    return {"error": "memory not found", "id": memory_id}


# ── Semantic / Procedural / Relations (read-only list views) ──────────────────

@app.get("/semantic")
async def compat_semantic(limit: int = 50):
    return await _list_memories(limit=limit)


@app.get("/procedural")
async def compat_procedural(limit: int = 50):
    seed = np.zeros(embedder.DIMS, dtype=np.float32)
    card = await _redis.execute_command("VCARD", mem_store.PROC_KEY)
    if not card or int(card) <= 1:
        return {"procedures": []}

    results_raw = await _redis.execute_command(
        "VSIM", mem_store.PROC_KEY, "FP32", seed.astype("float32").tobytes(),
        "COUNT", min(int(card), limit + 1), "WITHSCORES", "WITHATTRIBS"
    )

    items = []
    i = 0
    while i + 2 < len(results_raw):
        elem = results_raw[i]; raw = results_raw[i + 2]; i += 3
        elem_str = elem.decode() if isinstance(elem, bytes) else elem
        if elem_str == "__seed__":
            continue
        try:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
            continue
        items.append({"id": elem_str, **attrs})

    return {"procedures": items[:limit]}


@app.get("/relations")
async def compat_relations():
    stats = await graph_mod.graph_stats(_redis)
    return {"relations": stats}


# ── Viewer ────────────────────────────────────────────────────────────────────

@app.get("/viewer")
async def compat_viewer():
    idx = _os.path.join(_static_dir, "index.html")
    if _os.path.exists(idx):
        return FileResponse(idx)
    return {"message": "AgentMem Viewer", "version": "1.2.0"}
