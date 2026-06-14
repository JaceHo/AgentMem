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
import re
import time
from contextlib import asynccontextmanager

import httpx
import numpy as np
from fastapi import FastAPI, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ── Schemas (extracted to api/schemas/) ──────────────────────────────────────
from api.schemas.memory import RecallRequest, Message, StoreRequest
from api.schemas.consolidation import (
    CompactRequest, AnswerRequest, CompressSessionRequest,
)

# ── Search utilities (extracted to core/search.py) ───────────────────────────
from core.search import encode as _encode, vscan as _vscan, populate_bm25_from_redis

# ── Shared state (single source of truth) ────────────────────────────────────
from api import state

# ── Import refactored modules ────────────────────────────────────────────────
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
from core.utils import decode_bytes as _decode_bytes, decode_attrs as _decode_attrs, decode_json as _decode_json

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
from core.http import async_post_json
from services.consolidation_service import (
    do_consolidate as _do_consolidate,
    do_hard_prune as _do_hard_prune,
    crystallize_session_inline as _crystallize_session_inline,
)

logging.basicConfig(level=getattr(logging, settings.log_level), format=settings.log_format)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
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

_redis = None  # backward compat — delegates to state.redis via property-like access
_shutting_down = False  # set during lifespan shutdown to prevent new bg operations


def _get_redis():
    """Get the Redis client — always returns state.redis (single source of truth)."""
    r = state.redis
    if r is not None:
        return r
    # Fallback to legacy _redis global during startup before lifespan sets state.redis
    return _redis

# ── Task manager and counters — delegate to api.state (single source of truth) ─
_task_manager = state.task_manager
_AUTO_CONSOLIDATE_EVERY = state.AUTO_CONSOLIDATE_EVERY

def _spawn(coro, name: str = "bg") -> None:
    """Fire-and-forget a coroutine with TaskManager supervision."""
    state.spawn(coro, name=name)

# ── Thread-safe counters — delegate to api.state ─────────────────────────────
_stores_since_consolidation = state.stores_since_consolidation
_periodic_prune_counter = state.periodic_prune_counter
_store_attempts = state.store_attempts
_store_successes = state.store_successes
_store_skips = state.store_skips
_store_errors = state.store_errors
_store_latency_sum_ms = state.store_latency_sum_ms

# ── BM25 index — delegate to api.state ───────────────────────────────────────
_bm25_index = state.bm25_index


async def _encode_batch_async(texts: list[str]) -> list[np.ndarray]:
    """Batch-encode texts off the event loop using a single model forward pass.

    Uses asyncio.to_thread so the CPU-bound inference doesn't block the event loop.
    """
    if not texts:
        return []
    return list(await asyncio.to_thread(embedder.encode_batch, texts))

# ── Session handoff: pinned last-session summary ───────────────────────────────
# Key stores the most recent session summary so the NEXT session can always
# retrieve it regardless of query similarity (cross-session continuity bridge).
_PINNED_SESSION_KEY       = "mem:pinned:session_summary"
_CRYSTALLIZED_INDEX_KEY   = "mem:crystallized_index"   # sorted set: key → crystallized_at score


# ── Lifespan ──────────────────────────────────────────────────────────────────
async def _periodic_consolidate() -> None:
    """Background task: consolidate memory every hour; hard-prune every 24 hours."""
    global _periodic_prune_counter
    while True:
        await asyncio.sleep(3600)
        r = _get_redis()
        if r is None:
            continue
        try:
            card = await r.execute_command("VCARD", mem_store.FACT_KEY)
            if int(card or 0) > 5:
                await asyncio.wait_for(
                    _do_consolidate(r, _bm25_index, _spawn),
                    timeout=180.0,
                )
        except asyncio.TimeoutError:
            log.warning("[periodic_consolidate] timed out (180s limit)")
        except Exception as e:
            log.warning(f"[periodic_consolidate] error: {e}")
        prune_count = await _periodic_prune_counter.increment()
        if prune_count >= 24:
            await _periodic_prune_counter.reset()
            try:
                await _do_hard_prune(r, _bm25_index, _spawn)
            except Exception as e:
                log.warning(f"[periodic_consolidate] hard-prune error: {e}")


async def _backfill_crystallized_index(r) -> None:
    """One-time startup backfill: add any existing mem:crystallized:* keys to the sorted set index."""
    try:
        index_size = await r.zcard(_CRYSTALLIZED_INDEX_KEY)
        if index_size > 0:
            return  # index already populated
        cursor = 0
        pipe = r.pipeline(transaction=False)
        while True:
            cursor, keys = await r.scan(cursor, match="mem:crystallized:*", count=100)
            for key in keys:
                pipe.get(key)
            if cursor == 0:
                break
        raw_values = await pipe.execute()
        zadd_map: dict = {}
        for raw in raw_values:
            if not raw:
                continue
            try:
                d = _decode_json(raw)
                session_id = d.get("session_id", "")
                ts = d.get("crystallized_at", 0)
                if session_id and ts:
                    zadd_map[f"mem:crystallized:{session_id}"] = ts
            except Exception:
                continue
        if zadd_map:
            await r.zadd(_CRYSTALLIZED_INDEX_KEY, zadd_map)
            log.info(f"[crystallize] backfilled index with {len(zadd_map)} entries")
    except Exception as e:
        log.warning(f"[crystallize] index backfill failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis
    from api import state as _state
    log.info("Loading embedding model…")
    embedder._get_provider()  # warm up embedding model
    log.info("Connecting to Redis…")
    _redis = await mem_store.get_client()
    _state.redis = _redis  # share with compat/health/graph routes

    # Dimension guard: if existing vectorset has different dims than current
    # provider, force local (384-dim) to avoid corrupting vector search.
    try:
        for vset_key in (mem_store.FACT_KEY, mem_store.EPISODE_KEY, mem_store.PROC_KEY):
            info = await _get_redis().execute_command("VINFO", vset_key)
            if not info:
                continue
            info_dict = dict(zip(info[::2], info[1::2]))
            existing_dims = int(info_dict.get(b"vector-dim", info_dict.get("vector-dim", 0)))
            provider_dims = embedder._get_provider().dims
            if existing_dims and existing_dims != provider_dims:
                log.warning("Dimension mismatch in %s: existing=%d, provider=%s dims=%d — forcing local provider",
                            vset_key, existing_dims, embedder._get_provider().name, provider_dims)
                embedder._reset_provider("local")
                break  # no need to check further
    except Exception:
        pass  # no existing vectorset, safe to use any provider

    log.info("Ensuring vectorset indexes…")
    await mem_store.ensure_indexes(_get_redis())
    _spawn(_periodic_consolidate(), "consolidate")
    _spawn(_auto_crystallize(), "crystallize")  # Auto-crystallize sessions every 6h
    # Populate BM25 corpus from existing Redis facts (non-blocking, best-effort)
    _spawn(populate_bm25_from_redis(_get_redis(), state.bm25_index), "bm25-populate")
    # Backfill crystallized sorted-set index from existing keys (idempotent, skips if populated)
    _spawn(_backfill_crystallized_index(_get_redis()), "crystallize-index-backfill")
    log.info("AgentMem v%s ready (session-handoff+hard-prune+auto-graph+batch-MCP+compat+auto-crystallize)", APP_VERSION)

    # The MCP Streamable-HTTP transport is mounted as an ASGI sub-app, whose
    # own lifespan is NOT propagated by Starlette/FastAPI mounts. Run its
    # session manager here so the StreamableHTTPSessionManager task group is
    # initialized — otherwise every /mcp request 500s ("Task group is not initialized").
    _mcp_session_mgr = None
    try:
        from mcp_server import mcp as _mcp_srv
        _mcp_session_mgr = getattr(_mcp_srv, "session_manager", None)
    except Exception as _e:
        log.info("MCP session manager unavailable: %s", _e)

    if _mcp_session_mgr is not None:
        async with _mcp_session_mgr.run():
            log.info("MCP Streamable-HTTP session manager started")
            yield
    else:
        yield
    # Cancel tracked background tasks before closing Redis
    global _shutting_down
    _shutting_down = True
    await state.task_manager.shutdown(timeout=settings.bg_task_shutdown_timeout)
    from core.http import close_client
    await close_client()
    from mcp_server import close_client as _mcp_close_client
    await _mcp_close_client()
    await mem_store.close_pool()


app = FastAPI(title="AgentMem — Local Agent Memory Service", version=APP_VERSION, lifespan=lifespan)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Log full traceback for unhandled exceptions instead of silently returning 500."""
    import traceback
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    log.error("[unhandled] %s %s → %s\n%s", request.method, request.url.path, exc, "".join(tb))
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"detail": str(exc), "type": type(exc).__name__})


@app.middleware("http")
async def redis_guard(request: Request, call_next):
    """Return 503 if Redis is not connected (prevents NoneType crashes)."""
    if _get_redis() is None and request.url.path not in ("/health", "/docs", "/openapi.json", "/redoc"):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content={"detail": "Redis not connected"})
    return await call_next(request)


# Write-endpoint auth: if AGENTMEM_API_KEY is set, require X-API-Key header
# for mutating endpoints. Read endpoints (GET /recall, /health, /stats) remain open.
_WRITE_PREFIXES = ("/store", "/session/compress", "/session/clear",
                   "/admin/", "/consolidate", "/crystallize", "/feedback")

@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    """Require API key on write endpoints when AGENTMEM_API_KEY is configured."""
    if not settings.api_key:
        return await call_next(request)
    path = request.url.path
    needs_auth = any(path.startswith(p) for p in _WRITE_PREFIXES)
    if not needs_auth:
        return await call_next(request)
    provided = request.headers.get("X-API-Key", "")
    if provided != settings.api_key:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
    return await call_next(request)

# Request tracing: assign a unique ID to each request for log correlation
import uuid as _uuid

@app.middleware("http")
async def request_tracing(request: Request, call_next):
    """Assign a unique request ID and log request/response timing."""
    request_id = request.headers.get("X-Request-ID") or str(_uuid.uuid4())[:8]
    request.state.request_id = request_id

    import time as _time
    start = _time.monotonic()
    response = await call_next(request)
    elapsed_ms = int((_time.monotonic() - start) * 1000)

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{elapsed_ms}ms"

    # Structured access log for production monitoring
    log.info(
        "[access] %s %s → %d (%dms) rid=%s",
        request.method, request.url.path, response.status_code, elapsed_ms, request_id,
    )
    return response

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

# ── Extracted route modules ──────────────────────────────────────────────────
from api.routes.health import router as health_router
from api.routes.capability import router as capability_router
from api.routes.graph import router as graph_router
from api.routes.compat import router as compat_router
from api.routes.admin import router as admin_router
app.include_router(health_router)
app.include_router(capability_router)
app.include_router(graph_router)
app.include_router(compat_router)
app.include_router(admin_router)

# ── MCP transports: Streamable HTTP (preferred) + SSE (legacy) ───────────────
try:
    from mcp_server import build_mcp_app, build_sse_app
    app.mount("/mcp",     build_mcp_app(), name="mcp_http")
    app.mount("/mcp/sse", build_sse_app(), name="mcp_sse")
except ImportError as _mcp_err:
    import logging as _logging
    _logging.getLogger(__name__).info("MCP server not available (install 'mcp' package): %s", _mcp_err)
except Exception as _mcp_err:
    import logging as _logging
    _logging.getLogger(__name__).warning("MCP HTTP mount failed: %s", _mcp_err)

# ── WebSocket for real-time dashboard updates ─────────────────────────────────
_ws_clients: set[WebSocket] = set()


async def _ws_broadcast(event: str, data: dict) -> None:
    """Push an event to all connected WebSocket clients in parallel."""
    if not _ws_clients:
        return
    msg = json.dumps({"event": event, **data, "ts": time.time()})

    async def _send(ws: WebSocket) -> WebSocket | None:
        try:
            await ws.send_text(msg)
            return None
        except Exception:
            return ws

    results = await asyncio.gather(*[_send(ws) for ws in _ws_clients])
    dead = {ws for ws in results if ws is not None}
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
    token_budget: int | None = 1500,
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
    if token_budget is None:
        token_budget = settings.default_token_budget
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
    # Offload CPU-bound embedding to thread pool so it doesn't block the event loop
    emb = await asyncio.to_thread(_encode, text)
    recent = await mem_store.knn_search(_get_redis(), mem_store.EPISODE_KEY, emb, k=3)
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


# ── Session diversity: prevent one session dominating recall results ──────────

def _session_diversify(items: list[dict], max_per_session: int = 3) -> list[dict]:
    """Limit results from the same session to avoid recall dominated by one session.

    Preserves rank order within each session group. Items without a session_id
    (facts with no origin session) are always kept.
    """
    seen: dict[str, int] = {}
    result = []
    for item in items:
        sid = item.get("attrs", {}).get("session_id", "")
        if not sid:
            result.append(item)
            continue
        count = seen.get(sid, 0)
        if count < max_per_session:
            result.append(item)
            seen[sid] = count + 1
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
        raw = await _get_redis().execute_command("VGETATTR", mem_store.FACT_KEY, element_id)
        if not raw:
            return
        attrs = _decode_attrs(raw)
        existing_kws = set(attrs.get("keywords") or [])
        # Merge new keywords (deduplicated, capped at 12)
        merged_kws = list(existing_kws | set(new_keywords))[:12]
        attrs["keywords"] = merged_kws
        # Update topic if new one is more specific (longer = more specific)
        if new_topic and len(new_topic) > len(attrs.get("topic") or ""):
            attrs["topic"] = new_topic[:100]
        await _get_redis().execute_command("VSETATTR", mem_store.FACT_KEY, element_id, json.dumps(attrs))
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
                    _get_redis(), old_ent, new_ent, "contradicts",
                    confidence=0.9, source="contradiction_detector", bidirectional=True,
                )

    # Mark old fact as superseded
    await mem_store.soft_delete_fact(_get_redis(), element_id, "contradicted", reason="contradicted")
    log.info(f"[contradiction] fact {element_id} superseded: contradicted by new fact")


# ── Background store ──────────────────────────────────────────────────────────
async def _do_store(messages: list[Message], session_id: str) -> None:
    global _store_attempts, _store_successes, _store_skips, _store_errors, _store_latency_sum_ms
    if _shutting_down:
        return
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
        ep_emb = await asyncio.to_thread(_encode, summary[:500])
        similar = await mem_store.knn_search(_get_redis(), mem_store.EPISODE_KEY, ep_emb, k=1)
        is_dup = bool(similar and similar[0].get("score", 0.0) > 0.95)
        ep_saved = 0
        new_ep_uid = ""
        prev_ep_id = ""
        if not is_dup:
            # Causal chain: look up the previous episode for this session
            prev_ep_id = await mem_store.get_last_episode_id(_get_redis(), session_id)
            new_ep_uid = await mem_store.save_episode(
                _get_redis(), session_id, turn_text[:2000], ep_emb, lang, domain,
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

        # Batch-encode all fact contents off the event loop in one pass
        # instead of calling _encode() per fact (N sequential MiniLM inferences)
        fact_contents = [f.content for f in facts if not _contains_secret(f.content)]
        fact_embs = await _encode_batch_async(fact_contents)
        emb_iter = iter(fact_embs)

        for fact in facts:
            if _contains_secret(fact.content):
                continue
            f_emb = next(emb_iter)
            existing = await mem_store.knn_search(_get_redis(), mem_store.FACT_KEY, f_emb, k=1)
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
                _get_redis(), fact.content, fact.category, fact.confidence, f_emb, lang, domain,
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
            await persona_mod.update(_get_redis(), fact.category, fact.content)
            fact_saved += 1

        # 5b. Back-fill ep_type on newly saved episode + complete causal chain
        if new_ep_uid:
            ep_type = _infer_ep_type(facts)
            # Update ep_type attr on the episode we just saved
            try:
                raw = await _get_redis().execute_command("VGETATTR", mem_store.EPISODE_KEY, new_ep_uid)
                if raw:
                    ep_attrs = _decode_attrs(raw)
                    ep_attrs["ep_type"] = ep_type
                    await _get_redis().execute_command(
                        "VSETATTR", mem_store.EPISODE_KEY, new_ep_uid, json.dumps(ep_attrs)
                    )
            except Exception:
                pass  # VGETATTR not critical — ep_type stays "general"
            # Complete doubly-linked chain: tell prev episode its successor
            # (prev_ep_id was already fetched before save_episode above)
            if prev_ep_id and prev_ep_id != new_ep_uid:
                _spawn(
                    mem_store.update_episode_next_id(_get_redis(), prev_ep_id, new_ep_uid),
                    "ep-chain",
                )
            # Advance the session's last-episode pointer
            await mem_store.set_last_episode_id(_get_redis(), session_id, new_ep_uid)
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
        # Batch-encode procedure contents off the event loop
        if proc_facts:
            proc_contents = [pf.content for pf in proc_facts]
            proc_embs = await _encode_batch_async(proc_contents)
            for pf, p_emb in zip(proc_facts, proc_embs):
                existing = await mem_store.knn_search(_get_redis(), mem_store.PROC_KEY, p_emb, k=1)
                if not existing or existing[0].get("score", 0.0) < 0.90:
                    content_lower = pf.content.lower()
                    tools_in_proc = [t for t in _KNOWN_TOOLS if t in content_lower]
                    await mem_store.save_procedure(
                        _get_redis(), task=pf.content, procedure=pf.content,
                        embedding=p_emb, tools_used=tools_in_proc[:8],
                        domain=domain, language=lang,
                    )

        # 7. Record knowledge graph edges from persons + entities co-occurrences
        for fact in facts:
            if fact.persons or fact.entities:
                all_ents = (fact.persons or []) + (fact.entities or [])
                if len(all_ents) >= 2:
                    await graph_mod.record_entities(_get_redis(), all_ents, [])

        # 8. Accumulate rolling session summary (Tier 1 — layered controlled architecture)
        #    Append this turn's summary instead of overwriting, so Tier 1 grows across
        #    the whole session rather than holding only the last turn.
        await _accumulate_session(session_id, summary)

        # 9. Auto-consolidation: trigger after every settings.auto_consolidate_every stored facts
        global _stores_since_consolidation
        fact_count = await _stores_since_consolidation.increment(fact_saved)
        if fact_count >= settings.auto_consolidate_every:
            await _stores_since_consolidation.reset()
            _spawn(_do_consolidate(_get_redis(), _bm25_index, _spawn), "consolidate")
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
    existing = await mem_store.get_session_context(_get_redis(), session_id) or ""
    if not existing:
        await mem_store.set_session_context(_get_redis(), session_id, turn_summary[:1500])
        return
    combined = f"{existing}\n---\n{turn_summary}"
    if len(combined) > 1200:
        # MemAgent-style incremental overwrite: LLM decides what to keep
        combined = await summarizer.overwrite_update(existing, turn_summary, target_chars=900)
    await mem_store.set_session_context(_get_redis(), session_id, combined[:1500])


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

    ctx = await mem_store.get_session_context(_get_redis(), session_id)
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
    ep_emb  = await asyncio.to_thread(_encode, session_summary[:500])
    similar = await mem_store.knn_search(_get_redis(), mem_store.EPISODE_KEY, ep_emb, k=1)
    ep_saved = 0
    if not similar or similar[0].get("score", 0.0) < 0.95:
        await mem_store.save_episode(
            _get_redis(), session_id,
            f"[Session Summary] {session_summary[:2000]}",
            ep_emb, lang, domain,
        )
        ep_saved = 1

    # Hybrid fact extraction on the full session text
    facts = await extractor.extract_hybrid(
        [{"role": "user", "content": ctx}], ctx
    )
    fact_saved = 0
    # Batch-encode all fact contents off the event loop
    non_secret_facts = [f for f in facts if not _contains_secret(f.content)]
    if non_secret_facts:
        fact_contents = [f.content for f in non_secret_facts]
        fact_embs = await _encode_batch_async(fact_contents)
        emb_iter = iter(fact_embs)
    else:
        emb_iter = iter([])
    for fact in facts:
        if _contains_secret(fact.content):
            continue
        f_emb = next(emb_iter)
        existing = await mem_store.knn_search(_get_redis(), mem_store.FACT_KEY, f_emb, k=1)
        if existing and existing[0].get("score", 0.0) > 0.95:
            if fact.keywords and existing[0].get("score", 0.0) > 0.80:
                _spawn(_evolve_similar_fact(
                    existing[0].get("_element", ""),
                    fact.keywords, fact.topic,
                ))
            continue
        uid = await mem_store.save_fact(
            _get_redis(), fact.content, fact.category, fact.confidence,
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
        await persona_mod.update(_get_redis(), fact.category, fact.content)
        fact_saved += 1

    # Delete Tier 1 KV — session has been crystallised into long-term memory
    await _get_redis().delete(f"{mem_store.SESSION_PRE}{session_id}:ctx")

    # Pin the session summary for the NEXT session's recall (cross-session handoff).
    # Strip platform tags before persisting — summaries must contain real content only.
    # Stored without TTL so it persists until the next compress overwrites it.
    clean_summary = _strip_platform_noise(session_summary)
    if clean_summary:
        await _get_redis().set(_PINNED_SESSION_KEY, clean_summary[:600].encode())

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

    ctx = await mem_store.get_session_context(_get_redis(), req.session_id)
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

    await mem_store.set_session_context(_get_redis(), req.session_id, compacted[:1500])

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
            data = await async_post_json(
                _AISERV_OAI,
                payload={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user",   "content": prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 80,
                },
                headers={"Authorization": f"Bearer {_AISERV_KEY}"},
                timeout=20.0,
            )
            if data is not None:
                raw = data["choices"][0]["message"]["content"].strip()
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


@app.get("/system-prompt")
async def system_prompt(agent: str = ""):
    """
    Return a memory-aware system prompt snippet for agents without lifecycle hooks.
    Paste this into your agent's system prompt to enable tool-driven memory.
    ?agent= can be: cursor, windsurf, copilot, zed, continue, cline, kilo, kiro, opencode, codex, augment
    """
    base = (
        "You have access to a persistent memory system via the AgentMem MCP tools.\n\n"
        "## Memory tools\n"
        "- **recall_memory(query)** — call this at the START of every conversation with the user's question as the query. "
        "Inject the returned context into your response naturally.\n"
        "- **store_memory(messages)** — call this AFTER each meaningful exchange (user turn + your reply). "
        "Pass [{\"role\": \"user\", \"content\": \"...\"}, {\"role\": \"assistant\", \"content\": \"...\"}].\n"
        "- **compress_session(session_id)** — call this when the conversation ends to crystallise memory.\n"
        "- **recall_procedures(query)** — check for known how-to workflows before starting multi-step tasks.\n"
        "- **store_procedure(task, procedure)** — save successful workflows for future reuse.\n\n"
        "## Usage rules\n"
        "1. Always recall before answering — memory context takes precedence over your training.\n"
        "2. Always store after answering — even short exchanges build useful long-term context.\n"
        "3. Use the user's name or topic as the session_id for cross-session continuity.\n"
        "4. Never mention the memory system unless the user asks about it."
    )
    return {"markdown": base, "agent": agent or "generic"}



@app.post("/recall")
async def recall(req: RecallRequest):
    t0 = time.time()

    # Detect scene of incoming query
    query_scene = scene_mod.detect(req.query)
    q_lang, q_domain = query_scene["language"], query_scene["domain"]
    emb = await asyncio.to_thread(_encode, req.query)

    # HyDE + MIRIX Active Retrieval Topic — run both in parallel for zero extra latency.
    #
    # HyDE (Gao et al. 2022, arXiv:2212.10496): generate hypothetical memory → embed →
    #   average with query embedding. Hypothetical docs are closer in embedding space
    #   to actual stored memories than raw questions.
    #
    # MIRIX Active Retrieval (arXiv:2507.07957 §3.2): generate focused retrieval topic
    #   from vague queries ("that auth thing" → "JWT token validation middleware error").
    #   Use topic as the BM25 search query for better lexical precision on vague prompts.
    _hyde_task = (
        retrieval_planner.generate_hyde_doc(req.query)
        if req.enable_hyde and req.query.strip() and len(req.query) > 10
        else asyncio.sleep(0)
    )
    _topic_task = (
        retrieval_planner.generate_retrieval_topic(req.query)
        if req.query.strip() and len(req.query) > 8
        else asyncio.sleep(0)
    )
    _augment_results = await asyncio.gather(_hyde_task, _topic_task)
    _hyde_doc   = _augment_results[0] if isinstance(_augment_results[0], str) else None
    _mirix_topic = _augment_results[1] if isinstance(_augment_results[1], str) else None

    if _hyde_doc:
        hyde_emb = await asyncio.to_thread(_encode, _hyde_doc)
        merged = (emb * 0.5 + hyde_emb * 0.5).astype(np.float32)
        norm = np.linalg.norm(merged)
        emb = merged / norm if norm > 0 else merged

    # Use MIRIX topic for BM25 search when available (vague queries benefit most)
    _bm25_query = _mirix_topic if _mirix_topic else req.query

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
            _get_redis(), vset, emb, k, filter_expr=fexpr, bump_heat=True
        )

    # Detect if this is a capability query (needs tool context)
    is_cap_query = query_scene.get("is_capability_query", False)

    # Parallel: persona + env + session ctx + scene-filtered facts + episodes
    # + tool recall if capability query or include_tools=True
    # + symbolic structured pass (async, no extra latency via gather)
    # + pinned last-session summary (cross-session handoff bridge, index 6)
    gather_tasks = [
        persona_mod.get_context(_get_redis()),
        cap_mod.get_env_context(_get_redis()),
        mem_store.get_session_context(_get_redis(), req.session_id),
        _fetch(mem_store.FACT_KEY,    n_facts,    lang_filter_sym or None),
        _fetch(mem_store.EPISODE_KEY, n_episodes, lang_filter_sym or None),
        retrieval_planner.analyze_query_structure(req.query) if (req.query.strip() and req.enable_planning) else asyncio.sleep(0),  # symbolic layer (index 5) — LLM call only when enable_planning=True
        _get_redis().get(_PINNED_SESSION_KEY),                       # handoff bridge (index 6)
    ]
    if req.include_tools and is_cap_query:
        gather_tasks.append(cap_mod.recall_tools(_get_redis(), emb, k=6))
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
            results = await _get_redis().execute_command(*cmd)
        except Exception:
            return []
        procs = []
        idx = 0
        while idx + 2 < len(results):
            elem = results[idx]; score = results[idx+1]; raw = results[idx+2]
            idx += 3
            elem_str = _decode_bytes(elem)
            if elem_str == "__seed__":
                continue
            try:
                attrs = _decode_attrs(raw)
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
    
    # Crystallized digests (index 8): fetch recent auto-crystallized session summaries.
    # Uses a sorted set index (mem:crystallized_index, score=crystallized_at) instead of
    # SCAN so lookup is O(log N) regardless of total key count.
    async def _fetch_crystallized_digests() -> list[dict]:
        """Fetch top-3 most recent crystallized session digests via sorted-set index."""
        try:
            top_keys = await _get_redis().zrevrange(_CRYSTALLIZED_INDEX_KEY, 0, 9)
            if not top_keys:
                return []

            pipe = _get_redis().pipeline(transaction=False)
            for key in top_keys:
                pipe.get(key)
            raw_values = await pipe.execute()

            digests = []
            for raw in raw_values:
                if not raw:
                    continue
                try:
                    digest = _decode_json(raw)
                    digests.append(digest)
                except Exception:
                    continue

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
        _decode_bytes(_pinned_raw)
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
        ent_vecs = await _encode_batch_async(sym_entities[:3])
        sym_tasks = [
            mem_store.knn_search(_get_redis(), mem_store.FACT_KEY, vec, k=3,
                                 filter_expr=lang_filter_sym or None)
            for vec in ent_vecs
        ]
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
            # Batch-encode all planning queries off the event loop
            extra_embs = await _encode_batch_async(extra_queries)
            extra_tasks = [
                mem_store.knn_search(_get_redis(), mem_store.FACT_KEY, emb, n_facts,
                                     filter_expr=lang_filter_sym or None, bump_heat=True)
                for emb in extra_embs
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

    # Supplement with global search if scene results are sparse.
    # BM25 is pure in-memory (no I/O) but async — fold it into this gather so it
    # runs concurrently with the Redis supplement fetches at zero extra latency cost.
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

    supplement_tasks.append(_bm25_index.search(_bm25_query, k=n_facts))

    supp_results = await asyncio.gather(*supplement_tasks)
    facts_global = supp_results[0] if isinstance(supp_results[0], list) else []
    eps_global   = supp_results[1] if isinstance(supp_results[1], list) else []
    bm25_facts   = supp_results[2] if isinstance(supp_results[2], list) else []
    if bm25_facts:
        log.debug(f"[bm25] {len(bm25_facts)} hits for: {req.query[:60]!r}")

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

    # Session diversity: cap at 3 results per source session to prevent
    # one verbose session drowning out facts from other sessions.
    facts    = _session_diversify(facts,    max_per_session=3)
    episodes = _session_diversify(episodes, max_per_session=4)

    # SimpleMem reflection loop (opt-in): check if results sufficient, if not re-fetch
    if req.enable_reflection and facts:
        is_sufficient = await retrieval_planner.check_sufficiency(req.query, facts)
        if not is_sufficient:
            # One additional targeted pass with a different angle
            extra_q = await retrieval_planner.plan_queries(
                f"Find additional information about: {req.query}"
            )
            if extra_q:
                reflection_emb = await asyncio.to_thread(_encode, extra_q[0])
                reflection_facts = await mem_store.knn_search(
                    _get_redis(), mem_store.FACT_KEY, reflection_emb,
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
                graph_mod.entity_recall(_get_redis(), ent, emb, k=2)
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
            bridge_facts = await graph_mod.find_bridge_nodes(_get_redis(), facts, emb, k=3)
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
                all_tig = await _get_redis().hgetall(mem_store.TOOL_GRAPH_KEY)
                prefix  = f"{top_tool_elem}:"
                trans: dict[str, int] = {}
                for k_b, v_b in all_tig.items():
                    key_s = _decode_bytes(k_b)
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
        token_budget=req.token_budget or settings.default_token_budget,
        last_session_summary=_handoff,
        crystal_digests=crystal_digests or None,  # Auto-crystallized lessons learned
    )
    ms = int((time.time() - t0) * 1000)

    planning_info = f" planning={req.enable_planning}" if req.enable_planning else ""
    hyde_info = f" hyde={bool(_hyde_doc)}" if req.enable_hyde else ""
    topic_info = f" topic={_mirix_topic[:30]!r}" if _mirix_topic else ""
    log.info(f"[recall] session={req.session_id} lang={q_lang} domain={q_domain} "
             f"facts={len(facts)} ep={len(episodes)} tools={len(tools_raw)} procs={len(proc_ctx)}"
             f"{planning_info}{hyde_info}{topic_info} {ms}ms")

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
    ctx = await mem_store.get_session_context(_get_redis(), session_id)
    return {
        "session_id": session_id,
        "context":    ctx,
        "length":     len(ctx) if ctx else 0,
        "tier":       "Tier 1 / Session KV",
    }


# ── Consolidation logic now in services/consolidation_service.py ──────────────
# ═══════════════════════════════════════════════════════════════════════════════
# Absorbed agentmemory scaffold endpoints (no /agentmemory prefix)
# These provide full compatibility with rohitg00/agentmemory hooks & MCP tools.
# ═══════════════════════════════════════════════════════════════════════════════
# _compat_sid and _compat_check_auth are now in api/compat.py — imported above.
# _observe_internal is now in services/store_service.py — imported by route modules.


# ── Automatic Crystallization Background Task ────────────────────────────────
# _crystallize_session_inline now in services/consolidation_service.py

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
                cursor, keys = await _get_redis().scan(cursor, match=session_pattern, count=100)
                all_session_keys.extend(keys)
                if cursor == 0:
                    break

            if not all_session_keys:
                continue

            # Batch check which sessions are already crystallized + get session data
            # Pipeline: exists(crystal_key) + get(session_key) per session
            pipe = _get_redis().pipeline(transaction=False)
            key_map = []  # track (session_key, session_id, crystal_key) for each pipeline pair
            for key in all_session_keys:
                key_str = _decode_bytes(key)
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
                    session_obj = _decode_json(session_data)
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
            scanned = await _vscan(_get_redis(), mem_store.FACT_KEY, max_count=200)
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
                    await _get_redis().setex(
                        crystal_key,
                        90 * 86400,
                        json.dumps(digest, ensure_ascii=False)
                    )
                    # Maintain sorted-set index so /recall can use ZREVRANGE instead of SCAN
                    await _get_redis().zadd(
                        _CRYSTALLIZED_INDEX_KEY,
                        {crystal_key: digest["crystallized_at"]},
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


# ── Viewer ────────────────────────────────────────────────────────────────────

@app.get("/viewer")
async def compat_viewer():
    idx = _os.path.join(_static_dir, "index.html")
    if _os.path.exists(idx):
        return FileResponse(idx)
    return {"message": "AgentMem Viewer", "version": "1.2.0"}
