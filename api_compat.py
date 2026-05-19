"""
agentmemory API compatibility layer — v1.2.0

Implements the /agentmemory/* REST API contract from rohitg00/agentmemory
so their hooks, plugins, and connectors work directly with our Python/Redis backend.

Hook-critical endpoints (called by their 12 hooks):
  POST /agentmemory/session/start     — session-start hook
  POST /agentmemory/session/end       — session-end hook
  POST /agentmemory/observe           — post-tool-use, prompt-submit, subagent hooks
  POST /agentmemory/summarize         — stop hook
  POST /agentmemory/enrich            — pre-tool-use hook
  POST /agentmemory/context           — pre-compact hook
  POST /agentmemory/session/commit    — post-commit hook
  POST /agentmemory/claude-bridge/sync — pre-compact, session-end hooks
  POST /agentmemory/consolidate-pipeline — session-end hook
  POST /agentmemory/crystals/auto     — session-end hook

MCP tool endpoints (called by their 53 MCP tools):
  POST /agentmemory/search            — memory_recall
  POST /agentmemory/remember          — memory_save
  POST /agentmemory/forget            — memory_forget
  POST /agentmemory/compress-file     — memory_compress_file
  GET  /agentmemory/sessions          — list sessions
  GET  /agentmemory/observations      — list observations
  POST /agentmemory/patterns          — memory_patterns
  POST /agentmemory/file-context      — memory_file_history
  POST /agentmemory/smart-search      — smart search
  POST /agentmemory/timeline          — timeline

Utility endpoints:
  GET  /agentmemory/livez             — liveness probe
  GET  /agentmemory/health            — health check
  GET  /agentmemory/config/flags      — feature flags
  GET  /agentmemory/viewer            — web viewer
  GET  /agentmemory/profile           — user profile
  GET  /agentmemory/memories          — list memories
  GET  /agentmemory/export            — export data
  POST /agentmemory/import            — import data
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Request
from pydantic import BaseModel

log = logging.getLogger("mem.compat")

router = APIRouter(prefix="/agentmemory", tags=["agentmemory-compat"])

_redis = None


def _get_redis():
    """Get Redis client — prefers the shared pool from store.get_client().

    Falls back to the module-level _redis set by init_compat() for backward
    compatibility. This avoids the circular import from main._redis.
    """
    global _redis
    if _redis is None:
        from core import store as mem_store
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # Can't await get_client() in sync context — return None
            log.warning("[compat] Redis not initialized — call init_compat() at startup")
            return None
    return _redis


async def _r():
    """Async helper: get the Redis client via the shared connection pool."""
    from core import store as mem_store
    return await mem_store.get_client()


async def init_compat(redis_client) -> None:
    global _redis
    _redis = redis_client
    log.info("[compat] agentmemory API compatibility layer initialized")


def _check_auth(request: Request) -> dict | None:
    secret = os.getenv("AGENTMEMORY_SECRET", "")
    if not secret:
        return None
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {secret}":
        return {"status_code": 401, "error": "unauthorized"}
    return None


# ── Request models ─────────────────────────────────────────────────────────────

class SessionStartRequest(BaseModel):
    sessionId: str = ""
    session_id: str = ""
    project: str = ""
    cwd: str = ""

class SessionEndRequest(BaseModel):
    sessionId: str = ""
    session_id: str = ""
    project: str = ""
    cwd: str = ""

class ObserveRequest(BaseModel):
    hookType: str = ""
    sessionId: str = ""
    session_id: str = ""
    project: str = ""
    cwd: str = ""
    timestamp: str = ""
    data: dict = {}

class SummarizeRequest(BaseModel):
    sessionId: str = ""
    session_id: str = ""
    project: str = ""
    cwd: str = ""

class EnrichRequest(BaseModel):
    sessionId: str = ""
    session_id: str = ""
    project: str = ""
    cwd: str = ""
    files: list[str] = []
    query: str = ""

class ContextRequest(BaseModel):
    sessionId: str = ""
    session_id: str = ""
    project: str = ""
    cwd: str = ""
    query: str = ""
    token_budget: int = 1500

class SessionCommitRequest(BaseModel):
    sessionId: str = ""
    session_id: str = ""
    sha: str = ""
    message: str = ""
    branch: str = ""
    repo: str = ""
    cwd: str = ""

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    format: str = "full"
    token_budget: int | None = None
    sessionId: str = ""
    session_id: str = ""

class RememberRequest(BaseModel):
    content: str
    type: str = "fact"
    concepts: str = ""
    files: str = ""
    sessionId: str = ""
    session_id: str = ""

class ForgetRequest(BaseModel):
    query: str
    limit: int = 10
    dry_run: bool = True

class FileContextRequest(BaseModel):
    files: str
    sessionId: str = ""
    session_id: str = ""

class PatternsRequest(BaseModel):
    query: str = ""
    limit: int = 10

class SmartSearchRequest(BaseModel):
    query: str
    limit: int = 10
    depth: str = "auto"

class TimelineRequest(BaseModel):
    sessionId: str = ""
    session_id: str = ""
    from_: str = ""
    to: str = ""
    limit: int = 20

class ConsolidatePipelineRequest(BaseModel):
    sessionId: str = ""
    session_id: str = ""

class ClaudeBridgeSyncRequest(BaseModel):
    sessionId: str = ""
    session_id: str = ""
    project: str = ""
    cwd: str = ""

class CrystalsAutoRequest(BaseModel):
    sessionId: str = ""
    session_id: str = ""

class ImportRequest(BaseModel):
    data: dict = {}
    version: str = ""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sid(req) -> str:
    return req.sessionId or req.session_id or f"ses_{int(time.time())}"

async def _observe_internal(session_id: str, project: str, cwd: str,
                            hook_type: str, data: dict) -> dict:
    from main import _encode, _admission_gate
    main_redis = await _r()

    r = main_redis
    if r is None:
        return {"status": "error", "error": "redis not connected"}

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

    from main import _is_trivial, _is_injected, _contains_secret, _redact_secrets
    if _is_trivial(content) or _is_injected(content):
        return {"status": "ok", "action": "skipped_filter"}
    if _contains_secret(content):
        content = _redact_secrets(content)

    if not await _admission_gate(content):
        return {"status": "ok", "action": "skipped_gate"}

    emb = _encode(content)

    from core import store as mem_store
    uid = await mem_store.save_episode(
        r, session_id, content, emb,
        ep_type=hook_type,
    )

    try:
        from core import extractor
        facts = await extractor.extract_facts(content, session_id)
        for fact_text, fact_attrs in facts:
            fact_emb = _encode(fact_text)
            await mem_store.save_fact(
                r,
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
        log.warning(f"[compat.observe] fact extraction failed: {e}")

    return {"status": "ok", "action": "observed", "id": uid}


# ── Liveness / Health ─────────────────────────────────────────────────────────

@router.get("/livez")
async def livez():
    return {"status": "ok", "service": "agentmemory"}


@router.get("/health")
async def health():
    main_redis = await _r()
    try:
        await main_redis.ping()
        return {"status": "ok", "redis": "connected"}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@router.get("/config/flags")
async def config_flags():
    return {
        "graphExtractionEnabled": os.getenv("GRAPH_EXTRACTION_ENABLED", "false").lower() == "true",
        "consolidationEnabled": True,
        "autoCompressEnabled": True,
        "contextInjectionEnabled": os.getenv("AGENTMEMORY_INJECT_CONTEXT", "true").lower() == "true",
    }


# ── Session lifecycle ─────────────────────────────────────────────────────────

@router.post("/session/start")
async def session_start(req: SessionStartRequest, request: Request):
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    sid = _sid(req)
    project = req.project or req.cwd or os.getcwd()

    main_redis = await _r()
    from core import store as mem_store

    session_key = f"mem:session:{sid}"
    await main_redis.set(session_key, json.dumps({
        "session_id": sid,
        "project": project,
        "started_at": time.time(),
        "observations": 0,
    }), ex=14400)

    from core import persona as persona_mod
    persona_ctx = await persona_mod.get_context(main_redis)

    pinned_raw = await main_redis.get("mem:pinned:session_summary")
    last_summary = pinned_raw.decode() if pinned_raw else None

    context_parts = []
    if persona_ctx:
        context_parts.append(persona_ctx)
    if last_summary:
        context_parts.append(f"## Last Session Summary\n{last_summary[:600]}")

    context = "\n\n".join(context_parts) if context_parts else None

    return {"status": "ok", "sessionId": sid, "context": context}


@router.post("/session/end")
async def session_end(req: SessionEndRequest, request: Request, background_tasks: BackgroundTasks):
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    sid = _sid(req)

    async def _do_end():
        from main import _do_compress_session
        main_redis = await _r()
        try:
            await _do_compress_session(sid)
        except Exception as e:
            log.warning(f"[compat.session/end] compress failed: {e}")

    background_tasks.add_task(_do_end)
    return {"status": "ok", "sessionId": sid}


# ── Observe (most-called endpoint — all post-* hooks) ─────────────────────────

@router.post("/observe")
async def observe(req: ObserveRequest, request: Request, background_tasks: BackgroundTasks):
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    sid = _sid(req)
    project = req.project or req.cwd or ""
    hook_type = req.hookType or "observe"
    data = req.data or {}

    async def _do_observe():
        try:
            await _observe_internal(sid, project, req.cwd or "", hook_type, data)
        except Exception as e:
            log.warning(f"[compat.observe] background error: {e}")

    background_tasks.add_task(_do_observe)
    return {"status": "ok"}


# ── Summarize (stop hook) ─────────────────────────────────────────────────────

@router.post("/summarize")
async def summarize(req: SummarizeRequest, request: Request, background_tasks: BackgroundTasks):
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    sid = _sid(req)

    async def _do_summarize():
        from main import _do_compress_session
        main_redis = await _r()
        try:
            await _do_compress_session(sid)
        except Exception as e:
            log.warning(f"[compat.summarize] compress failed: {e}")

    background_tasks.add_task(_do_summarize)
    return {"status": "ok", "sessionId": sid}


# ── Enrich (pre-tool-use hook) ────────────────────────────────────────────────

@router.post("/enrich")
async def enrich(req: EnrichRequest, request: Request):
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    from main import _encode
    main_redis = await _r()
    from core import store as mem_store

    files = req.files or []
    query = req.query or ""

    file_contexts = []
    for f in files[:5]:
        fname = os.path.basename(f)
        emb = _encode(fname)
        results = await mem_store.knn_search(main_redis, mem_store.FACT_KEY, emb, k=3)
        for r in results:
            attrs = r.get("attrs", {})
            if attrs.get("content"):
                file_contexts.append(attrs["content"][:200])

    if query:
        emb = _encode(query)
        results = await mem_store.knn_search(main_redis, mem_store.FACT_KEY, emb, k=5)
        for r in results:
            attrs = r.get("attrs", {})
            if attrs.get("content"):
                file_contexts.append(attrs["content"][:200])

    context = "\n".join(file_contexts[:10]) if file_contexts else None
    return {"status": "ok", "context": context}


# ── Context (pre-compact hook) ────────────────────────────────────────────────

@router.post("/context")
async def context(req: ContextRequest, request: Request):
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    from main import _encode
    from main import _encode, _format_prepend
    main_redis = await _r()
    from core import store as mem_store, persona as persona_mod

    sid = _sid(req)
    query = req.query or ""
    budget = req.token_budget or 1500

    persona_ctx = await persona_mod.get_context(main_redis)
    session_ctx = await mem_store.get_session_context(main_redis, sid)

    pinned_raw = await main_redis.get("mem:pinned:session_summary")
    last_summary = pinned_raw.decode() if pinned_raw else None

    facts = []
    episodes = []
    if query:
        emb = _encode(query)
        facts = await mem_store.knn_search(main_redis, mem_store.FACT_KEY, emb, k=8)
        episodes = await mem_store.knn_search(main_redis, mem_store.EPISODE_KEY, emb, k=6)

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

@router.post("/session/commit")
async def session_commit(req: SessionCommitRequest, request: Request):
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    main_redis = await _r()

    sid = _sid(req)
    commit_key = f"mem:commit:{req.sha}"
    await main_redis.hset(commit_key, mapping={
        "sha": req.sha,
        "message": req.message,
        "branch": req.branch,
        "repo": req.repo,
        "session_id": sid,
        "timestamp": str(int(time.time() * 1000)),
    })

    return {"status": "ok", "sha": req.sha}


@router.get("/session/by-commit")
async def session_by_commit(sha: str = "", request: Request = None):
    main_redis = await _r()

    if not sha:
        return {"error": "sha parameter required"}

    commit_key = f"mem:commit:{sha}"
    data = await main_redis.hgetall(commit_key)
    if not data:
        return {"error": "commit not found", "sha": sha}

    result = {k.decode() if isinstance(k, bytes) else k:
              v.decode() if isinstance(v, bytes) else v
              for k, v in data.items()}
    return result


@router.get("/commits")
async def commits(branch: str = "", repo: str = "", limit: int = 20):
    main_redis = await _r()

    cursor, keys = await main_redis.scan(match="mem:commit:*", count=limit)
    results = []
    for key in keys:
        k = key.decode() if isinstance(key, bytes) else key
        data = await main_redis.hgetall(k)
        if data:
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

@router.post("/claude-bridge/sync")
async def claude_bridge_sync(req: ClaudeBridgeSyncRequest, request: Request):
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    from main import _encode
    from main import _encode, _format_prepend
    main_redis = await _r()
    from core import store as mem_store, persona as persona_mod

    sid = _sid(req)
    project = req.project or req.cwd or ""

    persona_ctx = await persona_mod.get_context(main_redis)
    session_ctx = await mem_store.get_session_context(main_redis, sid)

    pinned_raw = await main_redis.get("mem:pinned:session_summary")
    last_summary = pinned_raw.decode() if pinned_raw else None

    emb = _encode(project or "project context")
    facts = await mem_store.knn_search(main_redis, mem_store.FACT_KEY, emb, k=10)
    episodes = await mem_store.knn_search(main_redis, mem_store.EPISODE_KEY, emb, k=6)

    prepend = _format_prepend(
        facts=facts,
        episodes=episodes,
        session_ctx=session_ctx,
        persona_ctx=persona_ctx or "",
        token_budget=2000,
        last_session_summary=last_summary,
    )

    memory_md_path = os.path.expanduser("~/.claude/AGENTMEM_MEMORY.md")
    if prepend:
        try:
            os.makedirs(os.path.dirname(memory_md_path), exist_ok=True)
            with open(memory_md_path, "w") as f:
                f.write(prepend)
        except Exception as e:
            log.warning(f"[compat.claude-bridge] write failed: {e}")

    return {"status": "ok", "memoryPath": memory_md_path if prepend else None}


# ── Consolidate pipeline ──────────────────────────────────────────────────────

@router.post("/consolidate-pipeline")
async def consolidate_pipeline(req: ConsolidatePipelineRequest, request: Request,
                                background_tasks: BackgroundTasks):
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    async def _do_consolidate():
        from main import _do_consolidate
        main_redis = await _r()
        try:
            await _do_consolidate()
        except Exception as e:
            log.warning(f"[compat.consolidate-pipeline] error: {e}")

    background_tasks.add_task(_do_consolidate)
    return {"status": "ok"}


# ── Crystals auto ─────────────────────────────────────────────────────────────

@router.post("/crystals/auto")
async def crystals_auto(req: CrystalsAutoRequest, request: Request):
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err
    return {"status": "ok", "crystals": []}


# ── Search (MCP memory_recall) ────────────────────────────────────────────────

@router.post("/search")
async def search(req: SearchRequest, request: Request):
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    from main import _encode
    from main import _encode, _format_prepend
    main_redis = await _r()
    from core import store as mem_store, persona as persona_mod

    sid = _sid(req)
    emb = _encode(req.query)

    facts = await mem_store.knn_search(main_redis, mem_store.FACT_KEY, emb, k=req.limit)
    episodes = await mem_store.knn_search(main_redis, mem_store.EPISODE_KEY, emb, k=req.limit)

    persona_ctx = await persona_mod.get_context(main_redis)
    session_ctx = await mem_store.get_session_context(main_redis, sid)

    pinned_raw = await main_redis.get("mem:pinned:session_summary")
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

    if req.format == "narrative":
        prepend = _format_prepend(
            facts=facts, episodes=episodes,
            session_ctx=session_ctx, persona_ctx=persona_ctx or "",
            token_budget=budget, last_session_summary=last_summary,
        )
        return {"context": prepend}

    prepend = _format_prepend(
        facts=facts, episodes=episodes,
        session_ctx=session_ctx, persona_ctx=persona_ctx or "",
        token_budget=budget, last_session_summary=last_summary,
    )
    return {"context": prepend, "facts": len(facts), "episodes": len(episodes)}


# ── Remember (MCP memory_save) ────────────────────────────────────────────────

@router.post("/remember")
async def remember(req: RememberRequest, request: Request):
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    from main import _encode
    main_redis = await _r()
    from core import store as mem_store

    sid = _sid(req)
    content = req.content
    emb = _encode(content)

    uid = await mem_store.save_fact(
        main_redis,
        content=content,
        category=req.type or "fact",
        confidence=0.8,
        embedding=emb,
        keywords=[c.strip() for c in req.concepts.split(",") if c.strip()] if req.concepts else None,
        importance=0.8,
    )

    return {"status": "ok", "id": uid}


# ── Forget (MCP memory_forget) ────────────────────────────────────────────────

@router.post("/forget")
async def forget(req: ForgetRequest, request: Request):
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    from main import _encode
    main_redis = await _r()
    from core import store as mem_store

    emb = _encode(req.query)
    facts = await mem_store.knn_search(main_redis, mem_store.FACT_KEY, emb, k=req.limit)

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
            await main_redis.execute_command("VREM", mem_store.FACT_KEY, elem)
            deleted += 1

    return {"status": "ok", "deleted": deleted}


# ── Compress file ─────────────────────────────────────────────────────────────

@router.post("/compress-file")
async def compress_file(filePath: str = "", request: Request = None):
    if not filePath or not os.path.isfile(filePath):
        return {"error": "file not found", "path": filePath}

    try:
        with open(filePath, "r") as f:
            content = f.read()

        backup_path = filePath.replace(".md", ".original.md")
        if not os.path.exists(backup_path):
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

@router.get("/sessions")
async def sessions(limit: int = 20):
    main_redis = await _r()

    cursor, keys = await main_redis.scan(match="mem:session:*", count=limit)
    results = []
    for key in keys:
        k = key.decode() if isinstance(key, bytes) else key
        raw = await main_redis.get(k)
        if raw:
            try:
                data = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                results.append(data)
            except Exception:
                pass
    results.sort(key=lambda x: x.get("started_at", 0), reverse=True)
    return {"sessions": results[:limit]}


# ── Observations list ─────────────────────────────────────────────────────────

@router.get("/observations")
async def observations(sessionId: str = "", limit: int = 20):
    main_redis = await _r()
    from core import store as mem_store

    import numpy as np
    seed = np.zeros(mem_store.DIMS, dtype=np.float32)
    card = await main_redis.execute_command("VCARD", mem_store.EPISODE_KEY)
    if not card or int(card) <= 1:
        return {"observations": []}

    results_raw = await main_redis.execute_command(
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

@router.post("/file-context")
async def file_context(req: FileContextRequest, request: Request):
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    from main import _encode
    main_redis = await _r()
    from core import store as mem_store

    files = [f.strip() for f in req.files.split(",") if f.strip()]
    sid = _sid(req)

    all_results = []
    for f in files[:10]:
        fname = os.path.basename(f)
        emb = _encode(fname)
        results = await mem_store.knn_search(main_redis, mem_store.EPISODE_KEY, emb, k=5,
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

@router.post("/patterns")
async def patterns(req: PatternsRequest, request: Request):
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    from main import _encode
    main_redis = await _r()
    from core import store as mem_store

    query = req.query or "recurring patterns"
    emb = _encode(query)
    facts = await mem_store.knn_search(main_redis, mem_store.FACT_KEY, emb, k=req.limit)

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

@router.post("/smart-search")
async def smart_search(req: SmartSearchRequest, request: Request):
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    from main import _encode, _rrf_merge
    main_redis = await _r()
    from core import store as mem_store

    emb = _encode(req.query)

    facts = await mem_store.knn_search(main_redis, mem_store.FACT_KEY, emb, k=req.limit)
    episodes = await mem_store.knn_search(main_redis, mem_store.EPISODE_KEY, emb, k=req.limit)

    merged = _rrf_merge([facts, episodes], weights=[1.0, 0.8], limit=req.limit)

    return {
        "results": [{
            "content": m.get("content", "")[:300],
            "category": m.get("attrs", {}).get("category", ""),
            "score": round(m.get("score", 0), 4),
        } for m in merged],
    }


# ── Timeline ──────────────────────────────────────────────────────────────────

@router.post("/timeline")
async def timeline(req: TimelineRequest, request: Request):
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    main_redis = await _r()
    from core import store as mem_store

    import numpy as np
    seed = np.zeros(mem_store.DIMS, dtype=np.float32)
    card = await main_redis.execute_command("VCARD", mem_store.EPISODE_KEY)
    if not card or int(card) <= 1:
        return {"timeline": []}

    results_raw = await main_redis.execute_command(
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

@router.get("/profile")
async def profile():
    main_redis = await _r()
    from core import persona as persona_mod

    ctx = await persona_mod.get_context(main_redis)
    return {"profile": ctx or ""}


# ── Export / Import ───────────────────────────────────────────────────────────

@router.get("/export")
async def export_data():
    main_redis = await _r()
    from core import store as mem_store

    import numpy as np
    export = {"version": "1.2.0", "exported_at": time.time(), "facts": [], "episodes": []}

    for key, label in [(mem_store.FACT_KEY, "facts"), (mem_store.EPISODE_KEY, "episodes")]:
        card = await main_redis.execute_command("VCARD", key)
        if not card or int(card) <= 1:
            continue
        seed = np.zeros(mem_store.DIMS, dtype=np.float32)
        results = await main_redis.execute_command(
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


@router.post("/import")
async def import_data(req: ImportRequest, request: Request):
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    from main import _encode
    main_redis = await _r()
    from core import store as mem_store

    data = req.data
    imported = 0

    for item in data.get("episodes", []):
        content = item.get("content", "")
        if not content:
            continue
        emb = _encode(content)
        await mem_store.save_episode(
            main_redis,
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
            main_redis,
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

@router.get("/memories")
async def memories(limit: int = 50, category: str = ""):
    main_redis = await _r()
    from core import store as mem_store

    import numpy as np
    seed = np.zeros(mem_store.DIMS, dtype=np.float32)
    card = await main_redis.execute_command("VCARD", mem_store.FACT_KEY)
    if not card or int(card) <= 1:
        return {"memories": []}

    results_raw = await main_redis.execute_command(
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


@router.get("/memories/{memory_id}")
async def memory_detail(memory_id: str):
    main_redis = await _r()
    from core import store as mem_store

    try:
        raw = await main_redis.execute_command("VGETATTR", mem_store.FACT_KEY, memory_id)
        if raw:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            return {"id": memory_id, **attrs}
    except Exception:
        pass

    try:
        raw = await main_redis.execute_command("VGETATTR", mem_store.EPISODE_KEY, memory_id)
        if raw:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            return {"id": memory_id, **attrs}
    except Exception:
        pass

    return {"error": "memory not found", "id": memory_id}


# ── Semantic / Procedural / Relations (read-only list views) ──────────────────

@router.get("/semantic")
async def semantic(limit: int = 50):
    return await memories(limit=limit)


@router.get("/procedural")
async def procedural(limit: int = 50):
    main_redis = await _r()
    from core import store as mem_store

    import numpy as np
    seed = np.zeros(mem_store.DIMS, dtype=np.float32)
    card = await main_redis.execute_command("VCARD", mem_store.PROC_KEY)
    if not card or int(card) <= 1:
        return {"procedures": []}

    results_raw = await main_redis.execute_command(
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


@router.get("/relations")
async def relations_list():
    main_redis = await _r()
    from core import graph as graph_mod

    stats = await graph_mod.graph_stats(main_redis)
    return {"relations": stats}


# ── Viewer ────────────────────────────────────────────────────────────────────

@router.get("/viewer")
async def viewer():
    from fastapi.responses import FileResponse
    import os
    idx = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return {"message": "AgentMem Viewer", "version": "1.2.0"}
