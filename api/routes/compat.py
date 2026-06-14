"""Compat routes — agentmemory scaffold endpoints for hook & MCP compatibility."""

import json
import logging
import os
import time

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Request

from api import state
from api.compat import compat_sid as _compat_sid, check_auth as _compat_check_auth
from api.schemas.compat import (
    CompatSessionStartRequest, CompatSessionEndRequest,
    ObserveRequest, SummarizeRequest, EnrichRequest,
    ContextRequest, SessionCommitRequest,
    SearchRequest, RememberRequest, ForgetRequest,
    FileContextRequest, PatternsRequest, SmartSearchRequest,
    TimelineRequest, ClaudeBridgeSyncRequest, ImportRequest,
)
from core import embedder
from core import persona as persona_mod
from core import store as mem_store
from core.search import encode
from core.utils import decode_bytes, decode_attrs
from services.store_service import (
    format_prepend, do_compress_session, observe_internal, rrf_merge,
)

log = logging.getLogger("mem")
router = APIRouter(tags=["compat"])


@router.post("/session/start")
async def compat_session_start(req: CompatSessionStartRequest, request: Request):
    _compat_check_auth(request)

    r = state.redis
    sid = _compat_sid(req)
    project = req.project or req.cwd or os.getcwd()

    session_key = f"mem:session:{sid}"
    await r.set(session_key, json.dumps({
        "session_id": sid,
        "project": project,
        "started_at": time.time(),
        "observations": 0,
    }), ex=14400)

    persona_ctx = await persona_mod.get_context(r)

    pinned_raw = await r.get("mem:pinned:session_summary")
    last_summary = decode_bytes(pinned_raw)

    context_parts = []
    if persona_ctx:
        context_parts.append(persona_ctx)
    if last_summary:
        context_parts.append(f"## Last Session Summary\n{last_summary[:600]}")

    context = "\n\n".join(context_parts) if context_parts else None
    return {"status": "ok", "sessionId": sid, "context": context}


@router.post("/session/end")
async def compat_session_end(req: CompatSessionEndRequest, request: Request,
                              background_tasks: BackgroundTasks):
    _compat_check_auth(request)

    sid = _compat_sid(req)

    async def _do_end():
        try:
            await do_compress_session(sid)
        except Exception as e:
            log.warning("[session/end] compress failed: %s", e)

    background_tasks.add_task(_do_end)
    return {"status": "ok", "sessionId": sid}


@router.post("/observe")
async def observe(req: ObserveRequest, request: Request, background_tasks: BackgroundTasks):
    _compat_check_auth(request)

    sid = _compat_sid(req)
    project = req.project or req.cwd or ""
    hook_type = req.hookType or "observe"
    data = req.data or {}

    async def _do_observe():
        try:
            await observe_internal(sid, project, req.cwd or "", hook_type, data)
        except Exception as e:
            log.warning("[observe] background error: %s", e)

    background_tasks.add_task(_do_observe)
    return {"status": "ok"}


@router.post("/summarize")
async def compat_summarize(req: SummarizeRequest, request: Request,
                            background_tasks: BackgroundTasks):
    _compat_check_auth(request)

    sid = _compat_sid(req)

    async def _do_summarize():
        try:
            await do_compress_session(sid)
        except Exception as e:
            log.warning("[summarize] compress failed: %s", e)

    background_tasks.add_task(_do_summarize)
    return {"status": "ok", "sessionId": sid}


@router.post("/enrich")
async def enrich(req: EnrichRequest, request: Request):
    _compat_check_auth(request)

    r = state.redis
    files = req.files or []
    query = req.query or ""

    file_contexts = []
    for f in files[:5]:
        fname = os.path.basename(f)
        emb = encode(fname)
        results = await mem_store.knn_search(r, mem_store.FACT_KEY, emb, k=3)
        for res in results:
            attrs = res.get("attrs", {})
            if attrs.get("content"):
                file_contexts.append(attrs["content"][:200])

    if query:
        emb = encode(query)
        results = await mem_store.knn_search(r, mem_store.FACT_KEY, emb, k=5)
        for res in results:
            attrs = res.get("attrs", {})
            if attrs.get("content"):
                file_contexts.append(attrs["content"][:200])

    context = "\n".join(file_contexts[:10]) if file_contexts else None
    return {"status": "ok", "context": context}


@router.post("/context")
async def compat_context(req: ContextRequest, request: Request):
    _compat_check_auth(request)

    r = state.redis
    sid = _compat_sid(req)
    query = req.query or ""
    budget = req.token_budget or 1500

    persona_ctx = await persona_mod.get_context(r)
    session_ctx = await mem_store.get_session_context(r, sid)

    pinned_raw = await r.get("mem:pinned:session_summary")
    last_summary = decode_bytes(pinned_raw)

    facts = []
    episodes = []
    if query:
        emb = encode(query)
        facts = await mem_store.knn_search(r, mem_store.FACT_KEY, emb, k=8)
        episodes = await mem_store.knn_search(r, mem_store.EPISODE_KEY, emb, k=6)

    prepend = format_prepend(
        facts=facts, episodes=episodes,
        session_ctx=session_ctx, persona_ctx=persona_ctx or "",
        token_budget=budget, last_session_summary=last_summary,
    )
    return {"status": "ok", "context": prepend}


@router.post("/session/commit")
async def compat_session_commit(req: SessionCommitRequest, request: Request):
    _compat_check_auth(request)

    r = state.redis
    sid = _compat_sid(req)
    commit_key = f"mem:commit:{req.sha}"
    await r.hset(commit_key, mapping={
        "sha": req.sha,
        "message": req.message,
        "branch": req.branch,
        "repo": req.repo,
        "session_id": sid,
        "timestamp": str(int(time.time() * 1000)),
    })
    return {"status": "ok", "sha": req.sha}


@router.get("/session/by-commit")
async def session_by_commit(sha: str = ""):
    if not sha:
        return {"error": "sha parameter required"}
    r = state.redis
    commit_key = f"mem:commit:{sha}"
    data = await r.hgetall(commit_key)
    if not data:
        return {"error": "commit not found", "sha": sha}
    result = {decode_bytes(k): decode_bytes(v)
              for k, v in data.items()}
    return result


@router.get("/commits")
async def commits(branch: str = "", repo: str = "", limit: int = 20):
    r = state.redis
    _, keys = await r.scan(match="mem:commit:*", count=limit)
    if not keys:
        return {"commits": []}

    pipe = r.pipeline(transaction=False)
    for key in keys:
        k = decode_bytes(key)
        pipe.hgetall(k)
    pipe_results = await pipe.execute()

    results = []
    for data in pipe_results:
        if not data:
            continue
        entry = {decode_bytes(dk): decode_bytes(dv)
                 for dk, dv in data.items()}
        if branch and entry.get("branch") != branch:
            continue
        if repo and entry.get("repo") != repo:
            continue
        results.append(entry)
    results.sort(key=lambda x: x.get("timestamp", "0"), reverse=True)
    return {"commits": results[:limit]}


@router.post("/claude-bridge/sync")
async def claude_bridge_sync(req: ClaudeBridgeSyncRequest, request: Request):
    _compat_check_auth(request)

    r = state.redis
    sid = _compat_sid(req)
    project = req.project or req.cwd or ""

    persona_ctx = await persona_mod.get_context(r)
    session_ctx = await mem_store.get_session_context(r, sid)

    pinned_raw = await r.get("mem:pinned:session_summary")
    last_summary = decode_bytes(pinned_raw)

    emb = encode(project or "project context")
    facts = await mem_store.knn_search(r, mem_store.FACT_KEY, emb, k=10)
    episodes = await mem_store.knn_search(r, mem_store.EPISODE_KEY, emb, k=6)

    prepend = format_prepend(
        facts=facts, episodes=episodes,
        session_ctx=session_ctx, persona_ctx=persona_ctx or "",
        token_budget=2000, last_session_summary=last_summary,
    )

    memory_md_path = os.path.expanduser("~/.claude/AGENTMEM_MEMORY.md")
    if prepend:
        try:
            os.makedirs(os.path.dirname(memory_md_path), exist_ok=True)
            with open(memory_md_path, "w") as f:
                f.write(prepend)
        except Exception as e:
            log.warning("[claude-bridge] write failed: %s", e)

    return {"status": "ok", "memoryPath": memory_md_path if prepend else None}


@router.post("/consolidate-pipeline")
async def compat_consolidate_pipeline(request: Request, background_tasks: BackgroundTasks):
    _compat_check_auth(request)

    async def _do_consolidate_compat():
        try:
            from services.consolidation_service import do_consolidate
            await do_consolidate(state.redis, state.bm25_index, state.spawn)
        except Exception as e:
            log.warning("[consolidate-pipeline] error: %s", e)

    background_tasks.add_task(_do_consolidate_compat)
    return {"status": "ok"}


@router.post("/crystals/auto")
async def crystals_auto(request: Request):
    _compat_check_auth(request)
    return {"status": "ok", "crystals": []}


@router.post("/search")
async def compat_search(req: SearchRequest, request: Request):
    _compat_check_auth(request)

    r = state.redis
    sid = _compat_sid(req)
    emb = encode(req.query)

    facts = await mem_store.knn_search(r, mem_store.FACT_KEY, emb, k=req.limit)
    episodes = await mem_store.knn_search(r, mem_store.EPISODE_KEY, emb, k=req.limit)

    persona_ctx = await persona_mod.get_context(r)
    session_ctx = await mem_store.get_session_context(r, sid)

    pinned_raw = await r.get("mem:pinned:session_summary")
    last_summary = decode_bytes(pinned_raw)

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

    prepend = format_prepend(
        facts=facts, episodes=episodes,
        session_ctx=session_ctx, persona_ctx=persona_ctx or "",
        token_budget=budget, last_session_summary=last_summary,
    )
    return {"context": prepend, "facts": len(facts), "episodes": len(episodes)}


@router.post("/remember")
async def compat_remember(req: RememberRequest, request: Request):
    _compat_check_auth(request)

    r = state.redis
    content = req.content
    emb = encode(content)

    uid = await mem_store.save_fact(
        r,
        content=content,
        category=req.type or "fact",
        confidence=0.8,
        embedding=emb,
        keywords=[c.strip() for c in req.concepts.split(",") if c.strip()] if req.concepts else None,
        importance=0.8,
    )
    return {"status": "ok", "id": uid}


@router.post("/forget")
async def compat_forget(req: ForgetRequest, request: Request):
    _compat_check_auth(request)

    r = state.redis
    emb = encode(req.query)
    facts = await mem_store.knn_search(r, mem_store.FACT_KEY, emb, k=req.limit)

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
            await r.execute_command("VREM", mem_store.FACT_KEY, elem)
            await state.bm25_index.remove({elem})
            deleted += 1

    return {"status": "ok", "deleted": deleted}


@router.post("/compress-file")
async def compress_file(filePath: str = ""):
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


@router.get("/sessions")
async def compat_sessions(limit: int = 20):
    r = state.redis
    _, ctx_keys = await r.scan(match="mem:session:*:ctx", count=500)
    if not ctx_keys:
        return {"sessions": []}

    pipe = r.pipeline(transaction=False)
    for key in ctx_keys:
        pipe.get(key)
    raw_values = await pipe.execute()

    results = []
    for key, raw in zip(ctx_keys, raw_values):
        k = decode_bytes(key)
        parts = k.split(":")
        sid = parts[2] if len(parts) >= 4 else k
        ctx_str = decode_bytes(raw) or ""
        results.append({
            "session_id": sid,
            "status": "ended",
            "observation_count": 0,
            "summary": ctx_str[:300] if ctx_str else None,
        })
    return {"sessions": results[:limit]}


@router.get("/observations")
async def compat_observations(sessionId: str = "", limit: int = 20):
    r = state.redis
    seed = np.zeros(embedder._get_provider().dims, dtype=np.float32)
    card = await r.execute_command("VCARD", mem_store.EPISODE_KEY)
    if not card or int(card) <= 1:
        return {"observations": []}

    results_raw = await r.execute_command(
        "VSIM", mem_store.EPISODE_KEY, "FP32", seed.astype("float32").tobytes(),
        "COUNT", min(int(card), limit + 1), "WITHSCORES", "WITHATTRIBS"
    )

    items = []
    i = 0
    while i + 2 < len(results_raw):
        elem = results_raw[i]; raw = results_raw[i + 2]; i += 3
        elem_str = decode_bytes(elem)
        if elem_str == "__seed__":
            continue
        try:
            attrs = decode_attrs(raw)
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


@router.post("/file-context")
async def file_context(req: FileContextRequest, request: Request):
    _compat_check_auth(request)

    r = state.redis
    files = [f.strip() for f in req.files.split(",") if f.strip()]
    sid = _compat_sid(req)

    all_results = []
    for f in files[:10]:
        fname = os.path.basename(f)
        emb = encode(fname)
        results = await mem_store.knn_search(r, mem_store.EPISODE_KEY, emb, k=5,
                                              filter_expr=f'.session_id != "{sid}"')
        for res in results:
            attrs = res.get("attrs", {})
            all_results.append({
                "file": f,
                "content": attrs.get("content", "")[:300],
                "session_id": attrs.get("session_id", ""),
                "score": round(res.get("score", 0), 3),
            })
    return {"results": all_results}


@router.post("/patterns")
async def compat_patterns(req: PatternsRequest, request: Request):
    _compat_check_auth(request)

    r = state.redis
    query = req.query or "recurring patterns"
    emb = encode(query)
    facts = await mem_store.knn_search(r, mem_store.FACT_KEY, emb, k=req.limit)

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


@router.post("/smart-search")
async def smart_search(req: SmartSearchRequest, request: Request):
    _compat_check_auth(request)

    r = state.redis
    emb = encode(req.query)
    facts = await mem_store.knn_search(r, mem_store.FACT_KEY, emb, k=req.limit)
    episodes = await mem_store.knn_search(r, mem_store.EPISODE_KEY, emb, k=req.limit)

    merged = rrf_merge([facts, episodes], weights=[1.0, 0.8], limit=req.limit)

    return {
        "results": [{
            "content": m.get("content", "")[:300],
            "category": m.get("attrs", {}).get("category", ""),
            "score": round(m.get("score", 0), 4),
        } for m in merged],
    }


@router.post("/timeline")
async def compat_timeline(req: TimelineRequest, request: Request):
    _compat_check_auth(request)

    r = state.redis
    seed = np.zeros(embedder._get_provider().dims, dtype=np.float32)
    card = await r.execute_command("VCARD", mem_store.EPISODE_KEY)
    if not card or int(card) <= 1:
        return {"timeline": []}

    results_raw = await r.execute_command(
        "VSIM", mem_store.EPISODE_KEY, "FP32", seed.astype("float32").tobytes(),
        "COUNT", min(int(card), req.limit + 1), "WITHSCORES", "WITHATTRIBS"
    )

    items = []
    i = 0
    while i + 2 < len(results_raw):
        elem = results_raw[i]; raw = results_raw[i + 2]; i += 3
        elem_str = decode_bytes(elem)
        if elem_str == "__seed__":
            continue
        try:
            attrs = decode_attrs(raw)
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


@router.get("/profile")
async def compat_profile():
    r = state.redis
    if r is None:
        return {"profile": {}}
    persona_raw = await r.hgetall("mem:persona")
    fields = {
        decode_bytes(k): decode_bytes(v)
        for k, v in persona_raw.items()
    }
    return {"profile": fields}


@router.get("/export")
async def compat_export():
    r = state.redis
    export = {"version": "1.2.0", "exported_at": time.time(), "facts": [], "episodes": []}

    for key, label in [(mem_store.FACT_KEY, "facts"), (mem_store.EPISODE_KEY, "episodes")]:
        card = await r.execute_command("VCARD", key)
        if not card or int(card) <= 1:
            continue
        seed = np.zeros(embedder._get_provider().dims, dtype=np.float32)
        results = await r.execute_command(
            "VSIM", key, "FP32", seed.astype("float32").tobytes(),
            "COUNT", min(int(card), 5000), "WITHSCORES", "WITHATTRIBS"
        )
        i = 0
        while i + 2 < len(results):
            elem = results[i]; raw = results[i + 2]; i += 3
            elem_str = decode_bytes(elem)
            if elem_str == "__seed__":
                continue
            try:
                attrs = decode_attrs(raw)
            except Exception:
                continue
            export[label].append({"id": elem_str, **attrs})

    return export


@router.post("/import")
async def compat_import(req: ImportRequest, request: Request):
    _compat_check_auth(request)

    r = state.redis
    data = req.data
    imported = 0

    for item in data.get("episodes", []):
        content = item.get("content", "")
        if not content:
            continue
        emb = encode(content)
        await mem_store.save_episode(
            r,
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
        emb = encode(content)
        await mem_store.save_fact(
            r,
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


async def _list_memories(limit: int = 50, category: str = "") -> dict:
    """Shared logic for listing memories."""
    r = state.redis
    seed = np.zeros(embedder._get_provider().dims, dtype=np.float32)
    card = await r.execute_command("VCARD", mem_store.FACT_KEY)
    if not card or int(card) <= 1:
        return {"memories": []}

    results_raw = await r.execute_command(
        "VSIM", mem_store.FACT_KEY, "FP32", seed.astype("float32").tobytes(),
        "COUNT", min(int(card), limit + 1), "WITHSCORES", "WITHATTRIBS"
    )

    items = []
    i = 0
    while i + 2 < len(results_raw):
        elem = results_raw[i]; raw = results_raw[i + 2]; i += 3
        elem_str = decode_bytes(elem)
        if elem_str == "__seed__":
            continue
        try:
            attrs = decode_attrs(raw)
        except Exception:
            continue
        if category and attrs.get("category") != category:
            continue
        items.append({"id": elem_str, **attrs})

    return {"memories": items[:limit]}


@router.get("/memories")
async def compat_memories(limit: int = 50, category: str = ""):
    return await _list_memories(limit=limit, category=category)


@router.get("/memories/{memory_id}")
async def memory_detail(memory_id: str):
    r = state.redis
    try:
        raw = await r.execute_command("VGETATTR", mem_store.FACT_KEY, memory_id)
        if raw:
            attrs = decode_attrs(raw)
            return {"id": memory_id, **attrs}
    except Exception:
        pass

    try:
        raw = await r.execute_command("VGETATTR", mem_store.EPISODE_KEY, memory_id)
        if raw:
            attrs = decode_attrs(raw)
            return {"id": memory_id, **attrs}
    except Exception:
        pass

    return {"error": "memory not found", "id": memory_id}


@router.get("/semantic")
async def compat_semantic(limit: int = 50):
    return await _list_memories(limit=limit)


@router.get("/procedural")
async def compat_procedural(limit: int = 50):
    r = state.redis
    seed = np.zeros(embedder._get_provider().dims, dtype=np.float32)
    card = await r.execute_command("VCARD", mem_store.PROC_KEY)
    if not card or int(card) <= 1:
        return {"procedures": []}

    results_raw = await r.execute_command(
        "VSIM", mem_store.PROC_KEY, "FP32", seed.astype("float32").tobytes(),
        "COUNT", min(int(card), limit + 1), "WITHSCORES", "WITHATTRIBS"
    )

    items = []
    i = 0
    while i + 2 < len(results_raw):
        elem = results_raw[i]; raw = results_raw[i + 2]; i += 3
        elem_str = decode_bytes(elem)
        if elem_str == "__seed__":
            continue
        try:
            attrs = decode_attrs(raw)
        except Exception:
            continue
        items.append({"id": elem_str, **attrs})

    return {"procedures": items[:limit]}


@router.get("/relations")
async def compat_relations():
    from core import graph as graph_mod
    stats = await graph_mod.graph_stats(state.redis)
    return {"relations": stats}
