"""
OpenClaw Local Memory Service  v0.2.0
======================================
Replaces memos-cloud-openclaw-plugin with fully local, persistent memory.

Features:
  1. Heat-tiered recall  — frequently/recently accessed memories rank higher
  2. Scene isolation     — language + domain tags filter recall by context
  3. Evolving persona    — structured user profile updated from extracted facts
  4. Summarize-then-embed— long turns compressed before MiniLM embedding

Endpoints:
  POST /recall   — before_agent_start hook
  POST /store    — agent_end hook (async, non-blocking)
  GET  /health
  GET  /stats
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

import numpy as np
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

import embedder
import extractor
import heat as heat_mod
import persona as persona_mod
import scene as scene_mod
import store as mem_store
import summarizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [mem] %(message)s")
log = logging.getLogger("mem")

_redis = None

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


# ── Embedding cache ────────────────────────────────────────────────────────────
@lru_cache(maxsize=1024)
def _cached_encode(text: str) -> bytes:
    return embedder.encode(text).tobytes()

def _encode(text: str) -> np.ndarray:
    return np.frombuffer(_cached_encode(text), dtype=np.float32).copy()


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis
    log.info("Loading embedding model…")
    embedder.get_model()
    log.info("Connecting to Redis…")
    _redis = await mem_store.get_client()
    log.info("Ensuring vectorset indexes…")
    await mem_store.ensure_indexes(_redis)
    log.info("Memory service v0.2.0 ready on :18790")
    yield
    await _redis.aclose()


app = FastAPI(title="OpenClaw Memory Service", version="0.2.0", lifespan=lifespan)


# ── Request / Response models ──────────────────────────────────────────────────
class RecallRequest(BaseModel):
    query: str
    session_id: str = ""
    memory_limit_number: int = 6

class Message(BaseModel):
    role: str
    content: Any

class StoreRequest(BaseModel):
    messages: list[Message]
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

def _format_prepend(
    facts: list[dict],
    episodes: list[dict],
    session_ctx: str | None,
    persona_ctx: str,
) -> str:
    lines: list[str] = []

    if persona_ctx:
        lines += [persona_ctx, ""]

    if session_ctx:
        lines += ["## Current Session Context", session_ctx, ""]

    if facts:
        lines.append("## Long-Term Memory (Facts)")
        for i, f in enumerate(facts, 1):
            tag = f.get("category") or f.get("domain") or "?"
            lines.append(f"{i}. [{tag}] {f['content']}")
        lines.append("")

    if episodes:
        lines.append("## Recent Relevant Episodes")
        for i, e in enumerate(episodes, 1):
            lines.append(f"{i}. {e['content'][:300]}")
        lines.append("")

    return "\n".join(lines).strip()


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

        # 5. Rule-based fact extraction → dedup → save → update persona
        raw_msgs = [m.model_dump() for m in clean if m.role == "user"]
        facts = extractor.extract(raw_msgs)
        fact_saved = 0
        for fact in facts:
            f_emb = _encode(fact.content)
            existing = await mem_store.knn_search(_redis, mem_store.FACT_KEY, f_emb, k=1)
            if existing and existing[0].get("score", 0.0) > 0.95:
                continue
            await mem_store.save_fact(
                _redis, fact.content, fact.category, fact.confidence, f_emb, lang, domain
            )
            await persona_mod.update(_redis, fact.category, fact.content)
            fact_saved += 1

        # 6. Update session working memory
        await mem_store.set_session_context(_redis, session_id, user_text_raw[:400])

        ms = int((time.time() - t0) * 1000)
        log.info(f"[store] session={session_id} lang={lang} domain={domain} "
                 f"ep+{ep_saved} facts+{fact_saved} {ms}ms")

    except Exception as e:
        log.warning(f"[store] background error: {e}", exc_info=True)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    try:
        await _redis.ping()
        return {"status": "ok", "redis": "ok", "version": "0.2.0"}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.get("/stats")
async def stats():
    try:
        counts = await mem_store.vcard(_redis)
        persona_raw = await _redis.hgetall(mem_store.PERSONA_KEY
                                           if hasattr(mem_store, "PERSONA_KEY")
                                           else "mem:persona")
        counts["persona_fields"] = len(persona_raw)
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

    n_facts    = max(1, req.memory_limit_number // 2)
    n_episodes = req.memory_limit_number

    # Build filter expressions for scene isolation
    lang_filter   = f'.language == "{q_lang}"'
    domain_filter = f'.domain == "{q_domain}"' if q_domain != "general" else None

    # Recall strategy: scene-filtered first, supplement with global if sparse
    async def _fetch(vset: str, k: int, fexpr=None):
        return await mem_store.knn_search(
            _redis, vset, emb, k, filter_expr=fexpr, bump_heat=True
        )

    # Parallel: persona + session ctx + scene-filtered facts + scene-filtered episodes
    persona_ctx, session_ctx, facts_scene, eps_scene = await asyncio.gather(
        persona_mod.get_context(_redis),
        mem_store.get_session_context(_redis, req.session_id),
        _fetch(mem_store.FACT_KEY,    n_facts,    lang_filter),
        _fetch(mem_store.EPISODE_KEY, n_episodes, lang_filter),
    )

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
    facts_global   = supp_results[0] if isinstance(supp_results[0], list) else []
    eps_global     = supp_results[1] if isinstance(supp_results[1], list) else []

    # Merge, dedup by content, heat-rerank
    def _merge(primary, supplement, limit):
        seen = {i["content"] for i in primary}
        merged = list(primary) + [i for i in supplement if i["content"] not in seen]
        ranked = heat_mod.heat_rerank(merged)
        return ranked[:limit]

    facts    = _merge(facts_scene,   facts_global,  n_facts)
    episodes = _merge(eps_scene,     eps_global,    n_episodes)

    prepend = _format_prepend(facts, episodes, session_ctx, persona_ctx)
    ms = int((time.time() - t0) * 1000)

    log.info(f"[recall] session={req.session_id} lang={q_lang} domain={q_domain} "
             f"facts={len(facts)} ep={len(episodes)} {ms}ms")

    return {"prependContext": prepend if prepend else None, "latency_ms": ms}


@app.post("/store")
async def store_memory(req: StoreRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(_do_store, req.messages, req.session_id)
    return {"status": "queued"}
