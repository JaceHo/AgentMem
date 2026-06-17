"""
Consolidation Service - Memory lifecycle management.

Three-phase consolidation pipeline (SimpleMem Section 3.2 + LLM Wiki v2):
  Phase 1 — Decay:  Ebbinghaus forgetting curve + legacy importance decay
  Phase 2 — Merge:  cluster near-duplicates, LLM-merge, soft-delete losers
  Phase 3 — Prune:  soft-delete low-importance / low-confidence entries

Also handles:
  - Hard pruning (physical VREM of long-soft-deleted entries)
  - LLM fact merging
  - Session crystallization
"""

import asyncio
import json
import logging
import math
import time

import numpy as np

from core import extractor
from core import store as mem_store
from core.http import async_post_json
from core.search import BM25Index, async_encode_batch_chunked, encode, vscan

log = logging.getLogger("mem")


async def do_consolidate(
    redis,
    bm25_index: BM25Index,
    spawn_fn,
    similarity_threshold: float = 0.85,
    temporal_lambda: float = 0.03,
) -> dict:
    """
    Three-phase consolidation pipeline.

    Phase 1 — Decay:  Ebbinghaus forgetting curve on effective confidence,
                      legacy 0.9× importance for entries >90 days old.
    Phase 2 — Merge:  cluster near-duplicates (cosine×temporal ≥ threshold),
                      LLM-merge into keeper, soft-delete losers via superseded_by.
    Phase 3 — Prune:  soft-delete entries with importance < 0.05 or
                      effective confidence < 0.1.

    Args:
        redis:             aioredis client
        bm25_index:        BM25Index instance (will be reset if changes made)
        spawn_fn:          fire-and-forget coroutine spawner
        similarity_threshold: cosine×temporal affinity threshold for merge
        temporal_lambda:   temporal decay rate (λ=0.03 → 23-day half-life)

    Returns:
        Dict with decayed, merged, pruned counts and timing.
    """
    _OVERALL_TIMEOUT_S = 120.0
    _MAX_LLM_MERGES = 10

    t0 = time.time()
    now_ms = int(time.time() * 1000)
    NINETY_DAYS_MS = 90 * 86_400_000

    try:
        scanned = await asyncio.wait_for(
            vscan(redis, mem_store.FACT_KEY, max_count=5000),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        log.warning("[consolidate] vscan timed out (60s), retrying with smaller batch")
        try:
            scanned = await asyncio.wait_for(
                vscan(redis, mem_store.FACT_KEY, max_count=2000),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            log.error("[consolidate] vscan timed out again, skipping consolidation")
            return {"merged": 0, "decayed": 0, "pruned": 0, "ms": int((time.time() - t0) * 1000)}

    if not scanned:
        return {"merged": 0, "decayed": 0, "pruned": 0, "ms": 0}

    all_facts = [
        {"element": item["element"], "attrs": item["attrs"]}
        for item in scanned
        if item["attrs"].get("content") and not item["attrs"].get("superseded_by")
    ]

    if len(all_facts) < 2:
        return {"merged": 0, "decayed": 0, "pruned": 0, "total": len(all_facts), "ms": 0}

    # ── Phase 1: Decay ────────────────────────────────────────────────────────
    # Batch decay: collect all changed attrs, then write via pipeline to
    # avoid N individual Redis round-trips on large corpora.
    decay_updates: list[tuple[str, str]] = []  # (element, json_attrs)
    for fact in all_facts:
        changed = False
        ts = fact["attrs"].get("ts", now_ms)
        if (now_ms - ts) > NINETY_DAYS_MS:
            old_imp = fact["attrs"].get("importance", 0.5)
            new_imp = round(old_imp * 0.9, 4)
            fact["attrs"]["importance"] = new_imp
            changed = True
        eff_conf = mem_store.confidence_decay(fact["attrs"])
        if eff_conf != fact["attrs"].get("effective_confidence", -1):
            fact["attrs"]["effective_confidence"] = eff_conf
            changed = True
        if changed:
            decay_updates.append((fact["element"], json.dumps(fact["attrs"])))

    decayed_count = 0
    if decay_updates:
        # Pipeline: batch all VSETATTR into one round-trip
        try:
            async with redis.pipeline(transaction=False) as pipe:
                for elem, attrs_json in decay_updates:
                    pipe.execute_command("VSETATTR", mem_store.FACT_KEY, elem, attrs_json)
                await pipe.execute()
            decayed_count = len(decay_updates)
        except Exception as e:
            log.warning(f"[consolidate] pipeline decay failed, falling back one-by-one: {e}")
            # Fallback: individual writes
            for elem, attrs_json in decay_updates:
                try:
                    await redis.execute_command("VSETATTR", mem_store.FACT_KEY, elem, attrs_json)
                    decayed_count += 1
                except Exception as ex:
                    log.debug(f"[consolidate] decay VSETATTR failed: {ex}")

    # ── Phase 2: Merge ────────────────────────────────────────────────────────
    merged_count = 0
    superseded_elements: set[str] = set()

    # Encode fact contents in chunks — a single batch over ~1k+ facts times out
    # on remote embedding APIs (ReadTimeout at 15s).
    fact_contents = [f["attrs"]["content"] for f in all_facts]
    fact_vecs = await async_encode_batch_chunked(fact_contents)
    if fact_vecs is None:
        log.warning("[consolidate] encode_batch failed, skipping merge phase")

    if fact_vecs is not None:
        fact_embeddings: dict[str, np.ndarray] = {
            f["element"]: vec for f, vec in zip(all_facts, fact_vecs)
        }

        # Pre-compute normalized embedding matrix for fast cosine similarity
        # (avoids O(n) knn_search calls — uses numpy vectorized ops instead)
        elem_order = [f["element"] for f in all_facts]
        emb_matrix = np.stack([fact_embeddings[e] for e in elem_order]).astype(np.float32)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # avoid division by zero
        emb_normed = emb_matrix / norms

        fact_by_element: dict[str, dict] = {f["element"]: f for f in all_facts}

        for idx, fact in enumerate(all_facts):
            if fact["element"] in superseded_elements:
                continue

            if merged_count >= _MAX_LLM_MERGES:
                break

            if time.time() - t0 > _OVERALL_TIMEOUT_S:
                log.warning("[consolidate] overall timeout reached, stopping merge phase")
                break

            # In-memory cosine similarity (vectorized) — no Redis VSIM needed
            sims = emb_normed @ emb_normed[idx]
            # Get top-k similar indices (excluding self)
            top_indices = np.argsort(sims)[::-1]
            cluster = [fact]
            fact_ts = fact["attrs"].get("ts", 0)

            for sim_idx in top_indices[:10]:  # check top 10 candidates
                if sim_idx == idx:
                    continue
                sim_elem = elem_order[sim_idx]
                if sim_elem in superseded_elements:
                    continue

                cosine_sim = float(sims[sim_idx])
                s_fact = fact_by_element.get(sim_elem)
                s_ts = s_fact["attrs"].get("ts", 0) if s_fact else 0

                days_between = abs(fact_ts - s_ts) / 86_400_000
                temporal_factor = math.exp(-temporal_lambda * days_between)
                affinity = cosine_sim * temporal_factor

                if affinity >= similarity_threshold and s_fact:
                    cluster.append(s_fact)

            if len(cluster) < 2:
                continue

            contents = [c["attrs"]["content"] for c in cluster]
            merged_content = await llm_merge_facts(contents, spawn_fn=spawn_fn)
            if not merged_content:
                continue

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

            m_emb = await asyncio.to_thread(encode, merged_content)
            try:
                await redis.execute_command(
                    "VADD", mem_store.FACT_KEY, "FP32", m_emb.tobytes(),
                    keeper_element, "SETATTR", json.dumps(new_attrs)
                )
            except Exception as e:
                log.warning(f"[consolidate] failed to update keeper: {e}")
                continue

            for c in cluster[1:]:
                if c["element"] not in superseded_elements:
                    await mem_store.soft_delete_fact(redis, c["element"], keeper_element, reason="merged")
                    superseded_elements.add(c["element"])

            merged_count += 1

    # ── Phase 3: Prune ────────────────────────────────────────────────────────
    pruned_count = 0
    for fact in all_facts:
        if fact["element"] in superseded_elements:
            continue
        imp = fact["attrs"].get("importance", 1.0)
        eff_conf = fact["attrs"].get("effective_confidence", 1.0)
        if imp < 0.05 or eff_conf < 0.1:
            reason = "pruned" if imp < 0.05 else "confidence_expired"
            await mem_store.soft_delete_fact(redis, fact["element"], "pruned", reason=reason)
            superseded_elements.add(fact["element"])
            pruned_count += 1

    # ── Post-consolidation: incremental BM25 update ────────────────────────────
    if superseded_elements:
        await bm25_index.remove(superseded_elements)

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


async def do_hard_prune(redis, bm25_index: BM25Index, spawn_fn) -> dict:
    """
    Physically VREM entries soft-deleted >7 days, and stale episodes >180 days.

    Returns counts of entries removed from each vectorset.
    """
    t0 = time.time()
    now_ms = int(time.time() * 1000)
    SEVEN_DAYS_MS = 7 * 86_400_000
    SIX_MONTHS_MS = 180 * 86_400_000

    removed_facts = 0
    removed_eps = 0
    pruned_fact_ids: set[str] = set()

    async def _hard_prune_vset(vset_key: str, max_scan: int = 2000) -> int:
        nonlocal pruned_fact_ids
        items = await vscan(redis, vset_key, max_count=max_scan)
        if not items:
            return 0

        to_remove: list[str] = []
        for item in items:
            attrs = item["attrs"]
            ts = attrs.get("ts", now_ms)
            age_ms = now_ms - ts

            superseded = attrs.get("superseded_by", "")
            if superseded and age_ms > SEVEN_DAYS_MS:
                to_remove.append(item["element"])
                continue

            if vset_key == mem_store.EPISODE_KEY:
                if age_ms > SIX_MONTHS_MS and attrs.get("access_count", 0) == 0:
                    to_remove.append(item["element"])

        removed = 0
        for elem_str in to_remove:
            try:
                await redis.execute_command("VREM", vset_key, elem_str)
                removed += 1
                if vset_key == mem_store.FACT_KEY:
                    pruned_fact_ids.add(elem_str)
            except Exception:
                pass
        return removed

    removed_facts = await _hard_prune_vset(mem_store.FACT_KEY)
    removed_eps = await _hard_prune_vset(mem_store.EPISODE_KEY)

    if removed_facts > 0 and pruned_fact_ids:
        await bm25_index.remove(pruned_fact_ids)

    ms = int((time.time() - t0) * 1000)
    log.info(f"[hard_prune] removed facts={removed_facts} episodes={removed_eps} {ms}ms")
    return {"removed_facts": removed_facts, "removed_episodes": removed_eps, "ms": ms}


async def llm_merge_facts(contents: list[str], spawn_fn=None) -> str | None:
    """Use role-based routing to merge multiple similar facts into one.

    Tries up to 2 models with circuit-breaker awareness.
    Falls back to longest content if all models fail.
    """
    tried = []
    for attempt in range(2):
        model, _ = await extractor._resolve_nlp_model(exclude=tried[-1] if tried else None)
        if model in tried:
            break
        tried.append(model)

        if extractor.is_model_circuit_broken(model):
            continue

        facts_text = "\n".join(f"- {c}" for c in contents)
        prompt = (
            f"以下是关于同一主题的多条记忆，请合并为一条完整、准确的事实陈述。"
            f"保留所有不同的细节，去除重复。输出仅包含合并后的一句话，不要解释。\n\n{facts_text}"
        )

        try:
            data = await async_post_json(
                extractor.AISERV_URL,
                payload={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                    "temperature": 0.1,
                },
                headers={"Authorization": f"Bearer {extractor.AISERV_KEY}"},
                timeout=8.0,
            )
            if data is not None:
                content = data["choices"][0]["message"].get("content")
                if content and content.strip():
                    if spawn_fn:
                        spawn_fn(extractor._report_quality(model, +1), "quality")
                    return content.strip()
        except Exception as e:
            log.warning(f"[consolidate] LLM merge {model} failed: {e}")
            extractor.mark_model_failed(model)
            if spawn_fn:
                spawn_fn(extractor._report_quality(model, -1, reason="other"), "quality")

    return max(contents, key=len)


async def crystallize_session(
    redis,
    session_id: str,
    max_facts: int = 20,
) -> dict:
    """
    Crystallize a session — distill it into a structured digest.

    LLM Wiki v2: distills completed work into structured digest:
    question, findings, entities, lessons.
    """
    scanned = await vscan(redis, mem_store.FACT_KEY, max_count=5000)

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

    facts.sort(key=lambda a: a.get("importance", 0.5), reverse=True)
    top_facts = facts[:max_facts]

    digest = {
        "session_id": session_id,
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

    return digest


async def crystallize_session_inline(
    redis,
    session_id: str,
    session_obj: dict,
    max_facts: int = 20,
) -> dict | None:
    """
    Inline crystallization (same as /crystallize but without HTTP overhead).

    Returns digest dict or None if crystallization fails.
    """
    try:
        scanned = await vscan(redis, mem_store.FACT_KEY, max_count=5000)

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

        facts.sort(key=lambda a: a.get("importance", 0.5), reverse=True)
        top_facts = facts[:max_facts]

        fact_texts = [f.get("content", "") for f in top_facts[:10]]
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
            "entities": sorted(all_entities)[:20],
            "categories": sorted(set(f.get("category", "") for f in top_facts)),
            "crystallized_at": int(time.time() * 1000),
            "auto_crystallized": True,
        }

        return digest

    except Exception as e:
        log.warning(f"[crystallize] inline crystallization failed for {session_id}: {e}")
        return None
