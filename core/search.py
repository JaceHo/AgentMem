"""
Core search utilities — BM25 hybrid index, vector scan helper, and embedding cache.

Extracted from main.py to provide reusable search primitives across route modules.
"""

import asyncio
import atexit
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np

from core import embedder
from core.utils import decode_bytes, decode_attrs

log = logging.getLogger("mem")

# ── Dedicated embedding pool (Task #20) ───────────────────────────────────────
#
# Recall used to throw every embedding onto the *default* asyncio thread pool
# (asyncio.to_thread).  That pool is also used by BM25 rebuilds and any other
# to_thread() call, and it has no real concurrency ceiling — so a burst of
# concurrent recalls would spawn unbounded embedding threads that piled onto
# the CPU (or the embedding service) until latency exploded and requests timed
# out.  The practical ceiling was implicit and collapsed under load.
#
# We instead give embeddings their own bounded pool + semaphore.  The event
# loop stays free to accept and serve many concurrent recall requests (the
# I/O-bound Redis / augmentation work runs concurrently); only the CPU-bound
# embedding is serialized behind a fixed, tunable number of slots.  This is
# how a single-process server (in-process BM25 index + TaskManager state —
# which multi-worker would shard) raises its recall concurrency ceiling:
# turn the implicit saturation limit into an explicit, configurable knob.
#
# Tune via AGENTMEM_RECALL_CONCURRENCY / settings.recall_concurrency.  For a
# local CPU-bound embedder, keep it near the core count; for a remote (HTTP)
# embedder, raise it to absorb network latency.  Raising it directly raises
# the recall concurrency ceiling.

try:
    from config.settings import settings as _app_settings
    _RECALL_CONCURRENCY = max(1, int(_app_settings.recall_concurrency))
except Exception:
    # Settings unavailable (e.g. minimal import in a test harness) — fall back
    # to a cpu-aware default rather than crashing the import.
    _RECALL_CONCURRENCY = max(2, min(8, (os.cpu_count() or 4) + 4))

_embed_pool = ThreadPoolExecutor(
    max_workers=_RECALL_CONCURRENCY, thread_name_prefix="embed"
)
_embed_sem: asyncio.Semaphore | None = None


def _get_embed_sem() -> asyncio.Semaphore:
    """Lazily build the embedding semaphore inside a running loop.

    Creating asyncio.Semaphore at import time is unsafe across loop lifecycles,
    so we construct it on first use instead.
    """
    global _embed_sem
    if _embed_sem is None:
        _embed_sem = asyncio.Semaphore(_RECALL_CONCURRENCY)
    return _embed_sem


@atexit.register
def _shutdown_embed_pool() -> None:
    _embed_pool.shutdown(wait=False)


async def async_encode(text: str) -> np.ndarray:
    """Single-text encode off the event loop via the dedicated embed pool.

    Semaphore-bounded so concurrent recalls can't spawn unbounded embedding
    threads.  Propagates EmbeddingUnavailable (circuit-open) unchanged.
    """
    sem = _get_embed_sem()
    loop = asyncio.get_running_loop()
    async with sem:
        return await loop.run_in_executor(_embed_pool, embedder.encode, text)


async def async_encode_batch(texts: list[str]) -> list[np.ndarray]:
    """Batch encode via the dedicated embed pool, semaphore-bounded.

    Mirrors the old _encode_batch_async contract: propagates EmbeddingUnavailable
    but otherwise degrades gracefully — per-text failures fall back to a zero
    vector rather than killing the whole batch.
    """
    if not texts:
        return []
    sem = _get_embed_sem()
    loop = asyncio.get_running_loop()
    async with sem:
        try:
            return list(await loop.run_in_executor(_embed_pool, embedder.encode_batch, texts))
        except embedder.EmbeddingUnavailable:
            raise
        except Exception as e:
            log.warning(
                "[encode_batch] pool batch encode failed (%s: %s), "
                "falling back to per-text", type(e).__name__, e,
            )
            results: list[np.ndarray] = []
            for t in texts:
                try:
                    results.append(
                        await loop.run_in_executor(_embed_pool, embedder.encode, t)
                    )
                except embedder.EmbeddingUnavailable:
                    raise
                except Exception:
                    results.append(np.zeros(embedder.DIMS, dtype=np.float32))
            return results

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


# ── Tier text extraction ──────────────────────────────────────────────────────
#
# Each memory tier stores its text under different attribute names:
#
#   mem:facts       content, triple_str
#   mem:episodes    content
#   mem:procedures  task, procedure        (older rows have no `content` at all)
#
# The BM25 corpus used to be facts-only, which left ~79% of the text-bearing
# corpus unreachable by keyword search.  Every tier now contributes, and the
# text is assembled per tier so nothing is dropped on the floor.

TIER_TEXT_KEYS: dict[str, tuple[str, ...]] = {
    "mem:facts":     ("content", "triple_str"),
    "mem:episodes":  ("content", "summary"),
    "mem:procedures": ("task", "procedure", "content"),
}


def tier_of(vset: str) -> str:
    """Canonical tier name for a vectorset key (mem:facts -> facts)."""
    return vset.split(":", 1)[-1]


def extract_text(vset: str, attrs: dict) -> str:
    """Assemble the indexable text for an element of *vset* from its attrs.

    Returns "" when the element carries no text, so callers can skip it.
    """
    parts: list[str] = []
    for key in TIER_TEXT_KEYS.get(vset, ("content",)):
        val = attrs.get(key)
        if val and isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return " ".join(parts)


# ── Embedding cache ────────────────────────────────────────────────────────────

@lru_cache(maxsize=4096)
def cached_encode(text: str) -> bytes:
    """Cache raw embedding bytes (immutable, safe to share)."""
    return embedder.encode(text).tobytes()


def encode(text: str) -> np.ndarray:
    """Return a mutable float32 embedding vector for *text*."""
    return np.frombuffer(cached_encode(text), dtype=np.float32).copy()


def encode_batch(texts: list[str]) -> list[np.ndarray]:
    """Return mutable float32 embedding vectors for a batch of texts.

    Uses the embedder's native batch encoding which is significantly faster
    than calling encode() one-by-one (single forward pass vs N passes).
    """
    vecs = embedder.encode_batch(texts)
    return [np.asarray(v, dtype=np.float32) for v in vecs]


async def async_encode_batch_chunked(
    texts: list[str],
    *,
    batch_size: int = 64,
    timeout_per_batch: float = 60.0,
    max_retries: int = 2,
) -> list[np.ndarray] | None:
    """Encode texts in chunks off the event loop.

    Returns None if any chunk fails after retries — callers should skip
    operations that depend on the full embedding matrix (e.g. merge phase).
    Retries with exponential backoff to handle transient embedding service
    failures (502, 403, timeouts) common in production.
    """
    if not texts:
        return []
    all_vecs: list[np.ndarray] = []
    loop = asyncio.get_running_loop()
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset:offset + batch_size]
        for attempt in range(max_retries + 1):
            try:
                vecs = await asyncio.wait_for(
                    loop.run_in_executor(_embed_pool, encode_batch, batch),
                    timeout=timeout_per_batch,
                )
                all_vecs.extend(vecs)
                break
            except (asyncio.TimeoutError, Exception) as e:
                if attempt < max_retries:
                    backoff = 0.5 * (2 ** attempt)
                    log.info(
                        "[encode] chunk at offset %d attempt %d/%d failed, retrying in %.1fs (%s: %s)",
                        offset, attempt + 1, max_retries + 1, backoff, type(e).__name__, e,
                    )
                    await asyncio.sleep(backoff)
                    continue
                log.warning(
                    "[encode] chunk at offset %d failed after %d attempts (%s: %s)",
                    offset, max_retries + 1, type(e).__name__, e,
                )
                return None
    return all_vecs


# ── Zero-vector for VSIM scan-all ─────────────────────────────────────────────
# Lazily computed to avoid stale dimensions if provider changes at runtime
# (dimension guard in lifespan can reset provider, changing DIMS).

_zero_vec_cache: tuple[int, np.ndarray] | None = None


def _zero_vec() -> np.ndarray:
    """Return a zero vector matching the current embedding dimensions.

    Lazily computed and cached — if the provider changes (dimension guard),
    the next call picks up the new DIMS value automatically.
    """
    global _zero_vec_cache
    # Ensure provider is initialized so DIMS reflects the actual provider
    embedder._get_provider()
    dims = embedder.DIMS
    if _zero_vec_cache is None or _zero_vec_cache[0] != dims:
        _zero_vec_cache = (dims, np.zeros(dims, dtype=np.float32))
    return _zero_vec_cache[1]


async def vscan(r, vset_key: str, max_count: int = 200) -> list[dict]:
    """Best-effort full scan of a vectorset using zero-vector VSIM.

    **Not a reliable enumeration.**  VSIM is an approximate HNSW traversal and
    does not guarantee ``COUNT`` results.  On this instance it returns 99.9% of
    ``mem:facts`` but only ~13% of ``mem:procedures``, whose graph has degraded.
    Prefer ``core.store.enumerate_elements`` (shadow index) whenever
    completeness matters — consolidation, procedure scans, BM25 population.

    Kept for the paths that genuinely only need a sample, and as the fallback
    when no shadow index exists yet.  Logs the shortfall so silent truncation
    becomes visible.
    """
    card = await r.execute_command("VCARD", vset_key)
    if not card or int(card) <= 0:
        return []
    try:
        results = await r.execute_command(
            "VSIM", vset_key, "FP32", _zero_vec().tobytes(),
            "COUNT", min(int(card), max_count), "WITHSCORES", "WITHATTRIBS"
        )
    except Exception as e:
        log.warning("vscan VSIM failed on %s (%d elements): %s", vset_key, card, e)
        return []
    items = []
    i = 0
    while i + 2 < len(results):
        elem = results[i]
        score = results[i + 1]
        raw = results[i + 2]
        i += 3
        elem_str = decode_bytes(elem)
        if elem_str == "__seed__":
            continue
        try:
            attrs = decode_attrs(raw)
        except Exception:
            continue
        if attrs.get("_seed"):
            continue
        items.append({"element": elem_str, "score": float(score) if score else 0.0, "attrs": attrs})
    if items and len(items) < int(card) - 1:
        log.warning(
            "vscan on %s reached only %d of %d elements — the HNSW graph is "
            "degraded; use the shadow index for complete enumeration",
            vset_key, len(items), int(card) - 1,
        )
    return items


# ── BM25 in-memory index ──────────────────────────────────────────────────────

class BM25Index:
    """Copy-on-write BM25 corpus over fact contents + triple strings.

    Reads never block: search() reads the current _bm25 reference atomically (GIL).
    Rebuilds run in a thread pool (asyncio.to_thread) so the event loop isn't blocked
    during corpus construction.  Only one rebuild runs at a time (_rebuilding flag).
    Superseded facts (superseded_by != "") are excluded from search results.
    Rebuilt when corpus size changes by >=10 entries or when forced.
    """

    def __init__(self):
        # (uid, content, attrs, tier) — tier lets recall route a keyword hit
        # back to the vectorset it came from.
        self._docs: list[tuple[str, str, dict, str]] = []
        self._bm25 = None
        self._built_at_len: int = 0
        self._rebuild_threshold: int = 10
        self._add_lock = asyncio.Lock()      # protects list append only
        self._rebuild_lock = asyncio.Lock()  # prevents duplicate concurrent rebuilds
        # Versioned snapshot: _snap = (bm25_index, docs_list) always in sync.
        # search() reads _snap once, so scores and docs can never diverge.
        self._snap: tuple | None = None  # (bm25, docs_list) or None

    async def add(self, uid: str, content: str, attrs: dict, tier: str = "facts") -> None:
        """Add a document to the corpus. Triggers a background rebuild if threshold crossed."""
        if not content or not content.strip():
            return
        async with self._add_lock:
            self._docs.append((uid, content, attrs, tier))
        # Kick off a non-blocking rebuild if enough new docs have accumulated
        n = len(self._docs)
        if (self._bm25 is None or (n - self._built_at_len) >= self._rebuild_threshold) \
                and not self._rebuild_lock.locked():
            asyncio.create_task(self._rebuild_async())

    async def remove(self, uids: set[str]) -> None:
        """Remove documents by uid set. Incremental — avoids full reset+repopulate."""
        if not uids:
            return
        async with self._add_lock:
            before = len(self._docs)
            self._docs = [(u, c, a, t) for u, c, a, t in self._docs if u not in uids]
            removed = before - len(self._docs)
        if removed > 0 and not self._rebuild_lock.locked():
            asyncio.create_task(self._rebuild_async())

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize for BM25: English words >=3 chars + CJK bigrams."""
        text = text.lower()
        # Identifiers keep their underscores: `[a-z]{3,}` split ESCALATE_AT into
        # nothing searchable, so config keys and env names never matched a query
        # that named them exactly (measured 2026-10-27 on the KB corpus).
        en = re.findall(r'\b[a-z][a-z0-9_]{2,}\b', text)
        # CJK runs must be split into overlapping bigrams. Matching a whole run
        # (`[\u4e00-\u9fff]{2,}`) made one token out of "管理控制台在哪个地址", so a
        # query phrased differently shared no token with the passage at all
        # (measured 2026-10-27: every Chinese eval question missed in BM25).
        zh = [run[i:i + 2]
              for run in re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]+', text)
              for i in range(len(run) - 1)]
        return en + zh

    def _build_corpus(self, docs: list[tuple[str, str, dict, str]]):
        """CPU-bound corpus construction — runs in thread pool via asyncio.to_thread."""
        if not BM25_AVAILABLE:
            return None, 0
        corpus = []
        for _, content, attrs, _tier in docs:
            if attrs.get("superseded_by"):
                corpus.append([])
                continue
            text = content
            triple_str = attrs.get("triple_str", "")
            if triple_str:
                text = f"{text} {triple_str}"
            corpus.append(self._tokenize(text))
        return _BM25Okapi(corpus), len(docs)

    async def _rebuild_async(self) -> None:
        """Rebuild BM25 index in a thread pool without blocking reads or the event loop."""
        if self._rebuild_lock.locked():
            return
        async with self._rebuild_lock:
            docs_snapshot = list(self._docs)  # snapshot avoids holding add_lock during build
            if not docs_snapshot:
                return
            new_bm25, n = await asyncio.to_thread(self._build_corpus, docs_snapshot)
            if new_bm25 is None:
                return
            # Atomic reference swap — readers see either old or new, never partial
            self._bm25 = new_bm25
            self._built_at_len = n
            # Update snapshot atomically so search() always gets consistent (bm25, docs) pair
            self._snap = (new_bm25, docs_snapshot)

    async def search(self, query: str, k: int = 10) -> list[dict]:
        """Return top-k facts by BM25 score. Non-blocking: reads current index snapshot."""
        if not BM25_AVAILABLE:
            return []
        n = len(self._docs)
        if n == 0:
            return []
        # Trigger rebuild if needed, but don't wait — use whatever index is current
        if (self._bm25 is None or (n - self._built_at_len) >= self._rebuild_threshold) \
                and not self._rebuild_lock.locked():
            asyncio.create_task(self._rebuild_async())

        # Use consistent snapshot: bm25 index and docs list built from same data
        snap = self._snap
        if snap is None:
            return []
        bm25, docs_snapshot = snap

        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k * 2]
        results = []
        for idx in top_idx:
            if scores[idx] <= 0:
                break
            if idx >= len(docs_snapshot):
                continue
            uid, content, attrs, tier = docs_snapshot[idx]
            if attrs.get("superseded_by"):
                continue
            results.append({
                "content": content,
                "category": attrs.get("category", ""),
                "language": attrs.get("language", "en"),
                "domain":   attrs.get("domain", "general"),
                "score":    float(scores[idx]),
                "attrs":    attrs,
                "_element": uid,
                "tier":     tier,
            })
            if len(results) >= k:
                break
        return results

    async def populate_from_items(self, items: list[dict]) -> None:
        """Bulk-load from a vectorset scan (called at startup).

        Each item needs ``_element`` (or ``element``) and ``attrs``; the
        indexable text is derived per tier via :func:`extract_text`, so callers
        no longer have to know which attribute holds the body.

        Idempotent by element ID: re-populating an already-loaded element
        updates it in place instead of appending a duplicate.  Duplicate
        entries would inflate term frequencies and skew every BM25 score.
        """
        async with self._add_lock:
            existing = {uid for uid, _, _, _ in self._docs}
            added = 0
            for item in items:
                uid = item.get("_element") or item.get("element") or ""
                if not uid or uid in existing:
                    continue
                attrs = item.get("attrs", {})
                tier = item.get("tier") or tier_of(item.get("_vset", "mem:facts"))
                text = item.get("content") or extract_text("mem:" + tier, attrs)
                if not text:
                    continue
                self._docs.append((uid, text, attrs, tier))
                existing.add(uid)
                added += 1
        if added:
            log.info("[bm25] populated %d docs (tiers: %s)",
                     added, sorted({t for _, _, _, t in self._docs}))
        # Await the rebuild rather than spawning it: this only runs at startup,
        # and a fire-and-forget task leaves _snap == None so the first recall
        # after boot would silently return no keyword hits at all.
        await self._rebuild_async()

    async def stats(self) -> dict:
        """Corpus composition by tier — used by /health and diagnostics."""
        counts: dict[str, int] = {}
        for _, _, _, tier in self._docs:
            counts[tier] = counts.get(tier, 0) + 1
        return {"total": len(self._docs), "by_tier": counts}

    async def reset(self) -> None:
        async with self._add_lock:
            self._docs.clear()
        async with self._rebuild_lock:
            self._bm25 = None
            self._built_at_len = 0
            self._snap = None


async def populate_bm25_from_redis(r, bm25_index: BM25Index) -> None:
    """Load every memory tier from Redis into the BM25 corpus on startup.

    Two things were wrong with the previous version, and both silently degraded
    recall quality:

    1. It only scanned ``mem:facts``.  Episodes and procedures — the majority of
       the text-bearing corpus — had zero keyword coverage.
    2. It capped the scan at 5000 elements, so anything beyond that vanished
       from the keyword corpus with no warning.

    Enumeration now goes through the element-ID shadow index (``mem:idx:*``),
    which is exact, and falls back to ``vscan`` when the shadow index has not
    been populated yet.  Either way the shortfall is logged rather than dropped
    on the floor.
    """
    if not BM25_AVAILABLE:
        return
    from core import store as mem_store

    try:
        all_items: list[dict] = []
        for vset in (mem_store.FACT_KEY, mem_store.EPISODE_KEY, mem_store.PROC_KEY):
            # Gate on the shadow index, not on VCARD: a vectorset holding a
            # single real element (plus no __seed__ yet) has VCARD == 1 and
            # would be skipped by a naive "card <= 1" check.
            shadow = int(await r.scard(mem_store.idx_key(vset)))
            if shadow > 0:
                items = await mem_store.enumerate_elements(r, vset)
                try:
                    card = int(await r.execute_command("VCARD", vset))
                except Exception:
                    card = len(items)
                if len(items) < card:
                    log.warning(
                        "[bm25] %s: shadow index holds %d of %d elements",
                        vset, len(items), card,
                    )
            else:
                log.warning(
                    "[bm25] %s: no shadow index yet — falling back to vscan "
                    "(run the backfill to recover unreachable elements)", vset,
                )
                items = await vscan(r, vset, max_count=100_000)
                if not items:
                    continue

            for it in items:
                # normalize: enumerate_elements/vscan both say "element",
                # populate_from_items expects "_element"
                it["_element"] = it.get("_element") or it.get("element", "")
                it["tier"] = tier_of(vset)
                it["_vset"] = vset
            all_items.extend(items)

        await bm25_index.populate_from_items(all_items)
    except Exception as e:
        log.warning("populate_bm25_from_redis failed: %s", e)
