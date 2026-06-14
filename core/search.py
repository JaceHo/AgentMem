"""
Core search utilities — BM25 hybrid index, vector scan helper, and embedding cache.

Extracted from main.py to provide reusable search primitives across route modules.
"""

import asyncio
import re
from functools import lru_cache

import numpy as np

from core import embedder
from core.utils import decode_bytes, decode_attrs

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


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
    """Scan all elements in a vectorset using zero-vector VSIM.

    Returns list of {element, score, attrs} dicts, excluding __seed__ entries.
    """
    card = await r.execute_command("VCARD", vset_key)
    if not card or int(card) <= 1:
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
        self._docs: list[tuple[str, str, dict]] = []   # (uid, content, attrs)
        self._bm25 = None
        self._built_at_len: int = 0
        self._rebuild_threshold: int = 10
        self._add_lock = asyncio.Lock()      # protects list append only
        self._rebuild_lock = asyncio.Lock()  # prevents duplicate concurrent rebuilds

    async def add(self, uid: str, content: str, attrs: dict) -> None:
        """Add a fact to the corpus. Triggers a background rebuild if threshold crossed."""
        async with self._add_lock:
            self._docs.append((uid, content, attrs))
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
            self._docs = [(u, c, a) for u, c, a in self._docs if u not in uids]
            removed = before - len(self._docs)
        if removed > 0 and not self._rebuild_lock.locked():
            asyncio.create_task(self._rebuild_async())

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize for BM25: English words >=3 chars + CJK bigrams."""
        text = text.lower()
        en = re.findall(r'\b[a-z]{3,}\b', text)
        zh = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]{2,}', text)
        return en + zh

    def _build_corpus(self, docs: list[tuple[str, str, dict]]):
        """CPU-bound corpus construction — runs in thread pool via asyncio.to_thread."""
        if not BM25_AVAILABLE:
            return None, 0
        corpus = []
        for _, content, attrs in docs:
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
        bm25 = self._bm25  # atomic reference read (GIL guarantees this)
        if bm25 is None:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        docs_snapshot = self._docs  # read current list reference
        scores = bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k * 2]
        results = []
        for idx in top_idx:
            if scores[idx] <= 0:
                break
            if idx >= len(docs_snapshot):
                continue
            uid, content, attrs = docs_snapshot[idx]
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
            })
            if len(results) >= k:
                break
        return results

    async def populate_from_items(self, items: list[dict]) -> None:
        """Bulk-load from knn_search result items (called at startup)."""
        async with self._add_lock:
            for item in items:
                uid = item.get("_element", "")
                content = item.get("content", "")
                attrs = item.get("attrs", {})
                if uid and content:
                    self._docs.append((uid, content, attrs))
        # Trigger a single rebuild after bulk load
        asyncio.create_task(self._rebuild_async())

    async def reset(self) -> None:
        async with self._add_lock:
            self._docs.clear()
        async with self._rebuild_lock:
            self._bm25 = None
            self._built_at_len = 0


async def populate_bm25_from_redis(r, bm25_index: BM25Index) -> None:
    """Load all existing facts from Redis into the in-memory BM25 corpus on startup."""
    if not BM25_AVAILABLE:
        return
    try:
        from core import store as mem_store
        scanned = await vscan(r, mem_store.FACT_KEY, max_count=5000)
        items = [
            {"_element": item["element"], "content": item["attrs"]["content"], "attrs": item["attrs"]}
            for item in scanned
            if item["attrs"].get("content") and not item["attrs"].get("superseded_by")
        ]
        await bm25_index.populate_from_items(items)
    except Exception as e:
        log.warning("populate_bm25_from_redis failed: %s", e)
