"""
Core search utilities — BM25 hybrid index, vector scan helper, and embedding cache.

Extracted from main.py to provide reusable search primitives across route modules.
"""

import asyncio
import json
import re
from functools import lru_cache

import numpy as np

from core import embedder

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


# ── Embedding cache ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1024)
def cached_encode(text: str) -> bytes:
    """Cache raw embedding bytes (immutable, safe to share)."""
    return embedder.encode(text).tobytes()


def encode(text: str) -> np.ndarray:
    """Return a mutable float32 embedding vector for *text*."""
    return np.frombuffer(cached_encode(text), dtype=np.float32).copy()


# ── Zero-vector for VSIM scan-all ─────────────────────────────────────────────

_ZERO_VEC = np.zeros(embedder.DIMS, dtype=np.float32)


async def vscan(r, vset_key: str, max_count: int = 200) -> list[dict]:
    """Scan all elements in a vectorset using zero-vector VSIM.

    Returns list of {element, score, attrs} dicts, excluding __seed__ entries.
    """
    card = await r.execute_command("VCARD", vset_key)
    if not card or int(card) <= 1:
        return []
    try:
        results = await r.execute_command(
            "VSIM", vset_key, "FP32", _ZERO_VEC.tobytes(),
            "COUNT", min(int(card), max_count), "WITHSCORES", "WITHATTRIBS"
        )
    except Exception:
        return []
    items = []
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
        if attrs.get("_seed"):
            continue
        items.append({"element": elem_str, "score": float(score) if score else 0.0, "attrs": attrs})
    return items


# ── BM25 in-memory index ──────────────────────────────────────────────────────

class BM25Index:
    """Lazy-rebuild BM25 corpus over fact contents + triple strings.

    Thread-safe: uses asyncio.Lock to prevent concurrent mutation during search.
    Superseded facts (superseded_by != "") are excluded from search results.
    Rebuilt when corpus size changes by >=10 entries or when forced.
    """

    def __init__(self):
        self._docs: list[tuple[str, str, dict]] = []   # (uid, content, attrs)
        self._bm25 = None
        self._built_at_len: int = 0
        self._rebuild_threshold: int = 10
        self._lock = asyncio.Lock()

    async def add(self, uid: str, content: str, attrs: dict) -> None:
        """Add a fact to the corpus. Invalidates the BM25 index."""
        async with self._lock:
            self._docs.append((uid, content, attrs))

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize for BM25: English words >=3 chars + CJK bigrams."""
        text = text.lower()
        en = re.findall(r'\b[a-z]{3,}\b', text)
        zh = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]{2,}', text)
        return en + zh

    def _ensure_built(self) -> bool:
        """Rebuild BM25 index if corpus has changed enough. Returns False if corpus empty."""
        n = len(self._docs)
        if n == 0:
            return False
        if self._bm25 is None or (n - self._built_at_len) >= self._rebuild_threshold:
            corpus = []
            for _, content, attrs in self._docs:
                if attrs.get("superseded_by"):
                    corpus.append([])  # empty token list -> zero BM25 score
                    continue
                text = content
                triple_str = attrs.get("triple_str", "")
                if triple_str:
                    text = f"{text} {triple_str}"
                corpus.append(self._tokenize(text))
            self._bm25 = _BM25Okapi(corpus)
            self._built_at_len = n
        return True

    async def search(self, query: str, k: int = 10) -> list[dict]:
        """Return top-k facts by BM25 score. Returns [] if corpus empty or BM25 unavailable."""
        async with self._lock:
            if not BM25_AVAILABLE or not self._ensure_built():
                return []
            tokens = self._tokenize(query)
            if not tokens:
                return []
            scores = self._bm25.get_scores(tokens)
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k * 2]
            results = []
            for idx in top_idx:
                if scores[idx] <= 0:
                    break
                uid, content, attrs = self._docs[idx]
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
        async with self._lock:
            for item in items:
                uid = item.get("_element", "")
                content = item.get("content", "")
                attrs = item.get("attrs", {})
                if uid and content:
                    self._docs.append((uid, content, attrs))

    async def reset(self) -> None:
        async with self._lock:
            self._docs.clear()
            self._bm25 = None
            self._built_at_len = 0
