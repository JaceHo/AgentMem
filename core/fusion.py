"""Fusion & aggregation operators for hybrid recall.

Ported from EverMind-AI's ``everalgo`` algorithm library (Apache-2.0), which is the
algorithm layer of EverOS (https://github.com/EverMind-AI/EverOS).  The upstream
operators are deliberately pure functions with no I/O, so they port 1:1.

What came across:

* :func:`rrf`                 — reciprocal rank fusion over N ranked lists
* :func:`lr`                  — logistic-regression fusion of dense + sparse scores
* :func:`vector_anchored`     — dense-anchored fusion with BM25 saturation
* :func:`multi_field_weighting` — LR over an arbitrary number of named sources
                                 (agentmem fuses 4: scene / global / symbolic / bm25)
* :func:`cosine_to_lr_score`  — calibrate a raw cosine into an LR probability
* :func:`score_propagation`   — blend child score into parent score
* :func:`maxsim_group`        — child -> parent max-pool aggregation
* :func:`fit_lr_coefs`        — fit LR coefficients from labelled pairs

Why bother when ``main._rrf_merge`` already does weighted RRF?  Because RRF is
*rank*-based: it throws away score magnitude.  Two VSIM hits at cosine 0.91 and
0.62 contribute the same as two hits at 0.52 and 0.51.  LR / vector_anchored
consume the actual scores, which is what lets a strong dense hit survive a weak
sparse list (and vice versa) instead of being averaged away.

The coefficients in :data:`DEFAULT_LR_COEFS` were trained on EverOS's own score
distribution (LanceDB cosine + tantivy BM25).  AgentMem's scales differ (Redis
vectorset VSIM cosine + rank_bm25 Okapi), so treat them as a starting point and
refit with :func:`fit_lr_coefs` on your own recall logs before trusting them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = [
    "DEFAULT_LR_COEFS",
    "Candidate",
    "LRCoefs",
    "cosine_to_lr_score",
    "fit_lr_coefs",
    "from_dicts",
    "lr",
    "maxsim_group",
    "multi_field_weighting",
    "rrf",
    "score_propagation",
    "to_dicts",
    "vector_anchored",
]


# ─── Candidate ───────────────────────────────────────────────────────────────


@dataclass
class Candidate:
    """One ranked retrieval result.

    ``id`` is the dedup key.  ``source`` records which retriever produced it.
    ``metadata`` carries per-source side data; ``parent_id`` is read by
    :func:`maxsim_group` and :func:`score_propagation`.
    """

    id: str
    score: float
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy_with(self, score: float) -> "Candidate":
        return Candidate(id=self.id, score=score, source=self.source, metadata=dict(self.metadata))


def from_dicts(items: Sequence[dict], id_key: str = "id", score_key: str = "score") -> list[Candidate]:
    """Adapt agentmem's plain-dict recall results into :class:`Candidate` objects."""
    out: list[Candidate] = []
    for it in items:
        cid = it.get(id_key) or it.get("content") or ""
        if not cid:
            continue
        meta = {k: v for k, v in it.items() if k not in (id_key, score_key)}
        out.append(Candidate(id=str(cid), score=float(it.get(score_key, 0.0) or 0.0), source=str(meta.get("source", "")), metadata=meta))
    return out


def to_dicts(cands: Sequence[Candidate], extra: dict[str, Any] | None = None) -> list[dict]:
    """Flatten :class:`Candidate` objects back into agentmem's dict shape."""
    out: list[dict] = []
    for c in cands:
        row: dict[str, Any] = {"id": c.id, "score": c.score}
        if c.source:
            row["source"] = c.source
        row.update(c.metadata)
        if extra:
            row.update(extra)
        out.append(row)
    return out


# ─── RRF ─────────────────────────────────────────────────────────────────────


def rrf(*sources: Sequence[Candidate], k: int = 60) -> list[Candidate]:
    """Reciprocal Rank Fusion over N ranked lists.

    ``score(d) = sum_i 1 / (k + rank_i(d))``, equal weights across sources.
    """
    acc: dict[str, float] = {}
    keep: dict[str, Candidate] = {}
    for lst in sources:
        for rank, c in enumerate(lst, start=1):
            if not c.id:
                continue
            keep.setdefault(c.id, c)
            acc[c.id] = acc.get(c.id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(acc.items(), key=lambda kv: kv[1], reverse=True)
    return [keep[cid].copy_with(score) for cid, score in ranked]


# ─── Logistic-regression fusion ──────────────────────────────────────────────


class LRCoefs(tuple):
    """Trained LR coefficients for the 2-source (dense + sparse) fusion."""

    __slots__ = ()

    def __new__(cls, emb: float, bm25: float, intercept: float) -> "LRCoefs":
        return super().__new__(cls, (emb, bm25, intercept))

    @property
    def emb(self) -> float:
        return self[0]

    @property
    def bm25(self) -> float:
        return self[1]

    @property
    def intercept(self) -> float:
        return self[2]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"LRCoefs(emb={self.emb:.6f}, bm25={self.bm25:.6f}, intercept={self.intercept:.6f})"


DEFAULT_LR_COEFS = LRCoefs(
    emb=6.27473151675093,
    bm25=0.09395183408310023,
    intercept=-4.858095765012703,
)
"""EverOS production coefficients.  See module docstring: refit before trusting."""


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def multi_field_weighting(
    sources: dict[str, Sequence[Candidate]],
    *,
    weights: dict[str, float] | None = None,
    intercept: float = 0.0,
) -> list[Candidate]:
    """Fuse N named ranked lists into one LR probability per document.

    ``weights=None`` uses :data:`DEFAULT_LR_COEFS` (2-source mode, keys must be
    ``{"emb", "bm25"}``).  Otherwise caller supplies one coefficient per source
    name — this is the mode agentmem wants for its 4 recall passes.

    Documents missing from a source contribute 0 for that source, which is the
    standard "absent signal" convention: a doc only in the sparse list is scored
    as if its dense score were 0.
    """
    if weights is None:
        weights = {"emb": DEFAULT_LR_COEFS.emb, "bm25": DEFAULT_LR_COEFS.bm25}
        intercept = DEFAULT_LR_COEFS.intercept

    by_id: dict[str, dict[str, float]] = {}
    keep: dict[str, Candidate] = {}
    for name, lst in sources.items():
        for c in lst:
            if not c.id:
                continue
            keep.setdefault(c.id, c)
            by_id.setdefault(c.id, {})[name] = c.score

    scored: list[tuple[str, float]] = []
    for cid, sm in by_id.items():
        logit = intercept + sum(sm.get(name, 0.0) * coef for name, coef in weights.items())
        scored.append((cid, _sigmoid(logit)))

    scored.sort(key=lambda kv: kv[1], reverse=True)
    return [keep[cid].copy_with(score) for cid, score in scored]


def lr(
    emb_results: Sequence[Candidate],
    bm25_results: Sequence[Candidate],
    *,
    coefs: LRCoefs | None = None,
) -> list[Candidate]:
    """Logistic-regression fusion of dense + sparse candidates."""
    c = coefs or DEFAULT_LR_COEFS
    return multi_field_weighting(
        {"emb": list(emb_results), "bm25": list(bm25_results)},
        weights={"emb": c.emb, "bm25": c.bm25},
        intercept=c.intercept,
    )


def cosine_to_lr_score(sim: float, parent_bm25: float = 0.0, *, coefs: LRCoefs | None = None) -> float:
    """Calibrate a raw cosine similarity into an LR probability in ``[0, 1]``."""
    c = coefs or DEFAULT_LR_COEFS
    return _sigmoid(c.intercept + c.emb * sim + c.bm25 * parent_bm25)


# ─── Vector-anchored fusion ──────────────────────────────────────────────────


def vector_anchored(
    dense: Sequence[Candidate],
    sparse: Sequence[Candidate],
    *,
    saturation_k: float = 5.0,
    alpha: float = 0.7,
) -> list[Candidate]:
    """Dense-anchored fusion: ``alpha*dense + (1-alpha)*saturate(bm25)``.

    BM25 is squashed through ``s / (s + k)`` so its unbounded scale cannot
    dominate the bounded cosine.  A doc absent from one side inherits that
    side's *floor* (its worst observed score) rather than 0, which keeps
    single-list docs in contention instead of burying them.
    """
    vec: dict[str, float] = {c.id: c.score for c in dense if c.id}
    kw: dict[str, float] = {c.id: (c.score / (c.score + saturation_k) if c.score > 0 else 0.0) for c in sparse if c.id}

    vec_floor = min(vec.values()) if vec else 0.0
    kw_floor = min(kw.values()) if kw else 0.0

    keep: dict[str, Candidate] = {c.id: c for c in dense if c.id}
    for c in sparse:
        if c.id and c.id not in keep:
            keep[c.id] = c

    out: list[Candidate] = []
    for cid, c in keep.items():
        vs = vec.get(cid, vec_floor)
        ks = kw.get(cid, kw_floor)
        out.append(c.copy_with(alpha * vs + (1.0 - alpha) * ks))
    out.sort(key=lambda c: c.score, reverse=True)
    return out


# ─── Parent/child blending ───────────────────────────────────────────────────


def score_propagation(
    parents: Sequence[Candidate],
    children: Sequence[Candidate],
    *,
    alpha: float = 1.0,
) -> list[Candidate]:
    """Blend a child's score into its parent: ``final = alpha*child + (1-alpha)*parent``.

    ``alpha=1.0`` (default) means "trust the child completely" — used when the
    child is a finer-grained chunk (atomic fact) of the parent (episode).
    """
    plook = {p.id: p.score for p in parents}
    out: list[Candidate] = []
    for ch in children:
        pid = ch.metadata.get("parent_id", "")
        pscore = plook.get(pid, 0.0) if isinstance(pid, str) else 0.0
        out.append(ch.copy_with(alpha * ch.score + (1.0 - alpha) * pscore))
    return out


def maxsim_group(
    children: Sequence[Candidate],
    *,
    top_n: int = 20,
    min_score: float | None = None,
    parent_key: str = "parent_id",
) -> list[tuple[str, float]]:
    """MaxSim aggregation: group children by parent, keep the max child score.

    Pure aggregation — no I/O.  Returns ``[(parent_id, max_child_score), ...]``
    sorted descending, at most ``top_n`` entries, optionally filtered by
    ``min_score``.  The caller hydrates the parents (agentmem: one ``mget`` /
    pipeline instead of N round trips).
    """
    parent_max: dict[str, float] = {}
    for c in children:
        pid = c.metadata.get(parent_key)
        if not isinstance(pid, str) or not pid:
            continue
        if c.score > parent_max.get(pid, float("-inf")):
            parent_max[pid] = c.score

    if not parent_max:
        return []

    ranked = sorted(parent_max.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    if min_score is not None:
        ranked = [(pid, s) for pid, s in ranked if s >= min_score]
    return ranked


# ─── Coefficient fitting ─────────────────────────────────────────────────────


def fit_lr_coefs(
    pairs: Sequence[tuple[float, float, int]],
    *,
    l2: float = 1e-3,
    iters: int = 4000,
    lr_: float = 0.5,
) -> LRCoefs:
    """Fit ``(emb_coef, bm25_coef, intercept)`` from labelled pairs.

    ``pairs`` is ``[(dense_score, sparse_score, label)]`` with label 1 for a
    relevant document and 0 for a non-relevant one, harvested from recall logs
    where you know which hit the user actually used.

    Plain gradient descent on the logistic loss — no sklearn dependency, ~50
    lines, converges in well under a second for a few thousand pairs.  Use this
    instead of :data:`DEFAULT_LR_COEFS` once you have labels: those numbers are
    EverOS's, not yours.
    """
    if not pairs:
        return DEFAULT_LR_COEFS
    n = len(pairs)
    w = [1.0, 1.0, 0.0]  # emb, bm25, intercept
    for _ in range(iters):
        gw = [0.0, 0.0, 0.0]
        for d, s, y in pairs:
            x = (d, s, 1.0)
            p = _sigmoid(w[0] * x[0] + w[1] * x[1] + w[2] * x[2])
            err = p - y
            for j in range(3):
                gw[j] += err * x[j]
        for j in range(3):
            gw[j] = gw[j] / n + l2 * w[j]
            w[j] -= lr_ * gw[j]
    return LRCoefs(emb=w[0], bm25=w[1], intercept=w[2])
