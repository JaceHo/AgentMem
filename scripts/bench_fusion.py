#!/usr/bin/env python3
"""Fusion operator bake-off: agentmem's wRRF vs ported everalgo operators.

The question this answers: does throwing away score magnitude (RRF) cost
retrieval quality versus consuming the actual scores (LR / vector_anchored)?

Setup
-----
We synthesise a corpus where every document has a latent relevance ``r``, then
simulate four recall passes with *different score scales and different
reliability* — which is exactly agentmem's situation:

  scene    VSIM cosine, high reliability,  scale [0, 1]
  global   VSIM cosine, medium reliability, scale [0, 1]
  bm25     Okapi BM25,  keyword-blind,     scale [0, ~15]
  symbolic graph/entity hit, very high precision but only fires on ~15% of docs

Gold = top-M documents by latent relevance.

Metrics: MRR, nDCG@10, Recall@10, Precision@5.  20 seeds, 2000 docs each.

Run:  .venv/bin/python scripts/bench_fusion.py [--seeds 20] [--docs 2000]
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main as agentmem_main  # noqa: E402  (agentmem's production recall module)
from core import fusion  # noqa: E402

K = 10
TOP_M = 40  # gold set size


# ─── synthetic corpus ────────────────────────────────────────────────────────


def build_corpus(rng: random.Random, n_docs: int, cfg: dict | None = None):
    """Each doc gets a latent relevance and four pass scores.

    The passes disagree on scale and reliability on purpose:
      * scene/global return cosine in [0,1] but with different noise floors
      * bm25 is unbounded and mostly blind (only fires on lexical overlap)
      * symbolic is near-perfect when it fires, and fires rarely

    ``cfg`` overrides the generative knobs so we can check the conclusion is not
    an artefact of one particular synthetic assumption.
    """
    cfg = {**_DEFAULT_CFG, **(cfg or {})}
    docs = []
    for i in range(n_docs):
        # Latent relevance: heavy-tailed, like real memory hit distributions.
        r = rng.betavariate(cfg["rel_a"], cfg["rel_b"])

        # Two dense passes over the same latent signal, different noise.
        scene = min(1.0, max(0.0, r + rng.gauss(0, cfg["scene_noise"])))
        glob = min(1.0, max(0.0, r + rng.gauss(0, cfg["glob_noise"])))

        # BM25: lexical overlap is a *partial* signal. Amplify when the doc
        # happens to share vocabulary, otherwise near-zero regardless of r.
        lexical = rng.random() < cfg["lexical_rate"]
        if lexical:
            bm25 = max(0.0, r * cfg["bm25_gain"] + rng.gauss(2.0, 2.0))
        else:
            bm25 = max(0.0, rng.gauss(0.4, 0.5))

        # Symbolic: precise but rare. Fires on entity-bearing queries only.
        if rng.random() < cfg["sym_rate"]:
            symbolic = min(1.0, r + rng.gauss(0, 0.03))
        else:
            symbolic = 0.0

        docs.append(
            {
                "id": f"d{i}",
                "content": f"memory item {i}",
                "r": r,
                "scene": scene,
                "global": glob,
                "bm25": bm25,
                "symbolic": symbolic,
            }
        )
    return docs


_DEFAULT_CFG = {
    "rel_a": 1.2,
    "rel_b": 6.0,
    "scene_noise": 0.07,
    "glob_noise": 0.13,
    "lexical_rate": 0.35,
    "bm25_gain": 12.0,
    "sym_rate": 0.15,
}


def to_ranked(docs, key, k):
    """Top-k by a pass's score, as the Candidate shape agentmem passes around."""
    sel = sorted(docs, key=lambda d: d[key], reverse=True)[:k]
    return [fusion.Candidate(id=d["id"], score=d[key], source=key, metadata={"parent_id": ""}) for d in sel]


# ─── metrics ─────────────────────────────────────────────────────────────────


def ndcg_at_k(ranked_ids, gold_ids, k=K):
    dcg = 0.0
    for i, did in enumerate(ranked_ids[:k], start=1):
        if did in gold_ids:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold_ids), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate(ranked, gold_ids):
    ids = [c.id for c in ranked]
    mrr = 0.0
    for i, did in enumerate(ids, start=1):
        if did in gold_ids:
            mrr = 1.0 / i
            break
    rec = len(set(ids[:K]) & gold_ids) / len(gold_ids)
    prec = len(set(ids[:5]) & gold_ids) / 5.0
    return {"mrr": mrr, "ndcg": ndcg_at_k(ids, gold_ids), "recall": rec, "p5": prec}


# ─── fusion methods under test ───────────────────────────────────────────────


def method_wrrf(docs, k):
    """agentmem's production path: 4-list weighted RRF over dicts."""
    lists = [
        sorted(docs, key=lambda d: d["scene"], reverse=True)[:k],
        sorted(docs, key=lambda d: d["global"], reverse=True)[:k],
        sorted(docs, key=lambda d: d["symbolic"], reverse=True)[:k],
        sorted(docs, key=lambda d: d["bm25"], reverse=True)[:k],
    ]
    dicts = [
        [{"id": d["id"], "content": d["content"], "score": d["scene"]} for d in lst] for lst in lists
    ]
    fused = agentmem_main._rrf_merge(dicts, weights=[1.0, 1.0, 1.0, 1.0], k=60, limit=K)
    return [fusion.Candidate(id=f["id"], score=f["score"]) for f in fused]


def method_wrrf_tuned(docs, k):
    """Same, but with the query-type weights main.py already supports."""
    lists = [
        sorted(docs, key=lambda d: d["scene"], reverse=True)[:k],
        sorted(docs, key=lambda d: d["global"], reverse=True)[:k],
        sorted(docs, key=lambda d: d["symbolic"], reverse=True)[:k],
        sorted(docs, key=lambda d: d["bm25"], reverse=True)[:k],
    ]
    dicts = [[{"id": d["id"], "content": d["content"], "score": d[p]} for d in lst] for lst, p in
             zip(lists, ["scene", "global", "symbolic", "bm25"])]
    fused = agentmem_main._rrf_merge(dicts, weights=[1.0, 1.0, 1.4, 0.8], k=60, limit=K)
    return [fusion.Candidate(id=f["id"], score=f["score"]) for f in fused]


def method_lr_default(docs, k):
    """Ported LR fusion with EverOS's coefficients (2-source: emb=scene, bm25)."""
    scene = to_ranked(docs, "scene", k)
    glob = to_ranked(docs, "global", k)
    bm25 = to_ranked(docs, "bm25", k)
    # dense side = max(scene, global) — the two cosine passes agree on signal
    dense_map: dict[str, float] = {}
    for c in scene + glob:
        dense_map[c.id] = max(dense_map.get(c.id, 0.0), c.score)
    dense = [fusion.Candidate(id=i, score=s, source="dense") for i, s in dense_map.items()]
    return fusion.lr(dense, bm25)[:K]


def method_lr_fitted(docs, k, pairs):
    """Ported LR, coefficients fitted on this corpus's labelled pairs."""
    coefs = fusion.fit_lr_coefs(pairs)
    scene = to_ranked(docs, "scene", k)
    glob = to_ranked(docs, "global", k)
    bm25 = to_ranked(docs, "bm25", k)
    dense_map: dict[str, float] = {}
    for c in scene + glob:
        dense_map[c.id] = max(dense_map.get(c.id, 0.0), c.score)
    dense = [fusion.Candidate(id=i, score=s, source="dense") for i, s in dense_map.items()]
    return fusion.lr(dense, bm25, coefs=coefs)[:K]


def method_vec_anchored(docs, k):
    """Ported vector_anchored (alpha=0.7) on the same dense/sparse split."""
    scene = to_ranked(docs, "scene", k)
    glob = to_ranked(docs, "global", k)
    bm25 = to_ranked(docs, "bm25", k)
    dense_map: dict[str, float] = {}
    for c in scene + glob:
        dense_map[c.id] = max(dense_map.get(c.id, 0.0), c.score)
    dense = [fusion.Candidate(id=i, score=s, source="dense") for i, s in dense_map.items()]
    return fusion.vector_anchored(dense, bm25, alpha=0.7)[:K]


def method_lr_4source(docs, k, pairs):
    """Ported multi-field LR over all four passes, fitted on a disjoint corpus."""
    coefs4 = _fit_multi(pairs)
    w = {"scene": coefs4[0], "global": coefs4[1], "bm25": coefs4[2], "symbolic": coefs4[3]}
    src = {
        "scene": to_ranked(docs, "scene", k),
        "global": to_ranked(docs, "global", k),
        "bm25": to_ranked(docs, "bm25", k),
        "symbolic": to_ranked(docs, "symbolic", k),
    }
    return fusion.multi_field_weighting(src, weights=w, intercept=coefs4[4])[:K]
    # 4-source LR fit via the same gradient descent, one coef per source.
    coefs4 = _fit_multi(pairs4(docs, k))
    w = {"scene": coefs4[0], "global": coefs4[1], "bm25": coefs4[2], "symbolic": coefs4[3]}
    return fusion.multi_field_weighting(src, weights=w, intercept=coefs4[4])[:K]


def pairs4(docs, k):
    """Labelled pairs for the 4-source fit: (scene, global, bm25, sym, label)."""
    gold = {d["id"] for d in sorted(docs, key=lambda d: d["r"], reverse=True)[:TOP_M]}
    out = []
    for d in docs:
        out.append((d["scene"], d["global"], d["bm25"], d["symbolic"], 1 if d["id"] in gold else 0))
    return out


def _fit_multi(pairs, l2=1e-3, iters=3000, lr_=0.5):
    n = len(pairs)
    w = [0.0] * 5
    for _ in range(iters):
        gw = [0.0] * 5
        for *x, y in pairs:
            z = sum(wi * xi for wi, xi in zip(w[:4], x)) + w[4]
            p = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
            err = p - y
            for j in range(4):
                gw[j] += err * x[j]
            gw[4] += err
        for j in range(5):
            gw[j] = gw[j] / n + l2 * w[j]
            w[j] -= lr_ * gw[j]
    return w


# ─── driver ──────────────────────────────────────────────────────────────────


def run(seeds: int, n_docs: int, recall_k: int, cfg: dict | None = None, label: str = "") -> tuple:
    """Held-out evaluation.

    Coefficients are fitted on a *disjoint* corpus (different seed offset) and
    evaluated on the corpus under test.  Fitting and evaluating on the same gold
    labels would be in-sample leakage and would inflate the fitted methods.
    """
    agg: dict[str, list[dict]] = {}

    for seed in range(seeds):
        rng = random.Random(1000 + seed)
        docs = build_corpus(rng, n_docs, cfg)
        gold_ids = {d["id"] for d in sorted(docs, key=lambda d: d["r"], reverse=True)[:TOP_M]}

        # Fitting corpus: same generative process, independent draw.
        fit_rng = random.Random(900000 + seed)
        fit_docs = build_corpus(fit_rng, n_docs, cfg)
        fit_gold = {d["id"] for d in sorted(fit_docs, key=lambda d: d["r"], reverse=True)[:TOP_M]}

        pairs2 = []
        for d in fit_docs:
            dense = max(d["scene"], d["global"])
            pairs2.append((dense, d["bm25"], 1 if d["id"] in fit_gold else 0))

        methods = {
            "wRRF (production)": lambda: method_wrrf(docs, recall_k),
            "wRRF (tuned wts)": lambda: method_wrrf_tuned(docs, recall_k),
            "LR (everos coefs)": lambda: method_lr_default(docs, recall_k),
            "LR (fitted 2-src)": lambda: method_lr_fitted(docs, recall_k, pairs2),
            "vector_anchored": lambda: method_vec_anchored(docs, recall_k),
            "LR (fitted 4-src)": lambda: method_lr_4source(docs, recall_k, pairs4(fit_docs, recall_k)),
        }
        for name, fn in methods.items():
            res = evaluate(fn(), gold_ids)
            agg.setdefault(name, []).append(res)

    names = list(agg)
    means: dict[str, dict[str, float]] = {}
    for name in names:
        rs = agg[name]
        means[name] = {
            "mrr": statistics.mean(r["mrr"] for r in rs),
            "ndcg": statistics.mean(r["ndcg"] for r in rs),
            "recall": statistics.mean(r["recall"] for r in rs),
            "p5": statistics.mean(r["p5"] for r in rs),
        }

    if label:
        print(f"\n=== scenario: {label} ===")
        print(f"{'method':<22} {'MRR':>8} {'nDCG@10':>9} {'R@10':>8} {'P@5':>8}")
        print("-" * 60)
        base = None
        for name in names:
            m = means[name]
            if name == "wRRF (production)":
                base = m
            print(f"{name:<22} {m['mrr']:>8.4f} {m['ndcg']:>9.4f} {m['recall']:>8.4f} {m['p5']:>8.4f}")
        if base:
            print("Δ vs production wRRF:")
            print(f"{'method':<22} {'MRR':>10} {'nDCG@10':>10} {'R@10':>10} {'P@5':>10}")
            print("-" * 64)
            for name in names:
                m = means[name]
                d = [(m[k2] - base[k2]) / base[k2] * 100 if base[k2] else 0.0 for k2 in ("mrr", "ndcg", "recall", "p5")]
                print(f"{name:<22} {d[0]:>+9.2f}% {d[1]:>+9.2f}% {d[2]:>+9.2f}% {d[3]:>+9.2f}%")
    return means


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--docs", type=int, default=2000)
    ap.add_argument("--recall-k", type=int, default=100)
    ap.add_argument("--sweep", action="store_true", help="run the sensitivity sweep instead of the default scenario")
    a = ap.parse_args()

    if a.sweep:
        scenarios = {
            "default": {},
            "bm25 blind (lexical_rate=0.10)": {"lexical_rate": 0.10},
            "bm25 strong (lexical_rate=0.70)": {"lexical_rate": 0.70},
            "dense noisy (glob_noise=0.30)": {"glob_noise": 0.30},
            "symbolic rare (sym_rate=0.05)": {"sym_rate": 0.05},
            "symbolic common (sym_rate=0.45)": {"sym_rate": 0.45},
            "flat relevance (rel_b=1.5)": {"rel_b": 1.5},
        }
        all_means: dict[str, dict[str, dict[str, float]]] = {}
        for name, cfg in scenarios.items():
            all_means[name] = run(a.seeds, a.docs, a.recall_k, cfg, label=name)

        print("\n" + "=" * 78)
        print("SENSITIVITY SUMMARY — nDCG@10 Δ% vs production wRRF")
        print("=" * 78)
        hdr = f"{'scenario':<34}" + "".join(f"{m[:16]:>17}" for m in
              ["LR(everos)", "LR(fit2)", "vec_anchor", "LR(fit4)", "wRRF tuned"])
        print(hdr)
        print("-" * 78)
        key = {"LR(everos)": "LR (everos coefs)", "LR(fit2)": "LR (fitted 2-src)",
               "vec_anchor": "vector_anchored", "LR(fit4)": "LR (fitted 4-src)",
               "wRRF tuned": "wRRF (tuned wts)"}
        for sname, means in all_means.items():
            base = means["wRRF (production)"]["ndcg"]
            row = f"{sname:<34}"
            for short in ["LR(everos)", "LR(fit2)", "vec_anchor", "LR(fit4)", "wRRF tuned"]:
                v = means[key[short]]["ndcg"]
                row += f"{(v - base) / base * 100:>+16.1f}%"
            print(row)
        return 0

    run(a.seeds, a.docs, a.recall_k, label="default")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
