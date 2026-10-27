"""Unit tests for core/fusion.py — the operators ported from everalgo (EverOS).

These are pure-function tests: no Redis, no LLM, no network.  They pin the
algorithm contracts so a future refactor cannot silently change fusion
behaviour.
"""

import math

import pytest

from core import fusion


def C(id, score, source="", **meta):
    return fusion.Candidate(id=id, score=score, source=source, metadata=dict(meta))


# ─── RRF ─────────────────────────────────────────────────────────────────────


class TestRRF:
    def test_empty_input(self):
        assert fusion.rrf() == []

    def test_single_list_preserves_order(self):
        lst = [C("a", 0.9), C("b", 0.5), C("c", 0.1)]
        out = fusion.rrf(lst)
        assert [c.id for c in out] == ["a", "b", "c"]
        assert out[0].score == pytest.approx(1 / 61)
        assert out[1].score == pytest.approx(1 / 62)

    def test_two_lists_agree_beats_one_list_only(self):
        """A doc ranked #3 in both lists must outrank a doc ranked #1 in one list."""
        l1 = [C("x", 1.0), C("y", 0.9), C("z", 0.8)]
        l2 = [C("y", 1.0), C("x", 0.9), C("z", 0.8)]
        out = fusion.rrf(l1, l2)
        # y: 1/62 + 1/61 ; x: 1/61 + 1/62 ; z: 1/63 + 1/63
        assert out[0].id in ("x", "y")
        assert out[0].score > out[2].score

    def test_rrf_k_controls_smoothing(self):
        """Small k sharpens the top of the list; large k flattens rank gaps."""
        l1 = [C("a", 1.0), C("b", 0.9)]
        l2 = [C("a", 1.0), C("b", 0.9)]
        tight = fusion.rrf(l1, l2, k=1)
        loose = fusion.rrf(l1, l2, k=1000)
        gap_tight = tight[0].score - tight[1].score
        gap_loose = loose[0].score - loose[1].score
        assert gap_tight > gap_loose > 0

    def test_blank_ids_skipped(self):
        out = fusion.rrf([C("", 1.0), C("a", 0.5)])
        assert [c.id for c in out] == ["a"]


# ─── LR fusion ───────────────────────────────────────────────────────────────


class TestLR:
    def test_default_coefs_produce_valid_probabilities(self):
        emb = [C("a", 0.9), C("b", 0.5)]
        bm25 = [C("a", 12.0), C("c", 3.0)]
        out = fusion.lr(emb, bm25)
        assert all(0.0 <= c.score <= 1.0 for c in out)
        assert [c.id for c in out][0] == "a"

    def test_strong_dense_beats_strong_sparse_only(self):
        """LR is score-magnitude aware: a 0.95 cosine must beat a BM25-only doc."""
        emb = [C("dense_hit", 0.95)]
        bm25 = [C("kw_only", 4.0)]
        out = fusion.lr(emb, bm25)
        assert out[0].id == "dense_hit"

    def test_rrf_would_have_tied_that_case(self):
        """The behaviour LR fixes: RRF gives both docs equal rank-1 treatment."""
        emb = [C("dense_hit", 0.95)]
        bm25 = [C("kw_only", 4.0)]
        rrf_out = fusion.rrf(emb, bm25)
        assert abs(rrf_out[0].score - rrf_out[1].score) < 1e-12

    def test_custom_coefs(self):
        emb = [C("a", 0.8)]
        bm25 = [C("a", 10.0)]
        coefs = fusion.LRCoefs(emb=0.0, bm25=1.0, intercept=0.0)
        out = fusion.lr(emb, bm25, coefs=coefs)
        assert out[0].score == pytest.approx(1 / (1 + math.exp(-10)))

    def test_multi_field_weighting_four_sources(self):
        src = {
            "scene": [C("a", 0.9)],
            "global": [C("a", 0.8)],
            "bm25": [C("b", 20.0)],
            "symbolic": [C("c", 1.0)],
        }
        w = {"scene": 2.0, "global": 1.0, "bm25": 0.1, "symbolic": 3.0}
        out = fusion.multi_field_weighting(src, weights=w)
        # c: 3.0 ; a: 2*0.9+1*0.8+0.1*0 = 2.6 ; b: 0.1*20 = 2.0
        assert [c.id for c in out] == ["c", "a", "b"]

    def test_absent_source_contributes_zero(self):
        src = {"scene": [C("a", 0.9)], "bm25": []}
        out = fusion.multi_field_weighting(src, weights={"scene": 1.0, "bm25": 1.0}, intercept=0.0)
        assert out[0].score == pytest.approx(1 / (1 + math.exp(-0.9)))


# ─── vector_anchored ─────────────────────────────────────────────────────────


class TestVectorAnchored:
    def test_bm25_saturation(self):
        """Huge BM25 must not swamp a good cosine — that is the whole point."""
        dense = [C("semantic", 0.85)]
        sparse = [C("keyword", 1000.0)]
        out = fusion.vector_anchored(dense, sparse, alpha=0.7)
        assert out[0].id == "semantic"
        # saturated kw score -> 1000/1005 ~ 0.995 ; 0.7*0.85 + 0.3*0.995 = 0.8935
        assert out[0].score == pytest.approx(0.7 * 0.85 + 0.3 * (1000 / 1005))

    def test_alpha_zero_is_pure_sparse(self):
        """alpha=0 -> the dense score is ignored entirely.

        Note the floor semantics: a doc absent from the sparse list inherits the
        sparse floor, so it ties with the sparse-only doc rather than being
        zeroed out.  Both scores here must come purely from the sparse side.
        """
        dense = [C("a", 0.9)]
        sparse = [C("b", 10.0)]
        out = fusion.vector_anchored(dense, sparse, alpha=0.0)
        sat = 10.0 / (10.0 + 5.0)
        assert {c.id: c.score for c in out} == {"a": pytest.approx(sat), "b": pytest.approx(sat)}

    def test_alpha_one_is_pure_dense(self):
        dense = [C("a", 0.4)]
        sparse = [C("b", 10.0)]
        out = fusion.vector_anchored(dense, sparse, alpha=1.0)
        assert out[0].id == "a"

    def test_absent_side_uses_floor_not_zero(self):
        """A doc missing from one list inherits that list's floor, staying in contention."""
        dense = [C("a", 0.9), C("b", 0.3)]
        sparse = [C("b", 5.0)]
        out = fusion.vector_anchored(dense, sparse, alpha=0.7)
        # a has no bm25 -> gets kw_floor = 5/(5+5) = 0.5
        a_score = next(c.score for c in out if c.id == "a")
        assert a_score == pytest.approx(0.7 * 0.9 + 0.3 * 0.5)

    def test_empty_lists(self):
        assert fusion.vector_anchored([], []) == []


# ─── score_propagation / maxsim ──────────────────────────────────────────────


class TestParentChild:
    def test_score_propagation_alpha_one_trusts_child(self):
        parents = [C("ep1", 0.2)]
        children = [C("f1", 0.9, parent_id="ep1")]
        out = fusion.score_propagation(parents, children, alpha=1.0)
        assert out[0].score == pytest.approx(0.9)

    def test_score_propagation_alpha_zero_trusts_parent(self):
        parents = [C("ep1", 0.2)]
        children = [C("f1", 0.9, parent_id="ep1")]
        out = fusion.score_propagation(parents, children, alpha=0.0)
        assert out[0].score == pytest.approx(0.2)

    def test_score_propagation_unknown_parent_is_zero(self):
        parents = [C("ep1", 0.2)]
        children = [C("f1", 0.9, parent_id="missing")]
        out = fusion.score_propagation(parents, children, alpha=0.5)
        assert out[0].score == pytest.approx(0.45)

    def test_maxsim_group_keeps_max_per_parent(self):
        children = [
            C("f1", 0.4, parent_id="ep1"),
            C("f2", 0.9, parent_id="ep1"),
            C("f3", 0.7, parent_id="ep2"),
        ]
        out = fusion.maxsim_group(children, top_n=10)
        assert out == [("ep1", 0.9), ("ep2", 0.7)]

    def test_maxsim_group_respects_top_n(self):
        children = [C(f"f{i}", i / 10.0, parent_id=f"ep{i}") for i in range(1, 11)]
        out = fusion.maxsim_group(children, top_n=3)
        assert [pid for pid, _ in out] == ["ep10", "ep9", "ep8"]

    def test_maxsim_group_min_score_filter(self):
        children = [C("f1", 0.9, parent_id="ep1"), C("f2", 0.2, parent_id="ep2")]
        out = fusion.maxsim_group(children, min_score=0.5)
        assert out == [("ep1", 0.9)]

    def test_maxsim_group_skips_children_without_parent(self):
        children = [C("f1", 0.9), C("f2", 0.8, parent_id="ep1")]
        assert fusion.maxsim_group(children) == [("ep1", 0.8)]

    def test_maxsim_group_empty(self):
        assert fusion.maxsim_group([]) == []


# ─── coefficient fitting ─────────────────────────────────────────────────────


class TestFitLRCoefs:
    def test_recovers_separable_data(self):
        """Positives clearly above negatives on the dense axis -> large emb coef.

        The sparse feature is deliberately held constant: it carries no signal,
        and with a constant feature plus an intercept the coefficient is not
        identifiable, so we only assert on the dense coefficient.
        """
        pairs = []
        for i in range(200):
            pairs.append((0.90 + 0.001 * (i % 5), 1.0, 1))
            pairs.append((0.10 + 0.001 * (i % 5), 1.0, 0))
        coefs = fusion.fit_lr_coefs(pairs, iters=2000, lr_=0.5)
        assert coefs.emb > 3.0

    def test_recovers_sparse_signal_too(self):
        """When the sparse axis carries the signal, the fit must find it.

        The dense feature is held constant, so its coefficient is collinear with
        the intercept and not identifiable — only the sparse coefficient is
        asserted.
        """
        pairs = []
        for i in range(200):
            pairs.append((0.5, 12.0 + 0.01 * (i % 5), 1))
            pairs.append((0.5, 0.5 + 0.01 * (i % 5), 0))
        coefs = fusion.fit_lr_coefs(pairs, iters=3000, lr_=0.5)
        assert coefs.bm25 > 1.0

    def test_empty_pairs_returns_defaults(self):
        assert fusion.fit_lr_coefs([]) == fusion.DEFAULT_LR_COEFS

    def test_fitted_coefs_rank_relevant_first(self):
        """End-to-end: fit on a labelled draw, rank an unseen draw."""
        import random

        rng = random.Random(7)
        fit_pairs, eval_docs = [], []
        for _ in range(400):
            r = rng.betavariate(1.2, 6.0)
            d = min(1.0, max(0.0, r + rng.gauss(0, 0.07)))
            b = max(0.0, r * 12.0 + rng.gauss(2.0, 2.0)) if rng.random() < 0.35 else 0.0
            fit_pairs.append((d, b, 1 if r > 0.35 else 0))
        for _ in range(50):
            r = rng.betavariate(1.2, 6.0)
            d = min(1.0, max(0.0, r + rng.gauss(0, 0.07)))
            b = max(0.0, r * 12.0 + rng.gauss(2.0, 2.0)) if rng.random() < 0.35 else 0.0
            eval_docs.append((d, b, r))
        coefs = fusion.fit_lr_coefs(fit_pairs, iters=3000)
        ranked = sorted(eval_docs, key=lambda t: -(coefs.emb * t[0] + coefs.bm25 * t[1] + coefs.intercept))
        top5_r = sum(t[2] for t in ranked[:5]) / 5
        worst5_r = sum(t[2] for t in ranked[-5:]) / 5
        assert top5_r > worst5_r


# ─── dict adapters ───────────────────────────────────────────────────────────


class TestDictAdapters:
    def test_from_dicts_and_back(self):
        """``content`` is the fallback id — agentmem's own RRF dedups by content."""
        rows = [
            {"id": "a", "content": "x", "score": 0.9, "extra": 1},
            {"id": "", "content": "skip", "score": 0.5},
            {"content": "no id", "score": 0.1},
        ]
        cands = fusion.from_dicts(rows)
        assert [c.id for c in cands] == ["a", "skip", "no id"]
        back = fusion.to_dicts(cands)
        assert back[0]["id"] == "a"
        assert back[0]["score"] == 0.9
        assert back[0]["content"] == "x"
        assert back[0]["extra"] == 1

    def test_from_dicts_prefers_explicit_id(self):
        rows = [{"id": "f1", "content": "some text", "score": 0.7}]
        cands = fusion.from_dicts(rows)
        assert cands[0].id == "f1"

    def test_roundtrip_preserves_metadata(self):
        rows = [{"id": "f1", "score": 0.7, "parent_id": "ep1", "tier": "fact"}]
        cands = fusion.from_dicts(rows)
        assert cands[0].metadata["parent_id"] == "ep1"
        back = fusion.to_dicts(cands)
        assert back[0]["parent_id"] == "ep1"

    def test_cosine_to_lr_score_bounds(self):
        assert 0.0 <= fusion.cosine_to_lr_score(0.0) <= 1.0
        assert 0.0 <= fusion.cosine_to_lr_score(1.0) <= 1.0
        assert fusion.cosine_to_lr_score(1.0) > fusion.cosine_to_lr_score(0.0)
