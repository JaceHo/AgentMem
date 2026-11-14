"""House-word expansion must not wait on a translation that times out.

Measured 2026-11-02: a novel Chinese query spent 8.4s, the LLM returned nothing,
and that empty string was cached so the English BM25 pass never ran again.
"""
from core.kb import glossary_expansion


def test_a_watchdog_question_expands_without_a_model():
    out = glossary_expansion("看门狗跑了几轮才把失败登记、几轮才升级成 P0？")
    assert "watchdog" in out and "ESCALATE_AT" in out


def test_an_unrelated_question_adds_no_terms():
    assert glossary_expansion("公司 2026 Q3 的营收是多少？") == ""
