"""Local-first FrugalGPT extract cascade: fast first, wall budget, timeout circuit."""

from core import extractor as ex


def test_timeout_opens_circuit_on_first_miss(monkeypatch):
    ex._model_fail_times.clear()
    ex._model_fail_counts.clear()
    ex.mark_model_failed("slow-model", reason="timeout")
    assert ex.is_model_circuit_broken("slow-model")


def test_non_timeout_needs_two_strikes(monkeypatch):
    ex._model_fail_times.clear()
    ex._model_fail_counts.clear()
    ex.mark_model_failed("flaky", reason="other")
    assert not ex.is_model_circuit_broken("flaky")
    ex.mark_model_failed("flaky", reason="other")
    assert ex.is_model_circuit_broken("flaky")


def test_defaults_delegate_cascade_to_aiserv():
    assert "extract" in ex.AISERV_EXTRACT_MODEL
    assert ex.LLM_MAX_RETRIES == 0  # no client-side model lottery
    assert ex.EXTRACT_BUDGET_S <= 20.0


def test_preference_regex_works_without_llm():
    text = "我叫审计员，偏好用 bun 做前端验证，记住这件事。"
    facts = ex._regex_extract(text)
    cats = {f.category for f in facts}
    assert "preference" in cats or "identity" in cats
    assert any("bun" in f.content.lower() or "偏好" in f.content for f in facts)


def test_persona_context_capped():
    import asyncio
    from core import persona as persona_mod

    class FakeRedis:
        async def hgetall(self, key):
            return {
                b"name": ("X" * 400 + "; " + "Y" * 400).encode(),
                b"work": ("W" * 400 + "; " + "Z" * 400).encode(),
                b"preferences": ("prefer bun; prefer redis; prefer python; prefer more").encode(),
                b"location": b"somewhere far away with a long path description " * 5,
            }

    ctx = asyncio.run(persona_mod.get_context(FakeRedis()))
    assert ctx.startswith("## User Profile")
    assert len(ctx) <= persona_mod._MAX_CONTEXT_CHARS + 40
