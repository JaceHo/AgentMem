"""Persona injection must stay a short always-on prefix under the token budget."""

from main import _format_prepend


def test_persona_cannot_monopolize_budget():
    persona = "## User Profile\n" + ("- preferences: prefer bun over npm for agentmem audit. " * 40)
    facts = [{"content": f"fact-{i} about retrieval", "category": "general", "attrs": {}} for i in range(8)]
    episodes = [{"content": f"episode-{i} recent work", "attrs": {}} for i in range(6)]
    out = _format_prepend(
        facts, episodes, None, persona,
        token_budget=1200,
    )
    assert out.startswith("<cross_session_memory>")
    # Must still leave room for retrieved facts (not persona-only).
    assert "Long-Term Memory" in out or "fact-0" in out
    assert "User Profile" in out
