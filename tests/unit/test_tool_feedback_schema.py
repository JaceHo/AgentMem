"""tool-feedback accepts LAN hook aliases and rejects empty names."""

from api.schemas.capability import ToolFeedbackRequest


def test_aliases_name_and_tool():
    a = ToolFeedbackRequest.model_validate({"name": "WebSearch", "ok": True})
    assert a.tool_name == "WebSearch" and a.success is True
    b = ToolFeedbackRequest.model_validate({"tool": "Bash", "succeeded": False, "sessionId": "s1"})
    assert b.tool_name == "Bash" and b.success is False and b.session_id == "s1"


def test_strips_whitespace():
    r = ToolFeedbackRequest.model_validate({"tool_name": "  Edit  ", "success": True})
    assert r.tool_name == "Edit"
