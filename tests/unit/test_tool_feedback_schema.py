"""tool-feedback accepts LAN hook aliases; empty names soft-skip at the route."""

from api.schemas.capability import ToolFeedbackRequest
from api.schemas.consolidation import CompactRequest


def test_aliases_name_and_tool():
    a = ToolFeedbackRequest.model_validate({"name": "WebSearch", "ok": True})
    assert a.tool_name == "WebSearch" and a.success is True
    b = ToolFeedbackRequest.model_validate({"tool": "Bash", "succeeded": False, "sessionId": "s1"})
    assert b.tool_name == "Bash" and b.success is False and b.session_id == "s1"


def test_strips_whitespace():
    r = ToolFeedbackRequest.model_validate({"tool_name": "  Edit  ", "success": True})
    assert r.tool_name == "Edit"


def test_compact_coerces_null_and_sessionId_alias():
    assert CompactRequest.model_validate({"session_id": None}).session_id == ""
    assert CompactRequest.model_validate({"sessionId": "abc"}).session_id == "abc"
