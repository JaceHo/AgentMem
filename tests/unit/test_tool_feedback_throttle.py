"""Success tool-feedback is debounced; failures always persist."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import capability as cap_routes


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(cap_routes.router)
    fake_redis = MagicMock()
    monkeypatch.setattr(cap_routes.state, "redis", fake_redis)
    cap_routes._TOOL_FB_LAST_OK.clear()
    cap_routes._TOOL_FB_PENDING_OK.clear()
    return TestClient(app)


def test_success_debounced(client, monkeypatch):
    calls = {"n": 0}

    async def fake_atomic(r, key, success):
        calls["n"] += 1
        return {"success_count": calls["n"], "fail_count": 0, "capability_summary": ""}

    monkeypatch.setattr(cap_routes.cap_mod, "atomic_tool_feedback", fake_atomic)
    monkeypatch.setattr(cap_routes, "_TOOL_FB_SUCCESS_INTERVAL_S", 60.0)

    r1 = client.post("/tool-feedback", json={"tool_name": "Bash", "success": True})
    assert r1.status_code == 200 and r1.json().get("deferred") is not True
    r2 = client.post("/tool-feedback", json={"tool_name": "Bash", "success": True})
    assert r2.status_code == 200 and r2.json().get("deferred") is True
    assert calls["n"] == 1


def test_failure_always_writes(client, monkeypatch):
    calls = {"n": 0}

    async def fake_atomic(r, key, success):
        calls["n"] += 1
        return {"success_count": 0, "fail_count": calls["n"], "capability_summary": ""}

    monkeypatch.setattr(cap_routes.cap_mod, "atomic_tool_feedback", fake_atomic)
    monkeypatch.setattr(cap_routes, "_TOOL_FB_SUCCESS_INTERVAL_S", 60.0)

    for _ in range(3):
        r = client.post("/tool-feedback", json={"tool_name": "Bash", "success": False})
        assert r.status_code == 200 and not r.json().get("deferred")
    assert calls["n"] == 3
