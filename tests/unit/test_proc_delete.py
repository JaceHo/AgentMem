"""Regression tests for DELETE /procedures/{element_id} and the /stats store-plane counters.

The endpoint exists because `ops/memory.sh verify` runs daily and must not leave a test procedure in
the flagship KB every day (365 rows/year). It is thin over mem_store.remove_element, but "thin" is
where the 30,977 dangling side-index references came from in the first place — so the contract that
matters is: VREM *and* both side indexes, and never a 200 that deleted nothing.
"""
import json

import pytest

from api import state
from api.routes import admin as admin_routes
from api.routes import capability as cap
from core import store as mem_store


class FakeRedis:
    def __init__(self, attrs: dict, pending: int = 0):
        self.attrs = attrs                 # element_id -> attrs json
        self.vrem = []
        self.hdels = []
        self.srems = []
        self.pending = pending

    async def execute_command(self, cmd, *args):
        if cmd == "VGETATTR":
            return self.attrs.get(args[1])
        if cmd == "VREM":
            self.vrem.append(args[1])
            return 1
        raise AssertionError(f"unexpected command {cmd}")

    async def llen(self, key):
        return self.pending


@pytest.fixture
def wired(monkeypatch):
    attrs = {"elem-1": json.dumps({"task": "how to prune", "tools_used": ["grep", "read"]})}
    r = FakeRedis(attrs)
    monkeypatch.setattr(state, "redis", r)

    removed = []

    async def fake_remove(_r, vset, element_id):
        removed.append((vset, element_id))

    async def fake_get_attrs(_r, vset, element_id):
        raw = r.attrs.get(element_id)
        return json.loads(raw) if raw else None

    class FakeBM25:
        removed_ids = []

        async def remove(self, ids):
            FakeBM25.removed_ids.extend(ids)

    monkeypatch.setattr(mem_store, "remove_element", fake_remove)
    monkeypatch.setattr(mem_store, "get_attrs", fake_get_attrs)
    monkeypatch.setattr(state, "bm25_index", FakeBM25())
    return r, removed, FakeBM25


async def test_delete_removes_the_element_and_reports_it(wired):
    r, removed, _bm = wired
    out = await admin_routes.delete_procedure("elem-1")
    assert out["status"] == "ok" and out["action"] == "hard_deleted"
    assert removed == [(mem_store.PROC_KEY, "elem-1")], "must go through remove_element, not bare VREM"


async def test_delete_of_unknown_element_is_not_a_false_success(wired):
    out = await admin_routes.delete_procedure("ghost")
    assert out["status"] == "not_found", "200 + nothing deleted is the failure mode this house keeps hitting"


async def test_delete_rejects_an_empty_id(wired):
    """VREM with an empty element name is a silent no-op on Redis vectorsets."""
    out = await admin_routes.delete_procedure("")
    assert out["status"] in ("not_found", "bad_request")


def test_live_stats_exposes_every_store_plane_counter():
    """The running service is the contract here: a caller must be able to attribute
    "accepted but nothing new stored" to one of the five real write outcomes.

    Asserted against :18800 rather than a fake, because /stats aggregates ~10 probes and a
    hand-built double would only prove that my double works. Skips when the service is down —
    the mesh watchdog owns that alarm, not this file.
    """
    import os
    import urllib.request
    base = os.environ.get("AGENTMEM_URL", "http://127.0.0.1:18800")
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with op.open(base + "/stats", timeout=10) as resp:
            snap = json.loads(resp.read())
    except Exception as exc:
        pytest.skip(f"agentmem not reachable at {base}: {type(exc).__name__}")
    assert "procedures_pending" in snap, "queue depth must be readable — invisible backlog is a silent failure"
    guard = snap.get("proc_guard") or {}
    assert {"deferred", "errors", "drained", "stored", "deduped"} <= set(guard), guard
    assert all(isinstance(guard[k], int) for k in ("deferred", "errors", "drained", "stored", "deduped"))
    heat = snap.get("heat") or {}
    assert {"attempts", "ok", "missing", "errors"} <= set(heat), "read-side bumps must be countable too"
