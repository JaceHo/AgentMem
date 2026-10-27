"""KB plane tests — daily-use scenes against the real app + test Redis (db 15).

Progressive disclosure: ingest → index (Tier 0) → digest (Tier 1) → search
(Tier 2) → incremental re-ingest → remove. LLM calls are monkeypatched so the
suite is deterministic and runs without aiserv.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest  # noqa: E402

from core import kb as kb_mod  # noqa: E402


@pytest.fixture(autouse=True)
async def fresh_kb():
    """Isolate state per test: wipe the app's Redis db (the in-process app
    binds REDIS_URL db14, shared with the integration server) and reset the
    KB singleton (its BM25 index lives in RAM across tests)."""
    import redis as sync_redis

    from tests.conftest import TEST_REDIS_DB
    sync_redis.Redis(host="localhost", port=6379, db=TEST_REDIS_DB).flushdb()
    kb_mod._kb = kb_mod.KBStore()
    yield
    kb_mod._kb = None


@pytest.fixture
def kb_home(tmp_path):
    """Ingestion guard only allows paths under ~ — put fixtures in ~/.agentmem/."""
    d = Path(kb_mod.HOME) / ".agentmem" / f"test_kb_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    """Deterministic digests/enrichment/embeddings — no aiserv, no model downloads."""
    async def fake_llm(system, prompt, timeout=45.0):
        if "description" in prompt and "digest" in prompt:
            title = prompt.split('"')[1]
            return json.dumps({"description": f"Digest of {title} for tests",
                               "digest": f"# {title}\nTest digest body with ZQWKEY anchor."})
        return "This chunk belongs to the test fixture document."
    monkeypatch.setattr(kb_mod, "_llm", fake_llm)

    import hashlib
    import numpy as np

    dim = 32

    def fake_vec(text: str):
        h = hashlib.sha1(text.encode()).digest()
        v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        v = np.tile(v, dim // len(v) + 1)[:dim]
        norm = float(np.linalg.norm(v)) or 1.0
        return v / norm

    monkeypatch.setattr(kb_mod, "encode", fake_vec)
    monkeypatch.setattr(kb_mod, "encode_batch", lambda texts: [fake_vec(t) for t in texts])


async def _write_docs(d):
    d.mkdir(parents=True, exist_ok=True)
    (d / "deploy.md").write_text(
        "# Deployment Runbook\n"
        "## Rollout\n"
        "The XYZZY-DEPLOY flag gates the rollout. Deploy window is Sunday 02:00 UTC.\n"
        "## Rollback\n"
        "Rollback by reverting the helm release and flushing the queue.\n"
    )
    (d / "pricing.md").write_text(
        "# Pricing Handbook\n"
        "## Tiers\n"
        "Free tier: 100 calls/day. Pro tier: 50 dollars per month, 10k calls.\n"
        "The KWLPRC keyword anchors this section for retrieval tests.\n"
    )
    return d


@pytest.mark.asyncio
async def test_ingest_index_digest_search_daily_flow(kb_client, kb_home):
    """The core daily loop: ingest → Tier0 → Tier1 → Tier2."""
    d = await _write_docs(kb_home)

    r = await kb_client.post("/kb/ingest", json={"path": str(d), "enrich": True})
    assert r.status_code == 200
    body = r.json()
    assert len(body["ingested"]) == 2 and not body["failed"]
    deploy = next(x for x in body["ingested"] if x["title"].startswith("Deployment"))
    assert deploy["enriched"] > 0

    # Tier 0 — index lines carry id + description
    idx = (await kb_client.get("/kb/index")).json()["docs"]
    assert {x["title"] for x in idx} == {"Deployment Runbook", "Pricing Handbook"}
    assert all(x["description"] and x["chunks"] >= 1 for x in idx)
    deploy_id = next(x["id"] for x in idx if x["title"] == "Deployment Runbook")

    # Tier 1 — digest + section map
    meta = (await kb_client.get(f"/kb/doc/{deploy_id}/digest")).json()
    assert "Test digest body" in meta["digest"]
    assert set(meta["sections"]) >= {"Rollout", "Rollback"}
    assert meta["path"].endswith("deploy.md")

    # Tier 2 — exact-keyword BM25 recall (unique anchor words). NB: with fake
    # hash vectors the vector leg is noise, so assert presence, not rank #1.
    hits = (await kb_client.post("/kb/search", json={"query": "XYZZY-DEPLOY rollout sunday"})).json()["chunks"]
    assert any(h["title"] == "Deployment Runbook" and "XYZZY-DEPLOY" in h["text"] for h in hits)
    hits2 = (await kb_client.post("/kb/search", json={"query": "KWLPRC pricing tiers"})).json()["chunks"]
    assert any(h["title"] == "Pricing Handbook" and "KWLPRC" in h["text"] for h in hits2)


@pytest.mark.asyncio
async def test_incremental_reingest_skip_update_stable_id(kb_client, kb_home):
    """Unchanged → skipped; edited → updated under the SAME doc id."""
    d = await _write_docs(kb_home)
    first = (await kb_client.post("/kb/ingest", json={"path": str(d)})).json()
    ids = {x["title"]: x["doc"] for x in first["ingested"]}

    again = (await kb_client.post("/kb/ingest", json={"path": str(d)})).json()
    assert not again["ingested"] and len(again["skipped"]) == 2

    f = d / "deploy.md"
    f.write_text(f.read_text() + "\n## New section\nPostroll verification via KZNEWSTEP hook.\n")
    updated = (await kb_client.post("/kb/ingest", json={"path": str(d)})).json()
    assert len(updated["ingested"]) == 1
    assert updated["ingested"][0]["doc"] == ids["Deployment Runbook"]  # stable id

    hits = (await kb_client.post("/kb/search", json={"query": "KZNEWSTEP"})).json()["chunks"]
    assert hits and hits[0]["doc"] == ids["Deployment Runbook"]
    stale = (await kb_client.post("/kb/search", json={"query": "Rollback helm release"})).json()["chunks"]
    assert not [h for h in stale if h["section"] == "Rollback" and h["doc"] != ids["Deployment Runbook"]]


@pytest.mark.asyncio
async def test_remove_by_path(kb_client, kb_home):
    d = await _write_docs(kb_home)
    await kb_client.post("/kb/ingest", json={"path": str(d / "pricing.md")})

    r = await kb_client.post("/kb/remove", json={"doc": str(d / "pricing.md")})
    assert r.json()["status"] == "removed"

    idx = (await kb_client.get("/kb/index")).json()["docs"]
    assert not [x for x in idx if x["title"] == "Pricing Handbook"]
    hits = (await kb_client.post("/kb/search", json={"query": "KWLPRC"})).json()["chunks"]
    assert hits == []


@pytest.mark.asyncio
async def test_llm_outage_falls_back_extractive(kb_client, kb_home, monkeypatch):
    """When aiserv is down, ingest still succeeds with extractive digest."""
    async def dead_llm(system, prompt, timeout=45.0):
        raise ConnectionError("aiserv down")
    monkeypatch.setattr(kb_mod, "_llm", dead_llm)

    d = kb_home / "single"
    d.mkdir()
    (d / "note.md").write_text("# Server Notes\nPrimary DB runs on the jetson box, port 5432.\n")
    r = await kb_client.post("/kb/ingest", json={"path": str(d), "enrich": True})
    body = r.json()
    assert len(body["ingested"]) == 1 and body["ingested"][0]["enriched"] == 0

    doc = body["ingested"][0]["doc"]
    meta = (await kb_client.get(f"/kb/doc/{doc}/digest")).json()
    assert "LLM digest unavailable" in meta["digest"]
    hits = (await kb_client.post("/kb/search", json={"query": "jetson 5432"})).json()["chunks"]
    assert any("jetson" in h["text"] for h in hits)


@pytest.mark.asyncio
async def test_path_guard_and_missing_doc(kb_client, tmp_path):
    outside = tmp_path / "x.md"  # pytest tmp is under /private/var/... — outside home
    if str(outside.resolve()).startswith(kb_mod.HOME):
        pytest.skip("tmp is inside home on this machine")
    r = await kb_client.post("/kb/ingest", json={"path": str(outside)})
    assert r.status_code == 400
    assert (await kb_client.get("/kb/doc/kb_doesnotexist/digest")).status_code == 404
