"""Guards against the silent no-ops found on 2026-10-22.

Four regressions shared one shape: the code ran, returned HTTP 200, and did
nothing.

  1. ``force_str`` stringified Redis ``bytes`` (``decode_responses=False``), so
     heat bumps were keyed on ``"b'01M3AB…'"`` — a name no element has.
     1,362 recalls, ``access_count=0`` on 1,149/1,149 facts, ``[evolution]``
     never logged once.
  2. bump failures were swallowed (``except: pass`` / ``log.debug``), so the
     no-op was indistinguishable from success.
  3. ``estimate_tokens`` was ``len(text.split())`` — ~1 token for a whole
     Chinese paragraph — so ``token_budget`` could not cap anything.
  4. ``persona.name`` was a "longest entry ever seen" accumulator with no length
     or PII rules, so a national-ID-card file path was injected into every
     agent's context.

Each test here asserts the *failure signature*, not just the happy path: if
anyone reverts the fix, one of these goes red.
"""

import logging

import numpy as np
import pytest
import redis.asyncio as aioredis

from core import heat
from core import persona as persona_mod
from core import store as mem_store
from core.utils import decode_bytes, force_str
from utils.text_processing import estimate_tokens, truncate_to_tokens


@pytest.fixture
async def r():
    # decode_responses=False mirrors production (core/store.py:_get_pool) — the
    # bug only reproduces when replies come back as bytes.
    client = aioredis.Redis(host="localhost", port=6379, db=15, decode_responses=False)
    await client.flushdb()
    await mem_store.ensure_indexes(client)
    yield client
    await client.flushdb()
    await client.aclose()


@pytest.fixture
def bytes_r():
    """A client whose replies are bytes, for the pure-decoding tests."""
    return aioredis.Redis(host="localhost", port=6379, db=15, decode_responses=False)


def _vec(i: int, dims: int = 384) -> np.ndarray:
    v = np.zeros(dims, dtype=np.float32)
    v[i] = 1.0
    return v


async def _add_fact(r, uid: str, i: int, content: str = "a fact about testing") -> None:
    attr = {
        "content": content, "category": "work", "access_count": 0,
        "last_accessed": 0, "ts": 1790000000000, "importance": 0.8,
        "superseded_by": "", "_uid": uid,
    }
    import json
    await r.execute_command("VADD", mem_store.FACT_KEY, "FP32",
                            _vec(i).astype(np.float32).tobytes(), uid,
                            "SETATTR", json.dumps(attr))


# ── 1. bytes element names must decode ────────────────────────────────────────


class TestElementNameDecoding:
    def test_force_str_never_returns_a_bytes_repr(self):
        # The exact regression: str(b'01M3AB') == "b'01M3AB'".
        assert force_str(b"01M3AB") == "01M3AB"
        assert "b'" not in force_str(b"01M3JY19E9GNNA2YWQTZFJF4CX")
        assert force_str(b"01M3AB") == decode_bytes(b"01M3AB")

    def test_force_str_passes_through_and_stays_generic(self):
        assert force_str("plain") == "plain"
        assert force_str(None) == ""
        assert force_str(7) == "7"          # ints still stringify
        assert force_str(b"\xff") != ""     # invalid utf-8 must not raise

    async def test_knn_search_element_is_usable_as_a_vgetattr_key(self, r):
        """A hit's ``_element`` must be a real key, not a repr of one."""
        await _add_fact(r, "01TESTELEM000000000000AB", 3)
        hits = await mem_store.knn_search(r, mem_store.FACT_KEY, _vec(3), k=3)
        assert hits, "knn_search returned nothing"
        elem = hits[0]["_element"]
        assert isinstance(elem, str) and not elem.startswith("b'")
        raw = await r.execute_command("VGETATTR", mem_store.FACT_KEY, elem)
        assert raw, f"VGETATTR found nothing for element {elem!r} — the " \
                    "heat bump would silently no-op"

    async def test_proc_task_key_survives_the_round_trip(self, r):
        """mem:proc_tasks stores an element id; reading it back must be usable."""
        await _add_fact(r, "01TESTTASK000000000000CD", 4)
        task = "how to verify the memory loop end to end"
        await r.hset(mem_store.PROC_TASK_KEY, task, "01TESTTASK000000000000CD")
        # Same code path the store service uses to find an existing procedure.
        mapped = await r.hget(mem_store.PROC_TASK_KEY, task)
        assert force_str(mapped) == "01TESTTASK000000000000CD"


# ── 2. a no-op bump must be loud and counted ─────────────────────────────────


class TestBumpsAreLoud:
    async def test_bump_moves_access_count(self, r):
        await _add_fact(r, "01TESTBUMP000000000000EF", 5)
        await r.script_load(heat.BUMP_HEAT_LUA_SRC)
        assert await heat.atomic_bump(r, mem_store.FACT_KEY,
                                      "01TESTBUMP000000000000EF") is True
        from core.utils import decode_json
        attrs = decode_json(await r.execute_command(
            "VGETATTR", mem_store.FACT_KEY, "01TESTBUMP000000000000EF"))
        assert attrs["access_count"] == 1

    async def test_a_second_client_cannot_reuse_the_first_clients_script(self, r):
        """Lua Script handles are bound to their client, so they must never be cached globally.

        Measured 2026-10-24: a retention benchmark registered the bump script through a *sync*
        client first; the module global then handed that object to every async caller, whose
        ``await`` of an int died as "heat bump ERROR … object int can't be used in 'await'" —
        the whole suite's bumps went to False while /stats still counted attempts.
        """
        import redis as sync_redis

        await _add_fact(r, "01TESTCLIENT00000000000B1", 3)
        await r.script_load(heat.BUMP_HEAT_LUA_SRC)
        elem = "01TESTCLIENT00000000000B1"

        other = sync_redis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
        try:
            # The sync Script call runs EVALSHA server-side and only the returned int fails to
            # await, so this misuse reports False while still bumping. What must not happen is the
            # handle leaking onto the next client.
            assert await heat.atomic_bump(other, mem_store.FACT_KEY, elem) is False
        finally:
            other.close()

        assert await heat.atomic_bump(r, mem_store.FACT_KEY, elem) is True, \
            "a foreign client must not disable heat bumps"
        from core.utils import decode_json
        attrs = decode_json(await r.execute_command("VGETATTR", mem_store.FACT_KEY, elem))
        assert attrs["access_count"] >= 1, attrs

    async def test_bump_on_missing_element_warns_and_returns_false(self, r, caplog):
        with caplog.at_level(logging.WARNING, logger="mem"):
            ok = await heat.atomic_bump(r, mem_store.FACT_KEY, "b'01NOPE0000'")
        assert ok is False
        assert any("heat bump" in rec.message or "heat bump" in rec.getMessage()
                   for rec in caplog.records), \
            "a no-op bump must log at WARNING, never pass silently"

    async def test_many_bumps_report_real_count_not_the_request_size(self, r):
        """The old code returned ``len(elements)`` even when nothing moved."""
        await _add_fact(r, "01TESTMANY000000000000AA", 6)
        await r.script_load(heat.BUMP_HEAT_LUA_SRC)
        good = "01TESTMANY000000000000AA"
        bad = "b'01GHOST00000000000000BB'"
        bumped = await heat.atomic_bump_many(r, mem_store.FACT_KEY, [good, bad, bad])
        assert bumped == 1, "atomic_bump_many must count successes, not requests"

    async def test_stats_snapshot_exposes_the_counters_for_slash_stats(self, r):
        snap = heat.stats_snapshot()
        for key in ("attempts", "ok", "missing", "errors"):
            assert key in snap, f"/stats heat block lost {key!r}"
        before = dict(snap)
        await _add_fact(r, "01TESTSTAT000000000000CC", 7)
        await r.script_load(heat.BUMP_HEAT_LUA_SRC)
        await heat.atomic_bump(r, mem_store.FACT_KEY, "01TESTSTAT000000000000CC")
        await heat.atomic_bump(r, mem_store.FACT_KEY, "01DOESNOTEXIST00000000")
        after = heat.stats_snapshot()
        assert after["ok"] == before["ok"] + 1
        assert after["missing"] == before["missing"] + 1

    async def test_stats_route_includes_heat(self, agentmem_client):
        resp = await agentmem_client.get("/stats")
        assert resp.status_code == 200
        assert "heat" in resp.json(), "/stats must expose the bump counters"


# ── 3. CJK-aware token budget ────────────────────────────────────────────────


class TestTokenBudget:
    CJK = "记住用户的偏好是中文回答，并且要简短直接不要废话连篇。" * 20

    def test_chinese_is_not_one_token(self):
        # len(text.split()) == 1 for this string: the bug.
        assert estimate_tokens(self.CJK) > 100
        assert estimate_tokens(self.CJK) >= len(self.CJK) * 0.5

    def test_latin_estimate_is_in_the_right_ballpark(self):
        words = "the quick brown fox jumps over the lazy dog " * 10
        est = estimate_tokens(words)
        assert 50 < est < 400
        assert estimate_tokens("") == 0

    def test_truncate_respects_the_budget(self):
        out = truncate_to_tokens(self.CJK, 60)
        assert estimate_tokens(out) <= 60
        assert out and len(out) < len(self.CJK)

    def test_mixed_script_budget(self):
        mixed = "中文内容 mixed with English words 再来一段 more text here " * 12
        assert estimate_tokens(mixed) > estimate_tokens("x")
        assert estimate_tokens(truncate_to_tokens(mixed, 40)) <= 40

    async def test_recall_budget_is_honoured_end_to_end(self, agentmem_client):
        """POST /recall with token_budget=200 must not return 200 tokens of text."""
        probe = "memory-loop-acceptance-budget-probe-4f9a1c"
        await agentmem_client.post("/store", json={
            "sessionId": "guard-test", "content": f"{probe}: 这是一段用于验证 token 预算的中文长文本。" * 8,
        })
        resp = await agentmem_client.post("/recall", json={
            "query": probe, "sessionId": "guard-test", "token_budget": 200,
        })
        assert resp.status_code == 200
        ctx = (resp.json() or {}).get("prependContext") or ""
        assert estimate_tokens(ctx) <= 320, \
            f"token_budget=200 returned ~{estimate_tokens(ctx)} tokens"


# ── 4. persona hygiene: length caps + PII pointers ────────────────────────────


class TestPersonaHygiene:
    def test_path_and_id_document_are_rejected(self):
        for bad in (
            "身份证正反面存放在 ~/Documents/profile/resume/ 目录下",
            "Both sides of the national ID card are in /Users/jace/private/docs",
            "用户的身份证号是 110101199001011234",
            "护照扫描件在 ~/tmp/passport.jpg 里",
        ):
            assert persona_mod._should_reject(bad), f"PII pointer accepted: {bad!r}"

    def test_normal_profile_facts_still_pass(self):
        for ok in (
            "用户偏好中文回复，先给结论再给理由",
            "The user wants reports delivered before 09:30 each trading day",
            "用户常驻北京昌平，主做量化与安卓跨平台开发",
        ):
            assert not persona_mod._should_reject(ok), f"legit fact rejected: {ok!r}"

    def test_entries_are_capped_and_field_is_byte_bounded(self):
        long_entry = "身" * 400
        assert len(persona_mod._clamp_entry(long_entry)) <= persona_mod._MAX_ENTRY_LEN
        field = persona_mod._compose_field(["知" * 150] * 5)
        assert len(field.encode("utf-8")) <= persona_mod._MAX_FIELD_BYTES
        assert not field.endswith(";")            # never cut mid-entry

    def test_recency_wins_not_the_longest_ever_seen(self):
        """The old accumulator sorted by len desc, so one giant stale claim won
        the name field forever. Newest must come first now."""
        old_giant = "旧的很长的身份描述" + "。" * 300
        new_short = "Jace Ho，量化与 Android 工程师，坐标北京昌平"
        kept = persona_mod._best_entries([new_short, old_giant])
        assert kept[0] == persona_mod._clamp_entry(new_short)

    async def test_stored_pii_is_scrubbed_on_sanitize(self, r):
        await r.hset(persona_mod.PERSONA_KEY, "location",
                     "Both sides of the national ID card are stored in "
                     "~/Documents/profile/resume/ directory; 用户喜欢用中文写周报，先结论后细节")
        removed = await persona_mod.sanitize_stored(r)
        left = decode_bytes(await r.hget(persona_mod.PERSONA_KEY, "location"))
        assert "ID card" not in left and "~/Documents" not in left, left
        assert "中文" in left                       # the clean entry survives
        assert "location" in removed

    async def test_update_refuses_pii_and_logs_it(self, r, caplog):
        with caplog.at_level(logging.WARNING, logger="mem"):
            await persona_mod.update(
                r, "location",
                "身份证正反面在 ~/Documents/profile/resume/ 目录里存档备份")
        assert decode_bytes(await r.hget(persona_mod.PERSONA_KEY, "location")) \
            in (None, "")
        assert any("persona" in rec.getMessage() for rec in caplog.records), \
            "a rejected persona fact must be visible in the log"

    async def test_update_many_refuses_pii(self, r):
        await persona_mod.update_many(r, [
            ("location", "身份证正反面在 ~/Documents/profile/resume/ 目录里"),
            ("preference", "用户偏好中文回复，先给结论再给理由"),
        ])
        assert not await r.hget(persona_mod.PERSONA_KEY, "location")
        assert "中文" in decode_bytes(
            await r.hget(persona_mod.PERSONA_KEY, "preferences"))
