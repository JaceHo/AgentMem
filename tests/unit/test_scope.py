"""Track scoping: user vs agent, end to end through the vectorset.

EverOS files memories under exactly one of two tracks and Raven's
MemoryBackend contract names the track on every call. Before this, the store
answered every caller from one global corpus -- so the same facts surfaced for
every user and every agent, and ``user_id`` / ``agent_id`` were dropped on the
floor. These tests pin the three properties that make the merge real:

* the ids are written with the row (and not written at all when absent, so the
  pre-existing unscoped corpus keeps its exact shape);
* the index filters by them, so isolation is applied by VSIM rather than by
  discarding results after the fact;
* the synthesized persona is per-track, or the one tier that summarizes a user
  would be the one tier that leaks across users.
"""

import numpy as np
import pytest
import redis.asyncio as aioredis

from core import persona as persona_mod
from core import store as mem_store
from core.scope import BadScope, Scope, clean_id, scope_of, strict_scope


@pytest.fixture
async def r():
    client = aioredis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
    await client.flushdb()
    await mem_store.ensure_indexes(client)
    yield client
    await client.flushdb()
    await client.aclose()


def _emb(seed: float = 1.0) -> np.ndarray:
    v = np.zeros(384, dtype=np.float32)
    v[0] = seed
    return v


# ─── the id vocabulary ───────────────────────────────────────────────────────


class TestIds:
    def test_backend_native_ids_pass(self):
        # The shapes Raven and ordinary callers actually hand over.
        for ok in ("jace", "cli:20261028_101500", "DeepResearcher", "user@host",
                   "a.b_c-d", "team/ops", "x"):
            assert clean_id(ok) == ok

    def test_expression_metacharacters_are_refused(self):
        # The id is interpolated into a VSIM FILTER expression, so a quote or a
        # backslash would be an injection, not merely a bad id.
        for bad in ('a"b', "a\\b", "a b", "a==b", 'x"||.content!=""', "", "   ", None, 5):
            assert clean_id(bad) == ""

    def test_leading_operator_is_not_an_id(self):
        assert clean_id("==") == ""
        assert clean_id(".hidden") == ""

    def test_xor_resolves_to_one_track_or_none(self):
        assert scope_of("alice", None) == Scope(user_id="alice")
        assert scope_of(None, "Coder") == Scope(agent_id="Coder")
        assert scope_of(None, None) == Scope()
        # Both is a caller bug: guessing would file one party's memory under
        # the other, so it resolves to no track at all.
        assert scope_of("alice", "Coder") == Scope()

    def test_strict_scope_refuses_what_the_edge_cannot_honour(self):
        assert strict_scope("alice", "") == Scope(user_id="alice")
        with pytest.raises(BadScope):
            strict_scope("alice", "Coder")
        with pytest.raises(BadScope):
            strict_scope('a"b', "")


# ─── the shapes scope produces ───────────────────────────────────────────────


class TestShapes:
    def test_attributes_only_exist_when_a_track_does(self):
        assert Scope().attrs() == {}
        assert Scope(user_id="alice").attrs() == {"user_id": "alice"}
        assert Scope(agent_id="Coder").attrs() == {"agent_id": "Coder"}

    def test_filter_is_a_vsim_clause(self):
        assert Scope().filter() == ""
        assert Scope(user_id="alice").filter() == '.user_id == "alice"'
        assert Scope(agent_id="Coder").filter() == '.agent_id == "Coder"'

    def test_combine_ands_onto_an_existing_expression(self):
        assert Scope().combine(None) is None
        assert Scope().combine('.language == "en"') == '.language == "en"'
        assert Scope(user_id="alice").combine('.language == "en"') == (
            '.user_id == "alice" && .language == "en"'
        )

    def test_persona_is_keyed_per_track(self):
        assert Scope().persona_key() == "mem:persona"
        assert Scope(user_id="alice").persona_key() == "mem:persona:user:alice"
        assert Scope(agent_id="Coder").persona_key() == "mem:persona:agent:Coder"

    def test_matches_is_the_post_fetch_rule(self):
        alice = Scope(user_id="alice")
        assert alice.matches({"user_id": "alice"})
        assert not alice.matches({"user_id": "bob"})
        # A row written before tracks existed belongs to the global corpus and
        # never answers a scoped call.
        assert not alice.matches({})
        assert Scope().matches({})


# ─── the rows ────────────────────────────────────────────────────────────────


class TestRows:
    async def test_unscoped_write_keeps_the_old_attribute_shape(self):
        # Byte-shape stability matters: the live corpus was written without
        # these keys and must stay readable as the global one.
        attrs = mem_store._build_fact_attrs(content="c", category="fact")
        assert "user_id" not in attrs and "agent_id" not in attrs

    async def test_scoped_fact_carries_its_track(self, r):
        uid = await mem_store.save_fact(
            r, "alice prefers tabs", "preference", 0.9, _emb(),
            user_id="alice",
        )
        raw = await r.execute_command("VGETATTR", mem_store.FACT_KEY, uid)
        attrs = mem_store.decode_json(raw)
        assert attrs["user_id"] == "alice"
        assert "agent_id" not in attrs

    async def test_scoped_episode_carries_its_track(self, r):
        uid = await mem_store.save_episode(
            r, "s1", "a conversation", _emb(), agent_id="Coder",
        )
        attrs = mem_store.decode_json(
            await r.execute_command("VGETATTR", mem_store.EPISODE_KEY, uid)
        )
        assert attrs["agent_id"] == "Coder"

    async def test_the_index_isolates_tracks(self, r):
        await mem_store.save_fact(r, "alice likes tabs", "preference", 0.9, _emb(), user_id="alice")
        await mem_store.save_fact(r, "bob likes spaces", "preference", 0.9, _emb(), user_id="bob")
        await mem_store.save_fact(r, "legacy fact", "fact", 0.9, _emb())

        alice = await mem_store.knn_search(
            r, mem_store.FACT_KEY, _emb(), 10, filter_expr=Scope(user_id="alice").filter(),
        )
        assert [h["content"] for h in alice] == ["alice likes tabs"]

        # Unscoped still reads the whole corpus, including the legacy row.
        everything = await mem_store.knn_search(r, mem_store.FACT_KEY, _emb(), 10)
        assert {h["content"] for h in everything} == {
            "alice likes tabs", "bob likes spaces", "legacy fact",
        }

    async def test_batch_write_carries_the_same_track(self, r):
        await mem_store.save_facts_batch(r, [
            {"embedding": _emb(), "content": "one", "category": "fact", "user_id": "alice"},
            {"embedding": _emb(), "content": "two", "category": "fact", "user_id": "alice"},
        ])
        hits = await mem_store.knn_search(
            r, mem_store.FACT_KEY, _emb(), 10, filter_expr=Scope(user_id="alice").filter(),
        )
        assert {h["content"] for h in hits} == {"one", "two"}


# ─── the persona ─────────────────────────────────────────────────────────────


class TestPersona:
    async def test_scoped_persona_never_touches_the_global_hash(self, r):
        alice = Scope(user_id="alice").persona_key()
        await persona_mod.update(r, "identity", "alice is a data scientist", key=alice)

        assert "data scientist" in await persona_mod.get_context(r, key=alice)
        assert await persona_mod.get_context(r) == ""
        assert await r.hlen(persona_mod.PERSONA_KEY) == 0

    async def test_two_tracks_keep_two_profiles(self, r):
        await persona_mod.update_many(
            r, [("identity", "alice builds dashboards")],
            key=Scope(user_id="alice").persona_key(),
        )
        await persona_mod.update_many(
            r, [("identity", "bob ships kernels")],
            key=Scope(user_id="bob").persona_key(),
        )
        assert "dashboards" in await persona_mod.get_context(
            r, key=Scope(user_id="alice").persona_key())
        assert "kernels" in await persona_mod.get_context(
            r, key=Scope(user_id="bob").persona_key())
