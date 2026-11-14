"""Active working memory + intention store."""

from __future__ import annotations

import pytest

from core import focus as focus_mod


class FakeRedis:
    def __init__(self):
        self.h: dict[str, dict] = {}
        self.z: dict[str, dict] = {}

    async def hgetall(self, key):
        return dict(self.h.get(key, {}))

    async def hset(self, key, mapping=None, **kwargs):
        self.h.setdefault(key, {})
        if mapping:
            self.h[key].update({str(k): str(v) for k, v in mapping.items()})
        return True

    async def hget(self, key, field):
        return self.h.get(key, {}).get(field)

    async def expire(self, key, ttl):
        return True

    async def zadd(self, key, mapping):
        self.z.setdefault(key, {})
        self.z[key].update(mapping)
        return True

    async def zrevrange(self, key, start, end):
        items = sorted(self.z.get(key, {}).items(), key=lambda x: -x[1])
        ids = [i for i, _ in items]
        return ids[start:end + 1] if end >= 0 else ids[start:]

    async def zrem(self, key, *members):
        z = self.z.get(key, {})
        for m in members:
            z.pop(m, None)
        return True


@pytest.mark.asyncio
async def test_active_roundtrip():
    r = FakeRedis()
    cur = await focus_mod.set_active(r, "s1", goal="ship audit", notes="P0 nav")
    assert cur["goal"] == "ship audit"
    got = await focus_mod.get_active(r, "s1")
    assert got["goal"] == "ship audit" and got["notes"] == "P0 nav"


@pytest.mark.asyncio
async def test_intention_lifecycle():
    r = FakeRedis()
    # FakeRedis.hset for intentions uses field API differently — patch via direct
    class R2(FakeRedis):
        async def hset(self, key, *args, **kwargs):
            if args and len(args) >= 2:
                field, value = args[0], args[1]
                self.h.setdefault(key, {})[field] = value
                return 1
            return await super().hset(key, **kwargs)

    r = R2()
    item = await focus_mod.add_intention(r, text="remind about graph Go", cue="open console")
    assert item["status"] == "pending"
    listed = await focus_mod.list_intentions(r, status="pending")
    assert any(x["id"] == item["id"] for x in listed)
    done = await focus_mod.update_intention(r, item["id"], status="done")
    assert done["status"] == "done"
    listed2 = await focus_mod.list_intentions(r, status="pending")
    assert not any(x["id"] == item["id"] for x in listed2)
