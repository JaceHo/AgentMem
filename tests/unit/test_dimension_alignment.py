"""Width integrity: the model is measured, the index is the source of truth.

Why these exist (measured 2026-10-27): the embedder used to *declare* its width — a
class constant, a model table, or ``EMBEDDING_DIMS`` — and nothing compared that
declaration to the vectors actually returned. aiserv's ``/v1/embeddings`` ignores
the requested model and always proxies Jetson ``nomic-embed-text`` (768), so a
process configured for 1024 declared 1024, wrote 768, and every VADD was rejected
while ``POST /store`` still answered ``200 {"status":"queued"}``. Redis vector sets
fix their width on the first write and cannot be retyped, so this is silent,
permanent data loss — not a degraded-search condition.

The three properties pinned here:
  1. a provider's declared width never outranks a real vector;
  2. no vector leaves the embedder unchecked;
  3. the boot guard covers *every* vectorset, not three hard-coded keys, and fails
     closed instead of force-switching the provider.
"""

import numpy as np
import pytest

from core import dim_guard
from core import embedder as embed_mod


class _LyingProvider:
    """Declares 1024 dims, actually returns 768 — the aiserv shape."""

    name = "lying"

    def __init__(self, declared: int = 1024, actual: int = 768):
        self.dims = declared
        self.dims_verified = False
        self.model = "nomic-embed-text"
        self.actual = actual

    def encode(self, text: str) -> np.ndarray:
        return np.zeros(self.actual, dtype=np.float32)

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.encode(t) for t in texts]


@pytest.fixture
def provider(monkeypatch):
    """Install a provider without touching detection or the network."""
    holder = {"p": None}

    def install(p):
        holder["p"] = p
        monkeypatch.setattr(embed_mod, "_provider", p, raising=False)
        return p

    yield install
    monkeypatch.setattr(embed_mod, "_provider", holder["p"], raising=False)


def test_measurement_overrides_the_declaration(provider):
    p = provider(_LyingProvider(declared=1024, actual=768))
    embed_mod._measure_provider(p)
    assert p.dims == 768, "a declared width must never outrank the vector that came back"
    assert p.dims_verified is True


def test_encode_rejects_a_vector_of_the_wrong_width(provider):
    p = provider(_LyingProvider(declared=1024, actual=768))
    embed_mod._measure_provider(p)
    p.actual = 512                      # a gateway silently switching model mid-flight
    with pytest.raises(embed_mod.DimensionMismatch):
        embed_mod.encode("anything")


def test_batch_encode_rejects_one_bad_vector(provider):
    p = provider(_LyingProvider(declared=1024, actual=768))
    embed_mod._measure_provider(p)
    p.actual = 512
    with pytest.raises(embed_mod.DimensionMismatch):
        embed_mod.encode_batch(["a", "b"])


class _FakeRedis:
    """Enough of the Redis surface for the guard: scan / type / VDIM / VCARD."""

    def __init__(self, sets: dict[str, int], others: tuple[str, ...] = ("mem:proc_tasks",)):
        self.sets = sets
        self.keys = list(sets) + list(others)

    async def scan(self, cursor, match="*", count=500):
        return (0, [k.encode() for k in self.keys])

    async def type(self, key):
        return b"vectorset" if key in self.sets else b"set"

    async def execute_command(self, cmd, key):
        if cmd == "VDIM":
            return self.sets[key]
        if cmd == "VCARD":
            return 7
        raise AssertionError(f"unexpected command {cmd}")


@pytest.fixture
def measured(provider, monkeypatch):
    """A provider whose measured width is width — no model, no network."""
    def install(width: int):
        p = provider(_LyingProvider(declared=width, actual=width))
        embed_mod._measure_provider(p)
        return p
    monkeypatch.setattr(dim_guard, "_cache", {"at": 0.0, "dims": None, "bad": None})
    return install


async def test_guard_covers_sets_the_old_loop_never_saw(measured):
    measured(1024)
    r = _FakeRedis({"mem:facts": 1024, "mem:tools": 768}, others=("mem:kb:chunks",))
    snap = await dim_guard.snapshot(r)
    assert snap["ok"] is False
    assert set(snap["bad"]) == {"mem:tools"}, "a set added after the fact must still be checked"
    assert "mem:facts" in snap["sets"] and "mem:kb:chunks" not in snap["sets"]


async def test_boot_fails_closed_and_warn_is_explicit(measured):
    measured(1024)
    r = _FakeRedis({"mem:facts": 768})
    with pytest.raises(embed_mod.DimensionMismatch) as err:
        await dim_guard.enforce(r)
    assert "reembed_vectorsets" in str(err.value), "the error must carry the repair command"
    snap = await dim_guard.enforce(r, policy="warn")
    assert snap["ok"] is False          # boots, but reports the truth


async def test_write_gate_raises_and_is_cached(measured, monkeypatch):
    measured(1024)
    r = _FakeRedis({"mem:facts": 768})
    with pytest.raises(embed_mod.DimensionMismatch):
        await dim_guard.assert_writeable(r)
    # Second call inside the TTL must answer from cache, without a round trip.
    r.keys = []
    with pytest.raises(embed_mod.DimensionMismatch):
        await dim_guard.assert_writeable(r)


async def test_aligned_index_passes(measured):
    measured(1024)
    snap = await dim_guard.enforce(_FakeRedis({"mem:facts": 1024, "mem:tools": 1024}))
    assert snap["ok"] is True and snap["bad"] == {}
