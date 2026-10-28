"""Track scoping: whose memory a row belongs to, and how a query says so.

EverOS files memories under one of two tracks -- a person's episodes and
profile (``user_id``) or an agent's cases and skills (``agent_id``) -- and
Raven's MemoryBackend contract names the track on every call. A store that
ignores those ids answers every caller from one global corpus, so one user's
facts surface in another's prompt and an agent's procedures in everyone's.

This module is the one place that turns a track into the three shapes the
rest of the service needs:

- :meth:`Scope.attrs` -- the element attributes written alongside each row;
- :meth:`Scope.filter` -- the VSIM ``FILTER`` clause read alongside each query,
  so isolation is applied by the index rather than by discarding results after
  the fact;
- :meth:`Scope.persona_key` -- the synthesized profile is per-track too, or the
  tier would leak across the very boundary the rows respect.

An empty ``Scope`` is the unscoped, pre-existing behaviour: rows written
without ids stay readable by an unscoped caller and invisible to a scoped one.
That asymmetry is deliberate. Scoped reads must never fall back to the global
corpus, because the reason to pass a track is to not see the rest of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PERSONA_KEY = "mem:persona"
PERSONA_PRE = "mem:persona:"

#: Ids are interpolated into a VSIM FILTER expression, so the alphabet is the
#: guard: no quotes, backslashes, spaces or operators can reach the expression.
#: Matches what Raven hands over -- bare backend-native ids such as ``jace``,
#: ``cli:20261028_101500``, ``DeepResearcher``, ``user@host`` -- and refuses
#: anything that would need escaping.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/+-]{0,127}$")


def clean_id(value: object) -> str:
    """A track id safe to store and to interpolate, or ``""`` when unusable."""
    if not isinstance(value, str):
        return ""
    value = value.strip()
    return value if _ID_RE.match(value) else ""


@dataclass(frozen=True)
class Scope:
    """Exactly one track, or neither for the unscoped corpus.

    Construct through :func:`scope_of` so the XOR rule lives in one place:
    a caller that passes both ids has a bug, and answering it from either
    track would file one party's memory under another's.
    """

    user_id: str = ""
    agent_id: str = ""

    @property
    def is_global(self) -> bool:
        return not (self.user_id or self.agent_id)

    def attrs(self) -> dict[str, str]:
        """Attributes to write with a row. Empty when unscoped, so existing
        rows keep their exact byte shape and stay readable as they were."""
        if self.user_id:
            return {"user_id": self.user_id}
        if self.agent_id:
            return {"agent_id": self.agent_id}
        return {}

    def filter(self) -> str:
        """The VSIM FILTER clause selecting this track, or ``""``."""
        if self.user_id:
            return f'.user_id == "{self.user_id}"'
        if self.agent_id:
            return f'.agent_id == "{self.agent_id}"'
        return ""

    def combine(self, expr: str | None) -> str | None:
        """AND this scope onto another filter expression (``None`` when neither)."""
        parts = [p for p in (self.filter(), expr) if p]
        return " && ".join(parts) if parts else None

    def persona_key(self) -> str:
        if self.user_id:
            return f"{PERSONA_PRE}user:{self.user_id}"
        if self.agent_id:
            return f"{PERSONA_PRE}agent:{self.agent_id}"
        return PERSONA_KEY

    def track_suffix(self) -> str:
        """``user:<id>`` / ``agent:<id>`` for keys that need one lane per track."""
        if self.user_id:
            return f"user:{self.user_id}"
        if self.agent_id:
            return f"agent:{self.agent_id}"
        return ""

    def matches(self, attrs: dict | None) -> bool:
        """Whether an already-fetched row belongs to this track.

        For legs that cannot filter at the index -- BM25 hits, crystallized
        digests, anything read by id -- this is the same rule applied after
        the fact. An unscoped caller sees everything.
        """
        if self.is_global:
            return True
        attrs = attrs or {}
        if self.user_id:
            return attrs.get("user_id", "") == self.user_id
        return attrs.get("agent_id", "") == self.agent_id


def scope_of(user_id: object = None, agent_id: object = None) -> Scope:
    """Scope from a caller's two ids, enforcing the XOR.

    Both set is a caller bug (the contract says exactly one track); neither
    is the unscoped corpus. Both cases resolve to :class:`Scope` with no
    track rather than picking a winner, because guessing would file the
    write under the wrong owner.
    """
    user = clean_id(user_id)
    agent = clean_id(agent_id)
    if user and agent:
        return Scope()
    return Scope(user_id=user, agent_id=agent)


class BadScope(ValueError):
    """Ids that cannot name exactly one track."""


def strict_scope(user_id: object = None, agent_id: object = None) -> Scope:
    """:func:`scope_of` for a request boundary, where a bad id is an error.

    :func:`scope_of` is the in-process reading and resolves anything odd to
    the unscoped corpus. At an HTTP edge that would answer a scoped call from
    the global corpus, which is the leak scoping exists to prevent -- so the
    edge refuses instead, while the caller can still act on the message.
    """
    for name, raw in (("user_id", user_id), ("agent_id", agent_id)):
        if isinstance(raw, str) and raw.strip() and not clean_id(raw):
            raise BadScope(f"{name} has characters a track id cannot carry")
    if clean_id(user_id) and clean_id(agent_id):
        raise BadScope("user_id and agent_id are exclusive; pass one track")
    return scope_of(user_id, agent_id)
