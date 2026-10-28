"""Procedures written without a track must be visible to a scoped reader.

`/store-procedure` is how the cowork protocol records a how-to (`ops/memory.sh store --kind
procedure`), and `Scope.attrs()` deliberately writes no owner when the caller names no track.
The read paths used to answer that with a VSIM `.user_id == "…"` FILTER, which measured
2026-10-28 returned **0 of 51** procedures to `user_id=jace` and 0 to `agent_id=raven` — a
tier that is written by one path and unreachable from the other is not a memory.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.scope import Scope  # noqa: E402


def test_an_unowned_procedure_is_shared_memory():
    assert Scope().sees(None) is True
    assert Scope(user_id="jace").sees(None) is True
    assert Scope(agent_id="raven").sees("") is True


def test_a_track_owned_procedure_stays_on_its_track():
    assert Scope(user_id="jace").sees("jace") is True
    assert Scope(user_id="jace").sees("raven") is False
    assert Scope(agent_id="raven").sees("raven") is True
    assert Scope(agent_id="raven").sees("jace") is False


def test_a_global_reader_sees_everything_it_is_given():
    assert Scope().sees("jace") is True
    assert Scope().sees("someone-else") is True
