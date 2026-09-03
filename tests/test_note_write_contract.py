"""What the note write path accepts, refuses, and says about it.

``write_clip_notes`` had no test at this level at all. The note MODULE is
thoroughly covered (``tests/test_notes.py``), and the tool that uses it was
covered nowhere, which is how the two defects below survived: both live in the
few lines between the validator and the socket.

Measured 2026-09-02 against Live 12.4.5.

1. ``read_clip_notes`` hands back ``note_id`` on every note, and feeding that same
   list to ``write_clip_notes`` was refused with
   ``unknown key(s) ['note_id']``. Read a clip, change one velocity, write it
   back. That is the most ordinary edit in a DAW, and it was blocked. Meanwhile
   ``validate_note_dicts`` had already declared ``note_id`` tolerable and promised
   it "will be dropped". Nothing dropped it.

2. ``validate_note_dicts`` is the only place ``missing_velocity`` and
   ``empty_note_list`` are raised, and its report was discarded: only the second
   validation's warnings reached the caller. So an empty list with
   ``mode="replace"`` CLEARED THE CLIP and the warning that says so reached nobody.

These tests need no Ableton. The client is faked. The shapes it answers with are
the shapes Live really sent.
"""

from __future__ import annotations

from typing import Any

import pytest

from ableton_maestro import server
from ableton_maestro.music.notes import TOLERATED_NOTE_KEYS

#: One note exactly as Live handed it back on 2026-09-02, keys and all.
AS_LIVE_RETURNS_IT = {
    "pitch": 36,
    "start_time": 0.5,
    "duration": 0.125,
    "velocity": 96.0,
    "mute": False,
    "probability": 1.0,
    "velocity_deviation": 0.0,
    "release_velocity": 64.0,
    "note_id": 7,
}


class _FakeLive:
    """Answers ``notes_set`` and echoes it back for ``notes_get``."""

    def __init__(self) -> None:
        self.sent: dict[str, Any] = {}

    def send(self, handler: str, params: dict[str, Any]) -> dict[str, Any]:
        if handler == "notes_set":
            self.sent = params
            count = len(params["notes"])
            return {"before_count": 0, "after_count": count, "written": count}
        if handler == "notes_get":
            back = self.sent.get("notes") or []
            return {"notes": back, "count": len(back)}
        raise AssertionError(f"unexpected handler {handler!r}")


@pytest.fixture
def live(monkeypatch: pytest.MonkeyPatch) -> _FakeLive:
    """A faked Live that records what the tool would put on the socket."""
    fake = _FakeLive()
    monkeypatch.setattr(server, "_client_instance", lambda: fake)
    monkeypatch.setattr(server, "_run", lambda spec_id, **kw: {"ok": True, "value": 16.0})
    return fake


def _codes(result: dict[str, Any]) -> list[str]:
    return [w["code"] for w in result.get("warnings") or []]


def test_a_note_read_from_live_can_be_written_back(live: _FakeLive) -> None:
    """The round trip that was blocked: read, change one field, write.

    ``note_id`` is Live's handle for an existing note. It is identity, not
    content, and it is dropped rather than refused, which is what
    ``validate_note_dicts`` had promised all along.
    """
    edited = dict(AS_LIVE_RETURNS_IT, velocity=100.0)
    result = server.write_clip_notes(track=0, slot=0, notes=[edited])

    assert result["ok"] is True, result
    assert result["sent"] == 1
    assert result["input_keys_ignored"] == ["note_id"]
    assert "note_id" not in live.sent["notes"][0], "identity must not go back to Live"
    assert live.sent["notes"][0]["velocity"] == 100.0


def test_the_ignored_key_notice_is_stated_once_and_not_once_per_note(
    live: _FakeLive,
) -> None:
    """A forty-note round trip produced forty identical ``ignored_key`` warnings.

    The per-note warning is dropped from the result and ``input_keys_ignored``
    names the keys once for the whole call.
    """
    notes = [dict(AS_LIVE_RETURNS_IT, start_time=float(beat), note_id=beat)
             for beat in range(8)]
    result = server.write_clip_notes(track=0, slot=0, notes=notes)

    assert result["ok"] is True
    assert "ignored_key" not in _codes(result)
    assert result["input_keys_ignored"] == ["note_id"]


def test_a_note_without_velocity_says_so_in_the_result(live: _FakeLive) -> None:
    """``missing_velocity`` was computed by the first validation and discarded.

    The whole point of the warning is that 100 is a real default but an inherited
    one, so it has to be visible.
    """
    result = server.write_clip_notes(
        track=0, slot=0, notes=[{"pitch": 36, "start_time": 0.0, "duration": 0.25}]
    )
    assert result["ok"] is True
    assert "missing_velocity" in _codes(result)


def test_clearing_a_clip_warns_that_it_clears_the_clip(live: _FakeLive) -> None:
    """An empty list under ``replace`` empties the clip. That must be said out loud.

    This is the reason the discarded warnings mattered: the destructive case was
    the silent one.

    Once, and not twice. Both validation stages raise ``empty_note_list`` for an
    empty list, so merging the two reports naively said the same sentence twice.
    """
    result = server.write_clip_notes(track=0, slot=0, notes=[], mode="replace")
    codes = _codes(result)
    assert "empty_note_list" in codes, "clearing a clip has to announce itself"
    assert codes.count("empty_note_list") == 1, f"said more than once: {codes}"


def test_two_notes_with_the_same_problem_are_both_reported(live: _FakeLive) -> None:
    """De-duplicating warnings must not collapse different notes into one.

    Identical issues are merged by code AND by the note indices they point at, so
    a per-note warning still arrives once per note.
    """
    result = server.write_clip_notes(track=0, slot=0, notes=[
        {"pitch": 36, "start_time": 0.0, "duration": 0.25},
        {"pitch": 38, "start_time": 1.0, "duration": 0.25},
    ])
    assert result["ok"] is True
    assert _codes(result).count("missing_velocity") == 2, (
        "both notes are missing a velocity and both have to be named"
    )
    indices = [w["indices"] for w in result["warnings"]]
    assert indices == [[0], [1]], indices


def test_a_genuinely_unknown_key_is_still_refused(live: _FakeLive) -> None:
    """Tolerating Live's own keys must not turn into tolerating anything."""
    result = server.write_clip_notes(
        track=0, slot=0,
        notes=[{"pitch": 60, "start_time": 0.0, "duration": 1.0, "accent": 3}],
    )
    assert result["ok"] is False
    assert result["blocked"] is True
    codes = [issue["code"] for issue in result["validation"]["errors"]]
    assert "unknown_key" in codes
    assert not live.sent, "nothing may reach Live once the list is refused"


def test_the_pos_dur_mix_up_keeps_its_own_message(live: _FakeLive) -> None:
    """The hint no schema can express, and the reason unknown keys stay a refusal."""
    result = server.write_clip_notes(
        track=0, slot=0, notes=[{"pitch": 60, "pos": 0, "dur": 1}]
    )
    assert result["ok"] is False
    said = " ".join(issue["message"] for issue in result["validation"]["errors"])
    assert "from_tick_notes" in said
    assert "stacks the clip on beat 0" in said


def test_the_read_path_and_the_write_path_agree_on_what_a_note_carries() -> None:
    """The two lists of Live-added keys must not drift apart.

    ``server.UNMODELLED_NOTE_KEYS`` is what the READ path strips;
    ``notes.TOLERATED_NOTE_KEYS`` is what the WRITE path accepts and drops. A key
    the reader emits but the writer refuses breaks the round trip, which is
    exactly the defect this file was written for.
    """
    only_on_read = server.UNMODELLED_NOTE_KEYS - TOLERATED_NOTE_KEYS
    assert not only_on_read, (
        "read_clip_notes can emit these keys but write_clip_notes refuses them, "
        f"so a note cannot be written back: {sorted(only_on_read)}"
    )
