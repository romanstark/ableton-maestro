"""Tests for :mod:`ableton_maestro.music.notes`: the measured traps first.

No socket, no Live, no sleeping. Where a test needs a clip it uses
:class:`FakeMidiClip`, a faithful stand-in for the two LOM calls the
``notes_set`` handler actually makes (``docs/protocol.md`` §5.8):
``add_new_notes`` adds notes, ``remove_notes_extended`` removes a
pitch/time window. Replace is remove-then-write on top of those, which is
exactly what the handler does inside Live.

The three failures these tests exist for are all measured, all silent, and all
report success:

* wrong dict keys collapse every note onto beat 0 with duration 0.25.
* a write that appends instead of replacing left 86 notes where a 23-note
  correction to a 63-note clip was meant to leave 23.
* tick values sent as beats are 480 times too large and the clip looks empty.
"""

from __future__ import annotations

from typing import Any

import pytest

from ableton_maestro.models import Note
from ableton_maestro.music import notes as N

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def note(pitch: int, start: float, duration: float, velocity: float = 100.0) -> Note:
    """One note, spelled out: the tests never rely on a default time."""
    return Note(pitch=pitch, start_time=start, duration=duration, velocity=velocity)


def melody(count: int, *, pitch: int = 60, step: float = 0.5, duration: float = 0.4) -> list[Note]:
    """``count`` notes on a grid, each one distinguishable from the next."""
    return [
        note(pitch + (i % 12), round(i * step, 6), duration, 95 + (i % 20))
        for i in range(count)
    ]


class FakeMidiClip:
    """The two Live 11+ note calls the handler uses, and nothing else.

    ``add_new_notes`` is additive: that is the LOM's behaviour and the
    origin of the doubling. It is also the only write call ``Clip`` has:
    ``set_notes_extended`` does not exist (measured 2026-08-29 against Live
    12.4.5), so this stand-in does not offer it either.
    ``remove_notes_extended`` takes its arguments in the order
    ``(from_pitch, pitch_span, from_time, time_span)``, which is *not* the order
    of Live's older ``remove_notes`` and is the swap the handler's own comment
    warns about.
    """

    def __init__(self, notes: list[Note] | None = None) -> None:
        self.notes: list[Note] = list(notes or [])

    def get_notes_extended(self) -> list[Note]:
        return list(self.notes)

    def add_new_notes(self, specs: list[Note]) -> None:
        self.notes.extend(specs)

    def remove_notes_extended(
        self, from_pitch: float, pitch_span: float, from_time: float, time_span: float
    ) -> None:
        pitch_hi = from_pitch + pitch_span
        time_hi = from_time + time_span
        self.notes = [
            n
            for n in self.notes
            if not (from_pitch <= n.pitch < pitch_hi and from_time <= n.start_time < time_hi)
        ]


def notes_set(
    clip: FakeMidiClip,
    new_notes: list[Note],
    *,
    mode: str = "replace",
    swap_remove_args: bool = False,
) -> dict[str, Any]:
    """The handler's own logic, in ten lines (``docs/protocol.md`` §5.8).

    ``swap_remove_args`` feeds the remover the argument order of Live's older
    ``remove_notes``: the mistake that produces a plausible-looking wrong
    result rather than an error.
    """
    before_count = len(clip.notes)
    span = max((n.start_time + n.duration + 1.0 for n in [*clip.notes, *new_notes]), default=1.0)
    if mode == "replace":
        if swap_remove_args:
            # remove_notes(from_time, from_pitch, time_span, pitch_span): wrong here.
            clip.remove_notes_extended(0.0, 0, span, 128)
        else:
            # remove_notes_extended(from_pitch, pitch_span, from_time, time_span).
            clip.remove_notes_extended(0, 128, 0.0, span)
    clip.add_new_notes(list(new_notes))
    return {
        "mode": mode,
        "before_count": before_count,
        "after_count": len(clip.notes),
        "written": len(new_notes),
    }


# --------------------------------------------------------------------------- #
# The wrong-keys trap: every note on beat 0, one sixteenth long
# --------------------------------------------------------------------------- #


def test_validate_note_dicts_names_the_wrong_keys_before_a_note_exists() -> None:
    """``pos``/``dur`` is humanize.py's tick form.

    It must never reach a write.
    """
    report = N.validate_note_dicts([{"pos": 0, "dur": 240, "pitch": 60}])

    assert not report.ok
    codes = {issue.code for issue in report.errors}
    assert "unknown_key" in codes
    assert "missing_key" in codes

    unknown = report.by_code("unknown_key")[0].message
    assert "'dur'" in unknown or "dur" in unknown
    assert "pos" in unknown
    # The message has to explain the consequence, not only name the key.
    assert "start_time=0.0" in unknown
    assert "duration=0.25" in unknown
    assert "from_tick_notes" in unknown

    missing = " ".join(i.message for i in report.by_code("missing_key"))
    assert "start_time" in missing
    assert "duration" in missing


def test_validate_catches_the_resulting_shape_and_explains_it() -> None:
    """By the time the keys are gone, the fingerprint is all that is left."""
    stacked = [note(p, 0.0, 0.25) for p in (60, 62, 64, 67)]

    report = N.validate(stacked)

    assert not report.ok
    issues = report.by_code("stacked_on_beat_zero")
    assert len(issues) == 1
    issue = issues[0]
    assert issue.severity == "error"
    assert issue.indices == (0, 1, 2, 3)

    message = issue.message
    # It must say what happened, why, and what to check, not merely "invalid".
    assert "start_time=0.0" in message
    assert "duration=0.25" in message
    assert "wrong dict keys" in message
    assert ".get(" in message
    assert "reported success" in message
    assert "'pos'/'dur'" in message
    assert "from_tick_notes" in message


def test_stack_check_needs_two_notes_and_the_exact_fingerprint() -> None:
    """A single sixteenth on beat 0 is a note, not a symptom."""
    assert N.validate([note(60, 0.0, 0.25)]).ok
    # Duration 0.5 on beat 0 is not the (0.0, 0.25) pair a legacy reader substitutes.
    assert N.validate([note(60, 0.0, 0.5), note(62, 0.0, 0.5)]).ok
    assert not N.validate([note(60, 0.0, 0.25), note(62, 0.0, 0.25)]).ok


def test_partial_stack_is_a_warning_not_an_error() -> None:
    """One bad batch in an otherwise healthy clip: the measured shape."""
    healthy = [note(72, 4.0 + i, 0.5) for i in range(8)]
    bad_batch = [note(60 + i, 0.0, 0.25) for i in range(N.STACK_WARN_MIN_NOTES)]

    report = N.validate([*bad_batch, *healthy])

    assert report.ok, "a partial stack must not block a write on its own"
    warning = report.by_code("partial_stack_on_beat_zero")[0]
    assert warning.severity == "warning"
    assert len(warning.indices) == N.STACK_WARN_MIN_NOTES
    assert "63" in warning.message and "86" in warning.message


def test_note_from_dict_rejects_the_tick_form_outright() -> None:
    """The model refuses the same shape one layer down."""
    with pytest.raises(ValueError, match="unknown key"):
        Note.from_dict({"pos": 0, "dur": 240, "pitch": 60})
    with pytest.raises(ValueError, match="missing required key"):
        Note.from_dict({"pitch": 60})


def test_validation_report_is_json_safe() -> None:
    report = N.validate([note(p, 0.0, 0.25) for p in (60, 62)])
    payload = report.to_dict()

    assert payload["ok"] is False
    assert payload["note_count"] == 2
    assert payload["errors"][0]["code"] == "stacked_on_beat_zero"
    assert payload["errors"][0]["indices"] == [0, 1]
    assert "stacked_on_beat_zero" in payload["summary"]


# --------------------------------------------------------------------------- #
# Replace semantics: 63 + 23 = 86 was measured, and must not happen again
# --------------------------------------------------------------------------- #


def test_append_reproduces_the_measured_doubling() -> None:
    """A write that appends: 63 notes plus a 23-note correction gave 86."""
    clip = FakeMidiClip(melody(63))
    correction = melody(23, pitch=72, step=0.25)

    result = notes_set(clip, correction, mode="append")

    assert result["written"] == 23, "the reply counts what was sent"
    assert result["after_count"] == 86, "the clip holds something else entirely"
    assert len(clip.notes) == 86


def test_replace_writes_the_list_that_was_sent() -> None:
    clip = FakeMidiClip(melody(63))
    correction = melody(23, pitch=72, step=0.25)

    result = notes_set(clip, correction, mode="replace")

    assert result["before_count"] == 63
    assert result["written"] == 23
    assert result["after_count"] == 23
    assert N.diff(correction, clip.get_notes_extended()).is_empty


def test_applying_the_same_list_twice_yields_the_same_clip() -> None:
    """The property the module is built on. Twice written, once present."""
    clip = FakeMidiClip()
    line = melody(23, pitch=64)

    notes_set(clip, line, mode="replace")
    after_first = clip.get_notes_extended()
    notes_set(clip, line, mode="replace")
    after_second = clip.get_notes_extended()

    assert len(after_first) == 23
    assert len(after_second) == 23
    assert N.sort_notes(after_first) == N.sort_notes(after_second)
    assert N.diff(after_first, after_second).is_empty


def test_the_swapped_remove_argument_order_doubles_silently() -> None:
    """``remove_notes_extended`` and ``remove_notes`` disagree about order.

    Fed the older ``remove_notes`` order, the remover clears a pitch band no
    melody lives in, the write lands on top, and the result looks like success.
    """
    clip = FakeMidiClip(melody(63))
    correction = melody(23, pitch=72, step=0.25)

    result = notes_set(clip, correction, mode="replace", swap_remove_args=True)

    assert result["written"] == 23
    assert result["after_count"] == 86, "nothing was removed: a replace that appended"

    # And this is how a caller finds out: diff the read-back against the intent.
    difference = N.diff(correction, clip.get_notes_extended())
    assert not difference.is_empty
    assert len(difference.added) == 63
    assert "63 added" in difference.summary()


def test_replace_range_is_idempotent_and_merge_is_not() -> None:
    """The client-side mirror: replace_range replaces, merge concatenates."""
    base = N.from_beat_tuples([(0.0, 60, 0.5), (1.0, 62, 0.5), (2.0, 64, 1.0)])
    patch = N.from_beat_tuples([(1.0, 67, 0.5)])

    once = N.replace_range(base, patch, start=1.0, end=2.0)
    twice = N.replace_range(once, patch, start=1.0, end=2.0)

    assert [n.pitch for n in once] == [60, 67, 64]
    assert once == twice

    doubled = N.merge(base, base)
    assert len(doubled) == 2 * len(base)
    assert len(N.merge(base, base, dedupe_exact=True)) == len(base)


def test_replace_range_leaves_other_pitch_lanes_alone() -> None:
    """Replacing the hi-hat lane must not touch the kick."""
    drums = N.from_beat_tuples(
        [(0.0, 36, 0.1), (0.0, 42, 0.1), (0.5, 42, 0.1), (1.0, 36, 0.1)]
    )
    hats = N.from_beat_tuples([(0.25, 42, 0.1), (0.75, 42, 0.1)])

    out = N.replace_range(drums, hats, pitch_low=42, pitch_high=42)

    assert sorted(n.start_time for n in out if n.pitch == 36) == [0.0, 1.0]
    assert sorted(n.start_time for n in out if n.pitch == 42) == [0.25, 0.75]


def test_empty_list_written_as_replace_clears_the_clip_and_says_so() -> None:
    report = N.validate([])
    assert report.ok, "an empty write is legal, not an error"
    warning = report.by_code("empty_note_list")[0]
    assert warning.severity == "warning"
    assert "clears the clip" in warning.message


# --------------------------------------------------------------------------- #
# diff: what actually changed, as opposed to what was reported
# --------------------------------------------------------------------------- #


def test_diff_reports_added_removed_changed_and_unchanged() -> None:
    before = N.from_beat_tuples([(0.0, 60, 0.5), (1.0, 62, 0.5), (2.0, 64, 1.0)])
    after = [
        before[0],                                   # unchanged
        Note(pitch=62, start_time=1.0, duration=0.5, velocity=120),  # changed
        note(67, 3.0, 0.5),                          # added
    ]

    result = N.diff(before, after)

    assert [n.pitch for n in result.added] == [67]
    assert [n.pitch for n in result.removed] == [64]
    assert len(result.changed) == 1
    assert result.changed[0].fields == {"velocity": (100, 120)}
    assert result.unchanged == 1
    assert result.summary() == "1 added, 1 removed, 1 changed, 1 unchanged"
    assert not result.is_empty


def test_diff_of_a_list_against_itself_is_empty() -> None:
    line = melody(23)
    result = N.diff(line, list(line))
    assert result.is_empty
    assert result.unchanged == 23
    assert result.summary() == "0 added, 0 removed, 0 changed, 23 unchanged"


def test_diff_tolerates_the_float_velocity_live_reports() -> None:
    """Live reports velocity as a float. The model carries an int."""
    before = [Note(pitch=60, start_time=0.0, duration=1.0, velocity=100)]
    read_back = [Note(pitch=60, start_time=0.0, duration=1.0, velocity=100.00001)]

    assert N.diff(before, read_back).is_empty


def test_diff_by_pitch_reads_movement_as_a_change() -> None:
    """After a shift, "23 notes moved" is the useful sentence."""
    before = N.from_beat_tuples([(0.0, 60, 0.5), (1.0, 62, 0.5)])
    after = N.shift(before, 0.25)

    exact = N.diff(before, after)
    assert len(exact.added) == 2 and len(exact.removed) == 2

    by_pitch = N.diff(before, after, match_on="pitch")
    assert len(by_pitch.changed) == 2
    assert by_pitch.changed[0].fields["start_time"] == (0.0, 0.25)


def test_diff_notices_a_dropped_extension_field() -> None:
    before = [Note(pitch=60, start_time=0.0, duration=1.0, probability=0.5)]
    after = [Note(pitch=60, start_time=0.0, duration=1.0, probability=None)]

    result = N.diff(before, after)

    assert len(result.changed) == 1
    assert result.changed[0].fields["probability"] == (0.5, None)


def test_diff_rejects_an_unknown_match_mode() -> None:
    with pytest.raises(ValueError, match="match_on"):
        N.diff([], [], match_on="whatever")  # type: ignore[arg-type]


def test_diff_is_json_safe() -> None:
    payload = N.diff([note(60, 0.0, 1.0)], [note(62, 0.0, 1.0)]).to_dict()
    assert payload["added"][0]["pitch"] == 62
    assert payload["removed"][0]["pitch"] == 60
    assert payload["summary"].startswith("1 added, 1 removed")


# --------------------------------------------------------------------------- #
# transpose
# --------------------------------------------------------------------------- #


def test_transpose_round_trips_and_keeps_every_other_field() -> None:
    before = [
        Note(pitch=60, start_time=0.0, duration=0.5, velocity=97, probability=0.8),
        Note(pitch=64, start_time=1.0, duration=0.25, velocity=110, mute=True),
    ]

    up = N.transpose(before, 12)
    back = N.transpose(up, -12)

    assert [n.pitch for n in up] == [72, 76]
    assert back == N.sort_notes(before)
    assert up[0].probability == 0.8
    assert up[1].mute is True


def test_transpose_out_of_range_raises_by_default_and_names_the_pitches() -> None:
    high = [note(120, 0.0, 1.0), note(60, 1.0, 1.0)]

    with pytest.raises(ValueError) as excinfo:
        N.transpose(high, 12)

    message = str(excinfo.value)
    assert "132" in message
    assert "clamp" in message and "drop" in message


def test_transpose_clamp_and_drop_are_opt_in() -> None:
    high = [note(120, 0.0, 1.0), note(60, 1.0, 1.0)]

    assert [n.pitch for n in N.transpose(high, 12, out_of_range="clamp")] == [127, 72]
    assert [n.pitch for n in N.transpose(high, 12, out_of_range="drop")] == [72]

    low = [note(2, 0.0, 1.0)]
    assert [n.pitch for n in N.transpose(low, -12, out_of_range="clamp")] == [0]
    assert N.transpose(low, -12, out_of_range="drop") == []


def test_transpose_by_zero_is_a_sorted_copy() -> None:
    unsorted = [note(64, 2.0, 1.0), note(60, 0.0, 1.0)]
    out = N.transpose(unsorted, 0)
    assert [n.pitch for n in out] == [60, 64]
    assert out is not unsorted


# --------------------------------------------------------------------------- #
# quantize
# --------------------------------------------------------------------------- #


def test_quantize_snaps_to_the_grid_and_is_idempotent() -> None:
    off_grid = N.from_beat_tuples([(0.03, 60, 0.5), (0.98, 62, 0.5), (2.26, 64, 1.0)])

    once = N.quantize_times(off_grid, 0.25)
    twice = N.quantize_times(once, 0.25)

    assert [n.start_time for n in once] == [0.0, 1.0, 2.25]
    assert once == twice


def test_quantize_strength_moves_part_of_the_way() -> None:
    off_grid = N.from_beat_tuples([(0.03, 60, 0.5)])

    assert N.quantize_times(off_grid, 0.25, strength=0.5)[0].start_time == pytest.approx(0.015)
    assert N.quantize_times(off_grid, 0.25, strength=0.0) == list(off_grid)


def test_quantize_triplet_grid_survives_the_ordering_trap() -> None:
    """0.5 is a multiple of 1/6 but not of 1/3: test binary before triplet."""
    third = 1.0 / 3.0
    triplets = N.from_beat_tuples([(0.34, 60, 0.2), (0.66, 62, 0.2)])

    out = N.quantize_times(triplets, third)

    assert out[0].start_time == pytest.approx(third, abs=1e-6)
    assert out[1].start_time == pytest.approx(2 * third, abs=1e-6)


def test_quantize_ends_never_produces_a_zero_length_note() -> None:
    ornaments = [note(60, 0.02, 0.03), note(62, 1.01, 0.9)]

    out = N.quantize_times(ornaments, 0.25, quantize_ends=True)

    assert all(n.duration > 0 for n in out)
    assert out[0].duration == pytest.approx(0.25)


def test_quantize_rejects_impossible_arguments() -> None:
    with pytest.raises(ValueError, match="grid must be positive"):
        N.quantize_times([], 0.0)
    with pytest.raises(ValueError, match="strength"):
        N.quantize_times([], 0.25, strength=1.5)


# --------------------------------------------------------------------------- #
# shift and stretch
# --------------------------------------------------------------------------- #


def test_shift_round_trips() -> None:
    before = N.from_beat_tuples([(0.0, 60, 0.5), (1.0, 62, 0.5), (2.0, 64, 1.0)])

    moved = N.shift(before, 4.0)
    back = N.shift(moved, -4.0)

    assert [n.start_time for n in moved] == [4.0, 5.0, 6.0]
    assert back == before
    assert [n.duration for n in moved] == [n.duration for n in before]


def test_shift_before_zero_has_four_named_policies() -> None:
    before = N.from_beat_tuples([(0.0, 60, 0.5), (1.0, 62, 0.5), (2.0, 64, 1.0)])

    assert len(N.shift(before, -1.5)) == 1, "drop is the default"
    assert [n.start_time for n in N.shift(before, -1.5, before_zero="clamp")] == [0.0, 0.0, 0.5]
    assert [n.start_time for n in N.shift(before, -1.5, before_zero="keep")] == [-1.5, -0.5, 0.5]
    with pytest.raises(ValueError, match="before the clip start"):
        N.shift(before, -1.5, before_zero="error")


def test_shift_keep_leaves_something_validate_reports() -> None:
    """"keep" is honest only because validate() then says so."""
    kept = N.shift(N.from_beat_tuples([(0.0, 60, 0.5)]), -1.5, before_zero="keep")

    report = N.validate(kept)

    assert not report.ok
    assert report.errors[0].code == "negative_start_time"


def test_shift_drop_is_visible_in_a_diff_matched_on_pitch() -> None:
    """How many notes "drop" dropped is a diff, not a return value.

    It has to be ``match_on="pitch"``. Under the default every surviving note
    has moved as well, so it reads as a removal plus an addition and ``removed``
    counts the whole list rather than the losses.
    """
    before = N.from_beat_tuples([(0.0, 60, 0.5), (1.0, 62, 0.5), (2.0, 64, 1.0)])

    after = N.shift(before, -1.5)

    assert len(N.diff(before, after, match_on="pitch").removed) == 2
    assert len(N.diff(before, after).removed) == 3, "the default mode overcounts here"


def test_stretch_round_trips_around_its_origin() -> None:
    before = N.from_beat_tuples([(0.0, 60, 0.5), (1.0, 62, 0.5), (2.0, 64, 1.0)])

    slower = N.stretch(before, 2.0)
    back = N.stretch(slower, 0.5)

    assert [(n.start_time, n.duration) for n in slower] == [(0.0, 1.0), (2.0, 1.0), (4.0, 2.0)]
    assert back == before


def test_stretch_can_leave_durations_alone() -> None:
    before = N.from_beat_tuples([(0.0, 60, 0.5), (1.0, 62, 0.5)])

    out = N.stretch(before, 2.0, scale_durations=False)

    assert [n.start_time for n in out] == [0.0, 2.0]
    assert [n.duration for n in out] == [0.5, 0.5]


def test_stretch_origin_is_the_fixed_point() -> None:
    before = N.from_beat_tuples([(2.0, 60, 1.0), (4.0, 62, 1.0)])

    out = N.stretch(before, 0.5, origin=2.0)

    assert [n.start_time for n in out] == [2.0, 3.0]


def test_stretch_rejects_a_non_positive_factor() -> None:
    with pytest.raises(ValueError, match="factor must be positive"):
        N.stretch([], 0.0)


def test_stretching_past_the_clip_end_is_reported_when_the_length_is_known() -> None:
    """Live keeps the material. It never sounds."""
    before = N.from_beat_tuples([(0.0, 60, 1.0), (2.0, 62, 1.0)])

    stretched = N.stretch(before, 2.0)
    report = N.validate(stretched, clip_length=4.0)

    codes = {i.code for i in report.warnings}
    assert "starts_past_clip_end" in codes
    assert report.ok, "past the clip end is a warning, never a blocked write"


# --------------------------------------------------------------------------- #
# Ticks and beats: the off-by-480 disaster, in both directions
# --------------------------------------------------------------------------- #


def test_ticks_per_quarter_is_480() -> None:
    assert N.TICKS_PER_QUARTER == 480


@pytest.mark.parametrize(
    ("beats", "ticks"),
    [
        (0.0, 0),
        (0.25, 120),
        (0.5, 240),
        (0.75, 360),
        (1.0, 480),
        (1.5, 720),
        (4.0, 1920),
        (1.0 / 3.0, 160),
    ],
)
def test_beats_and_ticks_convert_both_ways(beats: float, ticks: int) -> None:
    assert N.beats_to_ticks(beats) == ticks
    assert N.ticks_to_beats(ticks) == pytest.approx(beats, abs=1e-6)


def test_tick_note_round_trip_is_exact_on_the_grid() -> None:
    tick_form = [
        {"pos": 0, "dur": 240, "pitch": 60},
        {"pos": 720, "dur": 480, "pitch": 64, "velocity": 110},
    ]

    beat_form = N.from_tick_notes(tick_form)
    back = N.to_tick_notes(beat_form)

    assert [(n.start_time, n.duration) for n in beat_form] == [(0.0, 0.5), (1.5, 1.0)]
    assert [(e["pos"], e["dur"], e["pitch"]) for e in back] == [
        (0, 240, 60),
        (720, 480, 64),
    ]
    assert back[1]["velocity"] == 110


def test_tick_round_trip_is_lossy_off_the_grid_as_documented() -> None:
    """480 per quarter is exact for ordinary subdivisions and nothing else."""
    seventh = [Note(pitch=60, start_time=1.0 / 7.0, duration=1.0)]

    back = N.from_tick_notes(N.to_tick_notes(seventh))

    assert back[0].start_time != pytest.approx(1.0 / 7.0, abs=1e-9)
    assert back[0].start_time == pytest.approx(1.0 / 7.0, abs=1e-3)


def test_from_tick_notes_refuses_to_substitute_a_missing_key() -> None:
    with pytest.raises(ValueError) as excinfo:
        N.from_tick_notes([{"pos": 0, "pitch": 60}])

    message = str(excinfo.value)
    assert "'dur'" in message
    assert "stack a whole clip on beat 0" in message


def test_from_tick_notes_velocity_fallback_only_fills_the_gaps() -> None:
    out = N.from_tick_notes(
        [{"pos": 0, "dur": 240, "pitch": 60}, {"pos": 240, "dur": 240, "pitch": 62,
                                              "velocity": 42}],
        velocity=77,
    )
    assert [n.velocity for n in out] == [77, 42]


def test_tick_values_used_as_beats_are_480_times_too_large() -> None:
    """The disaster itself: skip the conversion and bar 1 becomes bar 121."""
    tick_form = [{"pos": 1920, "dur": 480, "pitch": 60}]

    correct = N.from_tick_notes(tick_form)[0]
    naive = Note(
        pitch=tick_form[0]["pitch"],
        start_time=float(tick_form[0]["pos"]),
        duration=float(tick_form[0]["dur"]),
    )

    assert correct.start_time == 4.0
    assert naive.start_time == 1920.0
    assert naive.start_time / correct.start_time == 480.0
    assert naive.duration / correct.duration == 480.0

    # And validate() smells it, because 1920 beats is 480 bars of 4/4.
    report = N.validate([naive])
    warning = report.by_code("looks_like_ticks")[0]
    assert warning.severity == "warning"
    assert "480" in warning.message
    assert "from_tick_notes" in warning.message


def test_looks_like_ticks_does_not_fire_on_a_long_but_plausible_clip() -> None:
    """A note at beat 200 is bar 51: long, and entirely legal."""
    report = N.validate([note(60, 200.0, 1.0)])
    assert report.by_code("looks_like_ticks") == []


# --------------------------------------------------------------------------- #
# Odds and ends the read-modify-write cycle leans on
# --------------------------------------------------------------------------- #


def test_sort_order_is_start_then_pitch() -> None:
    jumbled = [note(64, 1.0, 0.5), note(60, 1.0, 0.5), note(62, 0.0, 0.5)]
    assert [n.pitch for n in N.sort_notes(jumbled)] == [62, 60, 64]


def test_span_reports_the_furthest_end_not_the_last_note() -> None:
    with_a_pedal = [note(36, 0.0, 8.0), note(72, 4.0, 0.5)]
    assert N.span(with_a_pedal) == (0.0, 8.0)
    assert N.span([]) == (0.0, 0.0)


def test_same_pitch_overlap_is_reported_polyphony_is_not() -> None:
    chord = [note(60, 0.0, 1.0), note(64, 0.0, 1.0), note(67, 0.0, 1.0)]
    assert N.find_overlaps(chord) == []
    assert N.validate(chord).by_code("same_pitch_overlap") == []

    doubled_lane = [note(60, 0.0, 2.0), note(60, 1.0, 1.0)]
    assert N.find_overlaps(doubled_lane) == [(0, 1)]
    assert N.validate(doubled_lane).by_code("same_pitch_overlap")


def test_monophonic_material_can_ask_for_the_stricter_check() -> None:
    chord = [note(60, 0.0, 1.0), note(64, 0.0, 1.0)]
    assert N.validate(chord, allow_overlap=False).by_code("polyphonic_overlap")


def test_dedupe_keeps_the_one_you_asked_for() -> None:
    twins = [
        Note(pitch=60, start_time=0.0, duration=1.0, velocity=80),
        Note(pitch=60, start_time=0.0, duration=2.0, velocity=120),
    ]

    assert [n.velocity for n in N.dedupe(twins)] == [80]
    assert [n.velocity for n in N.dedupe(twins, keep="last")] == [120]
    assert [n.velocity for n in N.dedupe(twins, keep="loudest")] == [120]
    assert [n.duration for n in N.dedupe(twins, keep="longest")] == [2.0]
    assert len(N.dedupe(twins, match_duration=True)) == 2


def test_to_dicts_and_from_dicts_survive_the_wire_shape() -> None:
    before = [Note(pitch=60, start_time=0.0, duration=0.5, velocity=97, probability=0.9)]

    wire = N.to_dicts(before)
    back = N.from_dicts(wire)

    assert wire[0]["probability"] == 0.9
    assert "velocity_deviation" not in wire[0], "an unset extension must not be invented"
    assert back == before
