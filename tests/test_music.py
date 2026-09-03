"""Tests for the theory and humanize halves of :mod:`ableton_maestro.music`.

Two halves, two kinds of claim.

:mod:`ableton_maestro.music.theory` is checked against musical facts that are
true everywhere (C major, A minor, ii-V-I, the intervals of a diatonic third)
plus the module's own measured constants, which it re-derives itself in
:func:`theory.self_check`. Nothing here needs Live, and nothing here is allowed
to open a socket.

:mod:`ableton_maestro.music.humanize` is checked for the three properties a
build script depends on: it is deterministic given a seed, it never leaves a
note where Live cannot draw it, and :func:`humanize.to_ableton` is the one place
ticks become beats. That last one is the 480 trap: skip the call and every
number is 480 times too large, with no error and an apparently empty clip.
"""

from __future__ import annotations

import random
import re

import pytest

from ableton_maestro.models import Note
from ableton_maestro.music import humanize as H
from ableton_maestro.music import theory as T

# --------------------------------------------------------------------------- #
# Names and units
# --------------------------------------------------------------------------- #


def test_note_names_count_c4_as_60() -> None:
    """Unambiguous in MIDI. Ableton draws the same note as C3."""
    assert T.note_name(60) == "C4"
    assert T.note_name(69) == "A4"
    assert T.note_name(0) == "C-1"
    assert T.note_name(127) == "G9"


def test_pitch_class_names_can_be_spelled_with_flats() -> None:
    assert T.pc_name(10) == "A#"
    assert T.pc_name(10, flats=True) == "Bb"
    assert T.pc_name(22, flats=True) == "Bb", "reduced modulo 12"


def test_chord_symbols_read_the_way_a_musician_writes_them() -> None:
    assert T.chord_symbol("Bb", "min") == "Bbm"
    assert T.chord_symbol("C", "maj") == "C"
    assert T.chord_symbol("G", "dom7") == "G7"
    assert T.chord_symbol("D", "min7b5") == "Dm7b5"


@pytest.mark.parametrize(
    ("signature", "beats"),
    [("4/4", 4.0), ("3/4", 3.0), ("6/8", 3.0), ("7/8", 3.5), ("5/4", 5.0), ("12/8", 6.0)],
)
def test_bar_beats_counts_in_quarter_notes_the_way_live_does(
    signature: str, beats: float
) -> None:
    assert T.bar_beats(signature) == beats


# --------------------------------------------------------------------------- #
# Scales
# --------------------------------------------------------------------------- #


def test_c_major_is_the_white_keys() -> None:
    assert sorted(T.pitch_classes("C", "major")) == [0, 2, 4, 5, 7, 9, 11]
    assert T.scale_pitches("C", "major", 60, 72) == [60, 62, 64, 65, 67, 69, 71, 72]


def test_a_minor_is_the_relative_of_c_major() -> None:
    """Same notes, different tonic: that is what "relative" means."""
    assert T.pitch_classes("A", "minor") == T.pitch_classes("C", "major")
    assert T.scale_pitches("A", "minor", 57, 69) == [57, 59, 60, 62, 64, 65, 67, 69]


def test_the_modes_are_rotations_of_one_another() -> None:
    white_keys = T.pitch_classes("C", "major")
    assert T.pitch_classes("D", "dorian") == white_keys
    assert T.pitch_classes("E", "phrygian") == white_keys
    assert T.pitch_classes("F", "lydian") == white_keys
    assert T.pitch_classes("G", "mixolydian") == white_keys
    assert T.pitch_classes("B", "locrian") == white_keys
    assert T.pitch_classes("C", "ionian") == white_keys


def test_the_minor_variants_carry_the_leading_tone_natural_minor_lacks() -> None:
    """A natural minor has G. A harmonic and melodic minor raise it to G#."""
    natural = T.pitch_classes("A", "minor")
    assert 7 in natural and 8 not in natural  # G natural, no G#

    harmonic = T.pitch_classes("A", "harmonic_minor")
    assert 8 in harmonic and 7 not in harmonic
    assert sorted(harmonic) == [0, 2, 4, 5, 8, 9, 11]  # A B C D E F G#

    melodic = T.pitch_classes("A", "melodic_minor")
    assert sorted(melodic) == [0, 2, 4, 6, 8, 9, 11]  # A B C D E F# G#


def test_the_five_and_six_note_scales_have_five_and_six_notes() -> None:
    assert len(T.pitch_classes("A", "minor_pent")) == 5
    assert len(T.pitch_classes("C", "major_pent")) == 5
    assert len(T.pitch_classes("A", "blues")) == 6
    assert len(T.pitch_classes("C", "whole_tone")) == 6
    # A minor pentatonic: A C D E G.
    assert sorted(T.pitch_classes("A", "minor_pent")) == [0, 2, 4, 7, 9]


def test_an_unknown_mode_is_refused_with_the_known_ones_named() -> None:
    with pytest.raises(KeyError, match="unknown mode"):
        T.pitch_classes("C", "klingon")


# --------------------------------------------------------------------------- #
# Chords
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("quality", "intervals"),
    [
        ("maj", (0, 4, 7)),
        ("min", (0, 3, 7)),
        ("dim", (0, 3, 6)),
        ("aug", (0, 4, 8)),
        ("sus2", (0, 2, 7)),
        ("sus4", (0, 5, 7)),
        ("maj7", (0, 4, 7, 11)),
        ("min7", (0, 3, 7, 10)),
        ("dom7", (0, 4, 7, 10)),
        ("min7b5", (0, 3, 6, 10)),
        ("dim7", (0, 3, 6, 9)),
    ],
)
def test_chord_qualities_are_the_textbook_interval_stacks(
    quality: str, intervals: tuple[int, ...]
) -> None:
    chord = T.build_chord("C", quality, octave=4)
    assert tuple(p - 60 for p in chord) == intervals


def test_build_chord_places_the_root_in_the_octave_it_is_given() -> None:
    assert T.build_chord("C", "maj", 4) == [60, 64, 67]
    assert T.build_chord("A", "min7", 3) == [57, 60, 64, 67]
    assert T.build_chord("G", "dom7", 3) == [55, 59, 62, 65]


def test_inversions_lift_the_lowest_notes_by_an_octave() -> None:
    assert T.build_chord("C", "maj", 4, inversion=1) == [64, 67, 72]
    assert T.build_chord("C", "maj", 4, inversion=2) == [67, 72, 76]
    # The pitch classes are unchanged. Only the position moves.
    assert {p % 12 for p in T.build_chord("C", "maj", 4, inversion=2)} == {0, 4, 7}


def test_an_unknown_quality_is_refused_with_the_known_ones_named() -> None:
    with pytest.raises(KeyError, match="unknown chord quality"):
        T.build_chord("C", "power")


def test_identify_chord_round_trips_every_quality() -> None:
    for quality in T.CHORD_QUALITIES:
        chord = T.build_chord("C", quality, octave=4)
        found = T.identify_chord(chord)
        assert found is not None, quality
        root, name = found
        assert root == "C"
        assert T.CHORD_QUALITIES[name] == T.CHORD_QUALITIES[quality], quality


def test_identify_chord_reads_an_inversion_back_to_its_root() -> None:
    assert T.identify_chord([64, 67, 72]) == ("C", "maj")
    assert T.identify_chord([51, 58, 66]) == ("D#", "min")
    assert T.identify_chord([51, 58, 66], flats=True) == ("Eb", "min")


def test_two_notes_are_not_a_chord() -> None:
    """A dyad does not determine a quality. Guessing is worse than silence."""
    assert T.identify_chord([58, 61]) is None
    assert T.identify_chord([60, 67]) is None
    assert T.identify_chord([60, 61, 62]) is None, "not a quality we carry"


# --------------------------------------------------------------------------- #
# Degrees: the quality comes from the scale, not from how the label is written
# --------------------------------------------------------------------------- #


def test_two_five_one_in_c_major() -> None:
    assert [T.degree_chord("C", d, "major") for d in ("ii", "V", "I")] == [
        ("D", "min"),
        ("G", "maj"),
        ("C", "maj"),
    ]


def test_two_five_one_with_sevenths_is_min7_dom7_maj7() -> None:
    """The '7' suffix takes the diatonic seventh.

    That is what makes ii-V-I work.
    """
    assert [T.degree_chord("C", d, "major") for d in ("ii7", "V7", "I7")] == [
        ("D", "min7"),
        ("G", "dom7"),
        ("C", "maj7"),
    ]


def test_the_diatonic_triads_of_c_major_are_the_ones_everyone_learns() -> None:
    expected = [
        ("C", "maj"), ("D", "min"), ("E", "min"), ("F", "maj"),
        ("G", "maj"), ("A", "min"), ("B", "dim"),
    ]
    labels = ("I", "ii", "iii", "IV", "V", "vi", "vii")
    assert [T.degree_chord("C", d, "major") for d in labels] == expected


def test_the_diatonic_triads_of_a_natural_minor() -> None:
    expected = [
        ("A", "min"), ("B", "dim"), ("C", "maj"), ("D", "min"),
        ("E", "min"), ("F", "maj"), ("G", "maj"),
    ]
    labels = ("i", "ii", "III", "iv", "v", "VI", "VII")
    assert [T.degree_chord("A", d, "minor") for d in labels] == expected


def test_case_does_not_decide_the_quality_the_scale_does() -> None:
    """'IV' in minor is still minor.

    The major chord there has an out-of-key note.
    """
    assert T.degree_chord("A", "IV", "minor") == ("D", "min")
    assert T.degree_chord("A", "iv", "minor") == ("D", "min")
    # In natural minor the fifth degree is minor, so v7 is min7 and not V7.
    assert T.degree_chord("A", "v7", "minor") == ("E", "min7")
    # A quality asked for by name is honoured.
    assert T.degree_chord("A", "Vmaj", "minor") == ("E", "maj")


def test_a_flat_key_is_spelled_with_flats() -> None:
    assert T.degree_chord("Bb", "VII", "minor") == ("Ab", "maj")
    assert T.degree_chord("Bb", "i", "minor") == ("Bb", "min")


def test_parse_degree_splits_accidental_numeral_and_suffix() -> None:
    assert T.parse_degree("iv") == (4, 0, "")
    assert T.parse_degree("bIIImaj7") == (3, -1, "maj7")
    assert T.parse_degree("V7") == (5, 0, "7")
    with pytest.raises(ValueError, match="unreadable degree label"):
        T.parse_degree("H9")


def test_degree_harmony_needs_seven_notes() -> None:
    with pytest.raises(ValueError, match="Degree harmony needs seven"):
        T.degree_chord("A", "i", "minor_pent")


def test_the_measured_pool_and_its_major_rotation() -> None:
    assert T.progression_pool("minor") == ("i", "III", "iv", "v", "VI", "VII")
    assert T.progression_pool("major") == ("I", "ii", "iii", "IV", "V", "vi")


def test_a_progression_reports_its_source_and_its_pool_membership() -> None:
    chords = T.progression("Bb", form="001")

    assert [c["chord"] for c in chords] == ["Bbm", "Ab", "Bbm", "Gb", "Ebm", "Gb", "Ab"]
    assert all(c["in_pool"] for c in chords)
    assert all("measured" in c["source"] for c in chords)

    own = T.progression("F#", degrees=["i", "VII", "VI"])
    assert [c["chord"] for c in own] == ["F#m", "E", "D"]
    assert all("NOT measured" in c["source"] for c in own)


# --------------------------------------------------------------------------- #
# The point of the whole module: a third is not a fixed number of semitones
# --------------------------------------------------------------------------- #


def test_a_diatonic_third_below_is_three_or_four_semitones_depending_where_it_sits() -> None:
    """A constant offset produces out-of-key notes, heard as wrong."""
    minor_third = 72 - T.transpose_diatonic(72, -2, "C", "major")   # C5 -> A4
    major_third = 71 - T.transpose_diatonic(71, -2, "C", "major")   # B4 -> G4

    assert minor_third == 3
    assert major_third == 4


def test_a_diatonic_harmony_line_stays_in_key_where_a_fixed_offset_would_not() -> None:
    melody = [72, 71, 69, 67, 65, 64, 62, 60]

    harmony = T.harmony_line(melody, -2, "C", "major", floor=None)
    naive = [p - 4 for p in melody]

    assert T.check_in_key(harmony, "C", "major") == []
    assert T.check_in_key(naive, "C", "major") != []


def test_transpose_diatonic_keeps_a_harmony_voice_out_of_the_cellar() -> None:
    assert T.transpose_diatonic(57, -2, "C", "major") == 65, "lifted above the G3 floor"
    assert T.transpose_diatonic(57, -2, "C", "major", floor=None) == 53


def test_an_out_of_key_starting_note_snaps_to_the_nearest_scale_tone() -> None:
    """Blindly subtracting semitones is exactly the mistake this prevents."""
    assert T.transpose_diatonic(61, -2, "C", "major", floor=None) == 57


def test_check_in_key_points_at_the_offending_index() -> None:
    assert T.check_in_key([60, 62, 64], "C", "major") == []
    assert T.check_in_key([60, 61, 62], "C", "major") == [(1, "C#4")]


# --------------------------------------------------------------------------- #
# Checks over time
# --------------------------------------------------------------------------- #


def test_check_against_chords_names_the_bar_and_the_note() -> None:
    findings = T.check_against_chords([(0.0, 60), (1.0, 61)], [[60, 64, 67]])

    assert len(findings) == 1
    assert "bar 1" in findings[0]
    assert "C#4" in findings[0]


def test_check_against_chords_tolerates_extensions_unless_told_not_to() -> None:
    """The 9th over a C major triad extends the chord.

    It does not contradict it.
    """
    ninth = [(0.0, 62)]

    assert T.check_against_chords(ninth, [[60, 64, 67]]) == []
    assert T.check_against_chords(ninth, [[60, 64, 67]], tolerate_extensions=False) != []


def test_check_against_chords_follows_an_explicit_schedule() -> None:
    """Two chords inside one bar: the case a per-bar list cannot express."""
    schedule = [(0.0, [60, 64, 67]), (2.0, [62, 65, 69])]   # C, then Dm on beat 3
    strict = {"tolerate_extensions": False}

    # D on beat 3 belongs to the Dm the schedule puts there.
    assert T.check_against_chords([(2.0, 62)], [[60, 64, 67]], schedule=schedule, **strict) == []
    # Without the schedule the whole bar is C major and the same note is out.
    assert T.check_against_chords([(2.0, 62)], [[60, 64, 67]], **strict) != []
    # And on beat 1 the schedule still says C major.
    assert T.check_against_chords([(0.0, 62)], [[60, 64, 67]], schedule=schedule, **strict) != []


def test_check_sections_finds_gaps_overlaps_and_a_wrong_total() -> None:
    assert T.check_sections([("A", 1, 8), ("B", 9, 8)], 16) == []

    gap = T.check_sections([("A", 1, 8), ("B", 10, 8)], 17)
    assert any("gap" in f for f in gap)

    overlap = T.check_sections([("A", 1, 8), ("B", 8, 8)], 15)
    assert any("overlap" in f for f in overlap)

    short = T.check_sections([("A", 1, 8)], 16)
    assert any("total length 8 bars, expected 16" in f for f in short)


# --------------------------------------------------------------------------- #
# The measured half: it re-derives its own constants
# --------------------------------------------------------------------------- #


def test_self_check_reproduces_every_measured_example() -> None:
    """A finding here is a module regression, not a new fact about music."""
    assert T.self_check() == []


def test_the_measured_voicing_puts_the_third_on_top() -> None:
    """The most usable number of the measurement: melody on the third."""
    chord = T.voicing("Eb", "min")

    assert chord == [51, 58, 66]
    root_pc = T.NAME_TO_PC["Eb"]
    assert (chord[-1] - root_pc) % 12 == 3, "top voice is the minor third"
    assert (chord[1] - chord[0]) == 7, "an open fifth at the bottom"
    assert T.VOICING_TOP_RANGE[0] <= chord[-1] <= T.VOICING_TOP_RANGE[1]
    assert T.VOICING_BASS_RANGE[0] <= chord[0] <= T.VOICING_BASS_RANGE[1]


def test_a_voicing_normalises_to_the_same_shape_in_every_octave() -> None:
    assert T.voicing_shape_of([51, 58, 66]) == ((1, 0), (5, 0), (3, 1))
    assert T.voicing_shape_of([39, 46, 54]) == ((1, 0), (5, 0), (3, 1))
    assert T.shape_text(T.voicing_shape_of([51, 58, 66])) == "1-5-3oct"


def test_a_thirdless_chord_gets_a_shape_that_fits_instead_of_a_guess() -> None:
    assert T.shape_fits("min", "1-5-3oct") is True
    assert T.shape_fits("sus2", "1-5-3oct") is False
    assert T.fitting_shape("sus2", "1-5-3oct") == "1-5-2oct"

    with pytest.raises(ValueError, match="has no degree 3"):
        T.voicing("C", "sus2")


def test_quality_degrees_reports_what_a_quality_actually_carries() -> None:
    assert T.quality_degrees("min") == (1, 3, 5)
    assert T.quality_degrees("sus2") == (1, 2, 5)
    assert T.quality_degrees("sus4") == (1, 4, 5)


def test_more_than_four_voices_leaves_the_measured_material() -> None:
    with pytest.raises(ValueError, match="measured are 2, 3 or 4 voices"):
        T.voicing("C", "min", voices=5)


# --------------------------------------------------------------------------- #
# Frequency: the lower bound is the whole point
# --------------------------------------------------------------------------- #


def test_hz_to_norm_reproduces_the_two_measured_points() -> None:
    for hz, measured in T.MEASURED_FILTER_POINTS:
        assert T.hz_to_norm(hz) == pytest.approx(measured, abs=1e-4)


def test_hz_to_norm_round_trips_and_clamps_rather_than_extrapolating() -> None:
    for hz in (20.0, 100.0, 1000.0, 20000.0):
        assert T.norm_to_hz(T.hz_to_norm(hz)) == pytest.approx(hz)

    assert T.hz_to_norm(5.0) == 0.0, "below the range, clamped and not extrapolated"
    assert T.hz_to_norm(50000.0) == 1.0
    assert T.norm_to_hz(0.5) == pytest.approx(632.5, abs=0.1)


def test_a_thirty_hertz_floor_would_be_wrong_at_the_bottom_and_invisible_at_the_top() -> None:
    """Check the formula at the bottom: at the top a wrong floor hides."""
    top_gap = abs(T.hz_to_norm(18939.0) - T.hz_to_norm(18939.0, lo=30.0))
    bottom_gap = abs(T.hz_to_norm(100.0) - T.hz_to_norm(100.0, lo=30.0))

    assert top_gap < 0.001, "0.9921 against 0.99162: nothing to see"
    assert bottom_gap > 0.04
    assert bottom_gap > 40 * top_gap

    # And this is what that costs: a value computed on 30 Hz, read on 20 Hz.
    misread = T.norm_to_hz(T.hz_to_norm(100.0, lo=30.0))
    assert misread == pytest.approx(71.9, abs=0.2), "572 cents off, and silent about it"


# --------------------------------------------------------------------------- #
# humanize: ticks, and the 480 trap
# --------------------------------------------------------------------------- #


def test_the_tick_grid_is_480_per_quarter() -> None:
    assert H.TICKS_PER_QUARTER == 480
    assert H.BAR_TICKS == 1920


@pytest.mark.parametrize(
    ("beats", "ticks"),
    [(0.0, 0), (0.25, 120), (0.5, 240), (0.75, 360), (1.0, 480), (4.0, 1920)],
)
def test_beats_and_ticks_convert_both_ways_without_a_tempo(beats: float, ticks: int) -> None:
    assert H.beats_to_ticks(beats) == ticks
    assert H.ticks_to_beats(ticks) == pytest.approx(beats)


def test_to_ableton_converts_ticks_to_beats() -> None:
    tick_notes = H.from_beats([(0.0, 60, 0.5), (0.5, 62, 0.5), (1.0, 64, 1.0)])
    assert [(n["pos"], n["dur"]) for n in tick_notes] == [(0, 240), (240, 240), (480, 480)]

    live_notes = H.to_ableton(tick_notes, velocity=96)

    assert all(isinstance(n, Note) for n in live_notes)
    assert [(n.pitch, n.start_time, n.duration) for n in live_notes] == [
        (60, 0.0, 0.5),
        (62, 0.5, 0.5),
        (64, 1.0, 1.0),
    ]
    assert [n.velocity for n in live_notes] == [96, 96, 96]


def test_skipping_to_ableton_makes_every_number_480_times_too_large() -> None:
    """A note for beat 1 lands on beat 480: bar 121 of a four-bar clip."""
    tick_note = H.from_beats([(1.0, 60, 1.0)])[0]

    converted = H.to_ableton([tick_note])[0]

    assert converted.start_time == 1.0
    assert tick_note["pos"] == 480
    assert tick_note["pos"] / converted.start_time == 480.0
    assert tick_note["dur"] / converted.duration == 480.0


def test_to_ableton_velocity_is_only_a_fallback() -> None:
    """After velocity_curve() every note has its own field.

    The velocity argument is then inert.
    """
    plain = H.from_beats([(0.0, 60, 1.0)])
    assert H.to_ableton(plain, velocity=42)[0].velocity == 42

    curved = H.velocity_curve(plain, shape="flat", base=88)
    assert H.to_ableton(curved, velocity=42)[0].velocity == 88, "the argument did nothing"


def test_to_ableton_carries_mute_and_the_live_11_extensions() -> None:
    tick_note = {"pos": 0, "dur": 480, "pitch": 60, "mute": True, "probability": 0.6}

    converted = H.to_ableton([tick_note])[0]

    assert converted.mute is True
    assert converted.probability == 0.6
    assert converted.velocity_deviation is None, "an unset extension is not invented"


def test_from_ableton_round_trips_the_tick_form() -> None:
    tick_notes = H.from_beats([(0.0, 60, 0.5), (0.5, 62, 0.5), (1.5, 64, 1.0)])

    back = H.from_ableton(H.to_ableton(tick_notes, velocity=96))

    assert [{k: n[k] for k in ("pos", "dur", "pitch")} for n in back] == tick_notes
    assert all(n["velocity"] == 96 for n in back)


def test_ms_to_ticks_needs_the_tempo_and_the_rule_of_thumb_only_holds_at_126() -> None:
    assert H.ms_to_ticks(350, 126) == 353
    assert H.ms_to_ticks(350, 70) == 196
    assert H.ticks_to_ms(1, 126) == pytest.approx(0.992, abs=0.001)
    assert H.ticks_to_ms(1, 90) == pytest.approx(1.389, abs=0.001)

    with pytest.raises(ValueError, match="bpm must be positive"):
        H.ms_to_ticks(350, 0)


def test_the_tick_form_guard_fires_when_the_order_is_wrong() -> None:
    """velocity_curve() after to_ableton() aborts loudly instead of doing nothing."""
    live_notes = H.to_ableton(H.from_beats([(0.0, 60, 1.0)]))

    with pytest.raises(ValueError) as excinfo:
        H.velocity_curve(live_notes)

    message = str(excinfo.value)
    assert "expects the tick form" in message
    assert "from_ableton()" in message

    with pytest.raises(ValueError, match="expects the tick form"):
        H.accent(live_notes, H.ACCENT_1_AND_3)


# --------------------------------------------------------------------------- #
# humanize: determinism
# --------------------------------------------------------------------------- #


def voices_for(count: int = 3) -> list[list[H.TickNote]]:
    line = [(i * 0.5, 60 + (i % 5), 0.5) for i in range(16)]
    return [H.from_beats(line) for _ in range(count)]


def test_the_same_seed_gives_the_same_scatter_twice() -> None:
    first = H.stagger_voices(voices_for(), seed=42, bpm=126)
    second = H.stagger_voices(voices_for(), seed=42, bpm=126)

    assert first == second


def test_a_different_seed_gives_a_different_scatter() -> None:
    assert H.stagger_voices(voices_for(), seed=42, bpm=126) != H.stagger_voices(
        voices_for(), seed=43, bpm=126
    )


def test_nothing_here_touches_the_global_random_state() -> None:
    """A build script that runs twice must write the same set twice."""
    random.seed(1234)
    expected = random.random()

    random.seed(1234)
    H.stagger_voices(voices_for(), seed=5, bpm=126)
    assert random.random() == expected


def test_voice_zero_stays_on_the_grid_as_the_reference() -> None:
    staggered = H.stagger_voices(voices_for(), seed=42, bpm=126, stagger_ms=0.0)
    plain = H.from_beats([(i * 0.5, 60 + (i % 5), 0.5) for i in range(16)])

    assert [n["pos"] for n in staggered[0]] == [n["pos"] for n in plain]
    assert [n["pos"] for n in staggered[1]] != [n["pos"] for n in plain]


def test_the_jitter_stays_inside_the_measured_cap() -> None:
    """The real example scattered with sigma 15 ms but only reached -20..+19 ms."""
    reference = H.stagger_voices(voices_for(4), seed=9, bpm=126, stagger_ms=0.0)
    limit = round(1.35 * H.ms_to_ticks(H.DEFAULT_JITTER_MS, 126))

    for voice in reference[1:]:
        for moved, straight in zip(voice, reference[0], strict=True):
            assert abs(moved["pos"] - straight["pos"]) <= limit


def test_the_deterministic_timing_operations_take_no_seed_at_all() -> None:
    line = H.from_beats([(0.0, 60, 0.5), (0.5, 62, 0.5), (1.0, 64, 0.5)])

    assert H.swing(line) == H.swing(line)
    assert H.micro_timing(line, 20) == H.micro_timing(line, 20)
    assert H.humanize(line, phrase_starts={0}, bpm=126) == H.humanize(
        line, phrase_starts={0}, bpm=126
    )


# --------------------------------------------------------------------------- #
# humanize: never leave a note where Live cannot draw it
# --------------------------------------------------------------------------- #


def test_micro_timing_and_stagger_clamp_at_the_clip_start() -> None:
    """Clip-local time begins at beat 0, so a pushed line cannot run past it."""
    pushed = H.micro_timing(H.from_beats([(0.0, 60, 1.0), (0.25, 62, 1.0)]), -5000)
    assert all(n["pos"] >= 0 for n in pushed)

    scattered = H.stagger_voices(voices_for(4), seed=3, bpm=126, jitter_ms=400.0)
    assert all(n["pos"] >= 0 for voice in scattered for n in voice)


def test_humanize_leaves_positive_positions_alone_and_keeps_durations_positive() -> None:
    line = H.from_beats([(0.0, 60, 0.5), (0.5, 62, 0.5), (1.0, 64, 0.5), (3.0, 65, 0.5)])

    out = H.humanize(line, phrase_starts={0, 3}, bpm=126)

    assert all(n["pos"] >= 0 for n in out)
    assert all(n["dur"] >= 1 for n in out)
    assert len(out) == len(line)


def test_a_negative_shift_is_the_callers_business_and_the_write_gate_catches_it() -> None:
    """humanize() does not clamp a negative shift: its docstring says not to use one.

    The guard is downstream and it is loud: the value never reaches Live,
    because :meth:`Note.validate` refuses a negative ``start_time``.
    """
    line = H.from_beats([(0.0, 60, 1.0)])

    shifted = H.humanize(line, bpm=126, shift_ticks=-480)
    assert shifted[0]["pos"] == -480

    with pytest.raises(ValueError, match=re.escape("start_time -1.0 is negative")):
        H.to_ableton(shifted)[0].validate()


# --------------------------------------------------------------------------- #
# humanize: velocity never leaves 1..127
# --------------------------------------------------------------------------- #


def test_the_velocity_clamp_floor_is_one_because_zero_is_a_note_off() -> None:
    assert H.VELOCITY_RANGE == (1, 127)
    assert H.VELOCITY_DEFAULT == 100
    assert H.VELOCITY_FULL == 127


@pytest.mark.parametrize("depth", [-500, -200, -20, 0, 20, 200, 500])
def test_velocity_curve_never_leaves_the_range(depth: int) -> None:
    line = H.from_beats([(i * 0.5, 60, 0.5) for i in range(16)])

    for shape in ("flat", "ramp", "arc"):
        out = H.velocity_curve(line, shape=shape, depth=depth, base=100)
        assert all(1 <= n["velocity"] <= 127 for n in out), (shape, depth)


@pytest.mark.parametrize(("amount", "ghost"), [(0, 0), (200, -200), (-200, 200), (40, -40)])
def test_accent_never_leaves_the_range(amount: int, ghost: int) -> None:
    hats = H.pattern_notes(H.GRID_SIXTEENTH, pitch=42, dur=120, velocity=100)

    out = H.accent(hats, H.ACCENT_1_AND_3, amount=amount, ghost=ghost)

    assert all(1 <= n["velocity"] <= 127 for n in out)


def test_velocity_curve_is_flat_by_default_because_the_corpus_is() -> None:
    """78 percent of the measured tracks carry one velocity across all their clips."""
    line = H.from_beats([(i * 0.5, 60, 0.5) for i in range(8)])

    out = H.velocity_curve(line)

    assert {n["velocity"] for n in out} == {H.VELOCITY_DEFAULT}


def test_velocity_curve_ramp_and_blocks_produce_the_documented_levels() -> None:
    line = H.from_beats([(0.0, 60, 1.0), (1.0, 60, 1.0), (2.0, 60, 1.0)])

    assert [n["velocity"] for n in H.velocity_curve(line, shape="ramp", depth=-20)] == [
        100, 90, 80,
    ]
    blocks = H.VELOCITY_BLOCKS_MEASURED["chord_melody_001"]
    assert [n["velocity"] for n in H.velocity_curve(line, blocks=blocks)] == [99, 99, 99]


def test_accent_adds_rather_than_sets_so_a_curve_survives_it() -> None:
    hats = H.pattern_notes(H.GRID_EIGHTH, pitch=42, dur=240, velocity=100)

    out = H.accent(hats, H.ACCENT_1_AND_3, amount=10, ghost=-20)

    assert [n["velocity"] for n in out] == [110, 80, 80, 80, 110, 80, 80, 80]


def test_accent_compares_bar_local_positions_across_the_bar_line() -> None:
    two_bars = H.pattern_notes(H.GRID_QUARTER, pitch=42, dur=480, velocity=100, bars=2)

    out = H.accent(two_bars, (0,), amount=20)

    assert [n["velocity"] for n in out] == [120, 100, 100, 100, 120, 100, 100, 100]


def test_velocity_curve_and_accent_reject_impossible_arguments() -> None:
    line = H.from_beats([(0.0, 60, 1.0)])

    with pytest.raises(ValueError, match="shape belongs to"):
        H.velocity_curve(line, shape="sawtooth")
    with pytest.raises(ValueError, match="steps must not be negative"):
        H.velocity_curve(line, steps=-1)
    with pytest.raises(ValueError, match="bar_ticks must be positive"):
        H.accent(line, (0,), bar_ticks=0)
    with pytest.raises(ValueError, match="tol must not be negative"):
        H.accent(line, (0,), tol=-1)


def test_to_ableton_does_not_clamp_a_hand_written_velocity_and_the_model_refuses_it() -> None:
    """The module's own values are already clamped; a hand-written field is not."""
    hand_written = [{"pos": 0, "dur": 480, "pitch": 60, "velocity": 200}]

    converted = H.to_ableton(hand_written)[0]

    assert converted.velocity == 200
    with pytest.raises(ValueError, match=re.escape("velocity 200 is outside 1..127")):
        converted.validate()


# --------------------------------------------------------------------------- #
# humanize: the timing operations themselves
# --------------------------------------------------------------------------- #


def test_swing_delays_the_offbeat_and_lengthens_its_predecessor() -> None:
    pair = H.from_beats([(0.0, 60, 0.5), (0.5, 62, 0.5)])

    swung = H.swing(pair, ratio=H.SWING_TRIPLET)

    assert [(n["pos"], n["dur"]) for n in swung] == [(0, 320), (320, 160)]
    assert swung[0]["dur"] + swung[1]["dur"] == 480, "the pair still fills its beat"


def test_swing_of_zero_is_a_copy_not_a_move() -> None:
    pair = H.from_beats([(0.0, 60, 0.5), (0.5, 62, 0.5)])

    straight = H.swing(pair, ratio=0.0)

    assert straight == pair
    assert straight is not pair


def test_swing_rejects_a_ratio_past_the_next_downbeat() -> None:
    with pytest.raises(ValueError, match="ratio belongs between"):
        H.swing([], ratio=0.6)
    with pytest.raises(ValueError, match="grid must be positive"):
        H.swing([], grid=0)


def test_micro_timing_pushes_the_phrase_start_hardest() -> None:
    line = H.from_beats([(0.0, 60, 0.5), (0.5, 62, 0.5), (1.0, 64, 0.5)])

    laid_back = H.micro_timing(line, 20, phrase_starts={0}, tail_weight=0.4)

    assert [n["pos"] for n in laid_back] == [20, 248, 488]


def test_extra_fields_ride_through_every_timing_operation() -> None:
    line = [{**n, "velocity": 42, "lyric": "ah"} for n in H.from_beats([(0.0, 60, 0.5),
                                                                       (0.5, 62, 0.5)])]

    for out in (
        H.humanize(line, phrase_starts={0}, bpm=126),
        H.swing(line),
        H.micro_timing(line, 10),
    ):
        assert all(n["velocity"] == 42 for n in out)
        assert all(n["lyric"] == "ah" for n in out)


def test_pattern_notes_builds_a_bar_local_grid() -> None:
    hats = H.pattern_notes(H.GRID_SIXTEENTH, pitch=42, dur=120, bars=4)
    assert len(hats) == 64
    assert hats[0]["pos"] == 0
    assert hats[-1]["pos"] == 4 * H.BAR_TICKS - 120

    three_three_two = H.pattern_notes(H.PATTERN_3_3_2, pitch=73, dur=240)
    assert [(n["pos"], n["dur"]) for n in three_three_two][:3] == [(0, 240), (360, 240), (720, 240)]


def test_pattern_notes_ties_through_when_no_duration_is_given() -> None:
    quarters = H.pattern_notes(H.GRID_QUARTER, pitch=42)
    assert all(n["dur"] == 480 for n in quarters)


def test_pattern_notes_refuses_positions_outside_the_bar() -> None:
    with pytest.raises(ValueError, match="positions are bar-local"):
        H.pattern_notes((0, 2000), pitch=42)
    with pytest.raises(ValueError, match="bars must be at least 1"):
        H.pattern_notes((0,), pitch=42, bars=0)


def test_velocity_report_says_single_level_when_it_is() -> None:
    hats = H.pattern_notes(H.GRID_QUARTER, pitch=42, velocity=100)

    text = H.velocity_report(hats)
    assert "4/4 notes carry one" in text
    assert "single-level" in text

    against_corpus = H.velocity_report(hats, role="Drums")
    assert "corpus Drums" in against_corpus

    with pytest.raises(ValueError, match="was not measured"):
        H.velocity_report(hats, role="Kazoo")


def test_velocity_report_names_the_missing_field_rather_than_inventing_one() -> None:
    text = H.velocity_report(H.from_beats([(0.0, 60, 1.0)]))
    assert "0 of 1 notes carry one" in text
    assert "fallback" in text


def test_report_counts_the_gaps_legato_closed() -> None:
    line = H.from_beats([(0.0, 60, 0.25), (1.0, 62, 0.25), (2.0, 64, 0.25)])

    closed = H.humanize(line, bpm=126)

    assert "gaps before 2, after 0" in H.report(line, closed)


# --------------------------------------------------------------------------- #
# Enharmonics, extended qualities, modes, and inversion checks
# --------------------------------------------------------------------------- #


def test_enharmonics_are_resolved_in_name_to_pc() -> None:
    assert T.NAME_TO_PC["Cb"] == 11
    assert T.NAME_TO_PC["Fb"] == 4
    assert T.NAME_TO_PC["B#"] == 0
    assert T.NAME_TO_PC["E#"] == 5
    assert T.build_chord("Cb", "maj", 4) == [71, 75, 78]


def test_extended_chord_qualities_build_and_identify() -> None:
    expected_intervals = {
        "7sus4": (0, 5, 7, 10),
        "add9": (0, 4, 7, 14),
        "madd9": (0, 3, 7, 14),
        "6_9": (0, 4, 7, 9, 14),
        "7b9": (0, 4, 7, 10, 13),
        "7#9": (0, 4, 7, 10, 15),
        "7#11": (0, 4, 7, 10, 18),
        "min11": (0, 3, 7, 10, 14, 17),
        "maj11": (0, 4, 7, 11, 14, 17),
    }
    for quality, ivs in expected_intervals.items():
        chord = T.build_chord("C", quality, octave=4)
        assert tuple(p - 60 for p in chord) == ivs
        found = T.identify_chord(chord)
        assert found is not None
        assert found[0] == "C"
        assert found[1] == quality


def test_new_modes_and_scales_are_accessible() -> None:
    # E Phrygian Dominant (5th mode of A harmonic minor)
    e_phryg_dom = T.pitch_classes("E", "phrygian_dominant")
    assert sorted(e_phryg_dom) == sorted(T.pitch_classes("A", "harmonic_minor"))

    # C Lydian Dominant: C D E F# G A Bb
    c_lyd_dom = T.pitch_classes("C", "lydian_dominant")
    assert sorted(c_lyd_dom) == [0, 2, 4, 6, 7, 9, 10]

    # C Altered (Super Locrian): C Db Eb Fb/E Gb Ab Bb -> 0, 1, 3, 4, 6, 8, 10
    c_alt = T.pitch_classes("C", "altered")
    assert sorted(c_alt) == [0, 1, 3, 4, 6, 8, 10]

    # 8-note scales
    assert len(T.pitch_classes("C", "diminished_hw")) == 8
    assert len(T.pitch_classes("C", "diminished_wh")) == 8
    assert len(T.pitch_classes("C", "bebop_dominant")) == 8


def test_check_against_chords_tolerates_extensions_on_inverted_chords() -> None:
    # C major in 1st inversion: [E4, G4, C5]
    inverted_c_major = [64, 67, 72]

    # Melody plays D5 (MIDI 74), which is the 9th above the root C
    melody = [(0.0, 74)]

    # With extension tolerance enabled, D5 must be accepted as the 9th of C
    findings = T.check_against_chords(melody, [inverted_c_major], tolerate_extensions=True)
    assert not findings, f"Expected clean check for 9th above C in inversion, got: {findings}"

    # Strict check (no extensions): D5 is not a chord tone of triad C-E-G
    strict_findings = T.check_against_chords(melody, [inverted_c_major], tolerate_extensions=False)
    assert len(strict_findings) == 1
    assert "D5 does not fit the chord" in strict_findings[0]

