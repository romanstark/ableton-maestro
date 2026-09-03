"""Music theory calculations and voicing models.

Scales, degrees, chords and progressions. Pure logic over standard data
structures (no network, socket, or LOM dependencies). Covers scales (modes,
harmonic/melodic minor, pentatonics, blues), chord qualities, diatonic harmony,
voicing shapes, voice-leading rules, and frequency conversions.

Key principles:
- Diatonic voice construction: harmony voices must follow scale intervals rather than
  constant semitone offsets (e.g. major vs. minor thirds).
- Voicing models: provides empirical voicing shapes, degree pool distributions, and
  voice-leading progression generators with register and pitch-range constraints.

Examples:
>>> note_name(69)
'A4'
>>> voicing("Eb", "min")
[51, 58, 66]
>>> [c["chord"] for c in progression("Bb", form="001")][:3]
['Bbm', 'Ab', 'Bbm']
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

NOTE_NAMES: list[str] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

#: Pitch classes written with flat accidentals for flat keys (e.g. Ab in Bb minor).
FLAT_NAMES: list[str] = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

NAME_TO_PC: dict[str, int] = {n: i for i, n in enumerate(NOTE_NAMES)}
NAME_TO_PC.update({
    "Db": 1,
    "Eb": 3,
    "Gb": 6,
    "Ab": 8,
    "Bb": 10,
    "H": 11,
    "Cb": 11,
    "Fb": 4,
    "B#": 0,
    "E#": 5,
})

MODES: dict[str, tuple[int, ...]] = {
    # Church modes
    "major": (0, 2, 4, 5, 7, 9, 11),
    "ionian": (0, 2, 4, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    # Minor variants: the leading tones natural minor does not have
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor": (0, 2, 3, 5, 7, 9, 11),
    # Additional modes and exotic scales
    "phrygian_dominant": (0, 1, 4, 5, 7, 8, 10),
    "lydian_dominant": (0, 2, 4, 6, 7, 9, 10),
    "altered": (0, 1, 3, 4, 6, 8, 10),
    "super_locrian": (0, 1, 3, 4, 6, 8, 10),
    "harmonic_major": (0, 2, 4, 5, 7, 8, 11),
    "diminished_hw": (0, 1, 3, 4, 6, 7, 9, 10),
    "diminished_wh": (0, 2, 3, 5, 6, 8, 9, 11),
    "bebop_dominant": (0, 2, 4, 5, 7, 9, 10, 11),
    # Five- and six-note scales
    "major_pent": (0, 2, 4, 7, 9),
    "minor_pent": (0, 3, 5, 7, 10),
    "blues": (0, 3, 5, 6, 7, 10),
    "whole_tone": (0, 2, 4, 6, 8, 10),
}

#: Chord qualities as semitone distances from the root.
CHORD_QUALITIES: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "7sus4": (0, 5, 7, 10),
    "maj6": (0, 4, 7, 9),
    "min6": (0, 3, 7, 9),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "dom7": (0, 4, 7, 10),
    "min7b5": (0, 3, 6, 10),
    "dim7": (0, 3, 6, 9),
    "minmaj7": (0, 3, 7, 11),
    "add9": (0, 4, 7, 14),
    "madd9": (0, 3, 7, 14),
    "6_9": (0, 4, 7, 9, 14),
    "maj9": (0, 4, 7, 11, 14),
    "min9": (0, 3, 7, 10, 14),
    "dom9": (0, 4, 7, 10, 14),
    "7b9": (0, 4, 7, 10, 13),
    "7#9": (0, 4, 7, 10, 15),
    "7#11": (0, 4, 7, 10, 18),
    "min11": (0, 3, 7, 10, 14, 17),
    "maj11": (0, 4, 7, 11, 14, 17),
    "dom13": (0, 4, 7, 10, 14, 21),
}

_CHORD_SYMBOL: dict[str, str] = {
    "maj": "",
    "min": "m",
    "dim": "dim",
    "aug": "aug",
    "sus2": "sus2",
    "sus4": "sus4",
    "7sus4": "7sus4",
    "maj6": "6",
    "min6": "m6",
    "maj7": "maj7",
    "min7": "m7",
    "dom7": "7",
    "min7b5": "m7b5",
    "dim7": "dim7",
    "minmaj7": "mMaj7",
    "add9": "add9",
    "madd9": "madd9",
    "6_9": "6/9",
    "maj9": "maj9",
    "min9": "m9",
    "dom9": "9",
    "7b9": "7b9",
    "7#9": "7#9",
    "7#11": "7#11",
    "min11": "m11",
    "maj11": "maj11",
    "dom13": "13",
}


# --------------------------------------------------------------------------- #
# Names, scales, chords: calculations, true everywhere
# --------------------------------------------------------------------------- #


def note_name(pitch: int) -> str:
    """Return MIDI note number formatted as scientific pitch name (e.g. 69 -> 'A4').

    Assumes C4 = 60.

    Args:
        pitch: MIDI note number (0 to 127).

    Returns:
        Formatted note string with octave number.
    """
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def pc_name(pc: int, flats: bool = False) -> str:
    """Return pitch class name for an integer pitch class (0 to 11).

    Args:
        pc: Integer pitch class modulo 12.
        flats: When True, uses flat accidentals (e.g. Db rather than C#).

    Returns:
        Pitch class name string.
    """
    return (FLAT_NAMES if flats else NOTE_NAMES)[pc % 12]


def chord_symbol(root: str, quality: str) -> str:
    """Return readable chord symbol string (e.g. ('Bb', 'min') -> 'Bbm').

    Args:
        root: Root note name.
        quality: Chord quality key from CHORD_QUALITIES.

    Returns:
        Formatted chord symbol string.
    """
    return f"{root}{_CHORD_SYMBOL.get(quality, quality)}"


def bar_beats(time_signature: str) -> float:
    """Return beats per measure in quarter notes for a time signature string.

    Args:
        time_signature: Time signature formatted as 'num/den' (e.g. '4/4', '6/8').

    Returns:
        Duration of one bar in quarter-note beats.
    """
    num, den = time_signature.split("/")
    return int(num) * 4.0 / int(den)


def pitch_classes(root: str, mode: str = "major") -> set[int]:
    """Return pitch classes (0 to 11) for a scale defined by root and mode.

    Args:
        root: Tonic note name.
        mode: Mode name from MODES dictionary.

    Returns:
        Set of pitch classes in the scale.

    Raises:
        KeyError: If mode or root name is unrecognized.
    """
    if mode not in MODES:
        raise KeyError(f"unknown mode {mode!r}. Known: {', '.join(sorted(MODES))}")
    base = NAME_TO_PC[root]
    return {(base + iv) % 12 for iv in MODES[mode]}


def scale_pitches(root: str, mode: str = "major", low: int = 36, high: int = 96) -> list[int]:
    """Return ascending MIDI pitches within bounds belonging to the specified scale.

    Args:
        root: Tonic note name.
        mode: Mode name from MODES dictionary.
        low: Minimum MIDI note number (inclusive).
        high: Maximum MIDI note number (inclusive).

    Returns:
        List of MIDI pitches in ascending order.
    """
    pcs = pitch_classes(root, mode)
    return [p for p in range(low, high + 1) if p % 12 in pcs]


def build_chord(root: str, quality: str = "maj", octave: int = 3, inversion: int = 0) -> list[int]:
    """Build a chord as a list of MIDI note numbers stacked in thirds.

    Args:
        root: Chord root note name.
        quality: Chord quality name from CHORD_QUALITIES.
        octave: Target root octave.
        inversion: Inversion index (0 = root position, 1 = first inversion, etc.).

    Returns:
        List of MIDI note numbers in ascending pitch order.

    Raises:
        KeyError: If chord quality is unrecognized.
    """
    if quality not in CHORD_QUALITIES:
        known = ", ".join(sorted(CHORD_QUALITIES))
        raise KeyError(f"unknown chord quality {quality!r}. Known: {known}")
    base = NAME_TO_PC[root] + 12 * (octave + 1)
    pitches = [base + iv for iv in CHORD_QUALITIES[quality]]
    for _ in range(inversion % max(1, len(pitches))):
        pitches = [*pitches[1:], pitches[0] + 12]
    return pitches


def transpose_diatonic(
    pitch: int,
    steps: int,
    root: str,
    mode: str = "major",
    floor: int | None = 55,
) -> int:
    """Transpose a MIDI pitch by scale steps within a key.

    Args:
        pitch: Input MIDI pitch.
        steps: Number of scale degrees to shift (positive for up, negative for down).
        root: Key root note name.
        mode: Mode name from MODES dictionary.
        floor: Optional minimum pitch threshold (defaults to 55 = G3). Pitches falling
            below floor are transposed up one octave. Pass None to disable.

    Returns:
        Transposed MIDI pitch.
    """
    scale = scale_pitches(root, mode, low=12, high=120)
    if not scale:
        return pitch

    if pitch in scale:
        index = scale.index(pitch)
    else:
        index = min(range(len(scale)), key=lambda i: (abs(scale[i] - pitch), i))

    target = scale[max(0, min(len(scale) - 1, index + steps))]
    if floor is not None:
        while target < floor:
            target += 12
    return target


def harmony_line(
    pitches: Sequence[int],
    steps: int,
    root: str,
    mode: str = "major",
    floor: int | None = 55,
) -> list[int]:
    """Transpose a sequence of melody pitches diatonically by scale degrees.

    Args:
        pitches: Sequence of MIDI note numbers.
        steps: Scale degrees to shift.
        root: Key root note name.
        mode: Mode name.
        floor: Optional minimum pitch threshold.

    Returns:
        List of harmonized MIDI note numbers.
    """
    return [transpose_diatonic(p, steps, root, mode, floor) for p in pitches]


def check_in_key(pitches: Iterable[int], root: str, mode: str = "major") -> list[tuple[int, str]]:
    """Identify notes in a sequence that do not belong to the specified key.

    Args:
        pitches: Iterable of MIDI pitches.
        root: Key root note name.
        mode: Mode name.

    Returns:
        List of (index, note_name) tuples for out-of-key notes.
    """
    pcs = pitch_classes(root, mode)
    return [(i, note_name(p)) for i, p in enumerate(pitches) if p % 12 not in pcs]


def check_against_chords(
    notes: Sequence[tuple[float, int]],
    chords: Sequence[Sequence[int]],
    beats_per_bar: float = 4.0,
    tolerate_extensions: bool = True,
    schedule: Sequence[tuple[float, Sequence[int]]] | None = None,
) -> list[str]:
    """Verify that melody notes align harmonically with chords at onset time.

    Args:
        notes: Sequence of (start_beat, pitch) tuples with timeline beat coordinates.
        chords: Sequence of chord pitch lists mapped cyclically per bar.
        beats_per_bar: Duration of one measure in quarter notes.
        tolerate_extensions: When True, permits 9th, 11th, and 13th extensions.
        schedule: Optional sequence of (start_beat, chord_pitches) for non-standard chord changes.

    Returns:
        List of descriptive strings for dissonant or misaligned notes.
    """
    findings: list[str] = []
    sched = sorted(schedule, key=lambda x: x[0]) if schedule else None

    for beat, pitch in notes:
        bar = int(beat // beats_per_bar)
        if sched:
            chord: Sequence[int] = sched[0][1]
            for start, candidate in sched:
                if start <= beat:
                    chord = candidate
                else:
                    break
        else:
            chord = chords[bar % len(chords)]

        chord_pcs = {c % 12 for c in chord}
        if tolerate_extensions and chord:
            identified = identify_chord(chord)
            if identified is not None:
                root_pc = NAME_TO_PC[identified[0]]
            else:
                root_pc = min(chord) % 12
            chord_pcs |= {(root_pc + iv) % 12 for iv in (2, 5, 9)}  # 9, 11, 13/6
        if pitch % 12 not in chord_pcs:
            findings.append(
                f"bar {bar + 1}, beat {beat % beats_per_bar:.2f}: "
                f"{note_name(pitch)} does not fit the chord"
            )
    return findings


def check_sections(sections: Sequence[tuple[str, int, int]], total_bars: int) -> list[str]:
    """Validate section boundaries for continuity, gaps, overlaps, and length.

    Args:
        sections: Sequence of (section_name, start_bar, length_in_bars) tuples (1-indexed bars).
        total_bars: Expected total project length in bars.

    Returns:
        List of warning strings detailing detected gaps, overlaps, or length discrepancies.
    """
    findings: list[str] = []
    cursor = 1
    for name, start, length in sorted(sections, key=lambda s: s[1]):
        if start > cursor:
            findings.append(f"gap: bars {cursor}-{start - 1} before {name!r}")
        elif start < cursor:
            findings.append(f"overlap: {name!r} starts at {start}, expected {cursor}")
        cursor = start + length
    if cursor - 1 != total_bars:
        findings.append(f"total length {cursor - 1} bars, expected {total_bars}")
    return findings


# --------------------------------------------------------------------------- #
# Voicing models and empirical degree pools
#
# Voicing shapes and degree distributions derived from empirical analysis of
# chord-melody structures across reference templates: 896 notes, 328 attacks,
# 65 harmony blocks, and 221 multi-voice attacks.
#
# Key structural patterns observed:
# - Low interval spacing: fifths, fourths, or octaves at the base; major seconds avoided.
# - Voice distribution: melody on chord thirds, 3-voice texture in middle register.
# - Single-voicing consistency: progressions typically transpose a unified voicing shape.
#
# Notes on pitch representations:
# - MIDI numbers are used throughout for unambiguous register indexing.
# - Frequency calculations adhere to standard logarithmic scaling.
# --------------------------------------------------------------------------- #

#: Lowest interval (bass to next voice) in semitones -> attack counts across 221 attacks.
VOICING_BOTTOM_COUNTS: dict[int, int] = {7: 82, 5: 65, 12: 34, 4: 20, 3: 12, 10: 5, 2: 3}

#: Prevalent lowest intervals (fifth, fourth, octave), representing 81.9% of attacks.
VOICING_BOTTOM_GOOD: tuple[int, ...] = (5, 7, 12)

#: Which chord degree the TOP note takes, in percent of 88 block placements.
#: Third together 58.0 (major 33.0 + minor 25.0), root 5.7. The single most
#: usable number of the measurement: put the melody on the third, not on the
#: root.
VOICING_TOP_DEGREE_PCT: dict[str, float] = {
    "3": 58.0,
    "5": 22.7,
    "9": 9.1,
    "1": 5.7,
    "11": 3.4,
    "#11": 1.1,
}

#: Window of the top voice, MIDI: 61..77 in 6 of 7 files. The seventh (001)
#: sits lower, 53..61.
VOICING_TOP_RANGE: tuple[int, int] = (61, 77)

#: Typical lowest note (the bass of the voicing), MIDI 46..58. The absolute
#: lowest note of the whole sample is 42, in exactly one file.
#: These voicings want a bass UNDER them. They are not one.
VOICING_BASS_RANGE: tuple[int, int] = (46, 58)
VOICING_BASS_ABSOLUTE_LOW: int = 42

#: Voice count: median 3, never more than 4. Per file 2 (001), 3 (005, 006,
#: 007), 4 (003, 004), mixed (002).
VOICING_MAX_VOICES: int = 4

#: Inversion -> percent of 65 harmony blocks. Root position is the normal case;
#: 004 is the only file standing throughout in second inversion, and once the
#: bass is not a chord tone at all.
VOICING_INVERSION_PCT: dict[int, float] = {0: 73.8, 2: 15.4, 1: 9.2}

#: Largest leap of the top voice across all seven files: 8 semitones (a minor
#: sixth, once, in 002). No octave leap anywhere.
VOICING_TOP_MAX_LEAP: int = 8

#: Semitone distances a chord degree may occupy. This is what keeps a degree
#: from ever being built with a fixed semitone value: it is always taken from
#: the actual chord quality, which is the point of this whole file.
_DEGREE_INTERVALS: dict[int, tuple[int, ...]] = {
    1: (0,),
    2: (1, 2),
    3: (3, 4),
    4: (5,),
    5: (6, 7, 8),
    6: (9,),
    7: (10, 11),
    9: (13, 14, 15),
    11: (17, 18),
    13: (20, 21, 22),
}


@dataclass(frozen=True)
class VoicingShape:
    """Representation of an empirical chord voicing shape.

    Attributes:
        degrees: Sequence of (chord_degree, octave_offset) relative to root.
        bottom: Description of lowest interval.
        evidence: Data source reference.
        note: Structural observations and usage guidelines.
    """

    degrees: tuple[tuple[int, int], ...]
    bottom: str
    evidence: str
    note: str = ""


#: Voicing shapes catalogued from empirical chord-melody reference templates.
VOICING_SHAPES: dict[str, VoicingShape] = {
    "1-3": VoicingShape(
        degrees=((1, 0), (3, 0)),
        bottom="third (26x minor, 28x major)",
        evidence="001, 56 attacks, dyads in parallel thirds",
        note=(
            "Two voices. Dyads do not fully specify chord quality; "
            "top voice in template 001 sits in a lower window (53..61) than other templates."
        ),
    ),
    "1-1oct-3oct": VoicingShape(
        degrees=((1, 0), (1, 1), (3, 1)),
        bottom="octave (12 st), 32x",
        evidence=(
            "002; the variant with fifth on top (1-1oct-5oct) is also attested. "
            "Third is voiced exclusively through the top voice."
        ),
    ),
    "1-5-3oct": VoicingShape(
        degrees=((1, 0), (5, 0), (3, 1)),
        bottom="fifth (7 st)",
        evidence="005 (46 of 46 attacks with a fifth at the bottom), 006 (36x)",
        note="Open fifth at the bottom, third an octave above bass. Standard pluck voicing.",
    ),
    "1-5-1oct-3oct": VoicingShape(
        degrees=((1, 0), (5, 0), (1, 1), (3, 1)),
        bottom="fifth (7 st)",
        evidence="005, Bbm block [46, 53, 58, 61]",
    ),
    "1-3-5-3oct": VoicingShape(
        degrees=((1, 0), (3, 0), (5, 0), (3, 1)),
        bottom="third (12x minor, 12x major, 1x fifth)",
        evidence="003, melody over full triads (chord-tone quota 69.2 percent)",
        note=(
            "Close third structure at the bottom with melody an octave above. "
            "Too close by the letter of rule 1 but attested 24 times, which is why "
            "check_voicing() reports a third at the bottom as a note rather than a "
            "rule violation."
        ),
    ),
    "5-1-5-3": VoicingShape(
        degrees=((5, -1), (1, 0), (5, 0), (3, 1)),
        bottom="fourth (5 st), 48 of 48",
        evidence="004, all six blocks identically transposed, second inversion throughout",
        note=(
            "Fifth in bass, root a fourth above, root octave, third on top. "
            "Consistent second-inversion transposition."
        ),
    ),
    "5-1-3": VoicingShape(
        degrees=((5, -1), (1, 0), (3, 0)),
        bottom="fourth (5 st), 12x",
        evidence="007, close three-voice position between MIDI 52 and 69",
    ),
    "1-5-2oct": VoicingShape(
        degrees=((1, 0), (5, 0), (2, 1)),
        bottom="fifth (7 st), derived from shape",
        evidence=(
            "Derived from reference audio recording (100-220 Hz band). Alternating "
            "triads with low correlation against major/minor triads indicate a "
            "thirdless sound (a-e-b pitch classes)."
        ),
        note=(
            "Root, fifth, and ninth on top (sus2 quality, 0, 2, 7). "
            "Satisfies rule 1 (fifth at base) and rule 2 (ninth on top)."
        ),
    ),
}

#: Default shape per voice count (2: "1-3", 3: "1-5-3oct", 4: "5-1-5-3").
VOICING_DEFAULT_SHAPE: dict[int, str] = {2: "1-3", 3: "1-5-3oct", 4: "5-1-5-3"}

#: Standard minor degree pool: diatonic minor degrees without the second degree.
MEASURED_DEGREE_POOL: tuple[str, ...] = ("i", "III", "iv", "v", "VI", "VII")

#: Degree pool represented as semitone distances from minor tonic.
_POOL_OFFSETS = frozenset((0, 3, 5, 7, 8, 10))


@dataclass(frozen=True)
class MeasuredProgression:
    """Reference chord progression example.

    Attributes:
        degrees: Diatonically normalized scale degree sequence.
        mode: Key mode name.
        key: Key tonic name.
        chords: Sequence of chord symbol strings.
        shape: Voicing shape identifier.
        harmonic_rhythm: Description of chord duration and change rate.
        note: Musical and analytical notes.
        deviation: Known differences between source and computed degrees.
    """

    degrees: tuple[str, ...]
    mode: str
    key: str
    chords: str
    shape: str
    harmonic_rhythm: str
    note: str = ""
    deviation: str = ""


MEASURED_PROGRESSIONS: dict[str, MeasuredProgression] = {
    "001": MeasuredProgression(
        degrees=("i", "VII", "i", "VI", "iv", "VI", "VII"),
        mode="minor",
        key="Bb",
        chords="Bbm - Ab - Bbm - Gb - Ebm - Gb - Ab",
        shape="1-3",
        harmonic_rhythm="1 chord per 2 bars; changes 0.5 beats BEFORE the barline (anticipation)",
    ),
    "002A": MeasuredProgression(
        degrees=("iv", "VII", "i", "VI", "v", "i"),
        mode="minor",
        key="F#",
        chords="B - E - F#m - D - C#m - F#",
        shape="1-1oct-3oct",
        harmonic_rhythm="1 chord per 1 to 2 bars, very uneven",
        note=(
            "Shortened from raw sequence (iv-VII-i-VII-VI-v-i). "
            "First and last chords measured thirdless (B with fifth, F# with octave)."
        ),
        deviation=(
            "two chords measured thirdless (B, F#); the recomputation names "
            "their diatonic quality (Bm, F#m)"
        ),
    ),
    "003": MeasuredProgression(
        degrees=("VI", "i", "VII", "v"),
        mode="minor",
        key="Bb",
        chords="Gb - Bbm - Ab - Fm",
        shape="1-3-5-3oct",
        harmonic_rhythm=(
            "1 chord per bar, then one over 2 bars, then an eighth-note turn (v) "
            "as a run-up back to VI"
        ),
    ),
    "004": MeasuredProgression(
        degrees=("vi", "iii", "V", "vi", "iii", "I"),
        mode="major",
        key="B",
        chords="G#m - Ebm - F# | G#m - Ebm - B",
        shape="5-1-5-3",
        harmonic_rhythm="4.25 / 3.50 / 8.00 beats, twice",
        note=(
            "Read as G# minor: i - v - VII | i - v - III. Eight bars (4+4) with open ending on V "
            "and closed on I. Recomputation yields D#m instead of Ebm (enharmonic equivalent)."
        ),
        deviation="spelling only: D#m (computed) vs Ebm (measured)",
    ),
    "005": MeasuredProgression(
        degrees=("i", "VII", "VI", "VII", "v", "VI"),
        mode="minor",
        key="Eb",
        chords="Ebm - Db - B | Db - Bbm - B",
        shape="1-5-3oct",
        harmonic_rhythm="1.0 to 4.0 beats, i.e. up to 2 chords per bar",
    ),
    "006": MeasuredProgression(
        degrees=("III", "iv", "v", "III", "iv", "v", "iv"),
        mode="minor",
        key="G#",
        chords="B - C# - Ebm - Ebsus4 | B - C# - Eb - C#sus2",
        shape="1-5-3oct",
        harmonic_rhythm="exactly 1 chord per bar, 8 bars",
        note=(
            "Accompaniment is thirdless (root and fifth only) with roots on degrees 3, 4, 5. "
            "Realized diatonically as III-iv-v to avoid out-of-key notes."
        ),
        deviation="spelling and the thirdless labels (sus) of the source; roots agree",
    ),
    "007": MeasuredProgression(
        degrees=("IV", "I", "I", "V", "iii", "vi", "iii", "vi"),
        mode="major",
        key="C",
        chords="F - C | C - G | Em - Am | Em - Am",
        shape="5-1-3",
        harmonic_rhythm="2 chords per bar, changing on beat 1.5",
        note=(
            "Equivalent to A minor: VI - III | III - VII | v - i | v - i. "
            "Bass remains stationary while upper voices perform double suspension."
        ),
    ),
}

#: Features absent from the reference progression corpus.
PROGRESSION_ABSENT: tuple[str, ...] = (
    "no V7 in seven files",
    "no V-I cadence",
    "no added leading tone as a melody note",
    "no borrowed degree, zero out-of-key notes",
    "the second degree (ii) occurs in none of the seven progressions",
)
PROGRESSION_ABSENT: tuple[str, ...] = (
    "no V7 in seven files",
    "no V-I cadence",
    "no added leading tone as a melody note",
    "no borrowed degree, zero out-of-key notes",
    "the second degree (ii) occurs in none of the seven progressions",
)


@dataclass(frozen=True)
class MeasuredVoicing:
    """One measured harmony block, kept so :func:`self_check` can reproduce it.

    ``pitches`` are the MIDI numbers as measured in the template named by
    ``source``; ``shape``, ``root`` and ``quality`` are the arguments that must
    make :func:`voicing` return exactly those pitches.
    """

    shape: str
    root: str
    quality: str
    pitches: tuple[int, ...]
    source: str


#: Six measured blocks, each from its own template. :func:`self_check` recomputes
#: them; a deviation means a regression in :func:`voicing`, not a new insight.
MEASURED_VOICINGS: tuple[MeasuredVoicing, ...] = (
    MeasuredVoicing("1-5-3oct", "Eb", "min", (51, 58, 66), "005 Ebm"),
    MeasuredVoicing("1-5-3oct", "B", "maj", (47, 54, 63), "006 B"),
    MeasuredVoicing("5-1-5-3", "G#", "min", (51, 56, 63, 71), "004 G#m"),
    MeasuredVoicing("5-1-5-3", "F#", "maj", (49, 54, 61, 70), "004 F#"),
    MeasuredVoicing("1-3-5-3oct", "Ab", "maj", (56, 60, 63, 72), "003 Ab"),
    MeasuredVoicing("5-1-3", "C", "maj", (55, 60, 64), "007 C"),
)

#: The first three measured harmony blocks of template 005 (Eb minor), MIDI.
#: ``voice_progression(progression("Eb", form="005"), shape="1-5-3oct")`` has to
#: reproduce them; the top voice moves by at most 8 semitones (measured).
MEASURED_005_BLOCKS: tuple[tuple[int, ...], ...] = ((51, 58, 66), (49, 56, 65), (47, 54, 63))

_ROMAN_TO_STEP: dict[str, int] = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7}
_ROMAN_BY_STEP: tuple[str, ...] = ("i", "ii", "iii", "iv", "v", "vi", "vii")


# --------------------------------------------------------------------------- #
# Degrees and progressions
# --------------------------------------------------------------------------- #


def parse_degree(degree: str) -> tuple[int, int, str]:
    """Parse Roman numeral scale degree into degree step, shift, and suffix.

    Args:
        degree: Degree label (e.g. 'iv', 'bIIImaj7', 'V7').

    Returns:
        Tuple of (step_1_to_7, chromatic_semitones, suffix).

    Raises:
        ValueError: If degree label cannot be parsed.
    """
    s = str(degree).strip()
    chroma = 0
    while s[:1] in ("b", "#"):
        chroma += -1 if s[0] == "b" else 1
        s = s[1:]
    i = 0
    while i < len(s) and s[i] in "iIvV":
        i += 1
    roman, suffix = s[:i].lower(), s[i:]
    if roman not in _ROMAN_TO_STEP:
        raise ValueError(
            f"unreadable degree label {degree!r}. Expected i..vii or I..VII, "
            "optionally preceded by b or #, followed by a chord quality from "
            "CHORD_QUALITIES or '7'."
        )
    return _ROMAN_TO_STEP[roman], chroma, suffix


def _match_quality(intervals: tuple[int, ...], degree: str, mode: str) -> str:
    """Find matching chord quality name for an interval tuple."""
    for name, ivs in CHORD_QUALITIES.items():
        if ivs == intervals:
            return name
    raise ValueError(
        f"degree {degree!r} in {mode!r} yields the semitone sequence {intervals}, "
        "which is not in CHORD_QUALITIES. Give the chord quality as a suffix."
    )


def degree_chord(
    key_root: str,
    degree: str,
    mode: str = "minor",
    flats: bool | None = None,
) -> tuple[str, str]:
    """Calculate root note name and chord quality for a scale degree in a key.

    Args:
        key_root: Key tonic note name.
        degree: Scale degree label (e.g. 'i', 'IV', 'VII', 'bIII').
        mode: Mode name (must be a 7-note diatonic mode).
        flats: Optional flag to force flat spelling (defaults based on key_root).

    Returns:
        Tuple of (chord_root_name, chord_quality).

    Raises:
        KeyError: If mode is unknown.
        ValueError: If mode does not have 7 notes or degree suffix is invalid.
    """
    if mode not in MODES:
        raise KeyError(f"unknown mode {mode!r}. Known: {', '.join(sorted(MODES))}")
    scale = MODES[mode]
    if len(scale) != 7:
        raise ValueError(
            f"mode {mode!r} has {len(scale)} notes. Degree harmony needs seven notes."
        )
    step, chroma, suffix = parse_degree(degree)
    if chroma and suffix not in CHORD_QUALITIES:
        raise ValueError(
            f"chromatic degree {degree!r}: no chord quality can be derived from "
            f"the scale for it. Append a quality (e.g. {degree!s}maj) or use build_chord()."
        )
    idx = step - 1
    if flats is None:
        flats = "b" in str(key_root)[1:]
    root_pc = (NAME_TO_PC[key_root] + scale[idx] + chroma) % 12

    t3 = (scale[(idx + 2) % 7] - scale[idx]) % 12
    t5 = (scale[(idx + 4) % 7] - scale[idx]) % 12
    t7 = (scale[(idx + 6) % 7] - scale[idx]) % 12

    if suffix in CHORD_QUALITIES:
        quality = suffix
    elif suffix == "":
        quality = _match_quality((0, t3, t5), degree, mode)
    elif suffix == "7":
        quality = _match_quality((0, t3, t5, t7), degree, mode)
    elif suffix in ("o", "dim"):
        quality = "dim"
    else:
        raise ValueError(
            f"unknown suffix {suffix!r} in {degree!r}. Allowed: '', '7', 'o' or "
            "a key from CHORD_QUALITIES."
        )
    return pc_name(root_pc, flats), quality


def _aeolian_offset(mode: str) -> int | None:
    """Calculate semitone distance from mode tonic to relative Aeolian tonic."""
    if mode not in MODES:
        return None
    target = {p % 12 for p in MODES[mode]}
    if len(target) != 7:
        return None
    for off in range(12):
        if {(off + a) % 12 for a in MODES["aeolian"]} == target:
            return off
    return None


def progression_pool(mode: str = "minor") -> tuple[str, ...]:
    """Return diatonic scale degree pool for a mode.

    Args:
        mode: Mode name (Aeolian rotation).

    Returns:
        Tuple of degree labels.

    Raises:
        ValueError: If mode is not an Aeolian rotation.
    """
    off = _aeolian_offset(mode)
    if off is None:
        raise ValueError(
            f"there is no measured pool for {mode!r}; requires aeolian or its rotations."
        )
    scale = MODES[mode]
    found: list[tuple[int, str]] = []
    for pool_off in sorted(_POOL_OFFSETS):
        pc = (off + pool_off) % 12
        for idx, iv in enumerate(scale):
            if iv % 12 == pc:
                t3 = (scale[(idx + 2) % 7] - iv) % 12
                roman = _ROMAN_BY_STEP[idx]
                found.append((idx, roman if t3 == 3 else roman.upper()))
                break
    return tuple(label for _, label in sorted(found))


def progression(
    key_root: str,
    mode: str | None = None,
    form: str = "001",
    rotate: int = 0,
    degrees: Sequence[str] | None = None,
    length: int | None = None,
    flats: bool | None = None,
) -> list[dict[str, Any]]:
    """Generate chord progression dictionaries for a key and form template.

    Args:
        key_root: Key tonic note name.
        mode: Optional mode override (defaults to template mode).
        form: Progression template key from MEASURED_PROGRESSIONS.
        rotate: Cyclic rotation offset for chords in the progression.
        degrees: Optional custom sequence of degree labels overriding form.
        length: Desired chord count (repeats or truncates progression cyclically).
        flats: Optional flag to force flat spelling.

    Returns:
        List of chord specification dictionaries with degree, root, quality, and symbol.

    Raises:
        KeyError: If form is not recognized.
    """
    if degrees is not None:
        seq = [str(d) for d in degrees]
        mode = mode or "minor"
        source = "caller's own sequence, NOT measured"
    else:
        if form not in MEASURED_PROGRESSIONS:
            raise KeyError(
                f"unknown form {form!r}. Measured are: {', '.join(sorted(MEASURED_PROGRESSIONS))}"
            )
        entry = MEASURED_PROGRESSIONS[form]
        seq = list(entry.degrees)
        mode = mode or entry.mode
        source = f"Chord Melody {form}, measured in {entry.key} {entry.mode}"
    if not seq:
        return []
    if rotate:
        r = rotate % len(seq)
        seq = seq[r:] + seq[:r]
    if length:
        seq = [seq[i % len(seq)] for i in range(int(length))]

    off = _aeolian_offset(mode)
    tonic = NAME_TO_PC[key_root]
    out: list[dict[str, Any]] = []
    for label in seq:
        root, quality = degree_chord(key_root, label, mode, flats=flats)
        in_pool: bool | None = None
        if off is not None:
            in_pool = (NAME_TO_PC[root] - tonic - off) % 12 in _POOL_OFFSETS
        out.append(
            {
                "degree": label,
                "root": root,
                "quality": quality,
                "chord": chord_symbol(root, quality),
                "in_pool": in_pool,
                "source": source,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Voicing: from a chord quality to concrete MIDI numbers
# --------------------------------------------------------------------------- #


def _degree_map(quality: str) -> dict[int, int]:
    """Map pitch class semitone offset to chord degree for a chord quality."""
    if quality not in CHORD_QUALITIES:
        known = ", ".join(sorted(CHORD_QUALITIES))
        raise KeyError(f"unknown chord quality {quality!r}. Known: {known}")
    out: dict[int, int] = {}
    for iv in CHORD_QUALITIES[quality]:
        for deg, allowed in _DEGREE_INTERVALS.items():
            if iv in allowed:
                out.setdefault(iv % 12, deg)
    return out


def _degree_semitone(quality: str, degree: int) -> int | None:
    """Return semitone interval for a specific chord degree within a quality."""
    dmap = _degree_map(quality)
    for iv in CHORD_QUALITIES[quality]:
        if dmap.get(iv % 12) == degree:
            return iv
    return None


def quality_degrees(quality: str) -> tuple[int, ...]:
    """Return tuple of chord degrees present in a chord quality.

    Args:
        quality: Chord quality name from CHORD_QUALITIES.

    Returns:
        Ascending tuple of integer chord degrees (e.g. (1, 3, 5)).

    Raises:
        KeyError: If chord quality is unrecognized.
    """
    return tuple(sorted(set(_degree_map(quality).values())))


def _shape_degrees(shape: str) -> tuple[tuple[int, int], ...]:
    """Return tuple of (degree, octave) pairs defining a voicing shape."""
    if shape not in VOICING_SHAPES:
        raise KeyError(f"unknown shape {shape!r}. Measured are: {', '.join(VOICING_SHAPES)}")
    return VOICING_SHAPES[shape].degrees


def _realize(root_pitch: int, quality: str, degrees: Sequence[tuple[int, int]]) -> list[int]:
    """Realize a voicing shape on a root pitch as sorted MIDI pitches.

    Args:
        root_pitch: Root MIDI pitch number.
        quality: Target chord quality.
        degrees: Sequence of (chord_degree, octave_offset) tuples.

    Returns:
        Sorted list of MIDI pitches.

    Raises:
        ValueError: If chord quality lacks one of the required degrees in the shape.
    """
    pitches: list[int] = []
    for deg, octave in degrees:
        iv = _degree_semitone(quality, deg)
        if iv is None:
            has = ", ".join(str(d) for d in quality_degrees(quality))
            raise ValueError(
                f"chord quality {quality!r} has no degree {deg} required by shape. "
                f"Quality provides degrees: {has}. "
                "Use fitting_shape() to resolve compatible shapes."
            )
        pitches.append(root_pitch + iv + 12 * octave)
    return sorted(pitches)


def _outside(value: int, lo: int, hi: int) -> int:
    """Calculate semitones by which value exceeds lo..hi range bounds (0 if within)."""
    return max(0, lo - value) + max(0, value - hi)


def shape_fits(quality: str, shape: str) -> bool:
    """Check whether a chord quality provides all degrees required by a voicing shape.

    Args:
        quality: Chord quality name.
        shape: Voicing shape identifier.

    Returns:
        True if all required degrees exist in the quality.
    """
    return all(_degree_semitone(quality, deg) is not None for deg, _ in _shape_degrees(shape))


def fitting_shape(quality: str, shape: str) -> str:
    """Return requested shape if compatible, or closest compatible alternative shape.

    Args:
        quality: Target chord quality name.
        shape: Preferred voicing shape identifier.

    Returns:
        Compatible voicing shape identifier.
    """
    if shape_fits(quality, shape):
        return shape
    want = len(_shape_degrees(shape))
    fitting = [name for name in VOICING_SHAPES if shape_fits(quality, name)]
    if not fitting:
        return shape  # _realize() will say which degree is missing
    return min(fitting, key=lambda name: abs(len(_shape_degrees(name)) - want))


def _candidates(
    root: str | int, quality: str, degrees: Sequence[tuple[int, int]]
) -> list[list[int]]:
    """Generate all octave realizations of a voicing shape within valid MIDI range."""
    pc = NAME_TO_PC[root] if isinstance(root, str) else int(root) % 12
    out: list[list[int]] = []
    for octave in range(-1, 10):
        v = _realize(pc + 12 * (octave + 1), quality, degrees)
        if v and v[0] >= 0 and v[-1] <= 127:
            out.append(v)
    return out


def voicing(
    root: str,
    quality: str = "min",
    voices: int = 3,
    shape: str | None = None,
    top_near: int | None = None,
    top_range: tuple[int, int] = VOICING_TOP_RANGE,
    bass_range: tuple[int, int] = VOICING_BASS_RANGE,
) -> list[int]:
    """Generate concrete MIDI pitches for a chord voicing based on empirical models.

    Args:
        root: Chord root note name.
        quality: Chord quality key from CHORD_QUALITIES.
        voices: Voice count (2, 3, or 4) used to pick default shape when shape is None.
        shape: Optional explicit voicing shape name from VOICING_SHAPES.
        top_near: Optional target MIDI pitch for the top melody voice.
        top_range: Acceptable MIDI pitch range for the top voice (defaults to 61 to 77).
        bass_range: Acceptable MIDI pitch range for lowest voice (defaults to 46 to 58).

    Returns:
        Ascending list of MIDI pitch numbers for the voiced chord.

    Raises:
        ValueError: If voice count is unsupported or pitches cannot fit MIDI range.
    """
    if shape is None:
        if voices not in VOICING_DEFAULT_SHAPE:
            raise ValueError(
                f"voices={voices!r}: measured are 2, 3 or 4 voices (median 3, never "
                "more than 4). For thicker chords use build_chord()."
            )
        shape = VOICING_DEFAULT_SHAPE[voices]
    degrees = _shape_degrees(shape)
    cands = _candidates(root, quality, degrees)
    if not cands:
        raise ValueError(f"no position for {root}{quality} with shape {shape!r} inside MIDI range")
    mid = (top_range[0] + top_range[1]) // 2

    if top_near is None:

        def rank(v: list[int]) -> tuple[int, int]:
            return (
                _outside(v[-1], *top_range) * 2 + _outside(v[0], *bass_range),
                abs(v[-1] - mid),
            )

    else:
        target = int(top_near)

        def rank(v: list[int]) -> tuple[int, int]:
            return (
                abs(v[-1] - target),
                _outside(v[-1], *top_range) * 2 + _outside(v[0], *bass_range),
            )

    return min(cands, key=rank)


def _as_chord(item: object, key_root: str | None, mode: str) -> tuple[str, str]:
    """Extract (root, quality) tuple from progression dict, tuple, or degree label."""
    if isinstance(item, dict):
        return str(item["root"]), str(item.get("quality", "min"))
    if isinstance(item, (tuple, list)) and len(item) == 2:
        return str(item[0]), str(item[1])
    if isinstance(item, str):
        if not key_root:
            raise ValueError(f"degree label {item!r} without key_root; key is required.")
        return degree_chord(key_root, item, mode)
    raise ValueError(
        f"unreadable chord {item!r}. Expected a dict from progression(), a "
        "(root, quality) pair, or a degree label."
    )


def _voice_rank(
    pitches: list[int],
    *,
    anchor: int,
    mid: int,
    top_range: tuple[int, int],
    bass_range: tuple[int, int],
) -> tuple[int, int, int]:
    """Score octave placement based on register constraints and voice-leading."""
    return (
        _outside(pitches[-1], *top_range) * 2 + _outside(pitches[0], *bass_range),
        abs(pitches[-1] - anchor),
        abs(pitches[-1] - mid),
    )


def voice_progression(
    chords: Sequence[object],
    key_root: str | None = None,
    mode: str = "minor",
    voices: int = 3,
    shape: str | None = None,
    top_range: tuple[int, int] = VOICING_TOP_RANGE,
    bass_range: tuple[int, int] = VOICING_BASS_RANGE,
    start_top: int | None = None,
) -> list[list[int]]:
    """Voice an entire progression using consistent voice-leading and register.

    Args:
        chords: Sequence of progression dictionaries, (root, quality) pairs, or degree strings.
        key_root: Key tonic note name (required when degree strings are supplied).
        mode: Mode name for degree resolution.
        voices: Voice count (2, 3, or 4).
        shape: Preferred voicing shape identifier.
        top_range: Target pitch range for the top voice.
        bass_range: Target pitch range for the bass voice.
        start_top: Optional target pitch for the first chord's top voice.

    Returns:
        List of voiced chords, each represented as an ascending list of MIDI pitch numbers.

    Raises:
        ValueError: If voice count is invalid or chord cannot be realized.
    """
    if shape is None:
        if voices not in VOICING_DEFAULT_SHAPE:
            raise ValueError(
                f"voices={voices!r}: measured are 2, 3 or 4 voices (median 3, never more than 4)."
            )
        shape = VOICING_DEFAULT_SHAPE[voices]
    _shape_degrees(shape)  # report an unknown shape right here
    mid = (top_range[0] + top_range[1]) // 2
    prev = int(start_top) if start_top is not None else mid

    out: list[list[int]] = []
    for item in chords:
        root, quality = _as_chord(item, key_root, mode)
        degrees = _shape_degrees(fitting_shape(quality, shape))
        cands = _candidates(root, quality, degrees)
        if not cands:
            raise ValueError(f"no position for {root}{quality} inside MIDI range")
        pick = min(
            cands,
            key=partial(
                _voice_rank, anchor=prev, mid=mid, top_range=top_range, bass_range=bass_range
            ),
        )
        out.append(pick)
        prev = pick[-1]
    return out


# --------------------------------------------------------------------------- #
# Reading a voicing back: identify, describe, check
# --------------------------------------------------------------------------- #


def identify_chord(pitches: Sequence[int], flats: bool = False) -> tuple[str, str] | None:
    """Identify root note and chord quality from a set of MIDI pitches.

    Compares pitch classes against CHORD_QUALITIES, preferring root-position interpretations.
    Requires at least 3 distinct pitch classes to avoid ambiguous dyad identification.

    Args:
        pitches: Sequence of MIDI note numbers.
        flats: When True, spells root note with flat accidentals.

    Returns:
        Tuple of (root_name, chord_quality), or None if pitches cannot be resolved.
    """
    pcs = {p % 12 for p in pitches}
    if len(pcs) < 3:
        return None
    bass_pc = min(pitches) % 12
    best: tuple[tuple[int, int], int, str] | None = None
    for root_pc in sorted(pcs):
        for name, ivs in CHORD_QUALITIES.items():
            if {(root_pc + iv) % 12 for iv in ivs} == pcs:
                rank = (len(ivs), 0 if root_pc == bass_pc else 1)
                if best is None or rank < best[0]:
                    best = (rank, root_pc, name)
    if best is None:
        return None
    return pc_name(best[1], flats), best[2]


def voicing_shape_of(
    pitches: Sequence[int],
    root: str | None = None,
    quality: str | None = None,
) -> tuple[tuple[int, int], ...] | None:
    """Extract normalized voicing shape ((degree, offset), ...) from MIDI pitches.

    Normalized to octave 0 of the lowest voice to allow transpositional comparisons.

    Args:
        pitches: Sequence of MIDI note numbers.
        root: Optional chord root name (inferred if None).
        quality: Optional chord quality name (inferred if None).

    Returns:
        Tuple of (chord_degree, octave_offset) pairs, or None if chord cannot be identified.
    """
    if root is None or quality is None:
        found = identify_chord(pitches)
        if found is None:
            return None
        root, quality = found
    dmap = _degree_map(quality)
    ps = sorted(pitches)
    root_pc = NAME_TO_PC[root] if isinstance(root, str) else int(root) % 12
    anchor = ps[0] - ((ps[0] - root_pc) % 12)
    out: list[tuple[int, int]] = []
    for p in ps:
        rel = p - anchor
        deg = dmap.get(rel % 12)
        if deg is None:
            return None
        out.append((deg, rel // 12))
    base = out[0][1]
    return tuple((d, o - base) for d, o in out)


def _shape_name_of(shape: Sequence[tuple[int, int]] | None) -> str | None:
    """Return matching standard shape identifier for a normalized shape tuple."""
    if not shape:
        return None
    norm = tuple((d, o) for d, o in shape)
    for name, spec in VOICING_SHAPES.items():
        base = spec.degrees[0][1]
        if tuple((d, o - base) for d, o in spec.degrees) == norm:
            return name
    return None


def shape_text(shape: Sequence[tuple[int, int]] | None) -> str:
    """Format voicing shape as readable string (e.g. '1-5-3oct').

    Args:
        shape: Sequence of (degree, octave_offset) tuples.

    Returns:
        Formatted shape string representation ('-' if empty).
    """
    if not shape:
        return "-"
    name = _shape_name_of(shape)
    if name is not None:
        return name
    parts: list[str] = []
    for d, o in shape:
        if o == 0:
            parts.append(f"{d}")
        elif o == 1:
            parts.append(f"{d}oct")
        else:
            parts.append(f"{d}+{o}oct")
    return "-".join(parts)


def voicing_facts(
    pitches: Sequence[int],
    root: str | None = None,
    quality: str | None = None,
) -> dict[str, Any]:
    """Compute measurable structural metrics for a chord voicing.

    Args:
        pitches: Sequence of MIDI note numbers.
        root: Optional chord root note name.
        quality: Optional chord quality name.

    Returns:
        Dictionary containing pitches, voice count, bass, top, span, intervals,
        inversion, degrees, and detected shape.
    """
    ps = sorted(int(p) for p in pitches)
    facts: dict[str, Any] = {
        "pitches": ps,
        "voices": len(ps),
        "bass": ps[0] if ps else None,
        "top": ps[-1] if ps else None,
        "span": (ps[-1] - ps[0]) if ps else None,
        "intervals": [ps[i + 1] - ps[i] for i in range(len(ps) - 1)],
        "bottom_interval": (ps[1] - ps[0]) if len(ps) > 1 else None,
        "chord": None,
        "root": None,
        "quality": None,
        "inversion": None,
        "bass_degree": None,
        "top_degree": None,
        "shape": None,
        "shape_name": None,
    }
    if not ps:
        return facts
    if root is None or quality is None:
        found = identify_chord(ps)
        if found is not None:
            root, quality = found
    if root is None or quality is None:
        return facts
    facts["root"], facts["quality"] = root, quality
    facts["chord"] = chord_symbol(root, quality)
    dmap = _degree_map(quality)
    root_pc = NAME_TO_PC[root]
    bass_deg = dmap.get((ps[0] - root_pc) % 12)
    facts["bass_degree"] = bass_deg
    facts["top_degree"] = dmap.get((ps[-1] - root_pc) % 12)
    facts["inversion"] = {1: 0, 3: 1, 5: 2, 7: 3}.get(bass_deg)
    shape = voicing_shape_of(ps, root, quality)
    facts["shape"] = shape
    facts["shape_name"] = _shape_name_of(shape)
    return facts


def check_voicing(
    voicings: Sequence[object],
    root: str | None = None,
    quality: str | None = None,
    top_range: tuple[int, int] = VOICING_TOP_RANGE,
    bass_range: tuple[int, int] = VOICING_BASS_RANGE,
) -> list[str]:
    """Evaluate one voicing or a progression against empirical voice-leading rules.

    Rules evaluated:
        1: Lowest interval spacing (avoids second intervals at the base).
        2: Top voice chord degree (prefers 3rd, 5th, or 9th).
        3: Voicing shape consistency across progression.
        4: Inversion and chord bass tone validity.
        5: Voice count bounds (2 to 4 voices).
        6: Register bounds for top and bass voices.

    Args:
        voicings: Single pitch sequence or sequence of voiced chords.
        root: Optional root note for single chord evaluation.
        quality: Optional chord quality for single chord evaluation.
        top_range: Expected pitch bounds for top voice.
        bass_range: Expected pitch bounds for bass voice.

    Returns:
        List of warning and informational strings detailing rule deviations.
    """
    if not voicings:
        return ["empty voicing: nothing to check."]
    if isinstance(voicings[0], (list, tuple)):
        seq = [[int(p) for p in v] for v in voicings]  # type: ignore[union-attr]
        multi = True
    else:
        seq = [[int(p) for p in voicings]]  # type: ignore[arg-type]
        multi = False

    findings: list[str] = []
    shapes: list[tuple[tuple[int, int], ...] | None] = []
    tops: list[int] = []

    for i, v in enumerate(seq):
        tag = f"chord {i + 1}: " if multi else ""
        ps = sorted(v)
        if not ps:
            findings.append(f"{tag}empty voicing.")
            continue
        facts = voicing_facts(ps, None if multi else root, None if multi else quality)
        shapes.append(facts["shape"])
        tops.append(ps[-1])

        if len(ps) > VOICING_MAX_VOICES:
            findings.append(
                f"{tag}rule 5 (voice count): {len(ps)} voices; in reference material "
                f"median is 3 and maximum is {VOICING_MAX_VOICES}."
            )
        elif len(ps) == 2:
            findings.append(
                f"{tag}note: two voices: a dyad does not fully specify chord quality."
            )
        elif len(ps) < 2:
            findings.append(f"{tag}a single note is not a voicing.")
            continue

        bottom = ps[1] - ps[0]
        if bottom <= 2:
            findings.append(
                f"{tag}rule 1 (no second at the bottom): lowest interval {bottom} st; "
                "measured 3 of 221 attacks (1.4%). "
                "Fifth (82x), fourth (65x), or octave (34x) expected."
            )
        elif bottom in (3, 4):
            good = ", ".join(str(k) for k in VOICING_BOTTOM_GOOD)
            findings.append(
                f"{tag}note: third at the bottom ({bottom} st): attested in 32 of 221 attacks, "
                f"while {good} st represent 181 of 221."
            )
        elif bottom not in VOICING_BOTTOM_COUNTS:
            known = ", ".join(str(k) for k in sorted(VOICING_BOTTOM_COUNTS))
            findings.append(
                f"{tag}rule 1: lowest interval {bottom} st not in reference distribution "
                f"(attested: {known} st)."
            )

        if facts["top_degree"] is None:
            names = " ".join(note_name(p) for p in ps)
            findings.append(f"{tag}chord not identifiable ({names}): rules 2 and 4 skipped.")
        else:
            top_deg = int(facts["top_degree"])
            top_deg = {2: 9, 4: 11}.get(top_deg, top_deg)
            if top_deg == 1:
                findings.append(
                    f"{tag}rule 2 (third on top): top note is the root "
                    "(attested in 5.7% of blocks, third on top in 58.0%)."
                )
            elif top_deg not in (3, 5, 9, 11):
                findings.append(
                    f"{tag}rule 2: top note is degree {top_deg} (reference blocks show "
                    "third: 58.0%, fifth: 22.7%, ninth: 9.1%, eleventh: 3.4%)."
                )
            if facts["inversion"] is None:
                findings.append(
                    f"{tag}rule 4 (inversion): bass {note_name(ps[0])} is not a chord tone."
                )
            elif facts["inversion"] == 1:
                findings.append(
                    f"{tag}note: first inversion "
                    "(9.2% of reference blocks; root position is 73.8%)."
                )

        if _outside(ps[-1], *top_range):
            findings.append(
                f"{tag}rule 6 (middle register): top voice {ps[-1]} ({note_name(ps[-1])}) "
                f"lies outside window {top_range[0]}..{top_range[1]}."
            )
        if _outside(ps[0], *bass_range):
            extra = ""
            if ps[0] < VOICING_BASS_ABSOLUTE_LOW:
                extra = f" (below sample lowest {VOICING_BASS_ABSOLUTE_LOW})"
            elif ps[0] < bass_range[0]:
                extra = " (below standard chord bass threshold)"
            findings.append(
                f"{tag}rule 6 (middle register): bass {ps[0]} ({note_name(ps[0])}) lies "
                f"outside {bass_range[0]}..{bass_range[1]}{extra}."
            )

    if multi and len(seq) > 1:
        known_shapes = [g for g in shapes if g is not None]
        distinct = set(known_shapes)
        if len(distinct) > 1:
            listed = " | ".join(shape_text(g) for g in sorted(distinct))
            findings.append(
                f"rule 3 (one shape, transposed): {len(distinct)} different shapes in "
                f"{len(seq)} chords. Reference template 004 uses one shape across all blocks. "
                f"Found: {listed}"
            )
        for i in range(len(tops) - 1):
            leap = abs(tops[i + 1] - tops[i])
            if leap > VOICING_TOP_MAX_LEAP:
                findings.append(
                    f"chord {i + 1}->{i + 2}: top voice leaps {leap} st "
                    f"({tops[i]} -> {tops[i + 1]}); maximum reference leap is "
                    f"{VOICING_TOP_MAX_LEAP} st (minor sixth)."
                )
    return findings


# --------------------------------------------------------------------------- #
# Frequency conversion for Live filter and EQ devices (Auto Filter, EQ Eight).
# Normalised control range 0.0..1.0 maps logarithmically:
# norm = log10(hz / 20) / log10(20000 / 20) = log10(hz / 20) / 3.
# Lower bound of 20.0 Hz measured against Live 11/12 parameter scaling. Other devices
# may have other bounds; VST plugins do not report theirs at all (measured).
#
# The lower bound is what decides accuracy. A wrong base quietly sets wrong frequencies
# with no error message: with 30 Hz instead of 20, a request for 100 Hz really lands at
# 71.9 Hz (572 cents off) and 1 kHz at 829.6 Hz (323 cents). So check at the bottom of
# the range. At the top end both formulas converge (at 18939 Hz, 0.99162 against
# 0.99211, measured 0.9921), and a wrong base does not show there.
# --------------------------------------------------------------------------- #

FILTER_HZ_MIN: float = 20.0
FILTER_HZ_MAX: float = 20000.0

#: Measured (Hz, normalized) control points for Live filter parameters.
MEASURED_FILTER_POINTS: tuple[tuple[float, float], ...] = ((13692.0, 0.9451), (18939.0, 0.9921))


def hz_to_norm(hz: float, lo: float = FILTER_HZ_MIN, hi: float = FILTER_HZ_MAX) -> float:
    """Convert audio frequency in Hertz to normalized 0.0..1.0 control parameter.

    Uses logarithmic mapping: norm = log10(hz / lo) / log10(hi / lo).
    Values outside lo..hi are clamped.

    Args:
        hz: Frequency in Hertz.
        lo: Lower frequency bound (defaults to 20.0 Hz).
        hi: Upper frequency bound (defaults to 20000.0 Hz).

    Returns:
        Normalized float value clamped to 0.0..1.0.
    """
    hz = max(lo, min(hi, float(hz)))
    return math.log(hz / lo) / math.log(hi / lo)


def norm_to_hz(value: float, lo: float = FILTER_HZ_MIN, hi: float = FILTER_HZ_MAX) -> float:
    """Convert normalized 0.0..1.0 control parameter to frequency in Hertz.

    Inverse of hz_to_norm: hz = lo * (hi / lo) ** value. Clamped to 0.0..1.0.

    Args:
        value: Normalized control value (0.0 to 1.0).
        lo: Lower frequency bound (defaults to 20.0 Hz).
        hi: Upper frequency bound (defaults to 20000.0 Hz).

    Returns:
        Frequency in Hertz.
    """
    value = max(0.0, min(1.0, float(value)))
    return lo * math.exp(value * math.log(hi / lo))


# --------------------------------------------------------------------------- #
# Self-check: recompute measured examples against reference dataset values
# --------------------------------------------------------------------------- #


def self_check() -> list[str]:
    """Verify consistency of empirical models against reference dataset points.

    Returns:
        List of error strings if any computed models deviate from reference values;
        empty list if all checks pass.
    """
    findings: list[str] = []

    pool = progression_pool("minor")
    if pool != MEASURED_DEGREE_POOL:
        findings.append(
            f"degree pool: progression_pool('minor') = {pool}, measured {MEASURED_DEGREE_POOL}"
        )

    for form, entry in sorted(MEASURED_PROGRESSIONS.items()):
        computed = " ".join(str(c["chord"]) for c in progression(entry.key, form=form))
        measured = entry.chords.replace(" - ", " ").replace(" | ", " ")
        if computed != measured and not entry.deviation:
            findings.append(
                f"progression {form}: computed {computed!r}, measured {measured!r} "
                "and no known deviation recorded"
            )

    for probe in MEASURED_VOICINGS:
        got = voicing(probe.root, probe.quality, shape=probe.shape)
        if tuple(got) != probe.pitches:
            findings.append(
                f"voicing {probe.shape} {chord_symbol(probe.root, probe.quality)}: "
                f"got {got}, measured {list(probe.pitches)} ({probe.source})"
            )

    voiced = voice_progression(progression("Eb", form="005"), shape="1-5-3oct")
    for i, measured_block in enumerate(MEASURED_005_BLOCKS):
        if tuple(voiced[i]) != measured_block:
            findings.append(f"005 block {i + 1}: got {voiced[i]}, measured {list(measured_block)}")
    leap = max(abs(voiced[i + 1][-1] - voiced[i][-1]) for i in range(len(voiced) - 1))
    if leap > VOICING_TOP_MAX_LEAP:
        findings.append(
            f"005 voiced: top voice leaps {leap} st, measured maximum is {VOICING_TOP_MAX_LEAP}"
        )

    for hz, norm in MEASURED_FILTER_POINTS:
        got_norm = hz_to_norm(hz)
        if abs(got_norm - norm) > 1e-4:
            findings.append(f"filter curve: {hz:g} Hz -> {got_norm:.5f}, measured {norm}")

    return findings


def _demo() -> None:
    """Display self-check status and reference voicing demonstrations."""
    print("Reference voicing and progression self-check:")
    print()
    print(f"  minor pool : {' '.join(progression_pool('minor'))}")
    print(f"  major pool : {' '.join(progression_pool('major'))} (rotation of minor pool)")
    print(f"  absent     : {'; '.join(PROGRESSION_ABSENT)}")
    print()
    for form, entry in sorted(MEASURED_PROGRESSIONS.items()):
        computed = " ".join(str(c["chord"]) for c in progression(entry.key, form=form))
        mark = f"   [{entry.deviation}]" if entry.deviation else "   [identical]"
        print(f"  {form:<5} {computed:<28} measured {entry.chords}{mark}")
    print()
    for probe in MEASURED_VOICINGS:
        got = voicing(probe.root, probe.quality, shape=probe.shape)
        symbol = chord_symbol(probe.root, probe.quality)
        print(
            f"  {probe.shape:<11} {symbol:<7} {got!s:<18} "
            f"measured {list(probe.pitches)!s:<18} ({probe.source})"
        )
    print()
    findings = self_check()
    if not findings:
        print("self_check(): all reference examples reproduced successfully.")
    else:
        print(f"self_check(): {len(findings)} finding(s)")
        for line in findings:
            print(f"  {line}")


if __name__ == "__main__":
    _demo()

