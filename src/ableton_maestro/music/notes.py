"""Note list manipulation, validation, and diffing for Live MIDI clips.

Provides pure operations over lists of :class:`~ableton_maestro.models.Note`:
transforms, safety validation, and change diffing. All operations assume
replace semantics (full note list in, full note list out).

Key operations:
- Replace-semantics helpers: ensures note updates replace existing notes rather
  than silently stacking duplicates.
- Time conversion: translates between Live's clip-local beat representation and
  standard 480 ticks/quarter resolution.
- Validation: catches missing keys, beat-zero stacking, out-of-range pitches,
  and unquantized note boundaries.

Examples:
>>> from ableton_maestro.models import Note
>>> notes = [Note(pitch=60, start_time=0.0, duration=0.5, velocity=100, mute=False)]
>>> louder = scale_velocities(notes, 1.2)
>>> louder[0].velocity
120
>>> diff(notes, louder).summary()
'0 added, 0 removed, 1 changed, 0 unchanged'
>>> stacked = [Note(pitch=p, start_time=0.0, duration=0.25, velocity=100, mute=False)
...            for p in (60, 62, 64)]
>>> validate(stacked).errors[0].code
'stacked_on_beat_zero'
>>> beats_to_ticks(1.5)
720
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from ableton_maestro.models import Note

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "BEATS_PER_BAR_4_4",
    "DEFAULT_VELOCITY",
    "FULL_VELOCITY",
    "GRID_TOLERANCE",
    "KNOWN_NOTE_KEYS",
    "LEGACY_DEFAULT_DURATION",
    "LEGACY_DEFAULT_START_TIME",
    "LEGACY_NOTE_KEYS",
    "PITCH_RANGE",
    "STACK_WARN_MIN_NOTES",
    "TICKS_PER_QUARTER",
    "TICKS_SUSPICION_BEATS",
    "TOLERATED_NOTE_KEYS",
    "VELOCITY_RANGE",
    "Issue",
    "NoteChange",
    "NoteDiff",
    "ValidationReport",
    "beats_to_ticks",
    "dedupe",
    "diff",
    "filter_by_pitch",
    "filter_by_time",
    "find_overlaps",
    "from_beat_tuples",
    "from_dicts",
    "from_tick_notes",
    "merge",
    "quantize_times",
    "replace_range",
    "scale_velocities",
    "shift",
    "sort_notes",
    "span",
    "stretch",
    "ticks_to_beats",
    "to_dicts",
    "to_tick_notes",
    "transpose",
    "validate",
    "validate_note_dicts",
    "without_tolerated_keys",
]

# --------------------------------------------------------------------- constants

#: Ticks per quarter note in music/humanize.py (480 ticks/quarter).
TICKS_PER_QUARTER = 480

#: Default measure duration in beats for 4/4 time signature.
BEATS_PER_BAR_4_4 = 4.0

#: Supported MIDI pitch range (0 to 127).
PITCH_RANGE = (0, 127)

#: Clamping range for MIDI note velocities (1 to 127).
VELOCITY_RANGE = (1, 127)

#: Standard Live Piano Roll drawing default and full-scale velocities. Measured over
#: the corpus (833 tracks from 103 projects): 648 tracks carry exactly one velocity
#: value across all of their clips, and where it is constant it is 100 in 472 tracks,
#: 127 in 65 and 91 in 9. Both numbers below are therefore measured rather than chosen.
#: The caveat that belongs with them: the sample leans heavily on tutorial rebuilds with
#: drawn notes, so the constant share may sit elsewhere in played material.
DEFAULT_VELOCITY = 100
FULL_VELOCITY = 127

#: Default fallbacks used by lenient readers for missing start_time and duration.
LEGACY_DEFAULT_START_TIME = 0.0
LEGACY_DEFAULT_DURATION = 0.25

#: Beat tolerance for equality comparisons.
GRID_TOLERANCE = 1e-4

#: Threshold above which start_time values likely represent unconverted ticks: 1000
#: beats is more than 250 bars of 4/4. Derived rather than measured, so it is a smell
#: test and :func:`validate` reports it as a warning, not a fact.
TICKS_SUSPICION_BEATS = 1000.0

#: Minimum note count required to flag a partial stack at beat zero. Not measured:
#: chosen so that an ordinary three-note chord on beat 0 does not trip the check.
STACK_WARN_MIN_NOTES = 4

_CORE_FIELDS: tuple[str, ...] = ("pitch", "start_time", "duration", "velocity", "mute")
_EXTENDED_FIELDS: tuple[str, ...] = ("probability", "velocity_deviation", "release_velocity")

#: Core note keys supported by the remote script protocol.
LEGACY_NOTE_KEYS = frozenset(_CORE_FIELDS)

#: All note keys recognized by the protocol.
KNOWN_NOTE_KEYS = frozenset(_CORE_FIELDS + _EXTENDED_FIELDS)

#: Keys returned by Live reads that are not part of the active write model.
TOLERATED_NOTE_KEYS = frozenset({"note_id", "pitch_bend_range", "pressure", "timbre", "slide"})

#: Number of decimal places used when rounding float beat values.
_ROUND_DIGITS = 9

_MISSING = object()

Severity = Literal["error", "warning"]
TimeSelect = Literal["onset", "overlap", "contained"]
OutOfRange = Literal["error", "clamp", "drop"]
BeforeZero = Literal["error", "drop", "clamp", "keep"]
KeepPolicy = Literal["first", "last", "loudest", "longest"]
MatchOn = Literal["pitch_time", "pitch"]


# ------------------------------------------------------------------- small tools


def _clean(value: float) -> float:
    """Round a beat value so float noise does not reach the wire format."""
    return round(float(value), _ROUND_DIGITS)


def _note_end(note: Note) -> float:
    """Return the end beat of a note in clip-local beats."""
    return note.start_time + note.duration


def _extended(note: Note, name: str) -> Any:
    """Read a Live 11+ field, or _MISSING when the model has none."""
    return getattr(note, name, _MISSING)


def _differs(before: Any, after: Any, tolerance: float) -> bool:
    """Compare two field values with numeric tolerance for floats."""
    if isinstance(before, bool) or isinstance(after, bool):
        return bool(before) != bool(after)
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return abs(float(before) - float(after)) > tolerance
    return before != after


def sort_notes(notes: Iterable[Note]) -> list[Note]:
    """Sort notes by (start_time, pitch, duration).

    Args:
        notes: Iterable of Note instances.

    Returns:
        Sorted list of Note instances.
    """
    return sorted(notes, key=lambda n: (n.start_time, n.pitch, n.duration))


def span(notes: Sequence[Note]) -> tuple[float, float]:
    """Return bounding (first_start, last_end) in clip-local beats.

    Args:
        notes: Sequence of Note instances.

    Returns:
        Tuple of (earliest_start_beat, latest_end_beat).
    """
    if not notes:
        return (0.0, 0.0)
    return (min(n.start_time for n in notes), max(_note_end(n) for n in notes))


def _in_time_range(note: Note, start: float, end: float, select: TimeSelect) -> bool:
    """Check range membership under one of the three selection rules."""
    if select == "onset":
        return start <= note.start_time < end
    if select == "contained":
        return note.start_time >= start and _note_end(note) <= end
    if select == "overlap":
        return note.start_time < end and _note_end(note) > start
    raise ValueError(f"select must be 'onset', 'overlap' or 'contained', not {select!r}")


def find_overlaps(
    notes: Sequence[Note],
    *,
    same_pitch_only: bool = True,
    tolerance: float = GRID_TOLERANCE,
) -> list[tuple[int, int]]:
    """Find index pairs of notes that overlap in time.

    Args:
        notes: Sequence of Note instances.
        same_pitch_only: When True, only check overlaps on the same pitch.
            When False, report all polyphonic overlaps across pitches.
        tolerance: Float beat tolerance for overlap comparisons.

    Returns:
        Sorted list of index pairs (i, j) with i < j referencing notes.
    """
    order = sorted(range(len(notes)), key=lambda i: (notes[i].start_time, notes[i].pitch))
    pairs: list[tuple[int, int]] = []

    if same_pitch_only:
        last_by_pitch: dict[int, int] = {}
        for idx in order:
            note = notes[idx]
            previous = last_by_pitch.get(note.pitch)
            if previous is not None and _note_end(notes[previous]) > note.start_time + tolerance:
                pairs.append((previous, idx) if previous < idx else (idx, previous))
            if previous is None or _note_end(notes[previous]) < _note_end(note):
                last_by_pitch[note.pitch] = idx
        return sorted(set(pairs))

    active: list[int] = []
    for idx in order:
        note = notes[idx]
        active = [j for j in active if _note_end(notes[j]) > note.start_time + tolerance]
        for j in active:
            pairs.append((j, idx) if j < idx else (idx, j))
        active.append(idx)
    return sorted(set(pairs))


# ------------------------------------------------------- read-modify-write helpers


def replace_range(
    notes: Sequence[Note],
    replacement: Sequence[Note] = (),
    *,
    start: float = 0.0,
    end: float = math.inf,
    pitch_low: int = PITCH_RANGE[0],
    pitch_high: int = PITCH_RANGE[1],
    select: TimeSelect = "onset",
) -> list[Note]:
    """Replace notes within a time and pitch window with replacement notes.

    Args:
        notes: Sequence of existing Note instances.
        replacement: Sequence of replacement Note instances.
        start: Window start in clip-local beats.
        end: Window end in clip-local beats.
        pitch_low: Lower pitch limit (inclusive).
        pitch_high: Upper pitch limit (inclusive).
        select: Window filter rule ('onset', 'contained', or 'overlap').

    Returns:
        Sorted list of Note instances.

    Raises:
        ValueError: If end < start or pitch_high < pitch_low.
    """
    if end < start:
        raise ValueError(f"end ({end}) is before start ({start})")
    if pitch_high < pitch_low:
        raise ValueError(f"pitch_high ({pitch_high}) is below pitch_low ({pitch_low})")

    kept = [
        n
        for n in notes
        if not (pitch_low <= n.pitch <= pitch_high and _in_time_range(n, start, end, select))
    ]
    return sort_notes([*kept, *replacement])


def transpose(
    notes: Sequence[Note],
    semitones: int,
    *,
    out_of_range: OutOfRange = "error",
) -> list[Note]:
    """Transpose all note pitches by a specified number of semitones.

    Args:
        notes: Sequence of Note instances.
        semitones: Pitch offset in semitones.
        out_of_range: Action on pitches falling outside 0..127 ('error', 'clamp', or 'drop').

    Returns:
        Sorted list of transposed Note instances.

    Raises:
        ValueError: If pitches fall out of range and out_of_range is 'error',
            or if out_of_range option is unrecognized.
    """
    low, high = PITCH_RANGE
    out: list[Note] = []
    offending: list[int] = []

    for note in notes:
        target = note.pitch + semitones
        if low <= target <= high:
            out.append(replace(note, pitch=target))
            continue
        if out_of_range == "drop":
            continue
        if out_of_range == "clamp":
            out.append(replace(note, pitch=min(high, max(low, target))))
            continue
        if out_of_range == "error":
            offending.append(target)
            continue
        raise ValueError(f"out_of_range must be 'error', 'clamp' or 'drop', not {out_of_range!r}")

    if offending:
        raise ValueError(
            f"transposing by {semitones} semitones puts {len(offending)} note(s) outside "
            f"MIDI {low}..{high} (e.g. {sorted(set(offending))[:5]}). "
            "Pass out_of_range='clamp' to pin them (intervals change) or 'drop' to lose them."
        )
    return sort_notes(out)


def _snap(value: float, grid: float, strength: float) -> float:
    """Pull value towards the nearest grid multiple by strength factor."""
    target = round(value / grid) * grid
    return _clean(value + strength * (target - value))


def quantize_times(
    notes: Sequence[Note],
    grid: float = 0.25,
    *,
    strength: float = 1.0,
    quantize_ends: bool = False,
    min_duration: float | None = None,
) -> list[Note]:
    """Snap note start times (and optionally durations) to a beat grid.

    Args:
        notes: Sequence of Note instances.
        grid: Beat grid subdivision (e.g. 1.0 quarter, 0.5 eighth, 0.25 sixteenth).
        strength: Quantization strength between 0.0 (no-op) and 1.0 (full snap).
        quantize_ends: Whether to snap note end positions and adjust duration.
        min_duration: Minimum duration when quantizing ends (defaults to grid).

    Returns:
        Sorted list of quantized Note instances.

    Raises:
        ValueError: If grid is non-positive or strength is outside 0.0..1.0.
    """
    if grid <= 0:
        raise ValueError(f"grid must be positive, got {grid}")
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be between 0.0 and 1.0, got {strength}")
    floor = grid if min_duration is None else min_duration

    out: list[Note] = []
    for note in notes:
        new_start = _snap(note.start_time, grid, strength)
        new_duration = note.duration
        if quantize_ends:
            new_end = _snap(_note_end(note), grid, strength)
            new_duration = max(_clean(new_end - new_start), floor)
        out.append(replace(note, start_time=new_start, duration=_clean(new_duration)))
    return sort_notes(out)


def scale_velocities(
    notes: Sequence[Note],
    factor: float = 1.0,
    *,
    offset: int = 0,
    floor: int = VELOCITY_RANGE[0],
    ceiling: int = VELOCITY_RANGE[1],
) -> list[Note]:
    """Scale and offset note velocities within clamp limits.

    Args:
        notes: Sequence of Note instances.
        factor: Multiplicative velocity scaling factor.
        offset: Additive velocity offset.
        floor: Lower velocity clamp bound (defaults to 1).
        ceiling: Upper velocity clamp bound (defaults to 127).

    Returns:
        Sorted list of Note instances with updated velocities.
    """
    out: list[Note] = []
    for note in notes:
        value = round(note.velocity * factor) + offset
        out.append(replace(note, velocity=min(ceiling, max(floor, value))))
    return sort_notes(out)


def shift(
    notes: Sequence[Note],
    delta: float,
    *,
    before_zero: BeforeZero = "drop",
) -> list[Note]:
    """Shift all note start times by delta beats.

    Args:
        notes: Sequence of Note instances.
        delta: Timing offset in beats (positive or negative).
        before_zero: Policy for notes landing before beat 0 ('drop', 'clamp', 'keep', or 'error').

    Returns:
        Sorted list of Note instances.

    Raises:
        ValueError: If a note lands before beat 0 and before_zero is 'error',
            or if before_zero is unrecognized.
    """
    out: list[Note] = []
    for note in notes:
        new_start = _clean(note.start_time + delta)
        if new_start >= 0.0:
            out.append(replace(note, start_time=new_start))
            continue
        if before_zero == "drop":
            continue
        if before_zero == "clamp":
            out.append(replace(note, start_time=0.0))
            continue
        if before_zero == "keep":
            out.append(replace(note, start_time=new_start))
            continue
        if before_zero == "error":
            raise ValueError(
                f"shifting by {delta} beats puts a note at {new_start}, before the clip "
                "start. Pass before_zero='drop', 'clamp' or 'keep' to say what should "
                "happen to it."
            )
        raise ValueError(
            f"before_zero must be 'error', 'drop', 'clamp' or 'keep', not {before_zero!r}"
        )
    return sort_notes(out)


def stretch(
    notes: Sequence[Note],
    factor: float,
    *,
    origin: float = 0.0,
    scale_durations: bool = True,
) -> list[Note]:
    """Scale note timing and duration around a fixed origin beat.

    Args:
        notes: Sequence of Note instances.
        factor: Time scaling factor (e.g. 2.0 = half time, 0.5 = double time).
        origin: Anchor beat position for the scaling transformation.
        scale_durations: Whether to scale note durations along with start times.

    Returns:
        Sorted list of stretched Note instances.

    Raises:
        ValueError: If factor is non-positive.
    """
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")
    out: list[Note] = []
    for note in notes:
        new_start = _clean(origin + (note.start_time - origin) * factor)
        new_duration = _clean(note.duration * factor) if scale_durations else note.duration
        out.append(replace(note, start_time=new_start, duration=new_duration))
    return sort_notes(out)


def filter_by_pitch(
    notes: Sequence[Note],
    low: int = PITCH_RANGE[0],
    high: int = PITCH_RANGE[1],
    *,
    exclude: bool = False,
) -> list[Note]:
    """Filter notes falling within or outside the specified pitch range.

    Args:
        notes: Sequence of Note instances.
        low: Minimum pitch (inclusive).
        high: Maximum pitch (inclusive).
        exclude: When True, keep notes outside the range; when False, keep notes inside.

    Returns:
        Sorted list of filtered Note instances.

    Raises:
        ValueError: If high < low.
    """
    if high < low:
        raise ValueError(f"high ({high}) is below low ({low})")
    inside = [n for n in notes if low <= n.pitch <= high]
    if not exclude:
        return sort_notes(inside)
    keep = {id(n) for n in inside}
    return sort_notes([n for n in notes if id(n) not in keep])


def filter_by_time(
    notes: Sequence[Note],
    start: float = 0.0,
    end: float = math.inf,
    *,
    select: TimeSelect = "onset",
    exclude: bool = False,
) -> list[Note]:
    """Filter notes falling within or outside the specified time window.

    Args:
        notes: Sequence of Note instances.
        start: Window start in clip-local beats.
        end: Window end in clip-local beats.
        select: Window filter rule ('onset', 'contained', or 'overlap').
        exclude: When True, keep notes outside the window; when False, keep notes inside.

    Returns:
        Sorted list of filtered Note instances.

    Raises:
        ValueError: If end < start.
    """
    if end < start:
        raise ValueError(f"end ({end}) is before start ({start})")
    inside = [n for n in notes if _in_time_range(n, start, end, select)]
    if not exclude:
        return sort_notes(inside)
    keep = {id(n) for n in inside}
    return sort_notes([n for n in notes if id(n) not in keep])


def merge(*note_lists: Sequence[Note], dedupe_exact: bool = False) -> list[Note]:
    """Concatenate multiple note sequences into a single sorted note list.

    Args:
        *note_lists: Variable sequences of Note instances.
        dedupe_exact: Whether to run dedupe() on the combined notes.

    Returns:
        Sorted list of merged Note instances.
    """
    combined: list[Note] = []
    for group in note_lists:
        combined.extend(group)
    if dedupe_exact:
        return dedupe(combined)
    return sort_notes(combined)


def dedupe(
    notes: Sequence[Note],
    *,
    time_tolerance: float = GRID_TOLERANCE,
    match_duration: bool = False,
    keep: KeepPolicy = "first",
) -> list[Note]:
    """Remove duplicate notes sharing the same pitch and start time.

    Args:
        notes: Sequence of Note instances.
        time_tolerance: Beat tolerance for matching start times.
        match_duration: When True, also require durations to match within tolerance.
        keep: Survivor selection policy ('first', 'last', 'loudest', or 'longest').

    Returns:
        Sorted list of deduped Note instances.

    Raises:
        ValueError: If keep policy is unrecognized.
    """
    ordered = sort_notes(notes)
    # pitch -> (position in `out`, the note currently occupying it)
    survivors: dict[int, tuple[int, Note]] = {}
    out: list[Note] = []

    for note in ordered:
        held = survivors.get(note.pitch)
        is_duplicate = (
            held is not None
            and abs(held[1].start_time - note.start_time) <= time_tolerance
            and (not match_duration or abs(held[1].duration - note.duration) <= time_tolerance)
        )
        if held is None or not is_duplicate:
            survivors[note.pitch] = (len(out), note)
            out.append(note)
            continue

        position, previous = held
        if _pick(previous, note, keep) is note:
            survivors[note.pitch] = (position, note)
            out[position] = note

    return sort_notes(out)


def _pick(previous: Note, candidate: Note, keep: KeepPolicy) -> Note:
    """Choose survivor of a duplicate pair under keep policy."""
    if keep == "first":
        return previous
    if keep == "last":
        return candidate
    if keep == "loudest":
        return candidate if candidate.velocity > previous.velocity else previous
    if keep == "longest":
        return candidate if candidate.duration > previous.duration else previous
    raise ValueError(f"keep must be 'first', 'last', 'loudest' or 'longest', not {keep!r}")


# ------------------------------------------------------------------- validation


@dataclass(frozen=True)
class Issue:
    """Represents a validation finding or defect in a note sequence.

    Attributes:
        code: Machine-readable issue identifier.
        severity: Issue severity ('error' or 'warning').
        message: Human-readable explanation.
        indices: Note indices associated with the finding.
    """

    code: str
    severity: Severity
    message: str
    indices: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert issue to JSON-serializable dictionary."""
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "indices": list(self.indices),
        }


@dataclass
class ValidationReport:
    """Summary of findings from a note validation pass.

    Attributes:
        note_count: Total number of notes inspected.
        issues: List of recorded Issue instances.
    """

    note_count: int
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return True if no error-level issues were recorded."""
        return not self.errors

    @property
    def errors(self) -> list[Issue]:
        """Return list of error-severity issues blocking a write."""
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        """Return list of warning-severity issues."""
        return [i for i in self.issues if i.severity == "warning"]

    def by_code(self, code: str) -> list[Issue]:
        """Return all recorded issues matching code."""
        return [i for i in self.issues if i.code == code]

    def summary(self) -> str:
        """Return concise single-line summary of counts and issue codes."""
        codes = ", ".join(sorted({i.code for i in self.issues})) or "none"
        return (
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s) "
            f"over {self.note_count} note(s); codes: {codes}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "ok": self.ok,
            "note_count": self.note_count,
            "errors": [i.to_dict() for i in self.errors],
            "warnings": [i.to_dict() for i in self.warnings],
            "summary": self.summary(),
        }


def validate(
    notes: Sequence[Note],
    *,
    clip_length: float | None = None,
    allow_overlap: bool = True,
    time_tolerance: float = GRID_TOLERANCE,
) -> ValidationReport:
    """Validate a sequence of Note instances prior to writing to Live.

    Inspects notes for unplayable parameters, negative start times, beat-zero
    stacking artifacts from misspelled keys, out-of-range values, and clip boundary overruns.

    Args:
        notes: Sequence of Note instances to validate.
        clip_length: Optional clip duration in beats for boundary overrun checks.
        allow_overlap: Whether to permit cross-pitch polyphonic overlaps.
        time_tolerance: Beat tolerance for equality comparisons.

    Returns:
        ValidationReport containing any errors or warnings found.
    """
    report = ValidationReport(note_count=len(notes))
    issues = report.issues

    if not notes:
        issues.append(
            Issue(
                "empty_note_list",
                "warning",
                "Empty note list. Written with mode='replace' this clears the clip; "
                "if the clip should be left alone, do not write at all.",
            )
        )
        return report

    low_pitch, high_pitch = PITCH_RANGE
    stacked: list[int] = []
    for index, note in enumerate(notes):
        if (
            abs(note.start_time - LEGACY_DEFAULT_START_TIME) <= time_tolerance
            and abs(note.duration - LEGACY_DEFAULT_DURATION) <= time_tolerance
        ):
            stacked.append(index)

        if note.duration <= 0:
            issues.append(
                Issue(
                    "non_positive_duration",
                    "error",
                    f"Note {index} (pitch {note.pitch}) has duration {note.duration}. "
                    "A note with no length is not a note.",
                    (index,),
                )
            )
        if not low_pitch <= note.pitch <= high_pitch:
            issues.append(
                Issue(
                    "pitch_out_of_range",
                    "error",
                    f"Note {index} has pitch {note.pitch}, outside MIDI {low_pitch}..{high_pitch}.",
                    (index,),
                )
            )
        if not 0 <= note.velocity <= FULL_VELOCITY:
            issues.append(
                Issue(
                    "velocity_out_of_range",
                    "error",
                    f"Note {index} has velocity {note.velocity}, outside 0..{FULL_VELOCITY}.",
                    (index,),
                )
            )
        elif note.velocity == 0:
            issues.append(
                Issue(
                    "zero_velocity",
                    "warning",
                    f"Note {index} has velocity 0, which is a note-off in the MIDI protocol. "
                    "Live's Piano Roll draws 1..127; whether the remote script passes a "
                    "0 through is unverified.",
                    (index,),
                )
            )
        if note.start_time < 0:
            issues.append(
                Issue(
                    "negative_start_time",
                    "error",
                    f"Note {index} starts at {note.start_time}, before the clip start. "
                    "Clip-local time begins at beat 0; a pickup goes into its own clip.",
                    (index,),
                )
            )
        elif note.start_time > TICKS_SUSPICION_BEATS:
            issues.append(
                Issue(
                    "looks_like_ticks",
                    "warning",
                    f"Note {index} starts at beat {note.start_time:.0f}, more than "
                    f"{TICKS_SUSPICION_BEATS / BEATS_PER_BAR_4_4:.0f} bars of 4/4. That "
                    f"looks like ticks ({TICKS_PER_QUARTER} per quarter). Run the list "
                    "through from_tick_notes() before writing it.",
                    (index,),
                )
            )

        probability = _extended(note, "probability")
        if (
            probability is not _MISSING
            and probability is not None
            and not 0.0 <= float(probability) <= 1.0
        ):
            issues.append(
                Issue(
                    "probability_out_of_range",
                    "warning",
                    f"Note {index} has probability {probability}, outside 0.0..1.0.",
                    (index,),
                )
            )

    _check_stack(issues, notes, stacked)
    if clip_length is not None:
        _check_clip_length(issues, notes, clip_length, time_tolerance)
    _check_overlaps(issues, notes, allow_overlap, time_tolerance)
    return report


def _check_stack(issues: list[Issue], notes: Sequence[Note], stacked: list[int]) -> None:
    """Report complete or partial beat-zero stacking artifacts."""
    if len(notes) >= 2 and len(stacked) == len(notes):
        issues.append(
            Issue(
                "stacked_on_beat_zero",
                "error",
                f"All {len(notes)} notes sit at start_time={LEGACY_DEFAULT_START_TIME} with "
                f"duration={LEGACY_DEFAULT_DURATION}, exactly the fingerprint of a note "
                "list built with the wrong dict keys. A reader that takes start_time and "
                "duration with .get(key, default) substitutes those two values without a "
                "word and stacks the whole clip on beat 0, and whatever wrote it "
                "reported success. "
                "Check the key names before writing: 'pos'/'dur' is humanize.py's tick "
                "form and has to go through from_tick_notes() first. If a stack of "
                f"{len(notes)} sixteenth notes on beat 0 is genuinely what was meant, this "
                "is a false positive, but check the keys anyway.",
                tuple(stacked),
            )
        )
    elif len(stacked) >= STACK_WARN_MIN_NOTES:
        issues.append(
            Issue(
                "partial_stack_on_beat_zero",
                "warning",
                f"{len(stacked)} of {len(notes)} notes carry the exact "
                f"({LEGACY_DEFAULT_START_TIME}, {LEGACY_DEFAULT_DURATION}) fingerprint of "
                "the wrong-keys default while the rest do not. That is the shape of "
                "one bad batch landing in an otherwise healthy clip. Measured precedent: a "
                "23-note batch stacked on top of 63 good notes and the clip ended up with "
                "86.",
                tuple(stacked),
            )
        )


def _check_clip_length(
    issues: list[Issue],
    notes: Sequence[Note],
    clip_length: float,
    tolerance: float,
) -> None:
    """Report note material extending beyond declared clip boundaries."""
    past_start = tuple(i for i, n in enumerate(notes) if n.start_time >= clip_length - tolerance)
    if past_start:
        issues.append(
            Issue(
                "starts_past_clip_end",
                "warning",
                f"{len(past_start)} note(s) start at or after the clip end "
                f"({clip_length} beats). Live keeps them (measured): a clip's declared "
                "length is regularly shorter than its note content, but they will never "
                "sound. Either lengthen the clip or drop them.",
                past_start,
            )
        )
    past_end = tuple(
        i
        for i, n in enumerate(notes)
        if n.start_time < clip_length - tolerance and _note_end(n) > clip_length + tolerance
    )
    if past_end:
        issues.append(
            Issue(
                "extends_past_clip_end",
                "warning",
                f"{len(past_end)} note(s) run past the clip end ({clip_length} beats). "
                "They sound, but the loop cuts their tail.",
                past_end,
            )
        )


def _check_overlaps(
    issues: list[Issue],
    notes: Sequence[Note],
    allow_overlap: bool,
    tolerance: float,
) -> None:
    """Report note overlap issues."""
    same_pitch = find_overlaps(notes, same_pitch_only=True, tolerance=tolerance)
    if same_pitch:
        flat = tuple(sorted({i for pair in same_pitch for i in pair}))
        issues.append(
            Issue(
                "same_pitch_overlap",
                "warning",
                f"{len(same_pitch)} overlapping pair(s) on the same pitch: a second "
                "note-on before the first note-off. Which one ends first is up to the "
                "instrument, and the answer is usually not the musical one.",
                flat,
            )
        )
    if allow_overlap:
        return
    everything = find_overlaps(notes, same_pitch_only=False, tolerance=tolerance)
    cross = [pair for pair in everything if notes[pair[0]].pitch != notes[pair[1]].pitch]
    if cross:
        flat = tuple(sorted({i for pair in cross for i in pair}))
        issues.append(
            Issue(
                "polyphonic_overlap",
                "warning",
                f"{len(cross)} overlapping pair(s) across different pitches, and this list "
                "was validated as monophonic material.",
                flat,
            )
        )


def validate_note_dicts(raw: Sequence[Any]) -> ValidationReport:
    """Validate raw note dictionary mappings prior to model conversion.

    Args:
        raw: Sequence of raw note dictionaries.

    Returns:
        ValidationReport containing errors or warnings for schema violations.
    """
    report = ValidationReport(note_count=len(raw))
    issues = report.issues

    if not raw:
        issues.append(
            Issue(
                "empty_note_list",
                "warning",
                "Empty note list. Written with mode='replace' this clears the clip.",
            )
        )
        return report

    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            issues.append(
                Issue(
                    "not_a_mapping",
                    "error",
                    f"Entry {index} is a {type(entry).__name__}, not a note dict: {entry!r}",
                    (index,),
                )
            )
            continue

        unknown = sorted(set(entry) - KNOWN_NOTE_KEYS - TOLERATED_NOTE_KEYS)
        if unknown:
            issues.append(
                Issue(
                    "unknown_key",
                    "error",
                    f"Entry {index} carries unknown key(s) {unknown}. Unknown keys raise "
                    "nothing and change nothing; a lenient reader drops them and falls "
                    f"back to its own defaults (start_time={LEGACY_DEFAULT_START_TIME}, "
                    f"duration={LEGACY_DEFAULT_DURATION}), which stacks the clip on beat 0 "
                    "and still reports success. 'pos'/'dur' is humanize.py's tick form: "
                    "run it through from_tick_notes().",
                    (index,),
                )
            )
        ignored = sorted(set(entry) & TOLERATED_NOTE_KEYS)
        if ignored:
            issues.append(
                Issue(
                    "ignored_key",
                    "warning",
                    f"Entry {index} carries {ignored}, which Live hands back on a read but "
                    "the note model does not write. It will be dropped.",
                    (index,),
                )
            )

        for key, legacy_default in (
            ("pitch", 60),
            ("start_time", LEGACY_DEFAULT_START_TIME),
            ("duration", LEGACY_DEFAULT_DURATION),
        ):
            if key not in entry:
                issues.append(
                    Issue(
                        "missing_key",
                        "error",
                        f"Entry {index} has no {key!r}. Do not carry on: a lenient reader "
                        f"substitutes {legacy_default} here silently and reports success.",
                        (index,),
                    )
                )
            elif not _is_number(entry[key]):
                issues.append(
                    Issue(
                        "non_numeric",
                        "error",
                        f"Entry {index} has {key}={entry[key]!r}, which is not a number.",
                        (index,),
                    )
                )

        if "velocity" not in entry:
            issues.append(
                Issue(
                    "missing_velocity",
                    "warning",
                    f"Entry {index} has no 'velocity'; {DEFAULT_VELOCITY} would be "
                    "substituted. That is Live's own drawing default and a measured one "
                    "(472 of 833 corpus tracks with a constant velocity use it), but say "
                    "it explicitly rather than inherit it.",
                    (index,),
                )
            )
        elif not _is_number(entry["velocity"]):
            issues.append(
                Issue(
                    "non_numeric",
                    "error",
                    f"Entry {index} has velocity={entry['velocity']!r}, not a number.",
                    (index,),
                )
            )

    return report


def _is_number(value: Any) -> bool:
    """Return True for int or float instances excluding booleans."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ------------------------------------------------------------------------ diffing


@dataclass(frozen=True)
class NoteChange:
    """Represents modified fields between a matched note pair.

    Attributes:
        before: Original Note instance.
        after: Modified Note instance.
        fields: Mapping of field name to (before_val, after_val) tuples.
    """

    before: Note
    after: Note
    fields: dict[str, tuple[Any, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert change to JSON-serializable dictionary."""
        return {
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "fields": {k: list(v) for k, v in self.fields.items()},
        }


@dataclass(frozen=True)
class NoteDiff:
    """Summary of differences between two note sequences.

    Attributes:
        added: Notes present in after but not in before.
        removed: Notes present in before but not in after.
        changed: Notes matched between lists with modified attributes.
        unchanged: Count of identical notes.
    """

    added: list[Note]
    removed: list[Note]
    changed: list[NoteChange]
    unchanged: int

    @property
    def is_empty(self) -> bool:
        """Return True if there are no additions, removals, or changes."""
        return not (self.added or self.removed or self.changed)

    def summary(self) -> str:
        """Return formatted single-line change summary."""
        return (
            f"{len(self.added)} added, {len(self.removed)} removed, "
            f"{len(self.changed)} changed, {self.unchanged} unchanged"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert diff to JSON-serializable dictionary."""
        return {
            "added": [n.to_dict() for n in self.added],
            "removed": [n.to_dict() for n in self.removed],
            "changed": [c.to_dict() for c in self.changed],
            "unchanged": self.unchanged,
            "summary": self.summary(),
        }


def diff(
    before: Sequence[Note],
    after: Sequence[Note],
    *,
    tolerance: float = GRID_TOLERANCE,
    match_on: MatchOn = "pitch_time",
    ignore_defaulted_extensions: bool = False,
) -> NoteDiff:
    """Compare two note sequences and report additions, removals, and changes.

    Args:
        before: Original Note sequence.
        after: Modified Note sequence.
        tolerance: Float beat tolerance for position comparisons.
        match_on: Match strategy ('pitch_time' or 'pitch').
        ignore_defaulted_extensions: When True, skip extensions absent on before side.

    Returns:
        NoteDiff object describing differences.

    Raises:
        ValueError: If match_on is unrecognized.
    """
    if match_on == "pitch_time":
        buckets = _bucket_by_pitch_time(before, tolerance)
        keys = _bucket_by_pitch_time(after, tolerance)
        return _pair_buckets(buckets, keys, tolerance, ignore_defaulted_extensions)
    if match_on == "pitch":
        return _pair_by_pitch(before, after, tolerance, ignore_defaulted_extensions)
    raise ValueError(f"match_on must be 'pitch_time' or 'pitch', not {match_on!r}")


def _bucket_by_pitch_time(
    notes: Sequence[Note], tolerance: float
) -> dict[tuple[int, int], list[Note]]:
    """Group notes by (pitch, start snapped to tolerance)."""
    buckets: dict[tuple[int, int], list[Note]] = {}
    for note in notes:
        key = (note.pitch, round(note.start_time / tolerance))
        buckets.setdefault(key, []).append(note)
    return buckets


def _pair_buckets(
    before: dict[tuple[int, int], list[Note]],
    after: dict[tuple[int, int], list[Note]],
    tolerance: float,
    ignore_defaulted: bool = False,
) -> NoteDiff:
    """Pair notes bucket by bucket; un-paired notes become additions or removals."""
    added: list[Note] = []
    removed: list[Note] = []
    changed: list[NoteChange] = []
    unchanged = 0

    for key in set(before) | set(after):
        old = sorted(before.get(key, []), key=lambda n: n.duration)
        new = sorted(after.get(key, []), key=lambda n: n.duration)
        for old_note, new_note in zip(old, new, strict=False):
            fields = _changed_fields(old_note, new_note, tolerance, ignore_defaulted)
            if fields:
                changed.append(NoteChange(before=old_note, after=new_note, fields=fields))
            else:
                unchanged += 1
        removed.extend(old[len(new) :])
        added.extend(new[len(old) :])

    return NoteDiff(
        added=sort_notes(added),
        removed=sort_notes(removed),
        changed=sorted(changed, key=lambda c: (c.after.start_time, c.after.pitch)),
        unchanged=unchanged,
    )


def _pair_by_pitch(
    before: Sequence[Note], after: Sequence[Note], tolerance: float,
    ignore_defaulted: bool = False,
) -> NoteDiff:
    """Pair notes per pitch in time order, so movement reads as a change."""
    old_by_pitch: dict[int, list[Note]] = {}
    new_by_pitch: dict[int, list[Note]] = {}
    for note in sort_notes(before):
        old_by_pitch.setdefault(note.pitch, []).append(note)
    for note in sort_notes(after):
        new_by_pitch.setdefault(note.pitch, []).append(note)

    added: list[Note] = []
    removed: list[Note] = []
    changed: list[NoteChange] = []
    unchanged = 0

    for pitch in set(old_by_pitch) | set(new_by_pitch):
        old = old_by_pitch.get(pitch, [])
        new = new_by_pitch.get(pitch, [])
        for old_note, new_note in zip(old, new, strict=False):
            fields = _changed_fields(old_note, new_note, tolerance, ignore_defaulted)
            if fields:
                changed.append(NoteChange(before=old_note, after=new_note, fields=fields))
            else:
                unchanged += 1
        removed.extend(old[len(new) :])
        added.extend(new[len(old) :])

    return NoteDiff(
        added=sort_notes(added),
        removed=sort_notes(removed),
        changed=sorted(changed, key=lambda c: (c.after.start_time, c.after.pitch)),
        unchanged=unchanged,
    )


def _changed_fields(
    before: Note, after: Note, tolerance: float, ignore_defaulted: bool = False
) -> dict[str, tuple[Any, Any]]:
    """Compare fields between notes, incorporating Live 11+ extensions."""
    fields: dict[str, tuple[Any, Any]] = {}
    for name in _CORE_FIELDS:
        old_value = getattr(before, name)
        new_value = getattr(after, name)
        if _differs(old_value, new_value, tolerance):
            fields[name] = (old_value, new_value)
    for name in _EXTENDED_FIELDS:
        old_value = _extended(before, name)
        new_value = _extended(after, name)
        if old_value is _MISSING and new_value is _MISSING:
            continue
        if ignore_defaulted and (old_value is _MISSING or old_value is None):
            continue
        old_plain = None if old_value is _MISSING else old_value
        new_plain = None if new_value is _MISSING else new_value
        if _differs(old_plain, new_plain, tolerance):
            fields[name] = (old_plain, new_plain)
    return fields


# ------------------------------------------------------------------- conversions


def beats_to_ticks(beats: float) -> int:
    """Convert beats to integer ticks at 480 ticks per quarter note.

    Args:
        beats: Duration in beats.

    Returns:
        Duration in integer ticks.

    >>> beats_to_ticks(1.5)
    720
    """
    return round(float(beats) * TICKS_PER_QUARTER)


def ticks_to_beats(ticks: float) -> float:
    """Convert integer ticks to quarter-note beats.

    Args:
        ticks: Duration in ticks.

    Returns:
        Duration in beats.

    >>> ticks_to_beats(720)
    1.5
    """
    return _clean(float(ticks) / TICKS_PER_QUARTER)


def from_tick_notes(
    tick_notes: Sequence[Mapping[str, Any]],
    *,
    velocity: int = DEFAULT_VELOCITY,
) -> list[Note]:
    """Convert tick-format dictionaries (pos, dur, pitch) into Note models.

    Args:
        tick_notes: Sequence of note mappings in tick format.
        velocity: Fallback velocity assigned if a note does not specify one.

    Returns:
        Sorted list of Note model instances.

    Raises:
        ValueError: If any entry is missing required pos, dur, or pitch keys.
    """
    out: list[Note] = []
    for index, entry in enumerate(tick_notes):
        for key in ("pos", "dur", "pitch"):
            if key not in entry:
                raise ValueError(
                    f"tick note {index} has no {key!r}: {dict(entry)!r}. Nothing is "
                    "substituted here: a missing pos or dur is enough to "
                    "stack a whole clip on beat 0."
                )
        note = Note(
            pitch=int(entry["pitch"]),
            start_time=ticks_to_beats(entry["pos"]),
            duration=ticks_to_beats(entry["dur"]),
            velocity=int(entry.get("velocity", velocity)),
            mute=bool(entry.get("mute", False)),
        )
        out.append(note)
    return sort_notes(out)


def to_tick_notes(notes: Sequence[Note]) -> list[dict[str, Any]]:
    """Convert Note models into tick-format dictionaries (pos, dur, pitch).

    Args:
        notes: Sequence of Note models.

    Returns:
        List of note mappings in tick format.
    """
    return [
        {
            "pos": beats_to_ticks(n.start_time),
            "dur": beats_to_ticks(n.duration),
            "pitch": int(n.pitch),
            "velocity": int(n.velocity),
            "mute": bool(n.mute),
        }
        for n in notes
    ]


def from_beat_tuples(
    seq: Sequence[tuple[float, int, float]],
    *,
    velocity: int = DEFAULT_VELOCITY,
) -> list[Note]:
    """Convert (start_beat, pitch, duration_beats) tuples into Note models.

    Args:
        seq: Sequence of (start_beat, pitch, duration_beats) tuples.
        velocity: Velocity value assigned to generated Note models.

    Returns:
        Sorted list of Note models.
    """
    return sort_notes(
        Note(
            pitch=int(pitch),
            start_time=_clean(start),
            duration=_clean(duration),
            velocity=int(velocity),
            mute=False,
        )
        for start, pitch, duration in seq
    )


def from_dicts(raw: Sequence[Mapping[str, Any]]) -> list[Note]:
    """Convert protocol note dictionaries into Note models via Note.from_dict.

    Args:
        raw: Sequence of note mappings.

    Returns:
        Sorted list of Note models.
    """
    return sort_notes(Note.from_dict(dict(entry)) for entry in raw)


def without_tolerated_keys(
    raw: Sequence[Any],
) -> tuple[list[Any], list[str]]:
    """Filter out non-model keys present in TOLERATED_NOTE_KEYS from note dicts.

    Measured 2026-09-02 against Live 12.4.5: read_clip_notes includes note_id
    on note dictionaries, which must be stripped before writing back to avoid schema errors.

    Args:
        raw: Sequence of raw note dictionaries or objects.

    Returns:
        Tuple of (filtered_notes, list_of_removed_key_names).

    >>> notes, dropped = without_tolerated_keys(
    ...     [{"pitch": 60, "start_time": 0.0, "duration": 1.0, "note_id": 7}]
    ... )
    >>> notes
    [{'pitch': 60, 'start_time': 0.0, 'duration': 1.0}]
    >>> dropped
    ['note_id']
    """
    dropped: set[str] = set()
    kept: list[Any] = []
    for entry in raw:
        # dict, not Mapping, to match validate_note_dicts -- the two run over the
        # same list and must agree on what counts as a note.
        if not isinstance(entry, dict):
            kept.append(entry)
            continue
        clean = {}
        for key, value in entry.items():
            if key in TOLERATED_NOTE_KEYS:
                dropped.add(key)
                continue
            clean[key] = value
        kept.append(clean)
    return kept, sorted(dropped)


def to_dicts(notes: Sequence[Note]) -> list[dict[str, Any]]:
    """Convert Note models to protocol-compliant dictionaries.

    Args:
        notes: Sequence of Note models.

    Returns:
        List of note dictionaries sorted by start_time and pitch.
    """
    return [n.to_dict() for n in sort_notes(notes)]

