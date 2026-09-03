"""Humanization, micro-timing, swing, and voice staggering for MIDI note lists.

Provides pure operations over note lists in standard 480 ticks/quarter resolution:
legato smoothing, breath pauses, swing grids, velocity shaping, and multi-voice jitter.

The interventions:
1. LEGATO       Close gaps inside a phrase. Without touching notes, vocal synthesizers
                or legato instruments cannot transition smoothly between syllables.
2. BREATH       Before every phrase, shorten the previous note rather than delaying
                the next one to keep rhythmic timing intact.
3. SWING        Delay even grid subdivisions for swing/groove feels.
4. DECOUPLE     Stagger multi-voice unison or chord arrangements with per-note random
                timing jitter rather than static voice offsets.
5. ATTACK       Velocity contouring over time (:func:`velocity_curve`) and metric
                grid accentuation (:func:`accent`).

Time units:
- Internal representation: integer ticks (480 per quarter note, ``TickNote``).
- Ableton representation: float beats (1.0 per quarter note, :class:`~ableton_maestro.models.Note`).
- Psychoacoustic parameters (breath duration, jitter, stagger): specified in milliseconds
  and converted using the tempo (BPM).
- Always use :func:`to_ableton` before sending notes to Live.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from ableton_maestro.models import Note

__all__ = [
    "ACCENT_1_AND_3",
    "BAR_TICKS",
    "DEFAULT_BREATH_MS",
    "DEFAULT_END_TRIM_MS",
    "DEFAULT_JITTER_MS",
    "DEFAULT_MIN_DURATION_MS",
    "DEFAULT_STAGGER_MS",
    "GRID_EIGHTH",
    "GRID_QUARTER",
    "GRID_SIXTEENTH",
    "OFFBEAT_EIGHTH",
    "OFFBEAT_SIXTEENTH",
    "PATTERNS",
    "PATTERN_3_3_2",
    "SWING_LIGHT",
    "SWING_TRIPLET",
    "TICKS_PER_QUARTER",
    "VELOCITY_BLOCKS_MEASURED",
    "VELOCITY_DEFAULT",
    "VELOCITY_FULL",
    "VELOCITY_MEASURED",
    "VELOCITY_RANGE",
    "TickNote",
    "accent",
    "beats_to_ticks",
    "from_ableton",
    "from_beats",
    "humanize",
    "micro_timing",
    "ms_to_ticks",
    "pattern_notes",
    "report",
    "stagger_voices",
    "swing",
    "ticks_to_beats",
    "ticks_to_ms",
    "to_ableton",
    "velocity_curve",
    "velocity_report",
]

TICKS_PER_QUARTER = 480

# Internal tick representation of a note: pos, dur, and pitch are integers,
# with pos and dur in ticks. Extra fields (velocity, mute, lyric, etc.) pass
# through untouched. Convert to Ableton Note instances using to_ableton().
TickNote = dict[str, Any]

# Default timing intervals in milliseconds (at 126 BPM, 350 ms is approximately 353 ticks).
DEFAULT_BREATH_MS = 350.0
DEFAULT_MIN_DURATION_MS = 90.0
DEFAULT_STAGGER_MS = 8.0
DEFAULT_END_TRIM_MS = 40.0
# Per-note jitter standard deviation across voices (reference: ~15 ms). See stagger_voices().
DEFAULT_JITTER_MS = 15.0

SWING_TRIPLET = 1.0 / 3.0  # Shuffle ratio: eighth-note pair becomes 2:1
SWING_LIGHT = 0.15  # Light swing ratio

# --- Attack strength parameters ---
# Reference measurements (833 tracks across 103 projects): 648 tracks use a single
# velocity across clips (typically 100 as Live's drawing default, or 127 for full scale).
VELOCITY_DEFAULT = 100
VELOCITY_FULL = 127

# Precautionary clamp bounds (1 to 127; MIDI velocity 0 indicates note-off).
VELOCITY_RANGE = (1, 127)

# Track role baseline statistics: (track_count, single_velocity_pct, median_range, std_dev).
# Used for contextual comparison in velocity_report().
#                                  tracks  const%  range   SD
VELOCITY_MEASURED: dict[str, tuple[int, int, int | None, float | None]] = {
    "Lead": (46, 78, 30, 8.8),
    "Riff": (49, 63, 47, 16.4),
    "Pluck": (13, 100, None, None),
    "Arp": (20, 90, 15, 3.4),
    "Stab": (22, 95, 30, 7.4),
    "Chord": (67, 66, 27, 6.6),
    "Pad": (44, 89, 22, 6.9),
    "Bass": (186, 85, 47, 10.6),
    "Drums": (180, 55, 51, 17.1),
    "Trigger": (92, 98, 15, 4.8),
    "FX": (24, 88, 65, 32.5),
}

BAR_TICKS = TICKS_PER_QUARTER * 4  # 1920 ticks per 4/4 measure.

# --- Metric attack patterns ---
# Bar-local grid positions in ticks (evaluated as pos % BAR_TICKS) for accent().
GRID_QUARTER = (0, 480, 960, 1440)
GRID_EIGHTH = tuple(range(0, BAR_TICKS, 240))
GRID_SIXTEENTH = tuple(range(0, BAR_TICKS, 120))
ACCENT_1_AND_3 = (0, 960)
OFFBEAT_EIGHTH = tuple(range(240, BAR_TICKS, 480))
OFFBEAT_SIXTEENTH = tuple(range(120, BAR_TICKS, 240))

# 3-3-2 subdivision pattern: attacks at 0, 0.75, 1.5, 2.25, 3.0, 3.5 beats, that is
# 3+3+3+3+2+2 sixteenths. The 3/4-beat spacing (dotted eighth) is the most frequent
# interval across the seven MIDI templates, with 116 of 306 intervals (37.9 percent),
# and `Chord Melody 007` writes out exactly these six positions. At 126 BPM a 3/4 beat
# is 357.1 ms.
PATTERN_3_3_2 = (0, 360, 720, 1080, 1440, 1680)

PATTERNS: dict[str, tuple[int, ...]] = {
    "quarter": GRID_QUARTER,
    "eighth": GRID_EIGHTH,
    "sixteenth": GRID_SIXTEENTH,
    "1_and_3": ACCENT_1_AND_3,
    "offbeat_eighth": OFFBEAT_EIGHTH,
    "offbeat_sixteenth": OFFBEAT_SIXTEENTH,
    "3_3_2": PATTERN_3_3_2,
}

# Empirical block velocity automation reference.
VELOCITY_BLOCKS_MEASURED: dict[str, tuple[tuple[int, int], ...]] = {
    "chord_melody_001": ((0, 99), (2 * BAR_TICKS, 90)),
}


# --------------------------------------------------------------- unit conversion
def ms_to_ticks(ms: float, bpm: float) -> int:
    """Convert duration from milliseconds to integer ticks at a given tempo.

    Args:
        ms: Duration in milliseconds.
        bpm: Beats per minute (must be positive).

    Returns:
        Integer duration in ticks (480 ticks/quarter).

    Raises:
        ValueError: If bpm is non-positive.
    """
    if bpm <= 0:
        raise ValueError("bpm must be positive")
    return round(ms * TICKS_PER_QUARTER * bpm / 60000.0)


def ticks_to_ms(ticks: int, bpm: float) -> float:
    """Convert duration from integer ticks to milliseconds at a given tempo.

    Args:
        ticks: Duration in ticks.
        bpm: Beats per minute (must be positive).

    Returns:
        Duration in milliseconds.

    Raises:
        ValueError: If bpm is non-positive.
    """
    if bpm <= 0:
        raise ValueError("bpm must be positive")
    return ticks * 60000.0 / (TICKS_PER_QUARTER * bpm)


def beats_to_ticks(beats: float) -> int:
    """Convert musical beat count to integer ticks (480 ticks per quarter note).

    Args:
        beats: Duration in quarter-note beats.

    Returns:
        Integer duration in ticks.
    """
    return round(beats * TICKS_PER_QUARTER)


def ticks_to_beats(ticks: int) -> float:
    """Convert integer ticks to quarter-note beat count.

    Args:
        ticks: Duration in ticks.

    Returns:
        Duration in quarter-note beats.
    """
    return ticks / float(TICKS_PER_QUARTER)


def _resolve(ticks: int | None, ms: float, bpm: float) -> int:
    """Resolve tick value directly if supplied, otherwise convert from milliseconds."""
    return int(ticks) if ticks is not None else ms_to_ticks(ms, bpm)


# ------------------------------------------------------------------------ timing
def humanize(
    notes: Sequence[TickNote],
    phrase_starts: Iterable[int] = (),
    bpm: float = 126.0,
    breath_ms: float = DEFAULT_BREATH_MS,
    shift_ms: float = 0.0,
    phrase_end_trim_ms: float = 0.0,
    min_duration_ms: float = DEFAULT_MIN_DURATION_MS,
    breath_ticks: int | None = None,
    shift_ticks: int | None = None,
    phrase_end_trim: int | None = None,
    min_duration: int | None = None,
) -> list[TickNote]:
    """Apply legato connection, breath pauses, and voice shifts in a single pass.

    Args:
        notes: Sequence of notes in tick format.
        phrase_starts: Note indices marking start of a new phrase (receives preceding breath).
        bpm: Tempo in beats per minute used for millisecond conversions.
        breath_ms: Duration in milliseconds for breath pauses before phrase starts.
        shift_ms: Global voice timing offset in milliseconds.
        phrase_end_trim_ms: Additional shortening in ms for phrase-final notes.
        min_duration_ms: Lower bound for note duration in milliseconds.
        breath_ticks: Optional explicit breath length in ticks (overrides breath_ms).
        shift_ticks: Optional explicit shift in ticks (overrides shift_ms).
        phrase_end_trim: Optional explicit end trim in ticks (overrides phrase_end_trim_ms).
        min_duration: Optional explicit minimum duration in ticks (overrides min_duration_ms).

    Returns:
        New list of notes in tick format with adjusted pos and dur fields.

    >>> notes = [{**n, "velocity": 42} for n in from_beats([(0.0, 69, 0.5),
    ...                                                     (1.0, 73, 0.5)])]
    >>> humanize(notes, phrase_starts={0}, bpm=126)[0]["velocity"]
    42
    """
    breath = _resolve(breath_ticks, breath_ms, bpm)
    shift = _resolve(shift_ticks, shift_ms, bpm)
    end_trim = _resolve(phrase_end_trim, phrase_end_trim_ms, bpm)
    min_dur = _resolve(min_duration, min_duration_ms, bpm)

    starts: set[int] = set(phrase_starts)
    ordered = sorted(notes, key=lambda n: n["pos"])
    out: list[TickNote] = []

    for i, note in enumerate(ordered):
        pos = note["pos"]
        end = pos + note["dur"]

        if i + 1 < len(ordered):
            next_pos = ordered[i + 1]["pos"]
            if (i + 1) in starts:
                end = min(end, next_pos - breath)  # breath
            else:
                end = next_pos  # legato
        if i + 1 == len(ordered) or (i + 1) in starts:
            end -= end_trim

        # min_duration must not eat the breath again: rather shorten the pause
        # than overlap the next note (monophonic line).
        dur = max(min_dur, end - pos)
        if i + 1 < len(ordered):
            dur = min(dur, max(1, ordered[i + 1]["pos"] - pos))

        out.append({**note, "pos": pos + shift, "dur": dur})
    return out


def _voice_rng(seed: int, voice_index: int) -> random.Random:
    """Return seeded random generator for one voice's jitter."""
    return random.Random(seed * 1000 + voice_index)


def stagger_voices(
    voice_notes: Sequence[Sequence[TickNote]],
    base_shifts: Sequence[int] | None = None,
    end_trims: Sequence[int] | None = None,
    phrase_starts: Iterable[int] = (),
    bpm: float = 126.0,
    breath_ms: float = DEFAULT_BREATH_MS,
    stagger_ms: float = DEFAULT_STAGGER_MS,
    end_trim_ms: float = DEFAULT_END_TRIM_MS,
    breath_ticks: int | None = None,
    jitter_ms: float = DEFAULT_JITTER_MS,
    seed: int = 0,
) -> list[list[TickNote]]:
    """Offset multiple voices against each other using fixed shifts and random jitter.

    Args:
        voice_notes: Sequence of note lists (one sequence per voice).
        base_shifts: Optional explicit tick offsets per voice.
        end_trims: Optional explicit phrase-end trims per voice.
        phrase_starts: Indices marking phrase starts.
        bpm: Tempo in beats per minute.
        breath_ms: Breath pause length in milliseconds.
        stagger_ms: Fixed progressive voice offset in milliseconds.
        end_trim_ms: Progressive phrase end trim in milliseconds.
        breath_ticks: Optional breath length in ticks.
        jitter_ms: Standard deviation of Gaussian per-note jitter in milliseconds.
        seed: Random seed for deterministic reproducibility.

    Returns:
        List of humanized note lists corresponding to input voices.

    Note:
        ``jitter_ms`` draws a fresh deviation around zero per note. That is what makes
        a choir: no singer is consistently late, every entry scatters anew.

        Measured on a backing vocal arrangement (three voices, 125 BPM): standard
        deviation 15.2 and 14.7 ms, range -20 to +19 ms, and across 33 notes 14 ahead
        of and 19 behind the reference. Hence ``jitter_ms = 15`` as the default.

        Notable in that example: both backing voices sat at median +11 and +12 ms, so
        not a rising offset per voice but a shared small offset that ``stagger_ms``
        alone would not produce.
    """
    count = len(voice_notes)
    step = ms_to_ticks(stagger_ms, bpm)
    trim_step = ms_to_ticks(end_trim_ms, bpm)
    jit = ms_to_ticks(jitter_ms, bpm)

    if base_shifts is None:
        base_shifts = [i * step for i in range(count)]
    if end_trims is None:
        end_trims = [i * trim_step for i in range(count)]

    out: list[list[TickNote]] = []
    for i, notes in enumerate(voice_notes):
        done = humanize(
            notes,
            phrase_starts=phrase_starts,
            bpm=bpm,
            breath_ms=breath_ms,
            breath_ticks=breath_ticks,
            shift_ticks=base_shifts[i],
            phrase_end_trim=end_trims[i],
        )
        if jit > 0 and i > 0:  # voice 0 stays the reference on the grid
            rnd = _voice_rng(seed, i)
            shifted: list[TickNote] = []
            for n in done:
                d = round(rnd.gauss(0.0, jit))
                # Capped at 1.35 sigma. The real example scatters with sigma
                # 15 ms yet only reaches -20..+19 ms, so the distribution is
                # clipped, not openly Gaussian.
                lim = round(1.35 * jit)
                d = max(-lim, min(lim, d))
                shifted.append({**n, "pos": max(0, n["pos"] + d)})
            done = shifted
        out.append(done)
    return out


def swing(
    notes: Sequence[TickNote],
    ratio: float = SWING_TRIPLET,
    grid: int = TICKS_PER_QUARTER // 2,
    keep_note_ends: bool = True,
) -> list[TickNote]:
    """Apply swing by delaying even metric grid subdivisions.

    Args:
        notes: Sequence of notes in tick format.
        ratio: Grid fraction to delay offbeats (e.g. 1/3 for triplet shuffle, 0.15 for light swing).
            Must be between 0.0 and 0.5.
        grid: Metric subdivision level in ticks (default 240 = eighth notes).
        keep_note_ends: When True, preserves note end positions and shortens offbeats.
            When False, shifts entire note duration.

    Returns:
        List of swung notes in tick format.

    Raises:
        ValueError: If ratio is outside 0.0..0.5 or grid is non-positive.

    >>> pairs = from_beats([(0.0, 60, 0.5), (0.5, 62, 0.5)])
    >>> [(n["pos"], n["dur"]) for n in swing(pairs, ratio=SWING_TRIPLET)]
    [(0, 320), (320, 160)]
    """
    if not 0.0 <= ratio <= 0.5:
        raise ValueError("ratio belongs between 0.0 and 0.5")
    if grid <= 0:
        raise ValueError("grid must be positive")

    delay = round(ratio * grid)
    if delay == 0:
        return [dict(n) for n in notes]

    pair = grid * 2
    tol = max(1, grid // 4)

    def is_offbeat(pos: int) -> bool:
        return abs((pos % pair) - grid) <= tol

    # 1) move the offbeats
    shifted: list[TickNote] = []
    moved_from: set[int] = set()
    for n in notes:
        if is_offbeat(n["pos"]):
            moved_from.add(n["pos"])
            dur = n["dur"] - delay if keep_note_ends else n["dur"]
            shifted.append({**n, "pos": n["pos"] + delay, "dur": max(1, dur)})
        else:
            shifted.append(dict(n))

    # 2) Stretch predecessor notes that ran into a moved onset, otherwise a gap
    #    tears open exactly where there used to be legato.
    for n in shifted:
        if n["pos"] + n["dur"] in moved_from and not is_offbeat(n["pos"]):
            n["dur"] += delay

    return shifted


def micro_timing(
    notes: Sequence[TickNote],
    ticks: int,
    phrase_starts: Iterable[int] = (),
    tail_weight: float = 0.4,
) -> list[TickNote]:
    """Apply systematic timing displacement across phrases.

    Positive values place notes behind the beat (laid back), negative values ahead (pushed).
    Maximum shift occurs at phrase starts and attenuates toward the grid by tail_weight.

    Args:
        notes: Sequence of notes in tick format.
        ticks: Maximum timing offset in ticks applied to phrase start.
        phrase_starts: Indices marking the beginning of phrases.
        tail_weight: Attenuation factor (0.0 to 1.0) applied to notes following phrase start.

    Returns:
        List of timing-adjusted notes in tick format.
    """
    starts = set(phrase_starts)
    out: list[TickNote] = []
    for i, note in enumerate(notes):
        weight = 1.0 if i in starts else tail_weight
        out.append({**note, "pos": max(0, note["pos"] + int(ticks * weight))})
    return out


# ------------------------------------------------------------------------ guards
def _require_tick_form(notes: Sequence[Any], where: str) -> None:
    """Validate that notes are in tick format dicts with 'pos' key."""
    for n in notes:
        if isinstance(n, Mapping) and "pos" in n:
            continue
        found = sorted(n.keys()) if isinstance(n, Mapping) else type(n).__name__
        raise ValueError(
            f"{where} expects the tick form {{pos, dur, pitch}}; this note has {found}. "
            "That is the Ableton form out of to_ableton(), which is too late at that point. "
            "Order: build notes -> attack -> timing -> to_ableton(). "
            "Use from_ableton() to come back."
        )


def _clamp_velocity(value: float, clamp: tuple[int, int] = VELOCITY_RANGE) -> int:
    lo, hi = int(clamp[0]), int(clamp[1])
    if lo > hi:
        raise ValueError("clamp belongs as (min, max) with min <= max")
    return max(lo, min(hi, round(value)))


def _field(note: Any, name: str, default: Any = None) -> Any:
    """Read one field from either note form (mapping key or dataclass attribute)."""
    if isinstance(note, Mapping):
        return note.get(name, default)
    return getattr(note, name, default)


# ------------------------------------------------------------------------ attack
def velocity_curve(
    notes: Sequence[TickNote],
    shape: str = "flat",
    base: int = VELOCITY_DEFAULT,
    depth: int = 0,
    steps: int = 0,
    blocks: Sequence[tuple[int, int]] | None = None,
    span: tuple[int, int] | None = None,
    clamp: tuple[int, int] = VELOCITY_RANGE,
) -> list[TickNote]:
    """Generate velocity contour over time (flat, ramp, arc, or stepped blocks).

    Sets the velocity field per note. Apply velocity_curve before accent() so
    that localized accents are added onto the contour.

    Args:
        notes: Sequence of notes in tick format.
        shape: Contour type ('flat', 'ramp', or 'arc').
        base: Baseline or starting velocity value. 100 is measured rather than chosen:
            472 of the 833 corpus tracks sit on it (VELOCITY_DEFAULT).
        depth: Dynamic range delta in velocity steps (signed).
        steps: Step rasterization count (0 = smooth per note, >0 = quantized blocks).
        blocks: Optional explicit (start_tick, velocity) blocks overriding shape and depth.
        span: Optional (start_tick, end_tick) bounds for the curve envelope.
        clamp: (min_vel, max_vel) bounds (defaults to 1..127).

    Returns:
        List of notes with updated velocity fields.

    Raises:
        ValueError: If shape is unrecognized or steps is negative.

    >>> notes = from_beats([(0.0, 60, 1.0), (1.0, 60, 1.0), (2.0, 60, 1.0)])
    >>> [n["velocity"] for n in velocity_curve(notes, shape="ramp", depth=-20)]
    [100, 90, 80]
    >>> notes = velocity_curve(notes, blocks=VELOCITY_BLOCKS_MEASURED["chord_melody_001"])
    >>> [n["velocity"] for n in notes]
    [99, 99, 99]
    """
    if not notes:
        return []
    _require_tick_form(notes, "velocity_curve()")
    if shape not in ("flat", "ramp", "arc"):
        raise ValueError("shape belongs to 'flat', 'ramp' or 'arc'")
    if steps < 0:
        raise ValueError("steps must not be negative")

    out: list[TickNote] = []

    if blocks:
        ordered_blocks = sorted(((int(t), int(v)) for t, v in blocks), key=lambda b: b[0])
        for n in notes:
            value: float = base
            for start, vel in ordered_blocks:
                if n["pos"] >= start:
                    value = vel
                else:
                    break
            out.append({**n, "velocity": _clamp_velocity(value, clamp)})
        return out

    if span is not None:
        t0, t1 = int(span[0]), int(span[1])
    else:
        t0 = min(n["pos"] for n in notes)
        t1 = max(n["pos"] for n in notes)
    width = t1 - t0

    for n in notes:
        x = 0.0 if width <= 0 else (n["pos"] - t0) / float(width)
        x = max(0.0, min(1.0, x))
        if steps > 0:
            k = min(steps - 1, int(x * steps))
            x = 0.0 if steps == 1 else k / float(steps - 1)
        if shape == "flat":
            value = float(base)
        elif shape == "ramp":
            value = base + depth * x
        else:  # "arc": parabola, crest in the middle
            value = base + depth * (4.0 * x * (1.0 - x))
        out.append({**n, "velocity": _clamp_velocity(value, clamp)})
    return out


def accent(
    notes: Sequence[TickNote],
    positions: Iterable[int],
    amount: int = 0,
    ghost: int = 0,
    base: int = VELOCITY_DEFAULT,
    bar_ticks: int = BAR_TICKS,
    tol: int = 0,
    clamp: tuple[int, int] = VELOCITY_RANGE,
) -> list[TickNote]:
    """Apply bar-local dynamic emphasis and ghosting to metric grid positions.

    Args:
        notes: Sequence of notes in tick format.
        positions: Bar-local tick positions (0 <= pos < bar_ticks) receiving accent amount.
        amount: Velocity delta added to accented grid positions.
        ghost: Velocity delta added to non-accented positions (typically negative).
        base: Default baseline velocity for notes lacking an explicit velocity field.
        bar_ticks: Measure length in ticks (defaults to 1920 for 4/4).
        tol: Position comparison tolerance in ticks.
        clamp: (min_vel, max_vel) clamping bounds.

    Returns:
        List of notes with accented velocity values.

    Raises:
        ValueError: If bar_ticks is non-positive or tol is negative.

    >>> hats = pattern_notes(GRID_EIGHTH, pitch=42, dur=240, velocity=100)
    >>> [n["velocity"] for n in accent(hats, ACCENT_1_AND_3, amount=10, ghost=-20)]
    [110, 80, 80, 80, 110, 80, 80, 80]
    """
    if not notes:
        return []
    _require_tick_form(notes, "accent()")
    if bar_ticks <= 0:
        raise ValueError("bar_ticks must be positive")
    if tol < 0:
        raise ValueError("tol must not be negative")

    grid = sorted(int(p) % bar_ticks for p in positions)
    out: list[TickNote] = []
    for n in notes:
        p = n["pos"] % bar_ticks
        hit = False
        for q in grid:
            d = abs(p - q)
            if min(d, bar_ticks - d) <= tol:  # the bar line is round
                hit = True
                break
        value = int(n.get("velocity", base)) + (amount if hit else ghost)
        out.append({**n, "velocity": _clamp_velocity(value, clamp)})
    return out


def pattern_notes(
    positions: Iterable[int],
    pitch: int,
    dur: int | None = None,
    bars: int = 1,
    bar_ticks: int = BAR_TICKS,
    velocity: int | None = None,
    offset: int = 0,
) -> list[TickNote]:
    """Convert bar-local rhythmic grid positions into playable note lists in ticks.

    Args:
        positions: Bar-local tick positions (0 <= pos < bar_ticks).
        pitch: MIDI note pitch number.
        dur: Optional note duration in ticks (if None, notes extend to next onset).
        bars: Number of measures to repeat the pattern.
        bar_ticks: Measure length in ticks (defaults to 1920 for 4/4).
        velocity: Optional fixed velocity assigned to generated notes.
        offset: Global timing shift in ticks applied to the whole pattern.

    Returns:
        List of generated notes in tick format.

    Raises:
        ValueError: If bar_ticks <= 0, bars < 1, or positions exceed bar bounds.

    >>> hats = pattern_notes(GRID_SIXTEENTH, pitch=42, dur=120, bars=4)
    >>> len(hats)
    64
    >>> [(n["pos"], n["dur"]) for n in pattern_notes(PATTERN_3_3_2, pitch=73, dur=240)][:3]
    [(0, 240), (360, 240), (720, 240)]
    """
    if bar_ticks <= 0:
        raise ValueError("bar_ticks must be positive")
    if bars < 1:
        raise ValueError("bars must be at least 1")
    grid = sorted(int(p) for p in positions)
    if not grid:
        return []
    if grid[0] < 0 or grid[-1] >= bar_ticks:
        raise ValueError(f"positions are bar-local: 0 <= p < {bar_ticks}")

    onsets = [offset + b * bar_ticks + p for b in range(bars) for p in grid]
    end = offset + bars * bar_ticks
    out: list[TickNote] = []
    for i, pos in enumerate(onsets):
        if dur is None:
            nxt = onsets[i + 1] if i + 1 < len(onsets) else end
            length = max(1, nxt - pos)
        else:
            length = max(1, int(dur))
        note: TickNote = {"pos": pos, "dur": length, "pitch": int(pitch)}
        if velocity is not None:
            note["velocity"] = _clamp_velocity(velocity)
        out.append(note)
    return out


def velocity_report(notes: Sequence[Any], role: str | None = None) -> str:
    """Analyze velocity distribution across notes against reference profiles.

    Args:
        notes: Sequence of notes in tick format or Ableton Note model format.
        role: Optional instrument role key from VELOCITY_MEASURED ('Lead', 'Bass', etc.).

    Returns:
        Formatted summary string describing levels, range, median, and standard deviation.

    Raises:
        ValueError: If role is not present in reference dataset.

    >>> velocity_report(pattern_notes(GRID_QUARTER, pitch=42, velocity=100))
    'velocity: 4/4 notes carry one | levels 1 (100..100, range 0) | median 100, \
SD 0.0 | single-level - the measured normal case'
    """
    vals = [int(v) for v in (_field(n, "velocity") for n in notes) if v is not None]
    if not vals:
        return (
            f"velocity: 0 of {len(notes)} notes carry one - "
            "to_ableton() then applies its fallback value"
        )

    ordered = sorted(vals)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    mean = sum(ordered) / float(n)
    sd = (sum((v - mean) ** 2 for v in ordered) / n) ** 0.5
    steps = len(set(ordered))

    text = (
        f"velocity: {n}/{len(notes)} notes carry one | "
        f"levels {steps} ({ordered[0]}..{ordered[-1]}, range {ordered[-1] - ordered[0]}) | "
        f"median {median:g}, SD {sd:.1f}"
    )
    if steps == 1:
        text += " | single-level - the measured normal case"
    if role:
        if role not in VELOCITY_MEASURED:
            known = ", ".join(sorted(VELOCITY_MEASURED))
            raise ValueError(f"role {role!r} was not measured. Known: {known}")
        tracks, const, span, msd = VELOCITY_MEASURED[role]
        if span is None:
            single = round(tracks * const / 100.0)
            text += (
                f" || corpus {role}: {single} of {tracks} tracks single-level "
                f"({const}%), none variable to compare against"
            )
        else:
            text += (
                f" || corpus {role}: {const}% of {tracks} tracks single-level; "
                f"when variable, range {span}, SD {msd:.1f}"
            )
    return text


# ---------------------------------------------------------------- format bridges
def from_beats(seq: Sequence[tuple[float, int, float]]) -> list[TickNote]:
    """Convert (beat, pitch, duration_beats) tuples to note list in ticks.

    Args:
        seq: Sequence of (start_beat, pitch, duration_beat) tuples.

    Returns:
        List of note mappings in tick format.

    >>> from_beats([(1.5, 64, 0.5)])
    [{'pos': 720, 'dur': 240, 'pitch': 64}]
    """
    return [
        {"pos": beats_to_ticks(b), "dur": beats_to_ticks(d), "pitch": int(p)} for (b, p, d) in seq
    ]


def to_ableton(notes: Sequence[TickNote], velocity: int = VELOCITY_DEFAULT) -> list[Note]:
    """Convert notes from tick format (480 ticks/quarter) to Ableton beat models.

    Converts tick pos and dur into quarter-note beat floats expected by the
    Ableton Remote Script wire protocol.

    Args:
        notes: Sequence of notes in tick format.
        velocity: Fallback velocity value for notes without an explicit velocity field.

    Returns:
        List of ableton_maestro.models.Note instances.

    Note:
        Skipping this conversion sends ticks where beats are expected, which is 480 times
        too large: beat 1 becomes beat 480, and writing beat 160 into a 16-bar clip puts
        the note past the end.

        The write on the other side defaults to ``mode="replace"``, implemented as
        remove-then-write inside one handler call. Live's own ``clip.set_notes()`` appends
        instead, which is why writing hi-hats in two passes silently doubles them
        (measured on the artefact 2026-08-22: 63 notes + 23 added = 86). Appending is
        still reachable but has to be asked for by name; the safe habit is to send the
        complete list in one call.
    """
    from ableton_maestro.models import Note  # local: music/ must not grow a hard cycle

    out: list[Note] = []
    for n in notes:
        fields: dict[str, Any] = {
            "pitch": int(n["pitch"]),
            "start_time": ticks_to_beats(n["pos"]),
            "duration": ticks_to_beats(n["dur"]),
            "velocity": int(n.get("velocity", velocity)),
        }
        if "mute" in n:
            fields["mute"] = bool(n["mute"])
        for extra in ("probability", "velocity_deviation", "release_velocity"):
            if extra in n:
                fields[extra] = n[extra]
        out.append(Note(**fields))
    return out


def from_ableton(notes: Sequence[Any]) -> list[TickNote]:
    """Convert Ableton beat models or wire mappings back into integer tick notes.

    Converts beat-based start_time and duration back to 480 ticks/quarter resolution.

    Args:
        notes: Sequence of Note models or wire-format dictionaries.

    Returns:
        List of note mappings in tick format.
    """
    out: list[TickNote] = []
    for n in notes:
        note: TickNote = {
            "pos": beats_to_ticks(float(_field(n, "start_time", 0.0))),
            "dur": beats_to_ticks(float(_field(n, "duration", 0.0))),
            "pitch": int(_field(n, "pitch", 0)),
        }
        velocity = _field(n, "velocity")
        if velocity is not None:
            note["velocity"] = round(velocity)
        for extra in ("mute", "probability", "velocity_deviation", "release_velocity"):
            value = _field(n, extra)
            if value is not None:
                note[extra] = value
        out.append(note)
    return out


def report(before: Sequence[TickNote], after: Sequence[TickNote]) -> str:
    """Generate concise comparison summary of note count, gaps, and total duration.

    Args:
        before: Sequence of notes prior to processing.
        after: Sequence of notes after processing.

    Returns:
        Formatted summary string.
    """

    def gaps(ns: Sequence[TickNote]) -> int:
        s = sorted(ns, key=lambda n: n["pos"])
        return sum(1 for a, b in pairwise(s) if a["pos"] + a["dur"] < b["pos"])

    total_before = max((n["pos"] + n["dur"] for n in before), default=0)
    total_after = max((n["pos"] + n["dur"] for n in after), default=0)
    return (
        f"notes {len(after)} | gaps before {gaps(before)}, after {gaps(after)} | "
        f"total length {total_before} -> {total_after} ticks"
    )
