"""Core domain models, enums, and data types for the Ableton Maestro protocol.

Defines wire protocol enums, shared data structures, Note models, and verification
types without dependencies on transport or external libraries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any

__all__ = [
    "NOTE_EXTENDED_FIELDS",
    "NOTE_FIELDS",
    "Access",
    "ErrorCode",
    "Interpolation",
    "Kind",
    "LaunchMode",
    "LaunchQuantization",
    "LomHandle",
    "MonitoringState",
    "Note",
    "NoteMode",
    "PathStatus",
    "SetResult",
    "Unit",
    "WarpMode",
    "validate_notes",
]


class PathStatus(str, Enum):
    """Verification status of a catalog path specification."""

    VERIFIED = "verified"   # Probed and verified against a running Live instance
    BROKEN = "broken"       # Probed and failed (e.g. path invalid or read-only)
    UNTESTED = "untested"   # Cataloged but not yet probed


class Access(str, Enum):
    """Permitted operations on a LOM path."""

    GET = "get"
    SET = "set"
    CALL = "call"
    OBSERVE = "observe"
    AUTOMATE = "automate"


class Kind(str, Enum):
    """Wire data type of a LOM value."""

    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STR = "str"
    ENUM = "enum"
    LIST = "list"
    OBJECT = "object"


class Unit(str, Enum):
    """Physical or normalized measurement unit of a parameter."""

    NORMALIZED = "normalized"
    DB = "db"
    HZ = "hz"
    SEMITONES = "semitones"
    BEATS = "beats"
    SECONDS = "seconds"
    PERCENT = "percent"
    NONE = "none"


class Interpolation(str, Enum):
    """Interpolation curve types for automation envelopes (docs/protocol.md §5.9)."""

    LINEAR = "linear"
    HOLD = "hold"
    EXPONENTIAL = "exponential"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"


class NoteMode(str, Enum):
    """Write mode for notes_set handler (docs/protocol.md §5.8)."""

    REPLACE = "replace"
    APPEND = "append"


class ErrorCode(str, Enum):
    """Structured error codes returned by the Remote Script (docs/protocol.md §4)."""

    UNKNOWN_HANDLER = "unknown_handler"
    BAD_PATH = "bad_path"
    NO_SUCH_PATH = "no_such_path"
    INDEX_OUT_OF_RANGE = "index_out_of_range"
    NOT_SETTABLE = "not_settable"
    METHOD_NOT_ALLOWED = "method_not_allowed"
    TYPE_ERROR = "type_error"
    LIVE_ERROR = "live_error"
    INTERNAL = "internal"
    SKIPPED = "skipped"


class WarpMode(IntEnum):
    """Clip warp mode enum matching Live.Clip.WarpMode (Live 12.4.5)."""

    BEATS = 0
    TONES = 1
    TEXTURE = 2
    REPITCH = 3
    COMPLEX = 4
    REX = 5
    COMPLEX_PRO = 6


class MonitoringState(IntEnum):
    """Track monitoring state enum matching Track.current_monitoring_state.

    Values:
        IN (0): Passes the input at all times, armed or not.
        AUTO (1): Passes the input only while the track is armed.
        OFF (2): Playback only. The ordinary setting for a recorded audio track.

    Measured 2026-08-31 against Live 12.4.5 by setting the property and reading
    Live's own switch; 1 follows by arithmetic. Live exposes no enum type for this
    property, so enum_names cannot reach it. The catalog row
    track.current_monitoring_state carries the corpus reading behind OFF.

    Note: A track fed by another track must sit on IN. On AUTO an unarmed track
    passes nothing through, which presents as silence with every other setting
    apparently correct.
    """

    IN = 0
    AUTO = 1
    OFF = 2


class LaunchMode(IntEnum):
    """Clip launch mode enum matching Live.Clip.LaunchMode."""

    TRIGGER = 0
    GATE = 1
    TOGGLE = 2
    REPEAT = 3


class LaunchQuantization(IntEnum):
    """Clip launch quantization enum matching Live.Clip.ClipLaunchQuantization.

    Measured 2026-09-01 against Live 12.4.5 through the enum_names handler
    (docs/protocol.md section 5.12).

    Note: This is not the enum song.clip_trigger_quantization uses. That property
    reads Live.Song.Quantization, which opens with q_no_q at 0, where this one opens
    with q_global at 0 and q_none at 1. Every shared member therefore sits one
    HIGHER here: one bar is 4 there and 5 here. The spellings differ as well -
    q_eighth, q_sixteenth and q_thirtysecond here against q_eight, q_sixtenth and
    q_thirtytwoth there. A value carried from one property to the other mis-sets it
    silently.
    """

    GLOBAL = 0
    NONE = 1
    BARS_8 = 2
    BARS_4 = 3
    BARS_2 = 4
    BAR = 5
    HALF = 6
    HALF_TRIPLET = 7
    QUARTER = 8
    QUARTER_TRIPLET = 9
    EIGHTH = 10
    EIGHTH_TRIPLET = 11
    SIXTEENTH = 12
    SIXTEENTH_TRIPLET = 13
    THIRTY_SECOND = 14


#: Base note fields supported across all Live versions.
NOTE_FIELDS: frozenset[str] = frozenset(
    {"pitch", "start_time", "duration", "velocity", "mute"}
)

#: Live 11+ note extension fields.
NOTE_EXTENDED_FIELDS: frozenset[str] = frozenset(
    {"probability", "velocity_deviation", "release_velocity"}
)


@dataclass(slots=True)
class Note:
    """Represents a single MIDI note in a clip.

    Times and durations are in clip-relative beats.

    Attributes:
        pitch: MIDI note number (0..127).
        start_time: Start time in beats relative to clip start.
        duration: Note length in beats (> 0).
        velocity: Velocity amount (1.0..127.0, default 100.0).
        mute: True if note is muted.
        probability: Note trigger probability (0.0..1.0, Live 11+).
        velocity_deviation: Velocity randomization range (-127.0..127.0, Live 11+).
        release_velocity: Note-off velocity (0.0..127.0, Live 11+).
    """

    pitch: int
    start_time: float
    duration: float
    velocity: float = 100.0
    mute: bool = False
    probability: float | None = None
    velocity_deviation: float | None = None
    release_velocity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize note to wire dictionary format (docs/protocol.md §5.8)."""
        data: dict[str, Any] = {
            "pitch": int(self.pitch),
            "start_time": float(self.start_time),
            "duration": float(self.duration),
            "velocity": float(self.velocity),
            "mute": bool(self.mute),
        }
        if self.probability is not None:
            data["probability"] = float(self.probability)
        if self.velocity_deviation is not None:
            data["velocity_deviation"] = float(self.velocity_deviation)
        if self.release_velocity is not None:
            data["release_velocity"] = float(self.release_velocity)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, index: int | None = None) -> Note:
        """Construct a Note from a wire dictionary.

        Requires pitch, start_time, and duration.
        """
        where = "note" if index is None else f"note {index}"
        if not isinstance(data, dict):
            raise ValueError(  # noqa: TRY004
                f"{where}: expected an object, got {type(data).__name__}"
            )

        allowed = NOTE_FIELDS | NOTE_EXTENDED_FIELDS
        unknown = sorted(k for k in data if k not in allowed)
        if unknown:
            raise ValueError(
                f"{where}: unknown key(s) {unknown}; valid keys are {sorted(allowed)}."
            )
        missing = sorted(k for k in ("pitch", "start_time", "duration") if k not in data)
        if missing:
            raise ValueError(
                f"{where}: missing required key(s) {missing}."
            )
        return cls(
            pitch=data["pitch"],
            start_time=data["start_time"],
            duration=data["duration"],
            velocity=data.get("velocity", 100.0),
            mute=bool(data.get("mute", False)),
            probability=data.get("probability"),
            velocity_deviation=data.get("velocity_deviation"),
            release_velocity=data.get("release_velocity"),
        )

    def validate(self, *, index: int | None = None) -> None:
        """Validate note values and bounds, raising ValueError upon error."""
        where = "note" if index is None else f"note {index}"
        problems: list[str] = []

        if isinstance(self.pitch, bool) or not isinstance(self.pitch, int):
            problems.append(f"pitch must be an int, got {self.pitch!r}")
        elif not 0 <= self.pitch <= 127:
            problems.append(f"pitch {self.pitch} is outside 0..127")

        for name in ("start_time", "duration", "velocity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(f"{name} must be a number, got {value!r}")
            elif not math.isfinite(float(value)):
                problems.append(f"{name} is {value!r}; NaN and infinity never reach Live")

        if _is_finite_number(self.start_time) and self.start_time < 0.0:
            problems.append(
                f"start_time {self.start_time} is negative; clip-local time starts at 0"
            )
        if _is_finite_number(self.duration) and self.duration <= 0.0:
            problems.append(f"duration {self.duration} must be greater than 0")
        if _is_finite_number(self.velocity) and not 1.0 <= float(self.velocity) <= 127.0:
            problems.append(f"velocity {self.velocity} is outside 1..127")

        problems.extend(_check_optional(self.probability, "probability", 0.0, 1.0))
        problems.extend(
            _check_optional(self.velocity_deviation, "velocity_deviation", -127.0, 127.0)
        )
        problems.extend(
            _check_optional(self.release_velocity, "release_velocity", 0.0, 127.0)
        )

        if problems:
            raise ValueError(f"{where}: " + "; ".join(problems))


def _is_finite_number(value: Any) -> bool:
    """Return True if value is a finite int or float (excluding bools)."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _check_optional(value: Any, name: str, low: float, high: float) -> list[str]:
    """Validate optional float field against bounds (min, max)."""
    if value is None:
        return []
    if not _is_finite_number(value):
        return [f"{name} must be a finite number or None, got {value!r}"]
    if not low <= float(value) <= high:
        return [f"{name} {value} is outside {low}..{high}"]
    return []


def validate_notes(notes: list[Note], *, allow_degenerate: bool = False) -> None:
    """Validate a list of Note objects before sending to Live.

    Performs individual note validation and checks for structural anomalies:
    - 3 or more notes stacked simultaneously at start_time 0.0 with 0.25 duration.
    - Duplicate note events with identical pitch and start_time.

    Args:
        notes: List of Note instances to validate.
        allow_degenerate: If True, bypasses anomaly checks.

    Raises:
        ValueError: If any note fails validation or an anomaly is detected.
    """
    for i, note in enumerate(notes):
        note.validate(index=i)

    if allow_degenerate or len(notes) < 2:
        return

    if len(notes) >= 3 and all(
        float(n.start_time) == 0.0 and float(n.duration) == 0.25 for n in notes
    ):
        raise ValueError(
            f"all {len(notes)} notes sit at start_time=0.0 with duration=0.25. That is "
            "the fingerprint of note keys that were never read, defaulted instead "
            "(measured). Check the key spelling against Note.to_dict(); pass "
            "allow_degenerate=True if this really is a stack of 16ths on beat 0."
        )

    seen: set[tuple[int, float]] = set()
    for i, note in enumerate(notes):
        key = (int(note.pitch), float(note.start_time))
        if key in seen:
            raise ValueError(
                f"note {i}: pitch {note.pitch} appears twice at start_time "
                f"{note.start_time}. Duplicated notes are the residue of an additive "
                "write (measured: 63 + 23 = 86 notes in one clip). Pass "
                "allow_degenerate=True to send them anyway."
            )
        seen.add(key)


@dataclass(slots=True)
class SetResult:
    """Result of a lom_set operation, carrying its read-back verification.

    See docs/protocol.md §5.4 for the wire shape.

    Attributes:
        path: Path that was modified.
        requested: Value passed into the set request.
        before: Value prior to the write.
        after: Value read back after the write.
        clamped: True if Live adjusted or clamped the requested value.
        changed: True if the value differed between before and after.
        display: Optional formatted display string from Live parameter.
    """

    path: str
    requested: Any
    before: Any
    after: Any
    clamped: bool
    changed: bool
    display: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SetResult:
        """Construct SetResult from the result payload of a lom_set reply."""
        requested = data.get("requested")
        before = data.get("before")
        after = data.get("after")
        return cls(
            path=str(data.get("path", "")),
            requested=requested,
            before=before,
            after=after,
            clamped=bool(data["clamped"]) if "clamped" in data else after != requested,
            changed=bool(data["changed"]) if "changed" in data else after != before,
            display=data.get("display"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize SetResult to dictionary representation."""
        data: dict[str, Any] = {
            "path": self.path,
            "requested": self.requested,
            "before": self.before,
            "after": self.after,
            "clamped": self.clamped,
            "changed": self.changed,
        }
        if self.display is not None:
            data["display"] = self.display
        return data

    def summary(self) -> str:
        """Return a single-line summary string describing the write outcome."""
        parts = [f"{self.path}: {self.before!r} -> {self.after!r}"]
        if self.clamped:
            parts.append(f"(requested {self.requested!r}, CLAMPED by Live)")
        if not self.changed:
            parts.append("(no change: it already held this value)")
        if self.display is not None:
            parts.append(f"[{self.display}]")
        return " ".join(parts)


@dataclass(slots=True, frozen=True)
class LomHandle:
    """Wire representation of an unresolved Live Object reference.

    See docs/protocol.md §7 for protocol details.

    Attributes:
        lom_class: Live Python class name (e.g. 'Track', 'Clip', 'DeviceParameter').
        path: Canonical LOM path to the object.
        name: Human-readable object name if available.
    """

    lom_class: str
    path: str
    name: str | None = None

    @staticmethod
    def is_handle(value: Any) -> bool:
        """Return True if value dictionary contains the '__lom__' key."""
        return isinstance(value, dict) and "__lom__" in value

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LomHandle:
        """Construct LomHandle from wire dictionary."""
        if not cls.is_handle(data):
            raise ValueError(f"not a LOM handle (no '__lom__' key): {data!r}")
        return cls(
            lom_class=str(data["__lom__"]),
            path=str(data.get("path", "")),
            name=data.get("name"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize LomHandle to wire dictionary."""
        data: dict[str, Any] = {"__lom__": self.lom_class, "path": self.path}
        if self.name is not None:
            data["name"] = self.name
        return data
