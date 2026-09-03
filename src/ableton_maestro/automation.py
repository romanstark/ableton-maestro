"""Client-side curve model for clip automation generation and verification.

Provides breakpoint curves (:class:`Curve`, :class:`Point`), curve builders (:func:`ramp`,
:func:`hold`, :func:`envelope`, :func:`adsr`, :func:`lfo`, :func:`probe_ramp`, :func:`merge`,
:func:`densify`), grid prediction (:func:`plan_steps`), and sampling/verification
(:func:`sample`, :func:`compare`, :func:`read_window`).

Operational rules:
- Live parameter default at t=0: ``automation_read`` returns the parameter's default value
  at t=0 rather than the curve. Verification via :func:`read_window` and :func:`compare`
  starts at :data:`READ_EPSILON` (> 0).
- Session clip envelope restriction: ``clip.automation_envelope(param)`` is available
  on Session clips only (returns None on Arrangement clips).
- Parameter write overwrites: writing automation to a parameter replaces any existing
  envelope on that parameter. Multiple segments should be combined with :func:`merge`.
- Median-based verification: :func:`compare` evaluates median error to avoid single-sample
  outlier bias.

Examples:
>>> curve = ramp(0.15, 0.95, beats=64.0, interpolation=Interpolation.EASE_IN)
>>> len(curve.points)
2
>>> round(curve.value_at(32.0), 4)
0.35
>>> plan_steps(curve, resolution=0.25).steps
257
>>> read_window(curve)[0] > 0.0          # never verify at t = 0
True
"""

from __future__ import annotations

import math
import random
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from statistics import fmean, median
from typing import Any, Self

from ableton_maestro.models import Interpolation

__all__ = [
    "DEFAULT_EXPONENT",
    "DEFAULT_READ_SAMPLES",
    "DEFAULT_RESOLUTION",
    "DEFAULT_TOLERANCE",
    "FILTER_HZ_MAX",
    "FILTER_HZ_MIN",
    "FLAT_EPSILON",
    "MAX_POINTS",
    "MAX_READ_SAMPLES",
    "MAX_STEPS",
    "MIN_READ_SAMPLES",
    "MIN_RESOLUTION",
    "READ_EPSILON",
    "SESSION_ONLY_NOTE",
    "WIRE_DECIMALS",
    "Comparison",
    "Curve",
    "LfoShape",
    "ParameterRange",
    "Point",
    "Sample",
    "StepPlan",
    "Verdict",
    "adsr",
    "clamp",
    "compare",
    "denormalise",
    "densify",
    "envelope",
    "filter_hz_to_normalised",
    "hold",
    "interpolation_name",
    "lfo",
    "merge",
    "normalise",
    "normalised_to_filter_hz",
    "parse_interpolation",
    "plan_steps",
    "probe_ramp",
    "ramp",
    "read_window",
    "sample",
    "to_write_params",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_EXPONENT = 2.0
"""Default exponent applied to exponential, ease_in, and ease_out curves."""

DEFAULT_RESOLUTION = 1.0 / 16.0
"""Default automation write step width in beats (1/16 note)."""

MIN_RESOLUTION = 1.0 / 128.0
"""Minimum supported automation step width in beats (1/128 note)."""

MAX_STEPS = 4000
"""Maximum step limit enforced per automation write request."""

MAX_POINTS = MAX_STEPS
"""Maximum breakpoint limit accepted in a single automation write request."""

READ_EPSILON = 1.0 / 64.0
"""Offset in beats (1/64 note) used to avoid parameter default artifact at t=0.

Measured: ``value_at_time(0.0)`` returns the parameter default, not the curve. A 1/64
beat is finer than the finest write resolution (:data:`MIN_RESOLUTION` = 1/128 is the
floor, 1/16 the default) and therefore still lands inside the first segment. Measured
against a piano-lead ride: reading at 0 gave ``first_value`` 0.4165 instead of 0.8997
and ``value_span`` 0.4588 instead of 0.4741.
"""

DEFAULT_READ_SAMPLES = 64
"""Default sample count for automation read operations."""

MIN_READ_SAMPLES = 2
"""Minimum allowed sample count for automation read operations."""

MAX_READ_SAMPLES = 512
"""Maximum allowed sample count for automation read operations."""

FLAT_EPSILON = 1e-9
"""Threshold below which parameter variance is considered static."""

DEFAULT_TOLERANCE = 0.02
"""Default error tolerance for curve verification comparisons.

Deliberately generous. Measured reference for what a good write looks like: 1024 points
restored from a backup came back with a median deviation of 0.0002 and a maximum of
0.0010.
"""

WIRE_DECIMALS = 6
"""Decimal precision applied when formatting values for transmission."""

FILTER_HZ_MIN = 20.0
"""Lower frequency bound in Hz for logarithmic filter parameters.

Auto Filter and EQ Eight ``Frequency`` run 0..1 over FILTER_HZ_MIN..FILTER_HZ_MAX,
logarithmically. Measured against the .als held next to ``automation_read``:
18939 Hz maps to 0.9921 and 13692 Hz to 0.9451.
"""

FILTER_HZ_MAX = 20000.0
"""Upper frequency bound in Hz for logarithmic filter parameters."""

SESSION_ONLY_NOTE = (
    "Automation exists only in Session clips: clip.automation_envelope() returns None for "
    "Arrangement clips. Write into the Session clip first, then duplicate to the Arrangement; "
    "never the reverse, and never into a copy that already exists."
)

_INTERPOLATION_ALIASES: dict[str, str] = {
    "linear": "LINEAR",
    "lin": "LINEAR",
    "hold": "HOLD",
    "step": "HOLD",
    "constant": "HOLD",
    "none": "HOLD",
    "exponential": "EXPONENTIAL",
    "exp": "EXPONENTIAL",
    "ease_in": "EASE_IN",
    "easein": "EASE_IN",
    "in": "EASE_IN",
    "logarithmic": "EASE_OUT",
    "log": "EASE_OUT",
    "ease_out": "EASE_OUT",
    "easeout": "EASE_OUT",
    "out": "EASE_OUT",
}


def interpolation_name(mode: Interpolation) -> str:
    """Return protocol wire string for an Interpolation enum member.

    Args:
        mode: Interpolation enum value.

    Returns:
        Lowercase string representation suitable for serialization.
    """
    value = getattr(mode, "value", None)
    return value if isinstance(value, str) else mode.name.lower()


def parse_interpolation(mode: Interpolation | str | None) -> Interpolation:
    """Parse interpolation mode string or enum into canonical Interpolation instance.

    Args:
        mode: Interpolation mode string, enum, or None (defaults to LINEAR).

    Returns:
        Interpolation enum member.

    Raises:
        ValueError: If interpolation mode string is unrecognized.
    """
    if mode is None:
        return Interpolation.LINEAR
    if isinstance(mode, Interpolation):
        return mode
    key = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
    member = _INTERPOLATION_ALIASES.get(key)
    if member is None:
        known = ", ".join(sorted(interpolation_name(m) for m in Interpolation))
        raise ValueError(f"unknown interpolation {mode!r}; use one of: {known}")
    return Interpolation[member]


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #


def _eased(fraction: float, mode: Interpolation, exponent: float) -> float:
    """Compute normalized progression fraction using specified easing curvature."""
    if fraction <= 0.0:
        return 0.0
    if fraction >= 1.0:
        return 1.0
    if mode is Interpolation.HOLD:
        return 0.0
    if mode is Interpolation.LINEAR:
        return fraction
    curve = exponent if exponent and exponent > 0.0 else DEFAULT_EXPONENT
    if mode is Interpolation.EASE_OUT:
        curve = 1.0 / curve
    return float(fraction**curve)


def _blend(
    start: float, end: float, fraction: float, mode: Interpolation, exponent: float
) -> float:
    """Interpolate between start and end values across fractional progress."""
    if fraction <= 0.0:
        return start
    if fraction >= 1.0:
        return end
    return start + (end - start) * _eased(fraction, mode, exponent)


def clamp(value: float, minimum: float | None = None, maximum: float | None = None) -> float:
    """Constrain value within optional lower and upper bounds."""
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _round(value: float) -> float:
    return round(float(value), WIRE_DECIMALS)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Point:
    """Represents a single automation breakpoint and optional segment curve settings.

    Attributes:
        time: Position in clip-local beats.
        value: Target parameter value.
        interpolation: Optional per-segment interpolation override.
        exponent: Optional per-segment curvature exponent override.
    """

    time: float
    value: float
    interpolation: Interpolation | None = None
    exponent: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", float(self.time))
        object.__setattr__(self, "value", float(self.value))
        if not math.isfinite(self.time) or not math.isfinite(self.value):
            raise ValueError(f"breakpoint must be finite, got time={self.time} value={self.value}")
        if self.time < 0.0:
            raise ValueError(f"breakpoint time must not be negative, got {self.time}")
        if self.interpolation is not None:
            object.__setattr__(self, "interpolation", parse_interpolation(self.interpolation))
        if self.exponent is not None:
            object.__setattr__(self, "exponent", float(self.exponent))

    def as_wire(self, *, explicit: bool = False) -> dict[str, Any]:
        """Format point as dictionary compliant with automation_write protocol."""
        out: dict[str, Any] = {"time": _round(self.time), "value": _round(self.value)}
        if explicit and self.interpolation is not None:
            out["interpolation"] = interpolation_name(self.interpolation)
            if self.exponent is not None:
                out["exponent"] = float(self.exponent)
        return out


class Sample(tuple[float, float]):
    """Represents a discrete (time, value) automation reading."""

    __slots__ = ()

    def __new__(cls, time: float, value: float) -> Self:
        return super().__new__(cls, (float(time), float(value)))

    @property
    def time(self) -> float:
        """Position in clip-local beats."""
        return self[0]

    @property
    def value(self) -> float:
        """Raw parameter value reading."""
        return self[1]

    def __repr__(self) -> str:
        return f"Sample(time={self.time!r}, value={self.value!r})"


PointLike = Point | Mapping[str, Any] | Sequence[float]


def _coerce_point(
    raw: PointLike,
    *,
    position: int,
    interpolation: Interpolation | None,
    exponent: float | None,
) -> Point:
    """Convert point mapping, tuple, or Point instance into a Point."""
    if isinstance(raw, Point):
        return raw
    if isinstance(raw, Mapping):
        time = raw.get("time", raw.get("beat"))
        value = raw.get("value")
        if time is None or value is None:
            raise ValueError(f"breakpoint {position} needs both 'time' and 'value': {raw!r}")
        mode = raw.get("interpolation", interpolation)
        exp = raw.get("exponent", exponent)
        return Point(
            float(time),
            float(value),
            parse_interpolation(mode) if mode is not None else None,
            float(exp) if exp is not None else None,
        )
    if isinstance(raw, Sequence) and not isinstance(raw, str | bytes) and len(raw) >= 2:
        return Point(float(raw[0]), float(raw[1]), interpolation, exponent)
    raise ValueError(
        f"breakpoint {position} must be a Point, a mapping with 'time'/'value', "
        f"or a [time, value] pair; got {raw!r}"
    )


@dataclass(frozen=True, slots=True)
class Curve:
    """Represents a sequence of automation breakpoints with interpolation settings.

    Attributes:
        points: Sequence of Point breakpoints sorted in ascending time order.
        interpolation: Default interpolation mode between breakpoints.
        exponent: Default curvature exponent for exponential easing segments.
    """

    points: tuple[Point, ...]
    interpolation: Interpolation = Interpolation.LINEAR
    exponent: float = DEFAULT_EXPONENT

    def __post_init__(self) -> None:
        object.__setattr__(self, "interpolation", parse_interpolation(self.interpolation))
        object.__setattr__(self, "exponent", float(self.exponent))
        points = tuple(self.points)
        if not points:
            raise ValueError("a curve needs at least one breakpoint")
        if len(points) > MAX_POINTS:
            raise ValueError(
                f"too many breakpoints ({len(points)}, max {MAX_POINTS}); the script rejects "
                "this outright rather than coarsening it; describe the shape with "
                "interpolation + resolution instead of sampling it yourself"
            )
        object.__setattr__(self, "points", tuple(sorted(points, key=lambda p: p.time)))

    # -- shape ------------------------------------------------------------- #

    def resolved(self) -> tuple[Point, ...]:
        """Return breakpoints with curve interpolation and exponent defaults applied."""
        return tuple(
            Point(
                p.time,
                p.value,
                p.interpolation if p.interpolation is not None else self.interpolation,
                p.exponent if p.exponent is not None else self.exponent,
            )
            for p in self.points
        )

    @property
    def start(self) -> float:
        """Time of first breakpoint in beats."""
        return self.points[0].time

    @property
    def end(self) -> float:
        """Time of final breakpoint in beats."""
        return self.points[-1].time

    @property
    def duration(self) -> float:
        """Total span between first and last breakpoint in beats."""
        return self.end - self.start

    @property
    def values(self) -> tuple[float, ...]:
        """Breakpoint values ordered chronologically."""
        return tuple(p.value for p in self.points)

    @property
    def value_range(self) -> tuple[float, float]:
        """Tuple of (min_value, max_value) across all breakpoints."""
        values = self.values
        return min(values), max(values)

    @property
    def span(self) -> float:
        """Difference between highest and lowest breakpoint value."""
        low, high = self.value_range
        return high - low

    def is_flat(self, epsilon: float = FLAT_EPSILON) -> bool:
        """Return True if total curve variation is below flat threshold."""
        return self.span < epsilon

    @property
    def has_mixed_modes(self) -> bool:
        """Return True if curve breakpoints utilize differing interpolation settings."""
        resolved = self.resolved()
        first = (resolved[0].interpolation, resolved[0].exponent)
        return any((p.interpolation, p.exponent) != first for p in resolved[1:])

    def value_at(self, time: float) -> float:
        """Evaluate curve value at beat offset time."""
        resolved = self.resolved()
        times = [p.time for p in resolved]
        if time <= times[0]:
            return resolved[bisect_right(times, times[0]) - 1].value
        if time >= times[-1]:
            return resolved[-1].value
        index = bisect_right(times, time) - 1
        left = resolved[index]
        right = resolved[index + 1]
        segment = right.time - left.time
        if segment <= 0.0:
            return right.value
        fraction = (time - left.time) / segment
        return _blend(
            left.value,
            right.value,
            fraction,
            left.interpolation or self.interpolation,
            left.exponent if left.exponent is not None else self.exponent,
        )

    # -- transforms -------------------------------------------------------- #

    def clamped(self, minimum: float | None, maximum: float | None) -> Curve:
        """Return copy with all breakpoint values constrained to [minimum, maximum]."""
        return Curve(
            tuple(
                Point(p.time, clamp(p.value, minimum, maximum), p.interpolation, p.exponent)
                for p in self.points
            ),
            interpolation=self.interpolation,
            exponent=self.exponent,
        )

    def shifted(self, delta: float) -> Curve:
        """Return copy shifted along the time axis by delta beats."""
        return Curve(
            tuple(Point(p.time + delta, p.value, p.interpolation, p.exponent) for p in self.points),
            interpolation=self.interpolation,
            exponent=self.exponent,
        )

    def with_interpolation(self, mode: Interpolation | str, exponent: float | None = None) -> Curve:
        """Return copy configured with specified default interpolation mode."""
        return Curve(
            self.points,
            interpolation=parse_interpolation(mode),
            exponent=self.exponent if exponent is None else float(exponent),
        )

    # -- wire -------------------------------------------------------------- #

    def wire_points(self, *, explicit: bool | None = None) -> list[dict[str, Any]]:
        """Format curve breakpoints for automation_write request."""
        emit = self.has_mixed_modes if explicit is None else explicit
        resolved = self.resolved() if emit else self.points
        return [p.as_wire(explicit=emit) for p in resolved]

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Iterator[Point]:
        return iter(self.points)


def to_write_params(
    curve: Curve,
    *,
    resolution: float | None = DEFAULT_RESOLUTION,
    explicit: bool | None = None,
) -> dict[str, Any]:
    """Construct parameters dictionary for automation_write protocol call.

    Args:
        curve: Source Curve instance.
        resolution: Step spacing in beats.
        explicit: Whether to include explicit per-point interpolation metadata.

    Returns:
        Dictionary suitable for transmission in automation_write request.
    """
    params: dict[str, Any] = {
        "points": curve.wire_points(explicit=explicit),
        "interpolation": interpolation_name(curve.interpolation),
        "exponent": float(curve.exponent),
    }
    if resolution is not None:
        params["resolution"] = float(resolution)
    return params


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def envelope(
    points: Iterable[PointLike],
    *,
    interpolation: Interpolation | str = Interpolation.LINEAR,
    exponent: float = DEFAULT_EXPONENT,
) -> Curve:
    """Construct a Curve instance from points, mappings, or tuples.

    Args:
        points: Sequence of (time, value) pairs, mappings, or Point instances.
        interpolation: Default interpolation mode for curve segments.
        exponent: Default curvature exponent for easing modes.

    Returns:
        Constructed Curve instance.

    Raises:
        ValueError: If points sequence is empty, invalid, or exceeds maximum points.
    """
    mode = parse_interpolation(interpolation)
    collected = [
        _coerce_point(raw, position=index, interpolation=None, exponent=None)
        for index, raw in enumerate(points)
    ]
    return Curve(tuple(collected), interpolation=mode, exponent=float(exponent))


def ramp(
    start_value: float,
    end_value: float,
    beats: float,
    *,
    start_time: float = 0.0,
    interpolation: Interpolation | str = Interpolation.LINEAR,
    exponent: float = DEFAULT_EXPONENT,
) -> Curve:
    """Construct a two-point transition curve between start_value and end_value.

    Args:
        start_value: Starting parameter value.
        end_value: Target parameter value at conclusion.
        beats: Duration of transition in beats.
        start_time: Initial offset in clip-local beats.
        interpolation: Interpolation mode for segment.
        exponent: Curvature exponent for easing modes.

    Returns:
        Two-point Curve representing the ramp.

    Raises:
        ValueError: If duration in beats is non-positive.
    """
    if beats <= 0.0:
        raise ValueError(f"beats must be greater than 0, got {beats}")
    return envelope(
        [(start_time, start_value), (start_time + beats, end_value)],
        interpolation=interpolation,
        exponent=exponent,
    )


def hold(value: float, beats: float, *, start_time: float = 0.0) -> Curve:
    """Construct a static hold curve maintaining a constant value across beats.

    Args:
        value: Constant parameter value to hold.
        beats: Duration in beats.
        start_time: Initial offset in clip-local beats.

    Returns:
        Curve instance with two identical points configured for hold interpolation.

    Raises:
        ValueError: If duration in beats is non-positive.
    """
    if beats <= 0.0:
        raise ValueError(f"beats must be greater than 0, got {beats}")
    return envelope(
        [(start_time, value), (start_time + beats, value)],
        interpolation=Interpolation.HOLD,
    )


def adsr(
    *,
    attack: float,
    decay: float,
    sustain: float,
    release: float,
    sustain_beats: float = 0.0,
    peak: float = 1.0,
    floor: float = 0.0,
    start_time: float = 0.0,
    attack_shape: Interpolation | str = Interpolation.EASE_OUT,
    decay_shape: Interpolation | str = Interpolation.EASE_IN,
    release_shape: Interpolation | str = Interpolation.EASE_IN,
    exponent: float = DEFAULT_EXPONENT,
) -> Curve:
    """Construct an Attack-Decay-Sustain-Release automation curve.

    Args:
        attack: Attack phase duration in beats.
        decay: Decay phase duration in beats.
        sustain: Sustain parameter level.
        release: Release phase duration in beats.
        sustain_beats: Duration to hold sustain level before release.
        peak: Peak value reached at apex of attack.
        floor: Baseline value at curve start and completion.
        start_time: Initial offset in clip-local beats.
        attack_shape: Interpolation mode for attack segment.
        decay_shape: Interpolation mode for decay segment.
        release_shape: Interpolation mode for release segment.
        exponent: Curvature exponent for easing segments.

    Returns:
        Constructed Curve representing ADSR envelope.

    Raises:
        ValueError: If any phase duration is negative or all durations are zero.
    """
    stages = {"attack": attack, "decay": decay, "release": release, "sustain_beats": sustain_beats}
    for name, length in stages.items():
        if length < 0.0:
            raise ValueError(f"{name} must not be negative, got {length}")
    if attack + decay + sustain_beats + release <= 0.0:
        raise ValueError("an ADSR shape needs at least one stage with a length")

    exp = float(exponent)
    t_peak = start_time + attack
    t_sustain = t_peak + decay
    t_release = t_sustain + sustain_beats

    points = [
        Point(start_time, floor, parse_interpolation(attack_shape), exp),
        Point(t_peak, peak, parse_interpolation(decay_shape), exp),
        Point(t_sustain, sustain, Interpolation.HOLD, exp),
    ]
    if sustain_beats > 0.0:
        points.append(Point(t_release, sustain, parse_interpolation(release_shape), exp))
    else:
        points[-1] = Point(t_sustain, sustain, parse_interpolation(release_shape), exp)
    points.append(Point(t_release + release, floor))
    return Curve(tuple(points), interpolation=Interpolation.LINEAR, exponent=exp)


class LfoShape(str, Enum):
    """Periodic waveform shapes supported by lfo generator."""

    SINE = "sine"
    TRIANGLE = "triangle"
    SAW_UP = "saw_up"
    SAW_DOWN = "saw_down"
    SQUARE = "square"
    RANDOM = "random"


def lfo(
    shape: LfoShape | str,
    beats: float,
    *,
    cycles: float = 1.0,
    low: float = 0.0,
    high: float = 1.0,
    start_time: float = 0.0,
    duty: float = 0.5,
    points_per_cycle: int = 16,
    seed: int | None = None,
) -> Curve:
    """Construct periodic LFO automation waveform across specified duration.

    Args:
        shape: Target waveform (sine, triangle, saw_up, saw_down, square, random).
        beats: Total duration in beats.
        cycles: Number of repetitions across duration.
        low: Minimum parameter value.
        high: Maximum parameter value.
        start_time: Initial offset in clip-local beats.
        duty: Duty cycle fraction for square waveform.
        points_per_cycle: Sample points per period for sine synthesis.
        seed: Optional PRNG seed for deterministic random shape generation.

    Returns:
        Curve instance generating the periodic waveform.

    Raises:
        ValueError: If beats or cycles is non-positive, or parameters are invalid.
    """
    kind = LfoShape(shape) if not isinstance(shape, LfoShape) else shape
    if beats <= 0.0:
        raise ValueError(f"beats must be greater than 0, got {beats}")
    if cycles <= 0.0:
        raise ValueError(f"cycles must be greater than 0, got {cycles}")
    if not 0.0 < duty < 1.0:
        raise ValueError(f"duty must lie strictly between 0 and 1, got {duty}")
    if points_per_cycle < 2:
        raise ValueError(f"points_per_cycle must be at least 2, got {points_per_cycle}")

    period = beats / cycles
    whole = max(1, math.ceil(cycles))
    mid = (low + high) / 2.0
    amplitude = (high - low) / 2.0
    end_time = start_time + beats
    points: list[Point] = []

    def add(time: float, value: float, mode: Interpolation | None = None) -> None:
        if time <= end_time + 1e-9:
            points.append(Point(min(time, end_time), value, mode))

    if kind is LfoShape.SINE:
        total = max(2, round(points_per_cycle * cycles) + 1)
        for index in range(total):
            time = start_time + beats * index / (total - 1)
            phase = 2.0 * math.pi * (time - start_time) / period
            add(time, mid + amplitude * math.sin(phase))
    elif kind is LfoShape.TRIANGLE:
        add(start_time, low)
        for cycle in range(whole):
            base = start_time + cycle * period
            add(base + period / 2.0, high)
            add(base + period, low)
    elif kind in (LfoShape.SAW_UP, LfoShape.SAW_DOWN):
        rise, fall = (low, high) if kind is LfoShape.SAW_UP else (high, low)
        for cycle in range(whole):
            base = start_time + cycle * period
            add(base, rise)
            add(base + period, fall)
            # A second breakpoint on the same beat is the reset. The script
            # skips the zero-length segment and writes it as a jump.
            if base + period < end_time - 1e-9:
                add(base + period, rise)
    elif kind is LfoShape.SQUARE:
        for cycle in range(whole):
            base = start_time + cycle * period
            add(base, high, Interpolation.HOLD)
            add(base + duty * period, low, Interpolation.HOLD)
        add(end_time, low if points and points[-1].value == low else high, Interpolation.HOLD)
    else:  # LfoShape.RANDOM: sample and hold
        rng = random.Random(seed)
        for cycle in range(whole):
            add(start_time + cycle * period, rng.uniform(low, high), Interpolation.HOLD)
        add(end_time, points[-1].value if points else low, Interpolation.HOLD)

    return Curve(tuple(points), interpolation=Interpolation.LINEAR)


def probe_ramp(
    beats: float,
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
    start_time: float = 0.0,
    margin: float = 0.25,
) -> Curve:
    """Construct a test ramp offset from parameter boundaries for verification.

    Starts at offset margin to avoid extreme values and the Live parameter default artifact at t=0.

    Args:
        beats: Duration of probe ramp in beats.
        minimum: Lower bound of target parameter range.
        maximum: Upper bound of target parameter range.
        start_time: Initial offset in clip-local beats.
        margin: Safety margin inward from bounds as a fraction of range.

    Returns:
        Two-point Curve spanning [margin, 1 - margin] within parameter range.

    Raises:
        ValueError: If maximum does not exceed minimum or margin is out of (0, 0.5).
    """
    if maximum <= minimum:
        raise ValueError(f"maximum must exceed minimum, got [{minimum}, {maximum}]")
    if not 0.0 < margin < 0.5:
        raise ValueError(f"margin must lie strictly between 0 and 0.5, got {margin}")
    low = denormalise(margin, minimum, maximum)
    high = denormalise(1.0 - margin, minimum, maximum)
    return ramp(low, high, beats, start_time=start_time)


def merge(*curves: Curve, gap: float = 1e-6) -> Curve:
    """Merge multiple curves sequentially on a single parameter into one Curve.

    Overriding curves replace earlier curve segments within their active time range,
    inserting boundary points at the seam to preserve slopes.

    Args:
        *curves: One or more Curve instances in priority order (later curves override).
        gap: Time buffer in beats around transition boundaries.

    Returns:
        Consolidated Curve instance containing merged points.

    Raises:
        ValueError: If no curves are provided.
    """
    if not curves:
        raise ValueError("merge needs at least one curve")
    merged = curves[0]
    for override in curves[1:]:
        base = merged.resolved()
        low, high = override.start, override.end
        kept = [p for p in base if p.time < low - gap or p.time > high + gap]
        if base[0].time < low - gap and low - gap >= 0.0:
            seam = low - gap
            kept.append(Point(seam, merged.value_at(seam), *_mode_at(merged, seam)))
        if base[-1].time > high + gap:
            seam = high + gap
            kept.append(Point(seam, merged.value_at(seam), *_mode_at(merged, seam)))
        kept.extend(override.resolved())
        merged = Curve(tuple(kept), interpolation=merged.interpolation, exponent=merged.exponent)
    return merged


def _mode_at(curve: Curve, time: float) -> tuple[Interpolation, float]:
    """Return interpolation mode and exponent governing the curve segment at time."""
    resolved = curve.resolved()
    times = [p.time for p in resolved]
    index = max(0, bisect_right(times, time) - 1)
    point = resolved[index]
    return (
        point.interpolation or curve.interpolation,
        point.exponent if point.exponent is not None else curve.exponent,
    )


def densify(
    curve: Curve,
    *,
    resolution: float = DEFAULT_RESOLUTION,
    max_points: int = MAX_POINTS,
) -> Curve:
    """Sample curve into densely spaced linear breakpoint segments.

    Args:
        curve: Source Curve instance to resample.
        resolution: Step spacing in beats between points.
        max_points: Maximum allowed points in resulting curve.

    Returns:
        Curve instance with linear interpolation and resampled breakpoints.

    Raises:
        ValueError: If resolution is non-positive or max_points is less than 2.
    """
    if resolution <= 0.0:
        raise ValueError(f"resolution must be greater than 0, got {resolution}")
    if max_points < 2:
        raise ValueError(f"max_points must be at least 2, got {max_points}")
    if curve.duration <= 0.0:
        return curve

    intervals = max(1, math.ceil(curve.duration / resolution))
    intervals = min(intervals, max_points - 1)
    points = [
        Point(
            _round(curve.start + curve.duration * index / intervals),
            _round(curve.value_at(curve.start + curve.duration * index / intervals)),
        )
        for index in range(intervals + 1)
    ]
    # Pin endpoints to prevent rounding drift.
    points[0] = Point(_round(curve.start), _round(curve.points[0].value))
    points[-1] = Point(_round(curve.end), _round(curve.points[-1].value))
    return Curve(tuple(points), interpolation=Interpolation.LINEAR)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def read_window(curve: Curve, *, epsilon: float = READ_EPSILON) -> tuple[float, float]:
    """Calculate start and end beat boundaries for automation_read verification.

    Offsets the starting beat to epsilon (> 0) to avoid Live returning parameter
    defaults at t=0.

    Args:
        curve: Target Curve to evaluate.
        epsilon: Minimum start offset in beats when curve starts at t=0.

    Returns:
        Tuple of (start_beat, end_beat).
    """
    start = curve.start if curve.start > 0.0 else epsilon
    end = curve.end
    if end <= start:
        end = start + 1.0
    return start, end


def sample(
    curve: Curve,
    n: int = DEFAULT_READ_SAMPLES,
    *,
    start: float | None = None,
    end: float | None = None,
    mirror_read: bool = True,
) -> list[Sample]:
    """Sample curve values at evenly spaced beat intervals.

    Args:
        curve: Source Curve to evaluate.
        n: Number of samples to generate.
        start: Optional start time override in beats.
        end: Optional end time override in beats.
        mirror_read: Whether to replicate automation_read windowing and sample limits.

    Returns:
        List of Sample instances containing (time, value) pairs.

    Raises:
        ValueError: If n is less than MIN_READ_SAMPLES.
    """
    if n < MIN_READ_SAMPLES:
        raise ValueError(f"n must be at least {MIN_READ_SAMPLES}, got {n}")
    count = min(int(n), MAX_READ_SAMPLES) if mirror_read else int(n)

    if mirror_read:
        window_start, window_end = read_window(curve)
    else:
        window_start, window_end = curve.start, curve.end
        if window_end <= window_start:
            window_end = window_start + 1.0
    t0 = window_start if start is None else float(start)
    t1 = window_end if end is None else float(end)
    if t1 <= t0:
        t1 = t0 + 1.0

    step = (t1 - t0) / float(count - 1)
    return [
        Sample(_round(t0 + index * step), curve.value_at(t0 + index * step))
        for index in range(count)
    ]


# --------------------------------------------------------------------------- #
# Script step planning
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StepPlan:
    """Automation step generation plan predicted for automation_write execution.

    Attributes:
        steps: Total discrete step segments planned.
        resolution: Effective step spacing in beats.
        requested_resolution: Original requested step spacing in beats.
        point_count: Number of input breakpoints.
        coarsened: True if resolution was coarsened to obey MAX_STEPS.
        notes: Diagnostic messages and operational warnings.
    """

    steps: int
    resolution: float
    requested_resolution: float
    point_count: int
    coarsened: bool
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return dictionary representation suitable for serialization."""
        return {
            "steps": self.steps,
            "resolution": self.resolution,
            "requested_resolution": self.requested_resolution,
            "point_count": self.point_count,
            "coarsened": self.coarsened,
            "notes": list(self.notes),
        }


def plan_steps(
    curve: Curve,
    *,
    resolution: float = DEFAULT_RESOLUTION,
    clip_length: float | None = None,
) -> StepPlan:
    """Simulate step calculation performed during automation_write.

    Predicts whether step resolution will be coarsened to fit within MAX_STEPS.

    Args:
        curve: Curve to be evaluated.
        resolution: Requested step spacing in beats.
        clip_length: Optional total clip length in beats.

    Returns:
        StepPlan detailing predicted step count, effective resolution, and notices.
    """
    points = curve.resolved()
    notes: list[str] = []

    requested = float(resolution)
    grid = requested
    if grid <= 0.0:
        grid = DEFAULT_RESOLUTION
        notes.append(f"non-positive resolution replaced by the script default {grid:g} beats")
    if grid < MIN_RESOLUTION:
        notes.append(
            f"resolution {grid:g} is finer than the script's floor "
            f"{MIN_RESOLUTION:g} beats and will be raised to it"
        )
        grid = MIN_RESOLUTION

    span = points[-1].time - points[0].time
    budget = MAX_STEPS - len(points) - 1
    coarsened = False
    if span > 0.0 and budget > 0 and (span / grid) > budget:
        before_coarsening = grid
        grid = span / float(budget)
        coarsened = True
        notes.append(
            f"the script will coarsen the step width from {before_coarsening:g} to {grid:g} beats "
            f"to stay under {MAX_STEPS} steps; it logs this to Live's log and nothing else"
        )
    elif budget <= 0:
        notes.append(
            f"{len(points)} breakpoints leave no step budget under {MAX_STEPS}; the script "
            "does not coarsen in this case"
        )

    steps = 0
    for left, right in pairwise(points):
        segment = right.time - left.time
        if segment <= 0.0:
            # Two breakpoints on the same beat: a jump, no steps.
            continue
        count = int(segment / grid)
        if count * grid < segment - 1e-9:
            count += 1
        steps += max(1, count)
    steps += 1  # Tail step holding final value

    if clip_length is not None and clip_length <= points[-1].time:
        notes.append(
            f"the last breakpoint sits at beat {points[-1].time:g}, at or past the clip length "
            f"{clip_length:g}; the tail step falls back to one grid step"
        )
    return StepPlan(
        steps=steps,
        resolution=grid,
        requested_resolution=requested,
        point_count=len(points),
        coarsened=coarsened,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


class Verdict(str, Enum):
    """Outcome status from curve verification comparison."""

    MATCH = "match"
    """All evaluated samples fall within tolerance."""

    DRIFT = "drift"
    """Median error is within tolerance, but individual sample deviations exceed threshold."""

    MISMATCH = "mismatch"
    """Median error exceeds tolerance threshold."""

    FLAT = "flat"
    """Envelope is static across the evaluated duration despite non-flat input."""

    MISSING = "missing"
    """No envelope was found on target parameter."""

    INCONCLUSIVE = "inconclusive"
    """Insufficient samples survived filtering for comparison."""


@dataclass(frozen=True, slots=True)
class Comparison:
    """Outcome of automation curve verification against read-back samples.

    Attributes:
        verdict: Comparison outcome classification.
        compared: Count of valid evaluated samples.
        ignored_at_zero: Count of samples discarded near t=0.
        unusable: Count of malformed or missing samples received.
        median_error: Median absolute deviation across compared samples.
        mean_error: Mean absolute deviation across compared samples.
        max_error: Maximum absolute deviation encountered.
        worst_time: Beat timestamp of sample with greatest deviation.
        worst_expected: Expected value at worst sample time.
        worst_actual: Actual value observed at worst sample time.
        expected_span: Value span of expected curve across window.
        actual_span: Value span observed in read-back samples.
        tolerance: Error tolerance threshold applied.
        notes: Diagnostic messages.
    """

    verdict: Verdict
    compared: int
    ignored_at_zero: int = 0
    unusable: int = 0
    median_error: float | None = None
    mean_error: float | None = None
    max_error: float | None = None
    worst_time: float | None = None
    worst_expected: float | None = None
    worst_actual: float | None = None
    expected_span: float = 0.0
    actual_span: float = 0.0
    tolerance: float = DEFAULT_TOLERANCE
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Return True if curve matches or exhibits minor drift within median tolerance."""
        return self.verdict in (Verdict.MATCH, Verdict.DRIFT)

    def summary(self) -> str:
        """Generate concise summary message for comparison outcome."""
        if self.verdict is Verdict.MISSING:
            return "no envelope in this clip: the curve never arrived (has_envelope: false)."
        if self.verdict is Verdict.INCONCLUSIVE:
            return (
                "nothing left to compare: every sample fell at beat 0 or inside the guard band "
                f"below beat {READ_EPSILON:g}, where Live returns the parameter default rather "
                "than the curve."
            )
        if self.verdict is Verdict.FLAT:
            return (
                f"an envelope exists but does not move (span {self.actual_span:.6f}) while a ride "
                f"of {self.expected_span:.4f} was requested. The curve is "
                f"written but ineffective."
            )
        assert self.median_error is not None and self.max_error is not None
        head = {
            Verdict.MATCH: "curve verified",
            Verdict.DRIFT: "curve present, individual samples off",
            Verdict.MISMATCH: "curve does not match",
        }[self.verdict]
        return (
            f"{head}: median error {self.median_error:.4f}, max {self.max_error:.4f} "
            f"over {self.compared} samples (tolerance {self.tolerance:g}); "
            f"{self.ignored_at_zero} sample(s) at or near beat 0 ignored. "
            "Proves the stored curve, not that it is audible."
        )

    def as_dict(self) -> dict[str, Any]:
        """Return dictionary representation suitable for serialization."""
        return {
            "verdict": self.verdict.value,
            "ok": self.ok,
            "compared": self.compared,
            "ignored_at_zero": self.ignored_at_zero,
            "unusable": self.unusable,
            "median_error": self.median_error,
            "mean_error": self.mean_error,
            "max_error": self.max_error,
            "worst": (
                None
                if self.worst_time is None
                else {
                    "time": self.worst_time,
                    "expected": self.worst_expected,
                    "actual": self.worst_actual,
                }
            ),
            "expected_span": self.expected_span,
            "actual_span": self.actual_span,
            "tolerance": self.tolerance,
            "summary": self.summary(),
            "notes": list(self.notes),
        }


def _coerce_actual(actual: Any) -> tuple[list[Sample], bool | None, int, list[str]]:
    """Normalize automation_read response into structured Sample instances."""
    notes: list[str] = []
    has_envelope: bool | None = None
    raw: Any = actual

    if isinstance(actual, Mapping):
        if "has_envelope" in actual:
            has_envelope = bool(actual["has_envelope"])
        for key in ("points", "samples"):
            if isinstance(actual.get(key), Sequence):
                raw = actual[key]
                break
        else:
            raw = []
        if actual.get("moves") is False:
            notes.append("the handler reports the envelope does not move (moves: false)")

    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise TypeError(f"cannot read samples out of {type(actual).__name__}")

    samples: list[Sample] = []
    unusable = 0
    for index, entry in enumerate(raw):
        if isinstance(entry, Mapping):
            time, value = entry.get("time"), entry.get("value")
        elif isinstance(entry, Sequence) and not isinstance(entry, str | bytes) and len(entry) >= 2:
            time, value = entry[0], entry[1]
        else:
            raise ValueError(f"sample {index} is neither a mapping nor a pair: {entry!r}")
        if value is None or time is None:
            unusable += 1
            continue
        samples.append(Sample(float(time), float(value)))

    samples.sort(key=lambda s: s.time)
    if unusable:
        notes.append(f"{unusable} sample(s) came back without a value and were skipped")
    return samples, has_envelope, unusable, notes


def compare(
    expected: Curve | Iterable[PointLike],
    actual: Any,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    ignore_before: float = READ_EPSILON,
    flat_epsilon: float = FLAT_EPSILON,
) -> Comparison:
    """Compare expected automation curve against read-back sample data.

    Evaluates expected curve values at actual sample timestamps. Samples at or below
    ignore_before (or beat 0) are excluded to avoid Live parameter default artifacts.

    Args:
        expected: Expected Curve instance or iterable of breakpoints.
        actual: Response data from automation_read or list of samples.
        tolerance: Maximum acceptable parameter value deviation.
        ignore_before: Beat threshold below which samples are excluded.
        flat_epsilon: Minimum span required to classify envelope as dynamic.

    Returns:
        Comparison instance detailing verification verdict and error metrics.
    """
    curve = expected if isinstance(expected, Curve) else envelope(expected)
    samples, has_envelope, unusable, notes = _coerce_actual(actual)

    if has_envelope is False:
        notes.append(
            "has_envelope is false. Either the write never landed, or a later write on the same "
            "parameter cleared it. Also check that this is a Session clip: "
            "clip.automation_envelope() is None for Arrangement clips."
        )
        return Comparison(
            verdict=Verdict.MISSING,
            compared=0,
            unusable=unusable,
            expected_span=curve.span,
            tolerance=tolerance,
            notes=tuple(notes),
        )

    usable = [s for s in samples if s.time >= ignore_before and s.time > 0.0]
    ignored = len(samples) - len(usable)
    if ignored:
        notes.append(
            f"{ignored} sample(s) before beat {ignore_before:g} ignored (beat 0 always): at "
            "time 0 Live returns the parameter's default value, not the curve (measured)."
        )
    if not usable:
        notes.append(
            "no sample survived past beat 0; read again with start > 0 "
            f"(the client default is {READ_EPSILON:g})."
        )
        return Comparison(
            verdict=Verdict.INCONCLUSIVE,
            compared=0,
            ignored_at_zero=ignored,
            unusable=unusable,
            expected_span=curve.span,
            tolerance=tolerance,
            notes=tuple(notes),
        )

    actual_values = [s.value for s in usable]
    actual_span = max(actual_values) - min(actual_values)
    expected_values = [curve.value_at(s.time) for s in usable]
    expected_span = max(expected_values) - min(expected_values)

    if actual_span < flat_epsilon and expected_span > tolerance:
        notes.append(
            "the envelope exists and does not move. It arrived and does nothing; check that a "
            "second write pass on this parameter did not replace it, and that the device's "
            "is_active is true (measured: 6 657 breakpoints rode a filter whose On was false)."
        )
        return Comparison(
            verdict=Verdict.FLAT,
            compared=len(usable),
            ignored_at_zero=ignored,
            unusable=unusable,
            expected_span=expected_span,
            actual_span=actual_span,
            tolerance=tolerance,
            notes=tuple(notes),
        )

    errors = [abs(exp - act) for exp, act in zip(expected_values, actual_values, strict=True)]
    worst_index = max(range(len(errors)), key=errors.__getitem__)
    med = float(median(errors))
    mx = float(errors[worst_index])

    if mx <= tolerance:
        verdict = Verdict.MATCH
    elif med <= tolerance:
        verdict = Verdict.DRIFT
        notes.append(
            "the shape is there but individual samples miss. Usual causes: a coarse write "
            "resolution (the curve is written as flat steps), or a sample landing on a step edge."
        )
    else:
        verdict = Verdict.MISMATCH
        notes.append(
            "the curve in Live is not the curve that was sent. The most common cause is a second "
            "write pass on the same parameter: the second deletes the first, with a success "
            "message either way (measured). Combine rides with merge() and write once."
        )

    if expected_span < flat_epsilon:
        notes.append(
            "the expected curve does not move, so this comparison cannot tell a written curve "
            "from a leftover one. Verify with a ride, not with a constant."
        )
    notes.append(
        "This compares sampled values only. Live cannot enumerate an envelope's breakpoints, so "
        "where they sit (and therefore whether a curve is damaged rather than merely late) is "
        "only visible in the .als."
    )
    return Comparison(
        verdict=verdict,
        compared=len(usable),
        ignored_at_zero=ignored,
        unusable=unusable,
        median_error=med,
        mean_error=float(fmean(errors)),
        max_error=mx,
        worst_time=usable[worst_index].time,
        worst_expected=expected_values[worst_index],
        worst_actual=actual_values[worst_index],
        expected_span=expected_span,
        actual_span=actual_span,
        tolerance=tolerance,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# Parameter scaling and units
# --------------------------------------------------------------------------- #


def normalise(value: float, minimum: float, maximum: float) -> float:
    """Map value within [minimum, maximum] to normalized range [0.0, 1.0].

    Args:
        value: Value to normalize.
        minimum: Lower bound of source range.
        maximum: Upper bound of source range.

    Returns:
        Normalized float between 0.0 and 1.0.

    Raises:
        ValueError: If minimum and maximum are equal.
    """
    if maximum == minimum:
        raise ValueError(f"empty parameter range [{minimum}, {maximum}]")
    return (float(value) - minimum) / (maximum - minimum)


def denormalise(fraction: float, minimum: float, maximum: float) -> float:
    """Map normalized fraction [0.0, 1.0] to target range [minimum, maximum].

    Args:
        fraction: Normalized progression fraction.
        minimum: Lower bound of target range.
        maximum: Upper bound of target range.

    Returns:
        Denormalized value in target range units.

    Note:
        Turning a normalised position into a displayed unit is only correct when the
        parameter's own scale is linear over its range, and measured on a third-party
        compressor plugin that is the exception rather than the rule: Gain and Knee are
        linear, but RMS length is ``v**2 * 100 ms``, Attack and Release are
        ``v**4 * 1000 ms``, Ratio is ``20**v`` and Threshold is roughly
        ``40 * log10(v)``. Read linearly, an attack of 10 ms is set to 316 ms, a factor
        of 30, with no error message.

        Look the parameter up before computing: the sweep results first, then
        ``lom_describe`` with the range, then ``str_for_value()``. Compute only when all
        three come up empty.
    """
    return minimum + (maximum - minimum) * float(fraction)


@dataclass(frozen=True, slots=True)
class ParameterRange:
    """Parameter scaling and boundary metadata reported by the Live Object Model.

    Attributes:
        minimum: Minimum parameter value.
        maximum: Maximum parameter value.
        quantized: True if parameter accepts discrete integer values only.
        unit: Optional physical unit label (e.g. 'dB', 'Hz', '%').
    """

    minimum: float = 0.0
    maximum: float = 1.0
    quantized: bool = False
    unit: str | None = None

    def __post_init__(self) -> None:
        if self.maximum == self.minimum:
            raise ValueError(f"empty parameter range [{self.minimum}, {self.maximum}]")

    def clamp(self, value: float) -> float:
        """Constrain value to parameter bounds [minimum, maximum]."""
        low, high = sorted((self.minimum, self.maximum))
        return clamp(value, low, high)

    def contains(self, value: float) -> bool:
        """Return True if value lies within parameter bounds [minimum, maximum]."""
        low, high = sorted((self.minimum, self.maximum))
        return low <= value <= high

    def normalise(self, value: float) -> float:
        """Calculate normalized fraction [0.0, 1.0] for value in this range."""
        return normalise(value, self.minimum, self.maximum)

    def denormalise(self, fraction: float) -> float:
        """Calculate parameter value corresponding to normalized fraction."""
        return denormalise(fraction, self.minimum, self.maximum)

    def clamp_curve(self, curve: Curve) -> Curve:
        """Return copy of curve with all breakpoints clamped within range."""
        low, high = sorted((self.minimum, self.maximum))
        return curve.clamped(low, high)


def filter_hz_to_normalised(hz: float) -> float:
    """Convert filter frequency in Hz to normalized [0.0, 1.0] parameter value.

    Uses a logarithmic scale across FILTER_HZ_MIN (20 Hz) and FILTER_HZ_MAX (20000 Hz).
    Measured against the .als held next to ``automation_read``: 18939 Hz maps to 0.9921
    and 13692 Hz to 0.9451.

    Args:
        hz: Frequency in Hertz.

    Returns:
        Normalized parameter value between 0.0 and 1.0.

    Raises:
        ValueError: If frequency is non-positive.

    Note:
        Beware the 30 Hz variant. Other scripts sometimes compute
        ``log(hz / 30) / log(20000 / 30)``. At the top the difference barely shows
        (18939 Hz gives 0.9916 instead of 0.9921); at the bottom it decides everything:
        ask for 100 Hz and the result lands on 71.9 Hz, 572 cents off, audibly wrong and
        silent about it. A ``FILTER_HZ_MIN = 30.0`` in another codebase is the tell.

        Most of the time the formula is not needed at all: a static control can be set
        from its display value by searching for the raw one. The formula is for
        automation point lists, where there is no display feedback.
    """
    if hz <= 0.0:
        raise ValueError(f"frequency must be greater than 0 Hz, got {hz}")
    decades = math.log10(FILTER_HZ_MAX / FILTER_HZ_MIN)
    return math.log10(float(hz) / FILTER_HZ_MIN) / decades


def normalised_to_filter_hz(value: float) -> float:
    """Convert normalized [0.0, 1.0] filter parameter value to frequency in Hz.

    Args:
        value: Normalized parameter value.

    Returns:
        Frequency in Hertz across 20 Hz to 20000 Hz range.
    """
    decades = math.log10(FILTER_HZ_MAX / FILTER_HZ_MIN)
    return FILTER_HZ_MIN * 10.0 ** (decades * float(value))
