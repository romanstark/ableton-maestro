"""Tests for :mod:`ableton_maestro.automation`: shapes, prediction, verdicts.

Pure logic, no socket. Where a test needs an ``automation_read`` reply it
builds the reply as data, in the shape ``docs/protocol.md`` §5.9 specifies:
``{"has_envelope": bool, "points": [[time, value], ...]}``.

Two measured rules carry most of this file:

* At ``time = 0`` Live returns the parameter's default, not the curve. Any
  helper that verifies must start past beat 0, and :func:`compare` must throw
  a beat-0 sample away instead of failing on it.
* The script interpolates server-side, so two breakpoints are a whole ramp.
  The client's job is to predict what the script will make of them.
"""

from __future__ import annotations

import math

import pytest

from ableton_maestro import automation as A
from ableton_maestro.models import Interpolation

# --------------------------------------------------------------------------- #
# Interpolation shapes: endpoints and midpoint
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("mode", "midpoint"),
    [
        (Interpolation.LINEAR, 0.5),
        (Interpolation.HOLD, 0.0),
        (Interpolation.EXPONENTIAL, 0.25),      # t ** 2
        (Interpolation.EASE_IN, 0.25),          # the same function, named for the case
        (Interpolation.EASE_OUT, math.sqrt(0.5)),  # t ** (1 / 2), also spelled "logarithmic"
    ],
)
def test_every_mode_pins_its_endpoints_and_its_midpoint(
    mode: Interpolation, midpoint: float
) -> None:
    """The shape applies to the progress, never to the value, so both ends hold."""
    curve = A.ramp(0.0, 1.0, 4.0, interpolation=mode)

    assert curve.value_at(0.0) == pytest.approx(0.0)
    assert curve.value_at(4.0) == pytest.approx(1.0)
    assert curve.value_at(2.0) == pytest.approx(midpoint)


@pytest.mark.parametrize(
    "mode",
    [
        Interpolation.LINEAR,
        Interpolation.HOLD,
        Interpolation.EXPONENTIAL,
        Interpolation.EASE_IN,
        Interpolation.EASE_OUT,
    ],
)
def test_no_mode_overshoots_or_reads_outside_its_own_range(mode: Interpolation) -> None:
    curve = A.ramp(0.2, 0.8, 4.0, interpolation=mode)

    values = [curve.value_at(t / 10.0) for t in range(41)]

    assert min(values) >= 0.2 - 1e-12
    assert max(values) <= 0.8 + 1e-12
    # Held before the start and after the end, the way the script holds it.
    assert curve.value_at(-5.0) == pytest.approx(0.2)
    assert curve.value_at(99.0) == pytest.approx(0.8)


def test_hold_stays_put_until_the_next_breakpoint_then_jumps() -> None:
    curve = A.ramp(0.0, 1.0, 4.0, interpolation=Interpolation.HOLD)

    assert curve.value_at(3.999) == pytest.approx(0.0)
    assert curve.value_at(4.0) == pytest.approx(1.0)


def test_ease_in_and_exponential_are_the_same_function() -> None:
    ease = A.ramp(0.0, 1.0, 4.0, interpolation=Interpolation.EASE_IN, exponent=3.0)
    exp = A.ramp(0.0, 1.0, 4.0, interpolation=Interpolation.EXPONENTIAL, exponent=3.0)

    grid = [t / 4.0 for t in range(17)]
    assert [ease.value_at(t) for t in grid] == [exp.value_at(t) for t in grid]
    assert ease.value_at(2.0) == pytest.approx(0.125)  # 0.5 ** 3


def test_ease_out_is_the_reciprocal_exponent_not_a_library_ease() -> None:
    """``t ** (1 / exponent)``, not ``1 - (1 - t) ** 2``."""
    curve = A.ramp(0.0, 1.0, 4.0, interpolation=Interpolation.EASE_OUT, exponent=4.0)

    assert curve.value_at(2.0) == pytest.approx(0.5**0.25)
    assert curve.value_at(2.0) != pytest.approx(1.0 - (1.0 - 0.5) ** 4)


def test_a_falling_segment_eases_the_progress_so_it_drops_late() -> None:
    falling = A.ramp(1.0, 0.0, 4.0, interpolation=Interpolation.EASE_IN)

    assert falling.value_at(2.0) == pytest.approx(0.75)
    assert falling.value_at(1.0) > 0.9


def test_a_non_positive_exponent_falls_back_the_way_the_script_does() -> None:
    curve = A.ramp(0.0, 1.0, 4.0, interpolation=Interpolation.EASE_IN, exponent=0.0)
    assert curve.value_at(2.0) == pytest.approx(0.5 ** A.DEFAULT_EXPONENT)


def test_per_point_modes_override_the_curve_default() -> None:
    curve = A.envelope(
        [
            {"time": 0.0, "value": 0.0, "interpolation": "hold"},
            {"time": 2.0, "value": 1.0},
            {"time": 4.0, "value": 0.0},
        ],
        interpolation=Interpolation.LINEAR,
    )

    assert curve.has_mixed_modes
    assert curve.value_at(1.0) == pytest.approx(0.0), "segment 1 holds"
    assert curve.value_at(3.0) == pytest.approx(0.5), "segment 2 is linear"
    # Mixed modes travel per point, not once at request level.
    wire = curve.wire_points()
    assert wire[0]["interpolation"] == "hold"
    assert wire[1]["interpolation"] == "linear"


def test_two_breakpoints_on_one_beat_read_as_a_jump() -> None:
    curve = A.envelope([(0.0, 0.0), (2.0, 1.0), (2.0, 0.2), (4.0, 0.2)])

    assert curve.value_at(1.999) > 0.9
    assert curve.value_at(2.0) == pytest.approx(0.2)


def test_parse_interpolation_accepts_the_alternative_spellings() -> None:
    """The alternative spellings a caller may already have in hand.

    ``logarithmic`` names the same curve as ``ease_out``; case, hyphens and
    spaces are irrelevant; ``None`` means linear. Anything else is refused by
    name rather than quietly straightened into a line.
    """
    assert A.parse_interpolation("logarithmic") is Interpolation.EASE_OUT
    assert A.parse_interpolation("LOG") is Interpolation.EASE_OUT
    assert A.parse_interpolation("Ease-In") is Interpolation.EASE_IN
    assert A.parse_interpolation(None) is Interpolation.LINEAR
    assert A.interpolation_name(Interpolation.EASE_OUT) == "ease_out"

    with pytest.raises(ValueError, match="unknown interpolation"):
        A.parse_interpolation("wobble")


# --------------------------------------------------------------------------- #
# Two points describe a ramp: the script interpolates, not the client
# --------------------------------------------------------------------------- #


def test_a_ramp_is_exactly_two_breakpoints() -> None:
    curve = A.ramp(0.15, 0.95, 64.0)

    assert len(curve.points) == 2
    assert curve.points[0].time == 0.0
    assert curve.points[-1].time == 64.0
    assert curve.span == pytest.approx(0.8)
    assert not curve.is_flat()


def test_to_write_params_sends_two_points_plus_the_mode_and_the_resolution() -> None:
    params = A.to_write_params(A.ramp(0.15, 0.95, 64.0, interpolation="ease_in"))

    assert params["points"] == [
        {"time": 0.0, "value": 0.15},
        {"time": 64.0, "value": 0.95},
    ]
    assert params["interpolation"] == "ease_in"
    assert params["exponent"] == A.DEFAULT_EXPONENT
    assert params["resolution"] == A.DEFAULT_RESOLUTION


def test_plan_steps_predicts_what_the_script_expands_those_two_points_into() -> None:
    plan = A.plan_steps(A.ramp(0.15, 0.95, 64.0), resolution=0.25)

    assert plan.point_count == 2
    assert plan.steps == 257           # 64 / 0.25, plus the tail step
    assert plan.coarsened is False
    assert plan.resolution == 0.25
    assert plan.notes == ()
    assert plan.as_dict()["steps"] == 257


def test_plan_steps_predicts_the_coarsening_before_the_round_trip() -> None:
    """The script coarsens by itself and logs it to Live's log, which nobody reads."""
    plan = A.plan_steps(A.ramp(0.0, 1.0, 4000.0), resolution=A.MIN_RESOLUTION)

    assert plan.coarsened is True
    assert plan.steps <= A.MAX_STEPS
    assert plan.resolution > A.MIN_RESOLUTION
    assert plan.requested_resolution == A.MIN_RESOLUTION
    assert any("coarsen" in n for n in plan.notes)


def test_plan_steps_reports_a_resolution_below_the_script_floor() -> None:
    plan = A.plan_steps(A.ramp(0.0, 1.0, 4.0), resolution=A.MIN_RESOLUTION / 4.0)

    assert plan.resolution == A.MIN_RESOLUTION
    assert any("floor" in n for n in plan.notes)


def test_too_many_breakpoints_are_refused_here_because_the_script_refuses_them() -> None:
    """Beyond the ceiling the script rejects outright: it does not coarsen."""
    with pytest.raises(ValueError, match="too many breakpoints"):
        A.Curve(tuple(A.Point(float(i), 0.5) for i in range(A.MAX_POINTS + 1)))


def test_densify_is_the_special_case_and_pins_its_endpoints() -> None:
    dense = A.densify(A.ramp(0.0, 1.0, 4.0), resolution=1.0)

    assert [(p.time, p.value) for p in dense.points] == [
        (0.0, 0.0),
        (1.0, 0.25),
        (2.0, 0.5),
        (3.0, 0.75),
        (4.0, 1.0),
    ]
    assert dense.points[0].value == 0.0
    assert dense.points[-1].value == 1.0


def test_densify_respects_a_point_ceiling() -> None:
    dense = A.densify(A.ramp(0.0, 1.0, 400.0), resolution=0.01, max_points=100)
    assert len(dense.points) == 100


def test_merge_makes_one_curve_out_of_two_rides_on_one_parameter() -> None:
    """Two write passes delete each other (measured). One curve, written once."""
    base = A.hold(0.2, 16.0)
    detail = A.ramp(0.4, 0.9, 4.0, start_time=8.0)

    merged = A.merge(base, detail)

    assert merged.value_at(4.0) == pytest.approx(0.2)
    assert merged.value_at(8.0) == pytest.approx(0.4)
    assert merged.value_at(12.0) == pytest.approx(0.9)
    assert merged.value_at(15.0) == pytest.approx(0.2), "the base survives outside the override"

    with pytest.raises(ValueError, match="at least one curve"):
        A.merge()


def test_builders_reject_a_length_of_zero() -> None:
    with pytest.raises(ValueError, match="beats must be greater than 0"):
        A.ramp(0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="beats must be greater than 0"):
        A.hold(0.5, 0.0)
    with pytest.raises(ValueError, match="at least one breakpoint"):
        A.envelope([])
    with pytest.raises(ValueError, match="must not be negative"):
        A.Point(-1.0, 0.5)


def test_adsr_stages_land_where_the_arguments_say() -> None:
    curve = A.adsr(attack=1.0, decay=1.0, sustain=0.6, release=2.0, sustain_beats=4.0, peak=1.0)

    assert curve.value_at(0.0) == pytest.approx(0.0)
    assert curve.value_at(1.0) == pytest.approx(1.0)
    assert curve.value_at(2.0) == pytest.approx(0.6)
    assert curve.value_at(5.0) == pytest.approx(0.6), "the sustain stage is genuinely flat"
    assert curve.value_at(8.0) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="at least one stage"):
        A.adsr(attack=0.0, decay=0.0, sustain=0.5, release=0.0)


def test_lfo_shapes_are_exact_without_dense_sampling() -> None:
    triangle = A.lfo("triangle", 4.0, cycles=1, low=0.0, high=1.0)
    assert [(p.time, p.value) for p in triangle.points] == [(0.0, 0.0), (2.0, 1.0), (4.0, 0.0)]

    square = A.lfo("square", 4.0, cycles=1)
    assert all(p.interpolation is Interpolation.HOLD for p in square.points)
    assert square.value_at(1.0) == pytest.approx(1.0)
    assert square.value_at(3.0) == pytest.approx(0.0)


def test_lfo_random_is_reproducible_from_its_seed() -> None:
    """A ride you cannot regenerate is a ride you cannot repair."""
    first = A.lfo("random", 8.0, cycles=4, seed=7)
    again = A.lfo("random", 8.0, cycles=4, seed=7)
    other = A.lfo("random", 8.0, cycles=4, seed=8)

    assert first.values == again.values
    assert first.values != other.values


def test_probe_ramp_stays_away_from_both_ends_of_the_range() -> None:
    """Verify probe distinguished from non-landing write at minimum value."""
    probe = A.probe_ramp(8.0, minimum=0.0, maximum=1.0, margin=0.25)

    assert probe.values == (0.25, 0.75)
    assert min(probe.values) > 0.0
    assert max(probe.values) < 1.0

    on_a_real_range = A.probe_ramp(8.0, minimum=-40.0, maximum=0.0, margin=0.2)
    assert on_a_real_range.values == (-32.0, -8.0)


# --------------------------------------------------------------------------- #
# The t = 0 rule: measured, and the reason verification starts at 1/64
# --------------------------------------------------------------------------- #


def test_read_window_never_starts_at_beat_zero() -> None:
    """The guard itself. At t = 0 Live hands back the parameter default."""
    start, end = A.read_window(A.ramp(0.0, 1.0, 64.0))

    assert start > 0.0
    assert start == A.READ_EPSILON
    assert end == 64.0


def test_read_window_keeps_a_start_that_is_already_past_zero() -> None:
    assert A.read_window(A.ramp(0.0, 1.0, 4.0, start_time=2.0)) == (2.0, 6.0)


def test_read_window_gives_a_single_point_curve_a_one_beat_window() -> None:
    start, end = A.read_window(A.envelope([(0.0, 0.5)]))

    assert start == A.READ_EPSILON
    assert end == pytest.approx(start + 1.0)


def test_sample_mirrors_the_read_grid_and_emits_no_sample_at_zero() -> None:
    samples = A.sample(A.ramp(0.0, 1.0, 4.0), 5)

    assert len(samples) == 5
    assert samples[0].time == A.READ_EPSILON
    assert all(s.time > 0.0 for s in samples)
    assert samples[-1].time == pytest.approx(4.0)
    assert samples[-1].value == pytest.approx(1.0)


def test_sample_clamps_to_the_handlers_own_limits() -> None:
    assert len(A.sample(A.ramp(0.0, 1.0, 4.0), 10_000)) == A.MAX_READ_SAMPLES
    with pytest.raises(ValueError, match=f"at least {A.MIN_READ_SAMPLES}"):
        A.sample(A.ramp(0.0, 1.0, 4.0), 1)


def test_sample_without_mirror_read_is_a_preview_and_starts_at_the_true_start() -> None:
    preview = A.sample(A.ramp(0.0, 1.0, 4.0), 5, mirror_read=False)

    assert preview[0].time == 0.0
    assert preview[0].value == pytest.approx(0.0)


def test_compare_discards_a_beat_zero_sample_and_counts_it() -> None:
    """0.8997 at t = 0 for a ramp that starts at 0.0: the measured artefact."""
    curve = A.ramp(0.0, 1.0, 8.0)
    honest = [[s.time, s.value] for s in A.sample(curve, 32)]
    poisoned = [[0.0, 0.8997], *honest]

    result = A.compare(curve, poisoned)

    assert result.verdict is A.Verdict.MATCH
    assert result.ignored_at_zero == 1
    assert result.compared == 32
    assert any("parameter's default" in n for n in result.notes)
    assert "at or near beat 0 ignored" in result.summary()


def test_compare_is_inconclusive_when_only_beat_zero_came_back() -> None:
    result = A.compare(A.ramp(0.0, 1.0, 8.0), [[0.0, 0.8997]])

    assert result.verdict is A.Verdict.INCONCLUSIVE
    assert result.ok is False
    assert result.compared == 0
    assert result.ignored_at_zero == 1
    assert "read again with start > 0" in " ".join(result.notes)


def test_a_sample_sitting_exactly_on_the_guard_band_survives() -> None:
    """A read taken over read_window() must lose nothing but the artefact."""
    curve = A.ramp(0.0, 1.0, 8.0)
    samples = [[A.READ_EPSILON, curve.value_at(A.READ_EPSILON)], [4.0, curve.value_at(4.0)]]

    result = A.compare(curve, samples)

    assert result.compared == 2
    assert result.ignored_at_zero == 0


# --------------------------------------------------------------------------- #
# sample() and compare() against each other, and compare()'s honesty
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "mode",
    [
        Interpolation.LINEAR,
        Interpolation.HOLD,
        Interpolation.EXPONENTIAL,
        Interpolation.EASE_IN,
        Interpolation.EASE_OUT,
    ],
)
def test_sample_and_compare_agree_within_tolerance(mode: Interpolation) -> None:
    """Feed compare() the curve's own samples and it must find the curve."""
    curve = A.ramp(0.1, 0.9, 16.0, interpolation=mode)
    reply = {"has_envelope": True, "points": [[s.time, s.value] for s in A.sample(curve, 64)]}

    result = A.compare(curve, reply)

    assert result.verdict is A.Verdict.MATCH
    assert result.ok
    assert result.compared == 64
    assert result.median_error == pytest.approx(0.0, abs=1e-6)
    assert result.max_error == pytest.approx(0.0, abs=1e-6)
    assert result.max_error <= result.tolerance


def test_compare_evaluates_the_expectation_at_the_actual_sample_times() -> None:
    """The two lists never have to share a grid."""
    curve = A.ramp(0.0, 1.0, 8.0)
    coarse = [[t, curve.value_at(t)] for t in (0.5, 1.7, 3.3, 6.9)]

    result = A.compare(curve, coarse)

    assert result.verdict is A.Verdict.MATCH
    assert result.compared == 4


def test_compare_reports_a_mismatch_rather_than_a_bare_false() -> None:
    curve = A.ramp(0.0, 1.0, 8.0)
    inverted = [[s.time, 1.0 - s.value] for s in A.sample(curve, 32)]

    result = A.compare(curve, inverted)

    assert result.verdict is A.Verdict.MISMATCH
    assert result.ok is False
    assert result.median_error is not None and result.median_error > result.tolerance
    assert result.worst_time is not None
    assert result.worst_expected is not None and result.worst_actual is not None
    assert any("second write pass" in n for n in result.notes)
    assert "curve does not match" in result.summary()


def test_compare_says_drift_when_the_shape_is_there_and_samples_are_not() -> None:
    """Median inside tolerance, one sample well outside it."""
    curve = A.ramp(0.0, 1.0, 8.0)
    samples = [[s.time, s.value] for s in A.sample(curve, 33)]
    samples[16][1] += 0.4

    result = A.compare(curve, samples)

    assert result.verdict is A.Verdict.DRIFT
    assert result.ok, "drift still means the ride is present"
    assert result.median_error is not None and result.median_error <= result.tolerance
    assert result.max_error is not None and result.max_error > result.tolerance
    assert result.worst_time == pytest.approx(samples[16][0])
    assert any("step edge" in n for n in result.notes)


def test_compare_distinguishes_flat_from_wrong() -> None:
    """An envelope that exists and does not move: written but ineffective."""
    curve = A.ramp(0.1, 0.9, 8.0)
    stuck = {"has_envelope": True, "points": [[s.time, 0.5] for s in A.sample(curve, 16)]}

    result = A.compare(curve, stuck)

    assert result.verdict is A.Verdict.FLAT
    assert result.ok is False
    assert result.actual_span == pytest.approx(0.0)
    assert result.expected_span > result.tolerance
    assert any("is_active" in n for n in result.notes)
    assert "written but ineffective" in result.summary()


def test_compare_distinguishes_missing_from_flat() -> None:
    """``has_envelope: false`` means the write never arrived."""
    result = A.compare(A.ramp(0.1, 0.9, 8.0), {"has_envelope": False, "points": []})

    assert result.verdict is A.Verdict.MISSING
    assert result.ok is False
    assert result.compared == 0
    assert any("Arrangement clips" in n for n in result.notes)
    assert "never arrived" in result.summary()


def test_compare_counts_samples_live_refused_instead_of_guessing_them() -> None:
    curve = A.ramp(0.0, 1.0, 8.0)
    reply = {
        "has_envelope": True,
        "points": [[1.0, curve.value_at(1.0)], [2.0, None], [4.0, curve.value_at(4.0)]],
    }

    result = A.compare(curve, reply)

    assert result.unusable == 1
    assert result.compared == 2
    assert any("without a value" in n for n in result.notes)


def test_compare_warns_that_a_flat_expectation_proves_nothing() -> None:
    flat = A.hold(0.5, 8.0)
    reply = [[s.time, 0.5] for s in A.sample(flat, 16)]

    result = A.compare(flat, reply)

    assert result.verdict is A.Verdict.MATCH
    assert any("Verify with a ride" in n for n in result.notes)


def test_compare_never_claims_the_result_is_audible() -> None:
    curve = A.ramp(0.1, 0.9, 8.0)
    result = A.compare(curve, [[s.time, s.value] for s in A.sample(curve, 16)])

    assert "Proves the stored curve, not that it is audible." in result.summary()
    assert any("only visible in the .als" in n for n in result.notes)
    assert result.as_dict()["verdict"] == "match"


def test_compare_accepts_the_samples_shape_and_a_bare_list() -> None:
    """A read may arrive under ``samples``, under ``points``, or as a bare list.

    ``compare`` reads all three, so a caller never has to unwrap the reply
    before it can be checked.
    """
    curve = A.ramp(0.0, 1.0, 8.0)
    points = [{"time": s.time, "value": s.value} for s in A.sample(curve, 16)]

    from_samples = A.compare(curve, {"samples": points})
    from_bare = A.compare(curve, points)

    assert from_samples.verdict is A.Verdict.MATCH
    assert from_bare.verdict is A.Verdict.MATCH


def test_compare_rejects_something_that_is_not_a_reply_at_all() -> None:
    with pytest.raises(TypeError, match="cannot read samples"):
        A.compare(A.ramp(0.0, 1.0, 4.0), 12.0)
    with pytest.raises(ValueError, match="neither a mapping nor a pair"):
        A.compare(A.ramp(0.0, 1.0, 4.0), [12.0])


def test_clamping_the_expectation_before_comparing_avoids_a_false_mismatch() -> None:
    """The script clamps on arrival; an unclamped expectation reports a mismatch."""
    written = A.ramp(0.5, 1.4, 8.0)
    landed = written.clamped(0.0, 1.0)
    reply = [[s.time, s.value] for s in A.sample(landed, 32)]

    assert A.compare(written, reply).verdict is A.Verdict.MISMATCH
    assert A.compare(landed, reply).verdict is A.Verdict.MATCH


def test_the_session_only_note_names_the_rule_and_its_direction() -> None:
    assert "Session" in A.SESSION_ONLY_NOTE
    assert "automation_envelope() returns None for Arrangement clips" in A.SESSION_ONLY_NOTE
    assert "never the reverse" in A.SESSION_ONLY_NOTE


# --------------------------------------------------------------------------- #
# Normalised values and display units
# --------------------------------------------------------------------------- #


def test_normalise_and_denormalise_are_inverses_over_a_real_range() -> None:
    for value in (-40.0, -30.0, -12.5, 0.0):
        fraction = A.normalise(value, -40.0, 0.0)
        assert A.denormalise(fraction, -40.0, 0.0) == pytest.approx(value)

    assert A.normalise(-20.0, -40.0, 0.0) == pytest.approx(0.5)
    assert A.denormalise(0.25, -40.0, 0.0) == pytest.approx(-30.0)


def test_normalise_extrapolates_outside_the_range_rather_than_clamping() -> None:
    assert A.normalise(10.0, 0.0, 1.0) == pytest.approx(10.0)
    assert A.denormalise(-1.0, 0.0, 1.0) == pytest.approx(-1.0)


def test_normalise_refuses_an_empty_range() -> None:
    with pytest.raises(ValueError, match="empty parameter range"):
        A.normalise(0.5, 1.0, 1.0)
    with pytest.raises(ValueError, match="empty parameter range"):
        A.ParameterRange(minimum=1.0, maximum=1.0)


def test_parameter_range_clamps_contains_and_converts() -> None:
    threshold = A.ParameterRange(minimum=-40.0, maximum=0.0, unit="dB")

    assert threshold.normalise(-10.0) == pytest.approx(0.75)
    assert threshold.denormalise(0.75) == pytest.approx(-10.0)
    assert threshold.clamp(5.0) == 0.0
    assert threshold.clamp(-100.0) == -40.0
    assert threshold.contains(-20.0)
    assert not threshold.contains(-41.0)


def test_parameter_range_clamps_a_whole_curve_for_the_comparison() -> None:
    unit = A.ParameterRange(minimum=0.0, maximum=1.0)

    clamped = unit.clamp_curve(A.ramp(-0.5, 1.5, 8.0))

    assert clamped.values == (0.0, 1.0)


def test_a_quantized_parameter_says_so_because_a_ride_over_one_is_a_staircase() -> None:
    switch = A.ParameterRange(minimum=0.0, maximum=1.0, quantized=True)
    assert switch.quantized is True


def test_filter_hz_reproduces_the_two_measured_anchors() -> None:
    assert A.filter_hz_to_normalised(18939.0) == pytest.approx(0.9921, abs=5e-5)
    assert A.filter_hz_to_normalised(13692.0) == pytest.approx(0.9451, abs=5e-5)


def test_filter_hz_round_trips_and_0_5_is_not_a_frequency() -> None:
    for hz in (20.0, 100.0, 632.5, 5000.0, 20000.0):
        assert A.normalised_to_filter_hz(A.filter_hz_to_normalised(hz)) == pytest.approx(hz)

    assert A.normalised_to_filter_hz(0.5) == pytest.approx(632.5, abs=0.1)
    assert A.filter_hz_to_normalised(A.FILTER_HZ_MIN) == pytest.approx(0.0)
    assert A.filter_hz_to_normalised(A.FILTER_HZ_MAX) == pytest.approx(1.0)


def test_the_thirty_hertz_variant_is_wrong_at_the_bottom_and_this_one_is_not() -> None:
    """Ask for 100 Hz with a 30 Hz floor and you land at 71.9 Hz, 572 cents off."""
    ours = A.filter_hz_to_normalised(100.0)
    theirs = math.log10(100.0 / 30.0) / math.log10(20000.0 / 30.0)

    assert A.normalised_to_filter_hz(ours) == pytest.approx(100.0)
    assert A.normalised_to_filter_hz(theirs) == pytest.approx(71.9, abs=0.2)
    # At the top the two are indistinguishable, which is why the check belongs low.
    top_ours = A.filter_hz_to_normalised(18939.0)
    top_theirs = math.log10(18939.0 / 30.0) / math.log10(20000.0 / 30.0)
    assert abs(top_ours - top_theirs) < 0.001


def test_filter_hz_refuses_a_non_positive_frequency() -> None:
    with pytest.raises(ValueError, match="greater than 0 Hz"):
        A.filter_hz_to_normalised(0.0)


def test_clamp_ignores_a_none_bound() -> None:
    assert A.clamp(5.0, None, 1.0) == 1.0
    assert A.clamp(-5.0, 0.0, None) == 0.0
    assert A.clamp(0.5, 0.0, 1.0) == 0.5
