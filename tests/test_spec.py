"""``spec.py``: the path builder, the pre-flight validator, and the grammar.

Everything in ``spec.py`` runs *before* anything reaches the socket, so all of it
is testable without Live, without a network, and in milliseconds. That is the
point of the module and the point of this file.

The behaviour under test is mostly refusal, and the refusals are not
interchangeable. Three of them carry the weight:

* An unknown argument is an error. ``build_path(spec, trak=3)`` must not
  quietly build ``song.tracks[0]...`` from a default. A keyword nobody reads
  vanishes without a sound and the operation proceeds against someone else's
  target: measured, that is how a set of automation curves ended up in slot 0
  while every call reported success (docs/architecture.md, 'read-back as a
  principle'). So the tests here do not stop at "it raised". They assert that
  the message names the argument, because an error that does not is nearly
  as bad as no error.
* Nothing is clamped. Live clamps, and ``lom_set``'s read-back is what makes
  that visible (``docs/protocol.md`` §5.4). A client that clamped first would
  turn a caller's mistake into a plausible-looking success.
* The grammar is a whitelist. No slices, no negative indices, no method
  calls, no expressions, because the resolver inside Live is a whitelist too,
  and a path this module accepts that the script rejects is a bug in one of the
  two ends.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from ableton_maestro.models import Access, Kind, PathStatus, Unit, WarpMode
from ableton_maestro.registry import default_registry
from ableton_maestro.spec import (
    ArgSpec,
    ParamSpec,
    PathSpec,
    build_path,
    name_placeholders_in,
    placeholders_in,
    require_access,
    validate_path,
    validate_value,
)

# --------------------------------------------------------------------------------
# Specs used across the file. Hand-built rather than taken from the catalog, so a
# catalog edit cannot change what these tests mean.
# --------------------------------------------------------------------------------


def volume_spec() -> PathSpec:
    """A normalised float on an indexed path: the archetypal row."""
    return PathSpec(
        id="track.volume",
        path="song.tracks[{track}].mixer_device.volume",
        access=[Access.GET, Access.SET],
        kind=Kind.FLOAT,
        range=(0.0, 1.0),
        unit=Unit.NORMALIZED,
        display="db",
        doc="Track volume. Normalised, not dB: 0.85 is 0 dB (measured).",
        params=[ParamSpec(name="track")],
    )


def clip_name_spec() -> PathSpec:
    """Two index placeholders, a string value."""
    return PathSpec(
        id="clip.name",
        path="song.tracks[{track}].clip_slots[{slot}].clip.name",
        access=[Access.GET, Access.SET],
        kind=Kind.STR,
        doc="Clip name.",
        params=[ParamSpec(name="track"), ParamSpec(name="slot")],
    )


def browser_item_spec() -> PathSpec:
    """A *segment name* placeholder next to an index placeholder."""
    return PathSpec(
        id="browser.item",
        path="app.browser.{root}.children[{index}]",
        access=[Access.GET],
        kind=Kind.OBJECT,
        doc="One browser entry.",
        params=[
            ParamSpec(
                name="root",
                kind=Kind.ENUM,
                enum=["instruments", "drums", "audio_effects"],
            ),
            ParamSpec(name="index"),
        ],
    )


def tempo_spec() -> PathSpec:
    """No placeholders at all."""
    return PathSpec(
        id="song.tempo",
        path="song.tempo",
        access=[Access.GET, Access.SET],
        kind=Kind.FLOAT,
        range=(20.0, 999.0),
        unit=Unit.NONE,
        doc="Tempo in BPM.",
    )


# --------------------------------------------------------------------------------
# build_path: substitution
# --------------------------------------------------------------------------------


def test_build_path_substitutes_one_index() -> None:
    """The template becomes a concrete, sendable path."""
    assert build_path(volume_spec(), track=3) == "song.tracks[3].mixer_device.volume"


def test_build_path_substitutes_several_placeholders() -> None:
    assert (
        build_path(clip_name_spec(), track=2, slot=7)
        == "song.tracks[2].clip_slots[7].clip.name"
    )


def test_build_path_leaves_a_template_without_placeholders_alone() -> None:
    assert build_path(tempo_spec()) == "song.tempo"


def test_build_path_fills_a_repeated_placeholder_everywhere() -> None:
    """One name may stand in two positions, and both get filled.

    ``placeholders_in`` keeps duplicates on purpose so a caller can see that it
    happened. The substitution must not stop at the first.
    """
    spec = PathSpec(
        id="test.repeated",
        path="song.tracks[{i}].devices[{i}]",
        access=[Access.GET],
        doc="A placeholder used twice.",
        params=[ParamSpec(name="i")],
    )
    assert build_path(spec, i=4) == "song.tracks[4].devices[4]"


def test_build_path_uses_an_optional_parameters_default() -> None:
    spec = PathSpec(
        id="test.optional",
        path="song.tracks[{track}].name",
        access=[Access.GET],
        kind=Kind.STR,
        doc="Optional index with a default.",
        params=[ParamSpec(name="track", required=False, default=0)],
    )
    assert build_path(spec) == "song.tracks[0].name"
    assert build_path(spec, track=5) == "song.tracks[5].name"


def test_build_path_accepts_an_int_enum_member() -> None:
    """An ``IntEnum`` is unwrapped to its value rather than stringified.

    ``str(WarpMode.TEXTURE)`` is not ``"2"`` on every Python, so relying on it
    would produce a path that differs between interpreters.
    """
    spec = PathSpec(
        id="test.enum_index",
        path="song.tracks[{track}].name",
        access=[Access.GET],
        kind=Kind.STR,
        doc="Index supplied as an IntEnum member.",
        params=[ParamSpec(name="track")],
    )
    assert build_path(spec, track=WarpMode.TEXTURE) == "song.tracks[2].name"


# --------------------------------------------------------------------------------
# build_path: an unknown argument is an error, and the error names it
# --------------------------------------------------------------------------------


def test_unknown_argument_is_rejected_and_the_message_names_it() -> None:
    """The headline refusal of this module.

    A misspelt keyword must not be dropped. It must raise, and the message must
    contain the offending name: a caller staring at "invalid arguments" still
    has to guess which one, and guessing is what this whole mechanism exists to
    remove.
    """
    with pytest.raises(ValueError) as excinfo:
        build_path(volume_spec(), trak=3)

    message = str(excinfo.value)
    assert "trak" in message, f"the rejected argument is not named: {message}"
    assert "track.volume" in message, f"the spec id is not named: {message}"
    assert "track" in message, f"the valid parameters are not listed: {message}"


def test_unknown_argument_is_rejected_even_alongside_a_correct_one() -> None:
    """A right argument does not excuse a wrong one.

    This is the realistic shape of the failure: the call looks like it worked
    because the part the author was thinking about did.
    """
    with pytest.raises(ValueError, match="slott"):
        build_path(clip_name_spec(), track=1, slot=2, slott=3)


def test_unknown_arguments_are_all_named_not_just_the_first() -> None:
    with pytest.raises(ValueError) as excinfo:
        build_path(clip_name_spec(), track=1, slot=2, colour="red", length=4.0)
    message = str(excinfo.value)
    assert "colour" in message
    assert "length" in message


def test_a_spec_with_no_parameters_still_rejects_an_argument() -> None:
    """``song.tempo`` takes nothing.

    ``track=3`` is a mistake, not a no-op.
    """
    with pytest.raises(ValueError) as excinfo:
        build_path(tempo_spec(), track=3)
    message = str(excinfo.value)
    assert "track" in message
    assert "none" in message, f"the message should say the row takes no parameters: {message}"


def test_unknown_argument_is_rejected_on_a_real_catalog_row() -> None:
    """The same refusal, through the shipped catalog rather than a fixture."""
    spec = default_registry().get("track.volume")
    with pytest.raises(ValueError, match="trakc"):
        build_path(spec, trakc=0)
    assert build_path(spec, track=0) == "song.tracks[0].mixer_device.volume"


# --------------------------------------------------------------------------------
# build_path: a missing placeholder is an error too
# --------------------------------------------------------------------------------


def test_missing_required_parameter_is_rejected() -> None:
    with pytest.raises(ValueError) as excinfo:
        build_path(volume_spec())
    message = str(excinfo.value)
    assert "track" in message
    assert "track.volume" in message


def test_missing_one_of_several_parameters_is_rejected() -> None:
    with pytest.raises(ValueError, match="slot"):
        build_path(clip_name_spec(), track=1)


def test_an_explicit_none_counts_as_missing_not_as_a_value() -> None:
    """``track=None`` must not render as the string ``"None"``."""
    with pytest.raises(ValueError, match="track"):
        build_path(volume_spec(), track=None)


# --------------------------------------------------------------------------------
# build_path: what an index may be
# --------------------------------------------------------------------------------


def test_an_integral_float_index_is_accepted() -> None:
    """JSON has one number type.

    Refusing ``3.0`` would punish the wire format.
    """
    assert build_path(volume_spec(), track=3.0) == "song.tracks[3].mixer_device.volume"


@pytest.mark.parametrize(
    ("value", "expected_in_message"),
    [
        (3.5, "whole number"),
        (float("nan"), "whole number"),
        (float("inf"), "whole number"),
        ("3", "int"),
        (-1, "negative"),
        ([3], "int"),
        (None, "missing"),
    ],
    ids=["fractional", "nan", "inf", "string", "negative", "list", "none"],
)
def test_an_index_that_is_not_a_whole_non_negative_number_is_rejected(
    value: Any, expected_in_message: str
) -> None:
    with pytest.raises(ValueError) as excinfo:
        build_path(volume_spec(), track=value)
    assert expected_in_message in str(excinfo.value)


def test_a_bool_index_is_rejected_even_though_bool_is_an_int() -> None:
    """``track=True`` would silently address track 1.

    ``bool`` is a subclass of ``int`` in Python, so every naive integer check
    passes it. The refusal has to be explicit or it does not happen.
    """
    with pytest.raises(ValueError) as excinfo:
        build_path(volume_spec(), track=True)
    assert "bool" in str(excinfo.value)


def test_an_index_outside_the_parameters_range_is_rejected() -> None:
    spec = PathSpec(
        id="test.bounded",
        path="song.tracks[{track}].name",
        access=[Access.GET],
        kind=Kind.STR,
        doc="A catalog-side sanity bound on the index.",
        params=[ParamSpec(name="track", range=(0, 15))],
    )
    assert build_path(spec, track=15) == "song.tracks[15].name"
    with pytest.raises(ValueError, match="maximum"):
        build_path(spec, track=16)


# --------------------------------------------------------------------------------
# build_path: segment-name placeholders are a different position with different rules
# --------------------------------------------------------------------------------


def test_a_segment_name_placeholder_substitutes_a_name() -> None:
    assert (
        build_path(browser_item_spec(), root="drums", index=6)
        == "app.browser.drums.children[6]"
    )


def test_the_two_placeholder_positions_are_told_apart() -> None:
    path = "app.browser.{root}.children[{index}]"
    assert name_placeholders_in(path) == {"root"}
    assert placeholders_in(path) == ["root", "index"]
    assert name_placeholders_in("song.tracks[{track}].name") == set()


def test_a_segment_name_outside_the_declared_enum_is_rejected() -> None:
    """The browser roots are a fixed set of names, not free text.

    Without the check a typo becomes a ``no_such_path`` from inside Live instead
    of a sentence naming the valid choices.
    """
    with pytest.raises(ValueError) as excinfo:
        build_path(browser_item_spec(), root="instrument", index=0)
    message = str(excinfo.value)
    assert "instrument" in message
    assert "instruments" in message, f"the valid names are not listed: {message}"


def test_a_segment_name_must_be_a_string() -> None:
    with pytest.raises(ValueError, match="string"):
        build_path(browser_item_spec(), root=0, index=0)


@pytest.mark.parametrize(
    "root",
    ["instruments.children", "children[0]", "", "in struments", "1st"],
    ids=["dotted", "indexed", "empty", "spaced", "leading-digit"],
)
def test_a_segment_name_that_is_not_an_identifier_is_rejected(root: str) -> None:
    """A name placeholder cannot smuggle extra path structure through.

    The value lands in an attribute position, so anything that is not a plain
    identifier would either build a path the resolver cannot parse or reach a
    different object than the caller named.
    """
    spec = PathSpec(
        id="test.free_name",
        path="app.browser.{root}.children[0]",
        access=[Access.GET],
        doc="A name placeholder with no enum, so only the grammar guards it.",
        params=[ParamSpec(name="root", kind=Kind.STR)],
    )
    with pytest.raises(ValueError):
        build_path(spec, root=root)


def test_an_index_placeholder_cannot_be_filled_with_a_name() -> None:
    """The positions are never interchangeable, in either direction."""
    with pytest.raises(ValueError, match="int"):
        build_path(browser_item_spec(), root="drums", index="drums")


# --------------------------------------------------------------------------------
# validate_value: coercion per kind
# --------------------------------------------------------------------------------


def kind_spec(kind: Kind, **extra: Any) -> PathSpec:
    """A minimal spec of the given kind, for value tests."""
    return PathSpec(
        id=f"test.{kind.value}",
        path="song.tempo",
        access=[Access.GET, Access.SET],
        kind=kind,
        doc=f"A {kind.value}-valued row.",
        **extra,
    )


def test_a_float_row_widens_an_int() -> None:
    result = validate_value(kind_spec(Kind.FLOAT), 1)
    assert result == 1.0
    assert isinstance(result, float)


def test_an_int_row_accepts_an_integral_float_and_refuses_a_fractional_one() -> None:
    spec = kind_spec(Kind.INT)
    assert validate_value(spec, 4.0) == 4
    with pytest.raises(ValueError, match="int"):
        validate_value(spec, 4.5)


@pytest.mark.parametrize("value", [True, False], ids=["true", "false"])
def test_a_bool_is_refused_for_a_numeric_row(value: bool) -> None:
    """``volume=True`` would otherwise arrive at Live as ``1.0``.

    And the read-back would confirm it, which is the part that makes this
    dangerous rather than merely wrong.
    """
    with pytest.raises(ValueError, match="bool"):
        validate_value(kind_spec(Kind.FLOAT), value)
    with pytest.raises(ValueError, match="bool"):
        validate_value(kind_spec(Kind.INT), value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_number_is_refused(value: float) -> None:
    """Every comparison with NaN is False, so a naive range check passes it.

    It would sail through validation and land on a parameter, so it is refused
    at the coercion step instead of at the bounds step.
    """
    assert not math.isfinite(value)
    with pytest.raises(ValueError):
        validate_value(kind_spec(Kind.FLOAT, range=(0.0, 1.0)), value)


def test_a_bool_row_takes_a_bool_and_the_ints_zero_and_one() -> None:
    spec = kind_spec(Kind.BOOL)
    assert validate_value(spec, True) is True
    assert validate_value(spec, 0) is False
    with pytest.raises(ValueError):
        validate_value(spec, 2)


@pytest.mark.parametrize("value", ["false", "true", "no", ""], ids=["false", "true", "no", "empty"])
def test_a_string_is_refused_for_a_bool_row(value: str) -> None:
    """A non-empty string is truthy, so ``"false"`` would switch this on."""
    with pytest.raises(ValueError):
        validate_value(kind_spec(Kind.BOOL), value)


def test_a_str_row_refuses_a_number() -> None:
    assert validate_value(kind_spec(Kind.STR), "Bass") == "Bass"
    with pytest.raises(ValueError, match="string"):
        validate_value(kind_spec(Kind.STR), 3)


def test_a_list_row_normalises_a_tuple_and_refuses_a_scalar() -> None:
    spec = kind_spec(Kind.LIST)
    assert validate_value(spec, (1, 2)) == [1, 2]
    with pytest.raises(ValueError, match="list"):
        validate_value(spec, 1)


def test_an_object_row_takes_a_handle_and_refuses_a_bare_value() -> None:
    """A Live object never travels as itself: it travels as a handle.

    ``docs/protocol.md`` §7 for the ``{"__lom__": ...}`` read shape, §5.4 for the
    ``{"__path__": ...}`` form a reference *write* takes.
    """
    spec = kind_spec(Kind.OBJECT)
    handle = {"__lom__": "Track", "path": "song.tracks[3]", "name": "Bass"}
    assert validate_value(spec, handle) == handle
    reference = {"__path__": "song.tracks[2]"}
    assert validate_value(spec, reference) == reference
    with pytest.raises(ValueError, match="handle"):
        validate_value(spec, "song.tracks[3]")


def test_an_enum_row_checks_membership() -> None:
    spec = kind_spec(Kind.ENUM, enum=[0, 1, 2])
    assert validate_value(spec, 2) == 2
    with pytest.raises(ValueError) as excinfo:
        validate_value(spec, 7)
    message = str(excinfo.value)
    assert "7" in message
    assert "[0, 1, 2]" in message, f"the allowed values are not listed: {message}"


def test_a_python_enum_member_is_unwrapped_to_its_value() -> None:
    spec = kind_spec(Kind.INT, enum=[0, 1, 2, 3, 4, 5, 6])
    assert validate_value(spec, WarpMode.TEXTURE) == 2


# --------------------------------------------------------------------------------
# validate_value: bounds, and the refusal to clamp
# --------------------------------------------------------------------------------


def test_a_value_inside_the_range_passes_including_both_bounds() -> None:
    spec = volume_spec()
    assert validate_value(spec, 0.0) == 0.0
    assert validate_value(spec, 0.85) == 0.85
    assert validate_value(spec, 1.0) == 1.0


@pytest.mark.parametrize(
    ("value", "word"), [(1.5, "maximum"), (-0.1, "minimum")], ids=["above", "below"]
)
def test_a_value_outside_the_range_is_refused_and_never_clamped(
    value: float, word: str
) -> None:
    """The client must not clamp. Live clamps, and the read-back reports it.

    ``lom_set`` returns ``requested``/``after``/``clamped`` precisely so a
    coerced write is visible (``docs/protocol.md`` §5.4). A client that clamped
    first would hand Live a value the caller never asked for and destroy the one
    signal that whole mechanism exists to produce.
    """
    with pytest.raises(ValueError) as excinfo:
        validate_value(volume_spec(), value)
    message = str(excinfo.value)
    assert word in message
    assert "track.volume" in message


def test_an_open_bound_is_open() -> None:
    spec = kind_spec(Kind.FLOAT, range=(0.0, None))
    assert validate_value(spec, 1_000_000.0) == 1_000_000.0
    with pytest.raises(ValueError, match="minimum"):
        validate_value(spec, -1.0)


def test_a_quantized_row_is_not_snapped_on_this_side() -> None:
    """A quantized parameter takes discrete steps.

    Where it lands is Live's answer. Measured: a requested 0.35 came back as
    0.25 or 0.50. The client passes the request through unchanged and lets the
    read-back report the landing, rather than guessing the step size and being
    wrong quietly.
    """
    spec = PathSpec(
        id="test.quantized",
        path="song.tempo",
        access=[Access.SET],
        kind=Kind.FLOAT,
        range=(0.0, 1.0),
        quantized=True,
        doc="A three-position switch in a 0..1 dress.",
    )
    assert validate_value(spec, 0.35) == 0.35


# --------------------------------------------------------------------------------
# The path grammar
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "song",
        "app",
        "song.tempo",
        "song.view.selected_track",
        "song.tracks[0]",
        "song.tracks[0].mixer_device.volume",
        "song.tracks[10].devices[1].parameters[23]",
        "app.browser.instruments.children[6]",
        "song.tracks[0].clip_slots[0].clip.name",
    ],
)
def test_a_legal_path_validates(path: str) -> None:
    validate_path(path)


@pytest.mark.parametrize(
    ("path", "expected_in_message"),
    [
        ("song.tracks[0:2]", "slice"),
        ("song.tracks[0:2].name", "slice"),
        ("song.tracks[-1]", "negative"),
        ("song.tracks[-1].name", "negative"),
        ("song.stop_all_clips()", "method call"),
        ("song.tracks[0].stop_all_clips()", "method call"),
        ("song.tracks[{track}].name", "placeholder"),
        ("song.tracks[007]", "leading zero"),
        (" song.tempo", "whitespace"),
        ("song.tempo\n", "whitespace"),
        ("song.tempo ", "whitespace"),
        ("song..tempo", "empty"),
        (".song.tempo", "empty"),
        ("song.tempo.", "empty"),
        ("song[0].tempo", "takes no index"),
        ("live.tracks[0]", "not resolvable"),
        ("tracks[0].name", "not resolvable"),
    ],
    ids=[
        "slice",
        "slice-midpath",
        "negative-index",
        "negative-index-midpath",
        "method-call",
        "method-call-midpath",
        "unsubstituted-placeholder",
        "leading-zero",
        "leading-space",
        "trailing-newline",
        "trailing-space",
        "doubled-dot",
        "leading-dot",
        "trailing-dot",
        "indexed-root",
        "unknown-root",
        "no-root",
    ],
)
def test_an_illegal_path_is_refused_with_a_reason(path: str, expected_in_message: str) -> None:
    """Each familiar mistake gets its own diagnosis, not "invalid syntax".

    Slices and method calls both look like ordinary Python and are both refused
    on purpose. A caller who wrote one deserves to be told which rule they hit.
    The trailing-newline case is why the patterns anchor on ``\\Z`` and not
    ``$``: ``$`` also matches before a final newline, so ``"song.tempo\\n"``
    would pass here and fail inside Live as a ``bad_path``.
    """
    with pytest.raises(ValueError) as excinfo:
        validate_path(path)
    assert expected_in_message in str(excinfo.value)


@pytest.mark.parametrize(
    "path", ["", None, 3, ["song", "tempo"]], ids=["empty", "none", "int", "list"]
)
def test_a_path_that_is_not_a_non_empty_string_is_refused(path: Any) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        validate_path(path)


def test_an_expression_in_a_path_is_refused() -> None:
    """The resolver walks attributes and indices. It is not an evaluator."""
    with pytest.raises(ValueError):
        validate_path("song.tracks[0].volume + 1")
    with pytest.raises(ValueError):
        validate_path("song.tracks[0] * 2")


def test_a_template_validates_only_when_placeholders_are_allowed() -> None:
    template = "song.tracks[{track}].mixer_device.volume"
    validate_path(template, allow_placeholders=True)
    with pytest.raises(ValueError, match="placeholder"):
        validate_path(template)


def test_a_segment_name_placeholder_validates_as_a_template() -> None:
    validate_path("app.browser.{root}.children[{index}]", allow_placeholders=True)


def test_a_template_may_not_hide_a_slice_or_a_negative_index() -> None:
    """``allow_placeholders`` widens exactly two positions and nothing else."""
    with pytest.raises(ValueError, match="slice"):
        validate_path("song.tracks[{a}:{b}]", allow_placeholders=True)
    with pytest.raises(ValueError, match="negative"):
        validate_path("song.tracks[-1].name", allow_placeholders=True)


def test_roots_none_checks_the_grammar_alone() -> None:
    """Useful in tests, never against a real script, which has fixed roots."""
    validate_path("live.tracks[0].name", roots=None)
    with pytest.raises(ValueError, match="slice"):
        validate_path("live.tracks[0:2]", roots=None)


def test_the_indexed_root_refusal_matches_the_script_and_lom_paths() -> None:
    """Neither root is a list: ``song`` is one Song, ``app`` one Application.

    The Remote Script answers ``bad_path: "the root segment takes no index"`` and
    ``lom.paths.parse`` refuses it too. This module is the client-side mirror of
    that resolver, so it has to refuse it as well: a path accepted here and
    rejected in Live is the divergence the module docstring calls a bug in one of
    the two ends.
    """
    from ableton_maestro.lom import paths as lom_paths

    for path in ("song[0].tempo", "app[0].browser"):
        with pytest.raises(ValueError):
            validate_path(path)
        with pytest.raises(lom_paths.PathSyntaxError):
            lom_paths.parse(path)


def test_a_bad_default_in_a_row_cannot_smuggle_a_malformed_path_out() -> None:
    """A malformed path never leaves :func:`build_path`.

    That holds whoever supplied the value. The placeholder here is filled from
    the parameter's own ``default``, so nothing the caller passed is involved.
    Measured against the current code, the guard that fires is the per-parameter
    coercion, not the whole-path re-check at the end, which is the better of the
    two, because it names the parameter rather than complaining about a string
    the caller never wrote. The re-check is still there as the backstop for any
    future route that gets past the coercions. Today every route is closed
    upstream of it.
    """
    spec = PathSpec(
        id="test.bad_default",
        path="app.browser.{root}.children[0]",
        access=[Access.GET],
        doc="A default that is not a legal segment name.",
        params=[ParamSpec(name="root", kind=Kind.STR, required=False, default="not a name")],
    )
    with pytest.raises(ValueError) as excinfo:
        build_path(spec)
    message = str(excinfo.value)
    assert "root" in message
    assert "test.bad_default" in message


# --------------------------------------------------------------------------------
# PathSpec: a row is validated against itself at construction
# --------------------------------------------------------------------------------


def test_a_placeholder_with_no_parameter_is_refused() -> None:
    with pytest.raises(ValueError) as excinfo:
        PathSpec(
            id="test.orphan_placeholder",
            path="song.tracks[{track}].name",
            access=[Access.GET],
            doc="No params entry.",
        )
    assert "track" in str(excinfo.value)


def test_a_parameter_that_appears_nowhere_is_refused() -> None:
    """An argument that lands nowhere is the failure this catalog prevents."""
    with pytest.raises(ValueError) as excinfo:
        PathSpec(
            id="test.orphan_param",
            path="song.tempo",
            access=[Access.GET],
            doc="A parameter with no placeholder.",
            params=[ParamSpec(name="track")],
        )
    assert "track" in str(excinfo.value)


def test_a_call_row_pointing_its_argument_at_params_is_told_where_it_belongs() -> None:
    with pytest.raises(ValueError) as excinfo:
        PathSpec(
            id="test.args_as_params",
            path="song.tracks[0]",
            access=[Access.CALL],
            method="delete_device",
            doc="Method argument filed as a path parameter.",
            params=[ParamSpec(name="index")],
        )
    assert "args" in str(excinfo.value)


def test_a_call_row_must_name_a_method() -> None:
    with pytest.raises(ValueError, match="method"):
        PathSpec(
            id="test.call_no_method",
            path="song.tracks[0]",
            access=[Access.CALL],
            doc="Callable but unnamed.",
        )


def test_a_method_without_call_access_is_refused() -> None:
    with pytest.raises(ValueError, match="call"):
        PathSpec(
            id="test.method_no_call",
            path="song.tracks[0]",
            access=[Access.GET],
            method="delete_device",
            doc="A method nothing will ever dial.",
        )


def test_args_on_a_non_call_row_are_refused() -> None:
    """On a get/set row an argument would be silently dropped."""
    with pytest.raises(ValueError) as excinfo:
        PathSpec(
            id="test.args_on_get",
            path="song.tempo",
            access=[Access.GET],
            doc="Arguments with nothing to call.",
            args=[ArgSpec(name="index", kind=Kind.INT)],
        )
    assert "index" in str(excinfo.value)


def test_a_required_argument_may_not_follow_an_optional_one() -> None:
    """``args`` goes on the wire positionally. The LOM takes no keywords."""
    with pytest.raises(ValueError, match="optional"):
        PathSpec(
            id="test.arg_order",
            path="song.tracks[0]",
            access=[Access.CALL],
            method="fire",
            doc="An unfillable argument order.",
            args=[
                ArgSpec(name="record_length", kind=Kind.FLOAT, required=False, default=1.0),
                ArgSpec(name="quantization", kind=Kind.INT, required=True),
            ],
        )


def test_one_name_may_not_be_both_a_parameter_and_an_argument() -> None:
    with pytest.raises(ValueError) as excinfo:
        PathSpec(
            id="test.name_collision",
            path="song.tracks[{index}]",
            access=[Access.CALL],
            method="delete_device",
            doc="One name, two destinations.",
            params=[ParamSpec(name="index")],
            args=[ArgSpec(name="index", kind=Kind.INT)],
        )
    assert "index" in str(excinfo.value)


def test_duplicate_parameter_names_are_refused() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        PathSpec(
            id="test.duplicate_param",
            path="song.tracks[{track}].name",
            access=[Access.GET],
            doc="Two parameters of one name.",
            params=[ParamSpec(name="track"), ParamSpec(name="track")],
        )


def test_an_index_parameter_must_be_an_int_and_a_name_parameter_must_not_be() -> None:
    with pytest.raises(ValueError, match="index"):
        PathSpec(
            id="test.index_as_str",
            path="song.tracks[{track}].name",
            access=[Access.GET],
            doc="An index typed as a string.",
            params=[ParamSpec(name="track", kind=Kind.STR)],
        )
    with pytest.raises(ValueError, match="segment name"):
        PathSpec(
            id="test.name_as_int",
            path="app.browser.{root}.children[0]",
            access=[Access.GET],
            doc="A segment name typed as an int.",
            params=[ParamSpec(name="root", kind=Kind.INT)],
        )


def test_an_optional_parameter_without_a_default_is_refused() -> None:
    """Every placeholder must end up filled or the path cannot be built."""
    with pytest.raises(ValueError, match="default"):
        PathSpec(
            id="test.optional_no_default",
            path="song.tracks[{track}].name",
            access=[Access.GET],
            doc="Optional with nothing to fall back on.",
            params=[ParamSpec(name="track", required=False)],
        )


def test_an_enum_row_must_list_its_values() -> None:
    with pytest.raises(ValueError, match="enum"):
        PathSpec(
            id="test.enum_no_values",
            path="song.tempo",
            access=[Access.GET],
            kind=Kind.ENUM,
            doc="An enum that validates nothing.",
        )


@pytest.mark.parametrize(
    "bounds",
    [(1.0, 0.0), (0.0, float("nan")), (float("inf"), 1.0), (0.0,), (0.0, 1.0, 2.0), ("a", "b")],
    ids=["inverted", "nan", "infinite", "one-element", "three-element", "not-numbers"],
)
def test_a_malformed_range_is_refused(bounds: Any) -> None:
    with pytest.raises(ValueError):
        PathSpec(
            id="test.bad_range",
            path="song.tempo",
            access=[Access.GET],
            kind=Kind.FLOAT,
            range=bounds,
            doc="A range that cannot be applied.",
        )


def test_a_row_must_declare_at_least_one_access() -> None:
    with pytest.raises(ValueError, match="access"):
        PathSpec(id="test.no_access", path="song.tempo", access=[], doc="Nothing may be done.")


def test_a_row_starts_out_untested() -> None:
    """Unproven is the default.

    A capability is a claim until somebody probes it.
    """
    spec = PathSpec(id="test.default_status", path="song.tempo", doc="A fresh row.")
    assert spec.status is PathStatus.UNTESTED
    assert spec.unit is Unit.NORMALIZED
    assert spec.verify == "read_back"
    assert spec.destructive is False
    assert spec.access == [Access.GET]


def test_strings_from_a_yaml_loader_are_converted_to_their_enums() -> None:
    """The registry hands over raw YAML. The dataclass normalises it itself."""
    spec = PathSpec(
        id="test.from_yaml",
        path="song.tracks[{track}].mixer_device.volume",
        access=["get", "set"],
        kind="float",
        unit="normalized",
        status="broken",
        range=[0.0, 1.0],
        doc="Built the way registry.py builds one.",
        params=[ParamSpec(name="track", kind="int")],
    )
    assert spec.access == [Access.GET, Access.SET]
    assert spec.kind is Kind.FLOAT
    assert spec.unit is Unit.NORMALIZED
    assert spec.status is PathStatus.BROKEN
    assert spec.range == (0.0, 1.0)
    assert spec.params[0].kind is Kind.INT


# --------------------------------------------------------------------------------
# require_access
# --------------------------------------------------------------------------------


def test_require_access_permits_what_the_catalog_declares() -> None:
    spec = volume_spec()
    require_access(spec, Access.GET)
    require_access(spec, Access.SET)
    assert spec.supports(Access.SET) is True
    assert spec.supports(Access.CALL) is False


def test_require_access_refuses_what_it_does_not_and_lists_what_it_does() -> None:
    with pytest.raises(ValueError) as excinfo:
        require_access(volume_spec(), Access.CALL)
    message = str(excinfo.value)
    assert "call" in message
    assert "track.volume" in message
    assert "get" in message and "set" in message


def test_call_access_here_is_not_the_last_word() -> None:
    """The catalog's ``call`` is a request, not the answer.

    The script's allowlist answers it. ``docs/protocol.md`` §6: the allowlist
    lives inside Live so it cannot be widened from outside, the two must agree,
    and the script wins. This test exists to pin that ``require_access`` is only
    the early, cheap refusal: it passes for a row the script may still refuse.
    """
    spec = PathSpec(
        id="test.callable",
        path="song.tracks[0]",
        access=[Access.CALL],
        method="a_method_the_script_has_never_heard_of",
        doc="Catalog says yes; only Live can say whether it happens.",
    )
    require_access(spec, Access.CALL)
    assert spec.method == "a_method_the_script_has_never_heard_of"
