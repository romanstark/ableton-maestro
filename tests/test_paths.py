"""Tests for :mod:`ableton_maestro.lom.paths` resolver mirror.

Pure string work, so there is nothing to fake: this module opens no socket and
never sees Live (docs/architecture.md, 'Layers'). What the tests are actually protecting
is the *rejection* half of the grammar in ``docs/protocol.md`` §6. A slice, a
negative index or a method call are all legal Python and all refused on purpose,
and each has to come back with a reason a caller can act on rather than
"invalid syntax": the module's own promise, and the difference between a
one-line fix and an afternoon.

The last group cross-checks the helpers against the real catalog. A helper
that builds ``song.tracks[3].mixer_device.volume`` while the catalog row for
``track.volume`` says something else is a bug that no unit test of either half
would catch on its own.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import pytest

from ableton_maestro.lom import paths
from ableton_maestro.lom.paths import PathSyntaxError, Segment
from ableton_maestro.registry import default_registry
from ableton_maestro.spec import build_path

# --------------------------------------------------------------------- parsing

LEGAL = [
    "song",
    "app",
    "song.tempo",
    "song.tracks[0]",
    "song.tracks[3].mixer_device.volume",
    "song.tracks[0].clip_slots[2].clip.warping",
    "song.tracks[0].devices[1].parameters[5].value",
    "song.view.selected_track",
    "song.master_track.mixer_device.volume",
    "song.return_tracks[1].devices[0]",
    "app.browser.instruments.children[6]",
    "song.tracks[10].name",
    "song._private_looking_name",
]


@pytest.mark.parametrize("path", LEGAL)
def test_legal_paths_parse(path: str) -> None:
    """Everything the grammar allows parses, and round-trips unchanged."""
    assert paths.is_valid(path)
    paths.validate(path)
    assert paths.unparse(paths.parse(path)) == path


def test_parse_returns_segments_root_first() -> None:
    """Segments come back in order, the root first, indices attached."""
    segments = paths.parse("song.tracks[3].devices[0].parameters[5]")
    assert segments[0] == Segment("song", None)
    assert segments[1] == Segment("tracks", 3)
    assert [str(s) for s in segments] == [
        "song",
        "tracks[3]",
        "devices[0]",
        "parameters[5]",
    ]
    assert segments[1].is_indexed
    assert not segments[0].is_indexed


def test_song_view_is_a_segment_not_a_root() -> None:
    """Verify ``song.view`` resolves through standard attribute resolution."""
    assert paths.ROOTS == ("song", "app")
    assert [str(s) for s in paths.parse("song.view.selected_track")] == [
        "song",
        "view",
        "selected_track",
    ]


# ------------------------------------------------------------------- rejection
#
# (path, a substring the reason must contain). The substring is the point of
# each row: the module promises a diagnosis, not a verdict.
ILLEGAL = [
    ("song.tracks[0:2]", "slice"),
    ("song.tracks[1:]", "slice"),
    ("song.tracks[-1]", "negative"),
    ("song.tracks[-1].name", "negative"),
    ("song.tracks[0].stop_all_clips()", "lom_call"),
    ("song.stop_playing()", "lom_call"),
    ("song.tracks[0+1]", "not a non-negative integer"),
    ("song.tracks[0].name * 2", "expression"),
    ("song.tempo + 1", "expression"),
    ("song..tracks[0]", "empty segment"),
    (".song.tracks[0]", "empty segment"),
    ("song.tracks[0].", "empty segment"),
    ("live.tracks[0]", "not a root"),
    ("Song.tempo", "not a root"),
    ("tracks[0].name", "not a root"),
    ("song[0]", "takes no index"),
    ("song.tracks[007]", "leading zero"),
    ("song.tracks[01]", "leading zero"),
    ("song.tracks[0", "unclosed"),
    ("song.tracks[0]x", "after ']'"),
    ("song.tracks[]", "empty index"),
    ("song.tracks[abc]", "not a non-negative integer"),
    # A dot inside the brackets splits the segment before the ']' is seen, so
    # the honest complaint is the unclosed bracket rather than the number.
    ("song.tracks[1.5]", "unclosed"),
    ("song.[0]", "expected an attribute name first"),
    ("song. tracks[0]", "whitespace"),
    ("song.tracks[{track}]", "{placeholder}"),
    ("song.9lives", "not a valid attribute name"),
    ("", "empty path"),
    ("   ", "empty path"),
]


@pytest.mark.parametrize(("path", "reason"), ILLEGAL)
def test_illegal_paths_are_refused_with_a_reason(path: str, reason: str) -> None:
    """Every rejection names what was wrong, and carries the wire error code."""
    assert not paths.is_valid(path)
    with pytest.raises(PathSyntaxError) as caught:
        paths.parse(path)
    assert reason in caught.value.reason
    assert caught.value.path == path
    # The Remote Script answers this class of failure with code "bad_path"
    # (docs/protocol.md §4). One name whichever side rejected it.
    assert caught.value.code == "bad_path"
    assert PathSyntaxError.code == "bad_path"


def test_a_non_string_is_refused_rather_than_coerced() -> None:
    """``parse(3)`` is a caller bug; saying so beats stringifying it."""
    with pytest.raises(PathSyntaxError) as caught:
        paths.parse(3)  # type: ignore[arg-type]
    assert "expected a string" in caught.value.reason


def test_is_valid_never_raises() -> None:
    """The predicate is a predicate: it is used where an exception is not wanted."""
    for candidate in ("", "song.tracks[-1]", "song.tracks[0:2]", "!!!", "song"):
        assert isinstance(paths.is_valid(candidate), bool)


def test_surrounding_whitespace_is_tolerated_but_inner_whitespace_is_not() -> None:
    assert paths.is_valid("  song.tracks[0]  ")
    assert not paths.is_valid("song.tracks [0]")


def test_syntax_error_reports_a_position_where_it_can_pin_one_down() -> None:
    with pytest.raises(PathSyntaxError) as caught:
        paths.parse("song.tracks[-1].name")
    assert caught.value.position == len("song.")


def test_unparse_refuses_an_empty_segment_list() -> None:
    with pytest.raises(ValueError, match="empty segment list"):
        paths.unparse([])


# ---------------------------------------------------------------------- build


def test_build_substitutes_every_placeholder() -> None:
    assert (
        paths.build("song.tracks[{track}].mixer_device.volume", track=3)
        == "song.tracks[3].mixer_device.volume"
    )
    assert (
        paths.build("song.tracks[{t}].devices[{d}].parameters[{p}]", t=0, d=1, p=2)
        == "song.tracks[0].devices[1].parameters[2]"
    )
    assert paths.build("song.tempo") == "song.tempo"


def test_build_substitutes_a_repeated_placeholder_once_per_occurrence() -> None:
    assert paths.build("song.tracks[{n}].devices[{n}]", n=2) == "song.tracks[2].devices[2]"


def test_build_fills_a_segment_name_placeholder() -> None:
    """The browser's roots are addressed by name, not by index (protocol §6)."""
    built = paths.build("app.browser.{root}.children[{index}]", root="instruments", index=6)
    assert built == "app.browser.instruments.children[6]"
    assert paths.is_valid(built)


def test_build_refuses_a_missing_argument() -> None:
    with pytest.raises(ValueError, match="missing argument"):
        paths.build("song.tracks[{track}].name")


def test_build_refuses_an_unknown_argument() -> None:
    """A keyword nobody reads is the failure class this project exists against.

    docs/architecture.md, 'read-back as a principle': an argument silently dropped is how a whole
    set of
    automation curves landed in slot 0 while every call reported success.
    """
    with pytest.raises(ValueError, match="unknown argument"):
        paths.build("song.tracks[{track}].name", track=1, slot=0)


def test_build_refuses_a_bool_as_an_index() -> None:
    """Verify boolean arguments are rejected in path index construction."""
    with pytest.raises(TypeError, match="bool"):
        paths.build("song.tracks[{track}]", track=True)


def test_build_refuses_a_negative_index() -> None:
    with pytest.raises(ValueError, match="negative"):
        paths.build("song.tracks[{track}]", track=-1)


def test_build_accepts_an_integral_float_and_refuses_a_fractional_one() -> None:
    """JSON has one number type; punishing a caller for ``3.0`` would be pedantry."""
    assert paths.build("song.tracks[{track}]", track=3.0) == "song.tracks[3]"
    with pytest.raises(TypeError, match="whole number"):
        paths.build("song.tracks[{track}]", track=3.5)


def test_build_validates_the_finished_path() -> None:
    """A name substitution that is not an identifier must not escape as a path."""
    with pytest.raises(PathSyntaxError):
        paths.build("app.browser.{root}", root="0bad")
    with pytest.raises(PathSyntaxError):
        paths.build("song.tracks[{track}].{attr}", track=0, attr="name()")


# --------------------------------------------------------------- parent / join


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("song.tracks[3].devices[0].parameters[5]", "song.tracks[3].devices[0].parameters"),
        ("song.tracks[3].devices[0].parameters", "song.tracks[3].devices[0]"),
        ("song.tracks[3].devices[0]", "song.tracks[3].devices"),
        ("song.tempo", "song"),
        ("song.view.selected_track", "song.view"),
    ],
)
def test_parent_peels_the_index_before_the_segment(path: str, expected: str) -> None:
    """The list is a real place in the model, so it gets its own step."""
    assert paths.parent(path) == expected


def test_parent_of_a_root_is_none() -> None:
    assert paths.parent("song") is None
    assert paths.parent("app") is None


def test_join_appends_a_segment_or_a_dotted_tail() -> None:
    assert paths.join("song.tracks[0]", "devices[2]") == "song.tracks[0].devices[2]"
    assert paths.join("song.tracks[0]", "mixer_device.volume") == (
        "song.tracks[0].mixer_device.volume"
    )
    assert paths.join("song.tracks[0]", Segment("name")) == "song.tracks[0].name"


def test_join_refuses_a_root_as_the_tail() -> None:
    """Two paths glued together is a mistake worth naming."""
    with pytest.raises(PathSyntaxError, match="is a root"):
        paths.join("song.tracks[0]", "song.tempo")


def test_join_refuses_an_empty_tail_and_a_bad_one() -> None:
    with pytest.raises(PathSyntaxError):
        paths.join("song.tracks[0]", "")
    with pytest.raises(PathSyntaxError):
        paths.join("song.tracks[0]", "devices[-1]")


# -------------------------------------------------------------------- helpers


def test_helpers_build_the_expected_paths() -> None:
    assert paths.track(0) == "song.tracks[0]"
    assert paths.return_track(1) == "song.return_tracks[1]"
    assert paths.master() == "song.master_track"
    assert paths.scene(4) == "song.scenes[4]"
    assert paths.clip_slot(2, 0) == "song.tracks[2].clip_slots[0]"
    assert paths.clip(2, 0) == "song.tracks[2].clip_slots[0].clip"
    assert paths.mixer(3) == "song.tracks[3].mixer_device"
    assert paths.send(0, 1) == "song.tracks[0].mixer_device.sends[1]"
    assert paths.device(3, 0) == "song.tracks[3].devices[0]"
    assert paths.parameter(3, 0, 5) == "song.tracks[3].devices[0].parameters[5]"


def test_every_helper_produces_a_parseable_path() -> None:
    """Whatever a helper returns must survive the grammar it is a shortcut for."""
    produced = [
        paths.track(7),
        paths.return_track(0),
        paths.master(),
        paths.scene(12),
        paths.clip_slot(1, 2),
        paths.clip(1, 2),
        paths.mixer(paths.master()),
        paths.send(0, 0),
        paths.device(paths.return_track(0), 1),
        paths.parameter(paths.master(), 0, 2),
    ]
    for path in produced:
        assert paths.is_valid(path), path


def test_helpers_accept_a_track_path_as_well_as_an_index() -> None:
    assert paths.device(paths.return_track(0), 1) == "song.return_tracks[0].devices[1]"
    assert paths.mixer(paths.master()) == "song.master_track.mixer_device"
    assert paths.parameter(paths.master(), 0, 2) == "song.master_track.devices[0].parameters[2]"


def test_the_master_track_has_no_sends() -> None:
    """A real LOM shape, refused here rather than by a round trip into Live."""
    with pytest.raises(ValueError, match="no sends"):
        paths.send(paths.master(), 0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: paths.track(-1),
        lambda: paths.scene(-2),
        lambda: paths.clip_slot(0, -1),
        lambda: paths.device(0, -3),
        lambda: paths.parameter(0, 0, -1),
        lambda: paths.return_track(-1),
    ],
)
def test_helpers_refuse_negative_indices(call: Callable[[], str]) -> None:
    with pytest.raises(ValueError, match="negative"):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda: paths.track(True),
        lambda: paths.device(True, 0),
        lambda: paths.parameter(0, 0, True),
        lambda: paths.track("song.tracks[0]"),
    ],
)
def test_helpers_refuse_a_bool_or_a_wrong_type(call: Callable[[], str]) -> None:
    with pytest.raises(TypeError):
        call()


def test_a_helper_given_a_malformed_track_path_refuses_it() -> None:
    with pytest.raises(PathSyntaxError):
        paths.device("song.tracks[-1]", 0)


# ------------------------------------------------- helpers against the catalog
#
# (helper result, catalog id, the kwargs that row's template takes). Both halves
# describe the same place in the object model; if they ever disagree, one of them
# is wrong and nothing else in the repo would notice.
CATALOG_PAIRS = [
    (paths.mixer(3) + ".volume", "track.volume", {"track": 3}),
    (paths.send(3, 1), "track.send", {"track": 3, "send": 1}),
    (paths.clip(2, 0), "clip_slot.clip", {"track": 2, "slot": 0}),
    (paths.clip_slot(2, 0), "clip_slot.fire", {"track": 2, "slot": 0}),
    (paths.parameter(3, 0, 5), "param.value", {"track": 3, "device": 0, "param": 5}),
    (paths.device(3, 0) + ".parameters[0]", "device.on", {"track": 3, "device": 0}),
    (
        paths.return_track(1) + ".mixer_device.volume",
        "return.volume",
        {"ret": 1},
    ),
    (paths.master(), "song.master_track", {}),
    (
        paths.parameter(paths.master(), 0, 2),
        "master_device.parameter",
        {"device": 0, "param": 2},
    ),
    (paths.scene(4) + ".name", "scene.name", {"scene": 4}),
]


@pytest.mark.parametrize(("built", "spec_id", "args"), CATALOG_PAIRS)
def test_a_helper_and_its_catalog_row_agree(built: str, spec_id: str, args: dict) -> None:
    spec = default_registry().get(spec_id)
    assert built == build_path(spec, **args)


_INDEX_PLACEHOLDER = re.compile(r"\[\{[A-Za-z_][A-Za-z0-9_]*\}\]")
_NAME_PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")


def test_every_catalog_template_parses_once_it_is_filled() -> None:
    """The catalog and this grammar are two descriptions of one resolver.

    A row whose path this module refuses would be a row the server could never
    send. Index placeholders get 0, name placeholders get a legal identifier.
    """
    registry = default_registry()
    assert registry.all(), "the packaged catalog is empty: the test proves nothing"
    for spec in registry.all():
        filled = _NAME_PLACEHOLDER.sub("instruments", _INDEX_PLACEHOLDER.sub("[0]", spec.path))
        assert paths.is_valid(filled), f"{spec.id}: {spec.path!r} -> {filled!r}"


def test_the_catalog_uses_no_root_this_module_does_not_know() -> None:
    for spec in default_registry().all():
        root = paths.parse(_NAME_PLACEHOLDER.sub("x", _INDEX_PLACEHOLDER.sub("[0]", spec.path)))[0]
        assert root.name in paths.ROOTS, f"{spec.id}: unknown root {root.name!r}"


# ------------------------------------------------------------------ describing


def test_describe_path_reads_as_english_and_gives_the_index_both_ways() -> None:
    """A human counts tracks from one, the protocol from zero. Say both."""
    assert describe("song.tracks[3].mixer_device.volume") == (
        "the song, track 4 (index 3), the mixer device, 'volume'"
    )
    assert describe("song.tracks[0].clip_slots[2].clip.warping") == (
        "the song, track 1 (index 0), clip slot 3 (index 2), the clip, 'warping'"
    )
    assert describe("song.tracks[0].devices[1].parameters[5]") == (
        "the song, track 1 (index 0), device 2 (index 1), parameter 6 (index 5)"
    )


def describe(path: str) -> str:
    """Local alias so the assertions above read as one line each."""
    return paths.describe_path(path)


def test_describe_path_knows_the_awkward_plurals() -> None:
    assert "clip slot 1 (index 0)" in describe("song.tracks[0].clip_slots[0]")
    assert "return track 2 (index 1)" in describe("song.return_tracks[1]")
    assert "locator 1 (index 0)" in describe("song.cue_points[0]")
    assert "the master track" in describe("song.master_track")


def test_describe_path_explains_a_bad_path_instead_of_raising() -> None:
    """It exists to make errors readable, so it must not become one."""
    text = describe("song.tracks[-1]")
    assert "is not a valid LOM path" in text
    assert "negative indices are not supported" in text
    assert describe("").startswith("'' is not a valid LOM path")
