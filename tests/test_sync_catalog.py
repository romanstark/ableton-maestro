"""The reconciliation rules of ``scripts/sync_catalog.py``, exercised offline.

Nothing here opens a socket or needs Ableton Live. The tool reads Live through
exactly two handlers, so a recorded ``{"describe": {...}, "get": {...}}`` payload
is a complete stand-in for one, which is also the ``--replay`` format users get,
so these tests exercise the shipped path rather than a test-only one.

What is actually under test is the deletion policy (``docs/catalog.md``,
"Three outcomes, not two"). Getting it wrong in the safe direction costs a rerun;
getting it wrong in the unsafe direction deletes a real capability, or plants a
``broken`` row that says something false about the Live Object Model and outlives
the session that wrote it. So each of the three outcomes has a test, and so does
every guard that keeps absence from being read as evidence.

Provenance of the payloads below: they are read from the source, so the answer
shapes are ``docs/protocol.md`` §5.6 and the error codes are the table in §4,
"Message shape", and hand-built to put one branch each under test. They are
*not* a recording of a real Live, and nothing here should be quoted as a
measurement of one. What they prove is that the tool draws the right conclusion
from a given answer, which is the half that can be checked without Ableton open.
"""

from __future__ import annotations

import io
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "scripts") not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import sync_catalog as sc
from test_catalog import ID_PATTERN, has_measurement_note

from ableton_maestro.models import Kind, PathStatus, Unit
from ableton_maestro.registry import Registry

# --------------------------------------------------------------------------------
# Fixtures: a tiny catalog, a tiny Remote Script, and a recorded Live
# --------------------------------------------------------------------------------

CATALOG = """\
# This comment must survive every edit this file ever receives.

# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------

- id: track.name
  path: song.tracks[{track}].name
  access: [get, set]
  kind: str
  unit: none
  destructive: false
  verify: read_back
  status: untested
  doc: >-
    Track name, a hypothesis until somebody reads it.
  params:
    - {name: track, kind: int, required: true}

- id: track.ghost
  path: song.tracks[{track}].ghost
  access: [get]
  kind: bool
  unit: none
  destructive: false
  verify: read_back
  status: untested
  doc: "A property nobody has ever seen."
  params:
    - {name: track, kind: int, required: true}

- id: track.raiser
  path: song.tracks[{track}].raiser
  access: [get]
  kind: bool
  unit: none
  destructive: false
  verify: read_back
  status: untested
  doc: >-
    A property Live has and refuses.
  params:
    - {name: track, kind: int, required: true}

- id: track.arm
  path: song.tracks[{track}].arm
  access: [get, set]
  kind: bool
  unit: none
  destructive: false
  verify: read_back
  status: untested
  doc: >-
    Arm flag. Group, return and main tracks have none.
  params:
    - {name: track, kind: int, required: true}

- id: track.stop_all
  path: song.tracks[{track}]
  access: [call]
  method: stop_all_clips
  kind: object
  unit: none
  destructive: false
  verify: none
  status: untested
  doc: >-
    Stop every clip on this track.
  params:
    - {name: track, kind: int, required: true}

# ---------------------------------------------------------------------------
# Devices - one shape, several Live classes
# ---------------------------------------------------------------------------

- id: device.name
  path: song.tracks[{track}].devices[{device}].name
  access: [get]
  kind: str
  unit: none
  destructive: false
  verify: read_back
  status: untested
  doc: >-
    Device name.
  params:
    - {name: track, kind: int, required: true}
    - {name: device, kind: int, required: true}

- id: simpler.sample_length
  path: song.tracks[{track}].devices[{device}].sample.length
  access: [get]
  kind: int
  unit: none
  destructive: false
  verify: read_back
  status: untested
  doc: >-
    Sample length in frames. Only a Simpler has one.
  params:
    - {name: track, kind: int, required: true}
    - {name: device, kind: int, required: true}

- id: rack.macro_count
  path: song.tracks[{track}].devices[{device}].macro_count
  access: [get]
  kind: int
  unit: none
  destructive: false
  verify: read_back
  status: untested
  doc: >-
    How many macros the rack has. Only a rack has any.
  params:
    - {name: track, kind: int, required: true}
    - {name: device, kind: int, required: true}

- id: scene.name
  path: song.scenes[{scene}].name
  access: [get]
  kind: str
  unit: none
  destructive: false
  verify: read_back
  status: untested
  doc: >-
    Scene name.
  params:
    - {name: scene, kind: int, required: true}
"""

SCRIPT = '''\
"""A stand-in Remote Script. Only the two module-level constants are read."""

METHOD_ALLOWLIST = frozenset([
    "Track.stop_all_clips",
    "Track.delete_device",
])

_DESCRIBE_SKIP = frozenset(["canonical_parent"])
'''

SPECIMENS: dict[str, str] = {
    "Song": "song",
    "Track": "song.tracks[0]",
    "Device": "song.tracks[0].devices[0]",
    "RackDevice": "song.tracks[0].devices[1]",
    "Scene": "song.scenes[0]",
}


def _describe(
    path: str,
    cls: str,
    properties: list[dict[str, Any]],
    children: list[dict[str, Any]] | None = None,
    methods: list[str] | None = None,
    allowed: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path,
        "class": cls,
        "properties": properties,
        "children": children or [],
        "methods": sorted(methods or []),
        "allowed_methods": sorted(allowed or []),
        "budget_left": 300,
    }
    result.update(extra)
    return {"ok": True, "result": result}


def _error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "code": code, "message": message}


def replay() -> dict[str, Any]:
    """One recorded run: a Track, two devices, an empty scene list."""
    return {
        "script_info": {
            "live_version": "12.4.5",
            "script_version": "0.4.0",
            "handlers": ["lom_get", "lom_describe"],
        },
        "describe": {
            "song": _describe("song", "Song", [], methods=["undo"]),
            "song.tracks[0]": _describe(
                "song.tracks[0]",
                "Track",
                [
                    {"name": "name", "type": "string", "settable": True, "value": "Drums"},
                    {
                        "name": "raiser",
                        "type": None,
                        "settable": None,
                        "error": "RuntimeError: Main and Return Tracks have no 'Arm' state!",
                    },
                    {
                        "name": "arm",
                        "type": None,
                        "settable": False,
                        "unavailable": "group, return and main tracks have no arm state "
                        "(can_be_armed=False)",
                    },
                    {"name": "color_index", "type": "int", "settable": None, "value": 12},
                    {"name": "canonical_parent", "type": "null", "settable": None,
                     "value": None},
                ],
                children=[
                    {
                        "name": "devices",
                        "type": "list",
                        "class": "Vector",
                        "is_collection": True,
                        "count": 4,
                        "path": "song.tracks[0].devices",
                    }
                ],
                methods=["stop_all_clips", "delete_device", "secret_method"],
                allowed=["stop_all_clips", "delete_device"],
            ),
            "song.tracks[0].devices[0]": _describe(
                "song.tracks[0].devices[0]",
                "Compressor",
                [{"name": "name", "type": "string", "settable": None, "value": "Compressor"}],
            ),
            "song.tracks[0].devices[1]": _describe(
                "song.tracks[0].devices[1]",
                "InstrumentGroupDevice",
                [{"name": "name", "type": "string", "settable": None, "value": "Rack"}],
            ),
            "song.tracks[0].devices[0].sample": _error(
                "no_such_path", "Compressor has no attribute 'sample'"
            ),
            "song.tracks[0].devices[1].sample": _error(
                "no_such_path", "InstrumentGroupDevice has no attribute 'sample'"
            ),
            "song.scenes[0]": _error("index_out_of_range", "song.scenes has 0 element(s)"),
        },
        "get": {
            "song.tracks[0].ghost": _error(
                "no_such_path", "Track has no attribute 'ghost'"
            ),
            "song.tracks[0].raiser": _error(
                "live_error", "RuntimeError: Main and Return Tracks have no 'Arm' state!"
            ),
            "song.tracks[0].devices[0].macro_count": _error(
                "no_such_path", "Compressor has no attribute 'macro_count'"
            ),
            "song.tracks[0].devices[1].macro_count": _error(
                "no_such_path", "InstrumentGroupDevice has no attribute 'macro_count'"
            ),
        },
    }


#: One extra catalog file, written with CRLF endings, used where a test needs to
#: prove that "unchanged" also means "not silently re-encoded". ``song.tempo`` is
#: absent from the recording on purpose: the run reaches it, learns nothing, and
#: therefore has no excuse to touch the file.
CRLF_FILE = (
    "# A second file, in the other line ending. Also prose that must survive.\n"
    "- id: song.tempo\n"
    "  path: song.tempo\n"
    "  access: [get]\n"
    "  kind: float\n"
    "  unit: none\n"
    "  destructive: false\n"
    "  verify: read_back\n"
    "  status: untested\n"
    '  doc: "Tempo in BPM."\n'
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A catalog directory and a stand-in Remote Script on disk."""
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "10-test.yaml").write_text(CATALOG, encoding="utf-8", newline="\n")
    (tmp_path / "script.py").write_text(SCRIPT, encoding="utf-8", newline="\n")
    return tmp_path


@pytest.fixture
def cli(workspace: Path, tmp_path: Path) -> Callable[..., int]:
    """Run ``sync_catalog.main`` end to end, offline, against the fixture.

    ``--replay`` is the shipped way to reconcile without a socket, so a CLI test
    exercises the same code path a user does, argument parsing and the write gate and
    the re-parse included, rather than a test-only shortcut past it.
    """

    def invoke(*extra: str, recorded: dict[str, Any] | None = None) -> int:
        recording = tmp_path / "replay.json"
        recording.write_text(
            json.dumps(recorded if recorded is not None else replay()), encoding="utf-8"
        )
        specimen_file = tmp_path / "specimens.json"
        # null removes a shipped default, so this is the whole specimen map replaced
        # rather than merged: the reference set's tracks are not this fixture's.
        specimen_file.write_text(
            json.dumps({**dict.fromkeys(sc.DEFAULT_SPECIMENS), **SPECIMENS}),
            encoding="utf-8",
        )
        return sc.main(
            [
                "--catalog",
                str(workspace / "catalog"),
                "--script",
                str(workspace / "script.py"),
                "--specimens",
                str(specimen_file),
                "--replay",
                str(recording),
                *extra,
            ]
        )

    return invoke


def run(workspace: Path, *, recorded: dict[str, Any] | None = None) -> sc.Report:
    """Reconcile the fixture catalog against a recorded run."""
    catalog = workspace / "catalog"
    registry = Registry.load(catalog)
    reader = sc.Reader(None, replay=recorded if recorded is not None else replay(), pace=0.0)
    return sc.Reconciler(
        registry,
        reader,
        {name: sc.Specimen(name, path) for name, path in SPECIMENS.items()},
        allowlist=sc._literal_string_set(workspace / "script.py", "METHOD_ALLOWLIST"),
        describe_skip=sc._literal_string_set(workspace / "script.py", "_DESCRIBE_SKIP"),
        catalog_root=catalog,
    ).run()


def verdict_of(report: sc.Report, spec_id: str) -> sc.RowFinding:
    for finding in report.findings:
        if finding.spec_id == spec_id:
            return finding
    raise AssertionError(f"{spec_id} was not judged at all")


def rows_on_disk(file: Path) -> dict[str, dict[str, Any]]:
    """The catalog file as raw YAML, keyed by id.

    Read as data rather than through :class:`Registry`, for the reason
    ``tests/test_catalog.py`` gives: the loader normalises, and a test that only
    asks the loader stops noticing the day the loader gets looser. What sync_catalog
    writes is text, so text is what gets checked.
    """
    parsed = yaml.safe_load(file.read_text(encoding="utf-8"))
    assert isinstance(parsed, list), f"{file.name} is no longer a list of rows"
    return {row["id"]: row for row in parsed}


def catalog_bytes(catalog: Path) -> dict[str, bytes]:
    """Every catalog file, verbatim: line endings and trailing bytes included."""
    return {file.name: file.read_bytes() for file in sorted(catalog.glob("*.yaml"))}


# --------------------------------------------------------------------------------
# The path template grammar
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "shape"),
    [
        ("song", "song"),
        ("song.tracks[0]", "song.tracks[]"),
        ("song.tracks[{track}].mixer_device", "song.tracks[].mixer_device"),
        ("app.browser.{root}.children[{i}]", "app.browser.{}.children[]"),
    ],
)
def test_shape_erases_indices_and_placeholders(path: str, shape: str) -> None:
    """Two paths naming the same *place* must compare equal, whatever the index."""
    assert sc.shape_of(sc.split_template(path)) == shape


def test_split_template_rejects_what_the_resolver_would() -> None:
    """The grammar is protocol §6's, so a leading zero is not a second spelling."""
    with pytest.raises(sc.TemplateError):
        sc.split_template("song.tracks[007]")


def test_target_of_splits_a_property_row_from_its_container() -> None:
    registry = Registry.load(_REPO_ROOT / "src" / "ableton_maestro" / "catalog")
    spec = registry.get("track.volume")
    target = sc.target_of(spec)
    assert target is not None
    assert target.container == "song.tracks[{track}].mixer_device"
    assert target.member == "volume"
    assert target.is_method is False


def test_target_of_makes_a_call_rows_whole_path_the_container() -> None:
    """A ``call`` row addresses the object; its member is the method name."""
    registry = Registry.load(_REPO_ROOT / "src" / "ableton_maestro" / "catalog")
    spec = registry.get("song.undo")
    target = sc.target_of(spec)
    assert target is not None
    assert target.container == spec.path
    assert target.member == "undo"
    assert target.is_method is True


# --------------------------------------------------------------------------------
# Resolving a container onto a specimen
# --------------------------------------------------------------------------------


def specimens() -> dict[str, sc.Specimen]:
    return {name: sc.Specimen(name, path) for name, path in SPECIMENS.items()}


def test_an_exact_shape_match_is_determined() -> None:
    resolutions, reason = sc.resolve_container("song.tracks[{track}]", specimens())
    assert reason is None
    assert [r.probe_path for r in resolutions] == ["song.tracks[0]"]
    assert resolutions[0].filled == ()


def test_a_deeper_container_borrows_the_longest_specimen_prefix() -> None:
    """``mixer_device`` has no specimen, so the Track's is extended by one segment."""
    resolutions, _ = sc.resolve_container("song.tracks[{track}].mixer_device", specimens())
    assert [r.probe_path for r in resolutions] == ["song.tracks[0].mixer_device"]
    # No index was invented, so this is still a determined probe.
    assert resolutions[0].filled == ()


def test_an_invented_index_is_recorded_as_filled() -> None:
    resolutions, _ = sc.resolve_container(
        "song.tracks[{track}].arrangement_clips[{clip}]", specimens()
    )
    assert [r.probe_path for r in resolutions] == ["song.tracks[0].arrangement_clips[0]"]
    assert resolutions[0].filled == ("arrangement_clips",)


def test_a_shape_two_specimens_share_resolves_to_both() -> None:
    """``devices[]`` is polymorphic and the union of both is the honest answer."""
    resolutions, _ = sc.resolve_container(
        "song.tracks[{track}].devices[{device}]", specimens()
    )
    assert sorted(r.probe_path for r in resolutions) == [
        "song.tracks[0].devices[0]",
        "song.tracks[0].devices[1]",
    ]


def test_a_segment_name_placeholder_is_bound_by_the_specimen() -> None:
    browser = {"BrowserItem": sc.Specimen("BrowserItem", "app.browser.instruments")}
    resolutions, _ = sc.resolve_container("app.browser.{root}", browser)
    assert [r.probe_path for r in resolutions] == ["app.browser.instruments"]


def test_a_name_placeholder_nothing_binds_is_unresolved() -> None:
    resolutions, reason = sc.resolve_container("app.browser.{root}", specimens())
    assert resolutions == []
    assert reason and "no specimen" in reason


# --------------------------------------------------------------------------------
# The three outcomes
# --------------------------------------------------------------------------------


def test_a_read_that_worked_confirms_the_row_and_flips_its_status(workspace: Path) -> None:
    report = run(workspace)
    finding = verdict_of(report, "track.name")
    assert finding.verdict == sc.CONFIRMED
    assert finding.new_status is PathStatus.VERIFIED
    assert has_measurement_note(finding.note), finding.note
    assert "song.tracks[0].name" in finding.note
    assert "No write" in finding.note


def test_no_such_path_on_a_determined_probe_deletes_the_row(workspace: Path) -> None:
    """The policy's first case: the property does not exist, so the row is fantasy."""
    report = run(workspace)
    finding = verdict_of(report, "track.ghost")
    assert finding.verdict == sc.MISSING
    assert [f.spec_id for f in report.deletions] == ["track.ghost"]


def test_live_error_keeps_the_row_and_marks_it_broken(workspace: Path) -> None:
    """The second case: it exists, Live refuses it, and that answers a real question."""
    report = run(workspace)
    finding = verdict_of(report, "track.raiser")
    assert finding.verdict == sc.REFUSED
    assert finding.new_status is PathStatus.BROKEN
    assert has_measurement_note(finding.note), finding.note
    assert finding not in report.deletions


def test_index_out_of_range_neither_deletes_the_row_nor_marks_it_broken(
    workspace: Path,
) -> None:
    """The third outcome, asserted hard: the rule that is easiest to get wrong.

    ``song.scenes[0]`` answers ``index_out_of_range``: the open set has no scenes.
    That is a fact about *this set* and none whatever about ``scene.name``, and it
    is the branch that plants false facts when it is got wrong, because both wrong
    answers look reasonable from inside the code:

    * deleting the row throws away a real capability on the evidence of a
      missing fixture, and the deletion is the one change nothing undoes;
    * marking it ``broken`` writes a false statement about the Live Object Model
      into a file that outlives the session that wrote it, and a ``broken`` row is
      read as settled, so the next person does not look.

    ``docs/catalog.md``, "Three outcomes, not two": the third case is the common
    one. So every channel through which this row could change is checked to be
    silent: the verdict, the proposed status, the deletion list, the status-change
    list, and last the file on disk after a full ``--write``, because a report that
    is right and a writer that is wrong still costs the catalog.
    """
    catalog = workspace / "catalog"
    report = run(workspace)
    finding = verdict_of(report, "scene.name")

    assert finding.verdict == sc.NOT_REACHED
    assert finding.new_status is None, "an empty scene list is not a fact about the row"
    assert "index_out_of_range" in finding.reason
    assert "scene.name" not in {f.spec_id for f in report.deletions}
    assert "scene.name" not in {f.spec_id for f in report.status_changes}

    out = io.StringIO()
    sc.print_report(report, out, verbose=True)
    _, _, not_reached = out.getvalue().partition("NOT REACHED")
    assert "scene.name" in not_reached, "silently untouched is not the same as reported"

    sc.CatalogSync(catalog).apply(
        report, sc.WritePlan(add=True, delete=True, status=True)
    )
    row = rows_on_disk(catalog / "10-test.yaml")["scene.name"]
    assert row["status"] == "untested"
    assert row["doc"].strip() == "Scene name.", "the row's doc gained a note it did not earn"


def test_no_such_path_on_a_derived_probe_is_not_a_deletion(workspace: Path) -> None:
    """A Compressor without ``macro_count`` says nothing about the rack rows.

    Two specimens share ``song.tracks[0].devices[]`` and neither of them is what
    this row means. Deleting on that evidence would remove a real capability from
    the catalog for good, so the finding stops at *not reached* and says which
    specimen would settle it.
    """
    report = run(workspace)
    finding = verdict_of(report, "rack.macro_count")
    assert finding.verdict == sc.NOT_REACHED
    assert "rack.macro_count" not in {f.spec_id for f in report.deletions}
    assert "derived" in finding.detail
    assert "--specimens" in finding.detail


def test_a_container_that_does_not_resolve_teaches_nothing_either(
    workspace: Path,
) -> None:
    """``devices[0].sample`` is a Compressor's missing attribute, not a missing row.

    The *container* answered ``no_such_path``, which is a fact about the device
    that happened to sit at index 0. Every ``simpler.*`` row underneath it stays
    exactly as it was.
    """
    report = run(workspace)
    finding = verdict_of(report, "simpler.sample_length")
    assert finding.verdict == sc.NOT_REACHED
    assert finding.new_status is None
    assert "simpler.sample_length" not in {f.spec_id for f in report.deletions}


def test_the_scripts_track_guard_is_not_lives_answer(workspace: Path) -> None:
    """``arm`` on an unarmable track is missing from the track, not from the class."""
    report = run(workspace)
    finding = verdict_of(report, "track.arm")
    assert finding.verdict == sc.NOT_REACHED
    assert "track guard" in finding.reason


def test_a_truncated_describe_is_never_evidence(workspace: Path) -> None:
    """A budget that ran out reported less than it saw; absence proves nothing."""
    recorded = replay()
    recorded["describe"]["song.tracks[0]"]["result"]["truncated"] = True
    recorded["describe"]["song.tracks[0]"]["result"]["budget_left"] = 0
    report = run(workspace, recorded=recorded)
    assert verdict_of(report, "track.ghost").verdict == sc.NOT_REACHED
    assert report.deletions == []


def test_a_call_row_whose_method_is_there_is_confirmed(workspace: Path) -> None:
    report = run(workspace)
    finding = verdict_of(report, "track.stop_all")
    assert finding.verdict == sc.CONFIRMED
    assert "allowlist" in finding.detail


# --------------------------------------------------------------------------------
# Proposals
# --------------------------------------------------------------------------------


def test_a_property_live_has_and_the_catalog_lacks_becomes_a_row(workspace: Path) -> None:
    report = run(workspace)
    proposed = {row.spec_id: row for row in report.new_rows}
    assert "track.color_index" in proposed
    row = proposed["track.color_index"]
    assert row.path == "song.tracks[{track}].color_index"
    assert row.access == ["get"]
    assert row.kind is Kind.INT
    assert row.status is PathStatus.VERIFIED
    assert [p.name for p in row.params] == ["track"]


def test_a_collection_child_becomes_a_list_row(workspace: Path) -> None:
    proposed = {row.spec_id: row for row in run(workspace).new_rows}
    assert proposed["track.devices"].kind is Kind.LIST
    assert proposed["track.devices"].unit is Unit.NONE


def test_an_allowlisted_method_becomes_an_untested_call_row(workspace: Path) -> None:
    """It was found, not called, so it may not carry a measurement note."""
    proposed = {row.spec_id: row for row in run(workspace).new_rows}
    row = proposed["track.delete_device"]
    assert row.access == ["call"]
    assert row.method == "delete_device"
    assert row.status is PathStatus.UNTESTED
    assert not has_measurement_note(row.doc), row.doc


def test_a_method_the_allowlist_refuses_is_a_finding_not_a_row(workspace: Path) -> None:
    """The script is the authority; widening it costs a Live restart (protocol §6).

    ``Track.secret_method`` exists in the running Live and the allowlist does not
    admit it. A catalog row for it would be a request the script refuses, so it is
    reported in its own list instead, and the report has to say what the decision
    costs, because that is the whole reason it is a decision and not a diff.
    """
    report = run(workspace)
    assert "track.secret_method" not in {row.spec_id for row in report.new_rows}
    assert "secret_method" not in {row.member for row in report.new_rows}
    assert ("Track", "secret_method") in {
        (gap.live_class, gap.method) for gap in report.allowlist_gaps
    }

    out = io.StringIO()
    sc.print_report(report, out, verbose=False)
    text = out.getvalue()
    heading, _, rest = text.partition("REACHABLE IF THE ALLOWLIST WERE WIDENED")
    assert rest, "the refused method has no section of its own in the report"
    assert "Track.secret_method()" in rest
    assert "restart" in rest, "widening the allowlist costs a Live restart; say so"
    assert "Track.secret_method" not in heading, "it must not appear as a proposed row"


def test_an_area_filter_narrows_what_is_judged_and_not_what_counts_as_new(
    workspace: Path,
) -> None:
    """``--area`` picks the rows to judge. It may not make an existing row look new.

    The right-hand side of "is this member already in the catalog?" is the *whole*
    catalog, never the ``--area`` subset. Filter both sides and the first
    ``--area track`` run proposes every device row over again, under a track id,
    and the append fails on the duplicate, or worse, does not.
    """
    registry = Registry.load(workspace / "catalog")
    rows = sc.select_rows(registry, ["track"])
    assert {spec.id for spec in rows} == {
        "track.name",
        "track.ghost",
        "track.raiser",
        "track.arm",
        "track.stop_all",
    }

    report = sc.Reconciler(
        registry,
        sc.Reader(None, replay=replay(), pace=0.0),
        {name: sc.Specimen(name, path) for name, path in SPECIMENS.items()},
        allowlist=sc._literal_string_set(workspace / "script.py", "METHOD_ALLOWLIST"),
        describe_skip=sc._literal_string_set(workspace / "script.py", "_DESCRIBE_SKIP"),
        catalog_root=workspace / "catalog",
        rows=rows,
    ).run()

    assert {f.spec_id for f in report.findings} == {spec.id for spec in rows}
    # `name` on a device is catalogued as `device.name`, in an area this run is not
    # looking at. It is still catalogued, so it is still not a discovery.
    assert "name" not in {row.member for row in report.new_rows}

    with pytest.raises(ValueError, match="no catalog area named"):
        sc.select_rows(registry, ["mixer"])


def test_an_attribute_lom_describe_skips_is_never_proposed(workspace: Path) -> None:
    """``canonical_parent`` walks back up the graph; a row for it is a trap."""
    report = run(workspace)
    assert "track.canonical_parent" not in {row.spec_id for row in report.new_rows}


def test_every_proposed_row_satisfies_the_catalog_schema_tests(workspace: Path) -> None:
    """The pairing ``tests/test_catalog.py`` enforces, checked before it is written."""
    for row in run(workspace).new_rows:
        assert ID_PATTERN.match(row.spec_id), row.spec_id
        assert row.doc.strip()
        note = has_measurement_note(row.doc)
        if row.status is PathStatus.UNTESTED:
            assert not note, f"{row.spec_id}: an untested row may not read as measured"
        else:
            assert note, f"{row.spec_id}: {row.status.value} without a measurement note"


#: A catalog whose two containers sit in the same area. ``Device.color`` and
#: ``Chain.color`` both want the id ``device.color``, and two rows with one id do
#: not load: a collision that only shows up in the re-parse *after* the file has
#: been written, where it takes every other discovery in the run with it.
COLLIDING_CATALOG = """\
- id: device.name
  path: song.tracks[{track}].devices[{device}].name
  access: [get]
  kind: str
  unit: none
  destructive: false
  verify: read_back
  status: untested
  doc: "Device name."
  params:
    - {name: track, kind: int, required: true}
    - {name: device, kind: int, required: true}

- id: device.chain_solo
  path: song.tracks[{track}].devices[{device}].chains[{chain}].solo
  access: [get]
  kind: bool
  unit: none
  destructive: false
  verify: read_back
  status: untested
  doc: "Solo on one chain of a rack."
  params:
    - {name: track, kind: int, required: true}
    - {name: device, kind: int, required: true}
    - {name: chain, kind: int, required: true}
"""


def test_two_containers_in_one_area_never_claim_the_same_id(tmp_path: Path) -> None:
    """Two discoveries in one run may not be given one id.

    The area is the id's first component and it comes from the exemplar row, so a
    device and a chain of that device both file under ``device``. Live reports
    ``color`` on each. Asking only the *catalog* whether ``device.color`` is taken
    answers "no" twice, and the duplicate is caught by ``Registry.load`` in the
    re-parse after the write: at which point the file is rolled back and every
    other row the run found is lost with it. So the run's own proposals count as
    taken, and the second one falls back to the qualified form.
    """
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "40-device.yaml").write_text(COLLIDING_CATALOG, encoding="utf-8", newline="\n")
    (tmp_path / "script.py").write_text(SCRIPT, encoding="utf-8", newline="\n")

    colour = {"name": "color", "type": "int", "settable": True, "value": 3}
    recorded = {
        "script_info": {"live_version": "12.4.5", "script_version": "0.4.0"},
        "describe": {
            "song.tracks[0].devices[0]": _describe(
                "song.tracks[0].devices[0]",
                "Compressor",
                [{"name": "name", "type": "string", "settable": False, "value": "Comp"}, colour],
            ),
            "song.tracks[0].devices[0].chains[0]": _describe(
                "song.tracks[0].devices[0].chains[0]",
                "Chain",
                [{"name": "solo", "type": "bool", "settable": True, "value": False}, colour],
            ),
        },
        "get": {},
    }
    report = sc.Reconciler(
        Registry.load(catalog),
        sc.Reader(None, replay=recorded, pace=0.0),
        {
            "Device": sc.Specimen("Device", "song.tracks[0].devices[0]"),
            "Chain": sc.Specimen("Chain", "song.tracks[0].devices[0].chains[0]"),
        },
        allowlist=frozenset(),
        describe_skip=frozenset({"canonical_parent"}),
        catalog_root=catalog,
    ).run()

    proposed = [row.spec_id for row in report.new_rows]
    assert len(proposed) == len(set(proposed)), f"two proposals share an id: {proposed}"
    assert set(proposed) == {"device.color", "device.chains_color"}

    sc.CatalogSync(catalog).apply(report, sc.WritePlan(add=True))
    registry = Registry.load(catalog)
    assert registry.get("device.color").path == "song.tracks[{track}].devices[{device}].color"
    assert (
        registry.get("device.chains_color").path
        == "song.tracks[{track}].devices[{device}].chains[{chain}].color"
    )


# --------------------------------------------------------------------------------
# A run that cannot name a Live version
# --------------------------------------------------------------------------------


def versionless() -> dict[str, Any]:
    """The recording, from a script whose handshake reports no ``live_version``."""
    recorded = replay()
    recorded["script_info"] = {
        "script_version": "0.4.0",
        "handlers": ["lom_get", "lom_describe"],
    }
    return recorded


def test_a_run_that_cannot_name_a_live_version_flips_no_status(workspace: Path) -> None:
    """No version, no measurement, and therefore no status.

    A note has to say which build it was true of; ``tests/test_catalog.py`` checks
    for a date *and* a ``Live <version>`` before it will accept ``verified`` or
    ``broken``. A handshake that reports neither leaves the note with nothing to
    name, so the reconciliation still runs and still reports what Live answered,
    it simply may not write a status, and a property it confirmed arrives
    ``untested`` with a doc that does not pretend otherwise.

    Deletions are the exception and stay: "this attribute does not exist" needs no
    version to be true, and the removed row leaves no status behind to pair.
    """
    report = run(workspace, recorded=versionless())

    assert any("Live version" in warning for warning in report.warnings), report.warnings
    assert report.status_changes == []
    assert verdict_of(report, "track.name").verdict == sc.CONFIRMED
    assert verdict_of(report, "track.name").new_status is None
    assert verdict_of(report, "track.raiser").verdict == sc.REFUSED
    assert verdict_of(report, "track.raiser").new_status is None
    assert [f.spec_id for f in report.deletions] == ["track.ghost"]

    for row in report.new_rows:
        assert row.status is PathStatus.UNTESTED, row.spec_id
        assert not has_measurement_note(row.doc), f"{row.spec_id}: untested, reads as measured"


def test_a_versionless_write_leaves_a_catalog_that_still_passes_its_own_tests(
    workspace: Path, cli: Callable[..., int]
) -> None:
    """The same rule, checked on the bytes: nothing claims a status it cannot date."""
    assert cli("--write", recorded=versionless()) == sc.CODE_OK
    rows = rows_on_disk(workspace / "catalog" / "10-test.yaml")
    for spec_id, row in rows.items():
        assert row["status"] == PathStatus.UNTESTED.value, f"{spec_id} was flipped anyway"
        assert not has_measurement_note(str(row.get("doc") or "")), spec_id


# --------------------------------------------------------------------------------
# Reading the Remote Script's own rules
# --------------------------------------------------------------------------------


def test_the_allowlist_is_read_from_the_script_and_never_copied(workspace: Path) -> None:
    allowlist = sc._literal_string_set(workspace / "script.py", "METHOD_ALLOWLIST")
    assert allowlist == frozenset({"Track.stop_all_clips", "Track.delete_device"})


def test_the_real_remote_script_still_parses() -> None:
    """The shipped script is the authority, and it is read, not imported.

  It cannot be imported outside Live (it pulls in ``_Framework``) so this is
    the check that the ``ast`` route still finds both constants after an edit.
    """
    script = _REPO_ROOT / "live-remote-script" / "__init__.py"
    allowlist = sc._literal_string_set(script, "METHOD_ALLOWLIST")
    assert "Song.undo" in allowlist
    assert sc._literal_string_set(script, "_DESCRIBE_SKIP")


def test_a_missing_constant_is_named_rather_than_guessed(tmp_path: Path) -> None:
    empty = tmp_path / "empty.py"
    empty.write_text("X = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="METHOD_ALLOWLIST"):
        sc._literal_string_set(empty, "METHOD_ALLOWLIST")


# --------------------------------------------------------------------------------
# Writing the catalog
# --------------------------------------------------------------------------------


def test_write_applies_every_class_of_change_and_the_file_still_loads(
    workspace: Path,
) -> None:
    """One pass, all three classes of change, and the result is still a catalog."""
    catalog = workspace / "catalog"
    report = run(workspace)
    changed = sc.CatalogSync(catalog).apply(
        report, sc.WritePlan(add=True, delete=True, status=True)
    )
    assert changed
    text = (catalog / "10-test.yaml").read_text(encoding="utf-8")
    assert "track.ghost" not in text
    assert "track.color_index" in text
    assert "This comment must survive" in text
    assert text.count("# Track") == 1
    # The whole point of the atomic re-parse: a bad edit is not a catalog.
    Registry.load(catalog)


def test_the_cli_write_leaves_a_catalog_that_loads_and_says_what_was_learned(
    workspace: Path, cli: Callable[..., int]
) -> None:
    """``--write`` end to end: the gate, the edit, the re-parse, the result.

    ``Registry.load`` is the assertion that matters. Everything this tool does is a
    line-level edit of hand-written YAML, so "it wrote something" and "it wrote a
    catalog" are different claims and only the second one counts.
    """
    catalog = workspace / "catalog"
    assert cli("--write") == sc.CODE_OK

    registry = Registry.load(catalog)
    ids = {spec.id for spec in registry.all()}
    assert "track.ghost" not in ids, "the fantasy row survived --write"
    assert registry.get("track.name").status is PathStatus.VERIFIED
    assert registry.get("track.raiser").status is PathStatus.BROKEN
    assert registry.get("scene.name").status is PathStatus.UNTESTED
    assert registry.get("track.color_index").kind is Kind.INT
    assert registry.get("track.delete_device").method == "delete_device"


def test_the_written_catalog_satisfies_the_pairing_test_catalog_enforces(
    workspace: Path, cli: Callable[..., int]
) -> None:
    """A row this tool marks ``verified`` carries a dated note: in both directions.

    ``tests/test_catalog.py`` fails a ``verified`` or ``broken`` row whose ``doc``
    records no measurement, *and* an ``untested`` row whose ``doc`` reads like one.
    Both halves are checked here on the bytes sync_catalog wrote, so a run that
    breaks the catalog's own test suite fails here first, next to the code that
    caused it.
    """
    assert cli("--write") == sc.CODE_OK
    rows = rows_on_disk(workspace / "catalog" / "10-test.yaml")
    assert rows, "the file lost every row"
    for spec_id, row in rows.items():
        doc = str(row.get("doc") or "")
        if row["status"] == PathStatus.UNTESTED.value:
            assert not has_measurement_note(doc), f"{spec_id}: untested, but reads as measured"
        else:
            assert has_measurement_note(doc), f"{spec_id}: {row['status']} with no receipt"
            assert "12.4.5" in doc, f"{spec_id}: the note does not name the build"


def test_comments_and_prose_survive_a_write(
    workspace: Path, cli: Callable[..., int]
) -> None:
    """The catalog is 22 000 lines of hand-written prose; a sync may not reflow it.

    This is why the writer is a line editor and not ``yaml.safe_load`` followed by
    ``yaml.dump``: the round trip is correct YAML and an unreviewable diff with
    every comment deleted. Each of the fixture's comment lines stands somewhere
    different: above the first row, above a section, above a section whose first
    row gets deleted, and all three have to come out the other side.
    """
    catalog = workspace / "catalog"
    before = (catalog / "10-test.yaml").read_text(encoding="utf-8")
    comments = [line for line in before.splitlines() if line.lstrip().startswith("#")]
    assert len(comments) >= 5, "the fixture stopped exercising this"

    assert cli("--write") == sc.CODE_OK
    after = (catalog / "10-test.yaml").read_text(encoding="utf-8")
    for comment in comments:
        assert comment in after, f"a write deleted the comment {comment!r}"
    assert "# This comment must survive every edit this file ever receives." in after
    assert "Track name, a hypothesis until somebody reads it." in after, "prose was reflowed away"


def test_a_second_run_over_its_own_output_proposes_nothing(workspace: Path) -> None:
    """Idempotence. A tool that appends a note every run is a tool nobody runs twice."""
    catalog = workspace / "catalog"
    first = run(workspace)
    sc.CatalogSync(catalog).apply(first, sc.WritePlan(add=True, delete=True, status=True))
    second = run(workspace)
    assert second.new_rows == []
    assert second.deletions == []
    assert second.status_changes == []


def test_a_plan_only_applies_what_it_names(workspace: Path) -> None:
    catalog = workspace / "catalog"
    report = run(workspace)
    sc.CatalogSync(catalog).apply(report, sc.WritePlan(status=True))
    text = (catalog / "10-test.yaml").read_text(encoding="utf-8")
    assert "track.ghost" in text            # --delete was not given
    assert "track.color_index" not in text  # --add was not given
    assert "status: verified" in text       # --status was


def test_crlf_line_endings_survive_an_edit_and_an_append(workspace: Path) -> None:
    """A CRLF catalog rewritten as LF turns a two-line edit into a whole-file diff."""
    catalog = workspace / "catalog"
    crlf = catalog / "20-crlf.yaml"
    crlf.write_text(
        "- id: song.tempo\n"
        "  path: song.tempo\n"
        "  access: [get]\n"
        "  kind: float\n"
        "  unit: none\n"
        "  destructive: false\n"
        "  verify: read_back\n"
        "  status: untested\n"
        '  doc: "Tempo in BPM."\n',
        encoding="utf-8",
        newline="\r\n",
    )
    recorded = replay()
    recorded["describe"]["song"] = _describe(
        "song",
        "Song",
        [{"name": "tempo", "type": "float", "settable": True, "value": 124.0}],
        methods=["undo"],
    )
    report = run(workspace, recorded=recorded)
    sc.CatalogSync(catalog).apply(report, sc.WritePlan(add=True, delete=True, status=True))

    raw = crlf.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n"), "the file gained an LF-only line"
    assert b"status: verified" in raw
    Registry.load(catalog)


def test_a_deletion_never_swallows_the_comment_above_it(workspace: Path) -> None:
    """A section header belongs to the section, not to the row underneath it."""
    catalog = workspace / "catalog"
    report = run(workspace)
    sc.CatalogSync(catalog).apply(report, sc.WritePlan(delete=True))
    text = (catalog / "10-test.yaml").read_text(encoding="utf-8")
    assert "# Devices - one shape, several Live classes" in text
    assert "\n\n\n" not in text, "a deletion left a double blank line behind"


def test_the_limits_block_records_every_deletion(workspace: Path) -> None:
    """The knowledge outlives the row: docs/limits.md §7 is where it goes."""
    report = run(workspace)
    block = sc.limits_block(report)
    assert "`track.ghost`" in block
    assert "Track" in block
    assert "2026-" in block or "20" in block  # the stamp carries a date
    assert block.startswith("### Removed by scripts/sync_catalog.py")


def test_nothing_is_written_without_a_plan(workspace: Path) -> None:
    catalog = workspace / "catalog"
    before = (catalog / "10-test.yaml").read_text(encoding="utf-8")
    sc.CatalogSync(catalog).apply(run(workspace), sc.WritePlan())
    assert (catalog / "10-test.yaml").read_text(encoding="utf-8") == before


# --------------------------------------------------------------------------------
# The CLI
# --------------------------------------------------------------------------------


def test_report_and_write_contradict_each_other(
    workspace: Path, cli: Callable[..., int]
) -> None:
    """Refusing to guess is the whole posture: one of the two has to go.

    ``--report`` promises to change nothing and ``--write`` promises to apply
    everything. Picking one silently would make the flag that was ignored mean
    whatever this tool felt like, on the one command where the user was explicit.
    """
    before = catalog_bytes(workspace / "catalog")
    assert cli("--report", "--write") == sc.CODE_USAGE
    assert catalog_bytes(workspace / "catalog") == before


@pytest.mark.parametrize("flags", [(), ("--report",), ("--verbose",), ("--json",)])
def test_reporting_changes_no_file_at_all(
    workspace: Path, cli: Callable[..., int], flags: tuple[str, ...]
) -> None:
    """Reporting is a read. Not "nearly unchanged": byte for byte unchanged.

    Compared as bytes, across every file in the directory, one of them with
    CRLF endings: a run that rewrote a file identically apart from its line
    endings, or that added a trailing newline, would pass a text comparison and
    still produce a whole-file diff for somebody to review.
    """
    catalog = workspace / "catalog"
    (catalog / "20-crlf.yaml").write_text(CRLF_FILE, encoding="utf-8", newline="\r\n")
    before = catalog_bytes(catalog)
    assert any(b"\r\n" in data for data in before.values())

    assert cli(*flags) == sc.CODE_OK
    assert catalog_bytes(catalog) == before
