"""Schema discipline for ``src/ableton_maestro/catalog/*.yaml``.

These are the schema tests for the catalog whose format ``docs/catalog.md`` defines.
The rules are enforced against the raw YAML, not against what
:class:`~ableton_maestro.registry.Registry` happened to make of it. That is
deliberate: the loader already rejects most of this, and a test that only asked
the loader would pass just as happily on the day somebody loosens the loader.
The data is the artefact under test, so the data is what gets read.

Nothing here opens a socket, imports Live, or needs Live installed. The catalog
is a text file; checking it is a text-file job.

Two conventions used throughout:

* Every offence is collected, then reported together. A catalog edit that
  breaks four rows should say so once, not four times across four runs.
* Failure messages explain the rule, not just the breach. These rows are
  hand-written and the message is the only feedback their author gets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml

from ableton_maestro.lom import paths as lom_paths
from ableton_maestro.models import Access, Kind, PathStatus, Unit
from ableton_maestro.registry import DEFAULT_CATALOG_DIR, ROW_KEYS, CatalogError, Registry
from ableton_maestro.spec import (
    MEANS_CAP,
    build_path,
    name_placeholders_in,
    placeholders_in,
    validate_path,
)

# --------------------------------------------------------------------------------
# Loading the catalog as data
# --------------------------------------------------------------------------------

CATALOG_DIR: Path = DEFAULT_CATALOG_DIR

#: How to legitimately move a row out of ``untested``. Repeated in several failure
#: messages because the whole point of the status column is that there is exactly
#: one way to change it.
HOW_TO_VERIFY = (
    "A status is flipped by measuring, never by editing:\n"
    "    python scripts/probe_paths.py --id <row id> --go --write-back\n"
    "against a throwaway set, with Live running. The probe writes back both the "
    "status and a sentence in `doc` saying what it actually did."
)


@dataclass(frozen=True)
class Row:
    """One catalog row, with the file and position it came from.

    The provenance travels with the row because every failure message in this
    file needs it: "a row is missing a doc" is not actionable, "30-clip.yaml row
    41 (clip.warping)" is.
    """

    file: str
    index: int
    data: dict[str, Any]

    @property
    def id(self) -> str:
        """The row's ``id``, or a stand-in that still locates it."""
        raw = self.data.get("id")
        return raw if isinstance(raw, str) and raw else f"<no id, {self.file} row {self.index}>"

    @property
    def where(self) -> str:
        """Human-readable position, for a failure message."""
        return f"{self.file} row {self.index} ({self.id})"

    def get(self, key: str, default: Any = None) -> Any:
        """Read one field of the raw row."""
        return self.data.get(key, default)


def catalog_files() -> list[Path]:
    """Every catalog file, in the order the registry concatenates them."""
    return sorted(CATALOG_DIR.glob("*.yaml"), key=lambda p: p.name)


def _load_rows() -> list[Row]:
    """Parse every catalog file into :class:`Row` objects.

    Parsing failures are left to :func:`test_every_catalog_file_parses`; a file
    that will not parse contributes no rows rather than taking every other test
    down with it, so one broken file yields one clear failure instead of thirty.
    """
    rows: list[Row] = []
    for file in catalog_files():
        try:
            raw = yaml.safe_load(file.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        if not isinstance(raw, list):
            continue
        for index, entry in enumerate(raw):
            if isinstance(entry, dict):
                rows.append(Row(file=file.name, index=index, data=entry))
    return rows


#: Loaded once at import: 1164 rows across five files, re-read by every test below
#: would be waste, and none of these tests mutate them.
ROWS: list[Row] = _load_rows()

VALID_ACCESS: frozenset[str] = frozenset(a.value for a in Access)
VALID_KIND: frozenset[str] = frozenset(k.value for k in Kind)
VALID_UNIT: frozenset[str] = frozenset(u.value for u in Unit)
VALID_STATUS: frozenset[str] = frozenset(s.value for s in PathStatus)

#: ``area.name``, lowercase, underscores allowed. A few rows carry a third
#: component (``song.groove_pool.grooves``); the area is still everything before
#: the first dot, which is what ``registry.area_of`` reads.
ID_PATTERN = re.compile(r"\A[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+\Z")

#: The three halves of a measurement note. All three must be present: a bare date
#: could be anything, and "measured" on its own is the claim without the receipt.
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_PROVENANCE = re.compile(r"\b(verified|measured|probed)\b", re.IGNORECASE)
_LIVE_VERSION = re.compile(r"\bLive\s*\d", re.IGNORECASE)

#: A verdict on the whole row, passed before anyone probed it. Each of these is
#: subjectless by construction -- it says the row is unverified, not what about it
#: is unverified -- which is exactly what a measurement further down contradicts.
#: A scoped sentence ("what an unknown name does is not measured") matches none of
#: them and is meant to survive.
_BLANKET_HEDGE = re.compile(
    r"expected from the LOM"
    r"|[;,]\s*unverified\b"
    r"|(?:^|\.\s+)unverified\b"
    r"|UNVERIFIED\s*[-–—]",
    re.IGNORECASE,
)


def has_measurement_note(doc: str) -> bool:
    """True when ``doc`` records a measurement: a date, a Live version, a verb.

  ``docs/catalog.md`` ("`verified` does not say *which* access) the `doc`
    does". The status column is coarse on purpose and the note is where the
    precision lives, so the note is what gets checked.
    """
    return bool(_ISO_DATE.search(doc) and _PROVENANCE.search(doc) and _LIVE_VERSION.search(doc))


def fail(headline: str, offences: list[str], *, explain: str = "") -> None:
    """Fail with every offence listed under one headline, or return quietly."""
    if not offences:
        return
    body = "\n".join(f"  - {line}" for line in offences)
    tail = f"\n\n{explain}" if explain else ""
    pytest.fail(f"{headline} ({len(offences)}):\n{body}{tail}", pytrace=False)


# --------------------------------------------------------------------------------
# The files themselves
# --------------------------------------------------------------------------------


def test_catalog_directory_holds_yaml_files() -> None:
    """The catalog exists and is a directory of files, not one file.

    ``docs/catalog.md``: splitting by LOM area is what lets several authors add
    rows without meeting in a diff. An empty directory here means the package
    was built without its data, which imports fine and then knows nothing.
    """
    assert CATALOG_DIR.is_dir(), f"catalog directory missing: {CATALOG_DIR}"
    files = catalog_files()
    assert files, f"no *.yaml files in {CATALOG_DIR}"


@pytest.mark.parametrize("file", catalog_files(), ids=lambda p: p.name)
def test_every_catalog_file_parses(file: Path) -> None:
    """Each file is valid YAML whose top level is a list of mappings."""
    try:
        raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - only on a broken edit
        pytest.fail(f"{file.name}: not valid YAML: {exc}", pytrace=False)

    assert isinstance(raw, list), (
        f"{file.name}: top level must be a list of catalog rows, "
        f"got {type(raw).__name__}. A mapping at the top level would load as one row."
    )
    assert raw, f"{file.name}: holds no rows"
    non_mappings = [i for i, entry in enumerate(raw) if not isinstance(entry, dict)]
    assert not non_mappings, f"{file.name}: rows {non_mappings} are not mappings"


def test_rows_were_loaded() -> None:
    """Guard against a silently empty ``ROWS`` making every other test vacuous."""
    assert len(ROWS) > 100, f"only {len(ROWS)} catalog rows parsed; something is wrong upstream"


# --------------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------------


def test_ids_are_unique_across_files() -> None:
    """No id appears twice, and a collision names both files.

    Ids are unique across the directory, not merely within a file
    (``docs/catalog.md``). A duplicate that silently shadowed an earlier row is
    the quiet failure this project exists to avoid, and a message naming only one
    of the two files sends the reader looking in the wrong place.
    """
    seen: dict[str, str] = {}
    offences: list[str] = []
    for row in ROWS:
        spec_id = row.data.get("id")
        if not isinstance(spec_id, str) or not spec_id:
            continue
        first = seen.get(spec_id)
        if first is not None:
            offences.append(f"{spec_id!r}: defined in {first} and again in {row.file}")
        else:
            seen[spec_id] = f"{row.file} row {row.index}"
    fail("duplicate catalog id(s)", offences)


def test_every_row_has_a_string_id_shaped_as_area_dot_name() -> None:
    """Every row carries an ``area.name`` id in lowercase.

    The area is the id prefix before the first dot and is how the registry groups
    rows (``registry.area_of``). It is derived rather than stored, so a
    mis-shaped id does not fail loudly: it quietly invents an area of one.
    """
    offences = [
        f"{row.where}: id {row.data.get('id')!r} is not `area.name` in lowercase"
        for row in ROWS
        if not (isinstance(row.data.get("id"), str) and ID_PATTERN.match(row.data["id"]))
    ]
    fail("mis-shaped catalog id(s)", offences)


def test_no_row_uses_an_unknown_key() -> None:
    """A row may only use keys :class:`~ableton_maestro.spec.PathSpec` can hold.

    A misspelt key is the archetypal silent loss: ``desctructive: true`` reads as
    a careful author and arms nothing.
    """
    offences = [
        f"{row.where}: unknown key(s) {sorted(k for k in row.data if k not in ROW_KEYS)}"
        for row in ROWS
        if any(k not in ROW_KEYS for k in row.data)
    ]
    fail(
        "unknown catalog key(s)",
        offences,
        explain=f"A row may use: {sorted(ROW_KEYS)}",
    )


# --------------------------------------------------------------------------------
# Placeholders and parameters
# --------------------------------------------------------------------------------


def _param_names(row: Row) -> list[str]:
    """The declared parameter names of a row, skipping malformed entries."""
    params = row.get("params") or []
    if not isinstance(params, list):
        return []
    return [p["name"] for p in params if isinstance(p, dict) and isinstance(p.get("name"), str)]


def test_every_placeholder_in_a_path_has_a_param() -> None:
    """``song.tracks[{track}]`` must declare a ``track`` parameter.

    Without one, :func:`~ableton_maestro.spec.build_path` has nothing to
    substitute and the brace reaches Live as a ``bad_path``, several layers away
    from the row that caused it.
    """
    offences: list[str] = []
    for row in ROWS:
        path = row.get("path")
        if not isinstance(path, str):
            continue
        declared = set(_param_names(row))
        missing = sorted(set(placeholders_in(path)) - declared)
        if missing:
            offences.append(f"{row.where}: path uses {missing} which no param declares")
    fail("placeholder(s) with no matching param", offences)


def test_every_param_appears_in_its_path() -> None:
    """And the other direction: a declared parameter must land somewhere.

    A parameter that names no placeholder is an argument the caller supplies and
    nobody reads: the failure class docs/architecture.md, 'read-back as a principle' is written
    against. On a
    ``call`` row the usual cause is a method argument filed under ``params``
    instead of ``args``.
    """
    offences: list[str] = []
    for row in ROWS:
        path = row.get("path")
        if not isinstance(path, str):
            continue
        in_path = set(placeholders_in(path))
        unused = sorted(set(_param_names(row)) - in_path)
        if unused:
            is_call = Access.CALL.value in (row.get("access") or [])
            hint = " (method arguments belong in `args`)" if is_call else ""
            offences.append(f"{row.where}: param(s) {unused} appear nowhere in {path!r}{hint}")
    fail("declared param(s) that land nowhere", offences)


def test_param_names_are_unique_within_a_row() -> None:
    """Two parameters of one name would fill one placeholder twice."""
    offences: list[str] = []
    for row in ROWS:
        names = _param_names(row)
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            offences.append(f"{row.where}: duplicate param name(s) {duplicates}")
    fail("duplicate param name(s)", offences)


def test_index_and_name_placeholders_declare_the_right_kind() -> None:
    """The two placeholder positions are validated differently and never mix.

    An index placeholder (``tracks[{track}]``) must be ``kind: int``; a segment
    name placeholder (``browser.{root}``) must be ``str`` or ``enum``, because the
    substituted value becomes an attribute name (``docs/protocol.md`` §6).
    """
    offences: list[str] = []
    for row in ROWS:
        path = row.get("path")
        params = row.get("params") or []
        if not isinstance(path, str) or not isinstance(params, list):
            continue
        by_name = name_placeholders_in(path)
        for param in params:
            if not isinstance(param, dict):
                continue
            name, kind = param.get("name"), param.get("kind")
            if name in by_name:
                if kind not in ("str", "enum"):
                    offences.append(
                        f"{row.where}: {name!r} names a path segment but is kind {kind!r}; "
                        "it must be 'str' or 'enum'"
                    )
            elif kind != "int":
                offences.append(
                    f"{row.where}: {name!r} stands for an index but is kind {kind!r}; "
                    "an index is always a whole number"
                )
    fail("placeholder kind mismatch(es)", offences)


# --------------------------------------------------------------------------------
# doc, and the status discipline
# --------------------------------------------------------------------------------


def test_every_row_has_a_non_empty_doc() -> None:
    """``doc`` is required and is load-bearing, not decoration.

    The catalog reaches the model at runtime as the ``ableton://catalog``
    resource; ``doc`` is what stops it guessing what ``0.5`` means
    (``docs/catalog.md``).
    """
    offences = [
        f"{row.where}: doc is missing or empty"
        for row in ROWS
        if not str(row.get("doc") or "").strip()
    ]
    fail("row(s) with no doc", offences)


def test_status_is_one_of_the_three() -> None:
    """``verified`` / ``broken`` / ``untested`` and nothing else.

    ``status: unknown`` or ``status: partial`` would load as a new fourth state
    that no reader has a rule for.
    """
    offences = [
        f"{row.where}: status {row.get('status')!r}"
        for row in ROWS
        if row.get("status") not in VALID_STATUS
    ]
    fail(
        "invalid status value(s)",
        offences,
        explain=f"Valid statuses: {sorted(VALID_STATUS)}. The starting value is 'untested'.",
    )


def test_no_row_claims_verified_without_a_measurement_note() -> None:
    """A ``verified`` row must carry the measurement that earned it.

    This is the project's core discipline and the one test worth reading twice.
    ``verified`` means *someone ran it against real Live and watched it work*, not
    that it looks right, not that the LOM documentation says so, not that a model
    was confident. Editing ``status: verified`` by hand converts a hypothesis into
    a fact without doing the work, and a false fact outlives the session that
    planted it.

    The status alone is coarse: a row may carry ``access: [get, set]`` and a read
    does not prove the write. So ``docs/catalog.md`` puts the precision in the
    note: every measurement appends a sentence to ``doc`` naming what was
    actually done, the date, and the Live version. That sentence is what this
    test looks for, because it is the part a hand edit forgets.
    """
    offences = [
        f"{row.where}: status is 'verified' but doc records no measurement"
        for row in ROWS
        if row.get("status") == PathStatus.VERIFIED.value
        and not has_measurement_note(str(row.get("doc") or ""))
    ]
    fail(
        "row(s) claiming 'verified' with nothing to show for it",
        offences,
        explain=(
            "'verified' means: someone ran it against real Live and watched it work.\n"
            "The doc must say so, in a sentence carrying a date and the Live version, "
            "e.g.\n"
            "    Read verified 2026-08-29 against Live 12.4.5; a write was NOT attempted "
            "on this row.\n"
            "    Measured 2026-08-29 against Live 12.4.5: read True, wrote False, read "
            "back False, restored.\n"
            "Those two sentences say different things, and a coarse status cannot. If a "
            "row above is one you edited by hand, put it back to 'untested' -- an "
            "unverified row with an honest doc is a documented hypothesis, a falsely "
            "verified one is a lie with a path attached.\n\n" + HOW_TO_VERIFY
        ),
    )


def test_broken_rows_carry_the_evidence_that_broke_them() -> None:
    """``broken`` is a measurement too, and needs the same receipt.

    ``docs/catalog.md`` names three outcomes, not two: a probe that failed
    because the target did not exist *in that set* teaches nothing about the row
    and must stay ``untested``. Only a probe that showed the property genuinely is
    not there earns ``broken``, and a ``broken`` row without its evidence is
    worse than a missing one, because it stops the next person from looking.
    """
    offences = [
        f"{row.where}: status is 'broken' but doc records no measurement"
        for row in ROWS
        if row.get("status") == PathStatus.BROKEN.value
        and not has_measurement_note(str(row.get("doc") or ""))
    ]
    fail(
        "row(s) marked 'broken' with no evidence",
        offences,
        explain=(
            "A row is 'broken' only when a probe showed the property genuinely is not "
            "there, with the evidence in doc. A target that was absent from the test set "
            "-- no rack device, no audio clip -- says something about the set and nothing "
            "about the row: leave that one 'untested'.\n\n" + HOW_TO_VERIFY
        ),
    )


def test_untested_rows_carry_no_measurement_note() -> None:
    """The inverse: a row that *was* measured must not still read ``untested``.

    A measurement note under ``status: untested`` means somebody probed the row
    and forgot the status, so the catalog under-reports what is known. That is the
    cheap direction of the same error and it hides real, paid-for knowledge.
    """
    offences = [
        f"{row.where}: doc records a measurement but status is still 'untested'"
        for row in ROWS
        if row.get("status") == PathStatus.UNTESTED.value
        and has_measurement_note(str(row.get("doc") or ""))
    ]
    fail(
        "measured row(s) still marked 'untested'",
        offences,
        explain=(
            "If the note is real, set the status to match what was measured "
            "('verified' or 'broken'). If the note is about something else -- a "
            "measurement of a different row, a date in prose -- reword it so it "
            "does not read as a probe of this row.\n\n" + HOW_TO_VERIFY
        ),
    )


def test_no_doc_calls_itself_unverified_and_then_records_the_measurement() -> None:
    """A doc must not open by hedging and close by measuring. Pick one.

    Rows are written before they are probed and updated afterwards, so a
    pre-measurement hedge -- "Expected from the LOM; unverified." -- survives the
    probe that settled it and ends up sitting a sentence above "Read verified
    2026-08-29 ... returned False". Both halves are in the same paragraph and they
    disagree. A reader takes whichever they reach first, and the hedge is first.

    What stays allowed is the useful case, and the distinction is the whole point:
    a measurement that covers part of a row leaves the rest genuinely unmeasured,
    and saying so is honest. The rule is that such a sentence must name its
    subject -- "What an unknown name does is not measured" -- rather than passing a
    verdict on the row as a whole. Only the subjectless forms are refused here.
    """
    offences = []
    for row in ROWS:
        doc = str(row.get("doc") or "")
        if not has_measurement_note(doc):
            continue
        hedge = _BLANKET_HEDGE.search(doc)
        if hedge:
            offences.append(
                f"{row.where}: doc records a measurement and still says {hedge.group(0)!r}"
            )
    fail(
        "doc(s) that hedge and measure in the same breath",
        offences,
        explain=(
            "Delete the hedge: once the row is measured it says nothing, and it "
            "contradicts the note below it. Where the hedge carried real information "
            "-- 'Expected from Live 11 and later' -- keep the information and drop "
            "only the verdict. Where something genuinely was NOT covered by the "
            "measurement, say what: 'What an unknown name does is not measured' "
            "passes, because it cannot be read as a verdict on the row.\n\n"
            + HOW_TO_VERIFY
        ),
    )


def test_every_row_a_doc_points_at_actually_exists() -> None:
    """Rows send readers to other rows. Those rows have to be there.

    "Write through its `.value` companion row", "read `param.min` and `param.max`
    instead of assuming", "Read `device.get_bank_count` first" -- the catalog is
    full of these, and they are how a reader gets from a row that refuses a write
    to the one that accepts it. A pointer to a row that was deleted or renamed
    strands them, silently, at exactly the moment they needed help.

    Deleting three rows and repathing four in one session broke six of these, so
    the class is real and it is invisible: nothing else in the suite reads a doc
    as a reference.

    The hard part is that a row id and the tail of a LOM path are spelled
    identically -- ``simpler.sample_start`` is a row, and
    ``...devices[0].view.sample_start`` ends in something shaped just like one. The
    lookbehind is what tells them apart: a real citation does not follow a dot or a
    closing bracket.
    """
    ids = {str(row.get("id")) for row in ROWS}
    areas = {i.split(".")[0] for i in ids if "." in i}
    if not areas:  # pragma: no cover - only when the catalog failed to load
        pytest.skip("no rows loaded")
    token = re.compile(
        r"(?<![.\]\w])\b(?:" + "|".join(sorted(areas)) + r")\.[a-z][a-z0-9_]*(?![\w.\[(])"
    )

    offences = []
    for row in ROWS:
        for match in token.finditer(str(row.get("doc") or "")):
            cited = match.group(0)
            if cited in ids or cited.endswith((".yaml", ".yml", ".py", ".md")):
                continue
            offences.append(f"{row.where}: doc cites {cited!r}, which is not a row in the catalog")
    fail(
        "doc(s) pointing at a row that does not exist",
        offences,
        explain=(
            "Either the row was renamed -- check whether the id moved area, the way "
            "the Simpler slice editors became sample.* when the object turned out to "
            "be the Sample -- or it was deleted, in which case the sentence has to "
            "say what is true now rather than pointing somewhere else."
        ),
    )


# --------------------------------------------------------------------------------
# access, method, args
# --------------------------------------------------------------------------------


def test_access_values_are_valid() -> None:
    """``access`` is a non-empty list drawn from the five known verbs."""
    offences: list[str] = []
    for row in ROWS:
        access = row.get("access")
        if not isinstance(access, list) or not access:
            offences.append(f"{row.where}: access must be a non-empty list, got {access!r}")
            continue
        unknown = [a for a in access if a not in VALID_ACCESS]
        if unknown:
            offences.append(f"{row.where}: unknown access value(s) {unknown}")
        if len(set(access)) != len(access):
            offences.append(f"{row.where}: access repeats a verb: {access}")
    fail(
        "invalid access declaration(s)",
        offences,
        explain=f"Valid access verbs: {sorted(VALID_ACCESS)}",
    )


def test_method_appears_if_and_only_if_access_includes_call() -> None:
    """``method`` and ``access: [call]`` imply each other, both ways.

    A ``method`` without ``call`` is a name nothing will ever dial; a ``call``
    without a ``method`` is a row the executor cannot dispatch. Either way the row
    reads as a capability and is not one, and ``docs/protocol.md`` §6 is clear
    that the catalog's ``call`` is only a *request*: the Remote Script keeps its
    own allowlist inside Live, the two must agree, and the script wins.
    """
    offences: list[str] = []
    for row in ROWS:
        access = row.get("access")
        callable_row = isinstance(access, list) and Access.CALL.value in access
        method = row.get("method")
        if callable_row and not method:
            offences.append(f"{row.where}: access includes 'call' but no method is named")
        if method and not callable_row:
            offences.append(
                f"{row.where}: names method {method!r} but access is {access!r} "
                "-- nothing will ever call it"
            )
    fail("method/call disagreement(s)", offences)


def test_args_appear_only_on_call_rows() -> None:
    """``args`` are what ``lom_call`` sends; on a get/set row they vanish.

    ``params`` and ``args`` look alike and mean opposite things
    (``docs/catalog.md``): a param is substituted into the path, an arg is handed
    to the method and never appears in the path at all.
    """
    offences: list[str] = []
    for row in ROWS:
        args = row.get("args")
        if not args:
            continue
        access = row.get("access")
        if not (isinstance(access, list) and Access.CALL.value in access):
            names = [a.get("name") for a in args if isinstance(a, dict)]
            offences.append(f"{row.where}: declares args {names} but access is {access!r}")
    fail("args on a non-call row", offences)


def test_required_args_never_follow_an_optional_one() -> None:
    """``args`` goes on the wire positionally, so the order has to be fillable.

    The Live Object Model takes no keyword arguments (``docs/protocol.md`` §5.5),
    so an optional argument sitting before a required one cannot be skipped,
    declaration order *is* call order.
    """
    offences: list[str] = []
    for row in ROWS:
        args = row.get("args") or []
        if not isinstance(args, list):
            continue
        seen_optional = False
        for arg in args:
            if not isinstance(arg, dict):
                continue
            if arg.get("required", True) and seen_optional:
                offences.append(
                    f"{row.where}: required argument {arg.get('name')!r} follows an optional one"
                )
            seen_optional = seen_optional or not arg.get("required", True)
    fail("unfillable argument order(s)", offences)


def test_a_name_is_never_both_a_param_and_an_arg() -> None:
    """One name with two destinations is how a value reaches the wrong place."""
    offences: list[str] = []
    for row in ROWS:
        args = row.get("args") or []
        if not isinstance(args, list):
            continue
        arg_names = {a["name"] for a in args if isinstance(a, dict) and "name" in a}
        overlap = sorted(arg_names & set(_param_names(row)))
        if overlap:
            offences.append(f"{row.where}: {overlap} declared as both a path param and an arg")
    fail("param/arg name collision(s)", offences)


# --------------------------------------------------------------------------------
# Types, flags and bounds
# --------------------------------------------------------------------------------


def test_kind_and_unit_are_valid() -> None:
    """``kind`` and ``unit``, where present, name a known member."""
    offences: list[str] = []
    for row in ROWS:
        if "kind" in row.data and row.data["kind"] not in VALID_KIND:
            offences.append(
                f"{row.where}: kind {row.data['kind']!r} is not one of {sorted(VALID_KIND)}"
            )
        if "unit" in row.data and row.data["unit"] not in VALID_UNIT:
            offences.append(
                f"{row.where}: unit {row.data['unit']!r} is not one of {sorted(VALID_UNIT)}"
            )
    fail("invalid kind/unit value(s)", offences)


def test_enum_rows_list_their_values() -> None:
    """``kind: enum`` without an ``enum`` list validates nothing at all."""
    offences = [
        f"{row.where}: kind is 'enum' but no enum values are listed"
        for row in ROWS
        if row.get("kind") == Kind.ENUM.value and not row.get("enum")
    ]
    fail("enum row(s) with no values", offences)


def test_flags_are_real_booleans() -> None:
    """``destructive`` and ``quantized`` must be YAML booleans, not strings.

    ``destructive: "false"`` is a non-empty string, and a non-empty string is
    truthy. Read as a flag it would arm the confirm guard on a harmless row,
    or, with the quotes the other way round, disarm it on a dangerous one.
    """
    offences: list[str] = []
    for row in ROWS:
        for flag in ("destructive", "quantized"):
            if flag in row.data and not isinstance(row.data[flag], bool):
                offences.append(
                    f"{row.where}: {flag}={row.data[flag]!r} is a "
                    f"{type(row.data[flag]).__name__}, not true/false"
                )
    fail("non-boolean flag(s)", offences)


def test_range_is_a_pair_of_finite_numbers_in_order() -> None:
    """``range`` is an inclusive ``[min, max]``, validated before sending.

    An inverted pair rejects every value; a one-element pair raises inside the
    bounds check instead of naming the row.
    """
    offences: list[str] = []
    for row in ROWS:
        bounds = row.get("range")
        if bounds is None:
            continue
        if not isinstance(bounds, list) or len(bounds) != 2:
            offences.append(f"{row.where}: range {bounds!r} is not a [min, max] pair")
            continue
        low, high = bounds
        for bound in (low, high):
            if bound is None:
                continue
            if isinstance(bound, bool) or not isinstance(bound, (int, float)):
                offences.append(f"{row.where}: range bound {bound!r} is not a number")
        numeric = [
            b for b in (low, high) if isinstance(b, (int, float)) and not isinstance(b, bool)
        ]
        if len(numeric) == 2 and numeric[0] > numeric[1]:
            offences.append(f"{row.where}: range is inverted: min {low} is above max {high}")
    fail("malformed range(s)", offences)


def test_verify_names_a_hook() -> None:
    """``verify`` names a verifier; it is never blank.

    The name is not required to be *implemented*: the executor answers an
    unknown verifier with ``verified: None`` and says so, which is honest, but a
    blank one would silently mean "no check" on a row whose author meant the
    opposite. :func:`test_catalog_coverage_report` prints which names currently
    have code behind them.
    """
    offences = [
        f"{row.where}: verify={row.data['verify']!r}"
        for row in ROWS
        if "verify" in row.data
        and not (isinstance(row.data["verify"], str) and row.data["verify"].strip())
    ]
    fail("blank verify hook(s)", offences)


# --------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------


def test_every_path_template_fits_the_grammar() -> None:
    """Each ``path`` parses as a catalog template.

    ``segment = name | name[int] | name[{placeholder}]``, rooted at ``song`` or
    ``app``. No slices, no negative indices, no method calls, no expressions
    (``docs/protocol.md`` §6).
    """
    offences: list[str] = []
    for row in ROWS:
        path = row.get("path")
        if not isinstance(path, str) or not path:
            offences.append(f"{row.where}: missing 'path'")
            continue
        try:
            validate_path(path, allow_placeholders=True)
        except ValueError as exc:
            offences.append(f"{row.where}: {exc}")
    fail("path template(s) outside the grammar", offences)


def _sample_args(row: Row) -> dict[str, Any]:
    """Concrete values for a row's placeholders, for a build-and-parse round trip.

    Indices get 0; a segment-name placeholder gets the first value of its
    declared ``enum`` (the browser roots are a fixed set of names, not free text).
    """
    path = str(row.get("path") or "")
    by_name = name_placeholders_in(path)
    args: dict[str, Any] = {}
    for param in row.get("params") or []:
        if not isinstance(param, dict):
            continue
        name = param.get("name")
        if not isinstance(name, str):
            continue
        if name in by_name:
            choices = param.get("enum") or []
            args[name] = choices[0] if choices else "instruments"
        else:
            args[name] = 0
    return args


def test_every_path_builds_and_parses_under_lom_paths() -> None:
    """Substituted, every catalog path survives ``lom/paths.parse()``.

    Two implementations of one grammar sit in this repo: ``spec.validate_path``
    (which built the string) and ``lom.paths.parse`` (which the rest of the code
    walks with), plus a third inside Live. A row that satisfies one and not the
    others is a row that fails somewhere the author will not be looking.
    """
    registry = Registry.load()
    offences: list[str] = []
    for spec in registry.all():
        row = next((r for r in ROWS if r.data.get("id") == spec.id), None)
        args = _sample_args(row) if row is not None else {}
        try:
            built = build_path(spec, **args)
        except ValueError as exc:
            offences.append(f"{spec.id}: build_path failed: {exc}")
            continue
        try:
            lom_paths.parse(built)
        except lom_paths.PathSyntaxError as exc:
            offences.append(f"{spec.id}: built {built!r} but lom.paths rejected it: {exc}")
    fail("path(s) that build but do not parse", offences)


def test_paths_are_rooted_where_the_resolver_can_reach() -> None:
    """Every path starts at ``song`` or ``app``.

    ``song.view`` is a segment of ``song``, not a root of its own, and the script
    resolves from nothing else (``docs/protocol.md`` §6).
    """
    offences: list[str] = []
    for row in ROWS:
        path = row.get("path")
        if not isinstance(path, str) or not path:
            continue
        root = path.split(".", 1)[0].split("[", 1)[0]
        if root not in lom_paths.ROOTS:
            offences.append(f"{row.where}: path {path!r} is rooted at {root!r}")
    fail("path(s) with an unreachable root", offences, explain=f"Roots: {list(lom_paths.ROOTS)}")


# --------------------------------------------------------------------------------
# The loader, against the real catalog and against a planted duplicate
# --------------------------------------------------------------------------------


def test_registry_loads_the_packaged_catalog() -> None:
    """The catalog the tests read is the catalog the registry builds."""
    registry = Registry.load()
    specs = registry.all()
    assert len(specs) == len(ROWS), (
        f"registry loaded {len(specs)} specs but {len(ROWS)} rows parsed from disk"
    )
    for spec in specs:
        assert registry.get(spec.id) is spec
    counts = registry.status_counts()
    assert set(counts) == VALID_STATUS, "status_counts must report every status, even at zero"
    assert sum(counts.values()) == len(specs)


def test_registry_names_both_files_on_a_duplicate_id(tmp_path: Path) -> None:
    """A collision across two files must name both, not just the second.

    Ids are unique across the directory, and a message naming one file sends the
    reader to the wrong half of the diff (``docs/catalog.md``).
    """
    row = (
        "- id: track.volume\n"
        "  path: song.tracks[{track}].mixer_device.volume\n"
        "  access: [get]\n"
        "  kind: float\n"
        "  status: untested\n"
        "  doc: A duplicate planted by the test suite.\n"
        "  params:\n"
        "    - {name: track, kind: int, required: true}\n"
    )
    (tmp_path / "10-first.yaml").write_text(row, encoding="utf-8")
    (tmp_path / "20-second.yaml").write_text(row, encoding="utf-8")

    with pytest.raises(CatalogError) as excinfo:
        Registry.load(tmp_path)

    message = str(excinfo.value)
    assert "track.volume" in message
    assert "10-first.yaml" in message, f"first file not named: {message}"
    assert "20-second.yaml" in message, f"second file not named: {message}"


def test_registry_refuses_a_row_whose_placeholder_has_no_param(tmp_path: Path) -> None:
    """The loader enforces the placeholder rule too, and names the file.

    The data-side tests above cover today's catalog; this covers tomorrow's edit.
    """
    (tmp_path / "10-bad.yaml").write_text(
        "- id: track.volume\n"
        "  path: song.tracks[{track}].mixer_device.volume\n"
        "  access: [get]\n"
        "  status: untested\n"
        "  doc: No params entry for {track}.\n",
        encoding="utf-8",
    )
    with pytest.raises(CatalogError) as excinfo:
        Registry.load(tmp_path)
    message = str(excinfo.value)
    assert "10-bad.yaml" in message
    assert "track" in message


def test_registry_refuses_an_unknown_key(tmp_path: Path) -> None:
    """A misspelt key is an error, not a shrug: it would otherwise be dropped."""
    (tmp_path / "10-typo.yaml").write_text(
        "- id: song.tempo\n"
        "  path: song.tempo\n"
        "  access: [get]\n"
        "  desctructive: true\n"
        "  status: untested\n"
        "  doc: A misspelt key.\n",
        encoding="utf-8",
    )
    with pytest.raises(CatalogError) as excinfo:
        Registry.load(tmp_path)
    assert "desctructive" in str(excinfo.value)


# --------------------------------------------------------------------------------
# Information, not assertion
# --------------------------------------------------------------------------------


def test_catalog_coverage_report(capsys: pytest.CaptureFixture[str]) -> None:
    """Print row counts per area and per status. Not a threshold.

    Coverage of the LOM is a fact about the catalog, not a bar to clear, and a
    number that fails a build teaches people to pad it. This prints what is
  there (including how much of it is still an unproven hypothesis) and
    asserts only that the catalog is not empty.
    """
    from ableton_maestro.registry import area_of  # local: only this report needs it

    registry = Registry.load()
    specs = registry.all()
    per_file: dict[str, int] = {}
    for row in ROWS:
        per_file[row.file] = per_file.get(row.file, 0) + 1

    areas: dict[str, dict[str, int]] = {}
    for spec in specs:
        bucket = areas.setdefault(area_of(spec.id), dict.fromkeys(VALID_STATUS, 0))
        bucket[spec.status.value] += 1

    verify_names: dict[str, int] = {}
    for row in ROWS:
        name = str(row.get("verify") or "read_back")
        verify_names[name] = verify_names.get(name, 0) + 1

    with capsys.disabled():
        print("\n" + "=" * 78)
        print(f"catalog coverage - {len(specs)} rows in {len(per_file)} files")
        print("=" * 78)
        for file in sorted(per_file):
            print(f"  {file:<20} {per_file[file]:>5} rows")

        print(f"\n  {'area':<18} {'rows':>6} {'verified':>9} {'broken':>7} {'untested':>9}")
        print("  " + "-" * 52)
        for area in sorted(areas, key=lambda a: -sum(areas[a].values())):
            counts = areas[area]
            total = sum(counts.values())
            print(
                f"  {area:<18} {total:>6} "
                f"{counts['verified']:>9} {counts['broken']:>7} {counts['untested']:>9}"
            )
        print("  " + "-" * 52)
        totals = registry.status_counts()
        print(
            f"  {'TOTAL':<18} {len(specs):>6} "
            f"{totals['verified']:>9} {totals['broken']:>7} {totals['untested']:>9}"
        )

        print("\n  access verbs:")
        for verb in sorted(VALID_ACCESS):
            count = sum(1 for s in specs if Access(verb) in s.access)
            print(f"    {verb:<10} {count:>5}")

        print("\n  verify hooks named by rows (an unimplemented hook reports 'not verified'):")
        from ableton_maestro.executor import VERIFIERS  # local: report only

        for name in sorted(verify_names):
            state = "implemented" if name in VERIFIERS else "no code behind it"
            print(f"    {name:<18} {verify_names[name]:>5}   {state}")

        destructive = sum(1 for s in specs if s.destructive)
        print(f"\n  destructive rows (executor demands confirm=True): {destructive}")
        print(
            "\n  Reminder: only a 'verified' row may be described as working in the "
            "README,\n  in a tool description, or to a user. Everything else is a "
            "hypothesis with a\n  path attached (docs/catalog.md).\n"
        )

    assert specs, "the catalog is empty"


# --------------------------------------------------------------------------------
# `means`: the catalog decoding what Live will not
# --------------------------------------------------------------------------------

#: Every row that carries a ``means`` map, named on purpose. Adding one is then a
#: deliberate diff and the moment to ask where the meaning came from: a
#: transcription of the row's own doc, a measurement, or somebody's memory of the
#: LOM reference. The third is the one this list exists to catch, and it has
#: already caught something: three of these rows had their enum WRONG before the
#: names were read off Live on 2026-08-31 (`device.type` claimed a value 3 that
#: does not exist, `groove.base` was short one value and shifted by one,
#: `wavetable.unison_mode` had two names that Live spells differently).
ROWS_WITH_MEANS: dict[str, int] = {
    "arrangement_clip.launch_mode": 4,
    "arrangement_clip.launch_quantization": 15,
    "arrangement_clip.warp_mode": 7,
    "chain.choke_group": 1,
    "clip.launch_mode": 4,
    "clip.launch_quantization": 15,
    "clip.warp_mode": 7,
    "device.type": 4,
    "eq8.global_mode": 3,
    "groove.base": 6,
    "master_device.type": 4,
    "param.automation_state": 3,
    "param.state": 3,
    "return.crossfade_assign": 3,
    "return_device.type": 4,
    "scene.tempo": 1,
    "simpler.pad_playback_mode": 3,
    "simpler.playback_mode": 3,
    "simpler.sample_slicing_style": 4,
    "simpler.sample_warp_mode": 7,
    "simpler.slicing_playback_mode": 3,
    "simpler.slicing_style": 4,
    "song.clip_trigger_quantization": 14,
    "song.count_in_duration": 4,
    "song.midi_recording_quantization": 9,
    "song.root_note": 12,
    "song.session_record_status": 3,
    "track.crossfade_assign": 3,
    "track.current_monitoring_state": 3,
    "track.fired_slot_index": 2,
    "track.panning_mode": 2,
    "track.playing_slot_index": 2,
    "wavetable.filter_routing": 3,
    "wavetable.mono_poly": 2,
    "wavetable.unison_mode": 7,
}


def test_only_the_declared_rows_carry_means() -> None:
    """A value→meaning map is a claim about Live, so it does not arrive by accident."""
    found = {row.id: len(row.means) for row in Registry.load().all() if row.means}
    assert found == ROWS_WITH_MEANS, (
        "the rows carrying `means` changed. Add or remove the entry above "
        "deliberately, and say in the row's doc where the meaning came from."
    )

#: Sentences that tell a reader the row's VALUE NAMES were never measured. These are
#: not blanket hedges -- they name their subject, so ``_BLANKET_HEDGE`` leaves them
#: alone by design -- but a row carrying a measured ``means`` block has settled that
#: subject, and then the sentence is simply false.
_NAMES_UNSOURCED = re.compile(
    r"Read from the LOM enum order[^.]*\."
    r"|numeric mapping is deliberately (?:NOT|not) listed"
    r"|mapping is deliberately not listed"
    r"|not in a documented order this project has confirmed",
    re.IGNORECASE,
)


def test_no_row_carries_a_measured_means_block_and_calls_it_unmeasured() -> None:
    """A row that decodes its own values has stopped being a row that cannot.

    ``means`` is the measured value->meaning map, so the sentence "read from the LOM
    enum order, not measured" is not a hedge that aged -- it is contradicted by the
    block directly above it. Four rows carried both at once
    (``song.clip_trigger_quantization``, ``song.midi_recording_quantization``,
    ``track.current_monitoring_state``, ``track.crossfade_assign``): the pass that
    added the measured names appended provenance and never retracted the old claim,
    and a reader takes whichever they reach first.

    :data:`_BLANKET_HEDGE` does not cover these and should not. It refuses
    *subjectless* verdicts, and these name their subject -- which is what makes them
    legitimate on a row with no ``means`` and false on a row with one. That is the
    whole distinction, so it needs its own check rather than a wider pattern.
    """
    offences = []
    for row in ROWS:
        if not row.get("means"):
            continue
        found = _NAMES_UNSOURCED.search(" ".join(str(row.get("doc") or "").split()))
        if found:
            offences.append(f"{row.where}: carries a measured `means` block and still says "
                            f"{' '.join(found.group(0).split())!r}")
    fail(
        "row(s) that measure their values and deny it in the same doc",
        offences,
        explain=(
            "Delete the sentence. The `means` block is the measurement, and the "
            "provenance line below it says where it came from. Where the sentence "
            "carried something else worth keeping -- 'note that the middle value is "
            '"none"\' -- keep that part and drop only the claim about sourcing.'
        ),
    )



def test_every_meaning_is_a_sentence_and_not_a_hedge() -> None:
    """Same discipline as ``doc``, on a field that rides along with a value.

    A meaning that hedges everything says nothing, and a meaning over the cap is
    trying to be documentation, which is what ``doc`` is for.
    """
    offences = []
    for row in Registry.load().all():
        for key, meaning in row.means.items():
            if len(meaning) > MEANS_CAP:
                offences.append(f"{row.id}: means[{key!r}] is {len(meaning)} chars, over the cap")
            hedge = _BLANKET_HEDGE.search(meaning)
            if hedge:
                offences.append(f"{row.id}: means[{key!r}] hedges: {hedge.group(0)!r}")
    fail("meaning(s) that hedge or overrun", offences)


def test_an_unmeasured_meaning_says_so() -> None:
    """The two sentinels taken from the LOM reference rather than from Live.

    ``-2`` on both slot-index rows is the LOM's convention. One of them has since
    been seen (every track read -2 right after ``stop()``, 2026-08-31) and one has
    not. Neither may read as settled, because the catalog's whole value is that
    the difference between "measured" and "documented elsewhere" survives.
    """
    registry = Registry.load()
    for row_id in ("track.playing_slot_index", "track.fired_slot_index"):
        meaning = registry.get(row_id).meaning_of(-2)
        assert meaning is not None, f"{row_id} lost its -2 entry"
        assert "NOT " in meaning or "not established" in meaning, (
            f"{row_id}: the -2 meaning comes from the LOM reference, not from a "
            f"measurement, and has to say so: {meaning!r}"
        )


def test_a_meaning_is_found_however_the_number_is_spelled() -> None:
    """``scene.tempo``'s -1.0 is written as a float and read back as one.

    This failed the first time it was tried: the YAML key stayed the string
    "-1.0" while the value from Live canonicalised to "-1", so the lookup missed.
    """
    row = Registry.load().get("scene.tempo")
    assert row.meaning_of(-1.0) is not None
    assert row.meaning_of(-1) is not None
    assert row.meaning_of("-1.0") is not None
    # And an ordinary tempo says nothing at all: the point of the whole field.
    assert row.meaning_of(124.0) is None


# --------------------------------------------------------------------------------
# Duplicate keys, which the loader forgives
# --------------------------------------------------------------------------------


class _DuplicateReportingLoader(yaml.SafeLoader):
    """A loader that records repeated mapping keys instead of quietly dropping them."""

    duplicates: ClassVar[list[tuple[str, int, str]]] = []


def _record_duplicates(loader: _DuplicateReportingLoader, node: Any, deep: bool = False) -> Any:
    seen: dict[Any, int] = {}
    for key_node, _value in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            _DuplicateReportingLoader.duplicates.append(
                (Path(loader.name).name, key_node.start_mark.line + 1, str(key))
            )
        else:
            seen[key] = key_node.start_mark.line + 1
    return yaml.constructor.SafeConstructor.construct_mapping(loader, node, deep)


_DuplicateReportingLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _record_duplicates
)


def test_no_catalog_row_declares_the_same_key_twice() -> None:
    """A repeated key is not a parse error, which is exactly why it survives.

    YAML says a mapping key must be unique; PyYAML does not enforce it and keeps
    the last value, so a row with two ``params:`` loads perfectly and every test
    over the loaded catalog passes. Thirty rows carried a dead empty ``params:``
    directly above the real one -- from the very first commit, through every audit
    since, until a stricter editor flagged them.

    The cost is not the dead line. It is that the file stops being valid YAML for
    anything less forgiving than the loader this project happens to use, and that
    a real double declaration -- two different ``doc`` fields, two ``status``
    values -- would be swallowed the same silent way.
    """
    _DuplicateReportingLoader.duplicates = []
    for path in sorted(CATALOG_DIR.glob("*.yaml")):
        with path.open(encoding="utf-8") as handle:
            yaml.load(handle, _DuplicateReportingLoader)
    offences = [
        f"{name}:{line}: key {key!r} is declared twice; the earlier value is discarded"
        for name, line, key in _DuplicateReportingLoader.duplicates
    ]
    fail(
        "catalog key(s) declared more than once",
        offences,
        explain=(
            "Delete the losing declaration. PyYAML keeps the LAST one, so the "
            "row behaves as its second value and the first is dead text."
        ),
    )


def test_no_doc_line_is_a_truncated_copy_of_the_next() -> None:
    """A re-wrap that goes wrong leaves the old line above the new one.

    The shape is unmistakable and silent: a line, then the same line continuing
    one word further. YAML folds both into the doc, so the row parses, loads and
    reads almost right -- a sentence that starts twice and finishes once. Two rows
    carried it, ``song.set_or_delete_cue`` and one browser row, from some pass
    between the first commit and the tone sweep.

    Nothing else here would catch it: it is not a duplicate key, not a stale
    number, not a bad citation. Just prose that says the same thing twice and then
    goes on.
    """
    offences = []
    for path in sorted(CATALOG_DIR.glob("*.yaml")):
        lines = path.read_text(encoding="utf-8").split("\n")
        for i in range(len(lines) - 1):
            first, second = lines[i].strip(), lines[i + 1].strip()
            if len(first) > 30 and first != second and second.startswith(first):
                offences.append(f"{path.name}:{i + 1}: {first[:70]!r} repeats into the next line")
    fail(
        "doc line(s) duplicated by a bad re-wrap",
        offences,
        explain="Delete the shorter, truncated line; the longer one is the real sentence.",
    )
