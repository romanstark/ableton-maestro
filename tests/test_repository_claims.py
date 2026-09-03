"""The repository's claims about itself, checked against the repository.

Prose goes stale in a way code does not. A number written into a sentence is
correct on the day it is written and silently wrong from the next commit onward;
a citation to a section survives the section being renamed; a path pasted out of
a terminal carries the account name of whoever ran it. None of that fails a test
that only exercises behaviour, and all of it reaches a reader as fact.

Each test here closes one class of that, found the expensive way -- by reading --
before it could reach a published repository:

* a count in prose that the code contradicts (the allowlist was described as 74
  entries when it held 130; a CLI warning offered to probe "680 rows" of a 1156-row
  catalog),
* a citation that does not resolve (``ARCHITECTURE.md`` in four places including a
  packaging include list, where it silently packaged nothing; ``catalog.yaml``,
  which has never existed; ``limits.md.4``, a section-numbering form that document
  does not use),
* a local path or account name pasted into a shipped file.

They are deliberately narrow. A test that tried to check every number in the
repository would be a test nobody trusts; these check the numbers that were
actually found wrong, and the citation forms this repository actually uses.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ableton_maestro.registry import default_registry

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "live-remote-script" / "__init__.py"
SERVER = ROOT / "src" / "ableton_maestro" / "server.py"


def tracked_files(*suffixes: str) -> list[Path]:
    """Files git actually tracks. Anything else is not published and not our problem."""
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()
    return [ROOT / f for f in out if f.endswith(suffixes)]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def fail(headline: str, offences: list[str]) -> None:
    if offences:
        body = "\n".join(f"  - {line}" for line in offences)
        pytest.fail(f"{headline}:\n{body}", pytrace=False)


# --------------------------------------------------------------------------------
# Counts written into prose
# --------------------------------------------------------------------------------


def allowlist_entries() -> set[str]:
    """The frozenset the Remote Script actually checks, read off its AST.

    Read rather than imported: the script runs inside Live's interpreter and
    imports ``Live`` and ``_Framework``, neither of which exists out here.
    """
    for node in ast.walk(ast.parse(read(SCRIPT))):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "METHOD_ALLOWLIST" for t in node.targets
        ):
            return {ast.literal_eval(e) for e in node.value.args[0].elts}
    raise AssertionError("METHOD_ALLOWLIST not found in the Remote Script")


def test_counts_in_prose_match_what_the_code_holds() -> None:
    """Every number the repository states about itself, recomputed.

    The patterns below are sentence SHAPES, not expected numbers: each finds where
    a count is being asserted and the test then compares whatever number is there
    against the code. That way a figure only has to be wrong once to fail, and a
    correct figure in a sentence nobody predicted does not have to be listed here.

    These are the shapes that were actually found carrying a wrong number, not
    every number in the repository. A test that claimed to check them all would be
    a test nobody could trust.
    """
    registry = default_registry()
    rows = len(registry.all())
    verified = registry.status_counts()["verified"]
    allow = allowlist_entries()
    tools = len(re.findall(r"^@mcp\.tool", read(SERVER), re.MULTILINE))

    # Each shape has to name the WHOLE catalog, not a count of some subset. "the
    # catalog carried 25 rows for properties that do not exist" is a measurement of
    # one reconciliation run and "36 rows -- every mixer fader" is a family; both
    # are true and neither is the total, so neither may be matched here.
    shapes: list[tuple[re.Pattern[str], int, str]] = [
        (re.compile(r"(\d[\d,]*)\s+rows across"), rows, "catalog rows"),
        (re.compile(r"(\d[\d,]*)\s+rows\s*[-–—]\s*\d[\d,]*\s+`?verified"), rows, "catalog rows"),
        (re.compile(r"(\d[\d,]*)\s+rows,\s*\d[\d,]*\s+are"), rows, "catalog rows"),
        # The same sentence shape, capturing the SECOND number. The shape above
        # asserts only its own group(1), so "1163 rows, 1126 are verified" had its
        # total checked and its verified count ignored -- and that count was wrong by
        # one in server.py's module docstring. Anchored on "rows," so it fires only
        # where the sentence already named the whole catalog: a family count such as
        # "31 of the 40 device rows are verified" must not be compared to the total.
        (re.compile(r"\d[\d,]*\s+rows,\s*(\d[\d,]*)\s+are\s+`{0,2}verified"),
         verified, "verified catalog rows"),
        (re.compile(r"(\d[\d,]*)\s+addressable places"), rows, "catalog rows"),
        (re.compile(r"it is (\d[\d,]*) rows\b"), rows, "catalog rows"),
        (re.compile(r"(\d[\d,]*)\s+entries covering"), len(allow), "allowlist entries"),
        (re.compile(r"\*\*(\d[\d,]*) tools\b"), tools, "MCP tools"),
    ]

    offences = []
    for path in tracked_files(".md", ".py", ".toml"):
        if path.name == Path(__file__).name:
            continue
        text = read(path)
        for pattern, expected, what in shapes:
            for m in pattern.finditer(text):
                got = int(m.group(1).replace(",", ""))
                if got != expected:
                    line = text[: m.start()].count("\n") + 1
                    offences.append(
                        f"{path.relative_to(ROOT).as_posix()}:{line}: says {got} {what}, "
                        f"the code holds {expected}"
                    )
    fail("count(s) in prose that the code contradicts", offences)


# --------------------------------------------------------------------------------
# Citations
# --------------------------------------------------------------------------------

_DOC_REF = re.compile(r"docs/(\w[\w-]*\.md)")
_SECTION_REF = re.compile(r"docs/(\w[\w-]*\.md)[\s`\"']*(?:§|section\s+)(\d+)", re.IGNORECASE)
_MD_LINK = re.compile(r"\]\((?!https?:|mailto:|#)([^)#]+)")

#: A quoted section name cited beside a document, with or without a section number in
#: between: ``docs/architecture.md, 'the restart tax'`` and
#: ``docs/limits.md §5, "Cost of introspection and of listeners"`` both match. The gap
#: is kept short so an unrelated quotation later in the same sentence is not read as
#: part of the citation.
_SECTION_NAME_REF = re.compile(
    r"docs/(\w[\w-]*\.md)[^\n'\"]{0,24}?['\"]([^'\"\n]{4,60})['\"]"
)

#: A quoted string that is a template for the reader to fill in, not a citation:
#: ``'<Class>.<method>'`` tells a user what to type. Angle brackets are the marker.
_PLACEHOLDER = re.compile(r"[<>]")


def test_every_cited_document_is_tracked() -> None:
    """Verify cited documentation files are tracked in version control.

    Existing on this disk is not the test. ``CLAUDE.md`` and ``docs/PRD.md`` are
    deliberately untracked: they are working notes, and using an assistant is not
    part of what this server does, so they sit in the working tree and would
    never reach a clone. A tracked file citing one of them is a broken link for
    everybody but the author, and checking the filesystem cannot see it.
    """
    tracked = {
        p.relative_to(ROOT).as_posix()
        for p in tracked_files(".md", ".py", ".yaml", ".toml", ".yml")
    }
    offences = []
    for path in tracked_files(".md", ".py", ".yaml", ".toml", ".yml"):
        if path.name == Path(__file__).name:
            continue
        text = read(path)
        for m in _DOC_REF.finditer(text):
            cited = f"docs/{m.group(1)}"
            if cited not in tracked:
                why = "is not tracked" if (ROOT / cited).exists() else "does not exist"
                offences.append(f"{path.relative_to(ROOT).as_posix()}: cites {cited}, which {why}")
        for name in ("CLAUDE.md", "docs/PRD.md"):
            if name in text:
                offences.append(
                    f"{path.relative_to(ROOT).as_posix()}: mentions {name}, which is "
                    "deliberately untracked and will not exist for a reader who clones"
                )
    fail("citation(s) a clone would not resolve", sorted(set(offences)))


def test_every_cited_section_number_exists() -> None:
    """``docs/protocol.md §8`` has to be a heading in that file.

    A section number survives a document being reorganised, and a reader sent to
    the wrong section reads the wrong rule with no sign anything is amiss.
    """
    headings: dict[str, set[str]] = {}
    for doc in (ROOT / "docs").glob("*.md"):
        headings[doc.name] = set(re.findall(r"^##\s+(\d+)\.", read(doc), re.MULTILINE))

    offences = []
    for path in tracked_files(".md", ".py", ".yaml", ".toml", ".yml"):
        for m in _SECTION_REF.finditer(read(path)):
            doc, number = m.group(1), m.group(2)
            if doc in headings and headings[doc] and number not in headings[doc]:
                offences.append(
                    f"{path.relative_to(ROOT).as_posix()}: cites docs/{doc} section {number}, "
                    f"which has sections {sorted(headings[doc], key=int)}"
                )
    fail("citation(s) to a section that is not there", sorted(set(offences)))


def test_every_cited_section_name_exists() -> None:
    """``docs/limits.md §5, 'Cost of introspection'`` has to be findable in that file.

    The sibling above checks the number; this checks the name beside it, which is the
    half that rots when a document is reorganised rather than renumbered. Three shipped
    files cited a ``docs/limits.md`` section by name that the document no longer carried,
    and ``scripts/probe_paths.py`` cited a phrase under a section number whose subject
    had changed entirely. Both read as precise references and resolved to nothing.

    A phrase counts as present if it appears in a heading or anywhere in the prose, so
    quoting a sentence rather than a heading is allowed. Only a phrase that is nowhere
    in the cited document fails.
    """
    docs = {p.name: read(p) for p in (ROOT / "docs").glob("*.md")}
    docs["README.md"] = read(ROOT / "README.md")

    offences = []
    for path in tracked_files(".md", ".py", ".yaml", ".toml", ".yml"):
        if path.name == Path(__file__).name:
            continue
        for m in _SECTION_NAME_REF.finditer(read(path)):
            doc, phrase = m.group(1), m.group(2).strip()
            if doc not in docs or not phrase or _PLACEHOLDER.search(phrase):
                continue
            if phrase.casefold() not in docs[doc].casefold():
                offences.append(
                    f"{path.relative_to(ROOT).as_posix()}: cites docs/{doc} "
                    f"{phrase!r}, which is not in that document"
                )
    fail("citation(s) naming a section that is not there", sorted(set(offences)))


def test_every_relative_markdown_link_resolves() -> None:
    """A link is relative to the file that holds it, which is where these break."""
    offences = []
    for path in tracked_files(".md"):
        for m in _MD_LINK.finditer(read(path)):
            target = m.group(1).strip()
            if not target or " " in target:
                continue  # not a path: parenthesised prose
            if not (path.parent / target).exists():
                offences.append(
                    f"{path.relative_to(ROOT).as_posix()}: link to {target} does not resolve"
                )
    fail("markdown link(s) that do not resolve", sorted(set(offences)))


# --------------------------------------------------------------------------------
# What must not be published
# --------------------------------------------------------------------------------

#: A Windows or POSIX home directory with a real account name in it. Probe output
#: pasted into a doc is how these arrive: one catalog row carried
#: 'C:\Users\<name>\Music\Live\...' as the measured value of song.file_path.
_HOME_PATH = re.compile(r"(?:[A-Za-z]:[\\/]+Users[\\/]+|/(?:home|Users)/)([A-Za-z0-9._-]+)")

#: Environment variables and documentation placeholders are the correct way to
#: write these, and must not trip the test.
_PLACEHOLDER = re.compile(r"%\w+%|\$\{?\w+\}?|<[^>]+>|\byour\b|\busername\b", re.IGNORECASE)


def test_no_shipped_file_carries_a_home_directory_path() -> None:
    """An account name in a published file is somebody's, and it is not the reader's."""
    offences = []
    for path in tracked_files(".md", ".py", ".yaml", ".yml", ".toml", ".cfg", ".txt"):
        if path.name == Path(__file__).name:
            continue
        for i, line in enumerate(read(path).splitlines(), 1):
            m = _HOME_PATH.search(line)
            if m and not _PLACEHOLDER.search(line):
                offences.append(
                    f"{path.relative_to(ROOT).as_posix()}:{i}: home directory of "
                    f"{m.group(1)!r} in a published file"
                )
    fail("published file(s) carrying a home directory path", offences)


# --------------------------------------------------------------------------------
# Reports that must be complete, because they read as complete
# --------------------------------------------------------------------------------


def _function(name: str) -> ast.FunctionDef:
    """One top-level function of ``server.py``, by name."""
    for node in ast.walk(ast.parse(read(SERVER))):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in server.py")


def _strings_in(node: ast.AST) -> set[str]:
    return {
        n.value
        for n in ast.walk(node)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }


def test_delete_track_measures_both_places_a_clip_can_live() -> None:
    """A confirmation step that omits half the losses is worse than none.

    Live keeps clips in two independent places: the Session grid and the
    Arrangement timeline. Until 2026-08-30 ``delete_track``'s dry run measured
    only the Session, while its message said the track would go "with its
    devices, its clips and its clip automation": a sentence that reads as a
    complete inventory. It was caught in a 53-track set with a full arrangement,
    where every track had to be hand-probed before a delete was safe.
    """
    strings = _strings_in(_function("delete_track"))
    missing = [
        row
        for row in ("clip_slot.has_clip", "arrangement_clip.name")
        if row not in strings
    ]
    fail("delete_track no longer measures", [f"{row}: a clip can live there" for row in missing])


def test_no_tool_reports_an_unqualified_clip_count() -> None:
    """``clip_count`` is a total, and this server never surveys the Arrangement.

    The snapshot sweeps Session slots. Reporting that as ``clip_count`` invites
    the reader to treat it as every clip in the set; in a project whose
    arrangement holds material across twenty tracks, it is not close. The field
    is ``session_clip_count`` and the qualifier is the whole point of it.
    """
    offences = []
    for path in tracked_files(".py"):
        if path.name == Path(__file__).name:
            continue
        for i, line in enumerate(read(path).splitlines(), 1):
            if re.search(r'"clip_count"|\bsnap\.clip_count\b', line):
                offences.append(
                    f"{path.relative_to(ROOT).as_posix()}:{i}: unqualified clip_count: "
                    "say session_clip_count, or count the Arrangement too"
                )
    fail("unqualified clip count(s)", offences)


def test_read_clip_notes_can_be_asked_a_narrow_question() -> None:
    """Reading one bar must not mean transferring the clip.

    Measured 2026-08-30: surveying note density across 59 drum clips meant a full
    read of each, and a single 384-note clip came back as 57k characters: past
    the tool-result cap. Live has taken a time window since Live 9 and the LOM
    has no ``note_count`` at all, so both of these are ours to offer.
    """
    arguments = {arg.arg for arg in _function("read_clip_notes").args.args}
    missing = sorted({"from_time", "time_span", "count_only"} - arguments)
    fail("read_clip_notes can no longer be asked for less than everything", missing)


def test_a_group_tracks_loss_report_does_not_read_as_empty() -> None:
    """A group holds no clips of its own, and four tracks.

    ``delete_track`` on a group answered "0 device(s), 0 Session clip(s), no
    Arrangement clip(s)": every count honest, the whole reading false, because
    all of them describe the container. The numbers below are the real ones:
    measured 2026-08-30 against Live 12.4.5, group "Vocal Bridge" at index 28
    with tracks 29–32 inside it and track 33 outside.
    """
    from ableton_maestro import server

    measured = {"is_grouped": {"values": [False] * 29 + [True] * 4 + [False] * 18}}
    tracks = {
        "values": [f"track {i}" for i in range(29)]
        + [
            "29 Bridge Lead (Voice 0)",
            "30 Bridge Double (Voice 1)",
            "31 Bridge HarmLow (Voice 2)",
            "32 Bridge HarmHigh (Voice 3)",
        ]
        + [f"track {i}" for i in range(33, 51)]
    }
    members = server._members_of_group(28, measured, tracks)
    assert [m["index"] for m in members] == [29, 30, 31, 32], members
    assert members[0]["name"] == "29 Bridge Lead (Voice 0)"

    warning = server._group_warning({"is_group": True, "contained_tracks": members})
    for owed in ("GROUP track", "CONTAINER", "4 track(s)", "was not measured"):
        assert owed in warning, f"{owed!r} missing from the group warning: {warning}"

    # An ordinary track must not acquire the warning.
    assert server._group_warning({"is_group": False}) == ""


def test_load_device_names_the_rows_that_aim_a_load() -> None:
    """A tool that says "I cannot aim this" owes the caller what can.

    ``load_device`` explained that ``Browser.load_item`` takes no target and
    follows Live's selection, and named ``view.selected_track`` as the way to aim
    it, then stopped at tracks. In a real session on 2026-08-31 that gap became
    a wrong conclusion: an effect landed at the end of a drum rack and the
    session concluded Live's API cannot load into a chain at all. Two catalog
    rows say otherwise, and the docstring never mentioned them.
    """
    doc = ast.get_docstring(_function("load_device")) or ""
    missing = [
        row for row in ("selected_track", "selected_chain", "selected_drum_pad")
        if row not in doc
    ]
    fail("load_device no longer names the row that aims a load", missing)


def test_every_read_only_handler_exists_in_the_script() -> None:
    """``READ_ONLY_HANDLERS`` must not name a handler the script does not have.

    Which handlers belong in that set is asserted in ``tests/test_client.py``,
    beside the client that uses it. This checks only the half that file cannot,
    namely that every name in it is a handler the Remote Script actually declares. A set
    naming a handler that does not exist gives a timeout class and a retry to
    nothing, and nothing complains.
    """
    from ableton_maestro.client import READ_ONLY_HANDLERS

    declared = script_handlers()
    fail(
        "handler(s) in READ_ONLY_HANDLERS that the Remote Script does not declare",
        sorted(set(READ_ONLY_HANDLERS) - declared),
    )


def script_handlers() -> set[str]:
    """The ``HANDLERS`` tuple the Remote Script declares, read off its AST."""
    for node in ast.walk(ast.parse(read(SCRIPT))):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "HANDLERS" for t in node.targets
        ):
            return {ast.literal_eval(e) for e in node.value.elts}
    raise AssertionError("HANDLERS not found in the Remote Script")


# --------------------------------------------------------------------------------
# A document does not narrate its own edit history
# --------------------------------------------------------------------------------

#: The self-reference, not the correction. "Widely assumed impossible; it is not,
#: and here is the measurement" is exactly what this repository is for. "This
#: section used to say X, and that was wrong" is a different sentence: it tells a
#: reader about a revision they never saw, it dates instantly, and the place for
#: it is the commit message that made the change.
_EDIT_HISTORY = re.compile(
    r"\b(?:this|the)\s+(?:row|docstring|document|section|module|page|file|paragraph|line|"
    r"list|catalog|project)\s+(?:used\s+to|said|claimed|briefly|got\s+(?:it\s+)?wrong|"
    r"had\s+it\s+wrong)"
    r"|\ban\s+earlier\s+version\s+of\s+(?:this|the)\b"
    r"|\bthe\s+first\s+version\s+of\s+this\b"
    r"|\buntil\s+it\s+was\s+measured\b"
    r"|\bsaid\s+the\s+opposite\b",
    re.IGNORECASE,
)

#: Inline code is a quotation, not a claim: CONTRIBUTING.md has to show the bad
#: form to forbid it.
_INLINE_CODE = re.compile(r"`[^`]*`|``[^`]*``")


def test_no_shipped_prose_narrates_its_own_edit_history() -> None:
    """Nobody reading this repository has seen an earlier draft of it.

    A retraction is worth keeping: somebody planned around a limit and needs to
    learn it was lifted. What that wants is the claim named as an assumption, not 
    the file's revision history: "widely assumed impossible; it is not, and here is 
    the measurement". The history goes in the commit that made the change, where it 
    is exact and where it does not age into a puzzle.

    ``tests/`` is exempt. A test docstring naming the defect it guards is stating
    the test's purpose, and that is the one place the earlier bug is the point.
    """
    offences = []
    for path in tracked_files(".md", ".py", ".yaml", ".yml"):
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith("tests/"):
            continue
        for i, line in enumerate(read(path).splitlines(), 1):
            match = _EDIT_HISTORY.search(_INLINE_CODE.sub("", line))
            if match:
                offences.append(f"{relative}:{i}: {match.group(0)!r} (write about the claim)")
    fail("prose narrating its own edit history", offences)


# --------------------------------------------------------------------------------
# Version numbers
# --------------------------------------------------------------------------------

#: A version literal in a line that is talking about the SCRIPT. Live versions and
#: loopback addresses are three-part numbers too, so the subject of the sentence
#: decides, not the shape of the number.
_SCRIPT_VERSION_CLAIM = re.compile(
    r"(?i)\bscript\b[^\n]{0,60}?(\d+\.\d+\.\d+)"
    r"|(\d+\.\d+\.\d+)[^\n]{0,20}?\bscript\b"
)

#: Files whose version literals are not claims: the package's own metadata, test
#: fixtures where a fake reply only has to round-trip, and the one module that
#: defines SCRIPT_VERSION in the first place.
_VERSION_EXEMPT = (
    "pyproject.toml",
    "src/ableton_maestro/__init__.py",
    "tests/",
    "live-remote-script/__init__.py",
)


def script_version() -> str:
    """``SCRIPT_VERSION`` from the Remote Script, read off its AST."""
    for node in ast.walk(ast.parse(read(SCRIPT))):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "SCRIPT_VERSION" for t in node.targets
        ):
            return str(ast.literal_eval(node.value))
    raise AssertionError("SCRIPT_VERSION not found in the Remote Script")


def test_no_prose_cites_a_remote_script_version() -> None:
    """A measurement is dated against a LIVE version, never against a script one.

    Three version numbers live here and they move independently: the Python
    package, ``SCRIPT_VERSION`` in the Remote Script, and ``PROTOCOL_VERSION``.
    Only the Live version belongs in a measurement note, because it is the thing
    being measured.

    A script version in prose is wrong in one of two ways and usually both. It
    ages on the next release: the protocol examples carried a stale one within a
    day of being written, which is why they now hold a placeholder. And it is
    often unmeasurable after the fact: catalog rows dated 2026-08-29 claimed to
    have been measured "with script 0.4.0" and "with Remote Script 0.6.0", one not
    yet released and the other never existing at all, and twelve more cited a
    0.5.0. Each was a real measurement rounded out with a detail nobody checked.

    The earlier cleanup of those searched for the wrong numbers one at a time and
    missed two. This searches for the shape instead.
    """
    current = script_version()
    offences = []
    for path in tracked_files(".md", ".py", ".yaml", ".yml", ".toml"):
        relative = path.relative_to(ROOT).as_posix()
        if relative == Path(__file__).name or relative.startswith(_VERSION_EXEMPT):
            continue
        lines = read(path).splitlines()
        for i, line in enumerate(lines, 1):
            for match in _SCRIPT_VERSION_CLAIM.finditer(line):
                found = match.group(1) or match.group(2)
                if found == current:
                    continue
                # "the Remote Script ... measured against Live 12.4.5" puts both
                # words in one line. What the number belongs to is decided by what
                # stands immediately before it -- and a wrapped paragraph can put
                # that word on the line above, which is why the previous line is
                # part of the lookback.
                at = line.index(found, match.start())
                before = (lines[i - 2] if i >= 2 else "") + " " + line[:at]
                if "live" in before[-12:].casefold():
                    continue
                offences.append(
                    f"{relative}:{i}: cites script version {found!r}; the script is "
                    f"{current} and a measurement is dated against Live"
                )
    fail("script version(s) cited in prose", offences)


def test_the_package_version_is_stated_once_and_agrees_with_itself() -> None:
    """Two literals, one number. The only pairing here that must be kept in step.

    ``pyproject.toml`` and ``ableton_maestro.__version__`` both carry it, and
    nothing compared them until this test. That is the same shape as the catalog
    count, which stood in seventeen places and was stale in four of them.

    The better fix is to state it once: hatchling reads a version out of a source
    file with ``dynamic = ["version"]`` and ``[tool.hatch.version] path = ...``,
    and it is not applied here because the build backend is not installed in this
    environment, so the change could not be verified by building. A build config
    edited blind is a worse trade than a duplication with a test on it.

    Note for anyone confused by a third number: ``importlib.metadata.version``
    reports whatever was last INSTALLED, which lags the source until the package
    is reinstalled. Nothing in this repository reads it, and it is not checked
    here for that reason.
    """
    import tomllib

    declared = tomllib.loads(read(ROOT / "pyproject.toml"))["project"]["version"]
    module = next(
        ast.literal_eval(node.value)
        for node in ast.walk(ast.parse(read(ROOT / "src" / "ableton_maestro" / "__init__.py")))
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "__version__" for t in node.targets)
    )
    assert declared == module, (
        f"pyproject.toml says {declared!r} and __init__.py says {module!r}. "
        "One number, two places, and they have drifted."
    )


# --------------------------------------------------------------------------------
# The Remote Script, which nothing else checks
# --------------------------------------------------------------------------------

#: Everything the Remote Script is allowed to import: the standard library of the
#: interpreter running these tests, plus what Live itself provides. ``Queue`` is
#: the Python 2 spelling and is not a 3.x stdlib name; the script keeps it behind
#: a ``try`` because Live's interpreter history is long.
_SCRIPT_MAY_IMPORT = {"Live", "_Framework", "Queue"}


def _script_tree() -> ast.AST:
    return ast.parse(read(SCRIPT))


def test_the_remote_script_compiles() -> None:
    """The one file in this repository that no other check would catch.

    ``ruff`` excludes ``live-remote-script/`` (it is not Python 3.11 code) and no
    test imports it, because importing it needs Live's own ``Live`` and
    ``_Framework`` modules. So a syntax error here survives the whole suite and
    surfaces as Live failing to load the Control Surface -- after an install, a
    ``__pycache__`` delete and a restart, which is the most expensive way in this
    project to learn about a typo.

    ``compile`` rather than ``ast.parse``: a ``continue`` outside a loop parses
    cleanly and only fails at compile time, which is exactly the kind of edit a
    prose sweep can leave behind.
    """
    try:
        compile(read(SCRIPT), str(SCRIPT), "exec")
    except SyntaxError as exc:
        pytest.fail(f"{SCRIPT.name}:{exc.lineno}: {exc.msg}")


def test_the_remote_script_avoids_syntax_lives_interpreter_may_not_have() -> None:
    """No f-strings and no annotations, because this file is imported by Live.

    The script says so itself: "no type annotations, no f-strings -- the
    interpreter is Live's, and its history is long". Every other file in the
    repository is Python 3.11+ with full type hints, so both are what a
    reformatter reaches for by reflex, and neither is caught by anything else --
    this file compiles fine under 3.12 and would still be refused by an older
    embedded interpreter.
    """
    offences = []
    for node in ast.walk(_script_tree()):
        if isinstance(node, ast.JoinedStr):
            offences.append(f"line {node.lineno}: f-string")
        elif isinstance(node, ast.AnnAssign):
            offences.append(f"line {node.lineno}: annotated assignment")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                offences.append(f"line {node.lineno}: return annotation on {node.name}()")
            args = node.args
            for arg in args.posonlyargs + args.args + args.kwonlyargs:
                if arg.annotation is not None:
                    offences.append(
                        f"line {node.lineno}: annotation on {node.name}() argument {arg.arg}"
                    )
    fail("syntax in the Remote Script that Live's interpreter may refuse", offences)


def test_the_remote_script_imports_only_what_live_provides() -> None:
    """Standard library plus Live's own modules. No third-party dependency, ever.

    The script is installed into the user's Ableton folder as a single file. There
    is no place to put a dependency and no installer to fetch one, so an import
    outside this set is not a bad choice -- it is a file that cannot load.
    """
    allowed = set(sys.stdlib_module_names) | _SCRIPT_MAY_IMPORT
    offences = []
    for node in ast.walk(_script_tree()):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module] if node.module else []
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            if top and top not in allowed:
                offences.append(f"line {node.lineno}: imports {name!r}, which Live will not have")
    fail("import(s) the Remote Script cannot rely on", offences)
