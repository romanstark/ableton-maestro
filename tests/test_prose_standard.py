"""The prose standard for docstrings and comments, enforced rather than reviewed.

Two classes of damage motivated these checks, both of them invisible to a test suite
that only exercises behaviour:

* A refactoring pass that rewrote docstrings for tone deleted measurements along with
  the prose that carried them. ``als/write.py`` lost every one of its 18 references to
  a measurement, including the provenance for the gzip container fields, leaving
  ``_GZIP_OS_BYTE = 0x0A`` as an unexplained constant. Nothing failed, because no test
  asserted any of those sentences.
* The same pass stamped ``Measured <date> against Live <version>`` onto a finding about
  MCP JSON-Schema flattening, which is a client-side property that was never measured
  against Live at all. The format this project treats as load-bearing was applied to
  something outside its scope.

:func:`test_measurements_are_not_deleted_silently` closes the first: a module may gain
measurements freely, and losing one costs a red test and a deliberate edit to the floor
below. :func:`test_a_live_dated_measurement_sits_in_a_module_that_reaches_live` closes
the second: the citation form is confined to the modules that can actually produce one.

:func:`test_no_docstring_or_comment_uses_an_em_dash` holds the punctuation standard
(PEP 8 prose, hyphens and commas rather than em-dashes). It reads the token stream
rather than the raw text, because three em-dashes in this repository are data, not
prose: two regex character classes and the separator ``docs/limits.md`` puts between an
entry title and its class. Replacing those breaks a parse, so the check is scoped to
docstrings and comments and never touches a string literal.
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EM_DASH = "—"

#: Floor per module: (references to a measurement, of which dated ones). Recorded
#: 2026-09-02. Raise a number when a module gains a measurement; lowering one is a
#: deliberate statement that the finding is gone for good, and belongs in the commit
#: message that does it.
PROVENANCE_FLOOR: dict[str, tuple[int, int]] = {
    "src/ableton_maestro/als/__init__.py": (2, 1),
    "src/ableton_maestro/als/read.py": (15, 1),
    "src/ableton_maestro/als/write.py": (17, 3),
    "src/ableton_maestro/automation.py": (9, 0),
    "src/ableton_maestro/lom/__init__.py": (3, 0),
    "src/ableton_maestro/lom/introspect.py": (8, 0),
    "src/ableton_maestro/models.py": (4, 2),
    "src/ableton_maestro/music/__init__.py": (14, 0),
    "src/ableton_maestro/music/humanize.py": (15, 0),
    "src/ableton_maestro/music/notes.py": (8, 1),
    "src/ableton_maestro/music/theory.py": (69, 0),
    "src/ableton_maestro/server.py": (70, 20),
    "scripts/install_script.py": (8, 2),
    "scripts/inventory.py": (10, 2),
    "scripts/probe_paths.py": (9, 0),
    "scripts/sync_catalog.py": (15, 6),
}

#: Modules allowed to date a finding against a Live version. Each one either talks to
#: Live over the socket, reads or writes the saved project file, or ships the catalog
#: rows those measurements are written onto. A module outside this set claiming a Live
#: measurement is describing something it cannot observe.
LIVE_FACING = (
    "src/ableton_maestro/server.py",
    "src/ableton_maestro/client.py",
    "src/ableton_maestro/executor.py",
    "src/ableton_maestro/models.py",
    "src/ableton_maestro/registry.py",
    "src/ableton_maestro/lom/",
    "src/ableton_maestro/als/",
    "src/ableton_maestro/music/notes.py",
    "src/ableton_maestro/music/theory.py",
    "src/ableton_maestro/music/humanize.py",
    "src/ableton_maestro/automation.py",
    "scripts/",
)

MEASURED = re.compile(r"measured", re.IGNORECASE)
DATED = re.compile(r"Measured 20\d{2}-\d{2}-\d{2}")


def python_files() -> list[Path]:
    """Return every Python file under src/ and scripts/, caches excluded."""
    out: list[Path] = []
    for root in ("src", "scripts"):
        out.extend(
            p for p in sorted((ROOT / root).rglob("*.py")) if "__pycache__" not in p.parts
        )
    return out


def relative(path: Path) -> str:
    """Return the repository-relative POSIX path of a file."""
    return path.relative_to(ROOT).as_posix()


def prose_tokens(source: str) -> list[tuple[int, str]]:
    """Return (line, text) for every docstring and comment token in source.

    A string literal that is not a docstring is excluded: it is runtime output or a
    pattern, not prose about the code.
    """
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    skip = (
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.COMMENT,
        tokenize.ENCODING,
    )
    out: list[tuple[int, str]] = []
    for i, token in enumerate(tokens):
        if token.type == tokenize.COMMENT:
            out.append((token.start[0], token.string))
            continue
        if token.type != tokenize.STRING:
            continue
        back = i - 1
        while back >= 0 and tokens[back].type in skip:
            back -= 1
        is_docstring = back < 0 or (
            tokens[back].type == tokenize.OP and tokens[back].string == ":"
        )
        if is_docstring:
            out.append((token.start[0], token.string))
    return out


def fail(headline: str, offences: list[str]) -> None:
    """Fail with one line per offence, or return when there are none."""
    if offences:
        body = "\n".join(f"  - {line}" for line in offences)
        pytest.fail(f"{headline}:\n{body}", pytrace=False)


def test_no_docstring_or_comment_uses_an_em_dash() -> None:
    """An em-dash in prose is a hyphen, comma, colon or pair of parentheses.

    Scoped to docstrings and comments. The three em-dashes this repository needs are
    all inside patterns, where the character is matched rather than read.
    """
    offences = []
    for path in python_files():
        for line, text in prose_tokens(read_source(path)):
            if EM_DASH in text:
                offences.append(f"{relative(path)}:{line}")
    fail("em-dash in a docstring or comment", offences)


def test_measurements_are_not_deleted_silently() -> None:
    """Every module keeps at least the measurements recorded in PROVENANCE_FLOOR.

    A measurement is expensive to make and cheap to delete while rewriting the sentence
    around it. This is the check that makes the deletion cost something.
    """
    offences = []
    for module, (floor_all, floor_dated) in sorted(PROVENANCE_FLOOR.items()):
        path = ROOT / module
        if not path.exists():
            offences.append(f"{module}: gone from the tree, floor still recorded")
            continue
        source = read_source(path)
        found_all = len(MEASURED.findall(source))
        found_dated = len(DATED.findall(source))
        if found_all < floor_all:
            offences.append(
                f"{module}: {found_all} references to a measurement, floor is {floor_all}"
            )
        if found_dated < floor_dated:
            offences.append(
                f"{module}: {found_dated} dated measurements, floor is {floor_dated}"
            )
    fail("measurements lost since the floor was recorded", offences)


def test_a_live_dated_measurement_sits_in_a_module_that_reaches_live() -> None:
    """``Measured <date> against Live <version>`` belongs where Live can be observed.

    The form is this project's evidence that a claim was checked rather than assumed.
    Applied to something no Live version has any bearing on, it spends that credibility
    on a guess.
    """
    offences = []
    for path in python_files():
        module = relative(path)
        if DATED.search(read_source(path)) and not module.startswith(LIVE_FACING):
            offences.append(f"{module}: dates a finding against Live but never reaches it")
    fail("Live-dated measurement outside a Live-facing module", offences)


def test_the_provenance_floor_names_only_files_that_exist() -> None:
    """A floor entry for a deleted or renamed module would never fail again."""
    missing = [module for module in sorted(PROVENANCE_FLOOR) if not (ROOT / module).exists()]
    fail("PROVENANCE_FLOOR names a file that is not in the tree", missing)


def read_source(path: Path) -> str:
    """Return the text of a Python file."""
    return path.read_text(encoding="utf-8")
