"""Install and manage the Ableton Maestro Remote Script in the User Library.

Usage:
    python scripts/install_script.py            # locate library, copy script, clear __pycache__
    python scripts/install_script.py --dry-run  # preview actions without modifying files
    python scripts/install_script.py --diff     # compare repository script with installed version
    python scripts/install_script.py --uninstall

Key installation mechanics:
- User Library discovery: parses ``Preferences/Library.cfg`` to resolve target directories,
  ranking candidates using the ``Ableton Folder Info`` marker.
- Control Surface registration: the folder name under ``Remote Scripts/`` determines
  the dropdown identifier in Live Preferences (``Remote Scripts/AbletonMaestro/``).
- Cache invalidation: removes stale ``__pycache__`` artifacts to ensure Live loads updated source.
- Bytecode verification: compares PEP 552 ``.pyc`` headers against source mtime and size.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "BACKUP_PREFIX",
    "DEFAULT_SOURCE",
    "FOREIGN_FOLDER_NAMES",
    "FOREIGN_PORT",
    "OUR_PORT",
    "SCRIPT_FOLDER_NAME",
    "InstalledScript",
    "LibraryCandidate",
    "LiveInstall",
    "SetupError",
    "candidates_for",
    "discover_user_libraries",
    "find_live_installs",
    "library_score",
    "main",
    "preferences_roots",
    "pyc_matches_source",
    "read_library_cfg",
    "scan_remote_scripts",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: The folder name under ``Remote Scripts``. This string is user-visible: it is
#: exactly what Live prints in Preferences -> Link, Tempo & MIDI -> Control
#: Surface. Changing it renames the dropdown entry and clears the selection.
SCRIPT_FOLDER_NAME = "AbletonMaestro"

#: Folder names that belong to other Ableton Remote Scripts, not to this one.
#: Writing into any of them is refused under every flag: this installer only
#: ever creates and writes its own folder. A tool that quietly overwrites its
#: neighbours is not one anybody should have to trust with a User Library.
FOREIGN_FOLDER_NAMES: frozenset[str] = frozenset({"abletonmcp", "ableton-mcp", "ableton_mcp"})

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO_ROOT / "live-remote-script" / "__init__.py"

#: Backups are ``__init__.py.bak-<YYYYmmdd-HHMMSS>`` beside the file they
#: replace. Never deleted by this script, including on ``--uninstall``.
BACKUP_PREFIX = "__init__.py.bak-"

#: The port this Remote Script listens on, and the neighbouring port it stays
#: off. 9877 is already in use by other Ableton Remote Scripts, and two servers
#: bound to one port fail in a way neither of them reports, so this one takes
#: 9878 and never argues about it.
OUR_PORT = 9878
FOREIGN_PORT = 9877

#: Live's own marker folder inside a User Library. Measured 2026-08-29: present
#: in the real User Library, absent from the folder ``<ProjectPath>`` named.
USER_LIBRARY_MARKER = "Ableton Folder Info"

#: The folders Live creates inside a User Library. Two or more of them together
#: are good secondary evidence when the marker folder is missing.
CLASSIC_SUBFOLDERS: tuple[str, ...] = (
    "Presets",
    "Samples",
    "Clips",
    "Defaults",
    "Grooves",
    "Templates",
    "Tunings",
    "MIDI Tools",
)

#: A source file that is not a Remote Script would install cleanly and then do
#: nothing, so the copy is refused unless the file carries Live's mandatory
#: entry point.
REQUIRED_SOURCE_MARKER = "def create_instance"

_VERSION_RE = re.compile(r"^Live\s+(\d+(?:\.\d+)*)\s*(\S*)$")

_VERSION_FIELDS = 4

#: How many Live versions a candidate names before the rest are summarised.
_VERSIONS_SHOWN = 3


class SetupError(RuntimeError):
    """Something the user has to fix, with the fix in the message."""


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LiveInstall:
    """One ``Live <version>`` preferences folder, and what its Library.cfg said.

    ``project_path`` and ``project_name`` are read straight out of
    ``Library.cfg``; neither is interpreted here. ``error`` carries the reason
    when the file could not be read, because a Live version whose preferences
    are unreadable is worth reporting rather than silently skipping.
    """

    version: tuple[int, ...]
    version_text: str
    folder: Path
    library_cfg: Path
    project_path: Path | None = None
    project_name: str | None = None
    error: str | None = None

    @property
    def preferences(self) -> Path:
        """The ``Preferences`` folder, where ``Log.txt`` lives too."""
        return self.folder / "Preferences"

    @property
    def sort_key(self) -> tuple[tuple[int, ...], int]:
        """Newest first when reversed; a plain release outranks a suffixed build."""
        padded = self.version + (0,) * (_VERSION_FIELDS - len(self.version))
        suffixed = 0 if self.version_text.split(" ", 1)[-1][-1:].isalpha() else 1
        return padded[:_VERSION_FIELDS], suffixed


@dataclass(frozen=True)
class LibraryCandidate:
    """A folder that might be the User Library, with the evidence for it.

    Kept as a candidate rather than a decision so that the reasoning survives
    into the output: when this script picks a folder, it can say why, and when
    it asks, the human sees the same evidence it saw.
    """

    path: Path
    origin: str
    score: int
    evidence: str
    from_versions: tuple[str, ...] = ()

    @property
    def exists(self) -> bool:
        """Whether the folder is actually there. A missing one is never installed into."""
        return self.score >= 0

    @property
    def remote_scripts(self) -> Path:
        """``<candidate>/Remote Scripts``, created on install if it is missing."""
        return self.path / "Remote Scripts"

    def describe(self) -> str:
        """Two lines for a list the user reads.

        The version list is capped at :data:`_VERSIONS_SHOWN`: on the reference
        machine 28 Live versions named the same library, and printing all 28
        turned the evidence line into a wall nobody reads.
        """
        shown = list(self.from_versions[:_VERSIONS_SHOWN])
        extra = len(self.from_versions) - len(shown)
        if extra > 0:
            shown.append(f"and {extra} older version(s)")
        via = f"{self.origin} of {', '.join(shown)}" if shown else self.origin
        return f"{self.path}\n      {self.evidence}; from {via}"


def preferences_roots() -> list[Path]:
    """Where Live keeps its per-version preferences folders on this platform.

    Windows is *measured*; macOS is written from the documented layout and is
    unverified. Linux returns nothing, because Live does not run there, and an
    invented path would only produce a confusing error later.
    """
    if sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return [base / "Ableton"]
    if sys.platform == "darwin":
        return [Path.home() / "Library" / "Preferences" / "Ableton"]
    return []


def find_live_installs(roots: Sequence[Path] | None = None) -> list[LiveInstall]:
    """Every ``Live <version>`` folder under the preferences roots, newest first.

    Non-version folders are skipped without comment: ``%APPDATA%\\Ableton`` also
    holds ``Live Reports``, which is not a Live version and has no
    ``Library.cfg`` (measured 2026-08-29, alongside 41 real version folders).
    """
    installs: list[LiveInstall] = []
    for root in preferences_roots() if roots is None else roots:
        if not root.is_dir():
            continue
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            match = _VERSION_RE.match(folder.name)
            if match is None:
                continue
            version = tuple(int(part) for part in match.group(1).split("."))
            cfg = folder / "Preferences" / "Library.cfg"
            project_path, project_name, error = read_library_cfg(cfg)
            installs.append(
                LiveInstall(
                    version=version,
                    version_text=folder.name,
                    folder=folder,
                    library_cfg=cfg,
                    project_path=project_path,
                    project_name=project_name,
                    error=error,
                )
            )
    installs.sort(key=lambda install: install.sort_key, reverse=True)
    return installs


def read_library_cfg(path: Path) -> tuple[Path | None, str | None, str | None]:
    """Read ``<ProjectPath>`` and ``<ProjectName>`` out of a ``Library.cfg``.

    Returns ``(project_path, project_name, error)``; exactly one of the first
    value and the error is meaningful. The file is XML with the shape::

        <Ableton>
          <ContentLibrary>
            <UserLibrary>
              <LibraryProject Id="2">
                <ProjectLocation />
                <ProjectName Value="User Library" />
                <ProjectPath Value="D:/Documents/Ableton" />

    ``<ProjectPath>`` uses forward slashes even on Windows; ``Path`` copes.

    Never raises for a bad file. A Live version with a broken ``Library.cfg`` is
    one line of output, not the end of the search: there are usually forty
    others.
    """
    try:
        tree = ET.parse(path)
    except FileNotFoundError:
        return None, None, "no Library.cfg (this version was never fully started?)"
    except ET.ParseError as exc:
        return None, None, f"Library.cfg did not parse: {exc}"
    except OSError as exc:
        return None, None, f"Library.cfg unreadable: {exc.__class__.__name__}: {exc}"

    project = tree.getroot().find("./ContentLibrary/UserLibrary/LibraryProject")
    if project is None:
        return None, None, "Library.cfg has no ContentLibrary/UserLibrary/LibraryProject"

    raw_path = _value_of(project, "ProjectPath")
    name = _value_of(project, "ProjectName")
    if not raw_path:
        return None, name, "Library.cfg carries an empty <ProjectPath>"
    return Path(raw_path), name, None


def _value_of(parent: ET.Element, tag: str) -> str | None:
    """The ``Value`` attribute of a child element, or ``None``."""
    child = parent.find(tag)
    if child is None:
        return None
    value = child.get("Value")
    return value.strip() if value else None


def library_score(path: Path) -> tuple[int, str]:
    """Rate how much a folder looks like a User Library, and say why.

    Higher is more certain. The evidence string goes straight into the output,
    because a path chosen for a reason the user can check beats a path chosen
    by a rule they have to trust.
    """
    if not path.is_dir():
        return -1, "does not exist"
    if (path / USER_LIBRARY_MARKER).is_dir():
        return 3, f"has {USER_LIBRARY_MARKER}/ - Live's own User Library marker"
    hits = [name for name in CLASSIC_SUBFOLDERS if (path / name).is_dir()]
    if len(hits) >= 2:
        return 2, "has " + ", ".join(f"{name}/" for name in hits[:3]) + " - Live's own layout"
    if (path / "Remote Scripts").is_dir():
        return 1, "has Remote Scripts/, but no other Live library marker"
    return 0, "exists, but looks nothing like a User Library"


def candidates_for(install: LiveInstall) -> list[LibraryCandidate]:
    """The folders this Live version's ``Library.cfg`` could be pointing at.

    Two, not one, and that is the measured part (see the module docstring):
    ``<ProjectPath>`` and ``<ProjectPath>/<ProjectName>``. On the machine this
    was written against, the second one was correct and the first one held a
    decoy ``Remote Scripts`` folder that Live never read.
    """
    if install.project_path is None:
        return []

    out = [
        LibraryCandidate(
            path=install.project_path,
            origin="<ProjectPath>",
            score=0,
            evidence="",
            from_versions=(install.version_text,),
        )
    ]
    name = (install.project_name or "").strip()
    if name and name not in {".", ".."} and not Path(name).is_absolute():
        out.append(
            LibraryCandidate(
                path=install.project_path / name,
                origin="<ProjectPath>/<ProjectName>",
                score=0,
                evidence="",
                from_versions=(install.version_text,),
            )
        )
    return [_scored(candidate) for candidate in out]


def _scored(candidate: LibraryCandidate) -> LibraryCandidate:
    """Fill in ``score`` and ``evidence`` from the folder on disk."""
    score, evidence = library_score(candidate.path)
    return LibraryCandidate(
        path=candidate.path,
        origin=candidate.origin,
        score=score,
        evidence=evidence,
        from_versions=candidate.from_versions,
    )


def discover_user_libraries(installs: Sequence[LiveInstall]) -> list[LibraryCandidate]:
    """Distinct User Library candidates across every Live version, best first.

    Deduplicated by resolved path, because forty version folders normally name
    the same library and forty identical lines help nobody. Ordering is by
    evidence first and by Live version second: a newer Live is a tiebreaker, not
    an argument.
    """
    merged: dict[Path, LibraryCandidate] = {}
    order: dict[Path, int] = {}
    for position, install in enumerate(installs):
        for candidate in candidates_for(install):
            key = _resolved(candidate.path)
            existing = merged.get(key)
            if existing is None:
                merged[key] = candidate
                order[key] = position
                continue
            if install.version_text not in existing.from_versions:
                merged[key] = LibraryCandidate(
                    path=existing.path,
                    origin=existing.origin,
                    score=existing.score,
                    evidence=existing.evidence,
                    from_versions=(*existing.from_versions, install.version_text),
                )
    return sorted(merged.values(), key=lambda c: (-c.score, order[_resolved(c.path)], str(c.path)))


def _resolved(path: Path) -> Path:
    """A comparable form of a path. Never raises; a missing folder still compares."""
    try:
        return path.resolve()
    except OSError:  # pragma: no cover - only on exotic filesystems
        return path


# --------------------------------------------------------------------------- #
# What is installed already
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class InstalledScript:
    """One folder under ``Remote Scripts``, as it sits on disk right now."""

    name: str
    folder: Path
    init_file: Path | None = None
    size: int | None = None
    mtime: float | None = None
    pycache: Path | None = None
    pyc_files: tuple[Path, ...] = ()
    pyc_current: bool | None = None
    backups: tuple[Path, ...] = field(default_factory=tuple)

    @property
    def is_foreign(self) -> bool:
        """True for a folder belonging to another Remote Script, never to ours.

        Reported so that the survey is complete, and never written to by this
        file under any flag.
        """
        return self.name.lower() in FOREIGN_FOLDER_NAMES

    @property
    def loadable(self) -> bool:
        """Live only offers a folder in the dropdown if it holds ``__init__.py``."""
        return self.init_file is not None

    def stale_note(self) -> str | None:
        """What the ``__pycache__`` says about which version Live would run."""
        if self.pycache is None or not self.pyc_files:
            return None
        if self.pyc_current is None:
            return "__pycache__ present; hash-based or unreadable, cannot tell which source it is"
        if self.pyc_current:
            return "__pycache__ present and matches __init__.py"
        return (
            "__pycache__ is STALE - it was compiled from a different __init__.py. "
            "Live prefers it and reports nothing. Delete it and restart Live."
        )


def scan_remote_scripts(user_library: Path) -> list[InstalledScript]:
    """Survey ``<User Library>/Remote Scripts``: every folder, ours and not.

    Read-only. Foreign folders are reported rather than filtered out: what else
    Live loads from the same place is part of the answer, and the listing is
    also the evidence that this installer left those folders alone.
    """
    base = user_library / "Remote Scripts"
    if not base.is_dir():
        return []

    scripts: list[InstalledScript] = []
    for folder in sorted(base.iterdir(), key=lambda p: p.name.lower()):
        if not folder.is_dir():
            continue
        init_file = folder / "__init__.py"
        has_init = init_file.is_file()
        size: int | None = None
        mtime: float | None = None
        if has_init:
            stat = init_file.stat()
            size, mtime = stat.st_size, stat.st_mtime

        pycache = folder / "__pycache__"
        pyc_files = tuple(sorted(pycache.glob("*.pyc"))) if pycache.is_dir() else ()
        pyc_current: bool | None = None
        if has_init and pyc_files:
            checks = [pyc_matches_source(pyc, init_file) for pyc in pyc_files]
            if any(check is False for check in checks):
                pyc_current = False
            elif all(check is True for check in checks):
                pyc_current = True

        scripts.append(
            InstalledScript(
                name=folder.name,
                folder=folder,
                init_file=init_file if has_init else None,
                size=size,
                mtime=mtime,
                pycache=pycache if pycache.is_dir() else None,
                pyc_files=pyc_files,
                pyc_current=pyc_current,
                backups=tuple(sorted(folder.glob(f"{BACKUP_PREFIX}*"))),
            )
        )
    return scripts


def pyc_matches_source(pyc: Path, source: Path) -> bool | None:
    """Was this ``.pyc`` compiled from *this* ``__init__.py``? Exactly, not by mtime.

    A timestamp-based ``.pyc`` carries the source's mtime and size in its
    16-byte PEP 552 header, so the question has a real answer instead of a
    heuristic. Returns ``None`` for a hash-based ``.pyc`` (PEP 552 flag bit 0)
    or an unreadable file: cases where there is nothing to compare, and saying
    "stale" would be a claim rather than a reading.

    *Measured 2026-08-29* against both scripts installed on the reference
    machine: Live's ``cpython-311`` ``.pyc`` files were timestamp-based
    (flags 0) and their embedded values equalled the sources' ``st_mtime`` and
    ``st_size`` exactly.
    """
    try:
        head = pyc.read_bytes()[:16]
        stat = source.stat()
    except OSError:
        return None
    if len(head) < 16:
        return None
    flags = int.from_bytes(head[4:8], "little")
    if flags & 0b1:
        return None
    embedded_mtime = int.from_bytes(head[8:12], "little")
    embedded_size = int.from_bytes(head[12:16], "little")
    mask = 0xFFFFFFFF
    return embedded_mtime == int(stat.st_mtime) & mask and embedded_size == stat.st_size & mask


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #


def _say(text: str = "") -> None:
    """Print one line to stdout. All human output goes through here."""
    print(text)


def _choose_user_library(
    candidates: Sequence[LibraryCandidate], *, assume_yes: bool
) -> LibraryCandidate:
    """Pick the User Library, asking when there is more than one honest answer.

    The brief is literal about this: several candidates means list them and ask.
    A wrong guess here is the most expensive failure this script has, because it
    produces an installation that looks perfect and that Live never reads.
    """
    roots = ", ".join(str(root) for root in preferences_roots()) or "(no known root)"
    if not candidates:
        raise SetupError(
            "No User Library found.\n"
            "  Live records it in Preferences\\Library.cfg under <ProjectPath>, and no\n"
            f"  readable Library.cfg was found under {roots}.\n"
            "  Start Live once so it writes its preferences, or pass the folder yourself:\n"
            '    python scripts/install_script.py --user-library "D:\\path\\to\\User Library"'
        )

    # A folder Live names but that is not on disk is a stale entry from an old
    # Live version, not a place to install into. It stays visible in --list.
    present = [candidate for candidate in candidates if candidate.exists]
    if not present:
        listed = "\n".join(
            f"    {candidate.path}   ({candidate.origin})" for candidate in candidates
        )
        raise SetupError(
            "Live named a User Library, but none of the folders exist:\n"
            f"{listed}\n"
            f"  Read from Library.cfg under {roots}.\n"
            "  Start Live once so it recreates its library, or pass the real folder:\n"
            '    python scripts/install_script.py --user-library "D:\\path\\to\\User Library"'
        )
    candidates = present

    if len(candidates) == 1:
        return candidates[0]

    _say("More than one folder could be the User Library. Live reads exactly one of them,")
    _say("and from the wrong one it loads nothing and says nothing. So: which is it?")
    _say()
    for number, candidate in enumerate(candidates, start=1):
        marker = " <- best evidence" if number == 1 else ""
        _say(f"  [{number}] {candidate.describe()}{marker}")
    _say()

    if assume_yes:
        _say(f"--yes: taking [1] {candidates[0].path}")
        return candidates[0]

    if not sys.stdin.isatty():
        raise SetupError(
            "Several User Library candidates and no terminal to ask in.\n"
            "  Re-run with the folder named explicitly:\n"
            f'    python scripts/install_script.py --user-library "{candidates[0].path}"\n'
            "  or accept the best-evidence one with --yes."
        )

    while True:
        try:
            answer = input(f"Which one? [1-{len(candidates)}, Enter for 1] ").strip()
        except EOFError as exc:  # pragma: no cover - interactive only
            raise SetupError("No answer given; nothing was installed.") from exc
        if not answer:
            return candidates[0]
        if answer.isdigit() and 1 <= int(answer) <= len(candidates):
            return candidates[int(answer) - 1]
        _say(f"  Please answer with a number from 1 to {len(candidates)}, or Enter.")


def _resolve_user_library(args: argparse.Namespace) -> tuple[Path, list[LibraryCandidate]]:
    """The User Library to work in, plus every candidate that was considered."""
    if args.user_library is not None:
        path = Path(args.user_library).expanduser()
        score, evidence = library_score(path)
        if score < 0:
            raise SetupError(f"--user-library points at a folder that does not exist:\n  {path}")
        if score == 0:
            _say(f"note: {path}")
            _say(f"      {evidence}. Using it anyway because you named it explicitly.")
        return path, [
            LibraryCandidate(path=path, origin="--user-library", score=score, evidence=evidence)
        ]

    installs = find_live_installs()
    if not installs:
        raise SetupError(
            "No Live version folder found under "
            + ", ".join(str(root) for root in preferences_roots() or [Path("(none)")])
            + ".\n"
            "  Either Live has never been started on this machine, or this platform is\n"
            "  not one this script knows. Pass the folder yourself with --user-library."
        )
    candidates = discover_user_libraries(installs)
    chosen = _choose_user_library(candidates, assume_yes=args.yes)
    return chosen.path, candidates


def _validate_folder_name(name: str) -> str:
    """Refuse a folder name that would escape the target or land on another script."""
    cleaned = name.strip().strip("/\\")
    if not cleaned or cleaned in {".", ".."} or any(sep in cleaned for sep in ("/", "\\")):
        raise SetupError(
            f"--name must be a single folder name, not {name!r}. "
            "It becomes the entry in Live's Control Surface dropdown."
        )
    if cleaned.lower() in FOREIGN_FOLDER_NAMES:
        raise SetupError(
            f"Refusing to install into {cleaned!r}.\n"
            "  That folder name belongs to a different Ableton Remote Script. This\n"
            "  installer only ever writes into its own folder, under any flag - it will\n"
            "  not overwrite somebody else's script.\n"
            f"  Leave --name at {SCRIPT_FOLDER_NAME}, or choose a name of your own."
        )
    return cleaned


def _read_source(path: Path) -> bytes:
    """Load the Remote Script to install, refusing anything that is not one."""
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise SetupError(
            f"Source file not found: {path}\n"
            "  Expected the Remote Script in the checkout at "
            "live-remote-script/__init__.py.\n"
            "  Run this from a clone of the repository, or pass --source."
        ) from exc
    except OSError as exc:
        raise SetupError(f"Source file unreadable: {path} ({exc})") from exc

    if REQUIRED_SOURCE_MARKER.encode() not in data:
        raise SetupError(
            f"{path} does not contain {REQUIRED_SOURCE_MARKER!r}.\n"
            "  Live's plugin interface requires that entry point, so this file would\n"
            "  install cleanly and then do nothing at all. Refusing to copy it."
        )
    return data


def _timestamp() -> str:
    """Local time, sortable, filename-safe."""
    return time.strftime("%Y%m%d-%H%M%S")


def _backup_path(target: Path) -> Path:
    """A free ``__init__.py.bak-<timestamp>`` beside ``target``."""
    base = target.with_name(f"{BACKUP_PREFIX}{_timestamp()}")
    if not base.exists():
        return base
    for counter in range(2, 100):  # pragma: no cover - same-second reruns only
        candidate = base.with_name(f"{base.name}-{counter}")
        if not candidate.exists():
            return candidate
    raise SetupError(f"Could not find a free backup name beside {target}")


def _clear_pycache(folder: Path, *, dry_run: bool) -> list[str]:
    """Delete ``__pycache__`` in the target folder. Part of the install, not tidying.

    *Measured, and in the docs in two places* (``docs/protocol.md`` §9 and
    ``docs/limits.md`` §4): Live loads the compiled version if
    one is there, so a fresh ``__init__.py`` beside a stale ``__pycache__``
    means the change did not happen, and Live reports nothing.
    """
    pycache = folder / "__pycache__"
    if not pycache.is_dir():
        return []
    names = sorted(item.name for item in pycache.iterdir())
    if dry_run:
        return names
    try:
        shutil.rmtree(pycache)
    except OSError as exc:
        raise SetupError(
            f"Could not delete {pycache}\n"
            f"  {exc.__class__.__name__}: {exc}\n"
            "  This is not cosmetic: Live prefers the compiled version and would keep\n"
            "  running the old script without saying so. Close Live and delete it by hand:\n"
            f'    Remove-Item -Recurse -Force "{pycache}"'
        ) from exc
    return names


def _note_foreign_scripts(scripts: Sequence[InstalledScript]) -> None:
    """Report another Remote Script found beside ours, and that it was left alone."""
    foreign = [script for script in scripts if script.is_foreign]
    if not foreign:
        return
    _say()
    _say("Another Remote Script sits in the same Remote Scripts folder, untouched:")
    for script in foreign:
        _say(f"  {script.folder}")
    _say(f"  Its own folder, its own port ({FOREIGN_PORT}, against {OUR_PORT} here). Nothing")
    _say("  above touches it. Whether two Control Surfaces can each run their own socket")
    _say("  server side by side is UNVERIFIED - nobody has measured it. If it turns out")
    _say("  not to hold, select one or the other in Preferences rather than both.")


def _print_manual_steps(target_folder: Path, name: str) -> None:
    """User instructions printed every time, covering manual Live configuration."""
    _say()
    _say("=" * 78)
    _say("NOT DONE YET. Live needs two things from you, and neither is optional.")
    _say("=" * 78)
    _say()
    _say(f"  1. Preferences -> Link, Tempo & MIDI -> Control Surface -> {name}")
    _say()
    _say(f"     The folder name IS the dropdown entry. It will read {name!r} because")
    _say(f"     the folder is called {name!r}:")
    _say(f"       {target_folder}")
    _say("     Pick it in any free slot. If it is not in the list, the script is in the")
    _say("     wrong folder - Live shows no error for that, it simply offers nothing.")
    _say()
    _say("  2. Quit Live completely and start it again.")
    _say()
    _say("     Remote Scripts load only at startup. There is no reload, no rescan and no")
    _say("     message. Closing the set is not enough; quit the application.")
    _say()
    _say("Then check it from this checkout:")
    _say()
    _say("  python -m ableton_maestro.client ping")
    _say("  python scripts/inventory.py")
    _say()


def do_install(args: argparse.Namespace) -> int:
    """Copy the script in, back up what was there, clear ``__pycache__``."""
    name = _validate_folder_name(args.name)
    source = Path(args.source).expanduser()
    data = _read_source(source)

    user_library, candidates = _resolve_user_library(args)
    target_folder = user_library / "Remote Scripts" / name
    target = target_folder / "__init__.py"
    existing = scan_remote_scripts(user_library)

    _say()
    _say(f"User Library : {user_library}")
    if candidates:
        _say(f"               {candidates[0].evidence}")
    _say(f"Source       : {source}  ({len(data):,} bytes)")
    _say(f"Target       : {target}")
    _say()

    chosen = _resolved(user_library)
    decoys = [
        (candidate, scan_remote_scripts(candidate.path))
        for candidate in candidates
        if candidate.exists and _resolved(candidate.path) != chosen
    ]
    decoys = [(candidate, scripts) for candidate, scripts in decoys if scripts]
    if decoys:
        _say("Remote Scripts also sit in folders Live is NOT reading:")
        for candidate, scripts in decoys:
            listed = ", ".join(script.name for script in scripts)
            _say(f"  {candidate.path / 'Remote Scripts'}  [{listed}]")
            _say(f"    {candidate.evidence}")
        _say("  Those are invisible to Live: no dropdown entry, no error, no log line.")
        _say("  That is exactly the trap this installer exists to avoid. Nothing above is")
        _say("  touched or deleted - if one of those folders is the one you meant, stop and")
        _say("  re-run with --user-library pointing at it.")
        _say()

    unchanged = target.is_file() and target.read_bytes() == data
    if args.dry_run:
        _say("--dry-run: nothing below is actually done.")
        _say()
        if not target_folder.is_dir():
            _say(f"  would create   {target_folder}")
        if target.is_file():
            if unchanged:
                _say(f"  identical      {target}  (no backup needed, no copy needed)")
            else:
                _say(f"  would back up  {target}  ->  {BACKUP_PREFIX}<timestamp>")
                _say(f"  would copy     {source}  ->  {target}")
        else:
            _say(f"  would copy     {source}  ->  {target}")
        removed = _clear_pycache(target_folder, dry_run=True)
        if removed:
            _say(f"  would delete   {target_folder / '__pycache__'}  ({len(removed)} file(s))")
        else:
            _say(f"  no __pycache__ in {target_folder}")
        _note_foreign_scripts(existing)
        _print_manual_steps(target_folder, name)
        return 0

    target_folder.mkdir(parents=True, exist_ok=True)

    if target.is_file() and not unchanged:
        backup = _backup_path(target)
        shutil.copy2(target, backup)
        _say(f"backed up      {target.name}  ->  {backup.name}")

    if unchanged:
        _say("already current: the installed file is byte-identical, nothing copied.")
    else:
        shutil.copy2(source, target)
        _say(f"copied         {len(data):,} bytes  ->  {target}")

    removed = _clear_pycache(target_folder, dry_run=False)
    if removed:
        _say(f"deleted        __pycache__ ({len(removed)} file(s): {', '.join(removed)})")
        _say("               Without this Live loads the stale compiled version and the")
        _say("               change appears not to have happened. Measured.")
    else:
        _say("no __pycache__ to delete.")

    _note_foreign_scripts(existing)
    _print_manual_steps(target_folder, name)
    return 0


def do_uninstall(args: argparse.Namespace) -> int:
    """Make Live stop loading the script, without deleting anything irreversibly.

    ``__init__.py`` is *renamed* to a timestamped backup rather than removed: a
    folder without ``__init__.py`` no longer appears in Live's dropdown, which
    is what uninstalling means here, and nothing the user might want back is
    destroyed. Removing the folder outright stays a deliberate manual step, and
    the exact command is printed.
    """
    name = _validate_folder_name(args.name)
    user_library, _ = _resolve_user_library(args)
    target_folder = user_library / "Remote Scripts" / name
    target = target_folder / "__init__.py"

    _say()
    _say(f"User Library : {user_library}")
    _say(f"Target       : {target_folder}")
    _say()

    if not target_folder.is_dir():
        _say(f"Nothing to uninstall: {target_folder} does not exist.")
        return 0

    if args.dry_run:
        _say("--dry-run: nothing below is actually done.")
        if target.is_file():
            _say(f"  would rename   {target.name}  ->  {BACKUP_PREFIX}<timestamp>")
        else:
            _say(f"  no {target.name} in that folder")
        removed = _clear_pycache(target_folder, dry_run=True)
        if removed:
            _say(f"  would delete   __pycache__  ({len(removed)} file(s))")
        return 0

    if target.is_file():
        backup = _backup_path(target)
        target.rename(backup)
        _say(f"renamed        {target.name}  ->  {backup.name}")
        _say("               Live no longer offers this folder in the Control Surface")
        _say("               dropdown. The file itself is kept, not deleted.")
    else:
        _say(f"no {target.name} in {target_folder} - it was already not loadable.")

    removed = _clear_pycache(target_folder, dry_run=False)
    if removed:
        _say(f"deleted        __pycache__ ({len(removed)} file(s))")

    _say()
    _say("Still to do by hand:")
    _say(f"  1. Preferences -> Link, Tempo & MIDI -> Control Surface: set the {name} slot")
    _say("     back to None.")
    _say("  2. Restart Live completely.")
    _say()
    remaining = sorted(item.name for item in target_folder.iterdir())
    if remaining:
        _say(f"The folder is kept because it still holds: {', '.join(remaining)}")
        _say("To remove it entirely, once you are sure you want the backups gone:")
        _say(f'  Remove-Item -Recurse -Force "{target_folder}"')
    else:
        target_folder.rmdir()
        _say(f"removed the now-empty folder {target_folder}")
    return 0


def do_diff(args: argparse.Namespace) -> int:
    """Unified diff of the repo's script against the installed one.

    Exit code is the answer: 0 identical, 1 different or not installed. That
    makes it usable as a check ("is Live's copy the one in this checkout?")
    and not only as something to read.
    """
    name = _validate_folder_name(args.name)
    source = Path(args.source).expanduser()
    user_library, _ = _resolve_user_library(args)
    target = user_library / "Remote Scripts" / name / "__init__.py"

    _say(f"repo      : {source}")
    _say(f"installed : {target}")
    _say()

    if not target.is_file():
        _say("Nothing is installed at that path.")
        _say("  python scripts/install_script.py")
        return 1

    source_text = source.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    target_text = target.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            target_text,
            source_text,
            fromfile=f"installed  {target}",
            tofile=f"repo       {source}",
            n=args.context,
        )
    )
    if not diff:
        _say("Identical. Live is running this checkout's script -")
        _say("assuming Live was restarted since it was copied; check __pycache__ with")
        _say("  python scripts/inventory.py")
        return 0

    sys.stdout.writelines(diff)
    added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    dropped = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
    _say()
    _say(f"Different: {added} line(s) in the repo that are not installed, {dropped} the other way.")
    _say("Install the repo version with:  python scripts/install_script.py")
    return 1


def do_list(args: argparse.Namespace) -> int:
    """Show the discovery: Live versions, Library.cfg readings, candidates, scripts."""
    installs = find_live_installs()
    _say()
    _say(f"Live version folders under {', '.join(str(r) for r in preferences_roots()) or '(none)'}")
    if not installs:
        _say("  (none found)")
    for install in installs[: args.limit]:
        if install.error:
            _say(f"  {install.version_text:<20} {install.error}")
        else:
            name = install.project_name or "(no <ProjectName>)"
            _say(f"  {install.version_text:<20} <ProjectPath> {install.project_path}")
            _say(f"  {'':<20} <ProjectName> {name}")
    if len(installs) > args.limit:
        _say(f"  ... and {len(installs) - args.limit} more (raise --limit to see them)")

    _say()
    _say("User Library candidates, best evidence first")
    candidates = discover_user_libraries(installs)
    if not candidates:
        _say("  (none - no readable Library.cfg)")
    for number, candidate in enumerate(candidates, start=1):
        _say(f"  [{number}] {candidate.describe()}")
        for script in scan_remote_scripts(candidate.path):
            flags = []
            if not script.loadable:
                flags.append("no __init__.py - Live ignores it")
            if script.is_foreign:
                flags.append(f"not ours; port {FOREIGN_PORT}")
            note = script.stale_note()
            if note:
                flags.append(note)
            _say(
                f"      Remote Scripts/{script.name}" + (f"  [{'; '.join(flags)}]" if flags else "")
            )
    _say()
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/install_script.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Install the Ableton Maestro Remote Script into Live's User Library.\n"
            "\n"
            "The User Library is read from Live's own Preferences\\Library.cfg, never\n"
            "guessed. The folder name under Remote Scripts becomes the entry in Live's\n"
            "Control Surface dropdown, and from the wrong folder Live loads nothing and\n"
            "says nothing at all."
        ),
        epilog=(
            "Exit codes: 0 fine, 1 something needs fixing (--diff: the files differ),\n"
            "2 usage error, 130 interrupted."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--uninstall",
        action="store_true",
        help="stop Live loading it: rename __init__.py to a timestamped backup, "
        "clear __pycache__. Never touches another script's folder.",
    )
    mode.add_argument(
        "--diff",
        action="store_true",
        help="unified diff of the repo's script against the installed one. "
        "Exits 0 when identical, 1 when it differs or nothing is installed.",
    )
    mode.add_argument(
        "--list",
        action="store_true",
        help="show what discovery found - Live versions, Library.cfg readings, "
        "User Library candidates, installed scripts - and change nothing.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="say exactly what would happen and change nothing. Works with --uninstall.",
    )
    parser.add_argument(
        "--user-library",
        metavar="PATH",
        help="skip discovery and use this folder as the User Library.",
    )
    parser.add_argument(
        "--name",
        default=SCRIPT_FOLDER_NAME,
        metavar="FOLDER",
        help=f"folder name under Remote Scripts (default: {SCRIPT_FOLDER_NAME}). "
        "This is what Live shows in the Control Surface dropdown. "
        "Names belonging to other Remote Scripts are refused.",
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        metavar="PATH",
        help="the __init__.py to install (default: live-remote-script/__init__.py "
        "in this checkout).",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="when several User Library candidates are found, take the best-evidence "
        "one instead of asking.",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=3,
        metavar="N",
        help="lines of context for --diff (default 3).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        metavar="N",
        help="how many Live version folders --list prints (default 8; there can be 40).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        0 on success, 1 when something needs fixing (including ``--diff``
        reporting a difference), 2 on misuse, 130 on Ctrl-C.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover - not a real stream
            pass

    args = _build_parser().parse_args(argv)
    if args.dry_run and (args.diff or args.list):
        print("--dry-run is meaningless with --diff or --list; both already change nothing.")
        return 2

    try:
        if args.list:
            return do_list(args)
        if args.diff:
            return do_diff(args)
        if args.uninstall:
            return do_uninstall(args)
        return do_install(args)
    except SetupError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("aborted; nothing was changed.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
