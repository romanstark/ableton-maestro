#!/usr/bin/env python
"""Reconcile the YAML path catalog against an active Ableton Live session.

Discovers, verifies, and updates catalog rows using ``lom_describe`` and ``lom_get``
across specimen objects in the open Live session.

Catalog update policies:
- ``no_such_path``: property is absent on confirmed specimen -> proposed for catalog deletion.
- ``live_error``: property exists but Live rejects access -> marked as ``status: broken``.
- ``index_out_of_range`` / missing container: inconclusive -> row left unchanged as ``untested``.

Evidence grades:
- Determined probes (exact specimen matches): permit status flips and deletions.
- Derived probes (polymorphic collections like devices[i]): permit additions and confirmations only.

Usage:
    python scripts/sync_catalog.py                    # reconcile and report
    python scripts/sync_catalog.py --area track       # one area only
    python scripts/sync_catalog.py --record run.json  # keep the raw answers
    python scripts/sync_catalog.py --replay run.json  # reconcile them again, offline
    python scripts/sync_catalog.py --write            # apply everything
    python scripts/sync_catalog.py --add --status     # apply only those two
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import textwrap
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# scripts/ is not a package and this file is run as a path, not as a module. A
# checkout that has not been `pip install -e .`-ed still has src/ next door.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# Everything below is imported after the path bootstrap above, which is why it is
# not at the very top of the file.
#
# The line-level catalog editor is `probe_paths.py`'s, imported rather than copied.
# Both tools edit the same 22 000 lines of hand-written YAML under the same two
# rules: never round-trip through yaml.dump, never touch a byte that did not have
# to change. Two copies of that logic would drift apart on the first change to
# either. What this file adds is the two operations probe_paths has no use for:
# inserting a row and removing one.
from probe_paths import (
    CatalogWriteError,
    RowLocation,
    _append_doc_note,
    _newline_of,
    _set_status,
    _split_lines,
    _wrap,
)

from ableton_maestro.client import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    AbletonClient,
    AbletonCommandError,
    AbletonConnectionError,
    AbletonError,
)
from ableton_maestro.models import Access, Kind, PathStatus, Unit
from ableton_maestro.registry import (
    DEFAULT_CATALOG_DIR,
    CatalogError,
    Registry,
    area_of,
)
from ableton_maestro.spec import ParamSpec, PathSpec, placeholders_in

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: One live object per catalogued class. Measured 2026-08-29 against Live 12.4.5:
#: every one of these resolved in the reference set. They are shipped as defaults,
#: not as facts about anybody else's set, because a different project has different tracks
#: at different indices, which is what ``--specimens`` is for.
#:
#: The key is the class the specimen is *meant* to be. The run reports the class
#: Live actually answered with, so a stale map says so instead of quietly
#: reconciling a Compressor against the rack rows.
DEFAULT_SPECIMENS: dict[str, str] = {
    "Song": "song",
    "Application": "app",
    "Application.View": "app.view",
    "Song.View": "song.view",
    "Track": "song.tracks[10]",
    "MixerDevice": "song.tracks[10].mixer_device",
    "DeviceParameter": "song.tracks[10].mixer_device.volume",
    "ClipSlot": "song.tracks[11].clip_slots[0]",
    "Clip": "song.tracks[11].clip_slots[0].clip",
    "Device": "song.tracks[10].devices[3]",
    "RackDevice": "song.tracks[5].devices[0]",
    "Chain": "song.tracks[10].devices[0].chains[0]",
    "Scene": "song.scenes[0]",
    "CuePoint": "song.cue_points[0]",
    "BrowserItem": "app.browser.instruments",
    "GroovePool": "song.groove_pool",
    "ReturnTrack": "song.return_tracks[0]",
    "MasterTrack": "song.master_track",
    "DrumPad": "song.tracks[4].devices[1].chains[0].devices[0].drum_pads[36]",
}

#: The Remote Script this run reads ``METHOD_ALLOWLIST`` and ``_DESCRIBE_SKIP``
#: out of. It is the repository copy, which is not necessarily the code Live
#: is running: Live loads a Remote Script only at startup (protocol §9). The two
#: are compared and a disagreement is reported rather than resolved.
DEFAULT_SCRIPT = _REPO_ROOT / "live-remote-script" / "__init__.py"

#: Pause between round trips. Nothing here writes, so this is politeness towards
#: Live's main thread rather than the hazard ``probe_paths`` paces against, but a
#: describe does run on that thread and a full catalog run is a few hundred of them.
DEFAULT_PACE = 0.05

#: The longer pause for anything that walks through a device. Loading plugins in
#: rapid succession can crash Live (measured, CONTRIBUTING.md *Warnings*).
#: Reading is not loading, so this is caution and not a measured requirement.
DEFAULT_DEVICE_PACE = 0.15

#: Column the folded ``doc`` blocks in the catalog wrap at. Must match
#: ``probe_paths._DOC_WIDTH``. The two write into the same blocks.
DOC_WIDTH = 96

CODE_OK = 0
CODE_USAGE = 1
CODE_WRITE_FAILED = 3

_BANNER = (
    "sync_catalog: reconciles the catalog against a running Live.\n"
    "  READ-ONLY against Live -- it sends lom_describe and lom_get, nothing else.\n"
    "  Writes to catalog/*.yaml only with --write / --add / --delete / --status.\n"
    "  The protocol is strictly serial (docs/protocol.md section 3): do not run\n"
    "  this while another client is talking to the script.\n"
)

#: A member name this tool is willing to turn into a catalog row. The id has to
#: survive ``tests/test_catalog.py::test_every_row_has_a_string_id_shaped_as_area_dot_name``
#: and the path has to survive the resolver's grammar, so anything else is
#: reported and not written.
_MEMBER_RE = re.compile(r"\A[a-z][a-z0-9_]*\Z")

#: What a measurement note has to name besides the date: the build the reading was
#: true of. Deliberately the same expression
#: ``tests/test_catalog.py::has_measurement_note`` applies, so this tool checks
#: *before* it writes what the test suite would check afterwards: a status it
#: cannot pair with a version is refused here rather than discovered by a failing
#: test, or worse, believed.
_LIVE_VERSION_RE = re.compile(r"\bLive\s*\d", re.IGNORECASE)

#: Catalog template segments: ``name``, ``name[7]``, ``name[{index}]``, ``{root}``.
_SEGMENT_RE = re.compile(
    r"\A(?:"
    r"\{(?P<name_ph>[A-Za-z_][A-Za-z0-9_]*)\}"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\[(?:(?P<index>0|[1-9][0-9]*)|\{(?P<index_ph>[A-Za-z_][A-Za-z0-9_]*)\})\])?"
    r")\Z"
)

#: How ``lom_describe`` names a value's type versus how the catalog names a kind.
#: ``null`` has no catalog member and never will: ``Kind`` describes what a value
#: looks like on the wire and "there was nothing there" is not a shape.
_KIND_FROM_WIRE: dict[str, Kind] = {
    "bool": Kind.BOOL,
    "int": Kind.INT,
    "float": Kind.FLOAT,
    "string": Kind.STR,
    "list": Kind.LIST,
    "object": Kind.OBJECT,
}

#: Verdicts a row can come out of the reconciliation with.
CONFIRMED = "confirmed"
#: Live class families where one concrete class is NOT evidence about its siblings.
#: A container shape can resolve to any member of a family depending on what the
#: user loaded, so a member missing on the class we happened to probe says nothing
#: about the row -- the row may well be written for the sibling.
#:
#: Measured 2026-08-30 against Live 12.4.5, and this table exists because of it:
#: ``chain.choke_group`` and ``chain.out_note`` were proposed for deletion after a
#: probe on ``song.tracks[10].devices[0].chains[0]``, which is a plain ``Chain``
#: inside an Audio Effect Rack. Both members exist on ``DrumChain`` -- read back as
#: 1 and 60 from a drum rack's pad chain at the same path shape. Deleting them
#: would have removed two real rows on the strength of the wrong specimen.
POLYMORPHIC_FAMILIES: dict[str, frozenset[str]] = {
    "Chain": frozenset({"Chain", "DrumChain"}),
    "DrumChain": frozenset({"Chain", "DrumChain"}),
    "Device": frozenset({"Device", "RackDevice", "PluginDevice", "MaxDevice"}),
    "RackDevice": frozenset(
        {
            "RackDevice",
            "InstrumentGroupDevice",
            "DrumGroupDevice",
            "AudioEffectGroupDevice",
            "MidiEffectGroupDevice",
        }
    ),
}

#: Live's own wording when the object on the path is ``None`` rather than the member
#: being absent. Measured 2026-08-30: ``clip.groove`` on a clip with no groove gives
#: "NoneType has no attribute 'name'".
_NULL_CONTAINER = re.compile(r"NoneType\S* has no attribute")

#: Live saying the OBJECT is the wrong kind, not that the member is broken.
#: Measured 2026-08-30 against Live 12.4.5: probing a MIDI clip for clip.warping
#: answers live_error "Warping is only available for Audio Clips" -- a statement
#: about the specimen, not about the row. Read as a refusal it would have demoted
#: twelve rows, three of which (warping, warp_mode, pitch_coarse) had been read,
#: written, read back and restored on an audio clip the same day.
_TYPE_PRECONDITION = re.compile(r"only (?:available )?(?:for|on|in) ", re.IGNORECASE)


MISSING = "missing"        # answered no_such_path -> delete
REFUSED = "refused"        # answered live_error -> stays, broken
NOT_REACHED = "not_reached"
MISMATCHED = "mismatched"  # it is there, but not the kind of thing the row says


# --------------------------------------------------------------------------- #
# The catalog path template grammar
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Segment:
    """One segment of a catalog path template or of a concrete path.

    Four shapes exist and they are not interchangeable (``docs/protocol.md`` §6):
    a plain ``name``, a ``name[7]`` with a literal index, a ``name[{track}]`` with
    an index placeholder, and a bare ``{root}`` standing for a segment *name*.
    The last one is why the browser's roots are reachable at all: some LOM
    collections are addressed by attribute rather than by number.
    """

    name: str | None
    index: int | None = None
    index_placeholder: str | None = None
    name_placeholder: str | None = None

    @property
    def is_name_placeholder(self) -> bool:
        """Return True for a bare ``{root}``, whose segment name is not fixed."""
        return self.name_placeholder is not None

    @property
    def has_index(self) -> bool:
        """True when this segment selects one element out of a collection."""
        return self.index is not None or self.index_placeholder is not None

    def render(self, *, fill: int = 0) -> str:
        """Render as a concrete segment.

        ``fill`` is substituted for a placeholder.
        """
        if self.is_name_placeholder:
            raise ValueError(f"segment name {{{self.name_placeholder}}} cannot be filled in")
        if self.index is not None:
            return f"{self.name}[{self.index}]"
        if self.index_placeholder is not None:
            return f"{self.name}[{fill}]"
        return str(self.name)

    def shape(self) -> str:
        """The index-free, placeholder-free form used to compare two paths."""
        if self.is_name_placeholder:
            return "{}"
        return f"{self.name}[]" if self.has_index else str(self.name)


class TemplateError(ValueError):
    """A path template did not fit the catalog grammar."""


def split_template(path: str) -> list[Segment]:
    """Split a catalog ``path``, template or concrete, into segments."""
    segments: list[Segment] = []
    for raw in path.split("."):
        match = _SEGMENT_RE.match(raw)
        if match is None:
            raise TemplateError(f"{path!r}: segment {raw!r} is not a legal path segment")
        if match.group("name_ph"):
            segments.append(Segment(name=None, name_placeholder=match.group("name_ph")))
            continue
        index = match.group("index")
        segments.append(
            Segment(
                name=match.group("name"),
                index=int(index) if index is not None else None,
                index_placeholder=match.group("index_ph"),
            )
        )
    return segments


def shape_of(segments: Sequence[Segment]) -> str:
    """The comparable shape of a path.

    ``song.tracks[10]`` becomes ``song.tracks[]``.
    """
    return ".".join(segment.shape() for segment in segments)


def unparse(segments: Sequence[Segment]) -> str:
    """Put a template back together, placeholders and all."""
    parts: list[str] = []
    for segment in segments:
        if segment.is_name_placeholder:
            parts.append("{" + str(segment.name_placeholder) + "}")
        elif segment.index is not None:
            parts.append(f"{segment.name}[{segment.index}]")
        elif segment.index_placeholder is not None:
            parts.append(f"{segment.name}[{{{segment.index_placeholder}}}]")
        else:
            parts.append(str(segment.name))
    return ".".join(parts)


@dataclass(frozen=True)
class RowTarget:
    """What one catalog row actually addresses: an object, and a member of it.

    For a property row the container is the path minus its last segment and the
    member is that segment's name: ``song.tracks[{t}].mixer_device.sends[{s}]``
    addresses ``sends`` on the mixer, not an anonymous element. For an
    ``access: [call]`` row the whole path *is* the container and the member is the
    method name, which is why the two carry a ``kind`` and are never mixed up.
    """

    container: str
    member: str
    is_method: bool


def target_of(spec: PathSpec) -> RowTarget | None:
    """The container and member a row addresses, or ``None`` when it has none.

    ``None`` is returned for a row whose path *is* a root object (``song``) with no
    call attached, and for one whose last segment is itself a name placeholder
    (``app.browser.{root}``): there is no member to ask about in either case.
    """
    if Access.CALL in spec.access:
        method = (spec.method or "").strip()
        if not method:
            return None
        return RowTarget(container=spec.path, member=method, is_method=True)

    segments = split_template(spec.path)
    if len(segments) < 2:
        return None
    last = segments[-1]
    if last.is_name_placeholder or last.name is None:
        return None
    return RowTarget(
        container=unparse(segments[:-1]), member=last.name, is_method=False
    )


# --------------------------------------------------------------------------- #
# Specimens
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Specimen:
    """One concrete live object standing in for a whole LOM class.

    ``area`` is optional and only ever used to file a *proposed* row. Several
    catalog areas can share one container shape: ``device``, ``rack``, ``simpler``
    and ``eq8`` all live on ``song.tracks[i].devices[j]``. The majority area is
    then a guess, and this is how a user replaces the guess with a decision.
    """

    class_name: str
    path: str
    area: str | None = None

    @property
    def segments(self) -> list[Segment]:
        return split_template(self.path)

    @property
    def shape(self) -> str:
        return shape_of(self.segments)


def load_specimens(file: Path | None) -> dict[str, Specimen]:
    """The specimen map: the shipped defaults, overlaid with ``file`` if given.

    The file is JSON, ``{"<class>": "<path>"}`` or
    ``{"<class>": {"path": "...", "area": "..."}}``. The two forms may be mixed. A
    value of ``null`` removes a shipped default, which is the honest way to say
    "this set has no rack", which beats pointing the specimen at something that is
    not one.
    """
    entries: dict[str, Any] = dict(DEFAULT_SPECIMENS)
    if file is not None:
        try:
            raw = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"--specimens {file}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"--specimens {file}: top level must be an object of class -> path")
        entries.update(raw)

    specimens: dict[str, Specimen] = {}
    for class_name, value in entries.items():
        if value is None:
            continue
        if isinstance(value, str):
            path, area = value, None
        elif isinstance(value, dict) and isinstance(value.get("path"), str):
            path = value["path"]
            area = value.get("area") if isinstance(value.get("area"), str) else None
        else:
            # ValueError even for a type fault, as everywhere else in this project:
            # one exception type keeps "was this refused before anything reached
            # Live?" answerable, and main() catches exactly it.
            raise ValueError(  # noqa: TRY004
                f"specimen {class_name!r}: expected a path string or "
                f'{{"path": "...", "area": "..."}}, got {value!r}'
            )
        try:
            split_template(path)
        except TemplateError as exc:
            raise ValueError(f"specimen {class_name!r}: {exc}") from exc
        specimens[class_name] = Specimen(class_name=class_name, path=path, area=area)
    return specimens


def _prefix_matches(specimen: Sequence[Segment], template: Sequence[Segment]) -> bool:
    """True when a specimen path names the same place as a template prefix.

    A template's ``{root}`` matches any plain concrete segment: that is the whole
    point of a name placeholder, and the specimen is what decides which root the
    run uses. Everything else must agree on both the name and on whether the
    segment carries an index: ``devices`` and ``devices[0]`` are different places.
    """
    if len(specimen) != len(template):
        return False
    for concrete, wanted in zip(specimen, template):
        if concrete.is_name_placeholder:
            return False
        if wanted.is_name_placeholder:
            if concrete.has_index:
                return False
            continue
        if concrete.name != wanted.name or concrete.has_index != wanted.has_index:
            return False
    return True


@dataclass(frozen=True)
class Resolution:
    """A container template resolved onto one concrete path in the open set."""

    container: str
    probe_path: str
    specimen: Specimen
    #: Segments this tool had to invent an index for. Empty means the specimen
    #: fixed every index in the path, which is what makes a probe *determined*.
    filled: tuple[str, ...]


def resolve_container(
    container: str, specimens: Mapping[str, Specimen], *, fill: int = 0
) -> tuple[list[Resolution], str | None]:
    """Resolve a container template to every probe path the specimens support.

    Returns ``(resolutions, reason_if_none)``. The longest specimen prefix wins.
    When several specimens tie at that length the container is *polymorphic* and
    all of them are probed, because the union of what they report is the only
    honest answer about a collection whose element class varies with the index.
    """
    try:
        segments = split_template(container)
    except TemplateError as exc:
        return [], str(exc)

    for cut in range(len(segments), 0, -1):
        head = segments[:cut]
        matched = [
            specimen
            for specimen in specimens.values()
            if _prefix_matches(specimen.segments, head)
        ]
        if not matched:
            continue

        tail = segments[cut:]
        if any(segment.is_name_placeholder for segment in tail):
            return [], (
                f"no specimen reaches {shape_of(segments)}: the segment name "
                f"{{{next(s.name_placeholder for s in tail if s.is_name_placeholder)}}} "
                "is a placeholder and nothing in the specimen map fills it"
            )
        filled = tuple(
            str(segment.name) for segment in tail if segment.index_placeholder is not None
        )
        suffix = "".join("." + segment.render(fill=fill) for segment in tail)
        return (
            [
                Resolution(
                    container=container,
                    probe_path=specimen.path + suffix,
                    specimen=specimen,
                    filled=filled,
                )
                for specimen in sorted(matched, key=lambda s: s.class_name)
            ],
            None,
        )

    return [], f"no specimen covers {shape_of(segments)} or any prefix of it"


# --------------------------------------------------------------------------- #
# Reading the Remote Script's own rules
# --------------------------------------------------------------------------- #


def _literal_string_set(script: Path, name: str) -> frozenset[str]:
    """Read a module-level ``frozenset([...])`` or ``[...]`` of strings.

    The Remote Script cannot be imported out here. It does
    ``from _Framework.ControlSurface import ControlSurface`` on line 77 and that
    module only exists inside Live, so its constants are *parsed* with ``ast``,
    never executed and never copied into this file. A copy would be a second
    source of truth for a rule whose whole design is that it has exactly one
    (protocol §6).
    """
    try:
        tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"cannot read {name} from {script}: {exc}") from exc

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            continue
        value: ast.expr = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in {"frozenset", "set"}
        ):
            if not value.args:
                return frozenset()
            value = value.args[0]
        try:
            literal = ast.literal_eval(value)
        except ValueError as exc:
            raise ValueError(f"{name} in {script} is not a literal: {exc}") from exc
        if not isinstance(literal, (list, tuple, set, frozenset)):
            raise ValueError(f"{name} in {script} is not a collection")  # noqa: TRY004
        return frozenset(str(item) for item in literal)

    raise ValueError(f"{script} has no module-level {name}")


# --------------------------------------------------------------------------- #
# Talking to Live (or to a recording of it)
# --------------------------------------------------------------------------- #


@dataclass
class Answer:
    """One handler's reply, success or structured failure.

    Never an exception.
    """

    ok: bool
    result: dict[str, Any] = field(default_factory=dict)
    code: str = ""
    message: str = ""

    def as_record(self) -> dict[str, Any]:
        if self.ok:
            return {"ok": True, "result": self.result}
        return {"ok": False, "code": self.code, "message": self.message}

    @classmethod
    def from_record(cls, raw: Any) -> Answer:
        if not isinstance(raw, dict):
            return cls(ok=False, code="not_recorded", message="malformed recording entry")
        if raw.get("ok"):
            result = raw.get("result")
            return cls(ok=True, result=dict(result) if isinstance(result, dict) else {})
        return cls(
            ok=False,
            code=str(raw.get("code") or "unspecified"),
            message=str(raw.get("message") or ""),
        )


class Reader:
    """Read-only access to Live, or to a recording of an earlier run.

    Every answer is memoised: a container is described once however many rows hang
    off it, and the same confirming ``lom_get`` is never sent twice. On a strictly
    serial socket that is not an optimisation, it is the difference between a run
    that takes twenty seconds and one that takes three minutes.
    """

    def __init__(
        self,
        client: AbletonClient | None,
        *,
        replay: Mapping[str, Any] | None = None,
        pace: float = DEFAULT_PACE,
        device_pace: float = DEFAULT_DEVICE_PACE,
    ) -> None:
        self._client = client
        self._replay = replay
        self._pace = pace
        self._device_pace = device_pace
        self.describes: dict[str, Answer] = {}
        self.gets: dict[str, Answer] = {}
        self.script_info: dict[str, Any] = {}
        self.round_trips = 0

    # ------------------------------------------------------------- one request
    def _recorded(self, kind: str, path: str) -> Answer:
        table = (self._replay or {}).get(kind)
        entry = table.get(path) if isinstance(table, Mapping) else None
        if entry is None:
            return Answer(
                ok=False,
                code="not_recorded",
                message=f"{kind} {path!r} is not in this recording",
            )
        return Answer.from_record(entry)

    def _send(self, kind: str, path: str) -> Answer:
        if self._replay is not None:
            return self._recorded(kind, path)
        client = self._client
        if client is None:  # pragma: no cover - guarded by the constructor's callers
            raise AbletonConnectionError("no client and no recording to read from")
        self.round_trips += 1
        try:
            if kind == "describe":
                return Answer(ok=True, result=client.describe(path))
            return Answer(ok=True, result=client.get(path))
        except AbletonCommandError as exc:
            return Answer(ok=False, code=exc.code, message=exc.message)
        finally:
            self._pause(path)

    def _pause(self, path: str) -> None:
        delay = self._device_pace if ".devices[" in path else self._pace
        if delay > 0:
            time.sleep(delay)

    # ------------------------------------------------------------- the handlers
    def describe(self, path: str) -> Answer:
        """Describe one object at depth 1, its full inventory (§5.6)."""
        if path not in self.describes:
            self.describes[path] = self._send("describe", path)
        return self.describes[path]

    def get(self, path: str) -> Answer:
        """Get one exact path, for its authoritative error code (§5.3)."""
        if path not in self.gets:
            self.gets[path] = self._send("get", path)
        return self.gets[path]

    def handshake(self) -> dict[str, Any]:
        """``script_info``, or whatever the recording kept of it (§5.2)."""
        if self._replay is not None:
            info = self._replay.get("script_info")
            self.script_info = dict(info) if isinstance(info, Mapping) else {}
            return self.script_info
        client = self._client
        if client is None:  # pragma: no cover - guarded by the constructor's callers
            raise AbletonConnectionError("no client and no recording to read from")
        self.round_trips += 1
        self.script_info = client.script_info()
        return self.script_info

    def recording(self) -> dict[str, Any]:
        """Everything this run read, in the shape ``--replay`` expects back."""
        return {
            "recorded": datetime.now().astimezone().isoformat(timespec="seconds"),
            "script_info": self.script_info,
            "describe": {path: answer.as_record() for path, answer in self.describes.items()},
            "get": {path: answer.as_record() for path, answer in self.gets.items()},
        }


# --------------------------------------------------------------------------- #
# What one describe told us about one object
# --------------------------------------------------------------------------- #


@dataclass
class Probe:
    """One ``lom_describe`` of one concrete object, unpacked."""

    resolution: Resolution
    answer: Answer
    live_class: str = ""
    truncated: bool = False
    #: name -> the describe entry, for names that read cleanly.
    readable: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: name -> the describe entry, for names Live listed and then refused.
    unreadable: dict[str, dict[str, Any]] = field(default_factory=dict)
    methods: set[str] = field(default_factory=set)
    allowed_methods: set[str] = field(default_factory=set)
    #: True when the script reported ``allowed_methods`` at all. An older script
    #: does not, and the on-disk allowlist is then the only thing to go on.
    reports_allowed: bool = False

    @property
    def probe_path(self) -> str:
        return self.resolution.probe_path

    @property
    def filled_indices(self) -> tuple[str, ...]:
        """Segments whose index this tool chose rather than a specimen.

        An empty tuple is what makes a probe *determined*: index 0 is a guess
        about the open set, and a guess must never become evidence about the
        catalog (``docs/catalog.md``, "Three outcomes, not two").
        """
        return self.resolution.filled

    @property
    def reached(self) -> bool:
        """The object was there and the describe finished.

        A *truncated* describe is not evidence of absence: the node budget stopped
        the walk at 400 attributes (``_DESCRIBE_MAX_NODES``) and what was not
        reported was not looked at.
        """
        return self.answer.ok and not self.truncated

    @property
    def names(self) -> set[str]:
        """Every name this object answered to, however it answered."""
        return set(self.readable) | set(self.unreadable) | self.methods

    @classmethod
    def unpack(cls, resolution: Resolution, answer: Answer) -> Probe:
        probe = cls(resolution=resolution, answer=answer)
        if not answer.ok:
            return probe
        result = answer.result
        probe.live_class = str(result.get("class") or "")
        probe.truncated = bool(result.get("truncated"))
        for entry in result.get("properties") or []:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str):
                continue
            item = dict(entry)
            if "error" in item or "unavailable" in item:
                probe.unreadable[item["name"]] = item
            else:
                probe.readable[item["name"]] = item
        for entry in result.get("children") or []:
            if isinstance(entry, Mapping) and isinstance(entry.get("name"), str):
                probe.readable[entry["name"]] = dict(entry)
        probe.methods = {m for m in result.get("methods") or [] if isinstance(m, str)}
        allowed = result.get("allowed_methods")
        if isinstance(allowed, list):
            probe.reports_allowed = True
            probe.allowed_methods = {m for m in allowed if isinstance(m, str)}
        return probe


@dataclass
class ContainerGroup:
    """Every catalog row on one container, and what Live said about it."""

    container: str
    shape: str
    rows: list[PathSpec] = field(default_factory=list)
    probes: list[Probe] = field(default_factory=list)
    unresolved: str | None = None

    @property
    def reached_probes(self) -> list[Probe]:
        return [probe for probe in self.probes if probe.reached]

    @property
    def reached(self) -> bool:
        return bool(self.reached_probes)

    @property
    def determined(self) -> bool:
        """True when absence here is evidence about the catalog.

        Three things must hold, and each of them has cost somebody an hour
        somewhere: exactly one specimen matched (a polymorphic ``devices[]`` is not
        one object but a family), this tool invented no index (index 0 is a guess
        about the set, not a statement about the model), and every probe finished
        (a truncated describe reports less than it saw).
        """
        if not (
            len(self.probes) == 1
            and not self.probes[0].filled_indices
            and self.probes[0].reached
        ):
            return False
        # A class with siblings is never on its own evidence: the same container
        # shape resolves to a different concrete class depending on what the user
        # loaded (measured -- see POLYMORPHIC_FAMILIES).
        return self.probes[0].live_class not in POLYMORPHIC_FAMILIES

    def readable_entry(self, member: str) -> dict[str, Any] | None:
        """``member``'s describe entry, from the first probe that read it."""
        for probe in self.reached_probes:
            if member in probe.readable:
                return probe.readable[member]
        return None

    def unreadable_entry(self, member: str) -> dict[str, Any] | None:
        for probe in self.reached_probes:
            if member in probe.unreadable:
                return probe.unreadable[member]
        return None

    def has_method(self, member: str) -> bool:
        return any(member in probe.methods for probe in self.reached_probes)

    def known(self, member: str) -> bool:
        """True when any reached probe answered to this name at all."""
        return any(member in probe.names for probe in self.reached_probes)


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #


@dataclass
class RowFinding:
    """What the run decided about one existing catalog row."""

    spec: PathSpec
    verdict: str
    probe_path: str = ""
    live_class: str = ""
    detail: str = ""
    #: A short, groupable category for the report's tally. The detail sentence is
    #: what a human reads. This is what the counts are made of, and deriving it by
    #: slicing the sentence would make the tally hostage to its wording.
    reason: str = ""
    #: Set only when the verdict implies a status the row does not carry.
    new_status: PathStatus | None = None
    note: str = ""

    @property
    def spec_id(self) -> str:
        return self.spec.id

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.spec_id,
            "path": self.spec.path,
            "verdict": self.verdict,
            "probe_path": self.probe_path,
            "live_class": self.live_class,
            "status": self.spec.status.value,
            "new_status": self.new_status.value if self.new_status else None,
            "reason": self.reason,
            "detail": self.detail,
            "note": self.note,
        }


@dataclass
class NewRow:
    """A row Live's surface justifies and the catalog does not have."""

    spec_id: str
    path: str
    access: list[str]
    kind: Kind
    unit: Unit
    status: PathStatus
    doc: str
    params: list[ParamSpec]
    method: str | None
    verify: str
    file: Path
    live_class: str
    probe_path: str
    member: str

    def render(self) -> list[str]:
        """The row as catalog YAML, in the house key order and block style."""
        lines = [
            f"- id: {self.spec_id}",
            f"  path: {self.path}",
            f"  access: [{', '.join(self.access)}]",
        ]
        if self.method:
            lines.append(f"  method: {self.method}")
        lines.append(f"  kind: {self.kind.value}")
        lines.append(f"  unit: {self.unit.value}")
        lines.append("  destructive: false")
        lines.append(f"  verify: {self.verify}")
        lines.append(f"  status: {self.status.value}")
        lines.append("  doc: >-")
        lines.extend(f"    {line}" for line in _wrap(self.doc, 4))
        if self.params:
            lines.append("  params:")
            for param in self.params:
                lines.append(f"    - {{{_render_param(param)}}}")
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.spec_id,
            "path": self.path,
            "access": self.access,
            "kind": self.kind.value,
            "status": self.status.value,
            "file": self.file.name,
            "live_class": self.live_class,
            "probe_path": self.probe_path,
            "member": self.member,
        }


def _render_param(param: ParamSpec) -> str:
    """One ``params`` entry in the catalog's compact flow style."""
    parts = [
        f"name: {param.name}",
        f"kind: {param.kind.value}",
        f"required: {'true' if param.required else 'false'}",
    ]
    if param.default is not None:
        parts.append(f"default: {json.dumps(param.default)}")
    if param.enum:
        parts.append("enum: [" + ", ".join(json.dumps(v) for v in param.enum) + "]")
    return ", ".join(parts)


@dataclass
class AllowlistGap:
    """A method Live has that the script's allowlist refuses.

    Not a catalog row, on purpose: the script is the authority and the catalog's
    ``access: [call]`` is only a request (protocol §6). Widening the allowlist
    costs every user a Live restart, so this is reported as a decision to take
    rather than a diff to apply.
    """

    live_class: str
    probe_path: str
    method: str

    def as_dict(self) -> dict[str, Any]:
        return {"class": self.live_class, "path": self.probe_path, "method": self.method}


# --------------------------------------------------------------------------- #
# The reconciliation
# --------------------------------------------------------------------------- #


@dataclass
class Report:
    """Everything one run concluded."""

    stamp: str
    live_version: str
    script_version: str
    groups: list[ContainerGroup] = field(default_factory=list)
    findings: list[RowFinding] = field(default_factory=list)
    new_rows: list[NewRow] = field(default_factory=list)
    rejected_rows: list[str] = field(default_factory=list)
    allowlist_gaps: list[AllowlistGap] = field(default_factory=list)
    allowlist_drift: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def by_verdict(self, verdict: str) -> list[RowFinding]:
        return [f for f in self.findings if f.verdict == verdict]

    @property
    def deletions(self) -> list[RowFinding]:
        return self.by_verdict(MISSING)

    @property
    def status_changes(self) -> list[RowFinding]:
        return [f for f in self.findings if f.new_status is not None]

    def counts(self) -> dict[str, int]:
        tally = Counter(f.verdict for f in self.findings)
        return {
            "rows_examined": len(self.findings),
            "confirmed": tally[CONFIRMED],
            "missing": tally[MISSING],
            "refused": tally[REFUSED],
            "mismatched": tally[MISMATCHED],
            "not_reached": tally[NOT_REACHED],
            "new_rows": len(self.new_rows),
            "status_changes": len(self.status_changes),
            "allowlist_gaps": len(self.allowlist_gaps),
        }


class Reconciler:
    """Walks the specimens, reads the catalog, and decides what disagrees."""

    def __init__(
        self,
        registry: Registry,
        reader: Reader,
        specimens: Mapping[str, Specimen],
        *,
        allowlist: frozenset[str],
        describe_skip: frozenset[str],
        catalog_root: Path,
        rows: Sequence[PathSpec] | None = None,
    ) -> None:
        self.registry = registry
        self.reader = reader
        self.specimens = specimens
        self.allowlist = allowlist
        self.describe_skip = describe_skip
        self.catalog_root = catalog_root
        self.rows = list(rows if rows is not None else registry.all())
        #: Filled in by :meth:`run` from the handshake, because a note without the
        #: Live version it was true of is a rumour (``tests/test_catalog.py``).
        self.stamp = ""
        #: Whether this run may write a status at all; see :meth:`run`.
        self.measurable = False
        self._file_of = _row_files(catalog_root)
        # Built from the WHOLE catalog, never from the --area subset: a member is
        # only "new" when nothing anywhere already claims it, and filtering the
        # left-hand side of that question would propose duplicates.
        self._catalogued = _index_by_member(registry.all())
        self._exemplars = _index_by_container(registry.all())
        # Every id that is spoken for, including the ones this run invents as it
        # goes. The registry alone is not enough: two containers filed under one
        # area can each discover the same member name -- Device.color and
        # Chain.color both want `device.color` -- and two rows with one id do not
        # load. That failure would land at the very end, in the re-parse after the
        # file is already written, and take every other discovery in the run down
        # with it.
        self._taken_ids = {spec.id for spec in registry.all()}

    # ------------------------------------------------------------------- run
    def run(self) -> Report:
        info = self.reader.handshake()
        self.stamp = _stamp(info)
        self.measurable = bool(_LIVE_VERSION_RE.search(self.stamp))
        report = Report(
            stamp=self.stamp,
            live_version=str(info.get("live_version") or "unknown"),
            script_version=str(info.get("script_version") or "unknown"),
        )
        handlers = info.get("handlers")
        if isinstance(handlers, list) and "lom_describe" not in handlers:
            report.warnings.append(
                "the script does not offer lom_describe; this run can conclude nothing"
            )
        if not self.measurable:
            report.warnings.append(
                "the handshake named no Live version, so this run may not write a "
                "measurement: no row is flipped to verified or broken, and a property "
                "Live confirmed is proposed as untested. A status whose note cannot say "
                "which build it was true of is a rumour, and tests/test_catalog.py "
                "rejects it (docs/catalog.md). Everything below is still reported"
            )

        groups = self._build_groups()
        report.groups = groups
        for group in groups:
            self._describe_group(group)

        for group in groups:
            for spec in group.rows:
                report.findings.append(self._judge(spec, group))
            self._propose(group, report)

        self._check_allowlist_drift(groups, report)
        return report

    # -------------------------------------------------------------- grouping
    def _build_groups(self) -> list[ContainerGroup]:
        """One group per distinct container, plus one per unused specimen.

        The second half matters: a class whose rows are all in another ``--area``,
        or which has no rows at all, is still worth describing: that is where the
        169 uncatalogued properties came from.
        """
        groups: dict[str, ContainerGroup] = {}
        for spec in self.rows:
            target = target_of(spec)
            if target is None:
                continue
            group = groups.get(target.container)
            if group is None:
                group = ContainerGroup(
                    container=target.container,
                    shape=shape_of(split_template(target.container)),
                )
                groups[target.container] = group
            group.rows.append(spec)

        covered = {
            resolution.probe_path
            for group in groups.values()
            for resolution in resolve_container(group.container, self.specimens)[0]
        }
        for specimen in self.specimens.values():
            if specimen.path in covered or specimen.path in groups:
                continue
            groups[specimen.path] = ContainerGroup(
                container=specimen.path, shape=specimen.shape
            )
        return list(groups.values())

    def _describe_group(self, group: ContainerGroup) -> None:
        resolutions, reason = resolve_container(group.container, self.specimens)
        if not resolutions:
            group.unresolved = reason
            return
        for resolution in resolutions:
            group.probes.append(
                Probe.unpack(resolution, self.reader.describe(resolution.probe_path))
            )

    # --------------------------------------------------------------- judging
    def _judge(self, spec: PathSpec, group: ContainerGroup) -> RowFinding:
        target = target_of(spec)
        if target is None:  # pragma: no cover - groups are built from targets
            return RowFinding(spec, NOT_REACHED, reason="no member",
                              detail="the row addresses no member")
        member = target.member

        if group.unresolved is not None:
            return RowFinding(
                spec, NOT_REACHED, reason="no specimen", detail=group.unresolved
            )
        if not group.reached:
            reason, detail = self._why_not_reached(group)
            return RowFinding(spec, NOT_REACHED, reason=reason, detail=detail)
        if member in self.describe_skip:
            return RowFinding(
                spec,
                NOT_REACHED,
                # The probe that answered, not merely the first one tried: on a
                # polymorphic container the first can be the one that failed, and a
                # report that names the wrong path sends the reader to the wrong object.
                probe_path=group.reached_probes[0].probe_path,
                reason="lom_describe skips this attribute",
                detail=(
                    f"lom_describe never follows {member!r} (_DESCRIBE_SKIP in the "
                    "Remote Script), so this run saw nothing either way"
                ),
            )

        probe = group.reached_probes[0]
        live_class = probe.live_class

        if target.is_method:
            return self._judge_method(spec, group, member, live_class)

        entry = group.readable_entry(member)
        if entry is not None:
            return self._confirm_property(spec, group, member, entry, live_class)

        refusal = group.unreadable_entry(member)
        if refusal is not None and "unavailable" in refusal:
            # The script's own track guard, not Live's answer. `arm` on a group
            # track is missing from that track, not from the Track class, and the
            # guard reports no_such_path (protocol §6 rule 4) which would read as
            # a fantasy row.
            return RowFinding(
                spec,
                NOT_REACHED,
                probe_path=probe.probe_path,
                live_class=live_class,
                reason="the Remote Script's track guard",
                detail=(
                    "the Remote Script's track guard refused this attribute on the "
                    f"specimen: {refusal.get('unavailable')}. That is a fact about "
                    "this track, not about the row"
                ),
            )
        return self._confirm_by_read(spec, group, member, live_class)

    def _judge_method(
        self, spec: PathSpec, group: ContainerGroup, member: str, live_class: str
    ) -> RowFinding:
        probe = group.reached_probes[0]
        if group.has_method(member):
            allowed = any(
                member in p.allowed_methods for p in group.reached_probes if p.reports_allowed
            ) or any(f"{p.live_class}.{member}" in self.allowlist for p in group.reached_probes)
            detail = (
                "on the running script's allowlist"
                if allowed
                else "NOT on the running script's allowlist: lom_call answers "
                "method_not_allowed until the allowlist is widened, which costs a "
                "Live restart (protocol section 6)"
            )
            return RowFinding(
                spec,
                CONFIRMED,
                probe_path=probe.probe_path,
                live_class=live_class,
                detail=f"{live_class}.{member}() exists; {detail}",
            )
        # Not in the method list. Confirm with a read: the resolver answers
        # method_not_allowed for a callable and no_such_path for a name that is
        # not there, so a read tells the two apart without invoking anything.
        return self._confirm_by_read(spec, group, member, live_class, expect_method=True)

    def _confirm_property(
        self,
        spec: PathSpec,
        group: ContainerGroup,
        member: str,
        entry: Mapping[str, Any],
        live_class: str,
    ) -> RowFinding:
        probe = group.reached_probes[0]
        finding = RowFinding(
            spec,
            CONFIRMED,
            probe_path=f"{probe.probe_path}.{member}",
            live_class=live_class,
            detail=_describe_entry(entry),
        )
        if spec.status is not PathStatus.VERIFIED:
            self._flip(
                finding,
                PathStatus.VERIFIED,
                f"Read verified {self.stamp} at {finding.probe_path} through "
                f"lom_describe: {_describe_entry(entry)}. No write, call, observation or "
                "automation was attempted on this row.",
            )
        return finding

    def _flip(self, finding: RowFinding, status: PathStatus, note: str) -> None:
        """Record a status change on ``finding``, if this run may claim one.

        A status and its note are one thing, not two (``docs/catalog.md``,
        "``verified`` does not say *which* access; the ``doc`` does"), and the note
        has to name the build the reading was true of. A handshake that reported no
        version leaves nothing to name, so the finding keeps its verdict, says why
        the status stayed put, and writes nothing. The alternative is a catalog that
        fails its own tests, which is the polite version of the real damage: a row
        that claims to have been measured and cannot say against what.
        """
        if not self.measurable:
            finding.detail += (
                f". The status was NOT changed to {status.value}: this run could not "
                "name a Live version, and a status whose note cannot say which build "
                "it was true of is not a measurement (docs/catalog.md)"
            )
            return
        finding.new_status = status
        finding.note = note

    def _confirm_by_read(
        self,
        spec: PathSpec,
        group: ContainerGroup,
        member: str,
        live_class: str,
        *,
        expect_method: bool = False,
    ) -> RowFinding:
        """Ask ``lom_get`` for the one answer the policy branches on."""
        probe = group.reached_probes[0]
        path = f"{probe.probe_path}.{member}"
        answer = self.reader.get(path)

        if answer.ok:
            detail = f"lom_get read it: {_short(answer.result.get('value'))}"
            verdict = MISMATCHED if expect_method else CONFIRMED
            if expect_method:
                detail = (
                    "lom_get resolved it as a value, so this is a property and not a "
                    f"method: {detail}. The row claims access: [call]"
                )
            return RowFinding(
                spec, verdict, probe_path=path, live_class=live_class, detail=detail
            )

        if answer.code == "method_not_allowed":
            if expect_method:
                return RowFinding(
                    spec,
                    CONFIRMED,
                    probe_path=path,
                    live_class=live_class,
                    detail=(
                        "the resolver refused it as a callable, which is how a method "
                        "that exists answers a read (protocol section 6 rule 2). "
                        "lom_describe did not list it, so it is not in dir() on this "
                        "specimen -- check the class before trusting the row"
                    ),
                )
            return RowFinding(
                spec,
                MISMATCHED,
                probe_path=path,
                live_class=live_class,
                detail=(
                    f"{live_class}.{member} is a method, and this row addresses it as a "
                    "property. It needs access: [call] and a method: field"
                ),
            )

        if answer.code == "live_error":
            if _TYPE_PRECONDITION.search(answer.message or ""):
                # Live is describing the SPECIMEN, not the row: "only available
                # for Audio Clips" means we probed a MIDI clip. Nothing is
                # learned about the catalog (measured 2026-08-30).
                return RowFinding(
                    spec,
                    NOT_REACHED,
                    probe_path=path,
                    live_class=live_class,
                    reason="the specimen is the wrong kind of object",
                    detail=(
                        f"answered live_error ({answer.message}), which names a "
                        "precondition on the OBJECT, not a fault in the row. "
                        "Nothing is written -- probe again on a specimen of the "
                        "kind the message asks for"
                    ),
                )

            if not group.determined:
                return RowFinding(
                    spec,
                    NOT_REACHED,
                    probe_path=path,
                    live_class=live_class,
                    reason="live_error on a derived probe",
                    detail=(
                        f"Live raised ({answer.message}), but the probe is derived, not "
                        f"determined: {self._why_derived(group)}. Nothing is written"
                    ),
                )
            finding = RowFinding(
                spec,
                REFUSED,
                probe_path=path,
                live_class=live_class,
                detail=f"lom_get answered live_error: {answer.message}",
            )
            if spec.status is not PathStatus.BROKEN:
                self._flip(
                    finding,
                    PathStatus.BROKEN,
                    f"Probed {self.stamp} at {path}: lom_get answered "
                    f'live_error - "{answer.message}". The property is there and Live '
                    "refuses it, so this row stays as a documented refusal rather than "
                    "being removed (docs/catalog.md).",
                )
            return finding

        if answer.code == "no_such_path":
            if _NULL_CONTAINER.search(answer.message or ""):
                # "NoneType has no attribute 'name'" says the CONTAINER was None,
                # not that the member is missing. Measured 2026-08-30 against Live
                # 12.4.5: the probed clip had no groove assigned, while
                # song.groove_pool.grooves[0].name reads "Swing 16ths 66" perfectly.
                # Reading a null container as an absent member would have deleted a
                # real row -- the null-versus-absent confusion, in the one place
                # where it costs a fact.
                return RowFinding(
                    spec,
                    NOT_REACHED,
                    probe_path=path,
                    live_class=live_class,
                    reason="the container on the path was None",
                    detail=(
                        f"answered no_such_path, but the message ({answer.message}) "
                        "says the container itself was None -- an unassigned optional "
                        "object, not a missing member. Nothing is written. Probe again "
                        "on a specimen where that object exists"
                    ),
                )
            if not group.determined:
                return RowFinding(
                    spec,
                    NOT_REACHED,
                    probe_path=path,
                    live_class=live_class,
                    reason="no_such_path on a derived probe",
                    detail=(
                        "answered no_such_path, but the probe is derived, not "
                        f"determined: {self._why_derived(group)}. Nothing is written -- "
                        f"add a specimen at {group.shape} with --specimens and run again"
                    ),
                )
            return RowFinding(
                spec,
                MISSING,
                probe_path=path,
                live_class=live_class,
                detail=f"lom_get answered no_such_path: {answer.message}",
            )

        return RowFinding(
            spec,
            NOT_REACHED,
            probe_path=path,
            live_class=live_class,
            reason=f"lom_get answered {answer.code}",
            detail=f"lom_get answered {answer.code}: {answer.message}",
        )

    def _why_not_reached(self, group: ContainerGroup) -> tuple[str, str]:
        """``(reason, detail)`` for a container no probe could describe."""
        reasons: list[str] = []
        parts: list[str] = []
        for probe in group.probes:
            if probe.answer.ok and probe.truncated:
                reasons.append("the describe was truncated")
                parts.append(
                    f"{probe.probe_path}: the describe was truncated by the node budget, "
                    "so what it did not report was not looked at"
                )
            elif not probe.answer.ok:
                reasons.append(f"lom_describe answered {probe.answer.code}")
                parts.append(
                    f"{probe.probe_path}: {probe.answer.code} - {probe.answer.message}"
                )
        reason = " / ".join(dict.fromkeys(reasons)) or "no probe was made"
        return reason, ("; ".join(parts) or "no probe was made")

    def _why_derived(self, group: ContainerGroup) -> str:
        if len(group.probes) > 1:
            classes = ", ".join(
                f"{p.resolution.specimen.class_name}->{p.live_class or '?'}" for p in group.probes
            )
            return f"{len(group.probes)} specimens share the shape {group.shape} ({classes})"
        filled = group.probes[0].filled_indices if group.probes else ()
        if filled:
            return "this tool chose the index for " + ", ".join(f"{n}[0]" for n in filled)
        return "the describe did not finish"

    # ------------------------------------------------------------- proposals
    def _propose(self, group: ContainerGroup, report: Report) -> None:
        """Everything Live reported on this container and not catalogued."""
        for probe in group.reached_probes:
            for member, entry in sorted(probe.readable.items()):
                # A name lom_describe deliberately does not follow is not a
                # discovery: _DESCRIBE_SKIP holds `canonical_parent`, which walks
                # back up the object graph and turns a recursive describe into a
                # cycle. Proposing a row for it would put a trap in the catalog.
                if member in self.describe_skip:
                    continue
                if self._is_catalogued(group.shape, member, is_method=False):
                    continue
                self._add_proposal(group, probe, member, entry, report, is_method=False)
            for member in sorted(probe.methods):
                if member in self.describe_skip:
                    continue
                if self._is_catalogued(group.shape, member, is_method=True):
                    continue
                allowed = (
                    member in probe.allowed_methods
                    if probe.reports_allowed
                    else f"{probe.live_class}.{member}" in self.allowlist
                )
                if not allowed:
                    report.allowlist_gaps.append(
                        AllowlistGap(
                            live_class=probe.live_class,
                            probe_path=probe.probe_path,
                            method=member,
                        )
                    )
                    continue
                self._add_proposal(group, probe, member, {}, report, is_method=True)

    def _is_catalogued(self, shape: str, member: str, *, is_method: bool) -> bool:
        return (shape, member, is_method) in self._catalogued

    def _add_proposal(
        self,
        group: ContainerGroup,
        probe: Probe,
        member: str,
        entry: Mapping[str, Any],
        report: Report,
        *,
        is_method: bool,
    ) -> None:
        if not _MEMBER_RE.match(member):
            report.rejected_rows.append(
                f"{probe.live_class}.{member} on {probe.probe_path}: the name does not fit "
                "a catalog id (lowercase, letters, digits, underscores). Add it by hand"
            )
            return

        exemplar = self._exemplars.get(group.shape)
        if exemplar is None:
            report.rejected_rows.append(
                f"{probe.live_class}.{member} on {probe.probe_path}: no catalog row exists "
                f"on {group.shape}, so there is no path template or params block to copy. "
                "Add one row of this container by hand and run again"
            )
            return

        area = probe.resolution.specimen.area or area_of(exemplar.id)
        spec_id = self._free_id(area, member, group.shape)
        if spec_id is None:
            report.rejected_rows.append(
                f"{probe.live_class}.{member} on {probe.probe_path}: every id this tool "
                f"would give it ({area}.{member}) is taken by another row. Add it by hand"
            )
            return

        container_template = _container_of(exemplar)
        path = f"{container_template}.{member}" if not is_method else container_template
        # Only the placeholders this path actually uses. The exemplar is picked by
        # container SHAPE, and a shape can be shorter than the exemplar's own path --
        # the browser roots are the measured case: an exemplar at
        # app.browser.{root}.children[{index}] lends its template to a row at
        # app.browser.{root}, and copying its params wholesale drags {index} along.
        # PathSpec then refuses the row ("parameter(s) ['index'] do not appear in
        # path"), which is the catalog's own guard against an argument that lands
        # nowhere -- correct, but it fails the whole file rather than this one row.
        # Measured 2026-08-30: this is what rolled 50-browser.yaml back on the first
        # --add run.
        used = set(placeholders_in(path))
        params = [param for param in exemplar.params if param.name in used]
        file = self._file_of.get(exemplar.id, self.catalog_root)

        if is_method:
            row = NewRow(
                spec_id=spec_id,
                path=path,
                access=["call"],
                kind=Kind.OBJECT,
                unit=Unit.NONE,
                status=PathStatus.UNTESTED,
                doc=_method_doc(probe, member),
                params=params,
                method=member,
                verify="none",
                file=file,
                live_class=probe.live_class,
                probe_path=probe.probe_path,
                member=member,
            )
        else:
            kind, unit = _kind_and_unit(entry)
            access = ["get"]
            if entry.get("settable") is True:
                access.append("set")
            row = NewRow(
                spec_id=spec_id,
                path=path,
                access=access,
                kind=kind,
                unit=unit,
                status=PathStatus.VERIFIED if self.measurable else PathStatus.UNTESTED,
                doc=_property_doc(
                    probe, member, entry, self.stamp, measurable=self.measurable
                ),
                params=params,
                method=None,
                verify="read_back",
                file=file,
                live_class=probe.live_class,
                probe_path=f"{probe.probe_path}.{member}",
                member=member,
            )
        report.new_rows.append(row)
        self._catalogued[(group.shape, member, is_method)] = row.spec_id
        self._taken_ids.add(row.spec_id)

    def _free_id(self, area: str, member: str, shape: str) -> str | None:
        """Return an id nothing claims, across the catalog's rows and this run's own."""
        taken = self._taken_ids
        candidate = f"{area}.{member}"
        if candidate not in taken:
            return candidate
        leaf = shape.rsplit(".", 1)[-1].removesuffix("[]")
        qualified = f"{area}.{leaf}_{member}"
        if _MEMBER_RE.match(f"{leaf}_{member}") and qualified not in taken:
            return qualified
        return None

    # ------------------------------------------------------------- allowlist
    def _check_allowlist_drift(self, groups: Sequence[ContainerGroup], report: Report) -> None:
        """Compare the allowlist on disk with the running script's own.

        Live loads a Remote Script at startup and keeps the compiled ``__pycache__``
        (protocol §9), so a checkout that is ahead of the running process is the
        normal state after an edit, and it is invisible from anywhere else.

        Comparison is by bare method name, not by ``Class.method``. The script
        matches an allowlist entry against the object's whole MRO, so a method
        listed as ``Device.store_chosen_bank`` is legitimately allowed on an
        ``Eq8Device``, which inherits it. Comparing the concrete class name against
        the literal entries reported that as drift on every subclass. Measured
        2026-08-30 against Live 12.4.5, where `Eq8Device` and `RackDevice` both
        raised a false alarm claiming Live ran "a newer copy than this working
        tree" while the checkout was in fact the source of the running script.
        Names are the honest comparison here: this tool cannot see an MRO, so it
        must not pretend to.
        """
        by_name = {entry.rsplit(".", 1)[-1] for entry in self.allowlist}
        for group in groups:
            for probe in group.reached_probes:
                if not probe.reports_allowed or not probe.live_class:
                    continue
                # Asymmetric on purpose, because the two directions are not the same
                # question and the tool cannot see an MRO.
                #
                # "the checkout allows it, Live does not" is only evidence when the
                # entry names THIS class. A base-class entry may not apply,
                # and a same-named entry on an unrelated class certainly does not:
                # `ClipSlot.create_audio_clip` must never be read as permission on a
                # Track, and the running script is right to refuse it (measured
                # 2026-08-30).
                #
                # "Live allows it, the checkout has no entry of that name anywhere"
                # is evidence either way, because no MRO can conjure a name that is
                # not in the file at all.
                exact = {
                    method
                    for method in probe.methods
                    if f"{probe.live_class}.{method}" in self.allowlist
                }
                only_disk = sorted(exact - probe.allowed_methods)
                only_live = sorted(
                    method for method in probe.allowed_methods if method not in by_name
                )
                if only_disk:
                    report.allowlist_drift.append(
                        f"{probe.live_class}: the checkout allows {only_disk} and the "
                        "running script does not. Live is running an older copy -- copy "
                        "the script, delete __pycache__, restart Live (protocol section 9)"
                    )
                if only_live:
                    report.allowlist_drift.append(
                        f"{probe.live_class}: the running script allows {only_live} and "
                        "no entry in the checkout carries that name. Either Live is "
                        "running a copy newer than this working tree, or the entry was "
                        "removed here after Live started"
                    )


def _stamp(info: Mapping[str, Any]) -> str:
    """``2026-08-30 against Live 12.4.5``: a date and the build it was true of.

    Both halves are load-bearing. ``tests/test_catalog.py`` looks for exactly this
    shape before it will let a row claim ``verified`` or ``broken``, because a
    status without a date and a version is a rumour.
    """
    version = str(info.get("live_version") or "unknown")
    return f"{datetime.now().astimezone().date().isoformat()} against Live {version}"


# --------------------------------------------------------------------------- #
# Turning a describe entry into catalog fields
# --------------------------------------------------------------------------- #


def _describe_entry(entry: Mapping[str, Any]) -> str:
    """One human sentence about what the describe said, for a doc note."""
    if entry.get("is_collection"):
        count = entry.get("count")
        cls = entry.get("class") or "collection"
        if count is None:
            return f"a {cls} that does not answer len() in this Live version"
        return f"a {cls} of {count} element(s)"
    if entry.get("type") and not entry.get("is_collection") and "value" not in entry:
        return f"a {entry['type']} object"
    kind = entry.get("type") or "unknown"
    if kind == "null":
        return "returned null, so the type is not settled by this reading"
    return f"returned {_short(entry.get('value'))} ({kind})"


def _kind_and_unit(entry: Mapping[str, Any]) -> tuple[Kind, Unit]:
    """The catalog ``kind`` and ``unit`` a described member deserves.

    ``unit`` is the honest half. :class:`~ableton_maestro.models.Unit` says an
    unknown unit is ``normalized``, because in this object model that is what an
    unexplained number nearly always is: a filter cutoff of 0.5 is not a
    frequency, a track volume of 0.85 is 0 dB (measured). A flag,
    a string, a list or an object carries no unit at all, and that is ``none``.
    """
    if entry.get("is_collection"):
        return Kind.LIST, Unit.NONE
    if "value" not in entry and entry.get("type"):
        return Kind.OBJECT, Unit.NONE
    kind = _KIND_FROM_WIRE.get(str(entry.get("type") or ""), Kind.OBJECT)
    if kind in {Kind.INT, Kind.FLOAT}:
        return kind, Unit.NORMALIZED
    return kind, Unit.NONE


def _property_doc(
    probe: Probe, member: str, entry: Mapping[str, Any], stamp: str, *, measurable: bool = True
) -> str:
    """The ``doc`` for a newly discovered property.

    It ends with a measurement sentence, and the pairing is not optional:
    ``docs/catalog.md`` requires a ``verified`` row to say what was done, and
    ``tests/test_catalog.py`` fails a status without a matching note *and* a note
    without a matching status. The sentence therefore names the exact path, the
    exact reading, and everything that was not tried. That last part is what a
    hand-written note forgets.

    ``measurable=False`` says the run could not name a Live version. The row then
    arrives ``untested`` and this sentence must not read as a measurement either,
    or the row fails the *other* half of that pairing: a note without a status is
    just as wrong as a status without a note, and it hides paid-for knowledge
    rather than inventing it.
    """
    settable = entry.get("settable")
    if settable is True:
        writability = (
            "Live reports it as settable, and this row therefore carries `set`, but no "
            "write has been attempted. "
        )
    elif settable is False:
        writability = "Live reports it as read-only. "
    else:
        writability = (
            "Whether it can be written is not knowable from a describe: Live's generated "
            "types use C-level descriptors that always look writable and raise at "
            "assignment time, so the script reports null rather than guessing. "
        )
    unit_note = ""
    kind, _unit = _kind_and_unit(entry)
    if kind in {Kind.INT, Kind.FLOAT}:
        unit_note = (
            "The unit is not established. A raw LOM number is normalised far more often "
            "than it is physical, and the display string only exists for a "
            "DeviceParameter that answers str_for_value() (docs/protocol.md section 7). "
        )
    if measurable:
        reading = (
            f"Read verified {stamp} at {probe.probe_path}.{member} through lom_describe: "
            f"{_describe_entry(entry)}. No write, call, observation or automation was "
            "attempted on this row."
        )
    else:
        reading = (
            f"lom_describe read it at {probe.probe_path}.{member} and it answered with "
            f"{_describe_entry(entry)} -- but the run that found it could not name a Live "
            "version, so that reading is not recorded here as a measurement and the row "
            "stays untested. Re-run scripts/sync_catalog.py against a script that reports "
            "one (docs/catalog.md)."
        )
    return (
        f"{probe.live_class}.{member}, found on the running object model and absent from "
        f"this catalog until scripts/sync_catalog.py proposed it. {writability}{unit_note}"
        f"{reading}"
    )


def _method_doc(probe: Probe, member: str) -> str:
    """The ``doc`` for a newly discovered, allowlisted method.

    Deliberately carries no date and no Live version. The row is ``untested``
    because nothing called the method, and ``tests/test_catalog.py`` fails an
    ``untested`` row whose doc reads as a measurement. The run's date and version
    are in the sync report instead, which is where a claim nobody paid for belongs.
    """
    return (
        f"{probe.live_class}.{member}() exists on the object at {probe.probe_path} in the "
        "running Live and is on the Remote Script's allowlist, so lom_call will attempt "
        "it. Its arguments were NOT read: the LOM publishes no signature over this "
        "channel, so `args` is empty and a call that needs arguments will answer "
        "type_error until somebody fills them in. What it returns is unknown, hence "
        "kind: object. Nothing has called it -- this row is a reachable hypothesis, not "
        "a capability, and the run that found it is recorded in the sync_catalog report "
        "rather than here, because an untested row may not carry a measurement note."
    )


def _short(value: object, limit: int = 60) -> str:
    """A value squeezed into a doc sentence without wrecking it."""
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


# --------------------------------------------------------------------------- #
# Indexing the catalog
# --------------------------------------------------------------------------- #


def _container_of(spec: PathSpec) -> str:
    target = target_of(spec)
    return target.container if target is not None else spec.path


def _index_by_member(specs: Iterable[PathSpec]) -> dict[tuple[str, str, bool], str]:
    """Map ``(container shape, member, is_method)`` to a row id.

    Built from the whole catalog.
    """
    index: dict[tuple[str, str, bool], str] = {}
    for spec in specs:
        target = target_of(spec)
        if target is None:
            continue
        try:
            shape = shape_of(split_template(target.container))
        except TemplateError:
            continue
        index.setdefault((shape, target.member, target.is_method), spec.id)
    return index


def _index_by_container(specs: Iterable[PathSpec]) -> dict[str, PathSpec]:
    """Return one exemplar row per container shape, the template a new row copies.

    A proposal inherits the exemplar's container template and its ``params`` block
    verbatim, which is what gives it ``{track}`` rather than a guessed placeholder
    name, and what puts it in the same file as its neighbours.
    """
    index: dict[str, PathSpec] = {}
    for spec in specs:
        target = target_of(spec)
        if target is None:
            continue
        try:
            shape = shape_of(split_template(target.container))
        except TemplateError:
            continue
        index.setdefault(shape, spec)
    return index


def _row_files(root: Path) -> dict[str, Path]:
    """``row id -> the file it lives in``, read from the raw YAML.

    Read from the text rather than from the registry because the registry
    deliberately does not remember: it concatenates the files into one namespace.
    """
    files = [root] if root.is_file() else sorted(root.glob("*.yaml"))
    index: dict[str, Path] = {}
    for file in files:
        with open(file, encoding="utf-8", newline="") as handle:
            for line in handle:
                match = _ROW_START.match(line.rstrip("\r\n"))
                if match is not None:
                    index.setdefault(match.group("id").strip("'\" "), file)
    return index


_ROW_START = re.compile(r"^-\s+id:\s*(?P<id>.+?)\s*(?:#.*)?$")


# --------------------------------------------------------------------------- #
# Writing the catalog
# --------------------------------------------------------------------------- #


@dataclass
class WritePlan:
    """Which classes of change this invocation is allowed to apply."""

    add: bool = False
    delete: bool = False
    status: bool = False

    @property
    def any(self) -> bool:
        return self.add or self.delete or self.status


class CatalogSync:
    """Applies a :class:`Report` to the catalog files, at the line level.

    Never through ``yaml.dump``. The catalog is 22 000 lines of hand-written prose
    and section comments. A round trip would delete every comment and reflow the
    whole thing into one unreviewable diff. Each edited file is re-parsed with
    :class:`~ableton_maestro.registry.Registry` before it is kept and put back
    unchanged if it stops loading, so a bad edit costs a message and not a catalog.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files = [root] if root.is_file() else sorted(root.glob("*.yaml"))
        if not self.files:
            raise CatalogWriteError(f"no catalog files under {root}")

    @staticmethod
    def read(file: Path) -> str:
        with open(file, encoding="utf-8", newline="") as handle:
            return handle.read()

    @staticmethod
    def write(file: Path, text: str) -> None:
        with open(file, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)

    def apply(self, report: Report, plan: WritePlan) -> list[str]:
        """Apply everything ``plan`` permits. Returns one line per change."""
        changed: list[str] = []
        deletions = report.deletions if plan.delete else []
        statuses = report.status_changes if plan.status else []
        additions = report.new_rows if plan.add else []

        # Deletions and status edits touch existing rows and are done first, per
        # file, bottom-up so that one edit never moves the line another edit found.
        edits: dict[Path, list[tuple[RowLocation, RowFinding, str]]] = {}
        for finding in deletions:
            location = self._locate(finding.spec_id)
            if location is None:
                raise CatalogWriteError(
                    f"{finding.spec_id}: no '- id: {finding.spec_id}' line under {self.root}; "
                    "refusing to guess where the row lives"
                )
            edits.setdefault(location.file, []).append((location, finding, "delete"))
        for finding in statuses:
            if finding in deletions:
                continue
            location = self._locate(finding.spec_id)
            if location is None:
                raise CatalogWriteError(
                    f"{finding.spec_id}: no '- id: {finding.spec_id}' line under {self.root}"
                )
            edits.setdefault(location.file, []).append((location, finding, "status"))

        for file, entries in edits.items():
            original = self.read(file)
            lines = _split_lines(original)
            entries.sort(key=lambda item: item[0].start, reverse=True)
            for location, finding, kind in entries:
                newline = _newline_of(lines, location.start)
                if kind == "delete":
                    _delete_row(lines, location)
                    changed.append(f"{file.name}: {finding.spec_id} REMOVED")
                else:
                    status = finding.new_status
                    if status is None:  # pragma: no cover - filtered above
                        continue
                    _set_status(lines, location, status.value, newline)
                    _append_doc_note(lines, location, finding.note, newline)
                    changed.append(f"{file.name}: {finding.spec_id} -> {status.value}")
            self._write_verified(file, "".join(lines), original)

        if additions:
            changed.extend(self._append(additions, report))
        return changed

    def _append(self, rows: Sequence[NewRow], report: Report) -> list[str]:
        """Append proposed rows to the file their container lives in."""
        changed: list[str] = []
        by_file: dict[Path, list[NewRow]] = {}
        for row in rows:
            by_file.setdefault(row.file if row.file.is_file() else self.files[0], []).append(row)

        for file, group in by_file.items():
            original = self.read(file)
            lines = _split_lines(original)
            # Line 0, not the last line: a file whose final line has no newline
            # would otherwise report LF and turn a CRLF catalog into a mixed one.
            newline = _newline_of(lines, 0)
            if lines and not lines[-1].endswith("\n"):
                lines.append(newline)
            block = [
                newline,
                f"# {'-' * 75}{newline}",
                f"# Appended by scripts/sync_catalog.py, {report.stamp}.{newline}",
                f"# Live reports these on the objects named in each row's doc; the{newline}",
                f"# catalog did not have them. Statuses come from the read that found{newline}",
                f"# them and from nothing else.{newline}",
                f"# {'-' * 75}{newline}",
            ]
            for row in group:
                block.append(newline)
                block.extend(line + newline for line in row.render())
                changed.append(f"{file.name}: + {row.spec_id} ({row.status.value})")
            lines.extend(block)
            self._write_verified(file, "".join(lines), original)
        return changed

    def _locate(self, spec_id: str) -> RowLocation | None:
        """The block of lines one row occupies.

        ``end`` stops at the last indented line of the row, not at the next
        ``- ``: a section-header comment between two rows belongs to the section,
        and a deletion that swallowed it would destroy hand-written prose to remove
        one property.
        """
        for file in self.files:
            lines = _split_lines(self.read(file))
            for index, line in enumerate(lines):
                match = _ROW_START.match(line.rstrip("\r\n"))
                if match is None or match.group("id").strip("'\" ") != spec_id:
                    continue
                end = index + 1
                for after in range(index + 1, len(lines)):
                    text = lines[after].rstrip("\r\n")
                    if not text.strip() or not text[:1].isspace():
                        break
                    end = after + 1
                return RowLocation(file=file, start=index, end=end)
        return None

    def _write_verified(self, file: Path, text: str, original: str) -> None:
        """Replace ``file`` atomically, and put it back if it stops parsing."""
        temporary = file.with_suffix(file.suffix + ".sync-tmp")
        self.write(temporary, text)
        temporary.replace(file)
        try:
            Registry.load(file)
        except CatalogError as exc:
            self.write(file, original)
            raise CatalogWriteError(
                f"{file.name} no longer parses after the edit, so the original was "
                f"restored and nothing in it was changed: {exc}"
            ) from exc


def _delete_row(lines: list[str], location: RowLocation) -> None:
    """Remove a row's own lines, and one blank line if that would leave two.

    Only the row's own lines. A section-header comment sitting above it belongs to
    the section and stays: deleting hand-written prose to remove one property would
    cost more than the property was worth, and an orphaned header is visible in the
    diff where a deleted paragraph is not.
    """
    before_blank = location.start > 0 and not lines[location.start - 1].strip()
    del lines[location.start : location.end]
    index = location.start
    if before_blank and index < len(lines) and not lines[index].strip():
        del lines[index]


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def print_summary(title: str, registry: Registry, out: Any) -> dict[str, int]:
    counts = registry.status_counts()
    total = len(registry.all())
    body = "  ".join(f"{name} {count}" for name, count in sorted(counts.items()))
    print(f"{title}: {total} rows  ({body})", file=out)
    return {"rows": total, **counts}


def print_report(report: Report, out: Any, *, verbose: bool) -> None:
    counts = report.counts()
    print(file=out)
    print(f"Live {report.live_version}, Remote Script {report.script_version}", file=out)
    for warning in report.warnings:
        print(f"  WARNING: {warning}", file=out)
    print(file=out)
    print(
        "  examined {rows_examined} row(s): {confirmed} confirmed, {missing} missing, "
        "{refused} refused by Live, {mismatched} mis-shaped, {not_reached} not reached".format(
            **counts
        ),
        file=out,
    )
    print(
        f"  proposals: {counts['new_rows']} new row(s), {len(report.deletions)} deletion(s), "
        f"{counts['status_changes']} status change(s)",
        file=out,
    )

    _section(out, "NEW ROWS -- Live has these, the catalog does not", report.new_rows and True)
    for row in report.new_rows:
        print(
            f"  + {row.spec_id:<40} {row.status.value:<9} {row.live_class}.{row.member} "
            f"-> {row.file.name}",
            file=out,
        )

    _section(
        out,
        "DELETIONS -- the target answered no_such_path, so the row is fantasy",
        bool(report.deletions),
    )
    for finding in report.deletions:
        print(f"  - {finding.spec_id:<40} {finding.probe_path}", file=out)
        print(f"      {finding.detail}", file=out)

    refused = report.by_verdict(REFUSED)
    _section(
        out,
        "BROKEN -- the target exists and Live refuses it; the row stays as a refusal",
        bool(refused),
    )
    for finding in refused:
        print(f"  ! {finding.spec_id:<40} {finding.probe_path}", file=out)
        print(f"      {finding.detail}", file=out)

    mismatched = report.by_verdict(MISMATCHED)
    _section(out, "MIS-SHAPED -- it exists, but not as the row describes it", bool(mismatched))
    for finding in mismatched:
        print(f"  ? {finding.spec_id:<40} {finding.detail}", file=out)

    statuses = [f for f in report.status_changes if f.verdict == CONFIRMED]
    _section(out, "STATUS -- confirmed by a read, and the row does not say so", bool(statuses))
    for finding in statuses:
        print(
            f"  * {finding.spec_id:<40} {finding.spec.status.value} -> "
            f"{finding.new_status.value if finding.new_status else '?'}",
            file=out,
        )

    _section(
        out,
        "REACHABLE IF THE ALLOWLIST WERE WIDENED -- not catalog rows; the script decides",
        bool(report.allowlist_gaps),
    )
    for gap in report.allowlist_gaps:
        print(f"  ~ {gap.live_class}.{gap.method}()  at {gap.probe_path}", file=out)
    if report.allowlist_gaps:
        print(
            "\n  Each of these costs every user a Live restart to enable "
            "(docs/protocol.md section 9). Add the '<Class>.<method>' line to "
            "METHOD_ALLOWLIST with its reason, then a catalog row.",
            file=out,
        )

    _section(out, "ALLOWLIST DRIFT", bool(report.allowlist_drift))
    for line in report.allowlist_drift:
        print(f"  ! {line}", file=out)

    _section(
        out,
        "NOT WRITTEN -- proposals this tool refuses to guess at",
        bool(report.rejected_rows),
    )
    for line in report.rejected_rows:
        print(f"  ? {line}", file=out)

    not_reached = report.by_verdict(NOT_REACHED)
    _section(out, "NOT REACHED -- nothing was learned about these rows", bool(not_reached))
    if not verbose and not_reached:
        by_reason = Counter(f.reason or "unstated" for f in not_reached)
        for reason, count in by_reason.most_common():
            print(f"  . {count:>4}  {reason}", file=out)
        print("  (--verbose lists them row by row)", file=out)
    else:
        for finding in not_reached:
            print(f"  . {finding.spec_id:<40} {finding.detail}", file=out)


def _section(out: Any, title: str, present: bool) -> None:
    if not present:
        return
    print(file=out)
    print(title, file=out)
    print("-" * len(title), file=out)


def limits_block(report: Report) -> str:
    """The deletions, as a block to paste into ``docs/limits.md`` §7.

    Deleting a row throws away the only place that knowledge was written down, so
    it is written out again before it goes. The table matches the one already in
    §7 so the two can be concatenated. This tool does not edit
    ``limits.md`` itself: that file is prose with an argument in it, and appending
    to it is a judgement call.
    """
    if not report.deletions:
        return ""
    lines = [
        f"### Removed by scripts/sync_catalog.py, {report.stamp}",
        "",
        "*Each of these was a catalog row naming a property the Live Object Model does",
        "not have; every one answered `no_such_path` on a determined probe. They were",
        "**deleted** rather than marked `broken`, because a catalog is an inventory of",
        "what exists and a fantasy row invites a caller to try (docs/catalog.md).*",
        "",
        "| Removed row | Class | Member | Path probed | Live's answer |",
        "|---|---|---|---|---|",
    ]
    for finding in report.deletions:
        target = target_of(finding.spec)
        member = target.member if target is not None else "?"
        message = finding.detail.split(": ", 1)[-1].replace("|", "\\|")
        lines.append(
            f"| `{finding.spec_id}` | `{finding.live_class or '?'}` | `{member}` | "
            f"`{finding.probe_path}` | {message} |"
        )
    return "\n".join(lines) + "\n"


def json_payload(report: Report, before: dict[str, int], after: dict[str, int]) -> dict[str, Any]:
    return {
        "stamp": report.stamp,
        "live_version": report.live_version,
        "script_version": report.script_version,
        "warnings": report.warnings,
        "catalog_before": before,
        "catalog_after": after,
        "counts": report.counts(),
        "new_rows": [row.as_dict() for row in report.new_rows],
        "deletions": [f.as_dict() for f in report.deletions],
        "refused": [f.as_dict() for f in report.by_verdict(REFUSED)],
        "mismatched": [f.as_dict() for f in report.by_verdict(MISMATCHED)],
        "status_changes": [f.as_dict() for f in report.status_changes],
        "not_reached": [f.as_dict() for f in report.by_verdict(NOT_REACHED)],
        "allowlist_gaps": [g.as_dict() for g in report.allowlist_gaps],
        "allowlist_drift": report.allowlist_drift,
        "rejected": report.rejected_rows,
        "containers": [
            {
                "container": group.container,
                "shape": group.shape,
                "rows": len(group.rows),
                "determined": group.determined,
                "unresolved": group.unresolved,
                "probes": [
                    {
                        "path": probe.probe_path,
                        "specimen": probe.resolution.specimen.class_name,
                        "live_class": probe.live_class,
                        "reached": probe.reached,
                        "code": probe.answer.code or None,
                    }
                    for probe in group.probes
                ],
            }
            for group in report.groups
        ],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/sync_catalog.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Reconcile src/ableton_maestro/catalog/*.yaml against a running Ableton "
            "Live, through lom_describe. Reports what Live has and the catalog does "
            "not, what the catalog has and Live does not, and what both agree on."
        ),
        epilog=textwrap.dedent(
            """\
            examples:
              python scripts/sync_catalog.py                     # report, change nothing
              python scripts/sync_catalog.py --area track --verbose
              python scripts/sync_catalog.py --record run.json   # keep the raw answers
              python scripts/sync_catalog.py --replay run.json   # reconcile offline
              python scripts/sync_catalog.py --add --status      # apply those two only
              python scripts/sync_catalog.py --write             # apply everything

            This tool never writes to Live and never calls a method: it sends
            lom_describe and lom_get and nothing else. The gate is on writing the
            CATALOG, and it will not do that without --write / --add / --delete /
            --status.

            A deletion is irreversible in the file, so every one of them is printed
            as a block for docs/limits.md section 7 before it happens. Keep it.
            """
        ),
    )

    what = parser.add_argument_group("what to do")
    what.add_argument(
        "--report",
        action="store_true",
        help="reconcile and print, change nothing -- the default",
    )
    what.add_argument(
        "--write", action="store_true", help="apply every proposed change to catalog/*.yaml"
    )
    what.add_argument("--add", action="store_true", help="apply only the new rows")
    what.add_argument(
        "--delete",
        action="store_true",
        help="apply only the deletions (rows whose target answered no_such_path)",
    )
    what.add_argument(
        "--status",
        action="store_true",
        help="apply only the status flips and their dated doc notes",
    )

    scope = parser.add_argument_group("scope")
    scope.add_argument(
        "--area",
        action="append",
        metavar="NAME",
        help="only rows in this catalog area (the id prefix: track, clip, ...); repeatable",
    )
    scope.add_argument(
        "--specimens",
        type=Path,
        metavar="FILE",
        help=(
            "JSON overriding the specimen map: {\"Track\": \"song.tracks[3]\"} or "
            "{\"Track\": {\"path\": \"...\", \"area\": \"track\"}}. null removes a default."
        ),
    )
    scope.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG_DIR,
        metavar="PATH",
        help="catalog directory or single file (default: the packaged catalog)",
    )
    scope.add_argument(
        "--script",
        type=Path,
        default=DEFAULT_SCRIPT,
        metavar="PATH",
        help="Remote Script to read METHOD_ALLOWLIST out of (default: this checkout's)",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--json", action="store_true", help="machine-readable report on stdout")
    output.add_argument(
        "--verbose", action="store_true", help="list every not-reached row instead of a tally"
    )
    output.add_argument(
        "--limits-report",
        type=Path,
        metavar="FILE",
        help="write the docs/limits.md paste-in block for the deletions to FILE",
    )
    output.add_argument(
        "--record",
        type=Path,
        metavar="FILE",
        help="save every answer Live gave, in the shape --replay reads back",
    )

    connection = parser.add_argument_group("connection")
    connection.add_argument(
        "--replay",
        type=Path,
        metavar="FILE",
        help="reconcile a recorded run instead of connecting to Live",
    )
    connection.add_argument("--host", default=DEFAULT_HOST, help=f"default {DEFAULT_HOST}")
    connection.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"default {DEFAULT_PORT}"
    )
    connection.add_argument(
        "--pace",
        type=float,
        default=DEFAULT_PACE,
        metavar="SECONDS",
        help=f"pause between round trips (default {DEFAULT_PACE})",
    )
    connection.add_argument(
        "--device-pace",
        type=float,
        default=DEFAULT_DEVICE_PACE,
        metavar="SECONDS",
        help=f"pause around paths that go through a device (default {DEFAULT_DEVICE_PACE})",
    )
    return parser


def select_rows(registry: Registry, areas: Sequence[str] | None) -> list[PathSpec]:
    """The rows this invocation is about, or raise :class:`ValueError`."""
    if not areas:
        return registry.all()
    known = {area_of(spec.id) for spec in registry.all()}
    unknown = sorted(set(areas) - known)
    if unknown:
        raise ValueError(
            f"no catalog area named {unknown}; the areas are {sorted(known)}"
        )
    wanted = set(areas)
    return [spec for spec in registry.all() if area_of(spec.id) in wanted]


def plan_from(args: argparse.Namespace) -> WritePlan:
    if args.write:
        return WritePlan(add=True, delete=True, status=True)
    return WritePlan(add=args.add, delete=args.delete, status=args.status)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Exit codes: 0 ran, 1 a usage or connection failure, 2 argparse's own usage
    error, 3 the catalog write failed and the files were put back as they were.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover - not a real stream
            pass

    parser = build_parser()
    args = parser.parse_args(argv)
    out = sys.stderr
    plan = plan_from(args)

    if args.report and plan.any:
        print(
            "usage error: --report changes nothing and contradicts --write/--add/"
            "--delete/--status. Drop one of them.",
            file=sys.stderr,
        )
        return CODE_USAGE
    print(_BANNER, file=out)

    try:
        registry = Registry.load(args.catalog)
    except CatalogError as exc:
        print(f"catalog error: {exc}", file=sys.stderr)
        return CODE_USAGE

    try:
        rows = select_rows(registry, args.area)
        specimens = load_specimens(args.specimens)
        allowlist = _literal_string_set(args.script, "METHOD_ALLOWLIST")
        describe_skip = _literal_string_set(args.script, "_DESCRIBE_SKIP")
    except ValueError as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return CODE_USAGE
    if not rows:
        print("no rows matched the selection.", file=sys.stderr)
        return CODE_USAGE

    replay: dict[str, Any] | None = None
    if args.replay is not None:
        try:
            replay = json.loads(args.replay.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"--replay {args.replay}: {exc}", file=sys.stderr)
            return CODE_USAGE
        if not isinstance(replay, dict):
            print(f"--replay {args.replay}: top level must be an object", file=sys.stderr)
            return CODE_USAGE

    before = print_summary("catalog before", registry, out)
    print(
        f"  reconciling {len(rows)} row(s) against {len(specimens)} specimen(s)"
        + (f" from {args.replay}" if replay is not None else f" via {args.host}:{args.port}"),
        file=out,
    )

    client: AbletonClient | None = None
    try:
        if replay is None:
            client = AbletonClient(args.host, args.port)
            client.connect()
        reader = Reader(
            client, replay=replay, pace=args.pace, device_pace=args.device_pace
        )
        report = Reconciler(
            registry,
            reader,
            specimens,
            allowlist=allowlist,
            describe_skip=describe_skip,
            catalog_root=args.catalog,
            rows=rows,
        ).run()
    except AbletonConnectionError as exc:
        print(f"connection: {exc}", file=sys.stderr)
        return CODE_USAGE
    except AbletonError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return CODE_USAGE
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("\naborted. Nothing was written; this tool reads Live and writes files.",
              file=sys.stderr)
        return 130
    finally:
        if client is not None:
            client.close()

    if args.record is not None:
        args.record.write_text(
            json.dumps(reader.recording(), indent=2, ensure_ascii=False, default=repr),
            encoding="utf-8",
        )
        source = "Live" if replay is None else "the replay, unchanged"
        print(
            f"\nrecorded {reader.round_trips} round trip(s) to {source} -> {args.record}",
            file=out,
        )

    print_report(report, out, verbose=args.verbose)

    block = limits_block(report)
    if block:
        if args.limits_report is not None:
            args.limits_report.write_text(block, encoding="utf-8")
            print(f"\nlimits.md block written to {args.limits_report}", file=out)
        else:
            print(file=out)
            print("PASTE INTO docs/limits.md SECTION 7 BEFORE APPLYING THE DELETIONS", file=out)
            print("=" * 78, file=out)
            print(block, file=out, end="")
            print("=" * 78, file=out)

    after = before
    if plan.any:
        try:
            changed = CatalogSync(args.catalog).apply(report, plan)
        except CatalogWriteError as exc:
            print(f"\nwrite failed: {exc}", file=sys.stderr)
            return CODE_WRITE_FAILED
        print(f"\napplied {len(changed)} change(s):", file=out)
        for line in changed:
            print(f"  {line}", file=out)
        try:
            after = print_summary("catalog after", Registry.load(args.catalog), out)
        except CatalogError as exc:  # pragma: no cover - _write_verified rules it out
            print(f"catalog no longer loads: {exc}", file=sys.stderr)
            return CODE_WRITE_FAILED
        print(
            "\nRun pytest before committing: tests/test_catalog.py enforces the pairing "
            "of a status with its note in both directions.",
            file=out,
        )
    else:
        print(
            "\nnothing was written. Pass --write, or --add / --delete / --status "
            "for one class of change at a time.",
            file=out,
        )

    if args.json:
        print(
            json.dumps(
                json_payload(report, before, after), indent=2, ensure_ascii=False, default=repr
            )
        )
    return CODE_OK


if __name__ == "__main__":
    raise SystemExit(main())
