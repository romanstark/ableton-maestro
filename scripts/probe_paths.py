#!/usr/bin/env python
"""Probe catalog paths against a live Ableton session and update status.

Tests catalog paths against a running Live instance to verify property readability,
writability, and value restoration, writing results back to YAML catalog files.

Probe outcomes:
- ``verified``: path reads successfully, writes a safe test value, and restores the original value.
- ``broken``: target exists but property is absent or write is rejected.
- ``inconclusive``: target container is missing in current session (no status change written).

Safety constraints:
- Dry-run by default; requires ``--go`` to execute commands or update catalog files.
- Destructive operations require ``--include-destructive`` plus interactive confirmation.
- Protects non-empty projects unless ``--i-know-what-im-doing`` is supplied.
- Paces writes and immediately aborts if value restoration fails.
"""

from __future__ import annotations

import argparse
import difflib
import json
import platform
import re
import sys
import textwrap
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# scripts/ is not a package and this file is run as a path, not as a module. A
# checkout that has not been `pip install -e .`-ed still has src/ next door.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_REPO_ROOT / "src"))

# Imported after the path bootstrap above, which is why they are not at the very
# top of the file.
import yaml

from ableton_maestro.client import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    AbletonClient,
    AbletonCommandError,
    AbletonConnectionError,
    AbletonError,
    AbletonTimeoutError,
)
from ableton_maestro.executor import Result, execute, execute_batch
from ableton_maestro.models import Access, Kind, PathStatus
from ableton_maestro.registry import (
    DEFAULT_CATALOG_DIR,
    CatalogError,
    Registry,
    area_of,
)
from ableton_maestro.spec import PathSpec, build_path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: The three rows that decide whether a generic LOM bridge reaches a clip's warp state
#: and pitch at all, which is the case docs/architecture.md argues under 'The restart
#: tax'. All three were read, written, read back and restored on 2026-08-29 against Live
#: 12.4.5 and now carry ``verified`` in the catalog. They keep a flag of their own
#: because probing them together is one invocation rather than three.
FIRST_THREE: tuple[str, ...] = ("clip.warping", "clip.warp_mode", "clip.pitch_coarse")

#: Rows whose write starts an *action* rather than storing a value: transport,
#: recording, arming. Restoring them afterwards stops playback again, but a
#: record pass that ran for two seconds has already put something in the set.
#:
#: This list is not derived from the catalog, and that is deliberate:
#: ``destructive`` marks what destroys, this marks what *acts*. Two different
#: questions, and folding them together would either arm the confirm prompt for
#: harmless rows or leave these ones unguarded.
ACTION_ROWS: frozenset[str] = frozenset(
    {
        "song.is_playing",
        "song.record_mode",
        "song.session_record",
        "song.overdub",
        "song.arrangement_overdub",
        "song.session_automation_record",
        "song.is_counting_in",
        "clip.is_playing",
        "arrangement_clip.is_playing",
        "track.arm",
        "track.implicit_arm",
        "return.arm",
    }
)

#: Pause between rows. Precaution, not a measurement: what *is* measured is that
#: loading plugins in rapid succession can crash Live (CONTRIBUTING.md *Warnings*).
#: A probe loads nothing, so this is cheap insurance against hammering Live's
#: main thread.
DEFAULT_PACE = 0.15

#: Pause after touching a path that goes through a device. Device parameters are
#: plugin code running next to a real-time audio thread; a round trip costs ~450 ms
#: anyway (docs/limits.md), so this is nearly free.
DEFAULT_DEVICE_PACE = 0.75

#: How much of a value a permanent catalog note may quote.
_NOTE_VALUE_CHARS = 80

#: Client-side refusal code, for a request the executor would not even build.
#: It never appears on the wire; docs/protocol.md section 4 lists the server's.
CODE_BAD_REQUEST = "bad_request"

#: Exit codes. ``2`` is left to argparse, which uses it for a usage error.
CODE_RESTORE_FAILED = 3
CODE_REFUSED = 4

#: Verdicts. ``INCONCLUSIVE`` is the one that must never be silently upgraded.
VERDICT_VERIFIED = "verified"
VERDICT_BROKEN = "broken"
VERDICT_INCONCLUSIVE = "inconclusive"
VERDICT_SKIPPED = "skipped"

#: Wire error codes that mean *this set has no such target*, never *this row is
#: wrong* (docs/protocol.md §4).
TARGET_ABSENT_CODES: frozenset[str] = frozenset({"index_out_of_range"})

#: How many alternative targets one row may be tried against (an audio clip, then
#: a MIDI clip, for instance) before it is called inconclusive.
MAX_CANDIDATES = 3

#: Live's own default track names, as a shape: "1 MIDI", "3 Audio", "2-Audio".
#: Read from Live's naming convention, not measured across versions or
#: locales, and it only feeds the "does this look like real work" heuristic, where
#: a false alarm costs a flag and a wrong pass costs somebody's set.
_DEFAULT_TRACK_NAME = re.compile(r"^\d+[\s_-]*(midi|audio)$", re.IGNORECASE)

#: Thresholds for that heuristic. Deliberately low.
GUARD_LIMITS: dict[str, int] = {
    "renamed tracks": 3,
    "clips": 8,
    "tracks": 12,
    "named scenes": 2,
    "devices": 8,
}

#: Float comparison tolerance for read-back, mirroring ``executor._same_value``.
#: A stored float returns through JSON, and bit-exact comparison would report a
#: mismatch on a value that round-tripped perfectly well.
_REL_TOL = 1e-6
_ABS_TOL = 1e-9

_BANNER = """
================================================================================
  probe_paths.py WRITES INTO A RUNNING ABLETON LIVE SET.

  Use a throwaway set. Not a production, not something you care about.
  Every write is read back and restored -- but a crash mid-probe is not
  undone, and Live's undo is not a backup.
================================================================================
"""


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #


def _same_value(first: object, second: object) -> bool:
    """Compare a read-back value with the one that was asked for.

    Mirrors ``executor._same_value``: bools compare as bools (Python's ``bool`` is
    an ``int`` subclass and ``True == 1``), numbers compare with a tolerance,
    everything else compares with ``==``.
    """
    if isinstance(first, bool) or isinstance(second, bool):
        return bool(first) is bool(second)
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return abs(float(first) - float(second)) <= max(
            _ABS_TOL, _REL_TOL * max(abs(float(first)), abs(float(second)))
        )
    return bool(first == second)


def _note_value(value: object) -> str:
    """A value as it should appear in a catalog note that lives forever.

    A scalar goes in verbatim: it *is* the measurement. A collection does not.
    ``song.tracks`` on the reference set is 53 handles, and pasting that into a
    ``doc`` would bury the sentence it belongs to and blow up the diff.

    ASCII only, deliberately: the note is written into a YAML file that other
    tools and other people read, and an ellipsis character there buys nothing.
    """
    if isinstance(value, (list, tuple)):
        return f"a list of {len(value)} item(s)"
    if isinstance(value, Mapping):
        handle = value.get("__lom__")
        return f"a {handle} handle" if handle else f"an object with {len(value)} field(s)"
    text = repr(value)
    return text if len(text) <= _NOTE_VALUE_CHARS else text[:_NOTE_VALUE_CHARS] + "..."


def _short(value: object, limit: int = 40) -> str:
    """A one-line rendering of a value, for a table cell."""
    if value is None:
        return "-"
    if isinstance(value, str):
        text = value if value else "''"
    elif isinstance(value, (list, tuple)):
        text = f"[{len(value)} items]"
    elif isinstance(value, Mapping):
        handle = value.get("__lom__")
        text = f"<{handle}>" if handle else "{...}"
    else:
        text = repr(value)
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _pause(seconds: float) -> None:
    """Wait between operations.

    Its own function so that the probe loop and the prober pace the same way, and
    so that ``--pace 0`` really is free rather than a zero-length sleep on every
    single row.
    """
    if seconds > 0:
        time.sleep(seconds)


def _parent_path(path: str) -> str | None:
    """The path one step up, or ``None`` at the root.

    ``song.tracks[0].clip_slots[0].clip.name`` -> ``song.tracks[0].clip_slots[0].clip``.
    An index step is stripped too: ``song.tracks[0]`` -> ``song.tracks``.
    """
    if path.endswith("]"):
        head = path[: path.rfind("[")]
        return head or None
    if "." not in path:
        return None
    return path.rsplit(".", 1)[0]


# --------------------------------------------------------------------------- #
# Transport wrapper
# --------------------------------------------------------------------------- #


class RecordingClient:
    """An :class:`AbletonClient` that keeps the raw reply of its last operation.

    The executor maps a wire reply onto a :class:`~ableton_maestro.executor.Result`,
    which carries everything an *operation* needs, and drops two fields a *probe*
    exists to report: the LOM ``type`` the script put on the value
    (docs/protocol.md §5.3) and, on a write, ``is_quantized``.

    Re-reading the path to recover them would cost a round trip and, worse, would
    read a value at a different moment than the one being reported. So the raw
    reply is kept here instead, and :attr:`last` is cleared at the start of every
    call: a stale reply misattributed to the next row is exactly the class of
    quiet error this project is built against.

    Implements the ``LomClient`` protocol of ``executor.py`` and nothing else;
    :attr:`inner` is there for the two things that go around the executor:
    ``lom_describe`` for ``call`` rows, and the raw restore of last resort.
    """

    def __init__(self, inner: AbletonClient) -> None:
        self.inner = inner
        self.last: dict[str, Any] = {}

    def get(self, path: str) -> Mapping[str, Any]:
        self.last = {}
        self.last = dict(self.inner.get(path))
        return self.last

    def set(self, path: str, value: object) -> Mapping[str, Any]:
        self.last = {}
        self.last = dict(self.inner.set(path, value))
        return self.last

    def call(self, path: str, method: str, args: Sequence[object]) -> Mapping[str, Any]:
        self.last = {}
        self.last = dict(self.inner.call(path, method, list(args)))
        return self.last

    def batch(self, ops: Sequence[Mapping[str, Any]], *, atomic: bool = False) -> Mapping[str, Any]:
        self.last = {}
        self.last = dict(self.inner.batch(ops, atomic=atomic))
        return self.last


# --------------------------------------------------------------------------- #
# The session survey
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrackInfo:
    """One track, as far as auto-picking a target needs to know it."""

    index: int
    name: str
    is_group: bool
    is_midi: bool
    device_count: int

    @property
    def renamed(self) -> bool:
        """True when the name is not one of Live's own defaults (a heuristic)."""
        return bool(self.name) and not _DEFAULT_TRACK_NAME.match(self.name)


@dataclass(frozen=True)
class ClipInfo:
    """One clip found in the session grid."""

    track: int
    slot: int
    is_audio: bool
    name: str


@dataclass
class Session:
    """What the probe learned about the set before it touched anything.

    Gathered with ``lom_batch``: four round trips rather than four hundred
    (docs/limits.md: one round trip costs ~450 ms, measured).
    """

    tracks: list[TrackInfo] = field(default_factory=list)
    scenes: list[str] = field(default_factory=list)
    returns: list[str] = field(default_factory=list)
    clips: list[ClipInfo] = field(default_factory=list)
    live_version: str = "unknown"
    script_version: str = "unknown"
    tempo: float | None = None
    incomplete: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- targets
    @property
    def playable_tracks(self) -> list[TrackInfo]:
        """Tracks that are not group tracks.

        Group tracks are the standing trap (docs/protocol.md §6): they have no arm state
        and no arrangement clips, and unguarded access raises. Nothing here
        auto-picks one.
        """
        return [t for t in self.tracks if not t.is_group]

    @property
    def audio_clips(self) -> list[ClipInfo]:
        return [c for c in self.clips if c.is_audio]

    @property
    def midi_clips(self) -> list[ClipInfo]:
        return [c for c in self.clips if not c.is_audio]

    def first_track_with_device(self) -> TrackInfo | None:
        return next((t for t in self.playable_tracks if t.device_count > 0), None)

    def summary_lines(self) -> list[str]:
        """What the probe saw, in the form it is shown to the user."""
        groups = [t for t in self.tracks if t.is_group]
        renamed = [t for t in self.tracks if t.renamed]
        named_scenes = [s for s in self.scenes if s]
        lines = [
            f"Live {self.live_version}, script {self.script_version}"
            + (f", tempo {self.tempo:g}" if self.tempo is not None else ""),
            f"tracks:  {len(self.tracks)}  ({len(groups)} group, "
            + f"{len(renamed)} renamed, "
            + f"{sum(t.device_count for t in self.tracks)} devices)",
            f"scenes:  {len(self.scenes)}  ({len(named_scenes)} named)",
            f"returns: {len(self.returns)}",
            f"clips:   {len(self.clips)}  "
            + f"({len(self.audio_clips)} audio, {len(self.midi_clips)} MIDI)",
        ]
        for track in self.tracks:
            flags = "".join(
                (
                    "G" if track.is_group else " ",
                    "M" if track.is_midi else "A",
                )
            )
            lines.append(
                f"  [{track.index:>2}] {flags} {track.name!r}"
                f"  devices={track.device_count}" + ("  (renamed)" if track.renamed else "")
            )
        for clip in self.clips[:12]:
            kind = "audio" if clip.is_audio else "midi"
            lines.append(f"  clip  track {clip.track} slot {clip.slot}: {kind} {clip.name!r}")
        if len(self.clips) > 12:
            lines.append(f"  ... and {len(self.clips) - 12} more clips")
        for note in self.incomplete:
            lines.append(f"  ! survey incomplete: {note}")
        return lines


def _batch_get(
    client: RecordingClient, registry: Registry, ops: Sequence[Mapping[str, Any]]
) -> list[Result]:
    """Run read-only catalog ops in one round trip; missing rows drop out.

    The survey is a diagnostic, not a measurement, so a catalog that has been
    trimmed (``--catalog`` pointing at one file, for instance) degrades the survey
    rather than stopping the run.
    """
    usable = [op for op in ops if _has_row(registry, str(op.get("id", "")))]
    if not usable:
        return []
    return execute_batch(client, registry, usable, verify=False)


def _has_row(registry: Registry, spec_id: str) -> bool:
    try:
        registry.get(spec_id)
    except KeyError:
        return False
    return True


def _first_index(path: str) -> int | None:
    """The first ``[n]`` in a concrete path, or ``None``.

    A blocked op never reaches this: its ``path`` is still the row's template and
    ``[{track}]`` is not a number. Returning ``None`` rather than raising keeps a
    survey from dying on one op that never went out.
    """
    match = re.search(r"\[(\d+)\]", path)
    return int(match.group(1)) if match else None


def _handles(result: Result | None) -> list[Mapping[str, Any]]:
    """The list of LOM handles a list-valued ``lom_get`` returned (§7)."""
    if result is None or not result.ok or not isinstance(result.value, list):
        return []
    return [item for item in result.value if isinstance(item, Mapping)]


def survey_session(client: RecordingClient, registry: Registry, *, max_slots: int) -> Session:
    """Read enough of the open set to pick targets and to judge whether to run.

    Four batches: the collections, then per-track flags, then the clip grid, then
    the clips themselves. Nothing here writes.
    """
    session = Session()

    try:
        info = client.inner.script_info()
        session.live_version = str(info.get("live_version") or "unknown")
        session.script_version = str(info.get("script_version") or "unknown")
    except AbletonError as exc:  # pragma: no cover - needs a live socket
        session.incomplete.append(f"script_info failed: {exc}")

    collections = _batch_get(
        client,
        registry,
        [
            {"id": "song.tracks"},
            {"id": "song.scenes"},
            {"id": "song.return_tracks"},
            {"id": "song.tempo"},
        ],
    )
    by_id = {r.id: r for r in collections}
    track_handles = _handles(by_id.get("song.tracks"))
    session.scenes = [str(h.get("name") or "") for h in _handles(by_id.get("song.scenes"))]
    session.returns = [str(h.get("name") or "") for h in _handles(by_id.get("song.return_tracks"))]
    tempo = by_id.get("song.tempo")
    if tempo is not None and tempo.ok and isinstance(tempo.value, (int, float)):
        session.tempo = float(tempo.value)
    if not track_handles:
        session.incomplete.append("song.tracks returned no track handles")
        return session

    flags = _batch_get(
        client,
        registry,
        [
            op
            for index in range(len(track_handles))
            for op in (
                {"id": "track.is_foldable", "track": index},
                {"id": "track.has_midi_input", "track": index},
                {"id": "track.devices", "track": index},
            )
        ],
    )
    per_track: dict[int, dict[str, object]] = {i: {} for i in range(len(track_handles))}
    for result in flags:
        if not result.ok:
            continue
        index = _first_index(result.path)
        if index is not None and index in per_track:
            per_track[index][result.id] = result.value

    for index, handle in enumerate(track_handles):
        values = per_track.get(index, {})
        devices = values.get("track.devices")
        session.tracks.append(
            TrackInfo(
                index=index,
                name=str(handle.get("name") or ""),
                is_group=bool(values.get("track.is_foldable")),
                is_midi=bool(values.get("track.has_midi_input")),
                device_count=len(devices) if isinstance(devices, list) else 0,
            )
        )

    slots = min(len(session.scenes), max_slots)
    candidates = [t.index for t in session.playable_tracks][:max_slots]
    if slots and candidates:
        grid = _batch_get(
            client,
            registry,
            [
                {"id": "clip_slot.has_clip", "track": track, "slot": slot}
                for track in candidates
                for slot in range(slots)
            ],
        )
        occupied = [
            (track, slot)
            for (track, slot), result in zip(
                [(t, s) for t in candidates for s in range(slots)], grid, strict=False
            )
            if result.ok and result.value is True
        ]
        details = _batch_get(
            client,
            registry,
            [
                op
                for track, slot in occupied
                for op in (
                    {"id": "clip.is_audio_clip", "track": track, "slot": slot},
                    {"id": "clip.name", "track": track, "slot": slot},
                )
            ],
        )
        paired = {(r.path, r.id): r for r in details}
        for track, slot in occupied:
            prefix = f"song.tracks[{track}].clip_slots[{slot}].clip"
            is_audio = paired.get((f"{prefix}.is_audio_clip", "clip.is_audio_clip"))
            name = paired.get((f"{prefix}.name", "clip.name"))
            session.clips.append(
                ClipInfo(
                    track=track,
                    slot=slot,
                    is_audio=(
                        bool(is_audio.value) if is_audio is not None and is_audio.ok else False
                    ),
                    name=str(name.value) if name is not None and name.ok else "",
                )
            )
    elif not slots:
        session.incomplete.append("no scenes, so no clip slots were scanned")

    return session


# --------------------------------------------------------------------------- #
# Safety gate: does this look like somebody's real work?
# --------------------------------------------------------------------------- #


def guard_signals(session: Session) -> dict[str, int]:
    """Count the things that make a set look like a production in progress."""
    return {
        "renamed tracks": sum(1 for t in session.tracks if t.renamed),
        "clips": len(session.clips),
        "tracks": len(session.tracks),
        "named scenes": sum(1 for s in session.scenes if s),
        "devices": sum(t.device_count for t in session.tracks),
    }


def guard_session(session: Session, *, override: bool, out: Any) -> bool:
    """Print what the probe saw and decide whether it may proceed.

    Returns True when the run may go ahead. The thresholds are low on purpose:
    a false alarm costs one flag, and a missed alarm costs somebody's set.
    """
    signals = guard_signals(session)
    tripped = {name: value for name, value in signals.items() if value >= GUARD_LIMITS[name]}

    print("This is the set the probe is looking at:", file=out)
    for line in session.summary_lines():
        print(f"  {line}", file=out)
    print(file=out)
    for name, value in signals.items():
        mark = "!!" if name in tripped else "  "
        print(f"  {mark} {name:<16} {value:>4}   (limit {GUARD_LIMITS[name]})", file=out)
    print(file=out)

    if not tripped:
        return True
    listed = ", ".join(f"{name}={value}" for name, value in tripped.items())
    if override:
        print(
            f"This set looks like real work ({listed}), and --i-know-what-im-doing\n"
            "was passed. Continuing. Consider saving before anything else happens.",
            file=out,
        )
        return True
    print(
        f"REFUSING TO RUN. This set looks like real work: {listed}.\n"
        "Open a throwaway set instead -- a few tracks, a few clips, nothing you\n"
        "would miss. If this really is a scratch set, pass --i-know-what-im-doing.",
        file=out,
    )
    return False


# --------------------------------------------------------------------------- #
# Choosing a target and a safe value
# --------------------------------------------------------------------------- #


def _needs_clip(spec: PathSpec) -> bool:
    return "clip_slots[{slot}].clip" in spec.path


def _needs_clip_slot(spec: PathSpec) -> bool:
    return "clip_slots[{slot}]" in spec.path and not _needs_clip(spec)


def _needs_device(spec: PathSpec) -> bool:
    return "devices[{device}]" in spec.path


def candidate_args(
    spec: PathSpec, session: Session, overrides: Mapping[str, object]
) -> tuple[list[dict[str, object]], str]:
    """Return the argument dicts to try for ``spec``, and why there might be none.

    More than one candidate exists for rows that address a clip: an audio clip is
    tried first, then a MIDI clip. Which of the two a row needs is not written
    down anywhere machine-readable (``clip.warping`` wants audio, ``clip.scale_name``
    wants MIDI), and guessing from the id would be a second, drifting catalog. So
    the probe simply tries, and reports which target answered.

    Explicit ``--track``/``--slot``/… always win over an auto-pick, on every
    candidate.
    """
    names = [p.name for p in spec.params]
    if not names:
        return [{}], ""

    base: dict[str, object] = {}
    for param in spec.params:
        if param.name == "root":
            allowed = list(param.enum or [])
            base[param.name] = (
                "instruments"
                if "instruments" in allowed
                else (allowed[0] if allowed else "instruments")
            )
        else:
            base[param.name] = 0

    variants: list[dict[str, object]] = []

    if _needs_clip(spec) and not ("track" in overrides and "slot" in overrides):
        picks = _ordered_clips(session)
        if "track" in overrides:
            # A pinned track wins: look for a clip on *that* track rather than
            # pairing its number with a slot that holds a clip somewhere else.
            picks = [clip for clip in picks if clip.track == overrides["track"]]
        elif not picks:
            return [], "no clip in this set"
        for clip in picks[:MAX_CANDIDATES]:
            variant = dict(base)
            variant["track"] = clip.track
            variant["slot"] = clip.slot
            variants.append(variant)
        if not variants:
            # The pinned track has no clip. Ask anyway and let Live answer: that
            # is a fact about this set, and the caller asked for this target.
            variants.append(dict(base))
    elif _needs_clip_slot(spec) and "track" not in overrides:
        variant = dict(base)
        if session.clips:
            variant["track"] = session.clips[0].track
            variant["slot"] = session.clips[0].slot
        elif session.playable_tracks:
            variant["track"] = session.playable_tracks[0].index
        else:
            return [], "no usable track in this set"
        variants.append(variant)
    elif _needs_device(spec) and "track" not in overrides:
        track = session.first_track_with_device()
        if track is None:
            return [], "no track carries a device in this set"
        variant = dict(base)
        variant["track"] = track.index
        variants.append(variant)
    elif "track" in names and "track" not in overrides:
        tracks = session.playable_tracks
        if not tracks:
            return [], "no non-group track in this set"
        variant = dict(base)
        variant["track"] = tracks[0].index
        variants.append(variant)
    else:
        variants.append(dict(base))

    for variant in variants:
        for name, value in overrides.items():
            if name in variant:
                variant[name] = value

    deduped: list[dict[str, object]] = []
    for variant in variants:
        if variant not in deduped:
            deduped.append(variant)
    return deduped, ""


def _ordered_clips(session: Session) -> list[ClipInfo]:
    """Clips in the order a probe should try them: audio, then MIDI, then rest."""
    ordered: list[ClipInfo] = []
    if session.audio_clips:
        ordered.append(session.audio_clips[0])
    if session.midi_clips:
        ordered.append(session.midi_clips[0])
    for clip in session.clips:
        if clip not in ordered:
            ordered.append(clip)
    return ordered


def safe_values(spec: PathSpec, current: object) -> list[tuple[object, str]]:
    """Safe write candidates for ``spec``, given what is there now.

    "Safe" means three things at once: inside the row's declared range, as close
    to the current value as still counts as a change, and trivially restorable.
    The maximum is never chosen: a probe that leaves a parameter at maximum
    is the measured accident this script exists to avoid.

    A candidate must differ from the current value. Writing the same value back
    would prove only that the property accepts assignment; it cannot tell a
    stored write from one Live silently discarded, which is the exact failure
    mode the read-back was introduced for (docs/architecture.md, 'read-back as a principle').

    A second candidate is offered where one exists, for the case where the first
    write comes back ``changed: false``: a quantized parameter can round a nudge
    straight back to where it started.
    """
    kind = spec.kind
    if kind is Kind.BOOL:
        if not isinstance(current, bool):
            return []
        return [(not current, "the other boolean state")]

    if kind is Kind.ENUM or (spec.enum and kind in (Kind.INT, Kind.STR)):
        members = [v for v in (spec.enum or []) if not _same_value(v, current)]
        return [(value, "another declared enum member") for value in members[:2]]

    if kind is Kind.INT:
        if not isinstance(current, int) or isinstance(current, bool):
            return []
        return [
            (value, "one step from the current value")
            for value in _numeric_candidates(float(current), spec, step=1.0)
        ]

    if kind is Kind.FLOAT:
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            return []
        step, why = _float_step(spec)
        return [
            (value, why)
            for value in _numeric_candidates(float(current), spec, step=step, integral=False)
        ]

    if kind is Kind.STR:
        if not isinstance(current, str):
            return []
        marked = (current + " [probe]")[:120]
        if marked == current:  # pragma: no cover - only for a 120-char name
            return []
        return [(marked, "the current text with a visible probe marker")]

    # LIST and OBJECT: a collection is read-only in the LOM and an object never
    # travels as itself (docs/protocol.md §7). There is nothing safe to write.
    return []


def _float_step(spec: PathSpec) -> tuple[float, str]:
    """How far to move a float, and the sentence that explains it.

    Two goals pull against each other. Small is safe: the value is restored a
    moment later, but a probe should never make a set lurch. Large is detectable:
    a step Live rounds straight back is indistinguishable from a write it silently
    discarded, and telling those two apart is the whole job.

    Step size is one percent of the declared range, capped at 1.0 in the
    parameter's own units. A normalised 0..1 parameter moves by 0.01; a tempo,
    whose range is 20..999, moves by 1 BPM rather than by ten.

    A row the catalog marks ``quantized`` gets five percent instead and no cap.
    A quantized parameter takes discrete steps (measured: a requested 0.35 came
    back as 0.25), and a nudge smaller than one step is swallowed whole.
    """
    low, high = spec.range if spec.range else (None, None)
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        span = abs(float(high) - float(low))
    else:
        span = 1.0
    if spec.quantized:
        return (span * 0.05) or 0.05, "five percent of the range (the row is quantized)"
    return min(span * 0.01, 1.0) or 0.01, "a one-percent nudge, restored afterwards"


def _numeric_candidates(
    current: float, spec: PathSpec, *, step: float, integral: bool = True
) -> list[object]:
    """Values ``step`` above and below ``current``, kept inside the row's range."""
    low, high = spec.range if spec.range else (None, None)
    out: list[object] = []
    for offset in (step, -step):
        value = current + offset
        if isinstance(low, (int, float)) and value < float(low):
            continue
        if isinstance(high, (int, float)) and value > float(high):
            continue
        # Never park a parameter at its declared maximum, even transiently.
        if isinstance(high, (int, float)) and _same_value(value, float(high)):
            continue
        candidate: object = round(value) if integral else float(value)
        if not _same_value(candidate, current):
            out.append(candidate)
    return out


# --------------------------------------------------------------------------- #
# The probe itself
# --------------------------------------------------------------------------- #


@dataclass
class ProbeOutcome:
    """What one row's probe established, and what it proposes to write back."""

    spec_id: str
    template: str
    verdict: str = VERDICT_SKIPPED
    path: str | None = None
    args: dict[str, object] = field(default_factory=dict)
    reachable: bool | None = None
    readable: bool | None = None
    writable: bool | None = None
    clamped: bool | None = None
    restored: bool | None = None
    lom_type: str | None = None
    display: str | None = None
    before: object = None
    wrote: object = None
    after: object = None
    error_code: str | None = None
    detail: str = ""
    note: str = ""
    restore_failed: bool = False

    @property
    def writes_back(self) -> bool:
        """True when this outcome may change the catalog row.

        An inconclusive probe never does: nothing was learned about the row, and
        a status written from nothing is worse than no status at all
        (docs/catalog.md, *Three outcomes, not two*).
        """
        return self.verdict in (VERDICT_VERIFIED, VERDICT_BROKEN) and bool(self.note)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.spec_id,
            "verdict": self.verdict,
            "path": self.path,
            "args": self.args,
            "reachable": self.reachable,
            "readable": self.readable,
            "writable": self.writable,
            "clamped": self.clamped,
            "restored": self.restored,
            "lom_type": self.lom_type,
            "display": self.display,
            "before": self.before,
            "wrote": self.wrote,
            "after": self.after,
            "error_code": self.error_code,
            "detail": self.detail,
            "note": self.note,
        }


class RestoreFailed(RuntimeError):
    """A written value could not be put back. The set is dirty; stop everything.

    Raised rather than returned: every further write would add to a mess the
    operator does not yet know about. The message carries the path and the
    original value so a human can repair it by hand, and :attr:`outcome` carries
    the half-finished measurement so the report still shows what was read and
    what was written.
    """

    def __init__(self, message: str, outcome: ProbeOutcome | None = None) -> None:
        super().__init__(message)
        self.outcome = outcome


class Prober:
    """Runs rows against Live, one at a time, restoring everything it touches."""

    def __init__(
        self,
        client: RecordingClient,
        registry: Registry,
        session: Session,
        *,
        overrides: Mapping[str, object],
        pace: float,
        device_pace: float,
        read_only: bool,
        include_calls: bool,
        include_destructive: bool,
        include_transport: bool,
        confirm: Any,
        out: Any,
    ) -> None:
        self.client = client
        self.registry = registry
        self.session = session
        self.overrides = dict(overrides)
        self.pace = pace
        self.device_pace = device_pace
        self.read_only = read_only
        self.include_calls = include_calls
        self.include_destructive = include_destructive
        self.include_transport = include_transport
        self.confirm = confirm
        self.out = out
        self._describes: dict[str, Mapping[str, Any] | None] = {}
        self.stamp = _stamp(session)

    # ---------------------------------------------------------------- driving
    def probe(self, spec: PathSpec) -> ProbeOutcome:
        """Probe one row end to end and return what was established."""
        outcome = ProbeOutcome(spec_id=spec.id, template=spec.path)

        # The destructive gate stands in front of every access, not only in front
        # of calls: ``execute`` refuses a destructive row for a *read* as well, and
        # a row that is confirmed here has to carry that confirmation onto the wire.
        if spec.destructive:
            if not self.include_destructive:
                return self._skip(
                    outcome,
                    "destructive row; pass --include-destructive and confirm it by hand",
                )
            if not self._confirm_destructive(spec):
                return self._skip(outcome, "destructive row not confirmed")
        if Access.CALL in spec.access:
            return self._probe_call(spec, outcome)

        candidates, why_not = candidate_args(spec, self.session, self.overrides)
        if not candidates:
            outcome.verdict = VERDICT_INCONCLUSIVE
            outcome.detail = why_not or "no target could be resolved for this row"
            return outcome

        read = self._read_first_reachable(spec, candidates, outcome)
        if read is None or not read.ok:
            return outcome

        if not spec.supports(Access.SET):
            outcome.verdict = VERDICT_VERIFIED
            outcome.detail = f"read {_short(outcome.before)} ({outcome.lom_type}); read-only row"
            outcome.note = (
                f"Read verified {self.stamp} with scripts/probe_paths.py on "
                f"{outcome.path} = {_note_value(outcome.before)} "
                f"(LOM type {outcome.lom_type}){_display_clause(outcome.display)}; a write "
                f"was NOT attempted on this row."
            )
            return outcome

        if self.read_only:
            outcome.verdict = VERDICT_VERIFIED
            outcome.detail = f"read {_short(outcome.before)}; --read-only, no write attempted"
            outcome.note = (
                f"Read verified {self.stamp} with scripts/probe_paths.py --read-only on "
                f"{outcome.path} = {_note_value(outcome.before)} "
                f"(LOM type {outcome.lom_type}){_display_clause(outcome.display)}; a write "
                f"was NOT attempted on this row."
            )
            return outcome

        if spec.id in ACTION_ROWS and not self.include_transport:
            outcome.verdict = VERDICT_INCONCLUSIVE
            outcome.detail = (
                "read only: writing this row starts playback, recording or arming; "
                "pass --include-transport to write it"
            )
            return outcome

        return self._probe_write(spec, outcome)

    def _skip(self, outcome: ProbeOutcome, reason: str) -> ProbeOutcome:
        outcome.verdict = VERDICT_SKIPPED
        outcome.detail = reason
        return outcome

    # ------------------------------------------------------------------ reads
    def _read_first_reachable(
        self, spec: PathSpec, candidates: Sequence[Mapping[str, object]], outcome: ProbeOutcome
    ) -> Result | None:
        """Read the row against each candidate target until one answers."""
        last_failure: Result | None = None
        for index, args in enumerate(candidates):
            outcome.args = dict(args)
            try:
                result = execute(
                    self.client,
                    self.registry,
                    spec.id,
                    verify=False,
                    confirm=spec.destructive,
                    **args,
                )
            except ValueError as exc:
                # A client-side refusal: an index the row's own ParamSpec rejects,
                # for instance. Nothing was sent, so nothing was learned.
                outcome.verdict = VERDICT_INCONCLUSIVE
                outcome.detail = f"could not build the request: {exc}"
                return None
            outcome.path = result.path
            if result.ok:
                outcome.reachable = True
                outcome.readable = True
                outcome.before = result.value
                outcome.lom_type = str(self.client.last.get("type") or "")
                outcome.display = _opt_str(self.client.last.get("display"))
                if index:
                    outcome.detail = f"resolved on candidate {index + 1}"
                return result
            last_failure = result
            outcome.error_code = result.code
            if not self._target_absent(result):
                break
            _pause(self.pace)

        outcome.reachable = False
        outcome.readable = False
        if last_failure is None:  # pragma: no cover - candidates is never empty here
            outcome.verdict = VERDICT_INCONCLUSIVE
            return None
        if self._target_absent(last_failure):
            outcome.verdict = VERDICT_INCONCLUSIVE
            outcome.detail = (
                f"[{last_failure.code}] {last_failure.message} "
                "-- this set has no such target, so nothing was learned about the row"
            )
            return None
        outcome.verdict = VERDICT_BROKEN
        outcome.detail = f"read failed [{last_failure.code}]: {last_failure.message}"
        outcome.note = (
            f"Probed {self.stamp} with scripts/probe_paths.py on {outcome.path}: the read "
            f"failed with {last_failure.code} ({last_failure.message}). The parent object "
            f"resolves, so the property is not there; marked broken."
        )
        return None

    def _target_absent(self, failure: Result) -> bool:
        """Report whether this failure is about the set or about the row.

        ``index_out_of_range`` always means the set. ``no_such_path`` is genuinely
        ambiguous: an empty clip slot answers ``clip`` with ``None``, and reading
        a property off it fails exactly like a misspelt property would. So the
        parent path is re-read: if the parent is absent, ``None``, or itself
        unreadable, the target is missing and the row keeps its status
        (docs/catalog.md, *Three outcomes, not two*).
        """
        code = failure.code or ""
        if code in TARGET_ABSENT_CODES:
            return True
        if code != "no_such_path":
            # bad_path, type_error, internal: about the request or the script, not
            # about this set. live_error is included here on purpose -- Live
            # raising is evidence about the row, and it should be seen.
            return False
        message = failure.message or ""
        if "NoneType has no attribute" in message:
            return True
        if "is not available on this track" in message:
            # The script's group-track guard (protocol §6): the target is the
            # wrong kind of track, which says nothing about the row.
            return True
        parent = _parent_path(failure.path)
        if parent is None:
            return False
        try:
            reply = self.client.inner.get(parent)
        except AbletonCommandError:
            return True
        except AbletonError:
            raise
        return reply.get("value") is None

    # ----------------------------------------------------------------- writes
    def _probe_write(self, spec: PathSpec, outcome: ProbeOutcome) -> ProbeOutcome:
        """Write a safe value, read it back, then put the original back."""
        candidates = safe_values(spec, outcome.before)
        if not candidates:
            outcome.verdict = VERDICT_INCONCLUSIVE
            outcome.detail = (
                f"read {_short(outcome.before)} ({outcome.lom_type}); no safe write value "
                f"could be derived for kind {spec.kind.value}"
            )
            return outcome

        write: Result | None = None
        for value, why in candidates:
            _pause(self._pace_for(spec))
            write = self._write(spec, outcome.args, value, outcome.path)
            outcome.wrote = value
            if not write.ok:
                break
            outcome.after = write.after
            outcome.clamped = write.clamped
            if write.changed:
                outcome.detail = why
                break
            # Accepted and nothing moved. A quantized parameter can round the
            # nudge back to where it was, so try the second candidate before
            # concluding that the write did nothing.

        if write is None or not write.ok:
            code = write.code if write else None
            message = write.message if write else "no reply"
            outcome.writable = False
            outcome.error_code = code
            if code == "not_settable":
                outcome.verdict = VERDICT_BROKEN
                outcome.detail = f"write refused [not_settable]: {message}"
                outcome.note = (
                    f"Probed {self.stamp} with scripts/probe_paths.py on {outcome.path}: the "
                    f"read succeeded ({_note_value(outcome.before)}) but lom_set was "
                    f"refused with "
                    f"not_settable ({message}). The property exists and is read-only in the "
                    f"LOM; the catalog's 'set' access does not hold. Marked broken."
                )
            elif code in TARGET_ABSENT_CODES:
                outcome.verdict = VERDICT_INCONCLUSIVE
                outcome.detail = f"write target vanished [{code}]: {message}"
            else:
                outcome.verdict = VERDICT_BROKEN
                outcome.detail = f"write failed [{code}]: {message}"
                outcome.note = (
                    f"Probed {self.stamp} with scripts/probe_paths.py on {outcome.path}: the "
                    f"read succeeded ({_note_value(outcome.before)}) but the write failed "
                    f"with {code} "
                    f"({message}). Marked broken."
                )
            return outcome

        original = write.before if write.before is not None else outcome.before
        outcome.before = original
        outcome.restored = self._restore(spec, outcome, original)

        if not write.changed:
            # Accepted, and the stored value did not move. That is either a write
            # Live discarded in silence -- the failure this whole project is built
            # against -- or a quantized parameter rounding the probe's step back to
            # where it started. Live itself says which: ``is_quantized`` comes off
            # the parameter (protocol section 5.4), and where it is set the probe
            # refuses to call the row broken on evidence it does not have.
            outcome.writable = False
            quantized = bool(self.client.last.get("is_quantized")) or spec.quantized
            outcome.detail = (
                f"write accepted but nothing changed: still {_short(write.after)} "
                f"after writing {_short(outcome.wrote)}"
                + (" (quantized -- the step may be smaller than one)" if quantized else "")
            )
            if quantized:
                outcome.verdict = VERDICT_INCONCLUSIVE
                return outcome
            outcome.verdict = VERDICT_BROKEN
            outcome.note = (
                f"Probed {self.stamp} with scripts/probe_paths.py on {outcome.path}: read "
                f"{_note_value(original)}, wrote {_note_value(outcome.wrote)}, read back "
                f"{_note_value(write.after)} -- the "
                f"write was accepted and stored nothing, and Live does not report the "
                f"parameter as quantized. {len(candidates)} value(s) were tried. Marked broken "
                f"so the next person does not rediscover it."
            )
            return outcome

        outcome.writable = True
        outcome.verdict = VERDICT_VERIFIED
        outcome.display = _opt_str(self.client.last.get("display")) or outcome.display
        quantized = bool(self.client.last.get("is_quantized"))
        clamp_clause = ""
        if write.clamped:
            clamp_clause = (
                f" Live clamped the write: requested {_note_value(outcome.wrote)}, stored "
                f"{_note_value(write.after)}" + (" (is_quantized)." if quantized else ".")
            )
        outcome.detail = (
            f"read {_short(original)} -> wrote {_short(outcome.wrote)} -> "
            f"read back {_short(write.after)} -> restored" + ("" if outcome.restored else " FAILED")
        )
        outcome.note = (
            f"Measured {self.stamp} with scripts/probe_paths.py on {outcome.path}: read "
            f"{_note_value(original)} (LOM type {outcome.lom_type}), wrote "
            f"{_note_value(outcome.wrote)}, read back {_note_value(write.after)}, restored to "
            f"{_note_value(original)}{_display_clause(outcome.display)}.{clamp_clause}"
        )
        return outcome

    def _write(
        self, spec: PathSpec, args: Mapping[str, object], value: object, path: str | None
    ) -> Result:
        """One ``lom_set`` through the executor, with its guards intact."""
        try:
            return execute(
                self.client,
                self.registry,
                spec.id,
                verify=False,
                confirm=spec.destructive,
                value=value,
                **args,
            )
        except ValueError as exc:
            return Result(
                ok=False,
                id=spec.id,
                path=path or spec.path,
                code=CODE_BAD_REQUEST,
                message=str(exc),
            )

    def _restore(self, spec: PathSpec, outcome: ProbeOutcome, original: object) -> bool:
        """Put the original value back, and prove it went back.

        Raises :class:`RestoreFailed` when it did not. This is the one place the
        script will stop the whole run: a parameter left where a probe put it is
        the accident CONTRIBUTING.md warns about in bold, and continuing would
        pile more writes on top of a set the operator does not know is dirty.
        """
        if original is None:
            return True
        _pause(self._pace_for(spec))
        back = self._write(spec, outcome.args, original, outcome.path)
        if back.ok and _same_value(back.after, original):
            return True

        # Last resort: go around the executor's client-side validation. Restoring
        # matters more than layering -- the value came out of Live in the first
        # place, so it is by definition a legal value for this property.
        if outcome.path:
            try:
                reply = self.client.inner.set(outcome.path, original)
                if _same_value(reply.get("after"), original):
                    return True
            except AbletonError:
                pass

        outcome.restore_failed = True
        outcome.restored = False
        why = (
            f"the write was accepted and stored {back.after!r} instead"
            if back.ok
            else f"the write failed with [{back.code}] {back.message}"
        )
        raise RestoreFailed(
            f"{spec.id}: could NOT restore {outcome.path} to its original value "
            f"{original!r} -- {why}. Put it back by hand:\n"
            f"    python -m ableton_maestro.client set {outcome.path} "
            f"{json.dumps(original, default=repr)}",
            outcome,
        )

    # ------------------------------------------------------------- call rows
    def _probe_call(self, spec: PathSpec, outcome: ProbeOutcome) -> ProbeOutcome:
        """Inspect a ``call`` row without invoking it, unless told otherwise.

        A method call has no read-back and no restore, so it is not a probe in the
        sense the rest of this script means. What *can* be established without
        side effects is whether the object exists, whether it has the method at
        all, and whether the script's allowlist will let it through; the last one
        matters because the catalog's ``access: [call]`` says what the server
        offers and the script's allowlist says what it will do, and where the two
        disagree the script wins (docs/protocol.md §6).

        That is short of ``verified`` and this script says so: the row keeps its
        status unless the method is genuinely absent, which *is* evidence of a
        broken row.
        """
        candidates, why_not = candidate_args(spec, self.session, self.overrides)
        if not candidates:
            outcome.verdict = VERDICT_INCONCLUSIVE
            outcome.detail = why_not or "no target could be resolved for this row"
            return outcome
        outcome.args = dict(candidates[0])

        try:
            path = _build(spec, outcome.args)
        except ValueError as exc:
            outcome.verdict = VERDICT_INCONCLUSIVE
            outcome.detail = f"could not build the request: {exc}"
            return outcome
        outcome.path = path

        described = self._describe(path)
        if described is None:
            outcome.reachable = False
            outcome.verdict = VERDICT_INCONCLUSIVE
            outcome.detail = "lom_describe could not resolve the object in this set"
            return outcome

        outcome.reachable = True
        outcome.lom_type = str(described.get("class") or "")
        methods = {str(m) for m in described.get("methods") or []}
        allowed = {str(m) for m in described.get("allowed_methods") or []}
        method = str(spec.method)

        if method not in methods and described.get("truncated"):
            # The describe hit the script's node budget, so ``methods`` is a
            # partial list and "the method is not there" is a conclusion the
            # evidence does not support.
            outcome.verdict = VERDICT_INCONCLUSIVE
            outcome.detail = (
                f"lom_describe on {path} was truncated at the script's node budget, so "
                f"the absence of {method!r} proves nothing"
            )
            return outcome

        if method not in methods:
            outcome.verdict = VERDICT_BROKEN
            outcome.detail = f"{outcome.lom_type} has no method {method!r}"
            outcome.note = (
                f"Probed {self.stamp} with scripts/probe_paths.py: lom_describe on "
                f"{path} reports class {outcome.lom_type} with no method {method!r}. The "
                f"row's method does not exist on this object; marked broken. The method was "
                f"NOT called."
            )
            return outcome

        if method not in allowed:
            outcome.verdict = VERDICT_INCONCLUSIVE
            outcome.detail = (
                f"{outcome.lom_type}.{method} exists but is NOT on the script's allowlist -- "
                "the catalog offers a call the Remote Script will refuse (protocol section 6)"
            )
            return outcome

        if not self.include_calls:
            outcome.verdict = VERDICT_INCONCLUSIVE
            outcome.detail = (
                f"{outcome.lom_type}.{method} exists and is allowlisted; not called "
                "(a call cannot be read back or restored -- pass --include-calls)"
            )
            return outcome

        _pause(self._pace_for(spec))
        try:
            result = execute(
                self.client,
                self.registry,
                spec.id,
                verify=False,
                confirm=spec.destructive,
                **outcome.args,
            )
        except ValueError as exc:
            outcome.verdict = VERDICT_INCONCLUSIVE
            outcome.detail = f"could not build the call: {exc}"
            return outcome

        if not result.ok:
            outcome.error_code = result.code
            if result.code in TARGET_ABSENT_CODES:
                outcome.verdict = VERDICT_INCONCLUSIVE
                outcome.detail = f"call failed [{result.code}]: {result.message}"
                return outcome
            outcome.verdict = VERDICT_BROKEN
            outcome.detail = f"call failed [{result.code}]: {result.message}"
            outcome.note = (
                f"Probed {self.stamp} with scripts/probe_paths.py: {method}() on {path} "
                f"failed with {result.code} ({result.message}). Marked broken."
            )
            return outcome

        outcome.verdict = VERDICT_VERIFIED
        outcome.after = result.value
        outcome.detail = f"called {method}() -> {_short(result.value)}"
        outcome.note = (
            f"Measured {self.stamp} with scripts/probe_paths.py: {method}() was called on "
            f"{path} and returned {_note_value(result.value)}. A call has no read-back and "
            f"nothing was "
            f"restored -- what the call did to the set was NOT verified here."
        )
        return outcome

    def _describe(self, path: str) -> Mapping[str, Any] | None:
        """``lom_describe`` for ``path``, cached; ``None`` when it does not resolve."""
        if path not in self._describes:
            _pause(self.pace)
            try:
                self._describes[path] = self.client.inner.describe(path)
            except AbletonCommandError:
                self._describes[path] = None
        return self._describes[path]

    def _confirm_destructive(self, spec: PathSpec) -> bool:
        """Make a human type the row id before anything destructive happens."""
        if self.confirm is None or not self.confirm.isatty():
            print(
                f"  refusing {spec.id}: destructive rows need an interactive terminal "
                "to confirm in.",
                file=self.out,
            )
            return False
        print(file=self.out)
        print(f"  DESTRUCTIVE ROW: {spec.id}", file=self.out)
        print(f"    path:   {spec.path}", file=self.out)
        print(f"    method: {spec.method}", file=self.out)
        print(f"    doc:    {textwrap.shorten(spec.doc, 300)}", file=self.out)
        print(
            "    This cannot be read back and cannot be restored.\n"
            f"    Type the row id ({spec.id}) to run it, anything else to skip: ",
            end="",
            file=self.out,
            flush=True,
        )
        try:
            typed = self.confirm.readline().strip()
        except (EOFError, KeyboardInterrupt):  # pragma: no cover - interactive only
            typed = ""
        return typed == spec.id

    # ------------------------------------------------------------------ pacing
    def _pace_for(self, spec: PathSpec) -> float:
        """A longer pause where a path goes through a device."""
        return self.device_pace if ".devices[" in spec.path else self.pace


def _build(spec: PathSpec, args: Mapping[str, object]) -> str:
    """Return the concrete path for ``spec`` with ``args``.

    Uses the same builder as the executor.
    """
    return build_path(spec, **args)


def _opt_str(raw: object) -> str | None:
    return None if raw is None else str(raw)


def _display_clause(display: str | None) -> str:
    return f", display {display!r}" if display else ""


def _stamp(session: Session) -> str:
    """The provenance stamp every note carries: date, Live version, OS.

    CONTRIBUTING.md asks for the Live version and the OS with every flipped row,
    because a ``verified`` row is a claim about one specific build, and that
    context is the difference between a measurement and a rumour.
    """
    return (
        f"{datetime.now().astimezone().date().isoformat()} against Live "
        f"{session.live_version} "
        f"on {platform.system()} {platform.release()}"
    )


# --------------------------------------------------------------------------- #
# Write-back: editing the catalog YAML without destroying it
# --------------------------------------------------------------------------- #

_ROW_START = re.compile(r"^-\s+id:\s*(?P<id>.+?)\s*(?:#.*)?$")
_KEY_LINE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z_][A-Za-z0-9_]*):(?P<rest>.*)$")
_BLOCK_SCALAR = re.compile(r"^[>|][0-9+-]*\s*(#.*)?$")

#: Column the folded ``doc`` blocks in the catalog wrap at.
_DOC_WIDTH = 96


class CatalogWriteError(RuntimeError):
    """The catalog could not be edited safely. Nothing was changed."""


@dataclass
class RowLocation:
    file: Path
    start: int
    end: int


class CatalogEditor:
    """Line-level editor for the catalog files.

    ``ruamel.yaml`` would round-trip comments, and it is not a dependency of
    this project (``pyproject.toml`` names two, on purpose). A ``yaml.safe_load``
    followed by ``yaml.dump`` would silently delete every comment and reflow all
    22 300 lines of catalog into one unreviewable diff, so this edits the lines
    it needs and leaves every other byte, including line endings, exactly as it
    found them.

    Two things make that safe rather than merely clever:

    * each edited file is re-parsed with :class:`~ableton_maestro.registry.Registry`
      before it is kept, and the original text is put back if it no longer loads;
    * the file is replaced atomically, so an interrupted write cannot leave half
      a catalog behind.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files = [root] if root.is_file() else sorted(root.glob("*.yaml"))
        if not self.files:
            raise CatalogWriteError(f"no catalog files under {root}")

    @staticmethod
    def read(file: Path) -> str:
        """Read a catalog file with its line endings intact.

        ``newline=""`` rather than ``Path.read_text``: the latter grew a
        ``newline`` argument only in 3.12, this project supports 3.11, and
        rewriting a CRLF catalog as LF would turn a two-line edit into a
        whole-file diff.
        """
        with open(file, encoding="utf-8", newline="") as handle:
            return handle.read()

    @staticmethod
    def write(file: Path, text: str) -> None:
        """Write a catalog file without translating a single line ending."""
        with open(file, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)

    def apply(self, outcomes: Sequence[ProbeOutcome]) -> list[str]:
        """Write every eligible outcome back, and return one line per change."""
        changed: list[str] = []
        by_file: dict[Path, list[tuple[ProbeOutcome, RowLocation]]] = {}
        for outcome in outcomes:
            if not outcome.writes_back:
                continue
            location = self._locate(outcome.spec_id)
            if location is None:
                raise CatalogWriteError(
                    f"{outcome.spec_id}: no '- id: {outcome.spec_id}' line found under "
                    f"{self.root}; refusing to guess where the row lives"
                )
            by_file.setdefault(location.file, []).append((outcome, location))

        for file, entries in by_file.items():
            original = self.read(file)
            lines = _split_lines(original)
            # Bottom-up, so an insertion never invalidates a location above it.
            entries.sort(key=lambda pair: pair[1].start, reverse=True)
            for outcome, location in entries:
                newline = _newline_of(lines, location.start)
                _set_status(lines, location, outcome.verdict, newline)
                _append_doc_note(lines, location, outcome.note, newline)
                changed.append(f"{file.name}: {outcome.spec_id} -> {outcome.verdict}")
            self._write_verified(file, "".join(lines), original)
        return changed

    def _locate(self, spec_id: str) -> RowLocation | None:
        """Find the block of lines that make up one row."""
        for file in self.files:
            lines = _split_lines(self.read(file))
            for index, line in enumerate(lines):
                match = _ROW_START.match(line.rstrip("\r\n"))
                if match is None or match.group("id").strip("'\" ") != spec_id:
                    continue
                end = len(lines)
                for after in range(index + 1, len(lines)):
                    if lines[after].startswith("- "):
                        end = after
                        break
                return RowLocation(file=file, start=index, end=end)
        return None

    def _write_verified(self, file: Path, text: str, original: str) -> None:
        """Replace ``file`` atomically, and put it back if it stops parsing."""
        temporary = file.with_suffix(file.suffix + ".probe-tmp")
        self.write(temporary, text)
        temporary.replace(file)
        try:
            Registry.load(file)
        except CatalogError as exc:
            self.write(file, original)
            raise CatalogWriteError(
                f"{file.name} no longer parses after the edit, so the original was "
                f"restored and nothing was changed in it: {exc}"
            ) from exc


def _split_lines(text: str) -> list[str]:
    """Split on newlines only, keeping the ending, so rejoining is exact.

    ``str.splitlines`` also breaks on form feed, U+0085 and U+2028. None of them
    belongs in a catalog file, but a doc string could carry one, and a split that
    the join does not undo would silently reshape a row.

    Note:
        The summary deliberately spells out "newlines" rather than using an escape.
        This docstring is not a raw string, so an escape here would resolve into a real
        line break and truncate the summary line.
    """
    parts = text.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def _newline_of(lines: list[str], index: int) -> str:
    line = lines[index] if index < len(lines) else ""
    return "\r\n" if line.endswith("\r\n") else "\n"


def _set_status(lines: list[str], location: RowLocation, status: str, newline: str) -> None:
    """Rewrite the row's ``status:`` value, keeping indent and trailing comment."""
    for index in range(location.start, min(location.end, len(lines))):
        match = _KEY_LINE.match(lines[index].rstrip("\r\n"))
        if match is None or match.group("key") != "status":
            continue
        rest = match.group("rest")
        comment = ""
        if "#" in rest:
            comment = "  " + rest[rest.index("#") :].strip()
        lines[index] = f"{match.group('indent')}status: {status}{comment}{newline}"
        return
    raise CatalogWriteError(
        f"row at line {location.start + 1} of {location.file.name} has no 'status:' line; "
        "every row must carry one (docs/catalog.md)"
    )


def _append_doc_note(lines: list[str], location: RowLocation, note: str, newline: str) -> None:
    """Append ``note`` to the row's ``doc``, in the style the row already uses.

    A folded block (``doc: >-``, every row in the catalog) gains the
    note as further lines of the same block, which is exactly how the
    measurements already in the catalog are written. A single-line quoted scalar
    (none are left in the catalog, but the form is still accepted) is converted
    into a folded block first, because appending inside the quotes would produce a
    line of several hundred characters (the median row's ``doc`` is 417) and the
    folded form is the file's house style.
    """
    if not note:
        return
    for index in range(location.start, min(location.end, len(lines))):
        match = _KEY_LINE.match(lines[index].rstrip("\r\n"))
        if match is None or match.group("key") != "doc":
            continue
        indent = match.group("indent")
        rest = match.group("rest").strip()
        if _BLOCK_SCALAR.match(rest):
            _append_to_block(lines, location, index, indent, note, newline)
        else:
            _convert_and_append(lines, index, indent, rest, note, newline)
        return
    raise CatalogWriteError(
        f"row at line {location.start + 1} of {location.file.name} has no 'doc:' line; "
        "doc is mandatory (docs/catalog.md) and the note has nowhere to go"
    )


def _append_to_block(
    lines: list[str],
    location: RowLocation,
    doc_index: int,
    indent: str,
    note: str,
    newline: str,
) -> None:
    """Insert ``note`` at the end of an existing block scalar."""
    key_indent = len(indent)
    continuation: int | None = None
    last_content = doc_index
    for index in range(doc_index + 1, min(location.end, len(lines))):
        text = lines[index].rstrip("\r\n")
        if not text.strip():
            continue
        depth = len(text) - len(text.lstrip(" "))
        if depth <= key_indent:
            break
        if continuation is None:
            continuation = depth
        last_content = index
    depth = continuation if continuation is not None else key_indent + 2
    block = [f"{' ' * depth}{line}{newline}" for line in _wrap(note, depth)]
    lines[last_content + 1 : last_content + 1] = block


def _convert_and_append(
    lines: list[str],
    doc_index: int,
    indent: str,
    rest: str,
    note: str,
    newline: str,
) -> None:
    """Turn a single-line ``doc:`` scalar into a folded block, then append."""
    try:
        existing = yaml.safe_load(rest)
    except yaml.YAMLError as exc:  # pragma: no cover - the row would not have loaded
        raise CatalogWriteError(f"cannot read the doc scalar {rest!r}: {exc}") from exc
    if not isinstance(existing, str):
        raise CatalogWriteError(f"doc is not a string but {type(existing).__name__}")

    depth = len(indent) + 2
    body = _wrap(existing, depth) + _wrap(note, depth)
    lines[doc_index : doc_index + 1] = [f"{indent}doc: >-{newline}"] + [
        f"{' ' * depth}{line}{newline}" for line in body
    ]


def _wrap(text: str, indent: int) -> list[str]:
    """Wrap one paragraph to the catalog's column, without leading spaces.

    Leading spaces matter: inside a folded scalar a more-indented line stops
    folding and is kept literally, which would change the meaning of the text
    rather than only its shape.
    """
    return textwrap.wrap(
        " ".join(text.split()),
        width=max(_DOC_WIDTH - indent, 40),
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def select_rows(registry: Registry, args: argparse.Namespace) -> list[PathSpec]:
    """Work out which rows this invocation is about, or raise :class:`ValueError`."""
    if args.first_three:
        rows = []
        for spec_id in FIRST_THREE:
            try:
                rows.append(registry.get(spec_id))
            except KeyError:
                raise ValueError(
                    f"--first-three needs the row {spec_id!r}, which is not in this catalog"
                ) from None
        return rows

    if not (args.id or args.area or args.status or args.all):
        raise ValueError(
            "nothing selected. Choose rows with --id, --area, --status or --all "
            "(--all is deliberately explicit: it is 1164 rows and a long run)."
        )

    rows = registry.all()
    if args.id:
        known = [spec.id for spec in rows]
        chosen: list[PathSpec] = []
        for spec_id in args.id:
            try:
                chosen.append(registry.get(spec_id))
            except KeyError:
                near = difflib.get_close_matches(spec_id, known, n=5, cutoff=0.5)
                hint = f" Did you mean: {', '.join(near)}?" if near else ""
                raise ValueError(f"unknown catalog id {spec_id!r}.{hint}") from None
        rows = chosen

    if args.area:
        wanted = {area.lower() for area in args.area}
        known_areas = {area_of(spec.id) for spec in registry.all()}
        unknown = sorted(wanted - known_areas)
        if unknown:
            raise ValueError(
                f"unknown area(s) {unknown}; the catalog has: {', '.join(sorted(known_areas))}"
            )
        rows = [spec for spec in rows if area_of(spec.id) in wanted]

    if args.status:
        wanted_status = {PathStatus(value) for value in args.status}
        rows = [spec for spec in rows if spec.status in wanted_status]

    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def print_dry_run(rows: Sequence[PathSpec], overrides: Mapping[str, object], out: Any) -> None:
    """List what a real run would touch, without opening a socket."""
    print(f"DRY RUN -- {len(rows)} row(s) selected. Nothing is sent to Live.", file=out)
    print(file=out)
    header = f"{'id':<34} {'status':<9} {'access':<20} {'kind':<6} plan"
    print(header, file=out)
    print("-" * len(header), file=out)
    for spec in rows:
        access = ",".join(a.value for a in spec.access)
        if spec.destructive:
            plan = "DESTRUCTIVE -- needs --include-destructive and a typed confirmation"
        elif Access.CALL in spec.access:
            plan = f"describe only ({spec.method}(); --include-calls to invoke it)"
        elif spec.supports(Access.SET):
            plan = "read, write a safe value, read back, restore"
        else:
            plan = "read only"
        if spec.id in ACTION_ROWS:
            plan += " [transport/record: needs --include-transport]"
        print(
            f"{spec.id:<34} {spec.status.value:<9} {access:<20} {spec.kind.value:<6} {plan}",
            file=out,
        )
        print(f"{'':<34} {spec.path}", file=out)
    print(file=out)
    if overrides:
        given = ", ".join(f"{k}={v!r}" for k, v in sorted(overrides.items()))
        print(f"Fixed indices from the command line: {given}", file=out)
    print(
        "Indices not given on the command line are auto-picked from the open set at\n"
        "run time (first non-group track, first clip, first track carrying a device).\n"
        "\n"
        "Add --go to actually probe. Nothing above has touched Live.",
        file=out,
    )


def print_report(outcomes: Sequence[ProbeOutcome], out: Any) -> None:
    """The per-row table: reachable / readable / writable / clamped / restored."""
    header = (
        f"{'id':<32} {'rch':<4}{'rd':<4}{'wr':<4}{'clmp':<5}{'rest':<5} "
        f"{'type':<7} {'verdict':<13} detail"
    )
    print(header, file=out)
    print("-" * min(len(header) + 40, 160), file=out)
    for outcome in outcomes:
        print(
            f"{outcome.spec_id:<32} "
            f"{_flag(outcome.reachable):<4}"
            f"{_flag(outcome.readable):<4}"
            f"{_flag(outcome.writable):<4}"
            f"{_plain_flag(outcome.clamped):<5}"
            f"{_flag(outcome.restored):<5} "
            f"{(outcome.lom_type or '-'):<7} "
            f"{outcome.verdict:<13} {outcome.detail}",
            file=out,
        )
        if outcome.path:
            extra = f"  {outcome.path}"
            if outcome.display:
                extra += f"   display={outcome.display!r}"
            print(extra, file=out)


def _flag(value: bool | None) -> str:
    """A column where False is bad news and should read like it."""
    if value is None:
        return "-"
    return "yes" if value else "NO"


def _plain_flag(value: bool | None) -> str:
    """A column where False is simply an answer (``clamped``, for instance)."""
    if value is None:
        return "-"
    return "yes" if value else "no"


def print_summary(outcomes: Sequence[ProbeOutcome], out: Any) -> dict[str, int]:
    counts: dict[str, int] = {
        VERDICT_VERIFIED: 0,
        VERDICT_BROKEN: 0,
        VERDICT_INCONCLUSIVE: 0,
        VERDICT_SKIPPED: 0,
    }
    for outcome in outcomes:
        counts[outcome.verdict] = counts.get(outcome.verdict, 0) + 1
    print(file=out)
    print(
        "  ".join(f"{name}: {count}" for name, count in counts.items()),
        file=out,
    )
    print(
        "\n'inconclusive' means the target was missing from THIS set, not that the row\n"
        "is wrong -- those rows keep their status and nothing is written for them.",
        file=out,
    )
    return counts


def print_proposed_edits(outcomes: Sequence[ProbeOutcome], out: Any) -> None:
    """Show the catalog changes a ``--write-back`` run would make."""
    pending = [o for o in outcomes if o.writes_back]
    if not pending:
        print("\nNo catalog row would change.", file=out)
        return
    print(
        f"\nProposed catalog edits ({len(pending)} row(s)) -- re-run with --write-back to apply:",
        file=out,
    )
    for outcome in pending:
        print(f"\n  {outcome.spec_id}: status -> {outcome.verdict}", file=out)
        for line in _wrap(outcome.note, 6):
            print(f"      {line}", file=out)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _placeholder_names(registry: Registry) -> list[str]:
    """Every ``{placeholder}`` name the catalog uses, so each gets its own flag."""
    names: set[str] = set()
    for spec in registry.all():
        names.update(param.name for param in spec.params)
    return sorted(names)


def build_parser(registry: Registry) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/probe_paths.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Probe catalog rows against a running Ableton Live: read them, write a "
            "safe value where the row allows it, read it back, and restore the "
            "original. Then flip the row's status from 'untested' to 'verified' or "
            "'broken'."
        ),
        epilog=textwrap.dedent(
            """\
            examples:
              # the three clip warp/pitch rows (see FIRST_THREE), in one command
              python scripts/probe_paths.py --first-three --go

              # one row against a target you choose
              python scripts/probe_paths.py --id clip.warping --track 0 --slot 0 --go

              # everything still untested in one area, writing the results back
              python scripts/probe_paths.py --area track --status untested --go --write-back

            Without --go this only lists what it WOULD do. That is on purpose: this
            script writes into a live set, and a dry run costs nothing.

            Use a throwaway set. Restore is automatic, a crash is not.
            """
        ),
    )

    selection = parser.add_argument_group("selecting rows")
    selection.add_argument(
        "--id", action="append", metavar="ID", help="a catalog row id; repeatable"
    )
    selection.add_argument(
        "--area",
        "--class",
        action="append",
        metavar="AREA",
        help="a catalog area (the id prefix: track, clip, song, device, ...); repeatable",
    )
    selection.add_argument(
        "--status",
        action="append",
        choices=[s.value for s in PathStatus],
        help="only rows with this status; repeatable",
    )
    selection.add_argument(
        "--all", action="store_true", help="every row in the catalog (a long run)"
    )
    selection.add_argument(
        "--first-three",
        action="store_true",
        help=(
            "exactly clip.warping, clip.warp_mode and clip.pitch_coarse -- the "
            "first measurement of the project, recorded in docs/limits.md section 5"
        ),
    )
    selection.add_argument("--limit", type=int, metavar="N", help="stop after N selected rows")

    targets = parser.add_argument_group(
        "targets",
        "Fix a path placeholder. Anything left out is auto-picked from the open set.",
    )
    for name in _placeholder_names(registry):
        targets.add_argument(
            f"--{name.replace('_', '-')}",
            dest=f"ph_{name}",
            metavar="N" if name != "root" else "NAME",
            help=f"value for the {{{name}}} placeholder",
        )

    safety = parser.add_argument_group("safety")
    safety.add_argument(
        "--go",
        action="store_true",
        help="actually connect to Live and probe. Without it this is a dry run.",
    )
    safety.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "say what would be probed and change nothing -- the default. It exists "
            "as a flag only because scripts/install_script.py has one and defaults "
            "the other way; passing both it and --go is refused rather than guessed."
        ),
    )
    safety.add_argument(
        "--read-only",
        action="store_true",
        help="never write anything; report only what could be read",
    )
    safety.add_argument(
        "--i-know-what-im-doing",
        dest="i_know",
        action="store_true",
        help="run even though the open set looks like real work",
    )
    safety.add_argument(
        "--include-destructive",
        action="store_true",
        help=(
            "allow rows marked destructive: true. Each one still has to be "
            "confirmed by typing its id at an interactive prompt."
        ),
    )
    safety.add_argument(
        "--include-calls",
        action="store_true",
        help=(
            "really invoke 'call' rows. Without it they are only inspected with "
            "lom_describe, because a call cannot be read back or restored."
        ),
    )
    safety.add_argument(
        "--include-transport",
        action="store_true",
        help=(
            "allow writes that start playback, recording or arming "
            f"({len(ACTION_ROWS)} rows). They are restored, but a record pass is not."
        ),
    )
    safety.add_argument(
        "--pace",
        type=float,
        default=DEFAULT_PACE,
        metavar="SECONDS",
        help=f"pause between operations (default {DEFAULT_PACE})",
    )
    safety.add_argument(
        "--device-pace",
        type=float,
        default=DEFAULT_DEVICE_PACE,
        metavar="SECONDS",
        help=(
            "pause around paths that go through a device "
            f"(default {DEFAULT_DEVICE_PACE}; loading plugins in rapid succession "
            "can crash Live -- measured)"
        ),
    )

    output = parser.add_argument_group("output and write-back")
    output.add_argument(
        "--write-back",
        action="store_true",
        help="update status and append a dated note to doc, in place, in the catalog",
    )
    output.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG_DIR,
        metavar="PATH",
        help="catalog directory or single file (default: the packaged catalog)",
    )
    output.add_argument("--json", action="store_true", help="machine-readable report on stdout")

    connection = parser.add_argument_group("connection")
    connection.add_argument("--host", default=DEFAULT_HOST, help=f"default {DEFAULT_HOST}")
    connection.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"default {DEFAULT_PORT}"
    )
    connection.add_argument(
        "--max-slots",
        type=int,
        default=16,
        metavar="N",
        help="how far the session survey scans tracks and scenes (default 16)",
    )
    return parser


def collect_overrides(args: argparse.Namespace, registry: Registry) -> dict[str, object]:
    """Turn the ``--track``/``--slot``/… flags into placeholder values."""
    overrides: dict[str, object] = {}
    for name in _placeholder_names(registry):
        raw = getattr(args, f"ph_{name}", None)
        if raw is None:
            continue
        if name == "root":
            overrides[name] = str(raw)
            continue
        try:
            overrides[name] = int(raw)
        except ValueError:
            flag = "--" + name.replace("_", "-")
            raise ValueError(f"{flag} takes a whole number, got {raw!r}") from None
    return overrides


def run_probe(
    args: argparse.Namespace,
    registry: Registry,
    rows: Sequence[PathSpec],
    overrides: Mapping[str, object],
    out: Any,
) -> tuple[list[ProbeOutcome], Session, int]:
    """Connect, survey, guard, probe. Returns the outcomes and an exit code."""
    outcomes: list[ProbeOutcome] = []
    client = AbletonClient(args.host, args.port)
    recording = RecordingClient(client)
    try:
        client.connect()
        session = survey_session(recording, registry, max_slots=args.max_slots)
        if not guard_session(session, override=args.i_know, out=out):
            return outcomes, session, CODE_REFUSED

        prober = Prober(
            recording,
            registry,
            session,
            overrides=overrides,
            pace=args.pace,
            device_pace=args.device_pace,
            read_only=args.read_only,
            include_calls=args.include_calls,
            include_destructive=args.include_destructive,
            include_transport=args.include_transport,
            confirm=sys.stdin,
            out=out,
        )
        exit_code = 0
        for index, spec in enumerate(rows, start=1):
            print(f"[{index}/{len(rows)}] {spec.id}", file=out, flush=True)
            try:
                outcomes.append(prober.probe(spec))
            except RestoreFailed as exc:
                partial = exc.outcome or ProbeOutcome(spec_id=spec.id, template=spec.path)
                # Nothing is written back for this row. The read and the write did
                # happen, but the set is dirty and the only thing that matters now
                # is that a human sees it.
                partial.verdict = VERDICT_SKIPPED
                partial.detail = str(exc)
                partial.restore_failed = True
                partial.note = ""
                outcomes.append(partial)
                print(file=out)
                print("=" * 78, file=out)
                print("  RESTORE FAILED -- THE SET IS DIRTY. Stopping the run.", file=out)
                print(f"  {exc}", file=out)
                print("=" * 78, file=out)
                return outcomes, session, CODE_RESTORE_FAILED
            _pause(args.pace)
        return outcomes, session, exit_code
    finally:
        client.close()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Exit codes: 0 ran (broken rows included; a broken row is a *result*),
    1 a usage or connection failure, 2 argparse's own usage error, 3 a restore
    failed and the set was left dirty, 4 the safety gate refused to run.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover - not a real stream
            pass

    out = sys.stderr
    try:
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_DIR)
        known, _ = pre.parse_known_args(argv)
        registry = Registry.load(known.catalog)
    except CatalogError as exc:
        print(f"catalog error: {exc}", file=sys.stderr)
        return 1

    parser = build_parser(registry)
    args = parser.parse_args(argv)

    print(_BANNER, file=out)
    try:
        rows = select_rows(registry, args)
        overrides = collect_overrides(args, registry)
    except ValueError as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return 1
    if not rows:
        print("no rows matched the selection.", file=sys.stderr)
        return 1
    if args.dry_run and args.go:
        print(
            "usage error: --dry-run and --go contradict each other. This script does "
            "nothing without --go, so --dry-run alone is enough.",
            file=sys.stderr,
        )
        return 1
    if args.write_back and not args.go:
        print(
            "usage error: --write-back needs --go. A dry run measures nothing, and a "
            "status written from nothing is worse than no status at all.",
            file=sys.stderr,
        )
        return 1
    if args.include_destructive and not args.go:
        print("note: --include-destructive does nothing in a dry run.", file=out)

    if not args.go:
        print_dry_run(rows, overrides, out)
        return 0

    try:
        outcomes, session, code = run_probe(args, registry, rows, overrides, out)
    except AbletonTimeoutError as exc:
        print(f"timeout: {exc}", file=sys.stderr)
        if exc.may_have_landed:
            print(
                "That was a write. A timed-out write may still have landed and it was "
                "NOT restored -- check the path by hand before probing again "
                "(docs/protocol.md section 8).",
                file=sys.stderr,
            )
        return CODE_RESTORE_FAILED
    except AbletonConnectionError as exc:
        print(f"connection: {exc}", file=sys.stderr)
        return 1
    except AbletonError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print(
            "\naborted. Anything written before the abort was restored as it went, but "
            "the row that was in flight may not have been -- check it.",
            file=sys.stderr,
        )
        return 130

    if outcomes:
        print(file=out)
        print_report(outcomes, out)
        print_summary(outcomes, out)

    if args.write_back:
        try:
            changed = CatalogEditor(args.catalog).apply(outcomes)
        except CatalogWriteError as exc:
            print(f"write-back failed: {exc}", file=sys.stderr)
            return 1
        print(f"\nwrote back {len(changed)} row(s):", file=out)
        for line in changed:
            print(f"  {line}", file=out)
        print(
            "\nSay in the pull request which Live version and OS this was measured on; "
            "the notes above carry it too.",
            file=out,
        )
    elif outcomes:
        print_proposed_edits(outcomes, out)

    if args.json:
        print(
            json.dumps(
                {
                    "live_version": session.live_version,
                    "script_version": session.script_version,
                    "session": {
                        "tracks": len(session.tracks),
                        "clips": len(session.clips),
                        "scenes": len(session.scenes),
                        "returns": len(session.returns),
                    },
                    "rows": [outcome.as_dict() for outcome in outcomes],
                },
                indent=2,
                ensure_ascii=False,
                default=repr,
            )
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
