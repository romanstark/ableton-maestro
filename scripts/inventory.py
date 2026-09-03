"""System and live session inventory diagnostic for Ableton Maestro.

Provides dual surveys:
1. Live session survey (requires running Live + active script): tracks, routing,
   devices, clips, return tracks, master chain, and scene topology.
2. Installation survey (runs offline): installed Live versions, User Library paths,
   Remote Script presence, log diagnostics, and listening port status.

Usage:
    python scripts/inventory.py                    # complete session and install inventory
    python scripts/inventory.py --json             # machine-readable JSON output
    python scripts/inventory.py --installation-only # inspect local setup without Live running

Key diagnostic checks:
- Connection troubleshooting: validates Live process state, Control Surface dropdown
  configuration, restart currency, and library directory selection.
- Parameter configuration advice: identifies VSTs exposing only ``Device On`` due
  to unconfigured parameter strips.
- Vector collection fallback: supports legacy Remote Script collection indexing via
  :class:`CountingClient`.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import textwrap
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Imported after the sys.path line above, so that `python scripts/inventory.py`
# works from any working directory. The discovery, the User Library scoring and
# the .pyc comparison live in the installer and are not duplicated here.
import install_script as installer

__all__ = ["build_report", "main", "probe_port", "read_control_surfaces"]

DEFAULT_HOST = "127.0.0.1"
OUR_PORT = installer.OUR_PORT
FOREIGN_PORT = installer.FOREIGN_PORT

#: Seconds for a bare TCP connect. Long enough that a busy Live still answers,
#: short enough that a diagnostic stays a diagnostic.
CONNECT_TIMEOUT = 3.0

#: Seconds for the two handshakes. ``script_info`` answers on the client thread
#: rather than Live's main thread (see the Remote Script's README), so it is
#: expected to be quick even while Live is busy.
HANDSHAKE_TIMEOUT = 8.0

#: How much of Live's ``Log.txt`` is read, from the end. The reference machine's
#: was 205 KB after one session; it grows without bound across a long-running
#: install, and only the most recent startup block is wanted.
LOG_TAIL_BYTES = 2 * 1024 * 1024

_SLOT_RE = re.compile(r'MidiRemoteScript\s+(\d+)\s+\[Control Surface="([^"]*)"')
_BLOCK_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T[\d:.]+):.*Midi Remote Scripts:")
_INIT_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T[\d:.]+):.*\(([A-Za-z0-9_.\- ]+)\) Initializing\.\.\.")

#: Parameter names per line under ``--parameters``. A configured VST instance
#: can reach 128 of them (measured ceiling, ``lom/introspect.VST_PARAMETER_SLOTS``).
_NAMES_PER_LINE = 6

#: How wide the device column in the track table gets before it is elided.
_DEVICE_COLUMN = 34

#: Wrap width for prose. Tables size themselves to their content instead.
_WIDTH = 96

#: The collections :class:`CountingClient` walks to supply a size. Everything
#: else in a ``lom_describe`` reply is passed through untouched, so a describe
#: can never turn into an unbounded walk of the whole object graph.
COUNTED_COLLECTIONS: frozenset[str] = frozenset(
    {"tracks", "return_tracks", "scenes", "devices", "parameters", "clip_slots", "cue_points"}
)

#: Indices probed per collection per round. One round of 64 covered every
#: collection on the 53-track reference set (measured 2026-08-29).
COLLECTION_BLOCK = 64

#: Default ceilings for the index walks. Each is a stop, not a claim: hitting one
#: is reported as "at least N", never as "exactly N".
MAX_COLLECTION = 512
MAX_DEVICES = 16
MAX_PARAMETERS = 8


# --------------------------------------------------------------------------- #
# Loading the package, which may not be installed yet
# --------------------------------------------------------------------------- #


def _load_maestro() -> tuple[ModuleType, ModuleType] | None:
    """Import the client and the introspection layer, or return ``None``.

    A cold start is exactly the case where ``pip install -e .`` has not been run
    yet, so a bare ``ImportError`` traceback would be the least useful possible
    answer. The checkout's ``src`` is tried as a fallback; failing that the
    installation survey still runs and the session survey reports why it did
    not.
    """
    for extra in (None, installer.REPO_ROOT / "src"):
        if extra is not None:
            path = str(extra)
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            from ableton_maestro import client as client_module
            from ableton_maestro.lom import introspect as introspect_module
        except ImportError:
            continue
        return client_module, introspect_module
    return None


# --------------------------------------------------------------------------- #
# Installation survey
# --------------------------------------------------------------------------- #


def probe_port(host: str, port: int, timeout: float = CONNECT_TIMEOUT) -> dict[str, Any]:
    """Open a TCP connection and close it again to see whether the port answers.

    The three outcomes are different diagnoses and are kept apart:

    * ``listening``: something accepted the connection. It does not say *what*;
      the handshake does that.
    * ``refused``: nothing is bound there. The usual, clean "not running".
    * ``no_answer``: the connect neither completed nor was refused inside the
      window. *Measured 2026-08-29* on the reference machine: a port with
      nothing bound to it timed out instead of refusing, so a firewall or filter
      can look exactly like a hang. Worth reporting as its own case instead of
      folding it into "closed".
    """
    started = time.monotonic()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
    except ConnectionRefusedError:
        state, detail = "refused", "nothing is listening on that port"
    except TimeoutError:
        state, detail = "no_answer", f"connect neither completed nor was refused in {timeout:.0f}s"
    except OSError as exc:
        state, detail = "error", f"{exc.__class__.__name__}: {exc}"
    else:
        state, detail = "listening", "a socket accepted the connection"
    finally:
        sock.close()
    return {
        "host": host,
        "port": port,
        "state": state,
        "detail": detail,
        "elapsed_ms": round((time.monotonic() - started) * 1000.0, 1),
    }


def _handshake(
    client_module: ModuleType, host: str, port: int, handler: str, timeout: float
) -> dict[str, Any]:
    """Ask a port who it is. Returns ``{"ok": bool, ...}``; never raises.

    ``handler`` differs per port on purpose: this script answers ``script_info``,
    while other Ableton Remote Scripts answer ``get_script_info``. The two names
    are deliberately kept apart (``docs/protocol.md`` §5.2), so asking the wrong
    handler of the wrong server is itself an identification.
    """
    client = client_module.AbletonClient(host, port, read_timeout=timeout, write_timeout=timeout)
    try:
        result = client.send(handler, timeout=timeout)
    except client_module.AbletonCommandError as exc:
        return {"ok": False, "handler": handler, "error": f"{exc.code}: {exc.message}"}
    except client_module.AbletonError as exc:
        return {"ok": False, "handler": handler, "error": str(exc)}
    finally:
        client.close()
    return {"ok": True, "handler": handler, "info": result}


def read_control_surfaces(log_file: Path) -> dict[str, Any]:
    """What Live's own ``Log.txt`` last recorded about Remote Scripts.

    Two readings, both from Live rather than from us:

    * the Control Surface slots, as Live printed them at its last startup,
      this is how "is AbletonMaestro actually selected?" gets an answer without
      a screenshot of the preferences dialog;
    * when each Control Surface last initialised, which is how "was Live
      restarted after the script was copied?" gets one.

    This is the last state Live logged, not a live reading: the log is
    written at startup and when the surfaces change, so between the two the file
    is silent. Timestamps are Live's local time and are compared against local
    file mtimes.

    Never raises. A missing or unreadable log is one field in the report.
    """
    out: dict[str, Any] = {
        "path": str(log_file),
        "available": False,
        "logged_at": None,
        "slots": {},
        "last_loaded": {},
        "error": None,
    }
    try:
        size = log_file.stat().st_size
        with log_file.open("rb") as handle:
            if size > LOG_TAIL_BYTES:
                handle.seek(size - LOG_TAIL_BYTES)
                handle.readline()  # drop the partial first line
            text = handle.read().decode("utf-8", errors="replace")
    except OSError as exc:
        out["error"] = f"{exc.__class__.__name__}: {exc}"
        return out

    out["available"] = True
    slots: dict[str, str] = {}
    stamp: str | None = None
    for line in text.splitlines():
        block = _BLOCK_RE.match(line)
        if block is not None:
            slots, stamp = {}, block.group(1)
            continue
        slot = _SLOT_RE.search(line)
        if slot is not None:
            slots[slot.group(1)] = slot.group(2)
            continue
        init = _INIT_RE.match(line)
        if init is not None:
            out["last_loaded"][init.group(2)] = init.group(1)

    out["slots"] = slots
    out["logged_at"] = stamp
    return out


def _iso_to_epoch(stamp: str | None) -> float | None:
    """Live's ISO timestamp as a local epoch, or ``None`` when it will not parse."""
    if not stamp:
        return None
    try:
        return time.mktime(time.strptime(stamp.split(".")[0], "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, OverflowError):
        return None


def _script_entry(script: installer.InstalledScript, source: bytes | None) -> dict[str, Any]:
    """One installed Remote Script as a report entry."""
    matches_repo: bool | None = None
    if source is not None and script.init_file is not None:
        try:
            matches_repo = script.init_file.read_bytes() == source
        except OSError:
            matches_repo = None
    return {
        "name": script.name,
        "folder": str(script.folder),
        "loadable": script.loadable,
        "is_foreign": script.is_foreign,
        "size": script.size,
        "modified": script.mtime,
        "modified_text": _when(script.mtime),
        "pycache": str(script.pycache) if script.pycache else None,
        "pycache_matches_source": script.pyc_current,
        "pycache_note": script.stale_note(),
        "matches_this_checkout": matches_repo,
        "backups": len(script.backups),
    }


def survey_installation(
    args: argparse.Namespace, client_module: ModuleType | None
) -> dict[str, Any]:
    """Everything knowable without Live running, plus the two port probes."""
    report: dict[str, Any] = {
        "platform": sys.platform,
        "preferences_roots": [str(root) for root in installer.preferences_roots()],
    }

    installs = installer.find_live_installs()
    newest = installs[0] if installs else None
    report["live_versions"] = {
        "count": len(installs),
        "newest": newest.version_text if newest else None,
        "newest_folder": str(newest.folder) if newest else None,
        "all": [install.version_text for install in installs],
    }

    if args.user_library is not None:
        chosen = Path(args.user_library).expanduser()
        score, evidence = installer.library_score(chosen)
        candidates = [
            installer.LibraryCandidate(
                path=chosen, origin="--user-library", score=score, evidence=evidence
            )
        ]
    else:
        candidates = installer.discover_user_libraries(installs)
        present = [candidate for candidate in candidates if candidate.exists]
        chosen = present[0].path if present else None

    report["user_library"] = {
        "path": str(chosen) if chosen else None,
        "evidence": next(
            (c.evidence for c in candidates if chosen and c.path == chosen),
            None,
        ),
        "candidates": [
            {
                "path": str(candidate.path),
                "origin": candidate.origin,
                "exists": candidate.exists,
                "evidence": candidate.evidence,
                "chosen": chosen is not None and candidate.path == chosen,
            }
            for candidate in candidates
        ],
    }

    try:
        source = installer.DEFAULT_SOURCE.read_bytes()
    except OSError:
        source = None
    report["repo_script"] = {
        "path": str(installer.DEFAULT_SOURCE),
        "readable": source is not None,
        "size": len(source) if source is not None else None,
    }

    scripts = installer.scan_remote_scripts(chosen) if chosen else []
    report["remote_scripts"] = [_script_entry(script, source) for script in scripts]
    report["foreign_present"] = any(script.is_foreign for script in scripts)
    report["ours"] = next(
        (
            entry
            for entry in report["remote_scripts"]
            if entry["name"].lower() == installer.SCRIPT_FOLDER_NAME.lower()
        ),
        None,
    )

    # Remote Scripts sitting somewhere Live does not read: the wrong-folder trap,
    # which produces no error anywhere and is therefore worth naming out loud.
    decoys = []
    for candidate in candidates:
        if not candidate.exists or (chosen is not None and candidate.path == chosen):
            continue
        found = installer.scan_remote_scripts(candidate.path)
        if found:
            decoys.append(
                {
                    "path": str(candidate.path / "Remote Scripts"),
                    "evidence": candidate.evidence,
                    "scripts": [script.name for script in found],
                }
            )
    report["invisible_to_live"] = decoys

    report["log"] = (
        read_control_surfaces(newest.preferences / "Log.txt")
        if newest
        else {"available": False, "error": "no Live version folder found"}
    )

    ports: dict[str, Any] = {}
    for port, handler, enabled in (
        (args.port, "script_info", True),
        (args.foreign_port, "get_script_info", args.probe_foreign),
    ):
        probe = probe_port(args.host, port, args.connect_timeout)
        if probe["state"] == "listening" and enabled and client_module is not None:
            probe["handshake"] = _handshake(
                client_module, args.host, port, handler, args.timeout or HANDSHAKE_TIMEOUT
            )
        ports[str(port)] = probe
    report["ports"] = ports

    # Which of the probed ports is the one the session survey uses. Not a
    # constant: --port exists, and a finding that names 9878 while the survey
    # tried 9999 is worse than no finding at all.
    report["our_port"] = args.port
    running = ports.get(str(args.port), {}).get("handshake", {})
    report["live_version_running"] = (
        running.get("info", {}).get("live_version") if running.get("ok") else None
    )
    return report


# --------------------------------------------------------------------------- #
# Session survey
# --------------------------------------------------------------------------- #


def _is_error(entry: Mapping[str, Any] | None) -> bool:
    """True for a batch result that is an error object rather than a value (§5.7)."""
    return entry is None or "code" in entry or entry.get("status") == "error"


def _error_text(entry: Mapping[str, Any]) -> str:
    """``code: message`` for an error result."""
    message = entry.get("message")
    code = entry.get("code", "error")
    return f"{code}: {message}" if message else str(code)


def _batched_get(
    client: Any, introspect: ModuleType, wanted: Sequence[str]
) -> list[dict[str, Any]]:
    """``lom_get`` over many paths in as few round trips as the batch limit allows.

    Results stay aligned with ``wanted``, errors included: in a survey, "this one
    path would not answer" is information, not a reason to lose the rest.
    """
    out: list[dict[str, Any]] = []
    limit = introspect.BATCH_OP_LIMIT
    for start in range(0, len(wanted), limit):
        chunk = wanted[start : start + limit]
        reply = client.batch([{"op": "get", "path": path} for path in chunk])
        results = reply.get("results")
        if not isinstance(results, list) or len(results) != len(chunk):
            raise RuntimeError(
                f"lom_batch answered {len(results) if isinstance(results, list) else '?'} "
                f"results for {len(chunk)} ops. Protocol §5.7 requires them in order and "
                "complete; the connection may be desynchronised - restart the survey."
            )
        out.extend(
            dict(item) if isinstance(item, Mapping) else {"code": "internal"} for item in results
        )
    return out


def probe_collection(
    client: Any,
    introspect: ModuleType,
    paths: Sequence[str],
    *,
    block: int,
    cap: int,
    warnings: list[str],
) -> dict[str, list[Any]]:
    """Count and read several LOM collections by walking their indices.

    Why this exists, and it is not a preference
    -------------------------------------------
    Live's collections are not Python lists. ``song.tracks``,
    ``track.devices`` and ``device.parameters`` are Live's own ``Vector``
    objects, and an ``isinstance`` against ``(list, tuple)`` catches none of
    them. A Remote Script that asks that question instead of
    ``is_lom_collection`` answers a ``lom_get`` on a collection with a bare
    *handle* (``{"__lom__": "Vector", ...}``) and lists it in ``lom_describe``
    as a child of type ``Vector`` with no count.

    *Measured 2026-08-29, Live 12.4.5, on a 53-track set:* against such a script,
    ``lom_get("song.tracks")`` gave ``{"__lom__": "Vector"}``;
    ``lom_describe("song")`` gave ``{"name": "tracks", "type": "Vector"}`` with
    no ``count``; and ``song.tracks[0].name`` answered ``'01 SC Trigger'``.
    Anything that counts a collection from those two replies reports zero
    tracks for a set with 53: a confident, wrong, silent answer, which is the
    one thing this project exists to refuse.

    The script in this repository no longer does that: a collection encodes as a
    list of element handles and a describe child carries a real ``count``
    (``docs/protocol.md`` §5.3, §5.6). This walk remains for an installation
    still running an older script, and for the device and parameter surveys,
    which want the element handles anyway. It indexes upwards until
    ``index_out_of_range``.

    It is cheap because it is batched and because the handles carry names:
    walking ``<track>.devices`` names every device in the same round trip that
    counts them. *Measured on that set:* 59 device collections in one round
    (1.8 s), 85 parameter collections in one round (1.4 s).

    Args:
        paths: collection paths, without the index.
        block: indices probed per collection per round.
        cap: stop after this many elements. The returned list being exactly
            ``cap`` long means "at least ``cap``", not "exactly ``cap``".
        warnings: appended to for any error that is *not* a bounds error: a
            collection that stops early for another reason is under-counted, and
            that must be visible rather than silent.

    Returns:
        ``{path: [element handles, in index order]}``.
    """
    values: dict[str, list[Any]] = {path: [] for path in paths}
    pending = list(paths)

    while pending:
        wanted: list[str] = []
        plan: list[tuple[str, int]] = []
        for path in pending:
            start = len(values[path])
            span = min(block, cap - start)
            if span <= 0:
                continue
            plan.append((path, span))
            wanted.extend(f"{path}[{index}]" for index in range(start, start + span))
        if not wanted:
            break

        replies = _batched_get(client, introspect, wanted)
        position = 0
        next_round: list[str] = []
        for path, span in plan:
            chunk = replies[position : position + span]
            position += span
            stopped = False
            for entry in chunk:
                if not _is_error(entry):
                    values[path].append(entry.get("value"))
                    continue
                stopped = True
                if entry.get("code") != "index_out_of_range":
                    # Successes were appended as we went, so the length is the
                    # index that just failed. A non-bounds failure means the walk
                    # stopped early and the count is too low - say so.
                    warnings.append(
                        f"{path}[{len(values[path])}]: {_error_text(entry)} - the index walk "
                        "stopped there, so this collection is under-counted"
                    )
                break
            if not stopped and len(values[path]) < cap:
                next_round.append(path)
        pending = next_round

    return values


class CountingClient:
    """A client proxy that fills in any collection size ``lom_describe`` omits.

    ``introspect.snapshot`` asks ``lom_describe("song")`` how many tracks,
    return tracks and scenes there are. An older Remote Script answered
    ``Vector`` with no count (see :func:`probe_collection`), and left alone the
    survey then reports an empty set. This proxy walks those collections and
    rewrites the ``children`` entries into the ``{"type": "list", "count": n}``
    shape ``introspect`` expects, so the survey works unmodified, with its
    group-track guards, its arm guard and its clip sweep all intact.

    The fix itself has been made, in the Remote Script's ``encode_value`` and
    ``lom_describe`` handler, where a Live ``Vector`` is now unrolled like a
    list. This proxy stays because deploying that fix costs a Live restart
    (``docs/protocol.md`` §9), and a diagnostic is exactly the tool that has to
    keep working against the installation as it is. Only a child whose reported
    type still ends in ``Vector`` is a candidate, so against a current script
    there is nothing here to walk.

    Only the collections named in :data:`COUNTED_COLLECTIONS` are walked. Every
    other child is passed through untouched, so this cannot turn a cheap describe
    into an unbounded one.
    """

    def __init__(
        self, inner: Any, introspect: ModuleType, *, block: int, cap: int, warnings: list[str]
    ) -> None:
        self._inner = inner
        self._introspect = introspect
        self._block = block
        self._cap = cap
        self._warnings = warnings
        self.walked = 0

    def get(self, path: str) -> Mapping[str, Any]:
        """Pass through."""
        return self._inner.get(path)

    def batch(self, ops: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Pass through."""
        return self._inner.batch(ops)

    def describe(self, path: str, **kwargs: Any) -> Mapping[str, Any]:
        """Describe, then supply the counts the script could not."""
        info = dict(self._inner.describe(path, **kwargs))
        children = list(info.get("children") or [])
        targets = [
            child
            for child in children
            if isinstance(child, Mapping)
            and child.get("name") in COUNTED_COLLECTIONS
            and str(child.get("type", "")).endswith("Vector")
            and child.get("path")
        ]
        if not targets:
            return info

        counted = probe_collection(
            self._inner,
            self._introspect,
            [str(child["path"]) for child in targets],
            block=self._block,
            cap=self._cap,
            warnings=self._warnings,
        )
        self.walked += len(targets)
        chosen = {id(child) for child in targets}
        rewritten: list[Any] = []
        for child in children:
            if id(child) not in chosen:
                rewritten.append(child)
                continue
            size = len(counted[str(child["path"])])
            rewritten.append({**child, "type": "list", "count": size})
            if size >= self._cap:
                self._warnings.append(
                    f"{child['path']}: the index walk stopped at the cap of {self._cap}; "
                    "there may be more"
                )
        info["children"] = rewritten
        return info


def survey_devices(
    client: Any,
    introspect: ModuleType,
    tracks: Sequence[Any],
    *,
    max_devices: int,
    max_parameters: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Every device on every track, with the ``Device On`` diagnosis.

    Two index walks, and each one doubles as its own read: the handles that come
    back from ``<track>.devices[i]`` carry the device's class and name, and the
    ones from ``<device>.parameters[j]`` carry the parameter names
    (``docs/protocol.md`` §7). So counting and naming cost the same round trip.

    The parameter walk stops at ``max_parameters``. That is enough to classify
    (the threshold between "sparse" and "playable" is
    ``introspect.CONFIGURED_THRESHOLD``), while a full walk of a configured VST
    would be up to 128 indices per instance (the measured ceiling). A device that
    hit the cap reports ``parameter_count: null`` and
    ``parameter_count_at_least``, because "8" and "at least 8" are different
    claims.

    A device whose parameter walk failed for a non-bounds reason is reported as
    ``unreadable`` and is not diagnosed. Calling a failed read "no
    parameters" would manufacture exactly the false conclusion this survey exists
    to prevent.
    """
    warnings: list[str] = []
    if not tracks:
        return [], warnings

    device_lists = probe_collection(
        client,
        introspect,
        [f"{track.path}.devices" for track in tracks],
        block=max_devices,
        cap=max_devices,
        warnings=warnings,
    )

    found: list[tuple[Any, int, dict[str, Any]]] = []
    for track in tracks:
        handles = device_lists.get(f"{track.path}.devices", [])
        if len(handles) >= max_devices:
            warnings.append(
                f"{track.path}.devices: walk capped at {max_devices}; raise --max-devices"
            )
        for index, handle in enumerate(handles):
            found.append((track, index, handle if isinstance(handle, dict) else {}))

    if not found:
        return [], warnings

    parameter_lists = probe_collection(
        client,
        introspect,
        [f"{track.path}.devices[{index}].parameters" for track, index, _ in found],
        block=max_parameters,
        cap=max_parameters,
        warnings=warnings,
    )

    out: list[dict[str, Any]] = []
    for track, index, handle in found:
        path = f"{track.path}.devices[{index}]"
        raw = parameter_lists.get(f"{path}.parameters", [])
        capped = len(raw) >= max_parameters
        views = [
            introspect.ParameterView(
                index=position,
                path=str((item or {}).get("path") or f"{path}.parameters[{position}]"),
                name=str((item or {}).get("name") or f"(parameter {position})"),
            )
            for position, item in enumerate(raw)
        ]
        view = introspect.DeviceView(
            path=path,
            name=str(handle.get("name") or "(unnamed)"),
            class_name=str(handle.get("__lom__") or "Device"),
            index=index,
            track=track.path,
            parameters=tuple(views),
            diagnosis=introspect.diagnose(views),
        )
        readable = bool(views)
        out.append(
            {
                "track": track.name,
                "track_path": track.path,
                "track_kind": track.kind,
                "index": index,
                "path": path,
                "name": view.name,
                "class": view.class_name,
                "readable": readable,
                "parameter_count": None if capped or not readable else view.parameter_count,
                "parameter_count_at_least": max_parameters if capped else None,
                "playable_count": None if capped or not readable else view.playable_count,
                "diagnosis": view.diagnosis.value if readable else "unreadable",
                "advice": view.advice if readable else None,
                "parameters": [param.name for param in views],
            }
        )
        if not readable:
            warnings.append(
                f"{path}.parameters: nothing readable, not even Live's own 'Device On'. "
                "That is not the unconfigured-plugin case; the path or the read failed."
            )
    return out, warnings


def survey_session(
    args: argparse.Namespace, modules: tuple[ModuleType, ModuleType]
) -> dict[str, Any]:
    """Survey tracks, returns, master, scenes and tempo, with every device on them.

    Runs ``introspect.snapshot`` through :class:`CountingClient`, which supplies
    a collection's size for any installation whose Remote Script still reports one
    as a ``Vector`` with no count; without it such a survey reports an empty set
    (see :func:`probe_collection`). Devices are surveyed separately, so
    ``snapshot`` is asked for ``devices=False``: this walk also brings back the
    class name and the parameter diagnosis.
    """
    client_module, introspect = modules
    report: dict[str, Any] = {"connected": False, "warnings": [], "error": None}
    warnings: list[str] = []

    client = client_module.AbletonClient(
        args.host,
        args.port,
        read_timeout=args.timeout or client_module.READ_TIMEOUT,
        write_timeout=args.timeout or client_module.WRITE_TIMEOUT,
    )
    counting = CountingClient(
        client,
        introspect,
        block=COLLECTION_BLOCK,
        cap=args.max_collection,
        warnings=warnings,
    )
    started = time.monotonic()
    try:
        snap = introspect.snapshot(
            counting,
            clips=not args.no_clips,
            devices=False,  # done below, with classes and the parameter diagnosis
            max_scenes=args.max_scenes,
        )
        tracks = [*snap.tracks, *snap.return_tracks]
        if snap.master is not None:
            tracks.append(snap.master)
        devices, device_warnings = (
            ([], [])
            if args.no_devices
            else survey_devices(
                client,
                introspect,
                tracks,
                max_devices=args.max_devices,
                max_parameters=args.max_parameters,
            )
        )
    except client_module.AbletonError as exc:
        report["error"] = str(exc)
        return report
    except (RuntimeError, introspect.IntrospectionError) as exc:
        report["error"] = f"{exc.__class__.__name__}: {exc}"
        return report
    finally:
        client.close()

    by_track: dict[str, list[dict[str, Any]]] = {}
    for device in devices:
        by_track.setdefault(device["track_path"], []).append(device)

    report.update(
        {
            "connected": True,
            "elapsed_ms": round((time.monotonic() - started) * 1000.0, 1),
            "song": dict(snap.song),
            "scene_count": snap.scene_count,
            "scene_names": list(snap.scene_names),
            "track_count": snap.track_count,
            "session_clip_count": snap.session_clip_count,
            "tracks": [track.to_dict() for track in snap.tracks],
            "return_tracks": [track.to_dict() for track in snap.return_tracks],
            "master": snap.master.to_dict() if snap.master is not None else None,
            "devices": devices,
            "devices_by_track": {path: len(items) for path, items in by_track.items()},
            "warnings": [*warnings, *snap.warnings, *device_warnings],
            "collections_walked": counting.walked,
            "taken_at": snap.taken_at,
        }
    )
    return report


# --------------------------------------------------------------------------- #
# Findings: what is wrong, and what to do about it
# --------------------------------------------------------------------------- #


def _finding(severity: str, what: str, fix: str) -> dict[str, str]:
    """One entry for findings list (``problem`` sets exit code, ``note`` does not)."""
    return {"severity": severity, "what": what, "fix": fix}


def _connection_findings(install: dict[str, Any], session: dict[str, Any]) -> list[dict[str, str]]:
    """The cold-start walk-through: four causes, each already checked on disk.

    Written as a numbered list rather than a guess, because every one of the
    four produces the same symptom (silence on the port) and three of them
    produce no error message anywhere in Live.
    """
    ours = install.get("ours")
    log = install.get("log", {})
    slots = log.get("slots") or {}
    selected = any(name.lower() == installer.SCRIPT_FOLDER_NAME.lower() for name in slots.values())
    library = install.get("user_library", {}).get("path")
    our_port = install.get("our_port", OUR_PORT)
    port = install.get("ports", {}).get(str(our_port), {})

    where = f"{port.get('host', DEFAULT_HOST)}:{port.get('port', our_port)}"
    socket_state = (
        "a socket answered, so Live is up and something is bound to the port"
        if port.get("state") == "listening"
        else f"no socket answered ({port.get('detail', 'nothing probed that port')})"
    )
    lines = [
        f"The client reported: {session.get('error') or 'no reply'}",
        "",
        "The four causes, in the order they actually occur:",
        "",
        "  1. Live is not running, or is still loading a set.",
        f"     -> {socket_state}",
        "",
        f"  2. {installer.SCRIPT_FOLDER_NAME} is not selected as a Control Surface.",
    ]
    if not log.get("available"):
        lines.append("     -> Live's Log.txt could not be read, so this one is unchecked.")
    elif selected:
        slot = next(
            key
            for key, name in slots.items()
            if name.lower() == installer.SCRIPT_FOLDER_NAME.lower()
        )
        logged = _iso_to_epoch(log.get("logged_at"))
        lines.append(
            f"     -> Live's log ({_when(logged) if logged else log.get('logged_at')}) has it "
            f"in slot {slot} - looks selected."
        )
    else:
        listed = ", ".join(f"{k}={v}" for k, v in sorted(slots.items())) or "(none logged)"
        lines.append(
            f"     -> NOT in Live's logged slots [{listed}]. Set it in Preferences -> "
            "Link, Tempo & MIDI -> Control Surface."
        )

    lines += ["", "  3. Live was not restarted completely after the script was copied."]
    loaded_at = _iso_to_epoch((log.get("last_loaded") or {}).get(installer.SCRIPT_FOLDER_NAME))
    written_at = ours.get("modified") if ours else None
    if loaded_at is None or written_at is None:
        lines.append("     -> not checkable here (no load entry in the log, or nothing installed).")
    elif loaded_at < written_at:
        lines.append(
            f"     -> Live last loaded it at {_when(loaded_at)}, but the file was written at "
            f"{_when(written_at)}. Live is running the OLDER version. Quit Live and start it again."
        )
    else:
        lines.append(f"     -> last loaded {_when(loaded_at)}, after the file was written. Fine.")

    lines += ["", "  4. The script is in a folder Live does not read."]
    if ours is None:
        lines.append(
            f"     -> nothing is installed at {library}\\Remote Scripts\\"
            f"{installer.SCRIPT_FOLDER_NAME}. Run: python scripts/install_script.py"
        )
    else:
        lines.append(f"     -> installed at {ours['folder']} (the User Library Live names).")
    if install.get("invisible_to_live"):
        for decoy in install["invisible_to_live"]:
            lines.append(
                f"     -> note: {', '.join(decoy['scripts'])} also sits in {decoy['path']}, "
                "which Live does not read. That folder produces no error and no dropdown entry."
            )

    return [
        _finding(
            "problem",
            f"The session was not surveyed: nothing usable answered on {where}.",
            "\n".join(lines),
        )
    ]


def collect_findings(
    install: dict[str, Any], session: dict[str, Any] | None
) -> list[dict[str, str]]:
    """Everything worth acting on, with the action attached.

    ``install`` is empty under ``--session-only`` and ``session`` is ``None``
    under ``--installation-only``. Neither may produce a finding: "not looked at"
    and "not there" are different answers, and reporting the first as the second
    is the exact mistake this file is built to avoid.
    """
    findings: list[dict[str, str]] = []
    if not install:
        return _session_findings(install, session, findings)

    if install.get("user_library", {}).get("path") is None:
        findings.append(
            _finding(
                "problem",
                "No User Library found.",
                "Live records it in Preferences\\Library.cfg under <ProjectPath>. Start Live "
                "once so it writes its preferences, or pass --user-library. "
                "See: python scripts/install_script.py --list",
            )
        )

    ours = install.get("ours")
    if install.get("user_library", {}).get("path") and ours is None:
        findings.append(
            _finding(
                "problem",
                f"{installer.SCRIPT_FOLDER_NAME} is not installed in the User Library.",
                "python scripts/install_script.py",
            )
        )
    elif ours is not None:
        if ours["pycache_matches_source"] is False:
            findings.append(
                _finding(
                    "problem",
                    "The installed __pycache__ was compiled from a different __init__.py.",
                    "Live prefers the compiled version and says nothing about it, so the change "
                    "appears not to have happened. Delete it and restart Live completely:\n"
                    f'  Remove-Item -Recurse -Force "{ours["pycache"]}"',
                )
            )
        if ours["matches_this_checkout"] is False:
            findings.append(
                _finding(
                    "note",
                    "The installed script differs from live-remote-script/__init__.py in this "
                    "checkout.",
                    "Not necessarily wrong - it may be an older or a deliberately edited copy. "
                    "See what changed: python scripts/install_script.py --diff",
                )
            )
        if not ours["loadable"]:
            findings.append(
                _finding(
                    "problem",
                    f"{ours['folder']} has no __init__.py, so Live does not offer it at all.",
                    "python scripts/install_script.py",
                )
            )

    for decoy in install.get("invisible_to_live", []):
        findings.append(
            _finding(
                "note",
                f"Remote Scripts in a folder Live does not read: {decoy['path']} "
                f"[{', '.join(decoy['scripts'])}].",
                "Live loads nothing from there and reports nothing about it. Harmless unless one "
                "of those is the copy you have been editing.",
            )
        )

    if install.get("foreign_present"):
        findings.append(
            _finding(
                "note",
                "Another Ableton Remote Script is installed alongside this one.",
                f"Its own folder, its own port ({FOREIGN_PORT} against {OUR_PORT}); nothing here "
                "touches it. Whether two Control Surfaces can each run their own socket server "
                "side by side is UNVERIFIED - nobody has measured it.",
            )
        )

    return _session_findings(install, session, findings)


def _session_findings(
    install: dict[str, Any],
    session: dict[str, Any] | None,
    findings: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Append session survey findings (split out for ``--session-only`` access)."""
    if session is not None and not session.get("connected"):
        findings.extend(_connection_findings(install, session))

    if session is not None and session.get("connected"):
        devices = session.get("devices", [])
        findings.extend(
            _configuration_finding(
                [device for device in devices if device["diagnosis"] == "unconfigured"],
                "report only 'Device On'",
            )
        )
        findings.extend(
            _configuration_finding(
                [
                    device
                    for device in devices
                    if device["diagnosis"] == "sparse" and "Plugin" in device["class"]
                ],
                "are plugins with barely any parameters exposed",
            )
        )
        if session.get("collections_walked"):
            findings.append(
                _finding(
                    "note",
                    "The track, return and scene counts above were obtained by walking indices, "
                    "not read from Live.",
                    "The Remote Script loaded in this Live reports its collections as type "
                    "'Vector' with no count, and answers lom_get on one with a handle rather "
                    "than a list - so nothing in the replies says how many tracks a set has "
                    "(measured 2026-08-29, Live 12.4.5). This survey therefore indexes upwards "
                    "until index_out_of_range. The counts are real; the round trips are the "
                    "price. The script in this repository already unrolls a Live Vector like a "
                    "list (docs/protocol.md 5.3 and 5.6) - installing it and restarting Live "
                    "makes these counts a single read.",
                )
            )
        for warning in session.get("warnings", []):
            findings.append(_finding("note", f"survey warning: {warning}", "read-only; no action"))

    return findings


def _configuration_finding(devices: Sequence[Mapping[str, Any]], what: str) -> list[dict[str, str]]:
    """One finding covering every device that shares a configuration diagnosis.

    The explanation is printed once, for the first device, and the rest are named
    as the same case. The text itself comes from
    ``introspect.configuration_advice``: one wording, one place to correct it.
    """
    if not devices:
        return []
    named = ", ".join(f"{d['name']} on {d['track']}" for d in devices[:4])
    if len(devices) > 4:
        named += f", and {len(devices) - 4} more"
    others = ", ".join(f"{d['name']} on {d['track']}" for d in devices[1:])
    return [
        _finding(
            "note",
            f"{len(devices)} device(s) {what}: {named}.",
            (devices[0]["advice"] or "") + (f"\nSame case, same fix: {others}." if others else ""),
        )
    ]


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    """Run the surveys the flags ask for and return the whole report as a dict.

    This is what ``--json`` prints verbatim, and what the table below renders.
    One structure, so the two can never disagree.
    """
    modules = _load_maestro()
    report: dict[str, Any] = {
        "generated_at": time.time(),
        "generated_at_text": _when(time.time()),
        "package_importable": modules is not None,
    }

    if not args.session_only:
        report["installation"] = survey_installation(args, modules[0] if modules else None)
    if not args.installation_only:
        if modules is None:
            report["session"] = {
                "connected": False,
                "error": "the ableton_maestro package could not be imported",
                "warnings": [],
            }
        else:
            report["session"] = survey_session(args, modules)

    report["findings"] = collect_findings(report.get("installation", {}), report.get("session"))
    if not report["package_importable"]:
        report["findings"].insert(
            0,
            _finding(
                "problem",
                "The ableton_maestro package is not importable by this interpreter "
                f"({sys.executable}).",
                'From the checkout:  pip install -e ".[dev]"\n'
                "Or run this with the checkout's own interpreter: "
                ".venv\\Scripts\\python.exe scripts/inventory.py",
            ),
        )
    return report


# --------------------------------------------------------------------------- #
# Human rendering
# --------------------------------------------------------------------------- #


def _when(epoch: float | None) -> str:
    """A local timestamp a human reads, or ``-``."""
    if epoch is None:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))


def _fit(text: str, width: int) -> str:
    """Pad or elide ``text`` to exactly ``width`` characters."""
    if len(text) <= width:
        return text.ljust(width)
    return text[: max(width - 3, 1)] + "..."


def _wrap(text: str, *, first: str, rest: str) -> list[str]:
    """Wrap one paragraph to :data:`_WIDTH`, keeping an indented command line intact.

    A line that is already a command to type is left alone: wrapping it would
    produce something that looks copy-pasteable and is not.
    """
    stripped = text.strip()
    if not stripped:
        return [""]
    # A line that indents itself is structure (a numbered step, a nested note),
    # and losing that indent turns a walk-through into a paragraph.
    own = text[: len(text) - len(text.lstrip())]
    head = first + own
    # A hanging indent only where the line already carries structure, so a
    # wrapped list item stays visibly inside its item and a plain paragraph
    # stays aligned with its own first line.
    tail = rest + own + ("   " if own else "")
    if len(head) + len(stripped) <= _WIDTH or stripped.startswith(("python ", "pip ", "Remove-")):
        return [head + stripped]
    return textwrap.wrap(
        stripped,
        width=_WIDTH,
        initial_indent=head,
        subsequent_indent=tail,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _table(rows: Sequence[Sequence[str]], headers: Sequence[str], indent: str = "  ") -> list[str]:
    """A fixed-width table. Columns size themselves to the content."""
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    out = [indent + "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))]
    out.append(indent + "  ".join("-" * width for width in widths))
    for row in rows:
        out.append(indent + "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return [line.rstrip() for line in out]


def _say(lines: Iterable[str] | str) -> None:
    """Print one line or many."""
    if isinstance(lines, str):
        print(lines)
        return
    for line in lines:
        print(line)


def _script_notes(entry: Mapping[str, Any]) -> list[str]:
    """What is worth saying about one installed Remote Script, one clause per line."""
    notes: list[str] = []
    if entry["is_foreign"]:
        notes.append(
            f"another Remote Script, not ours - port {FOREIGN_PORT}, and never touched by "
            "install_script.py"
        )
    if not entry["loadable"]:
        notes.append("no __init__.py, so Live does not offer this folder in the dropdown at all")
    if entry["matches_this_checkout"] is True:
        notes.append("byte-identical to live-remote-script/__init__.py in this checkout")
    elif entry["matches_this_checkout"] is False and not entry["is_foreign"]:
        notes.append(
            "differs from live-remote-script/__init__.py in this checkout "
            "(python scripts/install_script.py --diff)"
        )
    if entry["pycache_matches_source"] is False:
        notes.append(
            "STALE __pycache__ - compiled from a different __init__.py, and Live prefers it"
        )
    elif entry["pycache_matches_source"] is True:
        notes.append("__pycache__ matches the source beside it")
    elif entry["pycache"]:
        notes.append("__pycache__ present but hash-based or unreadable, so it cannot be classified")
    if entry["backups"]:
        notes.append(f"{entry['backups']} timestamped backup(s) kept in the folder")
    return notes


def _render_installation(install: dict[str, Any]) -> None:
    """The installation half of the table."""
    _say("")
    _say("INSTALLATION")
    versions = install["live_versions"]
    _say(f"  platform            {install['platform']}")
    _say(
        f"  Live, newest prefs  {versions['newest'] or '(none found)'}"
        f"   ({versions['count']} version folder(s) under "
        f"{', '.join(install['preferences_roots']) or 'no known root'})"
    )
    running = install.get("live_version_running")
    _say(f"  Live, running       {running or '(not reachable)'}")
    library = install["user_library"]
    _say(f"  User Library        {library['path'] or '(not found)'}")
    if library.get("evidence"):
        _say(f"                      {library['evidence']}")
    _say(
        f"  Repo script         {install['repo_script']['path']}"
        + (
            f"  ({install['repo_script']['size']:,} bytes)"
            if install["repo_script"]["size"]
            else "  (unreadable)"
        )
    )

    _say("")
    if install["remote_scripts"]:
        _say(f"  Remote Scripts in {library['path']}\\Remote Scripts")
        rows = [
            [
                entry["name"],
                f"{entry['size']:,}" if entry["size"] else "-",
                entry["modified_text"],
            ]
            for entry in install["remote_scripts"]
        ]
        _say(_table(rows, ["folder", "bytes", "modified"], indent="    "))
        for entry in install["remote_scripts"]:
            for note in _script_notes(entry):
                _say(f"      {entry['name']}: {note}")
    else:
        _say("  Remote Scripts      (none installed in that User Library)")

    if install["invisible_to_live"]:
        _say("")
        _say("  Remote Scripts in folders Live does NOT read:")
        for decoy in install["invisible_to_live"]:
            _say(f"    {decoy['path']}  [{', '.join(decoy['scripts'])}]")
            _say(f"      {decoy['evidence']}")
        _say("    No dropdown entry, no error, no log line. That is the whole trap.")

    log = install["log"]
    _say("")
    if log.get("available"):
        logged = _iso_to_epoch(log.get("logged_at"))
        stamp = _when(logged) if logged else (log.get("logged_at") or "?")
        _say(f"  Control Surface slots, from Live's Log.txt at {stamp}")
        slots = log.get("slots") or {}
        if slots:
            for number in sorted(slots, key=int):
                _say(f"    slot {number}   {slots[number]}")
        else:
            _say("    (no slot block found in the tail of the log)")
        _say("    This is the last state Live logged, not a live reading.")
        loaded = log.get("last_loaded") or {}
        if loaded:
            _say("")
            _say("  Control Surfaces Live last initialised (from the same log)")
            for name in sorted(loaded):
                when = _iso_to_epoch(loaded[name])
                _say(f"    {_fit(name, 20)} {_when(when) if when else loaded[name]}")
    else:
        _say(f"  Live's Log.txt      not read: {log.get('error') or 'not found'}")

    _say("")
    _say("  Ports")
    # Ours first: on a cold start it is the one being asked about.
    for key in sorted(install["ports"], key=lambda k: (k != str(OUR_PORT), int(k))):
        probe = install["ports"][key]
        who = ""
        handshake = probe.get("handshake")
        if handshake and handshake.get("ok"):
            info = handshake["info"]
            who = (
                f"  {info.get('name', '?')} {info.get('script_version', '?')}, "
                f"protocol {info.get('protocol_version', '?')}, "
                f"{len(info.get('handlers') or [])} handlers, "
                f"allowlist {info.get('allowlist_size', '?')}, "
                f"Live {info.get('live_version', '?')}"
            )
        elif handshake:
            who = f"  handshake '{handshake['handler']}' failed: {handshake['error']}"
        elif probe["state"] == "listening" and key == str(FOREIGN_PORT):
            who = "  something answers; --probe-foreign names what it is"
        _say(f"    {key}  {_fit(probe['state'], 11)} {probe['detail']}{who}")


def _render_session(session: dict[str, Any], args: argparse.Namespace) -> None:
    """The session half of the table."""
    _say("")
    _say("SESSION")
    if not session.get("connected"):
        # The client's full message is long and repeats what FINDINGS says
        # properly, so only its first sentence goes here.
        error = (session.get("error") or "no connection").split(". ")[0]
        _say(f"  not surveyed: {error}.")
        _say("  FINDINGS below walks the causes in the order they actually occur.")
        return

    song = session["song"]
    tempo = song.get("tempo")
    tempo_text = f"{tempo:.2f}" if isinstance(tempo, (int, float)) else "?"
    numerator = song.get("signature_numerator", "?")
    denominator = song.get("signature_denominator", "?")
    _say(
        f"  tempo {tempo_text}"
        f"   {numerator}/{denominator}"
        f"   playing {'yes' if song.get('is_playing') else 'no'}"
        f"   metronome {'on' if song.get('metronome') else 'off'}"
        f"   scenes {session['scene_count']}"
        f"   surveyed in {session['elapsed_ms']:.0f} ms"
    )
    _say(
        f"  {session['track_count']} track(s), "
        f"{len(session['return_tracks'])} return(s), "
        f"{len(session['devices'])} device(s), "
        f"{session['session_clip_count']} session clip(s)"
    )

    by_track: dict[str, list[dict[str, Any]]] = {}
    for device in session["devices"]:
        by_track.setdefault(device["track_path"], []).append(device)

    rows = []
    for track in [*session["tracks"], *session["return_tracks"], session["master"] or {}]:
        if not track:
            continue
        devices = by_track.get(track["path"], [])
        names = ", ".join(device["name"] for device in devices) or "-"
        clips = track.get("clips") or []
        rows.append(
            [
                str(track["index"]),
                _fit(track["name"] or "(unnamed)", 22).rstrip(),
                track["kind"],
                _yes_no(track.get("mute")),
                _yes_no(track.get("solo")),
                _yes_no(track.get("armed")),
                _fit(names, _DEVICE_COLUMN).rstrip(),
                str(len(clips)) if clips or track["kind"] in {"midi", "audio", "group"} else "-",
            ]
        )
    _say("")
    _say(_table(rows, ["#", "track", "kind", "mute", "solo", "arm", "devices", "clips"]))
    _say("  arm '-' means the question does not apply: groups, returns and the master have no")
    _say("  arm state, and asking raises inside Live (measured).")

    if session["devices"]:
        _say("")
        _say("  DEVICES")
        rows = []
        for device in session["devices"]:
            count = device["parameter_count"]
            floor = device["parameter_count_at_least"]
            if count is not None:
                params = str(count)
            elif floor is not None:
                params = f">={floor}"
            else:
                params = "?"
            rows.append(
                [
                    _fit(device["track"] or "(unnamed)", 18).rstrip(),
                    str(device["index"]),
                    _fit(device["name"], 24).rstrip(),
                    device["class"],
                    params,
                    device["diagnosis"],
                ]
            )
        _say(_table(rows, ["track", "#", "device", "class", "params", "state"]))
        _say("  'params' counts Live's own 'Device On' too. '>=N' means the index walk stopped")
        _say("  at --max-parameters, so the real number is that or larger - never assume it is N.")

        if args.parameters:
            _say("")
            _say("  PARAMETERS, as this instance currently exposes them")
            for device in session["devices"]:
                names = device["parameters"]
                _say(f"    {device['path']}  -  {device['name']}")
                if not names:
                    _say("      (none readable)")
                    continue
                for start in range(0, len(names), _NAMES_PER_LINE):
                    _say("      " + ", ".join(names[start : start + _NAMES_PER_LINE]))

    if session["scene_names"]:
        _say("")
        named = [f"{i}:{n}" for i, n in enumerate(session["scene_names"]) if n]
        _say(f"  scenes  {', '.join(named) if named else '(all unnamed)'}")


def _yes_no(value: bool | None) -> str:
    """``yes`` / ``no`` / ``-``. The third is not the second: see TrackView.armed."""
    if value is None:
        return "-"
    return "yes" if value else "no"


def _render_findings(findings: Sequence[Mapping[str, str]]) -> None:
    """What to do, with the reason attached. The part of the output that matters."""
    _say("")
    _say("FINDINGS")
    if not findings:
        _say("  Nothing to report. Everything the survey can check, checks out.")
        return
    for number, finding in enumerate(findings, start=1):
        tag = "PROBLEM" if finding["severity"] == "problem" else "note   "
        head = f"  [{number}] {tag}  "
        _say(_wrap(finding["what"], first=head, rest=" " * len(head)))
        for line in finding["fix"].splitlines():
            _say(_wrap(line, first=" " * 10, rest=" " * 10) if line.strip() else "")
    _say("")


def render(report: Mapping[str, Any], args: argparse.Namespace) -> None:
    """The whole human-readable report."""
    _say("")
    _say(f"ABLETON MAESTRO - INVENTORY            {report['generated_at_text']}")
    _say("A read-only survey. Nothing here writes anything to the set or to disk.")
    if "installation" in report:
        _render_installation(report["installation"])
    if "session" in report:
        _render_session(report["session"], args)
    _render_findings(report["findings"])
    if "session" in report and report["session"].get("connected"):
        _say("  A snapshot proves the state Live held at that moment. It says nothing about")
        _say("  the .als on disk, and nothing about how any of it sounds.")
        _say("")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/inventory.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Survey this Ableton Live set and this Ableton Maestro installation.\n"
            "\n"
            "Read-only: it reads the session over the socket and reads files on disk,\n"
            "and writes nothing anywhere. Doubles as the cold-start diagnostic - when\n"
            "nothing answers, FINDINGS walks the four causes in the order they occur."
        ),
        epilog=(
            "Exit codes: 0 nothing to fix, 1 at least one PROBLEM finding,\n"
            "2 usage error, 130 interrupted."
        ),
    )
    parser.add_argument("--json", action="store_true", help="the whole report as JSON on stdout.")

    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--session-only",
        action="store_true",
        help="skip the installation survey (files, log, ports).",
    )
    scope.add_argument(
        "--installation-only",
        action="store_true",
        help="skip the session survey. Live does not have to be running.",
    )

    parser.add_argument("--host", default=DEFAULT_HOST, help=f"default {DEFAULT_HOST}.")
    parser.add_argument(
        "--port", type=int, default=OUR_PORT, help=f"the Ableton Maestro port (default {OUR_PORT})."
    )
    parser.add_argument(
        "--foreign-port",
        type=int,
        default=FOREIGN_PORT,
        help=f"a second port, probed as a diagnostic only (default {FOREIGN_PORT}, where "
        "other Ableton Remote Scripts listen). Never used for anything else.",
    )
    parser.add_argument(
        "--probe-foreign",
        action="store_true",
        help="also send get_script_info to that second port, to name what is answering "
        "there. Off by default: a connect proves a listener, not which script.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=CONNECT_TIMEOUT,
        metavar="SECONDS",
        help=f"per-port TCP connect timeout (default {CONNECT_TIMEOUT:g}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="override the per-call timeout for the session survey (default: protocol §8).",
    )
    parser.add_argument(
        "--no-devices",
        action="store_true",
        help="skip the device survey. Saves two round trips and loses the "
        "'only Device On' diagnosis.",
    )
    parser.add_argument(
        "--no-clips",
        action="store_true",
        help="skip the session-clip sweep, whose cost is tracks x scenes - the only part "
        "of the survey that grows without bound.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=64,
        metavar="N",
        help="cap the clip sweep at N scenes (default 64).",
    )
    parser.add_argument(
        "--max-collection",
        type=int,
        default=MAX_COLLECTION,
        metavar="N",
        help=f"how far the index walk goes when counting tracks, returns and scenes "
        f"(default {MAX_COLLECTION}). Live's collections report no size, so they are "
        "counted by walking - hitting this cap is reported, never rounded off.",
    )
    parser.add_argument(
        "--max-devices",
        type=int,
        default=MAX_DEVICES,
        metavar="N",
        help=f"how far the index walk goes per device chain (default {MAX_DEVICES}).",
    )
    parser.add_argument(
        "--max-parameters",
        type=int,
        default=None,
        metavar="N",
        help=f"how far the index walk goes per device's parameters "
        f"(default {MAX_PARAMETERS}, or the measured 128-slot ceiling with --parameters). "
        "Enough to classify a device; raise it to count one exactly.",
    )
    parser.add_argument(
        "--parameters",
        action="store_true",
        help="list every parameter name per device, not just the count. "
        "--json always carries them.",
    )
    parser.add_argument(
        "--user-library",
        metavar="PATH",
        help="skip User Library discovery and survey this folder.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        0 when nothing needs fixing, 1 when a PROBLEM finding was recorded,
        2 on misuse, 130 on Ctrl-C.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover - not a real stream
            pass

    args = _build_parser().parse_args(argv)
    if args.max_scenes < 0:
        print("--max-scenes cannot be negative.", file=sys.stderr)
        return 2
    if min(args.max_collection, args.max_devices) < 1:
        print("--max-collection and --max-devices must be at least 1.", file=sys.stderr)
        return 2

    modules = _load_maestro()
    threshold = modules[1].CONFIGURED_THRESHOLD if modules else 6
    slots = modules[1].VST_PARAMETER_SLOTS if modules else 128
    if args.max_parameters is None:
        args.max_parameters = slots if args.parameters else MAX_PARAMETERS
    if args.max_parameters < threshold:
        # Below the threshold the walk cannot tell "sparse" from "playable", and
        # a diagnosis that might be wrong is worse than no diagnosis.
        print(
            f"--max-parameters must be at least {threshold}: below that the walk cannot "
            "distinguish a sparsely configured device from a fully configured one, and the "
            "state column would be a guess.",
            file=sys.stderr,
        )
        return 2

    try:
        report = build_report(args)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("aborted", file=sys.stderr)
        return 130

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        render(report, args)

    return 1 if any(f["severity"] == "problem" for f in report["findings"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
