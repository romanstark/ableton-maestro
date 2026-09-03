"""FastMCP server surface for Ableton Maestro.

Exposes high-level musical intent tools (note editing, mixing, automation, arrangement),
generic Live Object Model tools (lom_get, lom_set, lom_call, lom_batch, lom_describe,
lom_enums), and reference resources.

All catalogued operations route through the executor against the catalog registry,
enforcing type validation, parameter range checks, confirmation guards for destructive
actions, and post-write verification.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import MISSING
from dataclasses import fields as dataclass_fields
from enum import Enum
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from ableton_maestro import automation as auto
from ableton_maestro import toolargs
from ableton_maestro.als import read as als_reader
from ableton_maestro.als import write as als_writer
from ableton_maestro.client import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    AbletonClient,
    AbletonConnectionError,
    AbletonError,
    AbletonTimeoutError,
)
from ableton_maestro.executor import Result, UnknownSpecError, execute, execute_batch
from ableton_maestro.lom import introspect, paths
from ableton_maestro.models import Access, Note, PathStatus
from ableton_maestro.music import notes as note_ops
from ableton_maestro.registry import CatalogError, Registry, area_of, default_registry
from ableton_maestro.spec import PathSpec
from ableton_maestro.toolargs import BatchOpArg, NoteArg

# stderr, never stdout: stdout is the stdio transport and one stray print there
# corrupts the JSON-RPC stream.
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("ableton_maestro.server")

REPO_ROOT = Path(__file__).resolve().parents[2]
LIMITS_DOC = REPO_ROOT / "docs" / "limits.md"

# --------------------------------------------------------------------------- #
# Technical finding notices repeated across tool responses.
# --------------------------------------------------------------------------- #

STORED_NOT_AUDIBLE = (
    "The read-back proves the value is stored in Live. It does not prove the change is "
    "audible (the device may be disabled, the track muted, or the clip inactive), "
    "that it survives an undo or reload, or that the saved .als agrees."
)

RAN_NOT_EFFECTIVE = (
    "The method executed without raising an exception. Method calls have no read-back "
    "verification; verify the resulting state with a read operation."
)

NORMALISED_NOTE = (
    "Parameter values lie between min and max (typically 0.0..1.0 on native devices, where "
    "0.85 corresponds to 0 dB). Values represent internal normalized units rather than "
    "the physical display units shown by Live. Measured 2026-08-31 against Live 12.4.5: "
    "Max devices and VSTs define independent parameter ranges."
)

GROUP_TRACK_NOTE = (
    "Group tracks do not support arming and have no arrangement clips. Grouping or ungrouping "
    "inserts or removes a container track, shifting all subsequent track indices. "
    "Measured 2026-08-30 against Live 12.4.5: Re-read session state after grouping changes."
)

#: Track indices represent position offsets rather than stable identities.
#: Creating, deleting, or grouping tracks renumbers all subsequent tracks.
STALE_INDEX_NOTE = (
    "A track index represents a position offset rather than a stable identity. Creating a "
    "track anywhere but the end, deleting one, and grouping or ungrouping all renumber every "
    "track after the change, and nothing in the LOM detects that a path built earlier now "
    "points somewhere else. Measured 2026-08-30 against Live 12.4.5: after one insert and one "
    "delete, song.tracks[1] answered a different track's name with ok: true and no warning, "
    "and a slot the earlier survey called an empty audio track held a synth. The name is not a "
    "fallback either: Live prefixes its own track names with a position number, so inserting "
    "one track at index 0 renamed all six others in the same breath, and 2-Kit-OP 808 became "
    "3-Kit-OP 808 on down the set. Only a name you chose yourself survives that. Re-survey "
    "after any structural change, and compare on a name you set yourself if you compare on "
    "anything."
)

#: Methods excluded from the Remote Script allowlist, mapped to safer tool alternatives.
_TOOL_FOR_METHOD: dict[str, str] = {
    "value_at_time": "read_automation samples an envelope (Session clips only).",
    "insert_step": "write_automation builds and verifies curves with read-back comparison.",
    "clear_envelope": "clear_automation reads the curve before removing it.",
    "clear_all_envelopes": "clear_automation with all_envelopes=True.",
    "set_notes": "write_clip_notes (uses extended note API).",
    "remove_notes": "write_clip_notes (uses extended note API).",
    "get_notes": "read_clip_notes (uses extended note API with per-note metadata).",
}

SESSION_ONLY_SURVEY_NOTE = (
    "This survey covers the Session grid only. Clips on the Arrangement timeline are a "
    "second, independent set and are not counted here (count field is session_clip_count). "
    "A track empty in Session view can still hold arrangement clips; read "
    "song.tracks[N].arrangement_clips to inspect them (group tracks have no arrangement clips)."
)

CONFIGURE_NOTE = (
    "Plugins reporting only 'Device On' have not had parameters exposed via Live Configure "
    "mode (up to 128 parameter slots per instance, measured against Live 12.4.5)."
)

SESSION_ONLY_NOTE = auto.SESSION_ONLY_NOTE

NOTES_REPLACE_NOTE = (
    "mode='replace' removes existing notes within the range before writing new notes. "
    "LOM note writes use add_new_notes. Use mode='append' to retain existing notes."
)

BATCH_NOT_ATOMIC = (
    "Batch operations execute sequentially. The Live Object Model does not support "
    "transactional rollbacks. atomic=True halts execution at the first encountered error."
)

SAVED_STATE_NOTE = (
    "Reads the saved .als project file on disk. Changes made in an open Live session are "
    "only reflected after saving the project (Ctrl+S)."
)


def _port() -> int:
    """Return Remote Script port from ABLETON_MAESTRO_PORT or default."""
    raw = os.environ.get("ABLETON_MAESTRO_PORT")
    if raw:
        try:
            return int(raw)
        except ValueError:
            logger.warning("ABLETON_MAESTRO_PORT=%r is not an integer; using %d", raw, DEFAULT_PORT)
    return DEFAULT_PORT


def _connection_fix() -> list[str]:
    """Return troubleshooting checklist for Remote Script connection issues."""
    port = _port()
    return [
        "Is Ableton Live running at all?",
        (
            "Preferences -> Link, Tempo & MIDI: one of the Control Surface slots must be set to "
            "'AbletonMaestro'. The folder name under Remote Scripts matches the dropdown entry."
        ),
        (
            "The script must sit at <User Library>/Remote Scripts/AbletonMaestro/__init__.py. "
            "The User Library path is specified in Preferences/Library.cfg under <ProjectPath>."
        ),
        (
            "After changing the script, delete its __pycache__ and restart Live. "
            "Live loads Remote Scripts at startup and can use cached bytecode."
        ),
        (
            f"This server connects to port {port} (127.0.0.1). Port 9877 belongs to another "
            "Ableton bridge service."
        ),
        f"Cold-start check from a shell: python -m ableton_maestro.client ping --port {port}",
    ]


# --------------------------------------------------------------- client singleton
_client: AbletonClient | None = None


def _client_instance() -> AbletonClient:
    """Return lazily initialized process-wide AbletonClient singleton."""
    global _client
    if _client is None:
        _client = AbletonClient(DEFAULT_HOST, _port())
    return _client


mcp = FastMCP(
    name="ableton-maestro",
    instructions=(
        "Produce music inside a running Ableton Live over the Live Object Model. "
        "Start with get_session: it reports the script handshake and the set. Writes report "
        "before/after/clamped, a result never claims an effect it did not read back, and a "
        "read-back proves the stored value, never audibility. Parameter values are normalised, "
        "not dB or Hz. Automation exists only in Session clips. Look paths up in the "
        "ableton://catalog resource and reach anything without an intent tool through "
        "lom_get/lom_set/lom_call/lom_batch/lom_describe. ableton://limits says what is "
        "impossible and why."
    ),
)


# --------------------------------------------------------------------- helpers
def _means_for_path(path: str, value: Any) -> str | None:
    """Return catalog description for value at the given path, or None."""
    if value is None:
        return None
    registry, _err = _registry_or_error()
    if registry is None:
        return None
    found = [
        (row.id, meaning)
        for row in registry.rows_for_path(path)
        if (meaning := row.meaning_of(value)) is not None
    ]
    if not found:
        return None
    if len({meaning for _rid, meaning in found}) == 1:
        return found[0][1]
    return " | ".join(f"{rid}: {meaning}" for rid, meaning in found)


#: Longest row doc a catalog hint carries inline. The hint is rare by design, so
#: this can afford to be generous; the full row is one resource read away.
_HINT_DOC_CAP = 600


def _catalog_hint(path: str, *, writing: bool) -> dict[str, Any] | None:
    """Return catalog hints about a raw path when informative.

    ``lom_get`` and ``lom_set`` take a path, never a row id, so nothing the catalog has
    measured about a property would otherwise reach the call that touches it. The cost is
    concrete: the ``-2`` of ``playing_slot_index`` and the fact that muting a sidechain
    source does not cut a Post FX sidechain are both measured and both written on their
    row, yet neither is reachable from the call that touches the property.

    Silence is the default. A note that arrives on every result becomes noise, and then
    the notes that were working stop being read: the stale-index warning rides on every
    single ``get_session``, and an index was still lost. So this returns ``None`` unless
    the catalog actively disagrees with what the caller is doing:

    * the row is not ``verified``, meaning Live is known to refuse this property, e.g.
      ``song.master_track.mute``, which reads like the most ordinary write in the
      world and is one of three measured-broken property paths;
    * a write is aimed at a row that does not grant ``set``. That is the
      DeviceParameter trap: ``song.tracks[0].mixer_device.volume`` grants
      ``get``/``observe``/``automate`` and the value lives one level down at
      ``…volume.value``.

    A ``verified`` row being read normally adds nothing and says nothing. Neither
    does an uncatalogued path: the dynamic surface of a loaded plug-in is not in
    the catalog by design, so "no row" is the expected answer there rather than a
    finding.
    """
    registry, _err = _registry_or_error()
    if registry is None:
        return None
    rows = registry.rows_for_path(path)
    if not rows:
        return None

    unverified = [row for row in rows if row.status is not PathStatus.VERIFIED]
    unsettable = (
        [row for row in rows if Access.SET not in row.access] if writing else []
    )
    if not unverified and not unsettable:
        return None

    hint: dict[str, Any] = {"rows": [row.id for row in rows]}
    notes: list[str] = []
    if unverified:
        hint["status"] = {row.id: row.status.value for row in unverified}
        notes.append(
            "The catalog has measured this path and it is NOT verified, so a clean "
            "answer here is not evidence that the property works."
        )
    if unsettable:
        hint["not_settable_per_catalog"] = [row.id for row in unsettable]
        # No "write X instead" here on purpose. For a DeviceParameter the script
        # already says it, with the CONCRETE index -- measured 2026-08-31:
        # "DeviceParameter is a Live object, not a value - write
        # song.tracks[5].mixer_device.volume.value instead". Repeating that from
        # up here could only offer the template form with {track} still in it,
        # which is strictly worse than what the layer below already returned.
        # What the script cannot know is that the catalog MEASURED the access, so
        # that is all this says -- and it is the only warning for a row granting
        # no set whose property the script would otherwise write happily.
        notes.append(
            "The catalog grants no `set` on this path: the access was measured, and a "
            "write here is outside what any row documents."
        )
    # The doc of the row that fired, not of every row that matched, and not a
    # pointer to a resource: being a resource nobody fetched is how this
    # knowledge went missing in the first place. Capped because a few docs run
    # long, with the full row still one resource read away.
    fired = unverified + [row for row in unsettable if row not in unverified]
    hint["doc"] = {
        row.id: (row.doc if len(row.doc) <= _HINT_DOC_CAP else row.doc[:_HINT_DOC_CAP] + " […]")
        for row in fired
        if row.doc
    }
    notes.append(
        "Full rows: read the `ableton://catalog/" + fired[0].id + "` resource."
    )
    hint["note"] = " ".join(notes)
    return hint


def _registry_or_error() -> tuple[Registry | None, dict[str, Any] | None]:
    """Load the path catalog, or return the failure as a tool result."""
    try:
        return default_registry(), None
    except (CatalogError, OSError) as exc:
        return None, {
            "ok": False,
            "error": f"the path catalog could not be loaded: {exc}",
            "hint": (
                "The catalog is in src/ableton_maestro/catalog/*.yaml with schema "
                "docs/catalog.md. Intent tools route through the catalog; generic "
                "lom_* tools operate directly without catalog validation."
            ),
        }


def _connection_error(exc: AbletonConnectionError) -> dict[str, Any]:
    """Return connection failure dictionary with troubleshooting checklist."""
    return {
        "ok": False,
        "error": str(exc),
        "reachable": False,
        "host": DEFAULT_HOST,
        "port": _port(),
        "fix": _connection_fix(),
    }


def _timeout_error(exc: AbletonTimeoutError) -> dict[str, Any]:
    """Return timeout error dictionary with handler and uncertainty details."""
    return {
        "ok": False,
        "error": str(exc),
        "timed_out": True,
        "handler": exc.handler,
        "timeout_seconds": exc.timeout,
        "may_have_landed": exc.may_have_landed,
        "note": (
            "A write that timed out may have landed and is not retried "
            "automatically. Verify with a read before re-sending."
            if exc.may_have_landed
            else "A read timed out; nothing was changed."
        ),
    }


def _live_error(exc: AbletonError) -> dict[str, Any]:
    """Convert an AbletonError into a standardized error response dictionary."""
    if isinstance(exc, AbletonConnectionError):
        return _connection_error(exc)
    if isinstance(exc, AbletonTimeoutError):
        return _timeout_error(exc)
    payload: dict[str, Any] = {"ok": False, "error": str(exc)}
    code = getattr(exc, "code", None)
    if code is not None:
        payload["code"] = str(code)
    path = getattr(exc, "path", None)
    if path is not None:
        payload["path"] = path
    return payload


def _enum_value(value: Any) -> Any:
    """Unwrap Enum instances for JSON serialization recursively."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_enum_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(k): _enum_value(v) for k, v in value.items()}
    return value


def _result_dict(result: Result, spec: PathSpec | None = None) -> dict[str, Any]:
    """Format an executor Result into a standardized tool response dictionary."""
    ok = bool(result.ok) and not result.blocked
    out: dict[str, Any] = {"ok": ok, "id": result.id, "path": result.path}
    if result.code:
        out["code"] = result.code
    if result.message:
        out["message"] = result.message
    if result.blocked:
        out["blocked"] = True
    for name in ("value", "before", "after", "display"):
        found = getattr(result, name)
        if found is not None:
            out[name] = found
    for name in ("clamped", "changed"):
        found = getattr(result, name)
        if found is not None:
            out[name] = found
    if result.read_back is not None:
        out["read_back"] = result.read_back
    if result.after is not None or result.verified is not None:
        out["verified"] = result.verified
        if result.verified is None:
            out["verified_note"] = (
                "the re-read was too early to see this write; read the value again to settle it"
                if result.read_back == "not_observed"
                else "not checked: verification was not executed"
            )
    if spec is not None:
        for field_name, key in (("value", "means"), ("after", "after_means")):
            got = getattr(result, field_name)
            meaning = spec.meaning_of(got) if got is not None else None
            if meaning is not None:
                out[key] = meaning
        out["catalog_status"] = spec.status.value
        if spec.unit is not None:
            out["unit"] = spec.unit.value
        if spec.quantized:
            out["quantized"] = True
            out["quantized_note"] = (
                "This parameter takes discrete steps, so a written value can snap "
                "to the nearest step."
            )
        if spec.status.value != "verified":
            out["caveat"] = (
                f"catalog row {spec.id!r} is '{spec.status.value}': unverified against Live."
            )
    if ok and result.after is not None:
        out["proves"] = STORED_NOT_AUDIBLE
    if result.code == "not_settable" and ".value" in (result.message or ""):
        out["hint"] = (
            "This path addresses a Live object (such as DeviceParameter) rather than a value. "
            "Write to the '.value' companion path instead (e.g. param.value_raw)."
        )
    return out


def _run(
    spec_id: str, *, confirm: bool = False, verify: bool = True, **args: Any
) -> dict[str, Any]:
    """Execute one catalog row and return an honest ``{ok, …}`` dict."""
    registry, err = _registry_or_error()
    if err is not None:
        return err
    assert registry is not None
    try:
        spec: PathSpec | None = registry.get(spec_id)
    except KeyError:
        spec = None
    try:
        result = execute(
            _client_instance(), registry, spec_id, confirm=confirm, verify=verify, **args
        )
    except UnknownSpecError as exc:
        return {"ok": False, "error": str(exc)}
    except AbletonError as exc:
        return _live_error(exc)
    except (ValueError, TypeError, KeyError) as exc:
        return {"ok": False, "error": f"invalid arguments for {spec_id}: {exc}"}
    return _result_dict(result, spec)


def _run_batch(
    ops: Sequence[Mapping[str, Any]], *, atomic: bool = False, verify: bool = False
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    """Execute multiple catalog operations in a single round trip."""
    registry, err = _registry_or_error()
    if err is not None:
        return None, err
    assert registry is not None
    try:
        results = execute_batch(
            _client_instance(), registry, list(ops), atomic=atomic, verify=verify
        )
    except UnknownSpecError as exc:
        return None, {"ok": False, "error": str(exc)}
    except AbletonError as exc:
        return None, _live_error(exc)
    except (ValueError, TypeError, KeyError) as exc:
        return None, {"ok": False, "error": f"invalid batch: {exc}"}
    specs: list[PathSpec | None] = []
    for op in ops:
        try:
            specs.append(registry.get(str(op.get("id"))))
        except KeyError:
            specs.append(None)
    return [_result_dict(r, s) for r, s in zip(results, specs, strict=False)], None


def _read_fields(
    wanted: Sequence[tuple[str, dict[str, Any]]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Batch-read catalog rows and format into a dictionary keyed by field label."""
    ops = [{"id": spec_id, **args} for spec_id, args in wanted]
    results, err = _run_batch(ops)
    if err is not None:
        return None, err
    assert results is not None
    fields: dict[str, Any] = {}
    for (spec_id, _args), row in zip(wanted, results, strict=False):
        label = spec_id.split(".", 1)[1] if "." in spec_id else spec_id
        entry: dict[str, Any] = {"id": spec_id, "ok": row.get("ok", False)}
        for key in ("value", "display", "unit", "code", "message"):
            if key in row:
                entry[key] = row[key]
        fields[label] = entry
    return fields, None


def _value_of(fields: Mapping[str, Any], label: str, default: Any = None) -> Any:
    """Return extracted field value or default if read failed."""
    entry = fields.get(label)
    if isinstance(entry, Mapping) and entry.get("ok"):
        return entry.get("value", default)
    return default


#: Live collections are Vector objects that do not all expose len().
#: Probing indices sequentially in batches provides reliable count determination.
SCAN_CAP = 128
SCAN_LIMIT = 1024

COLLECTION_NOTE = (
    "Collection sizes are measured by probing indices until Live reports index_out_of_range. "
    "Live collection Vector objects do not all expose len() directly. "
    "'exact': False indicates the probe reached its ceiling."
)


def _scan(
    scans: Sequence[tuple[str, str, str, Mapping[str, Any]]], cap: int = SCAN_CAP
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Query indexed catalog collections in a single batch probe."""
    state: dict[str, dict[str, Any]] = {
        label: {"id": spec_id, "values": [], "count": 0, "exact": False, "stopped_by": None}
        for label, spec_id, _index_arg, _base in scans
    }
    pending: dict[str, int] = {label: 0 for label, _s, _i, _b in scans}

    while pending:
        ops: list[dict[str, Any]] = []
        spans: list[tuple[str, int, int]] = []
        for label, spec_id, index_arg, base in scans:
            start = pending.get(label)
            if start is None:
                continue
            spans.append((label, len(ops), cap))
            ops.extend({"id": spec_id, **dict(base), index_arg: start + i} for i in range(cap))
        if not ops:
            break
        results, err = _run_batch(ops)
        if err is not None:
            return state, err
        assert results is not None
        for label, offset, size in spans:
            entry = state[label]
            chunk = results[offset : offset + size]
            stopped = False
            for row in chunk:
                if not row.get("ok"):
                    entry["stopped_by"] = row.get("code") or "error"
                    stopped = True
                    break
                entry["values"].append(row.get("value"))
            entry["count"] = len(entry["values"])
            if stopped:
                entry["exact"] = True
                pending.pop(label, None)
            elif len(chunk) < size or entry["count"] >= SCAN_LIMIT:
                entry["exact"] = False
                pending.pop(label, None)
            else:
                pending[label] = entry["count"]
    return state, None


def _count(
    spec_id: str, index_arg: str, base: Mapping[str, Any], cap: int = SCAN_CAP
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return item count and items for an indexed catalog collection."""
    state, err = _scan([("n", spec_id, index_arg, base)], cap=cap)
    return state.get("n", {"count": 0, "exact": False, "values": []}), err


def _grew(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool | None:
    """Return True if collection count increased, or None if indeterminate."""
    if not before.get("exact") or not after.get("exact"):
        return None
    return int(after.get("count", 0)) > int(before.get("count", 0))


def _shrank(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool | None:
    """Return True if collection count decreased, or None if indeterminate."""
    if not before.get("exact") or not after.get("exact"):
        return None
    return int(after.get("count", 0)) < int(before.get("count", 0))


def _verdict(call_ok: bool, moved: bool | None) -> tuple[bool, dict[str, Any]]:
    """Combine operation status and count verification into result dictionary."""
    if moved is None:
        return bool(call_ok), {
            "verified": None,
            "verified_note": (
                "not checked: the count could not be measured. "
                + COLLECTION_NOTE
            ),
        }
    return bool(call_ok) and moved, {"verified": moved}


def _clip_path(track: int, slot: int) -> str:
    """Return canonical LOM path for a Session clip slot."""
    return paths.clip(track, slot)


#: Metadata keys attached by Live note queries that are excluded from Note models.
UNMODELLED_NOTE_KEYS = frozenset({"note_id"})

#: Per-note warning codes excluded from duplicate issue aggregation.
_PER_NOTE_WARNINGS = frozenset({"ignored_key"})


def _merged_warnings(*reports: note_ops.ValidationReport) -> list[dict[str, Any]]:
    """Merge and deduplicate validation warnings from note validation stages."""
    seen: set[tuple[str, tuple[int, ...]]] = set()
    merged: list[dict[str, Any]] = []
    for report in reports:
        for issue in report.warnings:
            if issue.code in _PER_NOTE_WARNINGS:
                continue
            fingerprint = (issue.code, tuple(issue.indices or ()))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            merged.append(issue.to_dict())
    return merged


def _parse_notes(raw: Sequence[Any]) -> tuple[list[Note], list[str]]:
    """Parse note dictionaries returned by Live, filtering unmodelled metadata keys."""
    dropped: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            cleaned.append(entry)
            continue
        keep = {}
        for key, value in entry.items():
            if key in UNMODELLED_NOTE_KEYS:
                dropped.add(key)
                continue
            keep[key] = value
        cleaned.append(keep)
    return note_ops.from_dicts(cleaned), sorted(dropped)


def _track_path_for(kind: str, index: int) -> str:
    """Return canonical LOM path for track, return track, or master track."""
    if kind == "master":
        return paths.master()
    if kind == "return":
        return paths.return_track(index)
    return paths.track(index)


def _device_parameter_names(
    track_path: str, device: int, cap: int = SCAN_CAP
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Probe parameter names for a device by index."""
    if track_path == paths.master():
        return _count("master_device.parameter_name", "param", {"device": device}, cap=cap)
    if track_path.startswith("song.return_tracks["):
        index = _trailing_index(track_path)
        if index is None:
            return {"count": 0, "exact": False, "values": []}, None
        return _count(
            "return_device.parameter_name", "param", {"return": index, "device": device}, cap=cap
        )
    index = _trailing_index(track_path)
    if index is None:
        return {"count": 0, "exact": False, "values": []}, None
    return _count("param.name", "param", {"track": index, "device": device}, cap=cap)


def _trailing_index(path: str) -> int | None:
    """Extract trailing integer index from LOM path segment, or None."""
    match = re.search(r"\[(\d+)\]$", path)
    return int(match.group(1)) if match else None


def _view_from_names(track_path: str, device: int, names: Sequence[Any]) -> introspect.DeviceView:
    """Construct DeviceView instance from probed parameter names."""
    device_path = f"{track_path}.devices[{device}]"
    parameters = tuple(
        introspect.ParameterView(index, f"{device_path}.parameters[{index}]", str(name))
        for index, name in enumerate(names)
    )
    return introspect.DeviceView(
        path=device_path,
        name=device_path,
        class_name="",
        index=device,
        track=track_path,
        parameters=parameters,
    )


def _send_index(key: str | int) -> int:
    """Convert send identifier ('A', 'B', or numeric index) to integer index."""
    if isinstance(key, int):
        return key
    text = str(key).strip()
    if text.isdigit():
        return int(text)
    if len(text) == 1 and text.upper().isalpha():
        return ord(text.upper()) - ord("A")
    return -1


# Catalog mapping specifications per track kind.
_TRACK_KINDS: dict[str, dict[str, Any]] = {
    "track": {
        "param": "track",
        "fields": (
            "track.name", "track.color", "track.mute", "track.solo", "track.is_frozen",
            "track.can_be_armed", "track.is_foldable", "track.is_grouped",
            "track.current_monitoring_state", "track.volume", "track.panning",
            "track.sends", "track.devices", "track.clip_slots", "track.arrangement_clips",
        ),
        "volume": "track.volume_value",
        "panning": "track.panning_value",
        "send": "track.send_value",
        "mute": "track.mute",
        "solo": "track.solo",
        "devices": "device.list",
        "device_name": "device.name",
        "device_class": "device.class_name",
        "device_active": "device.is_active",
        "device_parameters": "device.parameters",
        "device_param_name": "param.name",
        "parameter": "param.value_raw",
        "device_param": "track",
    },
    "return": {
        "param": "ret",
        "fields": (
            "return.name", "return.color", "return.mute", "return.solo",
            "return.volume", "return.panning", "return.sends", "return.devices",
            "return.clip_slots",
        ),
        "volume": "return.volume_value",
        "panning": "return.panning_value",
        "send": "return.send_value",
        "mute": "return.mute",
        "solo": "return.solo",
        "devices": "return_device.list",
        "device_name": "return_device.name",
        "device_class": "return_device.class_name",
        "device_active": "return_device.is_active",
        "device_parameters": "return_device.parameters",
        "device_param_name": "return_device.parameter_name",
        "parameter": "return_device.parameter_value",
        "device_param": "return",
    },
    "master": {
        "param": None,
        "fields": (
            "master.name", "master.color", "master.volume",
            "master.panning", "master.cue_volume", "master.crossfader", "master.devices",
        ),
        "volume": "master.volume_value",
        "panning": "master.panning_value",
        "send": None,
        "mute": None,
        "solo": None,
        "devices": "master_device.list",
        "device_name": "master_device.name",
        "device_class": "master_device.class_name",
        "device_active": "master_device.is_active",
        "device_parameters": "master_device.parameters",
        "device_param_name": "master_device.parameter_name",
        "parameter": "master_device.parameter_value",
        "device_param": None,
    },
}


def _kind_args(kind: str, index: int) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return path arguments for track kind or error if unsupported."""
    profile = _TRACK_KINDS.get(kind)
    if profile is None:
        return {}, {
            "ok": False,
            "error": f"kind must be one of {sorted(_TRACK_KINDS)}, not {kind!r}",
        }
    param = profile["param"]
    return ({} if param is None else {param: index}), None


def _clip_fields(track: int, slot: int) -> Sequence[tuple[str, dict[str, Any]]]:
    """Return catalog specifications queried by get_clip."""
    args = {"track": track, "slot": slot}
    ids = (
        "clip.name", "clip.is_midi_clip", "clip.is_audio_clip", "clip.length",
        "clip.looping", "clip.loop_start", "clip.loop_end", "clip.start_marker",
        "clip.end_marker", "clip.muted", "clip.color", "clip.gain", "clip.warping",
        "clip.warp_mode", "clip.pitch_coarse", "clip.pitch_fine",
        "clip.signature_numerator", "clip.signature_denominator", "clip.has_envelopes",
        "clip.velocity_amount", "clip.legato", "clip.launch_mode",
    )
    return [(spec_id, dict(args)) for spec_id in ids]


# --------------------------------------------------------------------- state
@mcp.tool()
def get_session(clips: bool = True, devices: bool = True, max_scenes: int = 64) -> dict[str, Any]:
    """Survey the running set: script handshake, song fields, tracks, clips, devices.

    Args:
        clips: Sweep Session slots for active clips.
        devices: Include device chains for each track.
        max_scenes: Maximum number of scenes to probe for Session clips.

    Returns:
        Dictionary with script info, session snapshot, track counts, and notes.

    Note:
        Measured 2026-08-29 against Live 12.4.5: Round-trip handshake takes ~450 ms.
        Group tracks do not support arm state ('armed' returns null).
        If Live collection introspection reports zero tracks, index probing is used as a fallback.
    """
    client = _client_instance()
    try:
        info = client.script_info()
    except AbletonError as exc:
        return _live_error(exc)
    try:
        snap = introspect.snapshot(
            client, clips=clips, devices=devices, max_scenes=max(0, int(max_scenes))
        )
    except AbletonError as exc:
        return _live_error(exc)
    except introspect.IntrospectionError as exc:
        return {"ok": False, "error": str(exc), "script": info}

    groups = [t.index for t in snap.group_tracks]
    result: dict[str, Any] = {
        "ok": True,
        "script": info,
        "session": snap.to_dict(),
        "track_count": snap.track_count,
        "session_clip_count": snap.session_clip_count,
        "group_track_indices": groups,
        "notes": [
            (
                "A snapshot is a photograph of Live's memory and goes stale the moment a "
                "human touches the mouse. It says nothing about the .als on disk."
            ),
            GROUP_TRACK_NOTE if groups else "No group tracks in this set.",
            SESSION_ONLY_SURVEY_NOTE,
            STALE_INDEX_NOTE,
        ],
    }

    if snap.track_count == 0:
        probed, err = _count("track.name", "track", {})
        if err is None and probed.get("count"):
            result["track_count"] = probed["count"]
            result["track_names"] = probed["values"]
            result["track_count_exact"] = probed["exact"]
            result["diagnostics"] = [
                (
                    "The snapshot came back with no tracks while probing found "
                    f"{probed['count']}. The snapshot is wrong, not the set: it reads child "
                    "counts out of lom_describe, and this Vector came back without one "
                    "(docs/protocol.md §5.6 promises a count). track_names "
                    "below was probed directly and is trustworthy; 'session' is not."
                ),
                COLLECTION_NOTE,
            ]
    return result


@mcp.tool()
def get_track(track: int, kind: str = "track") -> dict[str, Any]:
    """Read one track's mixer state, flags, device chain, sends and clip slots.

    Args:
        track: index in ``song.tracks`` (or ``song.return_tracks`` for
            ``kind="return"``). Ignored for ``kind="master"``.
        kind: ``track`` | ``return`` | ``master``.

    ``volume`` and ``panning`` are normalised, not dB: 0.85 is 0 dB (measured),
    and the scale is not linear. ``display`` carries the dB reading where Live
    offered one.

    ``arm`` is read only after ``can_be_armed`` says the question applies, because
    reading it on a group track raises inside Live and would break the whole batch
    (measured). It comes back as ``null`` where it does not apply.

    Device names, send values and which clip slots are filled are measured by
    probing indices in one batch, because Live's collections report no length
    (see ``notes``).
    """
    args, err = _kind_args(kind, track)
    if err is not None:
        return err
    profile = _TRACK_KINDS[kind]
    fields, err = _read_fields([(spec_id, dict(args)) for spec_id in profile["fields"]])
    if err is not None:
        return err
    assert fields is not None

    device_base = {} if profile["device_param"] is None else {profile["device_param"]: track}
    scans: list[tuple[str, str, str, Mapping[str, Any]]] = [
        ("devices", profile["device_name"], "device", device_base)
    ]
    if profile["send"] is not None:
        scans.append(("sends", profile["send"], "send", dict(args)))
    if kind == "track":
        scans.append(("clip_slots", "clip_slot.has_clip", "slot", {"track": track}))
    measured, err = _scan(scans)
    if err is not None:
        return err

    armed: Any = None
    can_arm = _value_of(fields, "can_be_armed")
    if kind == "track" and can_arm:
        arm = _run("track.arm", track=track)
        armed = arm.get("value") if arm.get("ok") else None

    notes: list[str] = [NORMALISED_NOTE, COLLECTION_NOTE]
    if _value_of(fields, "is_foldable") or _value_of(fields, "is_grouped"):
        notes.append(GROUP_TRACK_NOTE)

    slots = measured.get("clip_slots", {})
    return {
        "ok": True,
        "kind": kind,
        "index": None if profile["param"] is None else track,
        "fields": fields,
        "arm": armed,
        "arm_applies": bool(can_arm) if kind == "track" else False,
        "devices": measured.get("devices", {}).get("values", []),
        "device_count": measured.get("devices", {}).get("count"),
        "sends": measured.get("sends", {}).get("values", []),
        "clip_slot_count": slots.get("count"),
        "filled_slots": [i for i, filled in enumerate(slots.get("values", [])) if filled],
        "counts_exact": {label: entry.get("exact") for label, entry in measured.items()},
        "notes": notes,
    }


@mcp.tool()
def get_clip(track: int, slot: int) -> dict[str, Any]:
    """Read one Session clip's properties: length, loop, warp, pitch, markers, flags.

    Checks the slot is filled first, because a clip only exists while its slot
    has one and every property read would otherwise fail one by one.

    Notes are not properties; use ``read_clip_notes``. Automation is not a property
    either; use ``read_automation``.

    ``warping``, ``warp_mode`` and ``pitch_coarse`` are reachable only through a generic
    path, because a per-command bridge has no command for them and a missing command is
    easily mistaken for a missing capability (docs/architecture.md, 'the restart tax').
    Each was read, written, read back and restored on 2026-08-29 against Live 12.4.5, and
    all three rows are ``verified``. ``pitch_fine`` is read-verified only, with no write
    attempted on it, and the integer mapping behind ``warp_mode`` is still a hypothesis:
    confirm it against ``clip.available_warp_modes``, which is the authoritative per-clip
    list.

    Writing ``warping`` invalidates every time field this tool just returned. Measured
    2026-08-30 against Live 12.4.5 on a 1.14-unit audio clip at 120 BPM: turning warping
    on took ``loop_end`` from 0.5704 to 1.1408 and ``end_marker`` from 1.1408 to 2.2816,
    both doubled, while ``length`` did not follow. At 60 BPM the same toggle moved
    neither. The mechanism is not established and is not claimed; what is established is
    that the numbers move and that the factor tracks the tempo. Call this again after any
    write to ``warping`` rather than reusing what it said before.
    """
    has = _run("clip_slot.has_clip", track=track, slot=slot)
    if not has.get("ok"):
        return has
    if not has.get("value"):
        return {
            "ok": True,
            "track": track,
            "slot": slot,
            "has_clip": False,
            "note": "The slot is empty; there is no clip to read. create_clip fills it.",
        }

    fields, err = _read_fields(_clip_fields(track, slot))
    if err is not None:
        return err
    assert fields is not None
    is_midi = _value_of(fields, "is_midi_clip")
    return {
        "ok": True,
        "track": track,
        "slot": slot,
        "has_clip": True,
        "path": _clip_path(track, slot),
        "fields": fields,
        "is_midi": is_midi,
        "notes": [
            (
                "Times (length, loop_start, loop_end, markers) are BEATS and clip-local: "
                "beat 0 is the clip's own start, not the arrangement's."
            ),
            "gain is normalised, not dB.",
            (
                "Live keeps notes that lie past the clip end (measured); a declared length "
                "shorter than the note content is normal, and that material never sounds."
                if is_midi
                else "Audio clip: transpose it with pitch_coarse/pitch_fine, not with notes."
            ),
        ],
    }


@mcp.tool()
def get_devices(track: int, kind: str = "track") -> dict[str, Any]:
    """List a track's device chain: names, class names, on/off, parameter counts.

    The parameter count is worth reading first. A third-party plug-in reporting exactly
    one parameter has not been configured; see the ``configure_needed`` flag and the note
    it carries. That is a limit in Live, not in this server, and it is lifted only in
    Live's GUI.

    The parameter probe here only has to settle "exactly one, or more than one",
    which is what tells an unconfigured plug-in apart from a configured one. For
    the full list call ``describe`` with ``with_parameters=True``.

    Device order can be changed, though it is widely assumed to be fixed. The call sits
    on the Song rather than on the Device, which is why it reads as missing:
    ``Song.move_device(device, target_chain, position)``, catalogued as
    ``song.move_device`` and reachable through ``lom_call``. Measured 2026-08-30 against
    Live 12.4.5 on a scratch track: ``['Eq8', 'Compressor2', ...]`` became
    ``['Compressor2', 'Eq8', ...]``.

    Loading order is still worth planning, because it decides what survives. Effects are
    appended, and an instrument replaces the instrument already on the track (measured),
    taking its settings and its clip envelopes with it. Reordering afterwards is a
    convenience; a replaced instrument is not recoverable.
    """
    _, err = _kind_args(kind, track)
    if err is not None:
        return err
    profile = _TRACK_KINDS[kind]
    device_base = {} if profile["device_param"] is None else {profile["device_param"]: track}

    listed, err = _count(profile["device_name"], "device", device_base)
    if err is not None:
        return err
    count = int(listed.get("count", 0))
    if count == 0:
        return {
            "ok": True,
            "kind": kind,
            "index": None if profile["param"] is None else track,
            "device_count": 0,
            "devices": [],
            "note": "No devices on this track.",
        }

    detail_ops = [
        {"id": profile[key], **device_base, "device": index}
        for index in range(count)
        for key in ("device_class", "device_active")
    ]
    details, err = _run_batch(detail_ops)
    if err is not None:
        return err
    assert details is not None

    params, err = _scan(
        [
            (str(index), profile["device_param_name"], "param", {**device_base, "device": index})
            for index in range(count)
        ],
    )
    if err is not None:
        return err

    devices: list[dict[str, Any]] = []
    any_unconfigured = False
    for index in range(count):
        class_row, active_row = details[index * 2 : index * 2 + 2]
        entry = params.get(str(index), {})
        exact = bool(entry.get("exact"))
        param_count = int(entry.get("count", 0))
        unconfigured = exact and param_count == 1
        any_unconfigured = any_unconfigured or unconfigured
        devices.append(
            {
                "index": index,
                "name": listed["values"][index],
                "class_name": class_row.get("value") if class_row.get("ok") else None,
                "is_active": active_row.get("value") if active_row.get("ok") else None,
                "parameter_count": param_count if exact else None,
                "parameter_count_at_least": None if exact else param_count,
                "configure_needed": unconfigured,
            }
        )

    notes = [
        (
            "parameter_count counts what the LOM exposes, which is not the same as what the "
            "device has."
        ),
        (
            "Devices CAN be reordered: Song.move_device(device, target_chain, position), "
            "catalog row song.move_device, through lom_call (measured 2026-08-30). Loading "
            "order still matters, because an instrument replaces rather than appends."
        ),
        COLLECTION_NOTE,
    ]
    if any_unconfigured:
        notes.append(CONFIGURE_NOTE)
    return {
        "ok": True,
        "kind": kind,
        "index": None if profile["param"] is None else track,
        "device_count": count,
        "device_count_exact": listed.get("exact"),
        "devices": devices,
        "notes": notes,
    }


@mcp.tool()
def describe(path: str, depth: int = 1, with_parameters: bool = False) -> dict[str, Any]:
    """Introspect any live object: class, properties, children, and methods.

    This is the route to the dynamic surface: what a loaded plug-in actually exposes,
    which no catalog can know in advance because it is decided at runtime and, for
    third-party plug-ins, by what the user picked up in Configure mode.

    Args:
        path: a LOM path, e.g. ``song``, ``song.tracks[0]``,
            ``song.tracks[0].devices[1]``. Look shapes up in ``ableton://catalog``.
        depth: how far to descend. Deep describes over a whole set can be slow
            and the cost is unmeasured.
        with_parameters: for a device path, survey every parameter (name, value,
            min/max, quantized steps, display unit) instead of just the child
            counts, and diagnose an unconfigured plug-in rather than returning a
            useless one-entry list.

    Methods are never callable through a path. Use ``lom_call``, and only names on the
    script's own allowlist.

    Measured caveat (Live 12.4.5): the parameter survey depends on
    ``lom_describe`` reporting a ``count`` for the ``parameters`` child, and a
    Live ``Vector`` that refuses ``len()`` reports none, so the survey comes back
    with zero parameters on a device that has them. When that happens this tool falls
    back to probing parameter indices, reports the names it found, and says which
    of the two answers you are looking at.
    """
    try:
        paths.validate(path)
    except paths.PathSyntaxError as exc:
        return {"ok": False, "error": str(exc)}

    client = _client_instance()
    if with_parameters:
        head, sep, tail = path.rpartition(".devices[")
        if not sep or not tail.endswith("]") or not tail[:-1].isdigit():
            return {
                "ok": False,
                "error": (
                    f"with_parameters needs a device path such as song.tracks[0].devices[1]; "
                    f"{path!r} is not one. Call describe without it for anything else."
                ),
            }
        device_index = int(tail[:-1])
        try:
            view = introspect.describe_device(client, head, device_index)
        except AbletonError as exc:
            return _live_error(exc)
        except introspect.IntrospectionError as exc:
            return {"ok": False, "error": str(exc)}

        notes = [
            (
                "value lies on the device's OWN scale, bounded by min and max. display "
                "carries the physical reading, and unit is parsed out of display and "
                "belongs to it, not to value. The two scales coincide for linear "
                "controls and DIVERGE for logarithmic ones. Measured 2026-08-30 against "
                "Live 12.4.5 on a Max device: LPF value 8794.30 of min 20 / max 21000 "
                "displayed 288 Hz, Release value 3780.74 of 1.5..20000 displayed 27.0 ms, "
                "Volume value -33.73 of -70..10 displayed -8.5 dB - while every "
                "percentage control agreed to one decimal. So aim in value units and read "
                "display back; a value is never the number display shows."
            )
        ]
        if view.advice:
            notes.append(view.advice)
        if not view.reports_units:
            notes.append(
                "This device reports no units at all, so the value stays normalised and "
                "this server will not invent hertz for it. Two causes, and they look the "
                "same from here: a VST2 never reports units however it is configured "
                "(measured), and an instance whose strip holds only Device On has nothing "
                "to report units for."
            )

        payload: dict[str, Any] = {"ok": True, "path": view.path, "device": view.to_dict()}
        if view.parameter_count == 0:
            probed, err = _device_parameter_names(head, device_index)
            if err is None and probed.get("count"):
                payload["probed_parameters"] = [
                    {"index": i, "name": name} for i, name in enumerate(probed["values"])
                ]
                payload["parameter_count"] = probed["count"] if probed["exact"] else None
                payload["parameter_count_at_least"] = (
                    None if probed["exact"] else probed["count"]
                )
                notes.insert(
                    0,
                    (
                        "The survey in 'device' reports 0 parameters and is WRONG: probing "
                        f"found {probed['count']}. lom_describe emitted no count for this "
                        "Live Vector (docs/protocol.md §5.6 promises one), so the "
                        "introspection layer sees an empty list. Trust "
                        "'probed_parameters'; use lom_get on "
                        f"{head}.devices[{device_index}].parameters[i] for values."
                    ),
                )
                notes.append(COLLECTION_NOTE)
        payload["notes"] = notes
        return payload

    try:
        info = client.describe(path, depth=max(1, int(depth)))
    except AbletonError as exc:
        return _live_error(exc)
    return {
        "ok": True,
        "path": path,
        "described": info,
        "notes": [
            (
                "Properties read and write freely through a path; methods do not, they go "
                "through lom_call and only from the script's allowlist."
            ),
            (
                "A child of type 'Vector' is a collection and does not always carry a count "
                "(measured, Live 12.4.5). Probe its indices, or use get_track/get_devices, "
                "which do."
            ),
        ],
    }


# --------------------------------------------------------------------- build
@mcp.tool()
def create_track(kind: str = "midi", index: int = -1, name: str = "") -> dict[str, Any]:
    """Insert a MIDI, audio, or return track and return its resolved index.

    Args:
        kind: Track kind ('midi', 'audio', or 'return').
        index: Insertion index (-1 inserts at the end without shifting existing track indices).
            Ignored for return tracks.
        name: Optional track name for MIDI or audio tracks.

    Returns:
        Dictionary reporting track creation status and before/after track counts.

    Note:
        Inserting before existing tracks renumbers subsequent track indices.
        Measured 2026-08-30 against Live 12.4.5.
    """
    kinds = {
        "midi": ("song.create_midi_track", "track.name", "track"),
        "audio": ("song.create_audio_track", "track.name", "track"),
        "return": ("song.create_return_track", "return.name", "ret"),
    }
    chosen = kinds.get(kind.lower())
    if chosen is None:
        return {"ok": False, "error": f"kind must be one of {sorted(kinds)}, not {kind!r}"}
    spec_id, name_id, index_arg = chosen

    before, err = _count(name_id, index_arg, {})
    if err is not None:
        return err

    call_args: list[Any] = [] if kind.lower() == "return" else [int(index)]
    created = _run(spec_id, call_args=call_args)
    if not created.get("ok"):
        return created

    after, err = _count(name_id, index_arg, {})
    if err is not None:
        return err
    grew = _grew(before, after)
    ok, evidence = _verdict(bool(created.get("ok")), grew)

    landed: int | None = None
    if grew and after.get("exact"):
        landed = after["count"] - 1 if int(index) < 0 or kind.lower() == "return" else int(index)
    named: dict[str, Any] | None = None
    names_after = after.get("values")
    if name and landed is not None:
        named = _run(name_id, **{index_arg: landed}, value=name)
        if named.get("ok"):
            refreshed, refresh_err = _count(name_id, index_arg, {})
            if refresh_err is None and refreshed.get("values") is not None:
                names_after = refreshed["values"]

    return {
        "ok": ok,
        "kind": kind,
        "requested_index": int(index),
        "count_before": before.get("count"),
        "count_after": after.get("count"),
        "assumed_index": landed,
        "names_after": names_after,
        "created": created,
        "named": named,
        **evidence,
        "notes": [
            (
                "assumed_index is dead-reckoned from the count and the requested position; "
                "the LOM does not report the new track."
            ),
            "Inserting anywhere but the end renumbers every later track.",
            COLLECTION_NOTE,
        ],
    }


def _selected_chain_names() -> list[str]:
    """Return device names on the currently selected track in signal order.

    The order is the one Live sends. Measured 2026-09-01 against Live 12.4.5.

    An empty result is ambiguous on purpose and must stay that way here: the caller
    reports both chains it compared, so "nothing was displaced" cannot be confused with
    "nothing was looked at".
    """
    listed = _run("view.selected_track_devices")
    if not listed.get("ok"):
        return []
    value = listed.get("value")
    if not isinstance(value, list):
        return []
    return [str(device.get("name") or "?") for device in value if isinstance(device, dict)]


@mcp.tool()
def load_device(
    query: str = "",
    root: str = "",
    uri: str = "",
    item_path: str = "",
    limit: int = 12,
    confirm: bool = False,
) -> dict[str, Any]:
    """Search Live's browser and load a device onto the currently selected track.

    Two-step process: with confirm=False, searches browser and returns candidate items
    along with currently selected track info. With confirm=True and a chosen uri or item_path,
    loads the item.

    Args:
        query: Search string for browser items.
        root: Optional browser root category to search within.
        uri: Browser item URI returned from search step.
        item_path: Direct filesystem or library item path.
        limit: Maximum number of search candidates to return (defaults to 12).
        confirm: Confirmation flag required to execute loading.

    Returns:
        Dictionary containing search candidates or device load verification details.

    Note:
        Browser.load_item targets the current selection rather than an explicit track argument.
        Target track is determined by song.view.selected_track.
        Inside racks or drum racks, target destination is determined by rack.view.selected_chain
        or rack.view.selected_drum_pad.
        Instruments replace the track instrument, whereas audio/MIDI effects are appended.
    """
    client = _client_instance()
    if not query and not (confirm and (uri or item_path)):
        return {
            "ok": False,
            "error": "give a query to search for, or a uri (or item_path) from an "
                     "earlier search together with confirm=True to load one.",
        }
    params: dict[str, Any] = {"query": query, "limit": max(1, min(int(limit), 200))}
    if root:
        params["root"] = root

    selected = _run("view.selected_track")
    selected_handle = selected.get("value") if selected.get("ok") else None

    if not confirm or not (uri or item_path):
        try:
            found = client.send("browser_walk", params)
        except AbletonError as exc:
            return _live_error(exc)
        matches = found.get("matches") or []
        return {
            "ok": True,
            "loaded": False,
            "selected_track": selected_handle,
            "candidates": matches,
            "count": found.get("count"),
            "truncated": found.get("truncated"),
            "next_step": (
                "Select the target track in Live, then call again with "
                "uri=<the uri of your choice> and confirm=True."
            ),
            "warning": (
                "Loading an instrument replaces the instrument already on the selected track. "
                "Effects are appended; the chain can be reordered afterwards with song.move_device."
            ),
        }

    resolved: dict[str, Any] = {}
    if item_path:
        try:
            paths.validate(item_path)
        except paths.PathSyntaxError as exc:
            return {"ok": False, "error": f"invalid item_path: {exc}"}
    else:
        try:
            resolved = client.send("browser_walk", {"uri": uri})
        except AbletonError as exc:
            return _live_error(exc)
    if not item_path and not resolved.get("found"):
        if resolved.get("truncated"):
            return {
                "ok": False,
                "error": (
                    f"the browser walk ran out of budget before reaching {uri!r} "
                    f"(stopped by {resolved.get('truncated_by') or 'a limit'}). "
                    "Find the item with browser_walk or lom_get, then call this "
                    "tool again with item_path='<the item path>' and confirm=True."
                ),
                "resolved": resolved,
            }
        return {"ok": False, "error": f"no browser item with uri {uri!r}", "resolved": resolved}
    item_path = item_path or resolved.get("item_path")

    before = _run("view.selected_track")
    chain_before = _selected_chain_names()
    loaded = _run("browser.load_item", confirm=True, call_args=[{"__path__": item_path}])
    after = _run("view.selected_track")
    chain_after = _selected_chain_names()
    displaced = [name for name in chain_before if name not in chain_after]

    notes: list[str] = []
    if displaced:
        notes.append(
            "REPLACED: " + ", ".join(repr(name) for name in displaced) + " is gone from "
            "the track. An instrument load replaces the instrument that was there."
        )
    elif chain_before and chain_after:
        notes.append(
            f"Nothing was displaced: the chain went {chain_before} -> {chain_after}."
        )
    notes.extend(
        [
            "Verify with get_devices on the selected track.",
            "Pace further loads: loading plugins in rapid succession can crash Live.",
            RAN_NOT_EFFECTIVE,
        ]
    )
    return {
        "ok": bool(loaded.get("ok")),
        "loaded": bool(loaded.get("ok")),
        "uri": uri,
        "item_path": item_path,
        "item": resolved.get("item"),
        "selected_track_before": before.get("value") if before.get("ok") else None,
        "selected_track_after": after.get("value") if after.get("ok") else None,
        "chain_before": chain_before,
        "chain_after": chain_after,
        "displaced": displaced,
        "call": loaded,
        "notes": notes,
    }


@mcp.tool()
def create_clip(track: int, slot: int, length_beats: float = 4.0, name: str = "") -> dict[str, Any]:
    """Create an empty MIDI clip of specified length in a Session slot.

    Args:
        track: Track index in song.tracks (MIDI track required).
        slot: Scene/slot index in track.clip_slots.
        length_beats: Clip loop length in beats (defaults to 4.0).
        name: Optional clip name.

    Returns:
        Dictionary containing clip creation status and read-back verification.

    Note:
        Does not overwrite occupied slots; call delete_clip first if replacing.
    """
    has = _run("clip_slot.has_clip", track=track, slot=slot)
    if not has.get("ok"):
        return has
    if has.get("value"):
        return {
            "ok": False,
            "error": (
                f"track {track} slot {slot} already has a clip. create_clip does not "
                "overwrite; delete_clip first if replacing."
            ),
            "has_clip": True,
        }

    created = _run("clip_slot.create_clip", track=track, slot=slot, call_args=[float(length_beats)])
    if not created.get("ok"):
        return created

    named: dict[str, Any] | None = None
    if name:
        named = _run("clip.name", track=track, slot=slot, value=name)
    check, err = _read_fields(
        [
            ("clip_slot.has_clip", {"track": track, "slot": slot}),
            ("clip.length", {"track": track, "slot": slot}),
            ("clip.is_midi_clip", {"track": track, "slot": slot}),
        ]
    )
    if err is not None:
        return err
    assert check is not None
    return {
        "ok": bool(_value_of(check, "has_clip")),
        "track": track,
        "slot": slot,
        "path": _clip_path(track, slot),
        "requested_length_beats": float(length_beats),
        "read_back": check,
        "named": named,
        "call": created,
        "note": "Length is in beats. The clip is empty until write_clip_notes fills it.",
    }


@mcp.tool()
def delete_clip(track: int, slot: int, confirm: bool = False) -> dict[str, Any]:
    """Delete the clip in a Session slot. Requires confirm=True.

    When confirm=False, inspects slot and reports clip details that would be deleted.

    Args:
        track: Track index in song.tracks.
        slot: Scene/slot index in track.clip_slots.
        confirm: Confirmation flag required to execute deletion.

    Returns:
        Dictionary reporting deletion status or pending loss report.
    """
    fields, err = _read_fields(
        [
            ("clip_slot.has_clip", {"track": track, "slot": slot}),
            ("clip.name", {"track": track, "slot": slot}),
            ("clip.length", {"track": track, "slot": slot}),
            ("clip.is_midi_clip", {"track": track, "slot": slot}),
            ("clip.has_envelopes", {"track": track, "slot": slot}),
        ]
    )
    if err is not None:
        return err
    assert fields is not None
    if not _value_of(fields, "has_clip"):
        return {"ok": True, "deleted": False, "track": track, "slot": slot,
                "note": "The slot is already empty; nothing to delete."}

    note_count: int | None = None
    if _value_of(fields, "is_midi_clip"):
        try:
            got = _client_instance().send("notes_get", {"path": _clip_path(track, slot)})
            note_count = got.get("count")
        except AbletonError:
            note_count = None

    loss = {
        "name": _value_of(fields, "name"),
        "length_beats": _value_of(fields, "length"),
        "is_midi": _value_of(fields, "is_midi_clip"),
        "note_count": note_count,
        "has_envelopes": _value_of(fields, "has_envelopes"),
    }
    if not confirm:
        return {
            "ok": False,
            "deleted": False,
            "confirm_required": True,
            "would_lose": loss,
            "error": (
                f"delete_clip is destructive. This would delete the clip in track {track} "
                f"slot {slot} with its notes and any clip automation on it. Call again with "
                "confirm=True."
            ),
        }

    deleted = _run("clip_slot.delete_clip", confirm=True, track=track, slot=slot)
    after = _run("clip_slot.has_clip", track=track, slot=slot)
    still_there = after.get("value") if after.get("ok") else None
    return {
        "ok": bool(deleted.get("ok")) and still_there is False,
        "deleted": still_there is False,
        "track": track,
        "slot": slot,
        "lost": loss,
        "has_clip_after": still_there,
        "call": deleted,
    }


def _members_of_group(
    track: int,
    measured: Mapping[str, Any],
    tracks: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return nested tracks under a group track by index and name."""
    grouped = measured.get("is_grouped", {}).get("values", [])
    names = tracks.get("values", [])
    members: list[dict[str, Any]] = []
    index = track + 1
    while index < len(grouped) and grouped[index]:
        members.append(
            {"index": index, "name": names[index] if index < len(names) else None}
        )
        index += 1
    return members


def _group_warning(loss: Mapping[str, Any]) -> str:
    """The sentence a group track's own emptiness must not be allowed to imply."""
    if not loss.get("is_group"):
        return ""
    members = loss.get("contained_tracks") or []
    if not members:
        return (
            " This is a GROUP track. It holds no clips or devices of its own, so the "
            "counts above are about the container and not about its contents; no member "
            "tracks were found beside it, which is unusual enough to check by eye first."
        )
    listed = ", ".join(f"{m['index']} ({m['name']!r})" for m in members)
    return (
        f" This is a GROUP track and the counts above are about the CONTAINER, not its "
        f"contents: it nests {len(members)} track(s) - {listed} - with their own devices, "
        "clips and automation. What Live does with them when the group goes is NOT "
        "established by this server and was not measured; look at them before confirming."
    )


@mcp.tool()
def delete_track(track: int, confirm: bool = False) -> dict[str, Any]:
    """Delete a track and renumber subsequent tracks. Requires confirm=True.

    When confirm=False, reports track name, devices, filled Session slots,
    and Arrangement clips that would be removed.

    Args:
        track: Track index in song.tracks.
        confirm: Confirmation flag required to execute deletion.

    Returns:
        Dictionary reporting deletion status or pending loss report.

    Note:
        Deleting a track shifts all subsequent track indices down by one.
    """
    fields, err = _read_fields(
        [
            ("track.name", {"track": track}),
            ("track.is_foldable", {"track": track}),
        ]
    )
    if err is not None:
        return err
    assert fields is not None
    measured, err = _scan(
        [
            ("tracks", "track.name", "track", {}),
            ("devices", "device.name", "device", {"track": track}),
            ("clip_slots", "clip_slot.has_clip", "slot", {"track": track}),
            ("arrangement", "arrangement_clip.name", "clip", {"track": track}),
            ("is_grouped", "track.is_grouped", "track", {}),
        ]
    )
    if err is not None:
        return err

    before = measured.get("tracks", {})
    slots = measured.get("clip_slots", {})
    arrangement = measured.get("arrangement", {})
    not_applicable = arrangement.get("stopped_by") == "no_such_path"
    arrangement_names = [] if not_applicable else list(arrangement.get("values", []))
    loss = {
        "name": _value_of(fields, "name"),
        "devices": measured.get("devices", {}).get("values", []),
        "filled_slots": [i for i, filled in enumerate(slots.get("values", [])) if filled],
        "arrangement_clips": None if not_applicable else len(arrangement_names),
        "arrangement_clip_names": arrangement_names,
        "is_group": _value_of(fields, "is_foldable"),
    }
    if loss["is_group"]:
        loss["contained_tracks"] = _members_of_group(track, measured, before)
    if not confirm:
        return {
            "ok": False,
            "deleted": False,
            "confirm_required": True,
            "would_lose": loss,
            "track_count": before.get("count"),
            "error": (
                f"delete_track is destructive. This would delete track {track} "
                f"({loss['name']!r}) with its {len(loss['devices'])} device(s), "
                f"{len(loss['filled_slots'])} Session clip(s), "
                f"{'no' if not_applicable else len(arrangement_names)} Arrangement clip(s) "
                "and all of their clip automation, and shift every later track index "
                "down by one." + _group_warning(loss) + " Call again with confirm=True."
            ),
        }

    deleted = _run("song.delete_track", confirm=True, call_args=[int(track)])
    after, err = _count("track.name", "track", {})
    if err is not None:
        return err
    ok, evidence = _verdict(bool(deleted.get("ok")), _shrank(before, after))
    return {
        "ok": ok,
        "deleted_index": track,
        "lost": loss,
        "track_count_before": before.get("count"),
        "track_count_after": after.get("count"),
        "call": deleted,
        **evidence,
        "warning": (
            "Track indices after this one have shifted down by one. Re-read the session "
            "before using any index you held from before this call."
        ),
    }


@mcp.tool()
def delete_device(track: int, device: int, confirm: bool = False) -> dict[str, Any]:
    """Remove a device from a track device chain. Requires confirm=True.

    When confirm=False, reports device details and parameter count that would be removed.

    Args:
        track: Track index in song.tracks.
        device: Device index within track chain.
        confirm: Confirmation flag required to execute deletion.

    Returns:
        Dictionary reporting deletion status or pending loss report.

    Note:
        Deleting a device shifts subsequent device indices down by one.
        Measured 2026-08-30 against Live 12.4.5: Devices can be reordered via song.move_device.
    """
    fields, err = _read_fields(
        [
            ("device.name", {"track": track, "device": device}),
            ("device.class_name", {"track": track, "device": device}),
        ]
    )
    if err is not None:
        return err
    assert fields is not None
    before, err = _count("device.name", "device", {"track": track})
    if err is not None:
        return err
    parameters, err = _count("param.name", "param", {"track": track, "device": device})
    if err is not None:
        return err

    loss = {
        "name": _value_of(fields, "name"),
        "class_name": _value_of(fields, "class_name"),
        "parameter_count": parameters.get("count"),
        "parameter_count_exact": parameters.get("exact"),
    }
    if not confirm:
        return {
            "ok": False,
            "deleted": False,
            "confirm_required": True,
            "would_lose": loss,
            "device_count": before.get("count"),
            "error": (
                f"delete_device is destructive. This would remove device {device} "
                f"({loss['name']!r}) from track {track}, with every setting on it and every "
                "clip envelope aimed at it, and shift later devices down one index. Call "
                "again with confirm=True."
            ),
        }

    deleted = _run("track.delete_device", confirm=True, track=track, call_args=[int(device)])
    after, err = _count("device.name", "device", {"track": track})
    if err is not None:
        return err
    ok, evidence = _verdict(bool(deleted.get("ok")), _shrank(before, after))
    return {
        "ok": ok,
        "track": track,
        "deleted_index": device,
        "lost": loss,
        "device_count_before": before.get("count"),
        "device_count_after": after.get("count"),
        "devices_after": after.get("values"),
        "call": deleted,
        **evidence,
    }


# --------------------------------------------------------------------- music
@mcp.tool()
def write_clip_notes(
    track: int,
    slot: int,
    notes: list[NoteArg],
    mode: str = "replace",
    verify: bool = True,
) -> dict[str, Any]:
    """Write MIDI notes into a Session clip. Replaces notes by default.

    Args:
        track: Track index in song.tracks (MIDI track).
        slot: Scene/slot index in track.clip_slots.
        notes: List of note dictionaries with pitch, start_time, duration, velocity,
            and optional probability, velocity_deviation, release_velocity, mute.
        mode: Write mode, 'replace' (clears prior notes and writes) or 'append'.
        verify: When True, reads back notes and compares with input using diff.

    Returns:
        Dictionary containing write confirmation, validation reports, and optional diff.

    Note:
        Times are in beats and clip-local, so beat 0 is the clip's own start.

        A list straight from ``read_clip_notes`` can be written back. Live adds
        ``note_id`` to every note it hands out, which is identity rather than content,
        so the keys in :data:`~ableton_maestro.music.notes.TOLERATED_NOTE_KEYS` are
        dropped instead of refused and reported as ``input_keys_ignored``. Every other
        unrecognised key is an error.

        ``pitch``, ``start_time`` and ``duration`` are never defaulted. A reader that
        substitutes ``start_time=0.0`` and ``duration=0.25`` for missing keys turns a
        list built with ``pos``/``dur`` (the humanise spelling) into a pile of
        sixteenths stacked on beat 0, and reports success while doing it. Wrong keys are
        refused before anything is sent.

        ``replace`` is remove-then-write inside one handler call. Live's own
        ``set_notes`` appends, so a second write silently doubles a melody instead of
        correcting it (measured: 63 + 23 = 86 notes). Ask for ``append`` by name when
        adding is the intent. An empty list with ``mode="replace"`` clears the clip.

        A note does not come back bit-identical to the note sent. Times and durations
        return with a small deviation in both directions. Measured 2026-09-01 against
        Live 12.4.5: a sent 0.29 read back as 0.29000010406260407 and a sent 0.18 as
        0.17999994796869798, about 4e-7 relative. Re-measured 2026-09-02 against Live
        12.4.5 at 124 BPM: both values reproduced to every digit, a ``start_time`` of
        2.29 came back as 2.290000104062604, and a duration of 0.5 came back exactly.
        The cause is not established: it is not a tick grid (no grid of 96, 192, 480 or
        960 per quarter produces those numbers) and not a single float32 conversion
        (float32 of 0.29 is 0.28999999). That 0.5 survives while 0.29 does not is
        consistent with a binary-representable value passing through untouched, but the
        mechanism is not claimed here. At 124 BPM the error is around 50 nanoseconds, so
        musically it is nothing. It matters only for comparison: never test a note time
        for equality. The diff run here already uses a tolerance, which is why it
        reports ``0 changed`` for values that differ in the seventh decimal.
    """
    if mode not in ("replace", "append"):
        return {"ok": False, "error": f"mode must be 'replace' or 'append', not {mode!r}"}

    raw = toolargs.note_dicts(notes)
    report = note_ops.validate_note_dicts(raw)
    if not report.ok:
        return {
            "ok": False,
            "blocked": True,
            "error": "the note list did not validate; nothing was sent",
            "validation": report.to_dict(),
        }
    raw, dropped_on_input = note_ops.without_tolerated_keys(raw)
    try:
        parsed = note_ops.from_dicts(raw)
    except ValueError as exc:
        return {"ok": False, "blocked": True, "error": str(exc)}

    clip_path = _clip_path(track, slot)
    length = _run("clip.length", track=track, slot=slot)
    clip_length = length.get("value") if length.get("ok") else None
    second = note_ops.validate(
        parsed, clip_length=clip_length if isinstance(clip_length, (int, float)) else None
    )
    if not second.ok:
        return {
            "ok": False,
            "blocked": True,
            "error": "the note list did not validate; nothing was sent",
            "validation": second.to_dict(),
        }

    try:
        written = _client_instance().send(
            "notes_set",
            {"path": clip_path, "notes": note_ops.to_dicts(parsed), "mode": mode},
        )
    except AbletonError as exc:
        return _live_error(exc)

    result: dict[str, Any] = {
        "ok": True,
        "track": track,
        "slot": slot,
        "path": clip_path,
        "mode": mode,
        "sent": len(parsed),
        "before_count": written.get("before_count"),
        "after_count": written.get("after_count"),
        "written": written.get("written"),
        "warnings": _merged_warnings(report, second),
        "notes": [NOTES_REPLACE_NOTE, STORED_NOT_AUDIBLE],
    }
    if dropped_on_input:
        result["input_keys_ignored"] = dropped_on_input
    if written.get("dropped_fields"):
        result["dropped_fields"] = written["dropped_fields"]

    if not verify:
        result["verified"] = None
        result["verified_note"] = "not checked: verification was not requested"
        return result

    try:
        back = _client_instance().send("notes_get", {"path": clip_path})
    except AbletonError as exc:
        result["verified"] = None
        result["verified_note"] = f"read-back failed, so nothing is proven: {exc}"
        return result

    try:
        actual, dropped_keys = _parse_notes(back.get("notes") or [])
    except ValueError as exc:
        result["verified"] = None
        result["verified_note"] = f"read-back was not parseable, so nothing is proven: {exc}"
        return result

    expected = parsed if mode == "replace" else None
    if expected is None:
        result["verified"] = None
        result["verified_note"] = (
            "append mode: the clip now holds the old notes plus the new ones, so a diff "
            "against what was sent cannot decide anything. read_clip_notes shows the result."
        )
        result["note_count_after"] = back.get("count")
        return result

    changes = note_ops.diff(expected, actual, ignore_defaulted_extensions=True)
    if dropped_keys:
        result["read_back_keys_ignored"] = dropped_keys
    result["verified"] = changes.is_empty
    result["diff"] = {
        "summary": changes.summary(),
        "added": [n.to_dict() for n in changes.added[:20]],
        "removed": [n.to_dict() for n in changes.removed[:20]],
        "changed": [c.to_dict() for c in changes.changed[:20]],
        "unchanged": changes.unchanged,
        "truncated": max(len(changes.added), len(changes.removed), len(changes.changed)) > 20,
    }
    result["note_count_after"] = back.get("count")
    return result


@mcp.tool()
def read_clip_notes(
    track: int,
    slot: int,
    from_time: float | None = None,
    time_span: float | None = None,
    count_only: bool = False,
    check: bool = True,
) -> dict[str, Any]:
    """Read MIDI notes of a Session clip, query a specific window, or count notes.

    Args:
        track: Track index in song.tracks.
        slot: Scene/slot index in track.clip_slots.
        from_time: Start beat of window to query. Omit to query from beat 0.
        time_span: Window length in beats. Omit to query to the end of the clip.
        count_only: When True, returns note count without transferring note data.
        check: When True, performs validation checks on queried notes.

    Returns:
        Dictionary containing notes, count, window parameters, and validation report.

    Note:
        Times and positions are expressed in clip-local beats. Muted notes are included
        and carry ``mute: true``. ``time_span`` is a length in beats, not an end beat:
        beats 24 to 32 is ``from_time=24, time_span=8``.

        Windowing and counting exist because there was otherwise no way to ask a small
        question. Measured 2026-08-30: surveying note density across 59 drum clips meant
        reading every note of every clip, and one clip alone came back as 384 notes and
        57k characters, past the tool-result cap, so it spilled to a file. The LOM has no
        ``note_count`` on ``Clip`` at all, so counting meant transferring everything in
        order to measure it. ``count_only=True`` counts inside Live and carries back one
        integer.

        With ``count_only`` there is nothing to validate, so ``check`` is ignored. With a
        window, ``check`` sees the window and not the clip.

        Returned notes are sorted by time, then pitch. Live hands its notes back ordered
        by pitch, which is the one order a musical instruction never means: "the third
        note", "every other note" and "the last note" are all about when a note sounds.
        Measured 2026-08-30 against Live 12.4.5: four notes written at beats 0, 1, 2, 3
        with pitches 72, 60, 67, 62 came back 60, 62, 67, 72, so taking every other entry
        off the raw list picks alternating pitches rather than alternating beats, silently
        and plausibly. For Live's own order call ``get_notes_extended`` through
        ``lom_call``.
    """
    clip_path = _clip_path(track, slot)
    params: dict[str, Any] = {"path": clip_path}
    if from_time is not None:
        params["from_time"] = float(from_time)
    if time_span is not None:
        params["time_span"] = float(time_span)
    if count_only:
        params["count_only"] = True
    try:
        got = _client_instance().send("notes_get", params)
    except AbletonError as exc:
        return _live_error(exc)

    windowed = from_time is not None or time_span is not None
    if count_only:
        return {
            "ok": True,
            "track": track,
            "slot": slot,
            "path": clip_path,
            "count": got.get("count"),
            "window": got.get("window"),
            "note": (
                "Counted inside Live; no notes were carried back. "
                + ("The count is for the window, not the clip." if windowed else "")
            ).strip(),
        }

    raw = got.get("notes") or []
    if isinstance(raw, list):
        raw = sorted(
            raw,
            key=lambda note: (
                (note.get("start_time"), note.get("pitch"))
                if isinstance(note, dict)
                and isinstance(note.get("start_time"), (int, float))
                and isinstance(note.get("pitch"), (int, float))
                else (float("inf"), float("inf"))
            ),
        )
    payload: dict[str, Any] = {
        "ok": True,
        "track": track,
        "slot": slot,
        "path": clip_path,
        "count": got.get("count", len(raw)),
        "notes": raw,
        "api": got.get("api"),
        "note": "Times are beats, clip-local: beat 0 is the clip's own start.",
    }
    if windowed:
        payload["window"] = got.get("window")
        end_desc = (from_time or 0) + time_span if time_span is not None else "the end"
        payload["window_note"] = (
            "These are the notes of one window, not the clip. time_span is a length "
            f"in beats, so this window is {from_time or 0} to {end_desc}."
        )
    if not check:
        return payload
    try:
        parsed, dropped_keys = _parse_notes(raw)
    except ValueError as exc:
        payload["check"] = {"ok": False, "error": str(exc)}
        return payload
    if dropped_keys:
        payload["keys_ignored"] = dropped_keys
        payload["keys_ignored_note"] = (
            "Live adds these to every note it hands back; they are identity, not content, "
            "and this server does not model or write them."
        )
    length = _run("clip.length", track=track, slot=slot)
    clip_length = length.get("value") if length.get("ok") else None
    report = note_ops.validate(
        parsed, clip_length=clip_length if isinstance(clip_length, (int, float)) else None
    )
    low, high = note_ops.span(parsed) if parsed else (0.0, 0.0)
    payload["check"] = report.to_dict()
    payload["span_beats"] = [low, high]
    payload["clip_length"] = clip_length
    return payload


@mcp.tool()
def quantize_clip(
    track: int,
    slot: int,
    grid: float = 0.25,
    strength: float = 1.0,
    quantize_ends: bool = False,
) -> dict[str, Any]:
    """Quantize MIDI clip note positions and lengths to a specified beat grid.

    Reads notes, calculates quantized positions, writes back using mode='replace',
    and verifies the result.

    Args:
        track: Track index in song.tracks.
        slot: Scene/slot index in track.clip_slots.
        grid: Quantization grid in beats (0.25 = sixteenth notes, 0.5 = eighth notes).
        strength: Quantization strength from 0.0 (no change) to 1.0 (full snap).
        quantize_ends: When True, also quantizes note durations / end positions.

    Returns:
        Dictionary containing write result and quantization details.
    """
    if grid <= 0:
        return {"ok": False, "error": f"grid must be positive (beats), got {grid}"}
    if not 0.0 <= strength <= 1.0:
        return {"ok": False, "error": f"strength must be between 0.0 and 1.0, got {strength}"}

    clip_path = _clip_path(track, slot)
    try:
        got = _client_instance().send("notes_get", {"path": clip_path})
        before, _ignored = _parse_notes(got.get("notes") or [])
    except AbletonError as exc:
        return _live_error(exc)
    except ValueError as exc:
        return {"ok": False, "error": f"the clip's notes did not parse: {exc}"}
    if not before:
        return {"ok": True, "track": track, "slot": slot, "moved": 0,
                "note": "The clip has no notes; nothing to quantize."}

    after = note_ops.quantize_times(
        before, grid, strength=strength, quantize_ends=quantize_ends
    )
    changes = note_ops.diff(before, after, match_on="pitch")
    written = write_clip_notes(track, slot, note_ops.to_dicts(after), mode="replace", verify=True)
    written["quantize"] = {
        "grid_beats": grid,
        "strength": strength,
        "quantize_ends": quantize_ends,
        "note_count": len(before),
        "planned_changes": changes.summary(),
    }
    written.setdefault("notes", []).append(
        "Test binary subdivisions before triplets: 0.5 is not a multiple of 1/3 but is a "
        "multiple of 1/6, so a triplet-first test reads every eighth as a sixteenth triplet."
    )
    return written


@mcp.tool()
def transpose_clip(
    track: int, slot: int, semitones: int, out_of_range: str = "error"
) -> dict[str, Any]:
    """Transpose a MIDI clip by note pitch or an audio clip via pitch_coarse.

    Args:
        track: Track index in song.tracks.
        slot: Scene/slot index in track.clip_slots.
        semitones: Number of semitones to transpose (positive or negative).
        out_of_range: Action when transposed notes exceed MIDI range 0..127
            ('error', 'clamp', or 'drop').

    Returns:
        Dictionary reporting transposition status and details.

    Note:
        Measured 2026-08-29 against Live 12.4.5: Audio clips are transposed by setting
        clip.pitch_coarse (-48..48 semitones).
    """
    if out_of_range not in ("error", "clamp", "drop"):
        return {"ok": False, "error": "out_of_range must be 'error', 'clamp' or 'drop'"}

    kind = _run("clip.is_midi_clip", track=track, slot=slot)
    if not kind.get("ok"):
        return kind
    if not kind.get("value"):
        current = _run("clip.pitch_coarse", track=track, slot=slot)
        base = current.get("value") if current.get("ok") else None
        target = int(semitones) if base is None else int(base) + int(semitones)
        moved = _run("clip.pitch_coarse", track=track, slot=slot, value=target)
        moved["audio_clip"] = True
        moved["pitch_coarse_before"] = base
        moved["requested_semitones"] = int(semitones)
        moved["note"] = (
            "Audio clip: transposed with pitch_coarse, not with notes. pitch_coarse is "
            "catalogued as verified (measured 2026-08-29 on Live 12.4.5), but the -48..48 "
            "bound is Live's documentation rather than a probe; the read-back above is the "
            "evidence for this write."
        )
        return moved

    clip_path = _clip_path(track, slot)
    try:
        got = _client_instance().send("notes_get", {"path": clip_path})
        before, _ignored = _parse_notes(got.get("notes") or [])
    except AbletonError as exc:
        return _live_error(exc)
    except ValueError as exc:
        return {"ok": False, "error": f"the clip's notes did not parse: {exc}"}
    if not before:
        return {"ok": True, "track": track, "slot": slot, "moved": 0,
                "note": "The clip has no notes; nothing to transpose."}

    try:
        after = note_ops.transpose(before, int(semitones), out_of_range=out_of_range)
    except ValueError as exc:
        return {"ok": False, "blocked": True, "error": str(exc)}

    written = write_clip_notes(track, slot, note_ops.to_dicts(after), mode="replace", verify=True)
    written["transpose"] = {
        "semitones": int(semitones),
        "out_of_range": out_of_range,
        "notes_in": len(before),
        "notes_out": len(after),
        "dropped": len(before) - len(after),
    }
    return written


# ----------------------------------------------------------------------- mix
@mcp.tool()
def set_mix(
    track: int,
    volume: float | None = None,
    pan: float | None = None,
    sends: dict[str, float] | None = None,
    mute: bool | None = None,
    solo: bool | None = None,
    kind: str = "track",
) -> dict[str, Any]:
    """Set mixer parameters (volume, pan, sends, mute, solo) in a single round trip.

    Args:
        track: Track index in song.tracks (or song.return_tracks). Ignored for kind="master".
        volume: Normalized volume level 0.0 to 1.0 (0.85 corresponds to 0 dB).
        pan: Panning position from -1.0 (hard left) to 1.0 (hard right).
        sends: Dictionary of send amounts 0.0 to 1.0 keyed by send letter ('A', 'B') or index.
        mute: Track mute state boolean (unsupported on master track).
        solo: Track solo state boolean (unsupported on master track).
        kind: Track kind ('track', 'return', or 'master').

    Returns:
        Dictionary reporting mixer adjustments and before/after verification readings.
    """
    args, err = _kind_args(kind, track)
    if err is not None:
        return err
    profile = _TRACK_KINDS[kind]

    ops: list[dict[str, Any]] = []
    labels: list[str] = []
    for field_name, value in (("volume", volume), ("panning", pan), ("mute", mute),
                              ("solo", solo)):
        if value is None:
            continue
        spec_id = profile.get(field_name if field_name != "panning" else "panning")
        if spec_id is None:
            return {"ok": False, "error": f"{kind} tracks have no {field_name}"}
        ops.append({"id": spec_id, **args, "value": value})
        labels.append(field_name)

    if sends:
        send_id = profile.get("send")
        if send_id is None:
            return {"ok": False, "error": f"{kind} tracks have no sends"}
        for key, amount in sends.items():
            index = _send_index(key)
            if index < 0:
                return {
                    "ok": False,
                    "error": (
                        f"send key {key!r} is not a letter (A, B, …) or an index ('0', '1', …)"
                    ),
                }
            ops.append({"id": send_id, **args, "send": index, "value": amount})
            labels.append(f"send[{index}]")

    if not ops:
        return {"ok": False, "error": "nothing to set: give volume, pan, sends, mute or solo"}

    results, err = _run_batch(ops, verify=True)
    if err is not None:
        return err
    assert results is not None
    changes = {label: row for label, row in zip(labels, results, strict=False)}
    return {
        "ok": all(row.get("ok") for row in results),
        "kind": kind,
        "index": None if profile["param"] is None else track,
        "round_trips": 1,
        "changes": changes,
        "notes": [NORMALISED_NOTE, BATCH_NOT_ATOMIC, STORED_NOT_AUDIBLE],
    }


@mcp.tool()
def set_parameter(
    track: int,
    device: int,
    parameter: str,
    value: float,
    kind: str = "track",
) -> dict[str, Any]:
    """Set a device parameter by numeric index or name/glob and verify read-back.

    Args:
        track: Track index in song.tracks (or song.return_tracks). Ignored for kind="master".
        device: Device index within track chain.
        parameter: Parameter index (as string e.g. "1") or parameter name/glob pattern.
        value: Parameter target value.
        kind: Track kind ('track', 'return', or 'master').

    Returns:
        Dictionary reporting write status, resolved parameter index, and read-back value.

    Note:
        Parameter values are normalised, not the unit the device displays. Read ``min``
        and ``max`` (``describe`` with ``with_parameters=True``) rather than assuming
        a range.

        The curve from 0..1 onto the displayed unit is not linear and differs per
        device. On one third-party compressor Attack is ``v^4 * 1000 ms``, so writing
        "10 ms" linearly lands at 316 ms, a factor of 30, silently. Where the device
        reports a display, aim with the display; where it does not (all VST2), write
        normalised and calibrate by eye once.

        Measured 2026-08-30 against Live 12.4.5: a quantized parameter takes discrete
        steps, so a written 0.5 can legitimately read back as something else, and that
        is reported as a clamp rather than as a failure.
    """
    _, err = _kind_args(kind, track)
    if err is not None:
        return err
    profile = _TRACK_KINDS[kind]
    device_arg = profile["device_param"]
    base = {} if device_arg is None else {device_arg: track}
    head = _track_path_for(kind, track)

    matches: list[dict[str, Any]] = []
    resolved_by = "index"
    text = str(parameter).strip()
    if text.lstrip("-").isdigit():
        index = int(text)
    else:
        try:
            found = introspect.find_parameter(_client_instance(), head, device, text)
        except AbletonError as exc:
            return _live_error(exc)
        except (introspect.IntrospectionError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}
        if found:
            matches = [m.to_dict() for m in found]
            index = found[0].index
            resolved_by = "introspection"
        else:
            probed, perr = _device_parameter_names(head, device)
            if perr is not None:
                return perr
            view = _view_from_names(head, device, probed.get("values", []))
            ranked = introspect.rank_parameters(view, text) if view.parameters else []
            if not ranked:
                names = [p.name for p in view.parameters]
                shown = names[:30]
                out: dict[str, Any] = {
                    "ok": False,
                    "error": f"no parameter matching {text!r} on device {device}",
                    "available": shown,
                    "note": COLLECTION_NOTE,
                }
                # An Operator has 195 parameters. Handing back the first 30 with no
                # sign of that reads as the whole list, and a caller who does not
                # find their name concludes it is absent.
                if len(names) > len(shown):
                    out["available_truncated"] = True
                    out["available_count"] = len(names)
                    out["available_note"] = (
                        f"showing {len(shown)} of {len(names)} parameter names. For the "
                        "full list call describe with with_parameters=True on this device."
                    )
                if view.parameter_count <= 1:
                    out["advice"] = CONFIGURE_NOTE
                return out
            matches = [m.to_dict() for m in ranked]
            index = ranked[0].index
            resolved_by = "index probe"

    written = _run(
        profile["parameter"], **base, device=int(device), param=int(index), value=float(value)
    )
    written["parameter_index"] = index
    written["resolved_by"] = resolved_by
    written["matches"] = matches
    written["notes"] = [NORMALISED_NOTE, STORED_NOT_AUDIBLE]
    return written


# ---------------------------------------------------------------- automation
@mcp.tool()
def write_automation(
    track: int,
    slot: int,
    parameter: str,
    points: list[list[float]],
    interpolation: str = "linear",
    exponent: float = 2.0,
    resolution: float = 0.0625,
    clear_first: bool = True,
    verify: bool = True,
) -> dict[str, Any]:
    """Write an automation envelope curve into a Session clip and verify read-back.

    Args:
        track: Track index in song.tracks (Session clip).
        slot: Scene/slot index in track.clip_slots.
        parameter: LOM path to DeviceParameter (e.g. 'song.tracks[0].mixer_device.volume').
        points: Breakpoint coordinates as [[beat, value], ...] in clip-local beats.
        interpolation: Interpolation mode ('linear', 'hold', 'exponential', 'ease_in', 'ease_out').
        exponent: Exponent for non-linear interpolation shapes.
        resolution: Sampling step size in beats (defaults to 0.0625 = 1/16th note).
        clear_first: When True, clears prior envelope points before writing.
        verify: When True, samples envelope back and compares with generated curve.

    Returns:
        Dictionary reporting write status, sampling metrics, and verification comparison.

    Note:
        Automation envelopes can be written directly to Session clips;
        subsequently arrange the clip to transfer automation to the Arrangement timeline.
    """
    try:
        paths.validate(parameter)
    except paths.PathSyntaxError as exc:
        return {"ok": False, "error": f"invalid parameter path: {exc}"}
    try:
        curve = auto.envelope(points, interpolation=interpolation, exponent=exponent)
    except ValueError as exc:
        return {"ok": False, "blocked": True, "error": str(exc)}

    clip_path = _clip_path(track, slot)
    params: dict[str, Any] = {
        "path": clip_path,
        "parameter": parameter,
        "clear_first": bool(clear_first),
        **auto.to_write_params(curve, resolution=float(resolution)),
    }
    try:
        written = _client_instance().send("automation_write", params)
    except AbletonError as exc:
        return _live_error(exc)

    result: dict[str, Any] = {
        "ok": True,
        "track": track,
        "slot": slot,
        "path": clip_path,
        "parameter": parameter,
        "parameter_name": written.get("parameter_name"),
        "breakpoints_sent": written.get("written"),
        "steps_inserted": written.get("steps"),
        "resolution_beats": written.get("resolution"),
        "interpolation": written.get("interpolation"),
        "cleared_first": written.get("cleared"),
        "probe_time": written.get("probe_time"),
        "before_sample": written.get("before_sample"),
        "after_sample": written.get("after_sample"),
        "notes": [SESSION_ONLY_NOTE, STORED_NOT_AUDIBLE],
    }
    if not verify:
        result["verified"] = None
        result["verified_note"] = "not checked: verification was not requested"
        return result

    start, end = auto.read_window(curve)
    try:
        actual = _client_instance().send(
            "automation_read",
            {"path": clip_path, "parameter": parameter, "start": start, "end": end, "points": 64},
        )
    except AbletonError as exc:
        result["verified"] = None
        result["verified_note"] = f"read-back failed, so nothing is proven: {exc}"
        return result

    comparison = auto.compare(curve, actual)
    result["verified"] = comparison.ok
    result["comparison"] = comparison.as_dict()
    result["ok"] = comparison.ok
    return result


@mcp.tool()
def read_automation(
    track: int,
    slot: int,
    parameter: str,
    start: float | None = None,
    end: float | None = None,
    points: int = 64,
) -> dict[str, Any]:
    """Sample an envelope curve for a parameter on a Session clip.

    Args:
        track: Track index in song.tracks.
        slot: Scene/slot index in track.clip_slots.
        parameter: LOM path to the target DeviceParameter.
        start: Optional start position in clip-local beats (default skips initial beat 0 guard).
        end: Optional end position in clip-local beats.
        points: Number of sample points to evaluate (clamped 2 to 512, default 64).

    Returns:
        Dictionary with sampled values, range, and envelope presence flags.
    """
    try:
        paths.validate(parameter)
    except paths.PathSyntaxError as exc:
        return {"ok": False, "error": f"invalid parameter path: {exc}"}

    params: dict[str, Any] = {
        "path": _clip_path(track, slot),
        "parameter": parameter,
        "points": max(2, min(int(points), 512)),
    }
    if start is not None:
        params["start"] = float(start)
    if end is not None:
        params["end"] = float(end)
    try:
        got = _client_instance().send("automation_read", params)
    except AbletonError as exc:
        return _live_error(exc)

    return {
        "ok": True,
        "track": track,
        "slot": slot,
        **got,
        "notes": [
            (
                "These are samples, not breakpoints; the LOM does not enumerate support points. "
                "For explicit breakpoints, read the set file with als_read."
            ),
            (
                "Readings at or near beat 0 return the parameter default value."
            ),
            SESSION_ONLY_NOTE,
        ],
    }


@mcp.tool()
def clear_automation(
    track: int, slot: int, parameter: str = "", all_envelopes: bool = False, confirm: bool = False
) -> dict[str, Any]:
    """Delete automation envelopes on a Session clip. Requires confirm=True.

    When confirm=False, inspects clip and reports envelope existence without deleting.

    Args:
        track: Track index in song.tracks.
        slot: Scene/slot index in track.clip_slots.
        parameter: Target DeviceParameter LOM path when clearing a single envelope.
        all_envelopes: When True, clears all envelopes on the clip.
        confirm: Confirmation flag required to execute clearing.

    Returns:
        Dictionary reporting clearance status or pending envelope details.
    """
    if not parameter and not all_envelopes:
        return {
            "ok": False,
            "error": (
                "give parameter=<a DeviceParameter path> to clear one envelope, or "
                "all_envelopes=True to clear every envelope on the clip"
            ),
        }
    if parameter:
        try:
            paths.validate(parameter)
        except paths.PathSyntaxError as exc:
            return {"ok": False, "error": f"invalid parameter path: {exc}"}

    clip_path = _clip_path(track, slot)
    has = _run("clip.has_envelopes", track=track, slot=slot)
    existing: dict[str, Any] | None = None
    if parameter and not all_envelopes:
        try:
            existing = _client_instance().send(
                "automation_read", {"path": clip_path, "parameter": parameter, "points": 16}
            )
        except AbletonError:
            existing = None

    if not confirm:
        return {
            "ok": False,
            "cleared": False,
            "confirm_required": True,
            "clip_has_envelopes": has.get("value") if has.get("ok") else None,
            "would_lose": (
                "every envelope on this clip"
                if all_envelopes
                else {
                    "parameter": parameter,
                    "has_envelope": (existing or {}).get("has_envelope"),
                    "moves": (existing or {}).get("moves"),
                    "range": [(existing or {}).get("min"), (existing or {}).get("max")],
                }
            ),
            "error": (
                "clear_automation is destructive and cannot be undone here; "
                "pass confirm=True."
            ),
        }

    params: dict[str, Any] = {"path": clip_path}
    if all_envelopes:
        params["all"] = True
    else:
        params["parameter"] = parameter
    try:
        cleared = _client_instance().send("automation_clear", params)
    except AbletonError as exc:
        return _live_error(exc)

    after = _run("clip.has_envelopes", track=track, slot=slot)
    return {
        "ok": bool(cleared.get("cleared")),
        "track": track,
        "slot": slot,
        "cleared_all": bool(cleared.get("cleared_all")),
        "parameter": cleared.get("parameter"),
        "parameter_name": cleared.get("parameter_name"),
        "clip_has_envelopes_before": has.get("value") if has.get("ok") else None,
        "clip_has_envelopes_after": after.get("value") if after.get("ok") else None,
        "note": (
            "has_envelopes is the read-back. It is a per-clip flag, so it stays true while "
            "any other envelope remains."
        ),
    }


# --------------------------------------------------------------- arrangement
@mcp.tool()
def arrange(track: int, slot: int, at_beat: float, to_track: int | None = None) -> dict[str, Any]:
    """Duplicate a Session clip into the Arrangement timeline at a beat position.

    Args:
        track: Source track index in song.tracks.
        slot: Source scene/slot index in track.clip_slots.
        at_beat: Target start position on the Arrangement timeline in beats.
        to_track: Destination track index (defaults to source track).

    Returns:
        Dictionary containing placement status and arrangement clip counts.
    """
    destination = track if to_track is None else int(to_track)
    before, err = _count("arrangement_clip.name", "clip", {"track": destination})
    if err is not None:
        return err

    placed = _run(
        "arrangement_clip.duplicate_from_session",
        track=destination,
        call_args=[{"__path__": _clip_path(track, slot)}, float(at_beat)],
    )
    if not placed.get("ok"):
        return placed

    after, err = _count("arrangement_clip.name", "clip", {"track": destination})
    if err is not None:
        return err
    grew = _grew(before, after)
    ok, evidence = _verdict(bool(placed.get("ok")), grew)

    detail: dict[str, Any] | None = None
    if grew and after.get("exact") and after["count"]:
        index = after["count"] - 1
        fields, ferr = _read_fields(
            [
                ("arrangement_clip.name", {"track": destination, "clip": index}),
                ("arrangement_clip.start_time", {"track": destination, "clip": index}),
                ("arrangement_clip.end_time", {"track": destination, "clip": index}),
                ("arrangement_clip.length", {"track": destination, "clip": index}),
            ]
        )
        if ferr is None:
            detail = fields

    return {
        "ok": ok,
        "source": {"track": track, "slot": slot, "path": _clip_path(track, slot)},
        "destination_track": destination,
        "at_beat": float(at_beat),
        "arrangement_clip_count_before": before.get("count"),
        "arrangement_clip_count_after": after.get("count"),
        "last_clip": detail,
        "call": placed,
        **evidence,
        "notes": [
            SESSION_ONLY_NOTE,
            (
                "last_clip is the last entry in the destination track's arrangement list, "
                "which is the new one only if nothing else was added in between."
            ),
            COLLECTION_NOTE,
        ],
    }


@mcp.tool()
def set_locator(at_beat: float, name: str = "", confirm: bool = False) -> dict[str, Any]:
    """Place or remove an Arrangement cue locator. Requires confirm=True.

    Live's song.set_or_delete_cue toggles: adds a locator if absent, or deletes if present.
    When confirm=False, reports current locator positions and requires confirmation.

    Args:
        at_beat: Timeline position in beats.
        name: Optional name for newly created locator.
        confirm: Confirmation flag required to execute locator toggle.

    Returns:
        Dictionary reporting locator status and resulting cue points.
    """
    before, err = _count("cue.time", "cue", {})
    if err is not None:
        return err

    if not confirm:
        return {
            "ok": False,
            "confirm_required": True,
            "at_beat": float(at_beat),
            "locator_count": before.get("count"),
            "locator_times": before.get("values"),
            "error": (
                "set_or_delete_cue is a toggle: at a beat that already carries a locator it "
                "deletes it. Check locator_times above, then call again with confirm=True."
            ),
        }

    moved = _run("song.current_song_time", value=float(at_beat))
    if not moved.get("ok"):
        return moved
    toggled = _run("song.set_or_delete_cue", confirm=True)
    if not toggled.get("ok"):
        return toggled

    after, err = _count("cue.time", "cue", {})
    if err is not None:
        return err
    created = _grew(before, after)
    deleted = _shrank(before, after)
    ok, evidence = _verdict(bool(toggled.get("ok")), created)

    named: dict[str, Any] | None = None
    if created and name:
        for index, value in enumerate(after.get("values", [])):
            if isinstance(value, (int, float)) and abs(float(value) - float(at_beat)) < 1e-6:
                named = _run("cue.name", cue=index, value=name)
                break

    return {
        "ok": ok,
        "created": created,
        "deleted": deleted,
        "at_beat": float(at_beat),
        "playhead": moved,
        "locator_count_before": before.get("count"),
        "locator_count_after": after.get("count"),
        "locator_times_after": after.get("values"),
        "named": named,
        **evidence,
        "note": (
            "A locator was deleted, not created: there was already one at this position."
            if deleted
            else "The locator count is the evidence; the LOM does not report the new cue point."
        ),
    }


@mcp.tool()
def set_arrangement_time(at_beat: float) -> dict[str, Any]:
    """Move the Arrangement playhead to a beat position and return read-back position.

    Args:
        at_beat: Target song time in beats from timeline start.

    Returns:
        Dictionary with write confirmation and read-back song time.
    """
    result = _run("song.current_song_time", value=float(at_beat))
    result["note"] = (
        "Compare 'after' with requested beat to verify playhead position."
    )
    return result


# ----------------------------------------------------------------- transport
@mcp.tool()
def play(from_beat: float | None = None, continue_playing: bool = False) -> dict[str, Any]:
    """Start transport playback and read back is_playing status.

    Args:
        from_beat: Optional timeline position in beats to cue before playback.
        continue_playing: When True, resumes from current position rather than restarting.

    Returns:
        Dictionary reporting start status and is_playing confirmation.
    """
    moved: dict[str, Any] | None = None
    if from_beat is not None:
        moved = _run("song.current_song_time", value=float(from_beat))
        if not moved.get("ok"):
            return moved
    started = _run("song.continue_playing" if continue_playing else "song.start_playing")
    state = _run("song.is_playing")
    return {
        "ok": bool(started.get("ok")) and bool(state.get("value")),
        "is_playing": state.get("value") if state.get("ok") else None,
        "from_beat": from_beat,
        "playhead": moved,
        "call": started,
        "note": "is_playing is the read-back; the call itself proves only that it did not raise.",
    }


@mcp.tool()
def stop(clips: bool = False, quantized: bool = True) -> dict[str, Any]:
    """Stop transport playback and optionally stop all active Session clips.

    Args:
        clips: When True, also stops active Session clips via song.stop_all_clips.
        quantized: Whether clip stop adheres to global launch quantization.

    Returns:
        Dictionary reporting stop status and is_playing confirmation.
    """
    stopped = _run("song.stop_playing")
    clip_stop: dict[str, Any] | None = None
    if clips:
        clip_stop = _run("song.stop_all_clips", call_args=[bool(quantized)])
    state = _run("song.is_playing")
    return {
        "ok": bool(stopped.get("ok")) and state.get("value") is False,
        "is_playing": state.get("value") if state.get("ok") else None,
        "clips_stopped": clip_stop,
        "call": stopped,
    }


@mcp.tool()
def set_tempo(bpm: float) -> dict[str, Any]:
    """Set song tempo in BPM and read back stored value.

    Args:
        bpm: Target tempo in beats per minute (catalog validated 20.0 to 999.0).

    Returns:
        Dictionary reporting tempo write and read-back result.
    """
    result = _run("song.tempo", value=float(bpm))
    result["note"] = "BPM is a real unit here. Device parameters are not; those are normalised."
    return result


@mcp.tool()
def set_loop(
    enabled: bool | None = None, start: float | None = None, length: float | None = None
) -> dict[str, Any]:
    """Configure Arrangement loop brace parameters (enabled, start, length).

    Args:
        enabled: Optional loop enabled state boolean.
        start: Optional loop start position in beats.
        length: Optional loop length in beats.

    Returns:
        Dictionary reporting write status and before/after verification values.
    """
    ops: list[dict[str, Any]] = []
    labels: list[str] = []
    for label, spec_id, value in (
        ("enabled", "song.loop", enabled),
        ("start", "song.loop_start", start),
        ("length", "song.loop_length", length),
    ):
        if value is None:
            continue
        ops.append({"id": spec_id, "value": value})
        labels.append(label)
    if not ops:
        return {"ok": False, "error": "nothing to set: give enabled, start or length"}

    results, err = _run_batch(ops, verify=True)
    if err is not None:
        return err
    assert results is not None
    return {
        "ok": all(row.get("ok") for row in results),
        "round_trips": 1,
        "changes": dict(zip(labels, results, strict=False)),
        "note": "start and length are beats. length is a duration, not an end position.",
    }


# ---------------------------------------------------------- file (Channel B)
@mcp.tool()
def als_read(
    path: str, with_notes: bool = False, track: str = "", report: bool = False
) -> dict[str, Any]:
    """Read saved Ableton project (.als) or rack (.adg) structure directly from disk.

    Args:
        path: Filesystem path to .als or .adg file.
        with_notes: When True, parses clip notes and calculates note metrics.
        track: Optional track name or index to query a specific track instead of the entire project.
        report: When True, includes formatted human-readable summary report.

    Returns:
        Dictionary containing project metadata, track list, devices, and automation structure.

    Note:
        This is Channel B, a peer of the live connection rather than a fallback. It
        answers two questions the LOM never will: what is in someone else's project, and
        where an envelope's breakpoints actually sit.

        Automation lives in two places and the LOM can read only one of them. Measured
        over 174 professional projects: 52 (30 %) have clip envelopes but 159 (91 %) have
        track automation, and a tool that counted clip envelopes alone reported 110 of
        those projects as unautomated. The two layers are reported separately here and
        are never added together.
    """
    target = Path(path).expanduser()
    if not target.exists():
        return {"ok": False, "error": f"no such file: {target}"}
    try:
        project = als_reader.read_project(target, with_notes=with_notes)
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": f"could not read {target.name}: {exc}"}

    if track:
        selector: int | str = int(track) if track.lstrip("-").isdigit() else track
        try:
            found = project.track(selector)
        except (KeyError, IndexError) as exc:
            return {
                "ok": False,
                "error": f"no track {track!r} in {project.file}: {exc}",
                "tracks": [t.name for t in project.tracks],
            }
        return {
            "ok": True,
            "file": project.file,
            "track": {
                "name": found.name,
                "type": found.type,
                "volume": found.volume,
                "panning": found.panning,
                "sends": found.sends,
                "devices": [
                    {
                        "name": d.name,
                        "kind": d.kind,
                        "on": d.on,
                        "slots": d.param_count,
                        "configured_parameters": len(d.configured_parameters),
                        "sidechain": None if d.sidechain is None else vars(d.sidechain),
                    }
                    for d in found.devices
                ],
                "session_clips": len(found.session_clips),
                "arrangement_clips": len(found.arrangement_clips),
                "track_automation": [
                    {
                        "target": e.target,
                        "points": e.points,
                        "from": e.time_from,
                        "to": e.time_to,
                        "min": e.value_min,
                        "max": e.value_max,
                    }
                    for e in found.track_automation
                ],
            },
            "notes": [
                SAVED_STATE_NOTE,
                (
                    "Volume here is a linear factor (1.0 = 0 dB), not the normalized 0.85 "
                    "reported by LOM faders."
                ),
            ],
        }

    automation = als_reader.automation_of(project)
    payload: dict[str, Any] = {
        "ok": True,
        "file": project.file,
        "path": project.path,
        "live_version": project.live_version,
        "creator": project.creator,
        "tempo": project.tempo,
        "is_rack": project.is_rack,
        "track_count": len(project.tracks),
        "tracks": [
            {
                "name": t.name,
                "type": t.type,
                "devices": [d.name for d in t.devices],
                "session_clips": len(t.session_clips),
                "arrangement_clips": len(t.arrangement_clips),
                "track_automation": len(t.track_automation),
                "clip_envelopes": sum(
                    len(c.envelopes) for c in t.session_clips + t.arrangement_clips
                ),
            }
            for t in project.tracks
        ],
        "automation": {
            "clip_envelopes": len(automation.clip_envelopes),
            "track_automation": len(automation.track_automation),
            "automated": automation.automated,
            "note": (
                "Clip envelopes and track automation are reported separately. "
                "Only clip envelopes can be written via Session clips; "
                "track automation is read-only from saved sets."
            ),
        },
        "notes": [SAVED_STATE_NOTE],
    }
    if report:
        payload["report"] = als_reader.format_report(project, show_devices=True)
    if with_notes:
        payload["note_clips"] = [
            {"track": name, "clip": clip.name, "notes": clip.note_count,
             "where": clip.where, "fingerprint": clip.fingerprint}
            for name, clip in als_reader.unique_note_clips(project)
        ]
    return payload


def _set_open_in_live(target: Path) -> tuple[bool | None, str]:
    """Check whether target project is currently opened in running Live session.

    Asks Live for ``song.file_path``. Measured 2026-09-01: it returned the full path of
    the open set. A running Live holding a different set is therefore no reason to
    refuse, and a running Live holding this one is a reason no flag should override: the
    edit would survive on disk only until Live's next save, and then vanish with no
    error anywhere.

    Returns:
        ``(open_here, detail)``. ``None`` means Live could not be asked, which is not
        the same as no, and the caller keeps the conservative refusal in that case.
    """
    try:
        got = _client_instance().get("song.file_path")
    except AbletonError as exc:
        return None, f"Live could not be asked which set is open ({exc})"
    loaded = str(got.get("value") or "")
    if not loaded:
        return False, "Live has an unsaved set open, so it is not holding this file"
    try:
        same = Path(loaded).resolve() == target.resolve()
    except OSError:
        same = loaded.casefold() == str(target).casefold()
    if same:
        return True, f"Live has THIS set open: {loaded}"
    return False, f"Live is running but has a different set open: {loaded}"


@mcp.tool()
def als_write(
    path: str,
    operation: str,
    confirm: bool = False,
    target_track: str = "",
    source_track: str = "",
    device: int = 0,
    tap: str = "post",
    expression: str = "",
    value: str = "",
    attribute: str = "Value",
    index: int | None = None,
    create: bool = False,
    backup: str = "",
    allow_live_running: bool = False,
) -> dict[str, Any]:
    """Edit a saved .als project file on disk. Requires confirm=True.

    Supported operations: 'sidechain', 'configure', 'attribute', 'restore'.
    Target project file must not be currently open in Live unless confirmed.

    Args:
        path: Path to target .als project file.
        operation: Modification operation ('sidechain', 'configure', 'attribute', 'restore').
        confirm: Confirmation flag required to execute modifications.
        target_track: Name of track containing target device.
        source_track: Name of track providing trigger source (sidechain operation).
        device: Device index within target track.
        tap: Sidechain tap point ('pre' or 'post').
        expression: ElementTree path expression (attribute operation).
        value: New attribute value or configure assignment string ('<index>=<name>; ...').
        attribute: Attribute name to modify (attribute operation, defaults to 'Value').
        index: Target element match index when expression matches multiple elements.
        create: When True, creates attribute if missing.
        backup: Backup filepath required for restore operation.
        allow_live_running: Allow write when Live runs (auto-enabled if holding another set).

    Returns:
        Dictionary reporting write status, changes made, and verification report.

    Note:
        ``sidechain`` is here for the set Live is not holding, someone else's project
        wired on disk. Measured in a corpus of 174 projects: 74 of 82 in the
        house/trance BPM window use sidechain compression, median 6 wirings each. For
        the set on screen use ``lom_set`` and keep the read-back.

        ``configure`` fills a plugin's parameter strip, which no LOM call can do
        whatever the strip currently holds. Measured 2026-09-01 against Live 12.4.5: two
        parameters written into an instance whose strip held nothing came back on reopen
        with their real values, writable through ``lom_set`` with ``read_back: applied``.
        Live allocates 128 slots per instance and leaves them all in the file, so
        nothing is inserted; three fields of an existing element are filled.

        ``attribute`` sets one attribute anywhere in the file, addressed by an
        ElementTree expression. An expression matching more than one element is refused
        unless ``index`` picks one.
    """
    target = Path(path).expanduser()
    if operation not in ("sidechain", "attribute", "restore", "configure"):
        return {
            "ok": False,
            "error": (
                "operation must be 'sidechain', 'attribute', 'configure' or 'restore', "
                f"not {operation!r}"
            ),
        }
    if not confirm:
        return {
            "ok": False,
            "confirm_required": True,
            "operation": operation,
            "file": str(target),
            "error": (
                "als_write edits a saved project file. A backup is made first and can be "
                "restored, but the change is real and Live must be closed. Call again with "
                "confirm=True."
            ),
        }
    if not target.exists():
        return {"ok": False, "error": f"no such file: {target}"}

    open_here, why = _set_open_in_live(target)
    if open_here is True:
        return {
            "ok": False,
            "blocked": True,
            "error": (
                f"{why}. A write now would sit on disk until Live's next save and then be "
                "lost. Live does not have to be quit: close the SET (File > New, or open "
                "another set) and retry."
            ),
        }
    if open_here is False and not allow_live_running:
        allow_live_running = True

    try:
        if operation == "restore":
            if not backup:
                return {"ok": False, "error": "restore needs backup=<path to the backup file>"}
            restored = als_writer.restore_backup(backup, target)
            return {
                "ok": True,
                "operation": "restore",
                "restored_to": str(restored),
                "from_backup": backup,
                "note": "The backup was checked for being a parseable Live set before restoring.",
            }
        if operation == "sidechain":
            if not target_track or not source_track:
                return {
                    "ok": False,
                    "error": (
                        "sidechain needs target_track (the track carrying the compressor) "
                        "and source_track"
                    ),
                }
            result = als_writer.set_sidechain_source(
                target,
                target_track=target_track,
                source_track=source_track,
                device=int(device),
                tap=tap,  # type: ignore[arg-type]
                allow_live_running=allow_live_running,
            )
        elif operation == "configure":
            if not target_track:
                return {"ok": False, "error": "configure needs target_track=<track name>"}
            try:
                wanted = [
                    (int(part.split("=", 1)[0].strip()), part.split("=", 1)[1].strip())
                    for part in value.split(";")
                    if part.strip()
                ]
            except (IndexError, ValueError):
                return {
                    "ok": False,
                    "error": (
                        "configure takes value='<index>=<name>; <index>=<name>' where the "
                        "index is the position in the plugin's own parameter list. Get that "
                        "list with lom_call(device, 'get_parameter_names')."
                    ),
                }
            result = als_writer.configure_plugin_parameters(
                target,
                target_track,
                wanted,
                device_index=int(device),
                allow_live_running=allow_live_running,
            )
        else:
            if not expression:
                return {"ok": False, "error": "attribute needs expression=<ElementTree path>"}
            result = als_writer.set_attribute(
                target,
                expression,
                value,
                attribute=attribute,
                index=index,
                create=create,
                allow_live_running=allow_live_running,
            )
    except als_writer.AlsRefused as exc:
        return {
            "ok": False,
            "refused": True,
            "written": False,
            "backup_made": False,
            "error": str(exc),
            "note": (
                "Nothing was written and no backup was made. Close the set in Live and try again."
            ),
        }
    except als_writer.AlsWriteError as exc:
        return {"ok": False, "written": False, "error": str(exc)}
    except OSError as exc:
        return {"ok": False, "written": False, "error": f"file error: {exc}"}

    return {
        "ok": result.verified,
        "operation": operation,
        "file": str(result.file),
        "backup": str(result.backup),
        "verified": result.verified,
        "verify_failures": list(result.verify_failures),
        "changes": [
            {"what": c.what, "attribute": c.attribute, "before": c.before, "after": c.after,
             "created": c.created}
            for c in result.changes
        ],
        "size_before": result.size_before,
        "size_after": result.size_after,
        "live_check": {
            "live_running": result.live_check.live_running,
            "method": result.live_check.method,
            "detail": result.live_check.detail,
        },
        "measured_notes": list(result.notes),
        "notes": [
            "verified indicates the file was re-read from disk and edits verified.",
            "Undo with operation='restore' and the backup path.",
        ],
    }


# ------------------------------------------------------------ generic escape
@mcp.tool()
def lom_get(path: str) -> dict[str, Any]:
    """Read a property value by LOM path.

    Args:
        path: Dotted LOM path starting with 'song', 'app', or 'song.view'.

    Returns:
        Dictionary containing path, value, type, and optional parameter display string.
    """
    try:
        paths.validate(path)
    except paths.PathSyntaxError as exc:
        return {"ok": False, "error": str(exc)}
    hint = _catalog_hint(path, writing=False)
    try:
        result: dict[str, Any] = {"ok": True, **_client_instance().get(path)}
    except AbletonError as exc:
        failed = _live_error(exc)
        if hint is not None:
            failed["catalog"] = hint
        return failed
    if hint is not None:
        result["catalog"] = hint
    meaning = _means_for_path(path, result.get("value"))
    if meaning is not None:
        result["means"] = meaning
    return result


@mcp.tool()
def lom_set(path: str, value: Any) -> dict[str, Any]:
    """Write a property value by LOM path and return verification read-back.

    Args:
        path: Settable dotted LOM path.
        value: Value to set, or object reference dict {'__path__': '...'} for object properties.

    Returns:
        Dictionary reporting requested, before, after, clamped, and changed status. For a
        device parameter it also carries ``display`` and ``is_quantized``.

    Note:
        Live does not accept an out-of-range value silently, however often that is given
        as the reason to read back. Measured 2026-08-30 against Live 12.4.5: it refuses
        them out loud and stores nothing. ``...mixer_device.volume.value = 1.4`` answers
        ``live_error "Invalid value. Check the parameters range with min/max"`` and the
        value stays at 0.85; ``song.tempo = 5000`` answers "Tempo out of range"; a
        panning of 5 is refused the same way.

        What the read-back really catches, measured the same day, is quantised snap:
        ``<Compressor>.parameters[10].value = 0.4`` (Model: Peak / RMS / Expand) stored 0,
        reporting ``before 1, after 0, clamped: true, changed: true, read_back:
        "clamped", display: "Peak", is_quantized: true``. A caller who asks for a value
        between two steps gets a different one and is told nowhere else.

        Two more, measured separately: a write can apply asynchronously, so the first
        read is stale (``read_back: "not_observed"``, which is not ``clamped``), and a
        method call has no read-back at all. Live also ignores an unknown property name
        silently and reports success for a write that did nothing.

        A collection (``song.tracks``, ``device.parameters``) cannot be assigned. Address
        an element and set one of its properties.

        A Live object cannot travel as a plain value, and there are two cases behind
        that. ``song.tracks[0].mixer_device.volume`` is a DeviceParameter and the wanted
        value is one level down, so write ``...volume.value``. But a property whose value
        really is an object (``song.view.selected_track``, ``song.view.selected_scene``,
        ``song.view.detail_clip``, ``browser.hotswap_target``, ``clip.groove``, and on a
        device ``input_routing_type`` and ``input_routing_channel``) is written by passing
        a reference: ``value={"__path__": "song.tracks[2]"}``, resolved through the same
        resolver and guards as a ``lom_call`` argument (protocol §5.4). Handed a plain
        JSON value the first five answer ``not_settable`` (measured 2026-08-29 against
        Live 12.4.5). The two routing properties, a compressor's sidechain source and its
        tap point, take their reference out of
        ``<device>.available_input_routing_types`` and
        ``...available_input_routing_channels``, never a name string, because RoutingType
        carries ``category`` and ``display_name`` read-only and an ``attached_object``
        pointing at the Track (measured 2026-08-30, same Live).
    """
    try:
        paths.validate(path)
    except paths.PathSyntaxError as exc:
        return {"ok": False, "error": str(exc)}
    hint = _catalog_hint(path, writing=True)
    try:
        written = _client_instance().set(path, value)
    except AbletonError as exc:
        failed = _live_error(exc)
        if hint is not None:
            failed["catalog"] = hint
        return failed
    result: dict[str, Any] = {"ok": True, **written, "proves": STORED_NOT_AUDIBLE}
    if hint is not None:
        result["catalog"] = hint
    meaning = _means_for_path(path, result.get("after"))
    if meaning is not None:
        result["after_means"] = meaning
    return result


@mcp.tool()
def lom_call(path: str, method: str, args: list[Any] | None = None) -> dict[str, Any]:
    """Invoke an allowlisted method on a Live object.

    Args:
        path: Dotted LOM path to target object.
        method: Allowlisted method name to execute.
        args: Optional list of positional arguments.

    Returns:
        Dictionary reporting method execution status.
    """
    try:
        paths.validate(path)
    except paths.PathSyntaxError as exc:
        return {"ok": False, "error": str(exc)}
    try:
        called = _client_instance().call(path, method, args or [])
    except AbletonError as exc:
        failed = _live_error(exc)
        instead = _TOOL_FOR_METHOD.get(method)
        if instead is not None:
            failed["use_instead"] = instead
        return failed
    return {"ok": True, **called, "proves": RAN_NOT_EFFECTIVE}


@mcp.tool()
def lom_batch(ops: list[BatchOpArg], atomic: bool = False) -> dict[str, Any]:
    """Execute multiple raw LOM operations (get, set, call) in a single round trip.

    Args:
        ops: List of operation dictionaries ({'op': 'get'|'set'|'call', 'path': ...}).
        atomic: When True, stops execution on the first encountered error.

    Returns:
        Dictionary containing ordered results for each operation.

    Note:
        A batch runs inside one handler call, so nothing Live recomputes between
        operations is visible to a later operation in the same batch. A read that
        follows a transport jump in the same batch still reports the position from
        before the move. Measured 2026-09-01 against Live 12.4.5: four jumps
        interleaved with four reads returned four identical pre-jump values and no
        error at all. That is the failure shape to expect here. It does not raise; it
        returns a clean set of numbers that look like a measurement of a parameter which
        never changes, and the conclusion drawn from them is wrong. Sample a moving
        transport with one call per position, never inside a batch.
    """
    if not isinstance(ops, list) or not ops:
        return {"ok": False, "error": "ops must be a non-empty list of operations"}
    try:
        result = _client_instance().batch(toolargs.batch_ops(ops), atomic=bool(atomic))
    except AbletonError as exc:
        return _live_error(exc)
    for row in result.get("results") or []:
        if isinstance(row, dict) and row.get("status") == "success":
            meaning = _means_for_path(str(row.get("path") or ""), row.get("value"))
            if meaning is not None:
                row["means"] = meaning
    return {
        "ok": int(result.get("error_count", 0)) == 0,
        **result,
        "round_trips": 1,
        "note": BATCH_NOT_ATOMIC,
    }


@mcp.tool()
def lom_describe(path: str, depth: int = 1) -> dict[str, Any]:
    """Introspect an object generically by LOM path.

    Args:
        path: Target LOM path.
        depth: Traversal depth for nested child collections (defaults to 1).

    Returns:
        Dictionary reporting object class, properties, children, and allowed methods.
    """
    try:
        paths.validate(path)
    except paths.PathSyntaxError as exc:
        return {"ok": False, "error": str(exc)}
    try:
        return {"ok": True, **_client_instance().describe(path, depth=max(1, int(depth)))}
    except AbletonError as exc:
        return _live_error(exc)


@mcp.tool()
def lom_enums(type_name: str = "") -> dict[str, Any]:
    """Query member names and integer values for Live enum types.

    Args:
        type_name: Dotted Live enum name (e.g. 'Song.Quantization'). Omit to list enum types.

    Returns:
        Dictionary of enum member names and mappings.
    """
    params: dict[str, Any] = {}
    if type_name.strip():
        params["type"] = type_name.strip()
    try:
        found = _client_instance().send("enum_names", params)
    except AbletonError as exc:
        return _live_error(exc)
    out: dict[str, Any] = {"ok": True, **found}
    if "members" in found:
        out["note"] = (
            "Read from Live itself, so these are measured names. A catalog `means` "
            "entry built from them may say so; one built from the LOM reference may not."
        )
    else:
        out["note"] = (
            "Candidates found by their signature: every public attribute is a plain int. "
            "Match a catalog row by its enum length, then ask for that type by name."
        )
    return out


# ------------------------------------------------------------------ resources
def _row_payload(spec: PathSpec, *, doc_cap: int | None = None) -> dict[str, Any]:
    """Serialize a catalog PathSpec instance into a resource dictionary."""
    out: dict[str, Any] = {}
    for field in dataclass_fields(spec):
        value = getattr(spec, field.name)
        if field.name == "doc":
            continue
        default = field.default
        if default is not MISSING and default is not None and value == default:
            continue
        if field.name in ("params", "args"):
            out[field.name] = [getattr(entry, "name", str(entry)) for entry in value]
            continue
        out[field.name] = _enum_value(value)
    out.setdefault("id", spec.id)
    out.setdefault("path", spec.path)
    out.setdefault("status", spec.status.value)

    doc = " ".join((spec.doc or "").split())
    if doc_cap is not None and len(doc) > doc_cap:
        cut = doc[:doc_cap]
        space = cut.rfind(" ")
        out["doc"] = (cut[:space] if space > 40 else cut) + " ..."
        out["doc_truncated"] = True
    else:
        out["doc"] = doc
    return out


def _catalog_defaults() -> dict[str, Any]:
    """Return dictionary of default field values omitted from individual catalog rows."""
    defaults: dict[str, Any] = {}
    for field in dataclass_fields(PathSpec):
        if field.name in ("id", "path", "doc"):
            continue
        if field.default is not MISSING and field.default is not None:
            defaults[field.name] = _enum_value(field.default)
    defaults.setdefault("access", ["get"])
    defaults.setdefault("params", [])
    defaults.setdefault("args", [])
    return defaults


@mcp.resource("ableton://catalog")
def catalog_resource() -> str:
    """Return JSON string of entire LOM path catalog with status and metadata."""
    try:
        registry = default_registry()
    except (CatalogError, OSError) as exc:
        return json.dumps({"error": f"the path catalog could not be loaded: {exc}"}, indent=2)

    specs = registry.all()
    areas: dict[str, int] = {}
    for spec in specs:
        areas[area_of(spec.id)] = areas.get(area_of(spec.id), 0) + 1
    payload = {
        "count": len(specs),
        "status_counts": registry.status_counts(),
        "status_meaning": {
            "verified": "run against a real Live and watched to work",
            "broken": "run and it did not work; the reason is in doc",
            "untested": "nobody has tried it: a hypothesis with a path attached",
        },
        "areas": dict(sorted(areas.items())),
        "defaults": _catalog_defaults(),
        "full_docs_at": "ableton://catalog/{selector}: an area, a status, or a substring",
        "rows": [_row_payload(spec, doc_cap=160) for spec in specs],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


@mcp.resource("ableton://catalog/{selector}")
def catalog_filtered(selector: str) -> str:
    """Return filtered catalog rows matching area, status, or search query."""
    try:
        registry = default_registry()
    except (CatalogError, OSError) as exc:
        return json.dumps({"error": f"the path catalog could not be loaded: {exc}"}, indent=2)

    applied: dict[str, str] = {}
    specs = registry.by_area(selector)
    if specs:
        applied = {"area": selector}
    else:
        try:
            specs = registry.by_status(selector)
            applied = {"status": selector}
        except ValueError:
            specs = registry.search(selector)
            applied = {"search": selector}

    return json.dumps(
        {
            "filter": applied,
            "count": len(specs),
            "defaults": _catalog_defaults(),
            "rows": [_row_payload(spec) for spec in specs],
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.resource("ableton://session")
def session_resource() -> str:
    """Return JSON string snapshot of running Live session state."""
    client = _client_instance()
    try:
        info = client.script_info()
        snap = introspect.snapshot(client)
    except AbletonError as exc:
        return json.dumps(_live_error(exc), indent=2)
    except introspect.IntrospectionError as exc:
        return json.dumps({"error": str(exc)}, indent=2)
    return json.dumps(
        {
            "script": info,
            "session": snap.to_dict(),
            "note": (
                "Live's in-memory state at taken_at. Stale as soon as anyone touches Live, "
                "and unrelated to the saved .als."
            ),
        },
        ensure_ascii=False,
        indent=2,
    )


_LIMITS_FALLBACK: list[dict[str, str]] = [
    {"limit": "Render / export / bounce", "kind": "LOM", "class": "hard",
     "why": "The Live Object Model has no export, bounce or render call.",
     "instead": "A human does the final mixdown."},
    {"limit": "Save the project", "kind": "LOM", "class": "hard",
     "why": "There is no save call, so the .als on disk is only as fresh as the last Ctrl+S.",
     "instead": "A human presses Ctrl+S; Channel B reads the saved state."},
    {"limit": "Hearing", "kind": "physical", "class": "hard",
     "why": "Every value can be read and proved; whether it sounds good cannot.",
     "instead": "The machine measures, the human hears."},
    {"limit": "Reorder devices: an assumed limit that turned out not to be one",
     "kind": "none", "class": "not a limit",
     "why": "Song.move_device(device, target_chain, position) exists and works. The call "
            "is on the Song rather than on the Device, which is why it reads as missing. "
            "Measured 2026-08-30 against Live 12.4.5; the row song.move_device is "
            "verified and the method is on the script's allowlist.",
     "instead": "Nothing to work around: call it through lom_call. Loading order is still "
                "worth planning: effects are appended, an instrument replaces."},
    {"limit": "Group tracks", "kind": "LOM", "class": "hard",
     "why": "group_track and is_grouped are read-only.",
     "instead": "Ctrl+G in the GUI, and re-read afterwards: grouping shifts every index."},
    {"limit": "Set a sidechain source: widely assumed to be a hard LOM limit, and it is not",
     "kind": "none", "class": "not a limit",
     "why": "The routing surface hangs off the DEVICE, not the track: input_routing_type is "
            "the source, input_routing_channel the tap point, and "
            "available_input_routing_types / _channels list what fits. Measured 2026-08-30 "
            "against Live 12.4.5: a reference write read back applied, and display_name then "
            "read the source track's name. S/C On arms it, as an ordinary parameter.",
     "instead": "Nothing to work around: lom_set the reference, live, no file edit and no "
                "restart. als_write(operation='sidechain') stays for a project Live has "
                "not got open."},
    {"limit": "Write track automation", "kind": "LOM", "class": "hard",
     "why": "clip.automation_envelope() returns None for Arrangement clips.",
     "instead": "Write a Session clip, then arrange it. Never the reverse."},
    {"limit": "Unconfigured VST parameters", "kind": "Live", "class": "gui",
     "why": "The LOM shows only what was taken into the parameter strip via Configure; "
            "before that a plug-in reports exactly one parameter, Device On. 128 slots max.",
     "instead": "GUI work, per parameter and per instance."},
    {"limit": "Units on VST2", "kind": "Live", "class": "gui",
     "why": "str_for_value() returns nothing usable for VST2 (measured).",
     "instead": "Write normalised; calibrate against the plug-in's own display once."},
    {"limit": "Transactions", "kind": "LOM", "class": "protocol",
     "why": "lom_batch runs in order and cannot roll back; atomic only stops at the first error.",
     "instead": "Nothing. Do not present a batch as atomic."},
    {"limit": "Retry a timed-out write", "kind": "protocol", "class": "protocol",
     "why": "A write that timed out may have landed; repeating it can double a note list.",
     "instead": "Verify with a read. The client refuses to guess."},
    {"limit": "Two requests at once", "kind": "protocol", "class": "protocol",
     "why": "No framing and no request ids: replies correlate by order alone (measured).",
     "instead": "lom_batch."},
    {"limit": "Change the Remote Script without restarting Live", "kind": "Live",
     "class": "protocol",
     "why": "Live loads Remote Scripts only at startup and prefers a stale __pycache__.",
     "instead": "A catalog row instead of a handler, wherever that is possible."},
]


@mcp.resource("ableton://limits")
def limits_resource() -> str:
    """Return system limits specification and documentation reference."""
    payload: dict[str, Any] = {
        "classes": {
            "LOM": "Live's Object Model does not expose it. Only Ableton can change that.",
            "Live": "Live exposes it but gates it behind the GUI. Only the user can lift it.",
            "protocol": "Our wire format or script cannot carry it. Fixable by us.",
            "physical": "Not a software problem.",
        },
        "provenance": {
            "measured": "tested against running Live or a produced artefact",
            "read from the source": "derived from code, not executed",
            "estimated": "an expectation, explicitly not a promise",
            "unverified": "nobody has tried it",
        },
        "always_true": [
            STORED_NOT_AUDIBLE,
            BATCH_NOT_ATOMIC,
            SESSION_ONLY_NOTE,
            CONFIGURE_NOTE,
            GROUP_TRACK_NOTE,
        ],
    }

    try:
        text = LIMITS_DOC.read_text(encoding="utf-8")
    except OSError:
        payload["source"] = "built-in summary (docs/limits.md is not in this installation)"
        payload["limits"] = _LIMITS_FALLBACK
        return json.dumps(payload, ensure_ascii=False, indent=2)

    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        section = re.match(r"^##\s+(\d+)\.\s+(.*)$", line)
        if section is not None:
            current = {"number": int(section.group(1)), "title": section.group(2).strip(),
                       "entries": []}
            sections.append(current)
            continue
        # The em-dash here is data, not prose: it is the separator docs/limits.md puts
        # between an entry title and its class. Replacing it stops the parse.
        entry = re.match(r"^###\s+(.*?)(?:\s+—\s+class:\s+(\w+))?\s*$", line)
        if entry is not None and current is not None:
            current["entries"].append(
                {"title": entry.group(1).strip(), "class": entry.group(2) or "unclassified"}
            )
    payload["source"] = str(LIMITS_DOC)
    payload["sections"] = sections
    payload["limits"] = _LIMITS_FALLBACK
    payload["text"] = text
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main() -> None:
    """Serve Ableton Maestro MCP server over stdio transport."""
    logger.info(
        "ableton-maestro serving on stdio; Live bridge at %s:%d", DEFAULT_HOST, _port()
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
