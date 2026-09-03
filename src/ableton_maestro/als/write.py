"""Offline project file modifier for saved Ableton Live (.als) files.

Modifies .als files when Live does not have them open (e.g. sidechain routing,
plugin parameter configuration, XML attribute editing).

Safety guarantees:
- Timestamped backup created and verified before every write; undoable via restore_backup.
- Refusal when the target project is open in a running Live process.
- Atomic replacement using temporary files in the same directory via os.replace.
- Post-write verification: re-reads the file from disk to verify XML integrity and edits.

Sidechain routing notes:
- Routing target XML structure: SideChain/RoutedInput/Routable/Target
  (with fallback to SideChain/RoutedInput/Target).
- Source references use the track's XML Id attribute, not track position.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

__all__ = [
    "AlsRefused",
    "AlsWriteError",
    "Change",
    "LiveCheck",
    "PluginSlots",
    "SidechainSlot",
    "WriteResult",
    "configure_plugin_parameters",
    "list_plugin_slots",
    "list_sidechain_slots",
    "restore_backup",
    "set_attribute",
    "set_sidechain_source",
]

TRACK_TAGS = ("MidiTrack", "AudioTrack", "GroupTrack", "ReturnTrack", "MasterTrack", "MainTrack")
"""XML tags representing Ableton project tracks (includes Live 12 MainTrack)."""

NOT_ROUTED = "AudioIn/None"
"""Default target string for unrouted sidechain inputs.

Live hangs a sidechain input on every plugin, usually with ``OnOff/Manual = true``
and this target, so ``OnOff`` alone is never evidence of a wiring.

Note:
    Measured on one project: 62 sidechain-capable devices, 4 actually wired.
"""

TAPS: dict[str, tuple[str, str]] = {
    "post": ("PostFxOut", "Post FX"),
    "pre": ("PreFxOut", "Pre FX"),
}
"""Mapping from tap mode to target XML suffix and UI display label."""

Tap = Literal["post", "pre"]
"""Audio tap point for sidechain routing ('post' or 'pre')."""

# Live's exact prologue, measured on Live 12.4.3 sets. Written verbatim rather than
# through ElementTree, which quotes the declaration with apostrophes.
_XML_DECLARATION = b'<?xml version="1.0" encoding="UTF-8"?>\n'

# gzip container fields, measured from Live's own output: MTIME 0, FLG 0 (no embedded
# filename), XFL 0, OS 0x0a. Level 6 is zlib's default and lands 0.09 % to 0.29 %
# smaller than Live's own compressor across six sets. ElementTree drops XML comments
# and processing instructions, but real Live sets carry neither comments nor DOCTYPE.
_GZIP_LEVEL = 6
_GZIP_OS_BYTE = 0x0A

# A set saved seconds ago was almost certainly saved by Live, just now. Chosen, not
# measured: there is no measurement behind the number, only caution.
_RECENT_WRITE_SECONDS = 60.0

# Substring searched for in the process table, case-folded.
_LIVE_PROCESS_MARKER = "ableton live"


class AlsWriteError(Exception):
    """Base exception raised for project file modification errors."""


class AlsRefused(AlsWriteError):
    """Exception raised when a safety check prevents modifying the file."""


# ------------------------------------------------------------------- results
@dataclass(frozen=True)
class LiveCheck:
    """Process and filesystem locking status for target project file.

    Attributes:
        live_running: Whether an Ableton Live process was detected running, or None if unknown.
        method: Detection method used (process enumeration command).
        detail: Descriptive summary of detection check findings.
        seconds_since_write: Elapsed seconds since file last modified timestamp.
        recently_written: True if file was modified within recent write threshold.
        exclusive_open_failed: Error message if exclusive read-write open failed, else None.
    """

    live_running: bool | None
    method: str
    detail: str
    seconds_since_write: float
    recently_written: bool
    exclusive_open_failed: str | None


@dataclass(frozen=True)
class Change:
    """Record of modified XML attribute in project file.

    Attributes:
        what: Descriptive identifier of modified element or target.
        attribute: Name of XML attribute modified.
        before: Original attribute value before write, or None if newly created.
        after: Updated attribute value written to file.
        created: True if attribute was newly inserted into element.
    """

    what: str
    attribute: str
    before: str | None
    after: str
    created: bool


@dataclass(frozen=True)
class WriteResult:
    """Result and verification summary of a project file write operation.

    Attributes:
        file: Path to modified project file.
        backup: Path to verified timestamped backup copy.
        changes: Sequence of individual attribute modifications applied.
        verified: True if post-write verification re-read matched all edits.
        verify_failures: Descriptions of any attributes failing verification.
        live_check: Diagnostics from pre-write Live process and file lock inspection.
        size_before: File size in bytes prior to modification.
        size_after: File size in bytes after modification.
        notes: Additional contextual or diagnostic notes.

    Note:
        ``verified=False`` means the re-read did not confirm what was asked for. The
        file still parses, otherwise the backup would have been restored and an
        exception raised, but the change should not be trusted.

        ``notes`` carries measured context worth showing a human: facts from the corpus
        and from the file itself, never invented advice.
    """

    file: Path
    backup: Path
    changes: tuple[Change, ...]
    verified: bool
    verify_failures: tuple[str, ...]
    live_check: LiveCheck
    size_before: int
    size_after: int
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PluginSlots:
    """Parameter slot allocation for a plugin instance.

    Attributes:
        track: Name of track hosting the plugin.
        device_index: Zero-based index of plugin on track.
        plugin: Name of plugin device.
        slots: Total allocated parameter slot capacity.
        configured: Tuple of (slot_index, parameter_id, parameter_name) mappings.
        free: Number of remaining unconfigured parameter slots.

    Note:
        Live allocates the array up front and leaves the unused entries in the file,
        so ``slots`` is a constant per instance (128, measured 2026-09-01) and ``free``
        is what a Configure write has to fit into.
    """

    track: str
    device_index: int
    plugin: str
    slots: int
    configured: tuple[tuple[int, str, str], ...]
    free: int


@dataclass(frozen=True)
class SidechainSlot:
    """Sidechain routing state for a device.

    Attributes:
        track: Name of track hosting the device.
        track_id: XML ID of track hosting the device.
        track_tag: Element tag of track.
        device_index: Zero-based index of sidechain-capable device on track.
        device_tag: Element tag of device.
        target: Configured routing target string.
        source_display: Cached display string for routing source track.
        enabled: Enabled state of sidechain toggle.
        routed: True if sidechain target is active and routed to a valid source.

    Note:
        ``routed`` is the only trustworthy field for "is this actually wired".
        ``source_display`` keeps the last known track name even after the target has
        fallen back to ``AudioIn/None``. Measured on one project: 10 of 14 apparent
        sidechains were display strings over a dead target.
    """

    track: str
    track_id: str | None
    track_tag: str
    device_index: int
    device_tag: str
    target: str | None
    source_display: str | None
    enabled: bool
    routed: bool


@dataclass(frozen=True)
class _Edit:
    """Planned XML attribute modification and locator function."""

    what: str
    locate: Callable[[ET.Element, bool], ET.Element | None]
    attribute: str
    value: str
    create: bool = False


# --------------------------------------------------------------- gzip + XML
def _read_xml(path: Path) -> ET.Element:
    """Decompress and parse gzip-compressed XML from project file.

    Args:
        path: Filesystem path to project file.

    Returns:
        Parsed xml.etree.ElementTree.Element root.

    Raises:
        AlsWriteError: If file cannot be read or contains invalid XML.
    """
    try:
        with gzip.open(path, "rb") as handle:
            raw = handle.read()
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        raise AlsWriteError(f"{path} is not a readable gzip file: {exc}") from exc
    try:
        return ET.fromstring(raw)
    except ET.ParseError as exc:
        raise AlsWriteError(f"{path} is gzip but not parseable XML: {exc}") from exc


def _serialise(root: ET.Element) -> bytes:
    """Serialize XML tree to bytes using UTF-8 and CRLF line endings."""
    body = ET.tostring(root, encoding="utf-8", xml_declaration=False)
    return (_XML_DECLARATION + body + b"\n").replace(b"\n", b"\r\n")


def _gzip_bytes(payload: bytes) -> bytes:
    """Compress bytes payload into gzip format matching Live container conventions.

    The header is emitted by hand rather than through :mod:`gzip` so that MTIME stays
    zero and the OS byte matches Live's own output. See the container constants above
    for the measurement behind those fields.
    """
    compressor = zlib.compressobj(_GZIP_LEVEL, zlib.DEFLATED, -zlib.MAX_WBITS)
    body = compressor.compress(payload) + compressor.flush()
    header = b"\x1f\x8b\x08\x00" + b"\x00\x00\x00\x00" + b"\x00" + bytes([_GZIP_OS_BYTE])
    trailer = struct.pack("<II", zlib.crc32(payload) & 0xFFFFFFFF, len(payload) & 0xFFFFFFFF)
    return header + body + trailer


def _write_atomic(path: Path, data: bytes) -> None:
    """Write data to temporary sibling file and atomically replace destination path."""
    tmp = path.with_name(f".{path.name}.maestro-{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _format_value(value: str | bool | float) -> str:
    """Format Python primitive as attribute string according to Live XML conventions."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.10g}"
    text = str(value)
    if any(ord(char) < 32 for char in text):
        raise AlsWriteError("attribute values must not contain control characters")
    return text


def _append_child(parent: ET.Element, tag: str) -> ET.Element:
    """Append new child element to parent while preserving XML indentation."""
    existing = list(parent)
    child = ET.SubElement(parent, tag)
    if existing:
        child.tail = existing[-1].tail
        existing[-1].tail = parent.text
    else:
        closing = parent.tail or "\n"
        parent.text = closing + "\t"
        child.tail = closing
    return child


# --------------------------------------------------------------- safety net
def _inspect_environment(path: Path) -> LiveCheck:
    """Inspect process list and file access to evaluate whether project is in use.

    The result is weak evidence by design: there is no reliable lock to detect.

    Note:
        Measured 2026-08-29 against Live 12.4.5 with a set loaded: ``CreateFileW``
        with ``dwShareMode = 0`` (an exclusive open, which fails if any other process
        holds the file) succeeds on the very set Live has open. Live reads the .als
        into memory and closes the handle. Any claim that the file is locked while
        Live has it open is therefore wrong on Windows.
    """
    running: bool | None
    windows = sys.platform == "win32"
    command = ["tasklist", "/FO", "CSV", "/NH"] if windows else ["ps", "-A", "-o", "comm="]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=5,
            check=False,
            creationflags=creation_flags,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        running, method, detail = None, "unavailable", f"could not read the process table: {exc}"
    else:
        listing = completed.stdout.casefold()
        running = _LIVE_PROCESS_MARKER in listing
        method = " ".join(command)
        detail = "a process named 'Ableton Live' is running" if running else "no Live process found"

    stat = path.stat()
    age = max(0.0, time.time() - stat.st_mtime)

    exclusive_failure: str | None = None
    try:
        with open(path, "r+b"):
            pass
    except OSError as exc:
        exclusive_failure = str(exc)

    return LiveCheck(
        live_running=running,
        method=method,
        detail=detail,
        seconds_since_write=age,
        recently_written=age < _RECENT_WRITE_SECONDS,
        exclusive_open_failed=exclusive_failure,
    )


def _guard(path: Path, *, allow_live_running: bool) -> LiveCheck:
    """Validate safety checks before modifying project file on disk.

    Args:
        path: Path to target project file.
        allow_live_running: If True, bypass refusal on active Live processes.

    Returns:
        LiveCheck inspection record.

    Raises:
        AlsRefused: If safety checks determine the file may currently be in use.
    """
    check = _inspect_environment(path)

    if check.exclusive_open_failed is not None:
        raise AlsRefused(
            f"{path} cannot be opened for writing ({check.exclusive_open_failed}). "
            "Another process is holding it. This refusal has no override."
        )
    if check.live_running and not allow_live_running:
        raise AlsRefused(
            f"Ableton Live is running ({check.method}). A write to a set Live has open is "
            "lost without a word the next time Live saves. Close Live, or pass "
            "allow_live_running=True if you are certain Live does not have THIS set open. "
            "the check cannot tell which set is loaded."
        )
    if check.recently_written and not allow_live_running:
        raise AlsRefused(
            f"{path.name} was written {check.seconds_since_write:.0f} s ago, most likely by "
            "Live saving it just now. Wait, close Live, or pass allow_live_running=True."
        )
    return check


def _backup(path: Path, backup_dir: Path | None) -> Path:
    """Create timestamped copy of project file and verify SHA-256 digest match."""
    stamp = datetime.now(tz=UTC).astimezone().strftime("%Y-%m-%d %H%M%S")
    target_dir = backup_dir if backup_dir is not None else path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    destination = target_dir / f"{path.stem} [maestro {stamp}].als"
    counter = 1
    while destination.exists():
        destination = target_dir / f"{path.stem} [maestro {stamp}-{counter}].als"
        counter += 1

    shutil.copy2(path, destination)
    if _digest(destination) != _digest(path):
        destination.unlink(missing_ok=True)
        raise AlsWriteError(
            f"the backup of {path.name} did not match the original. Nothing was written."
        )
    return destination


def _digest(path: Path) -> str:
    """Compute SHA-256 checksum digest of file at path."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def restore_backup(backup: str | Path, target: str | Path) -> Path:
    """Restore backup copy over destination path after verifying XML integrity.

    Args:
        backup: Path to verified backup file.
        target: Destination path to restore over.

    Returns:
        Resolved Path of restored destination file.

    Raises:
        AlsWriteError: If backup file does not exist or fails XML parsing.
    """
    source = Path(backup).expanduser()
    destination = Path(target).expanduser()
    if not source.is_file():
        raise AlsWriteError(f"backup not found: {source}")
    _read_xml(source)
    _write_atomic(destination, source.read_bytes())
    return destination


# ------------------------------------------------------------ the edit loop
def _apply(root: ET.Element, edit: _Edit) -> Change:
    """Apply attribute edit to element in parsed XML tree."""
    element = edit.locate(root, edit.create)
    if element is None:
        raise AlsWriteError(f"{edit.what}: no such element in this set")
    before = element.get(edit.attribute)
    if before is None and not edit.create:
        raise AlsWriteError(
            f"{edit.what}: element has no {edit.attribute!r} attribute; pass create=True "
            "if you really mean to add one"
        )
    element.set(edit.attribute, edit.value)
    return Change(
        what=edit.what,
        attribute=edit.attribute,
        before=before,
        after=edit.value,
        created=before is None,
    )


def _verify(root: ET.Element, edits: Iterable[_Edit]) -> tuple[bool, tuple[str, ...]]:
    """Verify all planned edits against freshly re-parsed XML tree from disk."""
    failures: list[str] = []
    for edit in edits:
        try:
            element = edit.locate(root, False)
        except AlsWriteError as exc:
            failures.append(f"{edit.what}: could not be located after the write ({exc})")
            continue
        if element is None:
            failures.append(f"{edit.what}: element is gone after the write")
            continue
        actual = element.get(edit.attribute)
        if actual != edit.value:
            failures.append(f"{edit.what}: expected {edit.value!r} on re-read, found {actual!r}")
    return not failures, tuple(failures)


def _run(
    als_path: str | Path,
    plan: Callable[[ET.Element], tuple[list[_Edit], list[str]]],
    *,
    allow_live_running: bool,
    backup_dir: str | Path | None,
) -> WriteResult:
    """Execute transactional write pipeline: plan, guard, backup, write, and verify."""
    path = Path(als_path).expanduser().resolve()
    if not path.is_file():
        raise AlsWriteError(f"no such file: {path}")

    root = _read_xml(path)
    edits, notes = plan(root)
    if not edits:
        raise AlsWriteError("nothing to change")

    check = _guard(path, allow_live_running=allow_live_running)
    backup = _backup(path, Path(backup_dir).expanduser() if backup_dir is not None else None)
    size_before = path.stat().st_size

    changes = tuple(_apply(root, edit) for edit in edits)
    _write_atomic(path, _gzip_bytes(_serialise(root)))

    try:
        written = _read_xml(path)
    except AlsWriteError as exc:
        try:
            restore_backup(backup, path)
        except AlsWriteError as restore_failure:
            raise AlsWriteError(
                f"what was written to {path} is not a valid set ({exc}), and putting the backup "
                f"back failed too ({restore_failure}). The intact copy is at {backup}; copy it "
                "over the set by hand before doing anything else."
            ) from exc
        raise AlsWriteError(
            f"what was written to {path} did not read back as a valid set ({exc}). "
            f"The backup {backup} has been restored; the set is as it was."
        ) from exc
    verified, failures = _verify(written, edits)

    return WriteResult(
        file=path,
        backup=backup,
        changes=changes,
        verified=verified,
        verify_failures=failures,
        live_check=check,
        size_before=size_before,
        size_after=path.stat().st_size,
        notes=tuple(notes),
    )


# ------------------------------------------------------------ generic access
def set_attribute(
    als_path: str | Path,
    expression: str,
    value: str | bool | float,
    *,
    attribute: str = "Value",
    index: int | None = None,
    create: bool = False,
    allow_live_running: bool = False,
    backup_dir: str | Path | None = None,
) -> WriteResult:
    """Set attribute on matching XML element in an offline project file.

    Args:
        als_path: Path to target Ableton project file.
        expression: XPath search expression matching target element.
        value: New attribute value to write.
        attribute: Attribute name to modify (defaults to 'Value').
        index: Index of matching element if expression yields multiple matches.
        create: Whether to create attribute if not already present.
        allow_live_running: Whether to bypass active Live process refusal.
        backup_dir: Optional custom directory to store backup copy.

    Returns:
        WriteResult containing change summary and verification status.

    Raises:
        AlsRefused: If Live is detected running or file is locked.
        AlsWriteError: If expression is ambiguous, matches nothing, or write fails.

    Note:
        An expression matching more than one element is refused unless ``index`` picks
        one out of the match list, in document order. This guard stands between a typo
        like ``.//Manual`` and several thousand silently rewritten parameters: measured
        on one 5.3 MB set, ``.//Manual`` matches 4496 elements.

        ``create=True`` allows writing an attribute that is not there yet. The element
        itself is never created; this function edits a set, it does not build one.

        Values in the file are stored in their real unit, not normalised as Channel A
        reports them. Auto Filter ``Frequency`` is in Hertz and compressor ``Threshold``
        is a linear factor where 1 is 0 dB (measured). Read the value that is there
        before overwriting it, and do not carry a normalised number across from
        Channel A.

    >>> # threshold of the second Compressor2 in the set
    >>> set_attribute(als, ".//Compressor2/Threshold/Manual", 0.3, index=1)  # doctest: +SKIP
    """
    text = _format_value(value)

    def locate(root: ET.Element, _create: bool) -> ET.Element | None:
        """Resolve expression to exactly one element, or refuse on ambiguity."""
        matches = root.findall(expression)
        if not matches:
            return None
        if index is None:
            if len(matches) > 1:
                raise AlsWriteError(
                    f"{expression!r} matches {len(matches)} elements. Pass index=0.."
                    f"{len(matches) - 1} to choose one, or write a narrower expression."
                )
            return matches[0]
        if not 0 <= index < len(matches):
            raise AlsWriteError(
                f"{expression!r} matches {len(matches)} elements; index {index} is out of range"
            )
        return matches[index]

    def plan(root: ET.Element) -> tuple[list[_Edit], list[str]]:
        """Validate against loaded tree and construct single planned edit."""
        what = expression if index is None else f"{expression}[{index}]"
        if locate(root, False) is None:
            raise AlsWriteError(f"{expression!r} matches nothing in this set")
        edit = _Edit(what=what, locate=locate, attribute=attribute, value=text, create=create)
        return [edit], []

    return _run(als_path, plan, allow_live_running=allow_live_running, backup_dir=backup_dir)


# ------------------------------------------------------------------- tracks
def _track_name(track: ET.Element) -> str:
    """Return effective track display name, falling back to tag."""
    for candidate in ("./Name/EffectiveName", "./Name/UserName"):
        element = track.find(candidate)
        if element is not None and (element.get("Value") or "").strip():
            return element.get("Value") or track.tag
    return track.tag


def _tracks(root: ET.Element) -> list[ET.Element]:
    """Return all track elements in project in document order."""
    return [element for element in root.iter() if element.tag in TRACK_TAGS]


def _find_track(root: ET.Element, name: str, role: str) -> ET.Element:
    """Find track element matching name exactly, refusing if ambiguous or absent."""
    matches = [track for track in _tracks(root) if _track_name(track) == name]
    if not matches:
        known = ", ".join(sorted({_track_name(track) for track in _tracks(root)}))
        raise AlsWriteError(f"no {role} track named {name!r}. This set has: {known}")
    if len(matches) > 1:
        ids = ", ".join(str(track.get("Id")) for track in matches)
        raise AlsWriteError(
            f"{len(matches)} tracks are named {name!r} (Ids {ids}). Rename one in Live, or "
            "address the device with set_attribute."
        )
    return matches[0]


def _sidechain_devices(track: ET.Element) -> list[ET.Element]:
    """Return all devices on track with a SideChain child node, in document order.

    Descends the whole track, so devices nested inside racks are included. Measured on
    a project where all wired compressors sat one to three rack levels deep on group
    tracks: a reader that walked only the top device chain reported zero sidechains for
    a project holding 62 sidechain-capable devices.

    Live hangs a sidechain input on every plugin, so this list is long and most of it is
    unwired. It is a list of slots, not of findings.
    """
    return [element for element in track.iter() if element.find("./SideChain") is not None]


def _routable(device: ET.Element) -> ET.Element | None:
    """Locate routing container element carrying sidechain Target attribute."""
    sidechain = device.find("./SideChain")
    if sidechain is None:
        return None
    routed_input = sidechain.find("./RoutedInput")
    if routed_input is None:
        return None
    routable = routed_input.find("./Routable")
    return routable if routable is not None else routed_input


def _is_muted(track: ET.Element) -> bool | None:
    """Check whether track speaker toggle is muted (Mixer/Speaker/Manual == false)."""
    speaker = track.find("./DeviceChain/Mixer/Speaker/Manual")
    if speaker is None:
        return None
    return (speaker.get("Value") or "").strip().casefold() == "false"


def list_sidechain_slots(als_path: str | Path) -> tuple[SidechainSlot, ...]:
    """List all sidechain-capable devices across tracks in a saved project file.

    Args:
        als_path: Path to target Ableton project file.

    Returns:
        Tuple of SidechainSlot descriptors for all sidechain inputs.
    """
    root = _read_xml(Path(als_path).expanduser())
    slots: list[SidechainSlot] = []
    for track in _tracks(root):
        for position, device in enumerate(_sidechain_devices(track)):
            node = _routable(device)
            target = _value(node, "Target") if node is not None else None
            slots.append(
                SidechainSlot(
                    track=_track_name(track),
                    track_id=track.get("Id"),
                    track_tag=track.tag,
                    device_index=position,
                    device_tag=device.tag,
                    target=target,
                    source_display=_value(node, "UpperDisplayString") if node is not None else None,
                    enabled=(_value(device, "SideChain/OnOff/Manual") or "").casefold() == "true",
                    routed=bool(target) and target != NOT_ROUTED,
                )
            )
    return tuple(slots)


def _value(parent: ET.Element, path: str) -> str | None:
    """Return Value attribute of child element at relative path, or None."""
    element = parent.find(f"./{path}")
    return None if element is None else element.get("Value")


# ------------------------------------------------------- plugin Configure
# What Live writes into a parameter slot nobody has configured. Both markers matter:
# ParameterId -1 says "no plugin parameter behind this slot" and VisualIndex 2**30 - 1
# says "not in the strip". Measured 2026-09-01 on a set holding four plugin instances:
# 512 slots in the file, 128 per instance, and every unconfigured one carried exactly
# these two values.
_SLOT_UNSET_ID = "-1"
_SLOT_UNSET_VISUAL = "1073741823"


def _plugin_devices(track: ET.Element) -> list[ET.Element]:
    """Return all PluginDevice elements on track in chain order."""
    return [node for node in track.iter("PluginDevice")]


def _plugin_name(device: ET.Element) -> str:
    """Extract plugin name from Vst3PluginInfo or VstPluginInfo metadata."""
    for tag in ("Vst3PluginInfo", "VstPluginInfo"):
        info = device.find(f".//{tag}")
        if info is None:
            continue
        for name_tag in ("Name", "PlugName"):
            found = info.find(name_tag)
            if found is not None and found.get("Value"):
                return str(found.get("Value"))
    return "?"


def _parameter_slots(device: ET.Element) -> list[ET.Element]:
    """Return list of PluginFloatParameter slot elements in device parameter strip."""
    plist = device.find(".//ParameterList")
    if plist is None:
        raise AlsWriteError(
            f"the plugin {_plugin_name(device)!r} has no ParameterList in this set. "
            "Nothing can be configured into a device that has no slot array."
        )
    return plist.findall("PluginFloatParameter")


def list_plugin_slots(als_path: str | Path) -> tuple[PluginSlots, ...]:
    """Inspect parameter slot allocations for plugin devices in saved project file.

    Args:
        als_path: Path to target Ableton project file.

    Returns:
        Tuple of PluginSlots objects describing configured and free slots per plugin.
    """
    root = _read_xml(Path(als_path).expanduser().resolve())
    out: list[PluginSlots] = []
    for track in _tracks(root):
        for index, device in enumerate(_plugin_devices(track)):
            slots = _parameter_slots(device)
            taken = tuple(
                (position, str(slot.find("ParameterId").get("Value")), str(name))
                for position, slot in enumerate(slots)
                if (name := (slot.find("ParameterName").get("Value") or "")) != ""
            )
            out.append(
                PluginSlots(
                    track=_track_name(track),
                    device_index=index,
                    plugin=_plugin_name(device),
                    slots=len(slots),
                    configured=taken,
                    free=len(slots) - len(taken),
                )
            )
    return tuple(out)


def configure_plugin_parameters(
    als_path: str | Path,
    track_name: str,
    parameters: Sequence[tuple[int, str]],
    *,
    device_index: int = 0,
    allow_live_running: bool = False,
    backup_dir: str | Path | None = None,
) -> WriteResult:
    """Assign plugin parameter IDs to slots in the device parameter strip.

    Args:
        als_path: Path to target Ableton project file.
        track_name: Name of track hosting the target plugin device.
        parameters: Sequence of (plugin_parameter_index, display_name) pairs.
        device_index: Zero-based index of plugin device on track.
        allow_live_running: Whether to bypass active Live process refusal.
        backup_dir: Optional custom directory to store backup copy.

    Returns:
        WriteResult containing change records and verification outcome.

    Raises:
        AlsRefused: If Live is detected running or file is locked.
        AlsWriteError: If parameters cannot be configured or exceed slot capacity.

    Note:
        Nothing is inserted. Live allocates the slot array up front (128 slots per
        instance, measured 2026-09-01) and an unconfigured slot is a complete element
        already carrying its own automation and modulation targets. This fills three
        fields of a slot that is already there, which is why it is a small edit rather
        than a structural one.

        Three fields, and deliberately not more: ``ParameterName``, ``ParameterId`` and
        ``VisualIndex``. The slot's ``Manual`` value and its ``MidiControllerRange`` are
        left exactly as Live wrote them.
    """
    wanted = [(int(index), str(name)) for index, name in parameters]
    if not wanted:
        raise AlsWriteError("no parameters given; nothing to configure")
    for index, name in wanted:
        if index < 0:
            raise AlsWriteError(
                f"plugin index {index} is negative; -1 is the marker for an EMPTY slot and "
                "cannot be configured onto one"
            )
        if not name.strip():
            raise AlsWriteError(
                f"the parameter at index {index} has no name. A slot with an id and no name "
                "is the one state Live does not produce, and its meaning is unknown."
            )

    def plan(root: ET.Element) -> tuple[list[_Edit], list[str]]:
        track = _find_track(root, track_name, "target")
        devices = _plugin_devices(track)
        if not devices:
            raise AlsWriteError(f"no plugin device on {track_name!r}")
        if not 0 <= device_index < len(devices):
            raise AlsWriteError(
                f"device {device_index} does not exist; {track_name!r} carries "
                f"{len(devices)} plugin device(s)"
            )
        device = devices[device_index]
        plugin = _plugin_name(device)
        slots = _parameter_slots(device)

        free: list[int] = []
        used_names: set[str] = set()
        used_ids: set[str] = set()
        highest_visual = -1
        for position, slot in enumerate(slots):
            name = slot.find("ParameterName").get("Value") or ""
            if name == "":
                free.append(position)
                continue
            used_names.add(name)
            used_ids.add(str(slot.find("ParameterId").get("Value")))
            visual = slot.find("VisualIndex")
            if visual is not None and visual.get("Value") != _SLOT_UNSET_VISUAL:
                highest_visual = max(highest_visual, int(visual.get("Value")))

        if len(wanted) > len(free):
            raise AlsWriteError(
                f"{plugin!r} has {len(free)} free slot(s) of {len(slots)} and "
                f"{len(wanted)} parameter(s) were asked for. Live's array is fixed at "
                "128 per instance, so a plugin with more parameters than that is a choice, "
                "not an oversight."
            )
        clashes = [name for _index, name in wanted if name in used_names]
        if clashes:
            raise AlsWriteError(
                f"{plugin!r} already exposes {clashes}. Configuring the same parameter twice "
                "gives two strip entries for one control; remove it first if that is meant."
            )
        repeats = [index for index, _name in wanted if str(index) in used_ids]
        if repeats:
            raise AlsWriteError(
                f"{plugin!r} already has plugin index {repeats} in its strip under another "
                "name. One control, one slot."
            )

        edits: list[_Edit] = []
        for offset, (index, name) in enumerate(wanted):
            position = free[offset]
            visual = highest_visual + 1 + offset

            def locator(
                tag: str, at: int = position
            ) -> Callable[[ET.Element, bool], ET.Element | None]:
                def locate(tree: ET.Element, _create: bool) -> ET.Element | None:
                    found = _plugin_devices(_find_track(tree, track_name, "target"))
                    array = _parameter_slots(found[device_index])
                    return array[at].find(tag) if at < len(array) else None

                return locate

            edits.append(
                _Edit(f"{plugin} slot {position} name", locator("ParameterName"), "Value", name)
            )
            edits.append(
                _Edit(f"{plugin} slot {position} id", locator("ParameterId"), "Value", str(index))
            )
            edits.append(
                _Edit(
                    f"{plugin} slot {position} strip position",
                    locator("VisualIndex"),
                    "Value",
                    str(visual),
                )
            )

        notes = [
            (
                f"{plugin!r}: {len(wanted)} parameter(s) written into slot(s) "
                f"{[free[i] for i in range(len(wanted))]} of {len(slots)}; "
                f"{len(free) - len(wanted)} free afterwards."
            ),
            (
                "Manual and MidiControllerRange were left as Live wrote them. Whether "
                "Live adopts the plugin's current value or the placeholder when it loads "
                "the slot is NOT established - read device.parameters back through Live."
            ),
            "Live must be restarted, or the set reopened, before any of this is visible.",
        ]
        return edits, notes

    return _run(
        als_path,
        plan,
        allow_live_running=allow_live_running,
        backup_dir=backup_dir,
    )


# --------------------------------------------------------------- sidechain
def set_sidechain_source(
    als_path: str | Path,
    *,
    target_track: str,
    source_track: str,
    device: int = 0,
    tap: Tap = "post",
    allow_live_running: bool = False,
    backup_dir: str | Path | None = None,
) -> WriteResult:
    """Configure sidechain routing between tracks in an offline project file.

    Args:
        als_path: Path to target Ableton project file.
        target_track: Track hosting the receiving device (e.g. compressor).
        source_track: Track providing the sidechain trigger audio.
        device: Zero-based index of sidechain-capable device on target track.
        tap: Audio tap point ('post' for Post FX or 'pre' for Pre FX).
        allow_live_running: Whether to bypass active Live process refusal.
        backup_dir: Optional custom directory to store backup copy.

    Returns:
        WriteResult detailing applied attribute changes and verification status.

    Raises:
        AlsRefused: If Live is detected running or file is locked.
        AlsWriteError: If track/device cannot be found or routing is invalid.

    Note:
        The input gain at ``RoutedInput/Volume/Manual`` is deliberately left alone:
        99 % of 794 corpus occurrences sit at 1, and the strength of a pump comes from
        threshold and ratio on the compressor, not from the sidechain input.

        This is not a write the LOM cannot do, however often it is described that way.
        Measured 2026-08-30 against Live 12.4.5: a device carries ``input_routing_type``
        and ``input_routing_channel``, and both are writable over Channel A. For a set
        Live has open, do it there: it needs no backup, no closed Live and no restart.
        This function is for the set Live does not have open, where there is no LOM to
        ask.

        What this does not do, because none of it lives in the file: load the compressor
        (Channel A can, and must, do that first), give the trigger track an instrument,
        or make anything audible. A trigger track with no instrument is silence, and
        silence triggers no compressor (measured). A freshly loaded Live compressor also
        sits at threshold 0 dB and will never engage (measured); set that over Channel A.
    """
    if tap not in TAPS:
        raise AlsWriteError(f"tap must be one of {sorted(TAPS)}, not {tap!r}")
    suffix, display = TAPS[tap]

    def routable_field(tag: str) -> Callable[[ET.Element, bool], ET.Element | None]:
        """Create locator function for tag under device Routable container."""

        def locate(root: ET.Element, create: bool) -> ET.Element | None:
            node = _routable(_device_element(root, target_track, device))
            if node is None:
                return None
            return _descend(node, (tag,), create=create)

        return locate

    def onoff(root: ET.Element, create: bool) -> ET.Element | None:
        """Locate SideChain/OnOff/Manual element on device."""
        sidechain = _device_element(root, target_track, device).find("./SideChain")
        if sidechain is None:
            return None
        return _descend(sidechain, ("OnOff", "Manual"), create=create)

    def plan(root: ET.Element) -> tuple[list[_Edit], list[str]]:
        target = _find_track(root, target_track, "target")
        source = _find_track(root, source_track, "source")
        if source is target:
            raise AlsWriteError(
                "source and target are the same track, which is feedback, not ducking"
            )
        source_id = source.get("Id")
        if not source_id:
            raise AlsWriteError(
                f"{source_track!r} ({source.tag}) has no Id attribute and cannot be a sidechain "
                "source. Live's main track carries none."
            )
        node = _routable(_device_element(root, target_track, device))
        if node is None:
            raise AlsWriteError(
                f"the sidechain of device {device} on {target_track!r} has no RoutedInput. "
                "unexpected layout, refusing to guess"
            )

        prefix = f"{target_track} device {device}"
        edits = [
            _Edit(f"{prefix}: SideChain/OnOff/Manual", onoff, "Value", "true", create=True),
            _Edit(
                f"{prefix}: Target",
                routable_field("Target"),
                "Value",
                f"AudioIn/Track.{source_id}/{suffix}",
                create=True,
            ),
            _Edit(
                f"{prefix}: UpperDisplayString",
                routable_field("UpperDisplayString"),
                "Value",
                source_track,
                create=True,
            ),
            _Edit(
                f"{prefix}: LowerDisplayString",
                routable_field("LowerDisplayString"),
                "Value",
                display,
                create=True,
            ),
        ]
        notes = _sidechain_notes(previous=_value(node, "Target"), source=source, tap=tap)
        return edits, notes

    return _run(als_path, plan, allow_live_running=allow_live_running, backup_dir=backup_dir)


def _device_element(root: ET.Element, track_name: str, device_index: int) -> ET.Element:
    """Locate device_index-th sidechain-capable device on named track."""
    track = _find_track(root, track_name, "target")
    devices = _sidechain_devices(track)
    if not devices:
        raise AlsWriteError(
            f"no device on {track_name!r} has a sidechain input. Load a Compressor first. "
            "that the LOM can do."
        )
    if not 0 <= device_index < len(devices):
        tags = ", ".join(f"{i}:{element.tag}" for i, element in enumerate(devices))
        plural = "device" if len(devices) == 1 else "devices"
        raise AlsWriteError(
            f"device {device_index} does not exist; {track_name!r} has {len(devices)} "
            f"sidechain-capable {plural}: {tags}"
        )
    return devices[device_index]


def _descend(node: ET.Element, path: tuple[str, ...], *, create: bool) -> ET.Element | None:
    """Traverse path of child tags, optionally creating missing intermediate elements."""
    for tag in path:
        found = node.find(f"./{tag}")
        if found is None:
            if not create:
                return None
            found = _append_child(node, tag)
        node = found
    return node


def _sidechain_notes(*, previous: str | None, source: ET.Element, tap: str) -> list[str]:
    """Compile diagnostic notes regarding sidechain wiring and routing state.

    Reports measured context only. No advice is inferred that the file does not support.
    """
    notes: list[str] = []
    if previous and previous != NOT_ROUTED:
        notes.append(f"this input was already routed to {previous!r} and now is not")
    if _is_muted(source) is True:
        notes.append(
            "the source track is muted. That does not break the tap (the sidechain reads "
            "before the mixer) and 13 % of corpus sources are muted too"
        )
    if tap == "pre":
        notes.append("423 of 425 corpus wirings tap Post FX; Pre FX is the rare choice")
    notes.append(
        "writing the file is not proof Live reads it that way. Open the set and look at the "
        "device window: it should show External on, the source track, and the tap"
    )
    return notes
