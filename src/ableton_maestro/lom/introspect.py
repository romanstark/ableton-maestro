"""Dynamic surface inspection for loaded devices and session state at runtime.

Provides typed views of device parameters, fuzzy parameter matching and ranking,
and whole-session snapshot generation.

Introspection observations:
- A VST reports only ``Device On`` until parameters are added via Live's
  Configure mode. Measured on a 3rd-party VST2 plugin (Live 12.4.3): 1 parameter
  before, 5 after four clicks in the plugin GUI. In a sweep of 328 browser entries,
  88 had exactly 1 parameter (unconfigured instances).
- Live allocates a fixed array of 128 parameter slots per VST instance.
  Unconfigured slots carry the placeholder value ``0.1234567687`` in the ``.als``
  (measured on a 3rd-party VST plugin: 128 slots, 2 named).
- VST2 does not report parameter units. Measured across 2699 parameters of 242 plugins:
  898 report a unit (33 %), 898 of 2006 VST3 (45 %), 0 of 693 VST2 (0 %).
- Quantized parameters are indicated by ``is_quantized`` with discrete steps in ``value_items``.
- Group tracks have no arm state and no arrangement clips; iteration guards with
  ``can_be_armed`` and ``is_foldable``.
"""

from __future__ import annotations

import difflib
import fnmatch
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from ableton_maestro.lom import paths

__all__ = [
    "BATCH_OP_LIMIT",
    "CONFIGURED_THRESHOLD",
    "DEVICE_ON",
    "MEASURED_UNIT_TOKENS",
    "VST_PARAMETER_SLOTS",
    "ClipView",
    "DescribeCache",
    "DeviceView",
    "Diagnosis",
    "IntrospectionError",
    "LomClient",
    "ParameterMatch",
    "ParameterNotFoundError",
    "ParameterView",
    "SessionSnapshot",
    "TrackView",
    "configuration_advice",
    "describe_device",
    "diagnose",
    "find_parameter",
    "rank_parameters",
    "require_parameter",
    "snapshot",
    "unit_of",
]


# --------------------------------------------------------------- the client seam
@runtime_checkable
class LomClient(Protocol):
    """Structural protocol defining required synchronous LOM client operations."""

    def get(self, path: str) -> Mapping[str, Any]:
        """Fetch value and type for path via lom_get."""
        ...

    def describe(self, path: str) -> Mapping[str, Any]:
        """Fetch metadata, properties, and child collections for path via lom_describe."""
        ...

    def batch(self, ops: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Execute a sequence of batch operations via lom_batch."""
        ...


class IntrospectionError(RuntimeError):
    """Raised when an object inspected in the LOM does not match expected structure."""


class ParameterNotFoundError(LookupError):
    """Raised when parameter matching fails to find a suitable match."""


# ------------------------------------------------------------------- constants
DEVICE_ON = "Device On"
"""Live's default on/off parameter, located at index 0 of a device parameter list."""

CONFIGURED_THRESHOLD = 6
"""Parameter count threshold above which a hosted device is considered configured.

Measured rule of thumb from the plugin sweep: 1 means nothing is configured, 2 means
practically nothing, 3 to 5 is borderline and worth a look, 6 and up is immediately
usable. 138 of 242 installed plugins (57 %) were at or above it at sweep time, which is
a starting position rather than a limit.
"""

VST_PARAMETER_SLOTS = 128
"""Maximum parameter slots allocated by Live per VST plugin instance."""

BATCH_OP_LIMIT = 400
"""Maximum operations sent in a single lom_batch request before chunking."""

MEASURED_UNIT_TOKENS: tuple[str, ...] = (
    "dB",
    "Hz",
    "kHz",
    "ms",
    "s",
    "%",
    "°",
    "cents",
    "semitones",
    "LU",
    "LUFS",
    "×",
)
"""Recognized parameter unit suffix tokens."""

_EXTRA_UNIT_TOKENS: tuple[str, ...] = ("st", "bpm", "x", "cent", "semitone", "Hz.")
"""Additional recognized parameter unit suffix tokens."""

_UNIT_LOOKUP = {token.lower() for token in MEASURED_UNIT_TOKENS + _EXTRA_UNIT_TOKENS}

_DISPLAY_RE = re.compile(r"^[+-]?[\d.,]+\s*(?P<unit>[^\s\d.,+\-]+)$")

# Parameter attributes fetched during parameter batch survey.
_PARAMETER_FIELDS: tuple[str, ...] = (
    "name",
    "value",
    "min",
    "max",
    "is_quantized",
    "is_enabled",
    "original_name",
)

# Song properties surveyed during whole-session snapshot generation.
_SONG_FIELDS: tuple[str, ...] = (
    "tempo",
    "signature_numerator",
    "signature_denominator",
    "is_playing",
    "current_song_time",
    "song_length",
    "loop",
    "loop_start",
    "loop_length",
    "metronome",
    "record_mode",
    "session_record",
)

_TRACK_FIELDS: tuple[str, ...] = (
    "name",
    "color",
    "mute",
    "solo",
    "is_foldable",
    "can_be_armed",
    "has_midi_input",
)


# ------------------------------------------------------------------ typed views
class Diagnosis(str, Enum):
    """Classification of a device's parameter readiness."""

    OK = "ok"
    """Device exposes at least CONFIGURED_THRESHOLD parameters."""

    SPARSE = "sparse"
    """Device exposes between two and five parameters."""

    UNCONFIGURED = "unconfigured"
    """Device reports only the default Device On parameter."""

    NO_PARAMETERS = "no_parameters"
    """Device reports zero parameters."""


@dataclass(frozen=True)
class ParameterView:
    """Represents a single device parameter and its current properties.

    Attributes:
        index: Zero-based position within device parameter chain.
        path: Canonical LOM path to the parameter.
        name: Display name exposed in Live's UI.
        value: Current raw parameter value.
        display: Formatted string rendering of value from Live.
        minimum: Lower bound of parameter range.
        maximum: Upper bound of parameter range.
        is_quantized: Whether parameter uses discrete steps.
        is_enabled: Whether parameter is active.
        original_name: Original un-aliased name from plugin.
        value_items: Step labels if parameter is quantized.
        errors: Error messages encountered during inspection.
    """

    index: int
    path: str
    name: str
    value: float | None = None
    display: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    is_quantized: bool = False
    is_enabled: bool = True
    original_name: str | None = None
    value_items: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def unit(self) -> str | None:
        """Return parsed unit token from display string, or None."""
        return unit_of(self.display)

    @property
    def has_unit(self) -> bool:
        """Return True if display string contains a recognizable unit."""
        return self.unit is not None

    @property
    def steps(self) -> int | None:
        """Return number of discrete steps if quantized, or None."""
        if not self.is_quantized:
            return None
        return len(self.value_items) or None

    @property
    def is_device_on(self) -> bool:
        """Return True if parameter is Live's Device On switch."""
        return self.name == DEVICE_ON

    def to_dict(self) -> dict[str, Any]:
        """Convert parameter view to JSON-serializable dictionary."""
        data: dict[str, Any] = {
            "index": self.index,
            "path": self.path,
            "name": self.name,
            "value": self.value,
            "min": self.minimum,
            "max": self.maximum,
            "is_quantized": self.is_quantized,
            "is_enabled": self.is_enabled,
        }
        if self.display is not None:
            data["display"] = self.display
        if self.unit is not None:
            data["unit"] = self.unit
        if self.original_name and self.original_name != self.name:
            data["original_name"] = self.original_name
        if self.value_items:
            data["value_items"] = list(self.value_items)
        if self.errors:
            data["errors"] = list(self.errors)
        return data


@dataclass(frozen=True)
class DeviceView:
    """Represents an inspected device and its active parameters.

    Attributes:
        path: Canonical LOM path to the device.
        name: Display name of the device.
        class_name: LOM class name (e.g. 'PluginDevice', 'Eq8').
        index: Index of device within its track's device chain.
        track: Canonical path to parent track.
        is_active: Whether device is active and processing audio.
        parameters: Inspected parameters exposed by this device instance.
        diagnosis: Parameter readiness classification.
        errors: Inspection errors recorded during device traversal.
    """

    path: str
    name: str
    class_name: str
    index: int
    track: str
    is_active: bool | None = None
    parameters: tuple[ParameterView, ...] = ()
    diagnosis: Diagnosis = Diagnosis.OK
    errors: tuple[str, ...] = ()

    @property
    def parameter_count(self) -> int:
        """Total number of exposed parameters including Device On."""
        return len(self.parameters)

    @property
    def playable_count(self) -> int:
        """Number of controllable parameters excluding Device On."""
        return sum(1 for param in self.parameters if not param.is_device_on)

    @property
    def reports_units(self) -> bool:
        """Return True if at least one parameter displays a recognized unit."""
        return any(param.has_unit for param in self.parameters)

    @property
    def is_plugin(self) -> bool:
        """Return True if device is a hosted VST or AU plugin."""
        return "Plugin" in self.class_name

    @property
    def advice(self) -> str | None:
        """Actionable configuration advice if parameters are unconfigured."""
        return configuration_advice(self)

    def parameter(self, index: int) -> ParameterView:
        """Retrieve parameter view at index, or raise IndexError."""
        for param in self.parameters:
            if param.index == index:
                return param
        raise IndexError(
            f"{self.name!r} has no parameter at index {index}; "
            f"it reports {self.parameter_count} (0..{max(self.parameter_count - 1, 0)})"
        )

    def names(self, limit: int | None = 30) -> list[str]:
        """Return list of parameter names, optionally truncated."""
        names = [param.name for param in self.parameters]
        if limit is not None and len(names) > limit:
            return [*names[:limit], "..."]
        return names

    def to_dict(self) -> dict[str, Any]:
        """Convert device view to JSON-serializable dictionary."""
        data: dict[str, Any] = {
            "path": self.path,
            "index": self.index,
            "track": self.track,
            "name": self.name,
            "class": self.class_name,
            "is_active": self.is_active,
            "parameter_count": self.parameter_count,
            "playable_count": self.playable_count,
            "reports_units": self.reports_units,
            "diagnosis": self.diagnosis.value,
            "parameters": [param.to_dict() for param in self.parameters],
        }
        advice = self.advice
        if advice is not None:
            data["advice"] = advice
        if self.errors:
            data["errors"] = list(self.errors)
        return data


@dataclass(frozen=True)
class ParameterMatch:
    """Represents a scored candidate from parameter name resolution.

    Attributes:
        parameter: Matched ParameterView instance.
        score: Match score between 0.0 and 1.0.
        reason: Explanation of matching rule applied.
    """

    parameter: ParameterView
    score: float
    reason: str

    @property
    def name(self) -> str:
        """Return display name of the matched parameter."""
        return self.parameter.name

    @property
    def index(self) -> int:
        """Return index of the matched parameter."""
        return self.parameter.index

    def to_dict(self) -> dict[str, Any]:
        """Convert match result to JSON-serializable dictionary."""
        return {
            "index": self.parameter.index,
            "name": self.parameter.name,
            "path": self.parameter.path,
            "score": round(self.score, 3),
            "matched_by": self.reason,
        }


@dataclass(frozen=True)
class ClipView:
    """Represents an inspected session clip slot.

    Attributes:
        track: Zero-based track index.
        slot: Zero-based clip slot index.
        path: Canonical LOM path to the clip.
        name: Name of the clip.
        length: Duration in beats.
        is_midi: Whether clip is MIDI or audio.
    """

    track: int
    slot: int
    path: str
    name: str = ""
    length: float | None = None
    is_midi: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert clip view to JSON-serializable dictionary."""
        return {
            "track": self.track,
            "slot": self.slot,
            "path": self.path,
            "name": self.name,
            "length": self.length,
            "is_midi": self.is_midi,
        }


@dataclass(frozen=True)
class TrackView:
    """Represents an inspected track in a session snapshot.

    Attributes:
        index: Track index within its respective collection.
        path: Canonical LOM path to the track.
        name: Track display name.
        kind: Classification ('track', 'group', 'midi', 'audio', 'return', 'master').
        can_be_armed: Whether track supports arming.
        is_foldable: Whether track is a group track container.
        mute: Mute state.
        solo: Solo state.
        armed: Arm state, or None if track cannot be armed.
        color: RGB color integer.
        devices: Device name and class tuples.
        clips: Inspected session clip views.
    """

    index: int
    path: str
    name: str
    kind: str
    can_be_armed: bool = False
    is_foldable: bool = False
    mute: bool | None = None
    solo: bool | None = None
    armed: bool | None = None
    color: int | None = None
    devices: tuple[Mapping[str, str], ...] = ()
    clips: tuple[ClipView, ...] = ()

    @property
    def is_group(self) -> bool:
        """Return True if track is a group track container."""
        return self.kind == "group"

    def to_dict(self) -> dict[str, Any]:
        """Convert track view to JSON-serializable dictionary."""
        return {
            "index": self.index,
            "path": self.path,
            "name": self.name,
            "kind": self.kind,
            "mute": self.mute,
            "solo": self.solo,
            "armed": self.armed,
            "can_be_armed": self.can_be_armed,
            "color": self.color,
            "devices": [dict(device) for device in self.devices],
            "clips": [clip.to_dict() for clip in self.clips],
        }


@dataclass(frozen=True)
class SessionSnapshot:
    """Complete in-memory survey of song, tracks, devices, and clips.

    Attributes:
        song: Mapping of top-level song properties.
        tracks: Inspected regular and group tracks.
        return_tracks: Inspected return tracks.
        master: Master track view, or None.
        scene_count: Total number of scenes in set.
        scene_names: Names of surveyed scenes.
        taken_at: Unix timestamp when snapshot was collected.
        warnings: Warnings recorded during survey.
    """

    song: dict[str, Any] = field(default_factory=dict)
    tracks: tuple[TrackView, ...] = ()
    return_tracks: tuple[TrackView, ...] = ()
    master: TrackView | None = None
    scene_count: int = 0
    scene_names: tuple[str, ...] = ()
    taken_at: float = 0.0
    warnings: tuple[str, ...] = ()

    @property
    def track_count(self) -> int:
        """Return count of regular and group tracks."""
        return len(self.tracks)

    @property
    def group_tracks(self) -> tuple[TrackView, ...]:
        """Return tuple of group track views."""
        return tuple(track for track in self.tracks if track.is_group)

    @property
    def session_clip_count(self) -> int:
        """Return count of session clips found across surveyed tracks."""
        return sum(len(track.clips) for track in self.tracks)

    def to_dict(self) -> dict[str, Any]:
        """Convert session snapshot to JSON-serializable dictionary."""
        return {
            "song": dict(self.song),
            "scene_count": self.scene_count,
            "scene_names": list(self.scene_names),
            "tracks": [track.to_dict() for track in self.tracks],
            "return_tracks": [track.to_dict() for track in self.return_tracks],
            "master": self.master.to_dict() if self.master is not None else None,
            "taken_at": self.taken_at,
            "warnings": list(self.warnings),
        }


# ------------------------------------------------------------------------ cache
class DescribeCache:
    """Path-keyed cache of DeviceView instances with explicit invalidation.

    >>> cache = DescribeCache()
    >>> cache.put("song.tracks[0].devices[0]", "view")
    >>> cache.get("song.tracks[0].devices[0]")
    'view'
    >>> cache.invalidate("song.tracks[0]")
    1
    >>> cache.get("song.tracks[0].devices[0]") is None
    True
    """

    def __init__(self) -> None:
        self._entries: dict[str, Any] = {}
        self.hits = 0
        self.misses = 0

    def get(self, path: str) -> Any | None:
        """Retrieve cached value for path, or None."""
        value = self._entries.get(path)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def put(self, path: str, value: Any) -> None:
        """Store value under path in cache."""
        self._entries[path] = value

    def invalidate(self, path: str | None = None) -> int:
        """Evict path and descendant paths from cache; if None, evict all.

        Args:
            path: Path prefix to invalidate, or None to clear cache.

        Returns:
            Number of entries removed.
        """
        if path is None:
            return self.clear()
        prefix = path.strip()
        below = (f"{prefix}.", f"{prefix}[")
        doomed = [key for key in self._entries if key == prefix or key.startswith(below)]
        for key in doomed:
            del self._entries[key]
        return len(doomed)

    def clear(self) -> int:
        """Clear all entries from cache and return count of evicted items."""
        count = len(self._entries)
        self._entries.clear()
        return count

    def stats(self) -> dict[str, int]:
        """Return cache hit, miss, and size statistics."""
        return {"entries": len(self._entries), "hits": self.hits, "misses": self.misses}

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, path: object) -> bool:
        return path in self._entries


# ------------------------------------------------------------------ units
def unit_of(display: str | None) -> str | None:
    """Extract recognized unit suffix token from a display string.

    Args:
        display: Formatted parameter value string from Live.

    Returns:
        Unit token string, or None if no recognized unit suffix is present.

    >>> unit_of("-80.0 dB")
    'dB'
    >>> unit_of("353 Hz")
    'Hz'
    >>> unit_of("50%")
    '%'
    >>> unit_of("0.00") is None
    True
    >>> unit_of("-inf dB") is None
    True
    >>> unit_of(None) is None
    True
    """
    if not display:
        return None
    match = _DISPLAY_RE.match(display.strip())
    if match is None:
        return None
    token = match.group("unit").strip()
    return token if token.lower() in _UNIT_LOOKUP else None


# ------------------------------------------------------------------ devices
def describe_device(
    client: LomClient,
    track: int | str,
    device: int,
    *,
    cache: DescribeCache | None = None,
    with_value_items: bool = True,
) -> DeviceView:
    """Survey a device and return a typed DeviceView with its parameters.

    Args:
        client: LomClient instance for LOM communication.
        track: Track index or canonical path.
        device: Zero-based index within track device chain.
        cache: Optional DescribeCache for caching DeviceView instances.
        with_value_items: Whether to fetch discrete labels for quantized parameters.

    Returns:
        DeviceView representing device and its parameter state.

    Raises:
        IntrospectionError: If path does not reference a valid device object.
    """
    path = paths.device(track, device)
    if cache is not None:
        cached = cache.get(path)
        if isinstance(cached, DeviceView):
            return cached

    info = _payload(client.describe(path))
    properties = _properties(info)
    count = _child_count(info, "parameters")
    errors: list[str] = []
    if count is None:
        raise IntrospectionError(
            f"{paths.describe_path(path)} does not look like a device: "
            f"lom_describe reported class {info.get('class', '?')!r} with no "
            "'parameters' child"
        )

    parameters = _survey_parameters(client, path, count, errors)
    if with_value_items and any(param.is_quantized for param in parameters):
        parameters = _fill_value_items(client, parameters, errors)

    return _finish_device(
        path=path,
        info=info,
        properties=properties,
        parameters=parameters,
        device_index=device,
        track=track,
        errors=errors,
        cache=cache,
    )


def _finish_device(
    *,
    path: str,
    info: Mapping[str, Any],
    properties: Mapping[str, Any],
    parameters: tuple[ParameterView, ...],
    device_index: int,
    track: int | str,
    errors: list[str],
    cache: DescribeCache | None,
) -> DeviceView:
    """Assemble, diagnose and cache the finished view."""
    view = DeviceView(
        path=path,
        name=str(info.get("name") or properties.get("name") or "(unnamed)"),
        class_name=str(info.get("class") or properties.get("class_name") or "Device"),
        index=device_index,
        track=paths.parent(paths.parent(path)) or "song",
        is_active=_as_bool(properties.get("is_active")),
        parameters=parameters,
        diagnosis=diagnose(parameters),
        errors=tuple(errors),
    )
    if cache is not None:
        cache.put(path, view)
    return view
def diagnose(parameters: Sequence[ParameterView]) -> Diagnosis:
    """Classify parameter readiness by examining parameter count and names.

    Args:
        parameters: Sequence of ParameterView instances.

    Returns:
        Diagnosis indicating parameter configuration readiness.

    >>> diagnose([ParameterView(0, "p", DEVICE_ON)]) is Diagnosis.UNCONFIGURED
    True
    >>> diagnose([]) is Diagnosis.NO_PARAMETERS
    True
    >>> diagnose([ParameterView(i, "p", f"P{i}") for i in range(8)]) is Diagnosis.OK
    True
    """
    if not parameters:
        return Diagnosis.NO_PARAMETERS
    if len(parameters) == 1 and parameters[0].name == DEVICE_ON:
        return Diagnosis.UNCONFIGURED
    if len(parameters) < CONFIGURED_THRESHOLD:
        return Diagnosis.SPARSE
    return Diagnosis.OK


def configuration_advice(view: DeviceView) -> str | None:
    """Generate guidance for configuring unexposed parameters on hosted devices.

    Args:
        view: DeviceView to evaluate.

    Returns:
        Helpful configuration instructions string, or None if configured.
    """
    if view.diagnosis is Diagnosis.OK:
        return None

    if view.diagnosis is Diagnosis.NO_PARAMETERS:
        return (
            f"{view.name!r} reports no parameters at all, not even Live's own "
            f"{DEVICE_ON!r}. Every device has that one, so this path probably does not "
            "point at a device, or the survey failed - check the 'errors' field."
        )

    if view.diagnosis is Diagnosis.UNCONFIGURED:
        head = (
            f"{view.name!r} reports exactly one parameter, {DEVICE_ON!r}. That does not "
            "mean the plugin refuses to expose its controls - it means nothing has been "
            "taken into Live's parameter strip yet. device.parameters shows the "
            "configuration state of this instance, not the plugin's capability."
        )
    else:
        head = (
            f"{view.name!r} reports {view.parameter_count} parameters "
            f"({view.playable_count} besides {DEVICE_ON!r}). Below {CONFIGURED_THRESHOLD} "
            "usually means the parameter strip was barely configured, not that the "
            "plugin is limited."
        )

    return (
        f"{head} To expose more, in Live's Device View: open the plugin window, press "
        "Configure on the device, click the wanted controls in the plugin's OWN GUI "
        "(not in Live's strip), then switch Configure off. Measured on a 3rd-party VST2 plugin ("
        "Live 12.4.3): 1 parameter before, 5 after four clicks, and automation on the "
        f"new parameters holds. Limits: {VST_PARAMETER_SLOTS} slots per instance; "
        "the mapping is stored in the Live set, per parameter and per device instance, "
        "so a second instance of the same plugin starts empty again; and VST2 reports no "
        "units however configured, though VST3 may. Configure is GUI-only, but the mapping "
        "can also be written to the .als file directly (als/write.py configure_plugin_parameters)."
    )


def _survey_parameters(
    client: LomClient,
    device_path: str,
    count: int,
    errors: list[str],
) -> tuple[ParameterView, ...]:
    """Fetch all parameter properties for a device using batched LOM queries."""
    if count <= 0:
        return ()

    wanted: list[str] = []
    for index in range(count):
        base = f"{device_path}.parameters[{index}]"
        wanted.append(base)
        wanted.extend(f"{base}.{field_name}" for field_name in _PARAMETER_FIELDS)

    results = _get_many(client, wanted)
    stride = len(_PARAMETER_FIELDS) + 1
    parameters: list[ParameterView] = []

    for index in range(count):
        chunk = results[index * stride : (index + 1) * stride]
        base_path = f"{device_path}.parameters[{index}]"
        parameters.append(_parameter_from(index, base_path, chunk, errors))
    return tuple(parameters)


def _parameter_from(
    index: int,
    path: str,
    chunk: Sequence[Mapping[str, Any]],
    errors: list[str],
) -> ParameterView:
    """Construct ParameterView from chunk of batch query responses."""
    head = chunk[0] if chunk else {}
    fields = dict(zip(_PARAMETER_FIELDS, chunk[1:], strict=False))
    local: list[str] = []

    def field_value(name: str) -> Any:
        entry = fields.get(name)
        if entry is None:
            return None
        if _is_error(entry):
            local.append(f"{name}: {_error_text(entry)}")
            return None
        return entry.get("value")

    if _is_error(head):
        local.append(f"value: {_error_text(head)}")
        display = None
    else:
        display = head.get("display")
    value = _as_float(field_value("value"))

    name = field_value("name")
    original = field_value("original_name")
    view = ParameterView(
        index=index,
        path=path,
        name=str(name) if name is not None else f"parameter {index}",
        value=value,
        display=str(display) if display is not None else None,
        minimum=_as_float(field_value("min")),
        maximum=_as_float(field_value("max")),
        is_quantized=bool(_as_bool(field_value("is_quantized"))),
        is_enabled=_as_bool(field_value("is_enabled")) is not False,
        original_name=str(original) if original is not None else None,
        errors=tuple(local),
    )
    errors.extend(
        f"{path}: {problem}" for problem in local if not problem.startswith("original_name:")
    )
    return view


def _fill_value_items(
    client: LomClient,
    parameters: Sequence[ParameterView],
    errors: list[str],
) -> tuple[ParameterView, ...]:
    """Populate discrete step labels for quantized parameters."""
    quantised = [param for param in parameters if param.is_quantized]
    if not quantised:
        return tuple(parameters)

    results = _get_many(client, [f"{param.path}.value_items" for param in quantised])
    filled: dict[int, tuple[str, ...]] = {}
    for param, entry in zip(quantised, results, strict=False):
        if _is_error(entry):
            errors.append(f"{param.path}.value_items: {_error_text(entry)}")
            continue
        raw = entry.get("value")
        if isinstance(raw, (list, tuple)):
            filled[param.index] = tuple(str(item) for item in raw)

    return tuple(
        param
        if param.index not in filled
        else ParameterView(
            index=param.index,
            path=param.path,
            name=param.name,
            value=param.value,
            display=param.display,
            minimum=param.minimum,
            maximum=param.maximum,
            is_quantized=param.is_quantized,
            is_enabled=param.is_enabled,
            original_name=param.original_name,
            value_items=filled[param.index],
            errors=param.errors,
        )
        for param in parameters
    )


# ------------------------------------------------------------------ finding
def rank_parameters(
    view: DeviceView,
    name_or_pattern: str,
    *,
    limit: int = 8,
    floor: float = 0.35,
) -> list[ParameterMatch]:
    """Rank device parameters matching a name or glob pattern.

    Matching applies exact, case-insensitive, separator-insensitive,
    glob, prefix, substring, and difflib similarity scoring.

    Args:
        view: Inspected DeviceView instance.
        name_or_pattern: Target parameter name or glob pattern.
        limit: Maximum number of matches returned.
        floor: Minimum similarity score threshold.

    Returns:
        List of ParameterMatch instances sorted by descending score.

    Raises:
        ValueError: If name_or_pattern is empty.

    >>> view = DeviceView("d", "EQ", "Eq8", 0, "song.tracks[0]", parameters=(
    ...     ParameterView(0, "p0", DEVICE_ON),
    ...     ParameterView(1, "p1", "Dry/Wet"),
    ...     ParameterView(2, "p2", "Frequency A"),
    ... ))
    >>> [m.name for m in rank_parameters(view, "dry wet")]
    ['Dry/Wet']
    >>> [m.name for m in rank_parameters(view, "freq*")]
    ['Frequency A']
    >>> rank_parameters(view, "Dry/Wet")[0].reason
    'exact'
    """
    query = str(name_or_pattern).strip()
    if not query:
        raise ValueError("rank_parameters: empty search term")

    scored: list[ParameterMatch] = []
    for param in view.parameters:
        best = _score_parameter(param, query)
        if best is not None and best[0] >= floor:
            scored.append(ParameterMatch(param, best[0], best[1]))

    scored.sort(key=lambda match: (-match.score, match.parameter.index))
    return scored[:limit] if limit else scored


def find_parameter(
    client: LomClient,
    track: int | str,
    device: int,
    name_or_pattern: str,
    *,
    limit: int = 8,
    cache: DescribeCache | None = None,
    view: DeviceView | None = None,
) -> list[ParameterMatch]:
    """Find device parameters matching name or glob, returning ranked matches.

    Args:
        client: LomClient instance.
        track: Track index or path.
        device: Device chain index.
        name_or_pattern: Parameter name or pattern.
        limit: Maximum candidates returned.
        cache: Optional DescribeCache.
        view: Optional pre-fetched DeviceView.

    Returns:
        List of scored ParameterMatch instances.
    """
    resolved = view if view is not None else describe_device(client, track, device, cache=cache)
    return rank_parameters(resolved, name_or_pattern, limit=limit)


def require_parameter(
    client: LomClient,
    track: int | str,
    device: int,
    name_or_pattern: str,
    *,
    cache: DescribeCache | None = None,
    view: DeviceView | None = None,
) -> ParameterView:
    """Return top matching parameter view or raise ParameterNotFoundError.

    Args:
        client: LomClient instance.
        track: Track index or path.
        device: Device chain index.
        name_or_pattern: Parameter name or pattern.
        cache: Optional DescribeCache.
        view: Optional pre-fetched DeviceView.

    Returns:
        Best matching ParameterView.

    Raises:
        ParameterNotFoundError: If no parameter meets the matching threshold.
    """
    resolved = view if view is not None else describe_device(client, track, device, cache=cache)
    matches = rank_parameters(resolved, name_or_pattern, limit=5)
    if matches:
        return matches[0].parameter

    parts = [
        (
            f"no parameter matching {name_or_pattern!r} on {resolved.name!r} "
            f"({paths.describe_path(resolved.path)})."
        )
    ]
    advice = resolved.advice
    if advice is not None:
        parts.append(advice)
    else:
        parts.append(f"Available: {', '.join(resolved.names())}")
    raise ParameterNotFoundError(" ".join(parts))


def _score_parameter(param: ParameterView, query: str) -> tuple[float, str] | None:
    """Calculate highest match score and reason for parameter against query."""
    candidates: list[tuple[str, float, str]] = [(param.name, 0.0, "")]
    if param.original_name and param.original_name != param.name:
        candidates.append((param.original_name, 0.03, " on original_name"))

    best: tuple[float, str] | None = None
    for name, penalty, suffix in candidates:
        hit = _score_name(name, query)
        if hit is None:
            continue
        score, reason = hit[0] - penalty, hit[1] + suffix
        if best is None or score > best[0]:
            best = (score, reason)
    return best


def _score_name(name: str, query: str) -> tuple[float, str] | None:
    """Apply tiered string matching heuristics between candidate name and query."""
    if name == query:
        return 1.0, "exact"
    if name.lower() == query.lower():
        return 0.95, "case-insensitive"

    folded_name, folded_query = _fold(name), _fold(query)
    if folded_name == folded_query:
        return 0.90, "separator-insensitive"
    if any(char in query for char in "*?[") and fnmatch.fnmatch(name.lower(), query.lower()):
        return 0.85, "glob"
    if folded_name.startswith(folded_query):
        return 0.80, "prefix"
    if folded_query in folded_name:
        return 0.70, "substring"

    ratio = difflib.SequenceMatcher(None, folded_name, folded_query).ratio()
    if ratio >= 0.55:
        return ratio * 0.65, f"fuzzy {ratio:.2f}"
    return None


def _fold(name: str) -> str:
    """Strip whitespace and punctuation for separator-insensitive matching."""
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


# ------------------------------------------------------------------ session survey
def snapshot(
    client: LomClient,
    *,
    clips: bool = True,
    devices: bool = True,
    max_scenes: int | None = 64,
) -> SessionSnapshot:
    """Capture in-memory state of song, tracks, devices, and clips.

    Args:
        client: LomClient instance.
        clips: Whether to include session clip slots in snapshot.
        devices: Whether to include device chains in snapshot.
        max_scenes: Scene count ceiling for clip slot sweep.

    Returns:
        SessionSnapshot populated with session hierarchy.
    """
    warnings: list[str] = []
    song_info = _payload(client.describe("song"))
    song_props = _properties(song_info)

    track_count = _child_count(song_info, "tracks") or 0
    return_count = _child_count(song_info, "return_tracks") or 0
    scene_count = _child_count(song_info, "scenes") or 0

    song = {name: song_props[name] for name in _SONG_FIELDS if name in song_props}
    missing = [name for name in _SONG_FIELDS if name not in song]

    track_paths = [paths.track(index) for index in range(track_count)]
    return_paths = [paths.return_track(index) for index in range(return_count)]
    all_paths = [*track_paths, *return_paths, paths.master()]

    wanted = [f"song.{name}" for name in missing]
    for path in all_paths:
        wanted.extend(f"{path}.{field_name}" for field_name in _TRACK_FIELDS)
    results = _get_many(client, wanted)

    for name, entry in zip(missing, results[: len(missing)], strict=False):
        if not _is_error(entry):
            song[name] = entry.get("value")

    stride = len(_TRACK_FIELDS)
    offset = len(missing)
    raw_tracks: list[dict[str, Any]] = []
    for position, path in enumerate(all_paths):
        chunk = results[offset + position * stride : offset + (position + 1) * stride]
        raw_tracks.append(_track_fields(path, chunk))

    kinds = ["track"] * track_count + ["return"] * return_count + ["master"]
    for raw, kind in zip(raw_tracks, kinds, strict=False):
        raw["kind"] = _track_kind(raw, kind)

    armed = _read_arm(client, raw_tracks, warnings)
    device_names = _read_devices(client, raw_tracks, warnings) if devices else {}
    slots = scene_count if max_scenes is None else min(scene_count, max_scenes)
    if clips and scene_count > slots:
        warnings.append(
            f"clip sweep capped at {slots} of {scene_count} scenes (max_scenes); "
            "raise max_scenes for the rest"
        )
    clip_views = (
        _read_clips(client, raw_tracks[:track_count], slots, warnings) if clips and slots else {}
    )

    views = [
        TrackView(
            index=_trailing_index(raw["path"]) or 0,
            path=raw["path"],
            name=str(raw.get("name") or ""),
            kind=raw["kind"],
            can_be_armed=bool(raw.get("can_be_armed")),
            is_foldable=bool(raw.get("is_foldable")),
            mute=_as_bool(raw.get("mute")),
            solo=_as_bool(raw.get("solo")),
            armed=armed.get(raw["path"]),
            color=_as_int(raw.get("color")),
            devices=tuple(device_names.get(raw["path"], ())),
            clips=tuple(clip_views.get(raw["path"], ())),
        )
        for raw in raw_tracks
    ]

    return SessionSnapshot(
        song=song,
        tracks=tuple(views[:track_count]),
        return_tracks=tuple(views[track_count : track_count + return_count]),
        master=views[-1] if views else None,
        scene_count=scene_count,
        scene_names=_read_scene_names(client, slots, warnings) if slots else (),
        taken_at=time.time(),
        warnings=tuple(warnings),
    )


def _track_fields(path: str, chunk: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Map raw track query results to a dictionary of field values."""
    raw: dict[str, Any] = {"path": path}
    for name, entry in zip(_TRACK_FIELDS, chunk, strict=False):
        if not _is_error(entry):
            raw[name] = entry.get("value")
    return raw


def _track_kind(raw: Mapping[str, Any], family: str) -> str:
    """Determine track classification ('group', 'midi', 'audio', 'return', 'master')."""
    if family != "track":
        return family
    if _as_bool(raw.get("is_foldable")):
        return "group"
    has_midi = _as_bool(raw.get("has_midi_input"))
    if has_midi is None:
        return "track"
    return "midi" if has_midi else "audio"


def _read_arm(
    client: LomClient,
    raw_tracks: Sequence[Mapping[str, Any]],
    warnings: list[str],
) -> dict[str, bool | None]:
    """Query arm state for tracks that declare can_be_armed."""
    armable = [raw["path"] for raw in raw_tracks if raw.get("can_be_armed")]
    if not armable:
        return {}
    results = _get_many(client, [f"{path}.arm" for path in armable])
    armed: dict[str, bool | None] = {}
    for path, entry in zip(armable, results, strict=False):
        if _is_error(entry):
            warnings.append(f"{path}.arm: {_error_text(entry)}")
            continue
        armed[path] = _as_bool(entry.get("value"))
    return armed


def _read_devices(
    client: LomClient,
    raw_tracks: Sequence[Mapping[str, Any]],
    warnings: list[str],
) -> dict[str, list[dict[str, str]]]:
    """Fetch device name and class metadata for all tracks."""
    lists = _get_many(client, [f"{raw['path']}.devices" for raw in raw_tracks])
    counts: dict[str, int] = {}
    for raw, entry in zip(raw_tracks, lists, strict=False):
        if _is_error(entry):
            warnings.append(f"{raw['path']}.devices: {_error_text(entry)}")
            continue
        length = _sequence_length(entry.get("value"))
        if length:
            counts[raw["path"]] = length

    wanted: list[tuple[str, str, str]] = [
        (path, f"{path}.devices[{index}].name", f"{path}.devices[{index}].class_name")
        for path, length in counts.items()
        for index in range(length)
    ]
    if not wanted:
        return {}

    results = _get_many(
        client, [leg for _, name_path, class_path in wanted for leg in (name_path, class_path)]
    )
    found: dict[str, list[dict[str, str]]] = {path: [] for path in counts}
    for position, (owner, _name_path, _class_path) in enumerate(wanted):
        name_entry = results[position * 2] if position * 2 < len(results) else None
        class_entry = results[position * 2 + 1] if position * 2 + 1 < len(results) else None
        found[owner].append(
            {
                "name": _device_text(name_entry),
                "class": _device_text(class_entry),
            }
        )
    return found


def _device_text(entry: Any) -> str:
    """Extract string value from device entry or fallback to '(unreadable)'."""
    if entry is None or _is_error(entry):
        return "(unreadable)"
    value = entry.get("value")
    return str(value) if value else "(unreadable)"


def _read_clips(
    client: LomClient,
    raw_tracks: Sequence[Mapping[str, Any]],
    slots: int,
    warnings: list[str],
) -> dict[str, list[ClipView]]:
    """Survey session clip slots to identify populated clips and properties."""
    candidates = [raw for raw in raw_tracks if raw.get("kind") != "group"]
    if not candidates or slots <= 0:
        return {}

    probes = [
        (raw["path"], slot, f"{raw['path']}.clip_slots[{slot}].has_clip")
        for raw in candidates
        for slot in range(slots)
    ]
    probed = _get_many(client, [probe for _, _, probe in probes])
    filled: list[tuple[str, int]] = []
    for (path, slot, _), entry in zip(probes, probed, strict=False):
        if not _is_error(entry) and _as_bool(entry.get("value")):
            filled.append((path, slot))
    if not filled:
        return {}

    fields = ("name", "length", "is_midi_clip")
    wanted = [f"{path}.clip_slots[{slot}].clip.{name}" for path, slot in filled for name in fields]
    results = _get_many(client, wanted)

    clips: dict[str, list[ClipView]] = {}
    for position, (path, slot) in enumerate(filled):
        chunk = results[position * len(fields) : (position + 1) * len(fields)]
        values = {
            name: (None if _is_error(entry) else entry.get("value"))
            for name, entry in zip(fields, chunk, strict=False)
        }
        if all(value is None for value in values.values()):
            warnings.append(f"{path}.clip_slots[{slot}].clip: reported filled but unreadable")
        track_index = _trailing_index(path)
        clips.setdefault(path, []).append(
            ClipView(
                track=track_index if track_index is not None else -1,
                slot=slot,
                path=f"{path}.clip_slots[{slot}].clip",
                name=str(values.get("name") or ""),
                length=_as_float(values.get("length")),
                is_midi=_as_bool(values.get("is_midi_clip")),
            )
        )
    return clips


def _read_scene_names(client: LomClient, count: int, warnings: list[str]) -> tuple[str, ...]:
    """Retrieve scene names up to specified count."""
    results = _get_many(client, [f"song.scenes[{index}].name" for index in range(count)])
    names: list[str] = []
    for index, entry in enumerate(results):
        if _is_error(entry):
            warnings.append(f"song.scenes[{index}].name: {_error_text(entry)}")
            names.append("")
            continue
        names.append(str(entry.get("value") or ""))
    return tuple(names)


# ------------------------------------------------------------------ wire helpers
def _get_many(client: LomClient, wanted: Sequence[str]) -> list[dict[str, Any]]:
    """Execute batched get requests across paths, preserving order."""
    out: list[dict[str, Any]] = []
    for start in range(0, len(wanted), BATCH_OP_LIMIT):
        chunk = wanted[start : start + BATCH_OP_LIMIT]
        reply = _payload(client.batch([{"op": "get", "path": path} for path in chunk]))
        results = reply.get("results")
        if not isinstance(results, list):
            raise IntrospectionError(
                f"lom_batch returned no 'results' list (got keys {sorted(reply)})"
            )
        if len(results) != len(chunk):
            raise IntrospectionError(
                f"lom_batch returned {len(results)} results for {len(chunk)} ops; "
                "results are contractually in order and complete (docs/protocol.md §5.7)"
            )
        out.extend(item if isinstance(item, Mapping) else {"code": "internal"} for item in results)
    return [dict(item) for item in out]


def _payload(reply: Any) -> dict[str, Any]:
    """Unwrap result mapping from client response envelope."""
    if not isinstance(reply, Mapping):
        raise IntrospectionError(f"expected an object from the client, got {type(reply).__name__}")
    inner = reply.get("result")
    if "status" in reply and isinstance(inner, Mapping):
        return dict(inner)
    if "status" in reply and reply.get("status") == "error":
        raise IntrospectionError(f"{reply.get('code', 'error')}: {reply.get('message', '')}")
    return dict(reply)


def _is_error(entry: Mapping[str, Any] | None) -> bool:
    """Return True if batch response entry represents an error object."""
    if entry is None:
        return True
    return "code" in entry or entry.get("status") == "error"


def _error_text(entry: Mapping[str, Any]) -> str:
    """Format short error string from response entry."""
    code = entry.get("code", "error")
    message = entry.get("message")
    return f"{code}: {message}" if message else str(code)


def _properties(info: Mapping[str, Any]) -> dict[str, Any]:
    """Extract properties mapping from lom_describe output."""
    out: dict[str, Any] = {}
    raw = info.get("properties")
    if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if isinstance(item, Mapping) and "name" in item and "value" in item:
                out[str(item["name"])] = item["value"]
    return out


def _child_count(info: Mapping[str, Any], name: str) -> int | None:
    """Extract child count for named collection from lom_describe output."""
    raw = info.get("children")
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes)):
        return None
    for item in raw:
        if isinstance(item, Mapping) and item.get("name") == name:
            return _as_int(item.get("count")) or 0
    return None


def _sequence_length(value: Any) -> int | None:
    """Extract element count from list or count mapping."""
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, Mapping) and "count" in value:
        return _as_int(value.get("count"))
    return None


def _trailing_index(path: str) -> int | None:
    """Parse final collection index from canonical path."""
    try:
        return paths.parse(path)[-1].index
    except paths.PathSyntaxError:
        return None


def _as_float(value: Any) -> float | None:
    """Coerce value to float or return None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    """Coerce value to int or return None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool | None:
    """Coerce value to bool or return None if missing or invalid."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "on", "yes"}:
            return True
        if lowered in {"false", "0", "off", "no"}:
            return False
    return None
