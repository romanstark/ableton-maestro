"""LOM path grammar and path manipulation utilities.

Implements the path parser, validator, and builder mirroring the Remote Script's
LOM path resolver (docs/protocol.md §6).

Path grammar:
    root    = song | app
    segment = name | name "[" int "]"
    path    = root ("." segment)*

Examples:
    >>> parse("song.tracks[3].mixer_device.volume")[1]
    Segment(name='tracks', index=3)
    >>> build("song.tracks[{track}].clip_slots[{slot}].clip", track=2, slot=0)
    'song.tracks[2].clip_slots[0].clip'
    >>> parameter(3, 0, 5)
    'song.tracks[3].devices[0].parameters[5]'
    >>> parent("song.tracks[3].devices[0]")
    'song.tracks[3].devices'
    >>> describe_path("song.tracks[3].mixer_device.volume")
    "the song, track 4 (index 3), the mixer device, 'volume'"
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "ROOTS",
    "PathSyntaxError",
    "Segment",
    "build",
    "clip",
    "clip_slot",
    "describe_path",
    "device",
    "is_valid",
    "join",
    "master",
    "mixer",
    "parameter",
    "parent",
    "parse",
    "return_track",
    "scene",
    "send",
    "track",
    "unparse",
    "validate",
]

ROOTS: tuple[str, str] = ("song", "app")
"""Allowed root object identifiers (docs/protocol.md §6)."""

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SEGMENT_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[(0|[1-9][0-9]*)\])?")
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class PathSyntaxError(ValueError):
    """Raised when a path string violates the LOM path grammar (docs/protocol.md §6)."""

    code = "bad_path"

    def __init__(self, path: str, reason: str, position: int | None = None) -> None:
        self.path = path
        self.reason = reason
        self.position = position
        where = "" if position is None else f" at position {position}"
        super().__init__(f"bad path {path!r}{where}: {reason}")


@dataclass(frozen=True)
class Segment:
    """Represents a single path segment with optional list index.

    >>> str(Segment("tracks", 3))
    'tracks[3]'
    >>> str(Segment("mixer_device"))
    'mixer_device'
    """

    name: str
    index: int | None = None

    def __str__(self) -> str:
        return self.name if self.index is None else f"{self.name}[{self.index}]"

    @property
    def is_indexed(self) -> bool:
        """Return True if this segment indexes an element of a list."""
        return self.index is not None


# ------------------------------------------------------------------ parsing


def parse(path: str) -> list[Segment]:
    """Parse path into a list of Segment objects starting with the root segment.

    >>> parse("song")
    [Segment(name='song', index=None)]
    >>> [str(s) for s in parse("song.tracks[0].devices[1].parameters[2]")]
    ['song', 'tracks[0]', 'devices[1]', 'parameters[2]']
    >>> [str(s) for s in parse("song.view.selected_track")]
    ['song', 'view', 'selected_track']
    """
    if not isinstance(path, str):
        raise PathSyntaxError(str(path), f"expected a string, got {type(path).__name__}")

    lead = len(path) - len(path.lstrip())
    text = path.strip()
    if not text:
        raise PathSyntaxError(path, f"empty path; expected a root, one of {_root_list()}")

    segments: list[Segment] = []
    offset = lead
    for raw in text.split("."):
        segments.append(_parse_segment(raw, path, offset))
        offset += len(raw) + 1

    root = segments[0]
    if root.name not in ROOTS:
        raise PathSyntaxError(
            path,
            f"{root.name!r} is not a root; expected one of {_root_list()}",
            lead,
        )
    if root.is_indexed:
        raise PathSyntaxError(path, f"the root {root.name!r} takes no index", lead)
    return segments


def validate(path: str) -> None:
    """Validate path syntax, raising PathSyntaxError on failure."""
    parse(path)


def is_valid(path: str) -> bool:
    """Return True if path parses successfully without raising exceptions.

    >>> is_valid("song.tracks[0].name")
    True
    >>> is_valid("song.tracks[-1]")
    False
    >>> is_valid("song.tracks[0:2]")
    False
    >>> is_valid("song.tracks[0].stop_all_clips()")
    False
    >>> is_valid("live.tracks[0]")
    False
    """
    try:
        parse(path)
    except PathSyntaxError:
        return False
    return True


def unparse(segments: list[Segment]) -> str:
    """Combine a list of Segment objects into a dot-separated path string.

    >>> unparse([Segment("song"), Segment("tracks", 2), Segment("name")])
    'song.tracks[2].name'
    """
    if not segments:
        raise ValueError("cannot build a path from an empty segment list")
    return ".".join(str(segment) for segment in segments)


def _parse_segment(raw: str, path: str, offset: int) -> Segment:
    """Parse a single dot-delimited segment string into a Segment instance."""
    match = _SEGMENT_RE.fullmatch(raw)
    if match is None:
        raise PathSyntaxError(path, _segment_reason(raw), offset)
    name, index = match.group(1), match.group(2)
    return Segment(name, None if index is None else int(index))


def _segment_reason(raw: str) -> str:
    """Diagnose the specific grammar reason why a segment failed to parse."""
    if raw == "":
        return "empty segment (a doubled, leading or trailing '.')"
    if "{" in raw or "}" in raw:
        return (
            "an unsubstituted {placeholder}; a path template must go through "
            "build() before it is sent"
        )
    if "(" in raw or ")" in raw:
        return (
            "method calls are not reachable by path; call the method with lom_call, "
            "and only if the script's allowlist carries it (docs/protocol.md §6)"
        )

    head, bracket, tail = raw.partition("[")
    if bracket:
        close = tail.find("]")
        if close == -1:
            return "unclosed '['"
        if close != len(tail) - 1:
            return f"unexpected text {tail[close + 1 :]!r} after ']'"
        inner = tail[:close]
        if ":" in inner:
            return "slices are not supported; address one element at a time"
        if inner.startswith("-"):
            return "negative indices are not supported; count from 0"
        if inner.strip() == "":
            return "empty index; expected a non-negative integer"
        if not inner.isdigit():
            return f"index {inner!r} is not a non-negative integer"
        if len(inner) > 1 and inner.startswith("0"):
            return f"index {inner!r} has a leading zero; write the canonical form"

    if not head:
        return "segment starts with '['; expected an attribute name first"
    if not _NAME_RE.fullmatch(head):
        if head != head.strip():
            return f"{head!r} has surrounding whitespace"
        if any(char in head for char in "+-*/% "):
            return f"{head!r} is not an attribute name; expressions are not supported in a path"
        return (
            f"{head!r} is not a valid attribute name (letters, digits and underscore, "
            "not starting with a digit)"
        )
    return f"{raw!r} does not fit segment = name | name '[' int ']'"


def _root_list() -> str:
    """Return formatted string of allowed roots for error output."""
    return ", ".join(repr(root) for root in ROOTS) + " (and 'song.view')"


# ------------------------------------------------------------------ assembling


def build(template: str, **args: object) -> str:
    """Substitute {placeholder} variables in template and validate resulting path.

    >>> build("song.tracks[{track}].mixer_device.volume", track=3)
    'song.tracks[3].mixer_device.volume'
    >>> build("song.tracks[{t}].devices[{d}].parameters[{p}]", t=0, d=1, p=2)
    'song.tracks[0].devices[1].parameters[2]'
    >>> build("song.tempo")
    'song.tempo'
    """
    wanted = list(dict.fromkeys(_PLACEHOLDER_RE.findall(template)))
    missing = [name for name in wanted if name not in args]
    unknown = sorted(name for name in args if name not in set(wanted))
    if missing or unknown:
        complaints = []
        if missing:
            complaints.append(f"missing argument(s) {missing}")
        if unknown:
            complaints.append(f"unknown argument(s) {unknown}")
        raise ValueError(
            f"template {template!r}: {' and '.join(complaints)}; "
            f"it takes {wanted if wanted else 'none'}"
        )

    path = _PLACEHOLDER_RE.sub(lambda m: _render(template, m.group(1), args[m.group(1)]), template)
    if "{" in path or "}" in path:
        raise PathSyntaxError(
            path,
            f"a brace survived substitution of template {template!r}; "
            "placeholders must be written {name}",
        )
    validate(path)
    return path


def _render(template: str, name: str, value: object) -> str:
    """Render placeholder value into string form."""
    if isinstance(value, str):
        return value
    return str(_index(f"template {template!r}: {name}", value))


def parent(path: str) -> str | None:
    """Return parent path one step up the hierarchy, or None if path is a root.

    >>> parent("song.tracks[3].devices[0].parameters[5]")
    'song.tracks[3].devices[0].parameters'
    >>> parent("song.tracks[3].devices[0].parameters")
    'song.tracks[3].devices[0]'
    >>> parent("song.tempo")
    'song'
    >>> parent("song") is None
    True
    """
    segments = parse(path)
    last = segments[-1]
    if last.is_indexed:
        segments[-1] = Segment(last.name)
    elif len(segments) == 1:
        return None
    else:
        segments.pop()
    return unparse(segments)


def join(path: str, segment: str | Segment) -> str:
    """Append a segment or dotted tail to a base path.

    >>> join("song.tracks[0]", "devices[2]")
    'song.tracks[0].devices[2]'
    >>> join("song.tracks[0]", "mixer_device.volume")
    'song.tracks[0].mixer_device.volume'
    """
    validate(path)
    tail = str(segment).strip()
    if not tail:
        raise PathSyntaxError(path, "cannot join an empty segment")

    offset = 0
    for raw in tail.split("."):
        parsed = _parse_segment(raw, tail, offset)
        if parsed.name in ROOTS:
            raise PathSyntaxError(
                tail,
                f"{parsed.name!r} is a root and cannot be joined onto {path!r}",
                offset,
            )
        offset += len(raw) + 1
    return f"{path.strip()}.{tail}"


# ------------------------------------------------------------------ common shapes


def track(index: int) -> str:
    """Return path for a regular track.

    >>> track(0)
    'song.tracks[0]'
    """
    return f"song.tracks[{_index('track: index', index)}]"


def return_track(index: int) -> str:
    """Return path for a return track.

    >>> return_track(1)
    'song.return_tracks[1]'
    """
    return f"song.return_tracks[{_index('return_track: index', index)}]"


def master() -> str:
    """Return path for the master track.

    >>> master()
    'song.master_track'
    """
    return "song.master_track"


def scene(index: int) -> str:
    """Return path for a scene.

    >>> scene(4)
    'song.scenes[4]'
    """
    return f"song.scenes[{_index('scene: index', index)}]"


def clip_slot(track: int | str, slot: int) -> str:
    """Return path for a clip slot on the specified track.

    >>> clip_slot(2, 0)
    'song.tracks[2].clip_slots[0]'
    """
    base = _track_base(track, caller="clip_slot")
    return f"{base}.clip_slots[{_index('clip_slot: slot', slot)}]"


def clip(track: int | str, slot: int) -> str:
    """Return path for a clip within a clip slot.

    >>> clip(2, 0)
    'song.tracks[2].clip_slots[0].clip'
    """
    return f"{clip_slot(track, slot)}.clip"


def mixer(track: int | str) -> str:
    """Return path for a track's mixer device.

    >>> mixer(3)
    'song.tracks[3].mixer_device'
    >>> mixer(master())
    'song.master_track.mixer_device'
    """
    return f"{_track_base(track, caller='mixer')}.mixer_device"


def send(track: int | str, index: int) -> str:
    """Return path for a send parameter on a track mixer.

    >>> send(0, 1)
    'song.tracks[0].mixer_device.sends[1]'
    """
    base = _track_base(track, caller="send")
    if base == "song.master_track":
        raise ValueError("send: the master track has no sends")
    return f"{base}.mixer_device.sends[{_index('send: index', index)}]"


def device(track: int | str, index: int) -> str:
    """Return path for a device on a track.

    >>> device(3, 0)
    'song.tracks[3].devices[0]'
    >>> device(return_track(0), 1)
    'song.return_tracks[0].devices[1]'
    """
    base = _track_base(track, caller="device")
    return f"{base}.devices[{_index('device: index', index)}]"


def parameter(track: int | str, device: int, index: int) -> str:
    """Return path for a parameter on a device.

    >>> parameter(3, 0, 5)
    'song.tracks[3].devices[0].parameters[5]'
    >>> parameter(master(), 0, 2)
    'song.master_track.devices[0].parameters[2]'
    """
    base = _track_base(track, caller="parameter")
    device_index = _index("parameter: device", device)
    param_index = _index("parameter: index", index)
    return f"{base}.devices[{device_index}].parameters[{param_index}]"


def _track_base(on: int | str, *, caller: str) -> str:
    """Resolve track identifier (index or path string) to canonical track path."""
    if isinstance(on, bool):
        raise TypeError(f"{caller}: track must be an index or a track path, not a bool")
    if isinstance(on, int):
        return f"song.tracks[{_index(f'{caller}: track', on)}]"
    if isinstance(on, str):
        validate(on)
        return on.strip()
    raise TypeError(f"{caller}: track must be an index or a track path, got {type(on).__name__}")


def _index(label: str, value: object) -> int:
    """Validate and return non-negative integer index."""
    if isinstance(value, bool):
        raise TypeError(f"{label}: {value!r} is a bool, not an index")
    if isinstance(value, float):
        if not value.is_integer():
            raise TypeError(f"{label}: {value!r} is not a whole number")
        value = int(value)
    if not isinstance(value, int):
        raise TypeError(f"{label}: {value!r} is not an integer index")
    if value < 0:
        raise ValueError(
            f"{label}: {value} is negative; the grammar has no negative indices "
            "(docs/protocol.md §6): count from 0"
        )
    return value


# ------------------------------------------------------------------ explaining

_SINGULARS: dict[str, str] = {
    "clip_slots": "clip slot",
    "cue_points": "locator",
    "drum_pads": "drum pad",
    "return_chains": "return chain",
    "return_tracks": "return track",
    "value_items": "value item",
    "visible_tracks": "visible track",
    "arrangement_clips": "arrangement clip",
}

_NOUNS: dict[str, str] = {
    "app": "the Live application",
    "browser": "the browser",
    "clip": "the clip",
    "groove_pool": "the groove pool",
    "master_track": "the master track",
    "mixer_device": "the mixer device",
    "song": "the song",
    "view": "the view",
}


def describe_path(path: str) -> str:
    """Generate human-readable English description of a LOM path.

    >>> describe_path("song.tracks[3].mixer_device.volume")
    "the song, track 4 (index 3), the mixer device, 'volume'"
    >>> describe_path("song.tracks[0].clip_slots[2].clip.warping")
    "the song, track 1 (index 0), clip slot 3 (index 2), the clip, 'warping'"
    >>> describe_path("song.tracks[0].devices[1].parameters[5]")
    'the song, track 1 (index 0), device 2 (index 1), parameter 6 (index 5)'
    >>> describe_path("song.tracks[-1]")
    "'song.tracks[-1]' is not a valid LOM path: negative indices are not supported; count from 0"
    """
    try:
        segments = parse(path)
    except PathSyntaxError as exc:
        return f"{path!r} is not a valid LOM path: {exc.reason}"
    return ", ".join(_describe_segment(segment) for segment in segments)


def _describe_segment(segment: Segment) -> str:
    """Format a single Segment into its descriptive clause."""
    if segment.index is None:
        noun = _NOUNS.get(segment.name)
        return noun if noun is not None else repr(segment.name)
    return f"{_singular(segment.name)} {segment.index + 1} (index {segment.index})"


def _singular(name: str) -> str:
    """Return singular noun form for collection name."""
    known = _SINGULARS.get(name)
    if known is not None:
        return known
    stem = name[:-1] if name.endswith("s") and not name.endswith("ss") else name
    return stem.replace("_", " ")
