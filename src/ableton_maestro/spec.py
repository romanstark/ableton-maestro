"""Path catalog data types, path builder, and pre-flight validation.

Defines PathSpec and related models describing addressable places in the Live Object Model,
along with client-side path construction and parameter validation.

Design principles:
- Declarative catalog data: Represents LOM properties, methods, value types, ranges,
  and enums as structured specifications.
- Client-side pre-flight checks: Validates parameter types, required arguments,
  enums, and numeric bounds before requests reach the transport.
- Exact grammar validation: Enforces the LOM path grammar defined in docs/protocol.md §6.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ableton_maestro.models import Access, Kind, PathStatus, Unit

__all__ = [
    "NAME_PLACEHOLDER_PATTERN",
    "PATH_PATTERN",
    "PLACEHOLDER_PATTERN",
    "ROOTS",
    "SEGMENT_PATTERN",
    "TEMPLATE_PATH_PATTERN",
    "TEMPLATE_SEGMENT_PATTERN",
    "ArgSpec",
    "ParamSpec",
    "PathSpec",
    "build_path",
    "name_placeholders_in",
    "placeholders_in",
    "require_access",
    "validate_path",
    "validate_value",
]

# --------------------------------------------------------------------------------
# Path grammar (docs/protocol.md §6):
#   segment = name | name "[" int "]"
#   path    = segment ("." segment)*
# --------------------------------------------------------------------------------

_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
_INDEX = r"(?:0|[1-9][0-9]*)"
_PLACEHOLDER = r"\{" + _NAME + r"\}"

#: Pattern for a concrete path segment (e.g. 'tracks' or 'tracks[3]').
SEGMENT_PATTERN = re.compile(rf"\A{_NAME}(?:\[{_INDEX}\])?\Z")

#: Pattern for a template segment (e.g. 'tracks[{track}]' or '{root}').
_TEMPLATE_NAME = rf"(?:{_NAME}|{_PLACEHOLDER})"
TEMPLATE_SEGMENT_PATTERN = re.compile(
    rf"\A{_TEMPLATE_NAME}(?:\[(?:{_INDEX}|{_PLACEHOLDER})\])?\Z"
)

#: Complete concrete LOM path pattern.
PATH_PATTERN = re.compile(rf"\A{_NAME}(?:\[{_INDEX}\])?(?:\.{_NAME}(?:\[{_INDEX}\])?)*\Z")

#: Complete catalog template path pattern containing placeholders.
TEMPLATE_PATH_PATTERN = re.compile(
    rf"\A{_TEMPLATE_NAME}(?:\[(?:{_INDEX}|{_PLACEHOLDER})\])?"
    rf"(?:\.{_TEMPLATE_NAME}(?:\[(?:{_INDEX}|{_PLACEHOLDER})\])?)*\Z"
)

#: Pattern matching '{name}' placeholders within a template path.
PLACEHOLDER_PATTERN = re.compile(r"\{(" + _NAME + r")\}")

#: Root object names supported by the resolver (docs/protocol.md §6).
ROOTS: frozenset[str] = frozenset({"song", "app"})

#: Pattern matching placeholders that occupy an entire segment name rather than an index.
NAME_PLACEHOLDER_PATTERN = re.compile(r"(?:\A|\.)\{(" + _NAME + r")\}")


def name_placeholders_in(path: str) -> set[str]:
    """Return the placeholders in path that name a segment.

    For example, ``root`` in ``app.browser.{root}``.
    """
    return set(NAME_PLACEHOLDER_PATTERN.findall(path))


def placeholders_in(path: str) -> list[str]:
    """Return all placeholder names occurring in path in order of appearance."""
    return PLACEHOLDER_PATTERN.findall(path)


def validate_path(
    path: str,
    *,
    allow_placeholders: bool = False,
    roots: frozenset[str] | None = ROOTS,
) -> None:
    """Validate a path string against the LOM grammar (docs/protocol.md §6).

    Args:
        path: Path string to validate.
        allow_placeholders: If True, template placeholders like '{track}' are permitted.
        roots: Allowed root identifiers (default 'song', 'app'). Set to None to skip root check.

    Raises:
        ValueError: If the path contains invalid syntax, empty segments, or unknown roots.
    """
    if not isinstance(path, str) or not path:
        raise ValueError(f"path must be a non-empty string, got {path!r}")

    whole = TEMPLATE_PATH_PATTERN if allow_placeholders else PATH_PATTERN
    if not whole.match(path):
        segment_pattern = TEMPLATE_SEGMENT_PATTERN if allow_placeholders else SEGMENT_PATTERN
        for segment in path.split("."):
            if segment_pattern.match(segment):
                continue
            fault = _segment_fault(segment, allow_placeholders)
            raise ValueError(
                f"bad path {path!r}: segment {segment!r} {fault}. "
                "The grammar is segment = name | name[int]"
                + (" | name[{placeholder}]" if allow_placeholders else "")
                + " (docs/protocol.md section 6)."
            )
        raise ValueError(
            f"bad path {path!r}: segments must be joined by single dots, with no "
            "leading or trailing dot (docs/protocol.md section 6)."
        )

    if roots is not None:
        head_segment = path.split(".", 1)[0]
        head = head_segment.split("[", 1)[0]
        if head not in roots:
            raise ValueError(
                f"bad path {path!r}: root {head!r} is not resolvable; the script resolves "
                f"from {sorted(roots)} only (song.view is a segment of song, not a root)."
            )
        if head_segment != head:
            raise ValueError(
                f"bad path {path!r}: the root {head!r} takes no index; it is a single "
                "object, not a collection (docs/protocol.md section 6)."
            )


def _segment_fault(segment: str, allow_placeholders: bool) -> str:
    """Diagnose the specific syntax fault in an invalid path segment."""
    if not segment:
        return "is empty"
    if ":" in segment:
        return "looks like a slice; the resolver walks one index at a time, never a range"
    if "[-" in segment:
        return "has a negative index; indices are bounds-checked from 0 upwards"
    if "(" in segment or ")" in segment:
        return "looks like a method call; methods go through lom_call and an allowlist"
    if "{" in segment and not allow_placeholders:
        return "still contains a placeholder; substitute it with build_path() before sending"
    if segment != segment.strip():
        return "has surrounding whitespace"
    if re.match(rf"\A{_NAME}\[0[0-9]", segment):
        return "has a leading zero in its index; write the canonical form"
    return "is not a name, optionally followed by [int]"


@dataclass
class ParamSpec:
    """Specification for a {placeholder} parameter in a catalog path template.

    Attributes:
        name: Placeholder identifier.
        kind: Expected type (Kind.INT for indices, Kind.STR or Kind.ENUM for segment names).
        required: True if parameter must be provided when constructing concrete path.
        default: Default numeric or string value if optional.
        enum: Permitted values list for Kind.ENUM parameters.
        range: Inclusive (min, max) range tuple for numeric parameters.
    """

    name: str
    kind: Kind = Kind.INT
    required: bool = True
    default: int | None = None
    enum: list[Any] | None = None
    range: tuple[float | None, float | None] | None = None


@dataclass
class ArgSpec:
    """Specification for a positional method argument on a call catalog row.

    Attributes:
        name: Argument name for diagnostic and documentation purposes.
        kind: Expected argument type.
        required: True if argument must be supplied.
        default: Default argument value if optional.
        enum: List of permitted values if applicable.
        range: Inclusive (min, max) bounds if numeric.
        lom_object: True if argument is passed as a Live Object path dictionary ({"__path__": ...}).
        doc: Human-readable argument description.
    """

    name: str
    kind: Kind = Kind.OBJECT
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None
    range: tuple[float | None, float | None] | None = None
    lom_object: bool = False
    doc: str = ""

    def __post_init__(self) -> None:
        self.kind = self.kind if isinstance(self.kind, Kind) else Kind(self.kind)
        self.range = _as_pair(self.range) if self.range is not None else None
        if not self.name:
            raise ValueError("ArgSpec: name must not be empty")


@dataclass
class PathSpec:
    """Declarative specification for an addressable place or method in the LOM.

    Attributes:
        id: Unique catalog identifier.
        path: Path template string with optional placeholders.
        access: Permitted access verbs (GET, SET, CALL, AUTOMATE, OBSERVE).
        kind: Data type of the property value.
        range: Inclusive (min, max) numeric range.
        enum: Permitted value list for discrete properties.
        unit: Physical unit or normalization type (Unit.NORMALIZED, Unit.DB, etc.).
        display: Display formatting hint.
        quantized: True if the value takes discrete steps, not a continuous range.
        method: Method name for callable operations.
        verify: Verification hook identifier (default 'read_back').
        destructive: True if operation modifies or deletes project structure.
        status: Verification state (UNTESTED, VERIFIED, BROKEN).
        doc: Documentation description.
        means: Dictionary mapping raw numeric/string return values to human descriptions.
        params: List of path template placeholder definitions.
        args: List of positional method argument definitions for call operations.

    Note: range is a sanity bound on the catalog side, not a safety check. The Remote
    Script bounds-checks every index against the actual collection length, which is the
    only number that is ever right; a range here cannot know how many tracks a set has.
    """

    id: str
    path: str
    access: list[Access] = field(default_factory=lambda: [Access.GET])
    kind: Kind = Kind.OBJECT
    range: tuple[float | None, float | None] | None = None
    enum: list[Any] | None = None
    unit: Unit = Unit.NORMALIZED
    display: str | None = None
    quantized: bool = False
    method: str | None = None
    verify: str = "read_back"
    destructive: bool = False
    status: PathStatus = PathStatus.UNTESTED
    doc: str = ""
    means: dict[str, str] = field(default_factory=dict)
    params: list[ParamSpec] = field(default_factory=list)
    args: list[ArgSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._normalise()
        self._check_consistency()

    def _normalise(self) -> None:
        self.access = [a if isinstance(a, Access) else Access(a) for a in self.access]
        self.kind = self.kind if isinstance(self.kind, Kind) else Kind(self.kind)
        self.unit = self.unit if isinstance(self.unit, Unit) else Unit(self.unit)
        self.status = (
            self.status if isinstance(self.status, PathStatus) else PathStatus(self.status)
        )
        self.range = _as_pair(self.range)
        if self.means:
            self.means = {means_key(key): value for key, value in self.means.items()}
        for param in self.params:
            param.kind = param.kind if isinstance(param.kind, Kind) else Kind(param.kind)
            param.range = _as_pair(param.range)

    def meaning_of(self, value: Any) -> str | None:
        """Return the documented description for value from the means mapping, or None."""
        if not self.means:
            return None
        return self.means.get(means_key(value))

    def _check_consistency(self) -> None:
        """Validate internal consistency of catalog row definitions."""
        if not self.id:
            raise ValueError("PathSpec: id must not be empty")
        for key, meaning in self.means.items():
            if not isinstance(meaning, str) or not meaning.strip():
                raise ValueError(
                    f"{self.id}: means[{key!r}] is empty. A value that needs decoding and "
                    "has no text is worse than no entry, because the field being present "
                    "reads as an answer."
                )
            if len(meaning) > MEANS_CAP:
                raise ValueError(
                    f"{self.id}: means[{key!r}] is {len(meaning)} characters, over the "
                    f"{MEANS_CAP} cap. One sentence: it is attached to a value, not a "
                    "place to explain the property -- that is what doc is for."
                )
        if not self.access:
            raise ValueError(f"{self.id}: access must list at least one Access")

        validate_path(self.path, allow_placeholders=True)

        in_path = set(placeholders_in(self.path))
        declared = {p.name for p in self.params}
        if len(declared) != len(self.params):
            raise ValueError(f"{self.id}: duplicate parameter name(s) in params")
        missing = sorted(in_path - declared)
        if missing:
            raise ValueError(
                f"{self.id}: path uses placeholder(s) {missing} that no ParamSpec declares; "
                "build_path() would have nothing to substitute."
            )
        unused = sorted(declared - in_path)
        if unused:
            hint = (
                " Method arguments belong in 'args', not 'params' -- they are handed to "
                "lom_call, they never appear in the path."
                if Access.CALL in self.access
                else ""
            )
            raise ValueError(
                f"{self.id}: parameter(s) {unused} do not appear in path {self.path!r}; "
                "an argument that lands nowhere is the failure this catalog exists to "
                f"prevent.{hint}"
            )

        if self.args and Access.CALL not in self.access:
            names = sorted(a.name for a in self.args)
            raise ValueError(
                f"{self.id}: declares args {names} but access does not include 'call'. "
                "Arguments are only ever sent by lom_call; on a get/set row they would "
                "be silently dropped."
            )
        arg_names = [a.name for a in self.args]
        if len(set(arg_names)) != len(arg_names):
            raise ValueError(f"{self.id}: duplicate argument name(s) in args")
        overlap = sorted(set(arg_names) & declared)
        if overlap:
            raise ValueError(
                f"{self.id}: {overlap} declared as both a path parameter and a method "
                "argument. One name, two destinations, is exactly the ambiguity that "
                "sends a value to the wrong place."
            )
        seen_optional = False
        for arg in self.args:
            if arg.required and seen_optional:
                raise ValueError(
                    f"{self.id}: required argument {arg.name!r} follows an optional one. "
                    "args is positional (the LOM takes no keywords), so an optional "
                    "argument before a required one cannot be filled."
                )
            seen_optional = seen_optional or not arg.required
            _check_range_pair(self.id, f"argument {arg.name!r}", arg.range)

        by_name = name_placeholders_in(self.path)
        for param in self.params:
            if param.name in by_name:
                if param.kind not in (Kind.STR, Kind.ENUM):
                    raise ValueError(
                        f"{self.id}: parameter {param.name!r} stands for a segment name in "
                        f"{self.path!r}, so its kind must be 'str' or 'enum', not "
                        f"{param.kind.value!r}. The substituted value becomes an attribute "
                        "name (docs/protocol.md section 6)."
                    )
            elif param.kind is not Kind.INT:
                raise ValueError(
                    f"{self.id}: parameter {param.name!r} stands for an index in "
                    f"{self.path!r}, so its kind must be 'int', not {param.kind.value!r}. "
                    "An index is always a whole number (docs/protocol.md section 6)."
                )
            if not param.required and param.default is None:
                raise ValueError(
                    f"{self.id}: parameter {param.name!r} is optional but has no default; "
                    "every placeholder must end up filled or the path cannot be built."
                )
            _check_range_pair(self.id, f"parameter {param.name!r}", param.range)

        if Access.CALL in self.access and not self.method:
            raise ValueError(f"{self.id}: access includes 'call' but no method is named")
        if self.method and Access.CALL not in self.access:
            raise ValueError(
                f"{self.id}: names method {self.method!r} but access does not include 'call'"
            )
        if self.kind is Kind.ENUM and not self.enum:
            raise ValueError(f"{self.id}: kind is 'enum' but no enum values are listed")
        _check_range_pair(self.id, "range", self.range)

    def supports(self, access: Access) -> bool:
        """Return True if the catalog permits access on this path."""
        return access in self.access


#: Maximum character length for a single means entry.
MEANS_CAP = 240


def means_key(value: Any) -> str:
    """Return the canonical string key for looking up a value in the means dictionary.

    >>> means_key(-1), means_key(-1.0), means_key("-1"), means_key("-1.0")
    ('-1', '-1', '-1', '-1')
    >>> means_key(0.25), means_key("0.25")
    ('0.25', '0.25')
    >>> means_key(True), means_key(False)
    ('true', 'false')
    >>> means_key("q_bar")
    'q_bar'
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else repr(value)
    text = str(value)
    try:
        return means_key(int(text))
    except ValueError:
        pass
    try:
        return means_key(float(text))
    except ValueError:
        return text


def _as_pair(bounds: Any) -> Any:
    """Convert a two-element (min, max) sequence into a tuple."""
    if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
        return (bounds[0], bounds[1])
    return bounds


def _check_range_pair(spec_id: str, label: str, bounds: Any) -> None:
    """Validate a (min, max) pair for type, finiteness, and low <= high order."""
    if bounds is None:
        return
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise ValueError(f"{spec_id}: {label} must be a (min, max) pair, got {bounds!r}")
    low, high = bounds
    for bound in (low, high):
        if bound is None:
            continue
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            raise ValueError(  # noqa: TRY004
                f"{spec_id}: {label} bound {bound!r} is not a number"
            )
        if not math.isfinite(float(bound)):
            raise ValueError(f"{spec_id}: {label} bound {bound!r} is not finite")
    if low is not None and high is not None and float(low) > float(high):
        raise ValueError(f"{spec_id}: {label} is inverted: min {low} is above max {high}")


def build_path(spec: PathSpec, **args: Any) -> str:
    """Substitute args into spec's template path and validate concrete path.

    Example:
        build_path(track_volume, track=3) -> "song.tracks[3].mixer_device.volume"

    Raises:
        ValueError: If required parameters are missing, unknown parameters are passed,
            or values violate defined ranges or enums.
    """
    declared = {p.name: p for p in spec.params}
    unknown = sorted(k for k in args if k not in declared)
    if unknown:
        valid = sorted(declared)
        raise ValueError(
            f"{spec.id}: unknown argument(s) {unknown}; "
            f"valid parameter(s): {valid if valid else 'none'}."
        )

    path = spec.path
    by_name = name_placeholders_in(spec.path)
    for name, param in declared.items():
        raw = args.get(name, param.default)
        if raw is None:
            raise ValueError(
                f"{spec.id}: missing required parameter {name!r} for path {spec.path!r}"
            )
        if name in by_name:
            filled = _coerce_segment_name(spec.id, param, raw)
        else:
            filled = str(_coerce_index(spec.id, param, raw))
        path = path.replace("{" + name + "}", filled)

    validate_path(path)
    return path


def _coerce_segment_name(spec_id: str, param: ParamSpec, raw: Any) -> str:
    """Validate segment placeholder value against identifier syntax and enums."""
    if isinstance(raw, str):
        value = raw
    else:
        raise ValueError(  # noqa: TRY004
            f"{spec_id}: parameter {param.name!r} names a path segment, so it must be a "
            f"string, got {type(raw).__name__} {raw!r}."
        )
    if not re.match(rf"\A{_NAME}\Z", value):
        raise ValueError(
            f"{spec_id}: parameter {param.name!r} = {value!r} is not a legal segment name; "
            "it must look like a Python identifier (docs/protocol.md section 6)."
        )
    if param.enum is not None and value not in param.enum:
        raise ValueError(
            f"{spec_id}: parameter {param.name!r} = {value!r} is not one of "
            f"{sorted(map(str, param.enum))}."
        )
    return value


def _coerce_index(spec_id: str, param: ParamSpec, raw: Any) -> int:
    """Validate and coerce a path index placeholder to a non-negative integer."""
    if isinstance(raw, Enum):
        raw = raw.value
    if isinstance(raw, bool):
        raise ValueError(  # noqa: TRY004
            f"{spec_id}: {param.name}={raw!r} is a bool. Python's bool is a subclass of "
            "int, so this would quietly become an index of 0 or 1."
        )
    if isinstance(raw, int):
        index = raw
    elif isinstance(raw, float):
        if not math.isfinite(raw) or not raw.is_integer():
            raise ValueError(
                f"{spec_id}: {param.name}={raw!r} is not a whole number; an index cannot "
                "be fractional (docs/protocol.md section 6)."
            )
        index = int(raw)
    else:
        raise ValueError(  # noqa: TRY004
            f"{spec_id}: {param.name}={raw!r} is a {type(raw).__name__}, not an int. "
            "Path indices are integers; a string index would produce a path that only "
            "fails once it is inside Live."
        )

    if index < 0:
        raise ValueError(
            f"{spec_id}: {param.name}={index} is negative. The grammar has no negative "
            "indices; there is no counting from the end (docs/protocol.md section 6)."
        )
    if param.enum is not None and index not in param.enum:
        raise ValueError(
            f"{spec_id}: {param.name}={index} is not one of the allowed values {param.enum}"
        )
    _check_bounds(spec_id, param.name, index, param.range)
    return index


def validate_value(spec: PathSpec, value: Any) -> Any:
    """Validate and coerce a value against spec definition prior to sending to Live.

    Args:
        spec: PathSpec definition for target property.
        value: Value to write.

    Returns:
        Coerced value ready for wire encoding.

    Raises:
        ValueError: If value type, range, or enum validation fails.
    """
    if isinstance(value, Enum):
        value = value.value

    coerced = _coerce_value(spec.id, spec.kind, value)

    if spec.enum is not None and coerced not in spec.enum:
        raise ValueError(
            f"{spec.id}: {coerced!r} is not one of the allowed values {spec.enum}"
        )
    if spec.kind in (Kind.INT, Kind.FLOAT):
        _check_bounds(spec.id, "value", coerced, spec.range)
    return coerced


def _coerce_value(spec_id: str, kind: Kind, value: Any) -> Any:
    """Coerce value to the target Kind enum type."""
    if kind is Kind.BOOL:
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        raise ValueError(
            f"{spec_id}: expected a bool, got {value!r}. Strings are refused on purpose: "
            'every non-empty string is truthy, so "false" would switch this on.'
        )

    if kind is Kind.INT:
        if isinstance(value, bool):
            raise ValueError(f"{spec_id}: expected an int, got the bool {value!r}")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value)
        raise ValueError(f"{spec_id}: expected an int, got {value!r}")

    if kind is Kind.FLOAT:
        if isinstance(value, bool):
            raise ValueError(f"{spec_id}: expected a number, got the bool {value!r}")
        if isinstance(value, (int, float)):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(
                    f"{spec_id}: {value!r} is not finite. NaN compares False against every "
                    "bound, so it would pass a range check and reach Live unnoticed."
                )
            return number
        raise ValueError(f"{spec_id}: expected a number, got {value!r}")

    if kind is Kind.STR:
        if isinstance(value, str):
            return value
        raise ValueError(f"{spec_id}: expected a string, got {value!r}")

    if kind is Kind.ENUM:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(
                f"{spec_id}: an enum value must be an int or a string, got {value!r}"
            )
        return value

    if kind is Kind.LIST:
        if isinstance(value, (list, tuple)):
            return list(value)
        raise ValueError(f"{spec_id}: expected a list, got {value!r}")

    if isinstance(value, dict):
        return value
    raise ValueError(
        f"{spec_id}: expected a LOM handle object, got {value!r}. An object-valued path "
        "takes a handle such as {'__lom__': 'Track', 'path': 'song.tracks[3]'}."
    )


def _check_bounds(spec_id: str, label: str, value: Any, bounds: Any) -> None:
    """Validate numeric value against an inclusive (min, max) tuple."""
    if bounds is None:
        return
    low, high = bounds
    if low is not None and value < low:
        raise ValueError(
            f"{spec_id}: {label}={value!r} is below the allowed minimum {low}"
        )
    if high is not None and value > high:
        raise ValueError(
            f"{spec_id}: {label}={value!r} is above the allowed maximum {high}"
        )


def require_access(spec: PathSpec, access: Access) -> None:
    """Raise ValueError if the catalog row does not grant the requested Access mode."""
    if access not in spec.access:
        raise ValueError(
            f"{spec.id}: catalog does not permit {access.value!r} on this path; "
            f"it allows {[a.value for a in spec.access]}"
        )
