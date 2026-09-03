"""Typed argument models and schema flattening for nested MCP tool parameters.

Defines Pydantic models (NoteIn, BatchOp) and custom schema transformations to inline
nested object definitions into MCP tool input schemas. Inlining preserves field names and
types without requiring clients to resolve $ref pointers.

Nothing in this module talks to Live, so nothing here is dated against a Live version.
The contract it upholds is a client-side one: a tool schema is published fully inlined,
because a client that does not dereference ``$ref`` sees an untyped object where a typed
note or batch operation was declared. ``tests/test_tool_schema.py`` holds that contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    WithJsonSchema,
    field_validator,
    model_validator,
)

__all__ = [
    "BatchOp",
    "BatchOpArg",
    "NoteArg",
    "NoteIn",
    "batch_ops",
    "inline_schema_refs",
    "note_dicts",
]


class NoteIn(BaseModel):
    """MIDI note input model with clip-local beat timing.

    Required fields: pitch, start_time, duration.
    Optional fields: velocity, mute, probability, velocity_deviation, release_velocity.
    Extra keys pass through to allow validate_note_dicts to provide diagnostic messages.
    """

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "description": (
                "One MIDI note. Times are in beats, clip-local, where 0 is the "
                "clip start. Unknown keys are reported by name and refused."
            ),
            # extra="allow" allows validate_note_dicts to identify and diagnose unknown keys.
            # The schema contract specifies additionalProperties=False.
            "additionalProperties": False,
        },
    )

    pitch: int = Field(description="MIDI note number, 0..127. An integer, not a float.")
    start_time: float = Field(description="Onset in beats from the clip's start.")
    duration: float = Field(description="Length in beats. Must be greater than 0.")
    velocity: float | None = Field(
        default=None,
        description="1..127. Omitted or null both mean unspecified; Live default is 100.",
    )
    mute: bool | None = Field(default=None, description="True mutes this note.")
    probability: float | None = Field(
        default=None, description="Trigger chance 0..1 (Live 11+)."
    )
    velocity_deviation: float | None = Field(
        default=None, description="Per-hit velocity randomisation, -127..127 (Live 11+)."
    )
    release_velocity: float | None = Field(
        default=None, description="Note-off velocity 0..127 (Live 11+)."
    )

    @model_validator(mode="before")
    @classmethod
    def _an_explicit_null_reads_as_not_given(cls, data: Any) -> Any:
        """Treat an optional key sent as null as if it had been omitted.

        Ensures that optional fields explicitly passed as null inherit default
        values (e.g. default velocity 100) instead of failing numeric validation.
        Required fields remain validated (e.g. null pitch is rejected).
        """
        if not isinstance(data, dict):
            return data
        optional = {
            name for name, spec in cls.model_fields.items() if not spec.is_required()
        }
        if not any(value is None and key in optional for key, value in data.items()):
            return data
        return {
            key: value
            for key, value in data.items()
            if not (value is None and key in optional)
        }

    @field_validator(
        "pitch", "start_time", "duration", "velocity",
        "probability", "velocity_deviation", "release_velocity",
        mode="before",
    )
    @classmethod
    def _a_boolean_is_not_a_number(cls, value: Any) -> Any:
        """Reject boolean values where numeric values are required.

        Pydantic lax mode coerces True to 1 and False to 0. This validator
        ensures explicit numeric types are provided.
        """
        if isinstance(value, bool):
            raise ValueError(  # noqa: TRY004 - pydantic converts only ValueError and AssertionError
                "a boolean is not a number here; send a number"
            )
        return value


class BatchOp(BaseModel):
    """Single operation inside a lom_batch call.

    Requires op ('get', 'set', or 'call') and target LOM path.
    Additional keys are forbidden to prevent silent execution of malformed payloads.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "description": (
                "One raw LOM operation. Only valid operation keys are permitted."
            )
        },
    )

    op: Literal["get", "set", "call"] = Field(
        description="Read a property, write one, or call an allowlisted method."
    )
    path: str = Field(description="LOM path, e.g. 'song.tracks[0].name'.")
    value: Any = Field(
        default=None,
        description=(
            "Required for op='set'. May be {'__path__': '<path>'} to pass an object reference."
        ),
    )
    method: str | None = Field(
        default=None,
        description="Required for op='call'. Must be on the Remote Script allowlist.",
    )
    args: list[Any] | None = Field(
        default=None,
        description="Positional arguments for op='call'. Keyword arguments are not supported.",
    )


def inline_schema_refs(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Replace local $ref schema pointers with inline definitions.

    Flattens $ref definitions from $defs directly into the schema tree so that
    MCP clients receive fully qualified parameter definitions without requiring
    dereferencing.

    Any ``$ref`` that cannot be resolved locally is left in place, and ``$defs`` is kept
    alongside the tree when that happens, so an unresolvable pointer degrades to the
    original schema rather than to a broken one.

    Note:
        Not a Live measurement. This is a client-side property of the MCP schema, and
        ``tests/test_tool_schema.py`` asserts that no ``$ref`` or ``$defs`` survives in
        any published tool schema.
    """
    defs = schema.get("$defs") or {}
    if not defs:
        return dict(schema)

    def walk(node: Any, seen: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [walk(item, seen) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref[len("#/$defs/"):]
            if name in seen or name not in defs:
                return dict(node)
            target = walk(defs[name], seen | {name})
            beside = {key: value for key, value in node.items() if key != "$ref"}
            return {**target, **beside}
        return {key: walk(value, seen) for key, value in node.items()}

    stripped = {key: value for key, value in schema.items() if key != "$defs"}
    inlined = walk(stripped, frozenset())
    if _carries_a_ref(inlined):
        inlined["$defs"] = defs
    return inlined


def _carries_a_ref(node: Any) -> bool:
    """Return True if any node within the schema tree contains a $ref pointer."""
    if isinstance(node, list):
        return any(_carries_a_ref(item) for item in node)
    if not isinstance(node, dict):
        return False
    if "$ref" in node:
        return True
    return any(_carries_a_ref(value) for value in node.values())


#: Note argument schema annotation: validated by NoteIn with inlined JSON schema.
NoteArg = Annotated[NoteIn, WithJsonSchema(inline_schema_refs(NoteIn.model_json_schema()))]

#: Batch operation schema annotation: validated by BatchOp with inlined JSON schema.
BatchOpArg = Annotated[
    BatchOp, WithJsonSchema(inline_schema_refs(BatchOp.model_json_schema()))
]


def note_dicts(notes: Sequence[NoteIn | Mapping[str, Any] | Any]) -> list[Any]:
    """Convert NoteIn models or mappings to raw dictionaries.

    Preserves unset optional fields so downstream validation can distinguish
    between omitted keys and explicitly provided null values.
    """
    out: list[Any] = []
    for entry in notes:
        if isinstance(entry, NoteIn):
            out.append(entry.model_dump(exclude_unset=True))
        elif isinstance(entry, Mapping):
            out.append(dict(entry))
        else:
            out.append(entry)
    return out


def batch_ops(ops: Sequence[BatchOp | Mapping[str, Any] | Any]) -> list[Any]:
    """Convert BatchOp models or mappings to raw dictionaries.

    Preserves unset keys so that a set operation without a value remains
    distinguishable from one that explicitly writes null.
    """
    out: list[Any] = []
    for op in ops:
        if isinstance(op, BatchOp):
            out.append(op.model_dump(exclude_unset=True))
        elif isinstance(op, Mapping):
            out.append(dict(op))
        else:
            out.append(op)
    return out
