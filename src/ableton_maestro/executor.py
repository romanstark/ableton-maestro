"""Central execution engine for catalog-driven operations.

Coordinates path resolution, validation, safety guards, dispatch, and verification
for operations defined in the LOM catalog.

Design principles:
- Single execution pipeline: Validates arguments, checks destructive confirmations,
  and verifies access permissions before dispatching commands over the transport.
- Read-back verification: Evaluates returned values against requested parameters to
  confirm whether values were applied as requested, clamped, or deferred.
- Structured results: Maps all transport outcomes into a consistent Result dataclass
  containing status, before/after values, and verification flags.
"""

from __future__ import annotations

import difflib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ableton_maestro.models import Access
from ableton_maestro.spec import (
    ArgSpec,
    PathSpec,
    build_path,
    validate_path,
    validate_value,
)

if TYPE_CHECKING:
    from ableton_maestro.registry import Registry

# Reserved argument names for operation payloads (distinguished from path placeholders).
ARG_VALUE = "value"
ARG_CALL_ARGS = "call_args"
_RESERVED = frozenset({ARG_VALUE, ARG_CALL_ARGS})

# Client-side refusal codes for operations blocked prior to dispatch.
CODE_CONFIRM_REQUIRED = "confirm_required"
CODE_ACCESS_NOT_ALLOWED = "access_not_allowed"
CODE_BAD_REQUEST = "bad_request"
CODE_NO_RESULT = "no_result"

# Default verifier for 'set' operations (evaluates the read-back payload from lom_set).
DEFAULT_SET_VERIFY = "read_back"

# Catalog identifier indicating that no post-execution verification is available for a path.
VERIFY_NONE = "none"

# Float comparison tolerances for read-back verification against JSON-serialized numbers.
_REL_TOL = 1e-6
_ABS_TOL = 1e-9

# Start offset in beats for automation sampling (sampled > 0 to bypass Live's default t=0 value).
_VERIFY_START_BEATS = 0.05


@runtime_checkable
class LomClient(Protocol):
    """Transport protocol interface required by the execution engine."""

    def get(self, path: str) -> Mapping[str, Any]:
        """Read the property at the given LOM path (lom_get)."""

    def set(self, path: str, value: object) -> Mapping[str, Any]:
        """Write a value to the given LOM path and read it back (lom_set)."""

    def call(self, path: str, method: str, args: Sequence[object]) -> Mapping[str, Any]:
        """Invoke an allowlisted method on the object at the given LOM path (lom_call)."""

    def batch(
        self, ops: Sequence[Mapping[str, Any]], *, atomic: bool = False
    ) -> Mapping[str, Any]:
        """Execute multiple operations in a single round trip (lom_batch)."""


class UnknownSpecError(KeyError):
    """Raised when a requested catalog spec ID does not exist."""

    def __init__(self, message: str) -> None:
        super().__init__(message)

    def __str__(self) -> str:
        return str(self.args[0])


@dataclass
class Result:
    """Structured outcome of an operation executed through the catalog engine.

    Attributes:
        ok: True if the operation succeeded on the transport and Live side.
        id: Catalog entry identifier.
        path: Concrete LOM path that was addressed.
        code: Structured error or refusal code if the operation failed.
        message: Human-readable description of the error or status.
        value: The requested or read value.
        before: Previous value before a write operation.
        after: Resulting value read back from Live after a write.
        clamped: True if Live coerced or clamped the value during a write.
        changed: True if the value differed from its previous state.
        display: Human-readable display string formatted by Live if available.
        read_back: Raw read-back verdict from the script ('applied', 'clamped', 'not_observed').
        verified: True if post-condition verifier passed, False if failed, None if unverified.
        blocked: True if the operation was rejected locally before dispatch.
    """

    ok: bool
    id: str
    path: str
    code: str | None = None
    message: str | None = None
    value: object = None
    before: object = None
    after: object = None
    clamped: bool | None = None
    changed: bool | None = None
    display: str | None = None
    read_back: str | None = None
    verified: bool | None = None
    blocked: bool = False

    @classmethod
    def from_response(cls, resp: Mapping[str, Any], spec_id: str, path: str) -> Result:
        """Construct a Result instance from a wire protocol response dictionary."""
        ok, payload, code, message = _split_response(resp)
        echoed = payload.get("path") or resp.get("path")
        return cls(
            ok=ok,
            id=spec_id,
            path=str(echoed) if echoed else path,
            code=code,
            message=message,
            value=_first(payload, "value", "requested", "result"),
            before=payload.get("before"),
            after=payload.get("after"),
            clamped=_opt_bool(payload.get("clamped")),
            changed=_opt_bool(payload.get("changed")),
            display=_opt_str(payload.get("display")),
            read_back=_opt_str(payload.get("read_back")),
        )

    @classmethod
    def blocked_(cls, spec: PathSpec, reason: str, code: str) -> Result:
        """Construct a Result representing a client-side blocked operation."""
        return cls(ok=False, id=spec.id, path=spec.path, code=code, message=reason, blocked=True)


@dataclass
class _Prepared:
    """Pre-computed operation parameters ready for transport dispatch."""

    spec: PathSpec
    op: Access
    path: str
    value: object = None
    call_args: list[object] = field(default_factory=list)


def execute(
    client: LomClient,
    registry: Registry,
    spec_id: str,
    *,
    confirm: bool = False,
    verify: bool = True,
    **args: object,
) -> Result:
    """Execute a single catalog operation and return a structured Result.

    Args:
        client: Transport client implementing the LomClient protocol.
        registry: Spec registry containing catalog definitions.
        spec_id: Unique catalog entry ID (e.g. 'song.tempo', 'track.volume').
        confirm: Must be True to execute operations marked as destructive.
        verify: If True, executes the post-condition verifier defined in the spec.
        **args: Path parameters and payload arguments ('value' or 'call_args').

    Returns:
        A Result instance detailing execution status, returned values, and verification.

    Raises:
        UnknownSpecError: If spec_id is not found in the registry.
        ValueError: If arguments fail type or range validation.
    """
    spec = _lookup(registry, spec_id)
    op = _operation(spec, args)

    if spec.destructive and not confirm:
        return Result.blocked_(
            spec,
            f"{spec.id} is marked destructive; pass confirm=True to run it",
            CODE_CONFIRM_REQUIRED,
        )
    if not spec.supports(op):
        granted = ", ".join(a.value for a in spec.access) or "nothing"
        return Result.blocked_(
            spec,
            f"{spec.id} does not allow {op.value}; the catalog grants: {granted}",
            CODE_ACCESS_NOT_ALLOWED,
        )

    prepared = _prepare(spec, op, args)
    result = _dispatch(client, prepared)
    if verify:
        _run_verifier(client, prepared, args, result)
    return result


def execute_batch(
    client: LomClient,
    registry: Registry,
    ops: Sequence[Mapping[str, Any]],
    *,
    atomic: bool = False,
    verify: bool = False,
) -> list[Result]:
    """Execute multiple catalog operations in a single round-trip exchange.

    Args:
        client: Transport client implementing the LomClient protocol.
        registry: Spec registry containing catalog definitions.
        ops: Sequence of operation dictionaries, each containing 'id' and parameters.
        atomic: If True, execution halts upon the first error (note: LOM has no rollback).
        verify: If True, runs post-condition verifiers (may require additional round trips).

    Returns:
        List of Result instances corresponding to the input operations.
    """
    results: list[Result | None] = [None] * len(ops)
    prepared_by_index: dict[int, _Prepared] = {}
    wire_ops: list[Mapping[str, Any]] = []
    wire_index: list[int] = []

    for i, op_spec in enumerate(ops):
        outcome = _prepare_batch_op(registry, op_spec)
        if isinstance(outcome, Result):
            results[i] = outcome
            continue
        prepared_by_index[i] = outcome
        wire_ops.append(_wire_op(outcome))
        wire_index.append(i)

    if wire_ops:
        replies = _batch_replies(client, wire_ops, atomic=atomic)
        for slot, index in enumerate(wire_index):
            prepared = prepared_by_index[index]
            if slot < len(replies):
                results[index] = _result_for(prepared, replies[slot])
            else:
                results[index] = Result(
                    ok=False,
                    id=prepared.spec.id,
                    path=prepared.path,
                    code=CODE_NO_RESULT,
                    message=(
                        f"lom_batch returned {len(replies)} result(s) for {len(wire_ops)} op(s); "
                        "this op has no reply and its outcome is unknown"
                    ),
                )

    if verify:
        for index, prepared in prepared_by_index.items():
            result = results[index]
            if result is not None:
                _run_verifier(client, prepared, dict(ops[index]), result)

    return [r for r in results if r is not None]


# ------------------------------------------------------------------- internals


def _lookup(registry: Registry, spec_id: str) -> PathSpec:
    """Retrieve PathSpec for spec_id, or raise UnknownSpecError with nearest matches."""
    try:
        return registry.get(spec_id)
    except KeyError:
        known = [s.id for s in registry.all()]
        near = difflib.get_close_matches(spec_id, known, n=5, cutoff=0.5)
        hint = f"did you mean: {', '.join(near)}?" if near else "no similar id in the catalog"
        raise UnknownSpecError(f"unknown catalog id {spec_id!r}; {hint}") from None


def _operation(spec: PathSpec, args: Mapping[str, object]) -> Access:
    """Infer Access operation (GET, SET, or CALL) from spec and arguments."""
    if spec.method:
        if ARG_VALUE in args:
            raise ValueError(
                f"{spec.id}: row declares method {spec.method!r}, so it is a call; "
                f"pass {ARG_CALL_ARGS}=[...] instead of {ARG_VALUE}="
            )
        return Access.CALL
    if ARG_CALL_ARGS in args:
        raise ValueError(
            f"{spec.id}: row declares no method, so {ARG_CALL_ARGS} means nothing here"
        )
    return Access.SET if ARG_VALUE in args else Access.GET


def _prepare(spec: PathSpec, op: Access, args: Mapping[str, object]) -> _Prepared:
    """Validate parameters and build concrete LOM path prior to dispatch."""
    path_args = {k: v for k, v in args.items() if k not in _RESERVED}
    path = build_path(spec, **path_args)
    value = validate_value(spec, args[ARG_VALUE]) if op is Access.SET else None
    call_args = (
        _validate_call_args(spec, _as_args(args.get(ARG_CALL_ARGS)))
        if op is Access.CALL
        else []
    )
    return _Prepared(spec=spec, op=op, path=path, value=value, call_args=call_args)


def _validate_call_args(spec: PathSpec, given: list[object]) -> list[object]:
    """Validate positional method arguments against declared ArgSpec definitions."""
    if not spec.args:
        return given

    required = sum(1 for a in spec.args if a.required)
    if len(given) > len(spec.args) or len(given) < required:
        shape = ", ".join(a.name if a.required else f"[{a.name}]" for a in spec.args)
        raise ValueError(
            f"{spec.id}: {spec.method}() takes {shape or 'no arguments'} "
            f"({required} required, {len(spec.args)} total), got {len(given)}."
        )

    out: list[object] = []
    for index, arg in enumerate(spec.args):
        raw = given[index] if index < len(given) else arg.default
        if raw is None and arg.required:
            raise ValueError(f"{spec.id}: argument {arg.name!r} is required")
        out.append(_coerce_arg(spec, arg, raw))
    return out


def _coerce_arg(spec: PathSpec, arg: ArgSpec, raw: object) -> object:
    """Validate a single method argument for type, enum values, and numeric ranges."""
    if arg.lom_object:
        if isinstance(raw, Mapping) and "__path__" in raw:
            target = raw["__path__"]
        else:
            target = raw
        if not isinstance(target, str):
            raise ValueError(
                f"{spec.id}: argument {arg.name!r} is a Live object, so it travels as a "
                f"path string; got {type(target).__name__}. See docs/protocol.md §5.5."
            )
        validate_path(target)
        return {"__path__": target}

    if arg.enum is not None and raw not in arg.enum:
        raise ValueError(
            f"{spec.id}: argument {arg.name!r} = {raw!r} is not one of {sorted(map(str, arg.enum))}"
        )
    if arg.range is not None and isinstance(raw, (int, float)) and not isinstance(raw, bool):
        low, high = arg.range
        if (low is not None and raw < low) or (high is not None and raw > high):
            raise ValueError(
                f"{spec.id}: argument {arg.name!r} = {raw!r} is outside {arg.range}"
            )
    return raw


def _as_args(raw: object) -> list[object]:
    """Normalize positional call arguments to a list."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw]


def _dispatch(client: LomClient, prepared: _Prepared) -> Result:
    """Dispatch a prepared operation over the client transport and return a Result."""
    spec = prepared.spec
    try:
        if prepared.op is Access.GET:
            resp = client.get(prepared.path)
        elif prepared.op is Access.SET:
            resp = client.set(prepared.path, prepared.value)
        else:
            resp = client.call(prepared.path, str(spec.method), prepared.call_args)
    except Exception as exc:
        code = getattr(exc, "code", None)
        if code is None:
            raise
        return Result(
            ok=False,
            id=spec.id,
            path=prepared.path,
            code=str(code),
            message=str(getattr(exc, "message", None) or exc),
        )
    return _result_for(prepared, resp)


def _result_for(prepared: _Prepared, resp: Mapping[str, Any]) -> Result:
    """Construct a Result from transport reply, backfilling requested value."""
    result = Result.from_response(resp, prepared.spec.id, prepared.path)
    if prepared.op is Access.SET and result.value is None:
        result.value = prepared.value
    return result


def _prepare_batch_op(registry: Registry, op_spec: Mapping[str, Any]) -> _Prepared | Result:
    """Validate and prepare a single batch operation."""
    spec_id = op_spec.get("id")
    if not isinstance(spec_id, str) or not spec_id:
        raise ValueError(f"batch op is missing a string 'id': {op_spec!r}")
    spec = _lookup(registry, spec_id)
    confirm = bool(op_spec.get("confirm", False))
    args = {k: v for k, v in op_spec.items() if k not in ("id", "confirm")}

    try:
        op = _operation(spec, args)
    except ValueError as exc:
        return Result.blocked_(spec, str(exc), CODE_BAD_REQUEST)

    if spec.destructive and not confirm:
        return Result.blocked_(
            spec,
            f"{spec.id} is marked destructive; pass confirm=True on the op to run it",
            CODE_CONFIRM_REQUIRED,
        )
    if not spec.supports(op):
        granted = ", ".join(a.value for a in spec.access) or "nothing"
        return Result.blocked_(
            spec,
            f"{spec.id} does not allow {op.value}; the catalog grants: {granted}",
            CODE_ACCESS_NOT_ALLOWED,
        )
    try:
        return _prepare(spec, op, args)
    except (ValueError, TypeError, KeyError) as exc:
        return Result.blocked_(spec, str(exc), CODE_BAD_REQUEST)


def _wire_op(prepared: _Prepared) -> dict[str, Any]:
    """Format a prepared operation for a lom_batch request.

    See docs/protocol.md §5.7 for the wire shape.
    """
    if prepared.op is Access.GET:
        return {"op": "get", "path": prepared.path}
    if prepared.op is Access.SET:
        return {"op": "set", "path": prepared.path, "value": prepared.value}
    return {
        "op": "call",
        "path": prepared.path,
        "method": str(prepared.spec.method),
        "args": prepared.call_args,
    }


def _batch_replies(
    client: LomClient, wire_ops: Sequence[Mapping[str, Any]], *, atomic: bool
) -> list[Mapping[str, Any]]:
    """Dispatch a batch request and return individual operation response dictionaries."""
    resp = client.batch(wire_ops, atomic=atomic)
    ok, payload, code, message = _split_response(resp)
    if not ok:
        failure = {"code": code, "message": message}
        return [failure for _ in wire_ops]
    results = payload.get("results")
    if not isinstance(results, list):
        return []
    return [
        r if isinstance(r, Mapping) else {"code": "internal", "message": repr(r)}
        for r in results
    ]


def _split_response(
    resp: Mapping[str, Any],
) -> tuple[bool, Mapping[str, Any], str | None, str | None]:
    """Parse a wire response into (ok, payload, code, message).

    Handles standard response envelopes (status: success/error), un-nested batch item payloads,
    and direct error dictionaries.
    """
    status = resp.get("status")
    if status == "error":
        return False, {}, _opt_str(resp.get("code")), _opt_str(resp.get("message"))
    if status == "success":
        result = resp.get("result")
        # Direct result envelope check: lom_batch stamps status directly onto item payloads
        if isinstance(result, Mapping) and set(resp) <= {"status", "result"}:
            return True, result, None, None
        return True, resp, None, None
    if status is None:
        if "code" in resp:
            return False, {}, _opt_str(resp.get("code")), _opt_str(resp.get("message"))
        return True, resp, None, None
    return False, {}, "internal", f"unexpected status {status!r} in reply"


def _first(payload: Mapping[str, Any], *keys: str) -> object:
    """Return the first existing key from payload."""
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _opt_str(raw: object) -> str | None:
    return None if raw is None else str(raw)


def _opt_bool(raw: object) -> bool | None:
    return None if raw is None else bool(raw)


def _same_value(a: object, b: object) -> bool:
    """Compare requested and read-back values with float tolerance."""
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) is bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=_REL_TOL, abs_tol=_ABS_TOL)
    return bool(a == b)


def _run_verifier(
    client: LomClient, prepared: _Prepared, args: Mapping[str, Any], result: Result
) -> None:
    """Execute post-condition verification hook for a completed operation."""
    if not result.ok:
        return
    name = prepared.spec.verify or (DEFAULT_SET_VERIFY if prepared.op is Access.SET else None)
    if not name or name == VERIFY_NONE:
        return
    verifier = VERIFIERS.get(name)
    if verifier is None:
        result.verified = None
        result.message = result.message or f"unknown verifier {name!r}; not verified"
        return
    result.verified = verifier(client, prepared.spec, args, result)


# -------------------------------------------------------------------- verifiers


def _verify_read_back(
    client: LomClient, spec: PathSpec, args: Mapping[str, Any], result: Result
) -> bool | None:
    """Verify stored value matches requested value using read-back payload.

    Returns None if read_back is 'not_observed' (indicating asynchronous or deferred update).
    """
    if result.read_back == "not_observed":
        return None
    if result.after is None or result.value is None:
        return None
    return _same_value(result.after, result.value)


def _verify_envelope_moves(
    client: LomClient, spec: PathSpec, args: Mapping[str, Any], result: Result
) -> bool | None:
    """Verify that an automation envelope exists and is non-flat (min != max).

    Sampled strictly at t > 0 to avoid reading Live's default parameter value at t=0.
    """
    reader = getattr(client, "automation_read", None)
    if not callable(reader):
        return None
    parameter = args.get("parameter")
    if parameter is None:
        return None

    start = max(_as_float(args.get("start"), 0.0), _VERIFY_START_BEATS)
    end = _as_float(args.get("end"), 0.0)
    try:
        resp = reader(result.path, parameter, start=start, end=end if end > start else None)
    except (TypeError, ValueError):
        return None
    if not isinstance(resp, Mapping):
        return None

    ok, payload, _code, _message = _split_response(resp)
    if not ok:
        return None
    if payload.get("has_envelope") is False:
        return False
    moves = payload.get("moves")
    if isinstance(moves, bool):
        return moves
    low, high = payload.get("min"), payload.get("max")
    if low is None or high is None:
        return None
    return not _same_value(low, high)


def _as_float(raw: object, default: float) -> float:
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _read_bool(client: LomClient, path: str) -> bool | None:
    """Query a boolean property value over the transport client."""
    getter = getattr(client, "get", None)
    if not callable(getter):
        return None
    try:
        resp = getter(path)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(resp, Mapping):
        return None
    ok, payload, _code, _message = _split_response(resp)
    value = payload.get("value")
    return bool(value) if ok and isinstance(value, bool) else None


def _slot_of(result: Result) -> str:
    """Derive the clip slot path from an addressed clip path."""
    return result.path.removesuffix(".clip")


def _verify_slot_holds_a_clip(
    client: LomClient, spec: PathSpec, args: Mapping[str, Any], result: Result
) -> bool | None:
    """Verify that a clip slot currently contains a clip."""
    return _read_bool(client, f"{_slot_of(result)}.has_clip")


def _verify_slot_is_empty(
    client: LomClient, spec: PathSpec, args: Mapping[str, Any], result: Result
) -> bool | None:
    """Verify that a clip slot is currently empty."""
    held = _read_bool(client, f"{_slot_of(result)}.has_clip")
    return None if held is None else not held


def _verify_envelope_present(
    client: LomClient, spec: PathSpec, args: Mapping[str, Any], result: Result
) -> bool | None:
    """Verify that a clip contains at least one automation envelope."""
    return _read_bool(client, f"{result.path}.has_envelopes")


def _verify_envelope_absent(
    client: LomClient, spec: PathSpec, args: Mapping[str, Any], result: Result
) -> bool | None:
    """Verify that all envelopes on the clip have been cleared."""
    if spec.method != "clear_all_envelopes":
        return None
    held = _read_bool(client, f"{result.path}.has_envelopes")
    return None if held is None else not held


#: Post-condition verification handlers mapped by catalog 'verify' identifier.
VERIFIERS: dict[str, Callable[[LomClient, PathSpec, Mapping[str, Any], Result], bool | None]] = {
    "read_back": _verify_read_back,
    "envelope_moves": _verify_envelope_moves,
    "has_clip": _verify_slot_holds_a_clip,
    "no_clip": _verify_slot_is_empty,
    "envelope_present": _verify_envelope_present,
    "envelope_absent": _verify_envelope_absent,
}
