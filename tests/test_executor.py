"""Executor tests against a fake client: no socket anywhere in this file.

:mod:`ableton_maestro.executor` declares what it needs from the transport as a
:class:`~ableton_maestro.executor.LomClient` protocol precisely so that it can be
exercised with a hand-written stand-in, and :class:`FakeClient` is that stand-in: it
records every call and hands back canned replies. A method the test did not arrange a
reply for raises, so "nothing was sent" is provable rather than assumed, which is what
the guard tests below actually assert.

The catalog rows here are built by hand instead of loaded from
``src/ableton_maestro/catalog/``. The subject is the executor, and a test that fails
because somebody edited a YAML row would be testing the wrong thing.

Two exception classes *are* imported from ``client.py``: the executor's contract is
that a failure carrying a ``code`` becomes a failed :class:`Result` while anything
else (a timeout above all) is re-raised, and the honest way to test that contract is
against the exceptions the real client raises rather than against a look-alike.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest

from ableton_maestro.client import AbletonCommandError, AbletonTimeoutError
from ableton_maestro.executor import (
    CODE_ACCESS_NOT_ALLOWED,
    CODE_BAD_REQUEST,
    CODE_CONFIRM_REQUIRED,
    CODE_NO_RESULT,
    LomClient,
    Result,
    UnknownSpecError,
    execute,
    execute_batch,
)
from ableton_maestro.models import Access, Kind, Unit
from ableton_maestro.registry import Registry
from ableton_maestro.spec import ArgSpec, ParamSpec, PathSpec

# --------------------------------------------------------------------------- #
# The fake client
# --------------------------------------------------------------------------- #

Reply = Mapping[str, Any] | Callable[..., Mapping[str, Any]]


class FakeClient:
    """Records every call; answers only what the test arranged for.

    A reply may be a mapping (returned as-is) or a callable (invoked with the call's
    arguments), which is how a test makes the reply depend on the path it was asked
    about. A call with no arranged reply raises :class:`AssertionError`: the guard
    tests below turn on *nothing being sent*, and a fake that quietly returned an
    empty dict would let a leaked request pass unnoticed.
    """

    def __init__(self, **arranged: Reply) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self._arranged = arranged

    def _answer(self, name: str, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        self.calls.append((name, args, kwargs))
        reply = self._arranged.get(name)
        if reply is None:
            raise AssertionError(
                f"the fake client was asked for {name!r}{args!r} and no reply was arranged. "
                "If this call was supposed to be blocked, that is the bug"
            )
        return reply(*args, **kwargs) if callable(reply) else dict(reply)

    # -- the LomClient protocol -------------------------------------------
    def get(self, path: str) -> Mapping[str, Any]:
        return self._answer("get", path)

    def set(self, path: str, value: object) -> Mapping[str, Any]:
        return self._answer("set", path, value)

    def call(self, path: str, method: str, args: Sequence[object]) -> Mapping[str, Any]:
        return self._answer("call", path, method, list(args))

    def batch(
        self, ops: Sequence[Mapping[str, Any]], *, atomic: bool = False
    ) -> Mapping[str, Any]:
        return self._answer("batch", [dict(op) for op in ops], atomic=atomic)

    # -- inspection --------------------------------------------------------
    @property
    def names(self) -> list[str]:
        """The method name of every call made, in order."""
        return [name for name, _args, _kwargs in self.calls]

    def only(self, name: str) -> tuple[Any, ...]:
        """The positional arguments of the single call to ``name``."""
        matching = [args for called, args, _kwargs in self.calls if called == name]
        assert len(matching) == 1, f"expected exactly one {name!r} call, got {len(matching)}"
        return matching[0]


class FakeClientWithAutomation(FakeClient):
    """A client that also offers ``automation_read`` (optional protocol feature).

    The base class deliberately does not, because ``envelope_moves`` has to answer
    "could not verify" rather than failing when the transport cannot read an envelope.
    """

    def automation_read(
        self,
        path: str,
        parameter: object,
        *,
        start: float | None = None,
        end: float | None = None,
    ) -> Mapping[str, Any]:
        return self._answer("automation_read", path, parameter, start=start, end=end)


def echoing_set(**over: Any) -> Callable[..., Mapping[str, Any]]:
    """A ``lom_set`` reply that echoes the path and stores what was asked for.

    The script always echoes the path it wrote (§5.4), and ``envelope_moves`` reads
    the echoed path back out of the result, so a fake that answered with a stand-in
    string would be testing a reply the script never sends.
    """

    def _reply(path: str, value: object) -> Mapping[str, Any]:
        body: dict[str, Any] = {
            "path": path,
            "requested": value,
            "before": 0.0,
            "after": value,
            "clamped": False,
            "changed": True,
        }
        body.update(over)
        return body

    return _reply


def raising(exc: Exception) -> Callable[..., Mapping[str, Any]]:
    """A reply that raises: the transport failing rather than the script refusing."""

    def _reply(*_args: Any, **_kwargs: Any) -> Mapping[str, Any]:
        raise exc

    return _reply


def test_fake_client_satisfies_the_protocol() -> None:
    """If the stand-in drifts from :class:`LomClient`, every test below is fiction."""
    assert isinstance(FakeClient(), LomClient)
    assert isinstance(FakeClientWithAutomation(), LomClient)


# --------------------------------------------------------------------------- #
# Catalog rows, built by hand
# --------------------------------------------------------------------------- #


def volume_spec() -> PathSpec:
    """A plain settable float: the mixer row the read-back was designed for."""
    return PathSpec(
        id="track.volume",
        path="song.tracks[{track}].mixer_device.volume",
        access=[Access.GET, Access.SET, Access.AUTOMATE],
        kind=Kind.FLOAT,
        range=(0.0, 1.0),
        unit=Unit.NORMALIZED,
        display="db",
        params=[ParamSpec(name="track")],
        doc="Track volume, normalised. 0.85 is 0 dB (measured).",
    )


def name_spec() -> PathSpec:
    """Read-only here on purpose, so that a ``set`` has something to be refused by."""
    return PathSpec(
        id="track.name",
        path="song.tracks[{track}].name",
        access=[Access.GET],
        kind=Kind.STR,
        unit=Unit.NONE,
        params=[ParamSpec(name="track")],
        doc="Track name.",
    )


def quantized_spec() -> PathSpec:
    """A stepped device parameter: where Live's silent clamping actually happens."""
    return PathSpec(
        id="device.parameter_value",
        path="song.tracks[{track}].devices[{device}].parameters[{parameter}].value",
        access=[Access.GET, Access.SET],
        kind=Kind.FLOAT,
        range=(0.0, 1.0),
        quantized=True,
        params=[ParamSpec(name="track"), ParamSpec(name="device"), ParamSpec(name="parameter")],
        doc="A quantized parameter takes discrete steps; 0..1 is not a continuum.",
    )


def envelope_spec() -> PathSpec:
    """A row whose write is checked by re-reading the envelope.

    ``parameter`` is both the path's index placeholder and the argument
    ``envelope_moves`` hands to ``automation_read``. That overlap is not decoration:
    :func:`~ableton_maestro.spec.build_path` refuses any keyword the row does not
    declare, so a verifier argument can only reach the hook by also being a
    placeholder.
    """
    return PathSpec(
        id="device.parameter_automated",
        path="song.tracks[{track}].devices[{device}].parameters[{parameter}].value",
        access=[Access.GET, Access.SET, Access.AUTOMATE],
        kind=Kind.FLOAT,
        range=(0.0, 1.0),
        verify="envelope_moves",
        params=[ParamSpec(name="track"), ParamSpec(name="device"), ParamSpec(name="parameter")],
        doc="Written as automation, verified by re-reading the curve.",
    )


def delete_clip_spec() -> PathSpec:
    """Destructive, and therefore gated behind ``confirm=True``."""
    return PathSpec(
        id="clip_slot.delete_clip",
        path="song.tracks[{track}].clip_slots[{slot}]",
        access=[Access.CALL],
        method="delete_clip",
        destructive=True,
        verify="none",
        kind=Kind.OBJECT,
        params=[ParamSpec(name="track"), ParamSpec(name="slot")],
        doc="Deletes the clip in this slot. There is no undo in the LOM.",
    )


def clear_envelope_spec() -> PathSpec:
    """A call row whose argument is a Live object, so it travels as ``{"__path__"}``."""
    return PathSpec(
        id="clip.clear_envelope",
        path="song.tracks[{track}].clip_slots[{slot}].clip",
        access=[Access.CALL],
        method="clear_envelope",
        verify="none",
        kind=Kind.OBJECT,
        params=[ParamSpec(name="track"), ParamSpec(name="slot")],
        args=[ArgSpec(name="parameter", kind=Kind.OBJECT, lom_object=True, doc="the parameter")],
        doc="Clears one envelope. protocol.md §5.5.",
    )


def undo_spec() -> PathSpec:
    """A call row that nothing on its own path can verify, hence ``verify: none``."""
    return PathSpec(
        id="song.undo",
        path="song",
        access=[Access.CALL],
        method="undo",
        verify="none",
        kind=Kind.OBJECT,
        doc="One step back. Measured 2026-08-29 against Live 12.4.5: it works.",
    )


def registry(*specs: PathSpec) -> Registry:
    return Registry(list(specs))


# --------------------------------------------------------------------------- #
# Guards: refused before anything is sent
# --------------------------------------------------------------------------- #


def test_destructive_without_confirm_is_blocked_and_nothing_is_sent() -> None:
    """The executor enforces ``confirm=True``, not the caller."""
    client = FakeClient()
    result = execute(client, registry(delete_clip_spec()), "clip_slot.delete_clip", track=1, slot=2)

    assert result.blocked is True
    assert result.ok is False
    assert result.code == CODE_CONFIRM_REQUIRED
    assert "confirm=True" in str(result.message)
    assert client.calls == [], "a blocked destructive op must not touch the wire"
    assert result.path == "song.tracks[{track}].clip_slots[{slot}]", (
        "the concrete path was never built, and inventing one would misreport "
        "what did not happen"
    )


def test_destructive_with_confirm_goes_through() -> None:
    """The gate is a gate, not a wall."""
    client = FakeClient(call={"path": "song.tracks[1].clip_slots[2]", "method": "delete_clip"})
    result = execute(
        client,
        registry(delete_clip_spec()),
        "clip_slot.delete_clip",
        track=1,
        slot=2,
        confirm=True,
    )

    assert result.ok is True
    assert client.only("call") == ("song.tracks[1].clip_slots[2]", "delete_clip", [])


def test_access_not_granted_is_blocked_with_an_explanation() -> None:
    """A row that grants only ``get`` refuses a ``set``, and says what it does allow."""
    client = FakeClient()
    result = execute(client, registry(name_spec()), "track.name", track=0, value="Kick")

    assert result.blocked is True
    assert result.code == CODE_ACCESS_NOT_ALLOWED
    message = str(result.message)
    assert "does not allow set" in message
    assert "the catalog grants: get" in message
    assert client.calls == []


def test_a_call_row_refuses_a_value_argument() -> None:
    """``value=`` on a method row is a caller mistake, and it is named as one."""
    client = FakeClient()
    with pytest.raises(ValueError, match="call_args"):
        execute(client, registry(undo_spec()), "song.undo", value=1)
    assert client.calls == []


def test_unknown_id_raises_with_the_near_matches_named() -> None:
    """A mistyped id is the commonest way to get nothing done and believe otherwise."""
    with pytest.raises(UnknownSpecError) as caught:
        execute(FakeClient(), registry(volume_spec(), name_spec()), "track.volumee", track=0)

    text = str(caught.value)
    assert text.startswith("unknown catalog id 'track.volumee'")
    assert "track.volume" in text


# --------------------------------------------------------------------------- #
# Validation happens before sending
# --------------------------------------------------------------------------- #


def test_out_of_range_value_never_reaches_the_client() -> None:
    """Step 4 of :func:`execute` is client-side, so a bad value never enters Live."""
    client = FakeClient()
    with pytest.raises(ValueError, match="above the allowed maximum"):
        execute(client, registry(volume_spec()), "track.volume", track=0, value=1.4)
    assert client.calls == []


def test_unknown_placeholder_never_reaches_the_client() -> None:
    """A silently dropped keyword is the failure class this project exists to kill."""
    client = FakeClient()
    with pytest.raises(ValueError, match="unknown argument"):
        execute(client, registry(volume_spec()), "track.volume", trak=0, value=0.5)
    assert client.calls == []


def test_a_bool_is_not_a_number_and_is_refused_before_sending() -> None:
    """Verify boolean volume arguments are rejected despite int subclassing."""
    client = FakeClient()
    with pytest.raises(ValueError, match="bool"):
        execute(client, registry(volume_spec()), "track.volume", track=0, value=True)
    assert client.calls == []


def test_call_argument_arity_is_checked_before_sending() -> None:
    client = FakeClient()
    with pytest.raises(ValueError, match="takes parameter"):
        execute(client, registry(clear_envelope_spec()), "clip.clear_envelope", track=0, slot=0)
    assert client.calls == []


def test_a_lom_object_argument_travels_as_a_path_and_is_bounds_checked() -> None:
    """protocol.md §5.5: the same resolver, the same guards, no wider reach."""
    spec = clear_envelope_spec()
    client = FakeClient(call={"path": "x", "method": "clear_envelope", "result": None})
    execute(
        client,
        registry(spec),
        "clip.clear_envelope",
        track=0,
        slot=0,
        call_args=["song.tracks[0].devices[0].parameters[3]"],
    )
    assert client.only("call")[2] == [
        {"__path__": "song.tracks[0].devices[0].parameters[3]"}
    ]

    refusing = FakeClient()
    with pytest.raises(ValueError, match="negative index"):
        execute(
            refusing,
            registry(spec),
            "clip.clear_envelope",
            track=0,
            slot=0,
            call_args=["song.tracks[-1]"],
        )
    assert refusing.calls == [], (
        "a __path__ argument must not reach anything a lom_get path could not"
    )


# --------------------------------------------------------------------------- #
# The read-back (docs/architecture.md, 'read-back as a principle', protocol.md §5.4)
# --------------------------------------------------------------------------- #


def test_read_back_fields_survive_into_the_result() -> None:
    """Verify every field reported by ``lom_set`` is mapped in read-back."""
    client = FakeClient(
        set={
            "path": "song.tracks[3].mixer_device.volume",
            "requested": 0.85,
            "before": 0.7,
            "after": 0.85,
            "clamped": False,
            "changed": True,
            "display": "0.0 dB",
        }
    )
    result = execute(client, registry(volume_spec()), "track.volume", track=3, value=0.85)

    assert result.ok is True
    assert result.path == "song.tracks[3].mixer_device.volume"
    assert result.value == 0.85
    assert result.before == 0.7
    assert result.after == 0.85
    assert result.clamped is False
    assert result.changed is True
    assert result.display == "0.0 dB"
    assert result.verified is True, "after == requested, so read_back holds"
    assert client.only("set") == ("song.tracks[3].mixer_device.volume", 0.85)


def test_a_clamped_write_is_reported_as_clamped_not_as_success() -> None:
    """Verify quantized parameter snapping detection on write verification.

    Measured: a requested 0.35 came back as 0.25. The handler ran, so
    ``ok`` is True, but the value that is stored is not the value that was asked for,
    and both ``clamped`` and a failed ``read_back`` say so. A caller that reports this
    as "done" is exactly what docs/architecture.md, 'read-back as a principle' was written against.
    """
    client = FakeClient(
        set={
            "path": "song.tracks[0].devices[0].parameters[5].value",
            "requested": 0.35,
            "before": 0.0,
            "after": 0.25,
            "clamped": True,
            "changed": True,
        }
    )
    result = execute(
        client,
        registry(quantized_spec()),
        "device.parameter_value",
        track=0,
        device=0,
        parameter=5,
        value=0.35,
    )

    assert result.ok is True
    assert result.clamped is True
    assert result.after == 0.25
    assert result.value == 0.35
    assert result.verified is False, "read_back is what turns a silent clamp loud"


def test_a_no_op_write_is_not_an_error_but_is_reported() -> None:
    """``changed: false`` is legitimate, and the caller still gets to know."""
    client = FakeClient(
        set={
            "path": "song.tracks[0].mixer_device.volume",
            "requested": 0.85,
            "before": 0.85,
            "after": 0.85,
            "clamped": False,
            "changed": False,
        }
    )
    result = execute(client, registry(volume_spec()), "track.volume", track=0, value=0.85)

    assert result.ok is True
    assert result.changed is False
    assert result.verified is True


def test_a_script_that_omits_requested_still_gets_a_read_back() -> None:
    """We know what we asked for; the comparison does not depend on the echo."""
    client = FakeClient(
        set={"path": "song.tracks[0].mixer_device.volume", "before": 0.7, "after": 0.85}
    )
    result = execute(client, registry(volume_spec()), "track.volume", track=0, value=0.85)

    assert result.value == 0.85
    assert result.verified is True


def test_read_back_is_tolerant_of_float_round_tripping() -> None:
    """Verify float comparisons tolerate JSON serialization precision limits."""
    client = FakeClient(
        set={
            "path": "song.tracks[0].mixer_device.volume",
            "requested": 0.85,
            "before": 0.0,
            "after": 0.8500000000000001,
        }
    )
    result = execute(client, registry(volume_spec()), "track.volume", track=0, value=0.85)
    assert result.verified is True


def test_a_get_reports_the_value_and_verifies_nothing() -> None:
    """``read_back`` has nothing to compare on a read, and ``None`` is never a pass."""
    client = FakeClient(get={"path": "song.tracks[0].name", "value": "Kick", "type": "string"})
    result = execute(client, registry(name_spec()), "track.name", track=0)

    assert result.ok is True
    assert result.value == "Kick"
    assert result.verified is None
    assert client.names == ["get"]


# --------------------------------------------------------------------------- #
# Verifiers
# --------------------------------------------------------------------------- #


def test_envelope_verifier_runs_and_its_verdict_lands_in_the_result() -> None:
    client = FakeClientWithAutomation(
        set=echoing_set(),
        automation_read={"has_envelope": True, "min": 0.1, "max": 0.9, "moves": True},
    )
    result = execute(
        client,
        registry(envelope_spec()),
        "device.parameter_automated",
        track=0,
        device=1,
        parameter=4,
        value=0.7,
    )

    assert result.verified is True
    assert client.names == ["set", "automation_read"]


def test_envelope_verifier_never_samples_at_time_zero() -> None:
    """Measured: at ``time=0`` Live returns the parameter's default, not the curve.

    A verifier that sampled there would call a flat envelope moving, or a moving one
    flat, depending on where the default sits, so the start offset has to be > 0.
    """
    client = FakeClientWithAutomation(
        set=echoing_set(),
        automation_read={"has_envelope": True, "min": 0.1, "max": 0.9},
    )
    execute(
        client,
        registry(envelope_spec()),
        "device.parameter_automated",
        track=0,
        device=1,
        parameter=4,
        value=0.7,
    )

    _name, args, kwargs = client.calls[-1]
    assert args == ("song.tracks[0].devices[1].parameters[4].value", 4)
    assert kwargs["start"] > 0.0


def test_a_flat_envelope_fails_the_verifier() -> None:
    """A written curve that reports success and then sits flat is the silent no-op."""
    client = FakeClientWithAutomation(
        set=echoing_set(),
        automation_read={"has_envelope": True, "min": 0.5, "max": 0.5, "moves": False},
    )
    result = execute(
        client,
        registry(envelope_spec()),
        "device.parameter_automated",
        track=0,
        device=1,
        parameter=4,
        value=0.7,
    )
    assert result.verified is False


def test_a_client_without_automation_read_cannot_verify_and_says_so() -> None:
    """"Could not check" is reported as ``None``: never quietly as a pass."""
    client = FakeClient(set=echoing_set())
    result = execute(
        client,
        registry(envelope_spec()),
        "device.parameter_automated",
        track=0,
        device=1,
        parameter=4,
        value=0.7,
    )
    assert result.verified is None
    assert client.names == ["set"]


def test_verify_false_skips_the_hook_only() -> None:
    """The operation still runs and the reply is still mapped; only the check is off."""
    client = FakeClient(
        set={
            "path": "song.tracks[0].devices[0].parameters[5].value",
            "requested": 0.35,
            "before": 0.0,
            "after": 0.25,
            "clamped": True,
        }
    )
    result = execute(
        client,
        registry(quantized_spec()),
        "device.parameter_value",
        track=0,
        device=0,
        parameter=5,
        value=0.35,
        verify=False,
    )

    assert result.ok is True
    assert result.clamped is True, "the script's own clamp flag is not a verifier"
    assert result.verified is None


def test_verify_none_is_a_statement_not_a_missing_verifier() -> None:
    """``verify: none`` means nothing on this path can prove it (docs/catalog.md).

    It is the second most common spelling in the catalog: 386 rows carry it against
    734 on ``read_back``, and it must not be mistaken for a verifier name this build
    does not know: that stamped "unknown verifier 'none'" onto every one of those results.
    """
    client = FakeClient(call={"path": "song", "method": "undo", "result": None})
    result = execute(client, registry(undo_spec()), "song.undo")

    assert result.ok is True
    assert result.verified is None
    assert result.message is None


def test_an_unknown_verifier_name_stays_loud() -> None:
    """Verify missing parameter names raise appropriate catalog error."""
    spec = PathSpec(
        id="track.volume",
        path="song.tracks[{track}].mixer_device.volume",
        access=[Access.GET, Access.SET],
        kind=Kind.FLOAT,
        verify="sounds_good",
        params=[ParamSpec(name="track")],
        doc="A verifier nobody implemented.",
    )
    client = FakeClient(
        set={"path": "song.tracks[0].mixer_device.volume", "requested": 0.5, "after": 0.5}
    )
    result = execute(client, registry(spec), "track.volume", track=0, value=0.5)

    assert result.verified is None
    assert "sounds_good" in str(result.message)


# --------------------------------------------------------------------------- #
# Failures from the transport
# --------------------------------------------------------------------------- #


def test_a_structured_command_error_becomes_a_failed_result() -> None:
    """A refusal from the script is an outcome, and the ``code`` travels unchanged."""
    error = AbletonCommandError(
        "lom_set",
        {"path": "song.tracks[0].mixer_device.volume", "value": 0.5},
        {
            "status": "error",
            "code": "not_settable",
            "message": "volume is read-only on a group track",
            "path": "song.tracks[0].mixer_device.volume",
        },
    )
    client = FakeClient(set=raising(error))
    result = execute(client, registry(volume_spec()), "track.volume", track=0, value=0.5)

    assert result.ok is False
    assert result.code == "not_settable"
    assert result.message == "volume is read-only on a group track"
    assert result.verified is None, "a failed op is never verified"


def test_a_timeout_is_not_turned_into_a_result() -> None:
    """Verify timed-out writes are not assumed to have failed (protocol §8).

    Mapping it onto ``ok=False`` would state an outcome nobody knows, so the
    exception is left to reach the caller, who verifies with a read.
    """
    client = FakeClient(set=raising(AbletonTimeoutError("lom_set", 20.0, "no reply")))
    with pytest.raises(AbletonTimeoutError) as caught:
        execute(client, registry(volume_spec()), "track.volume", track=0, value=0.5)
    assert caught.value.may_have_landed is True


# --------------------------------------------------------------------------- #
# execute_batch: one round trip (protocol.md §5.7, docs/limits.md)
# --------------------------------------------------------------------------- #


def batch_reply(*entries: Mapping[str, Any]) -> dict[str, Any]:
    """A ``lom_batch`` payload: results in order, plus the two counters."""
    errors = sum(1 for e in entries if e.get("status") == "error" or "code" in e)
    return {
        "results": [dict(e) for e in entries],
        "ok_count": len(entries) - errors,
        "error_count": errors,
    }


def test_execute_batch_is_exactly_one_round_trip() -> None:
    """Twenty round trips at ~450 ms each, or one. That is why ``lom_batch`` exists."""
    client = FakeClient(
        batch=batch_reply(
            {"path": "song.tracks[0].mixer_device.volume", "value": 0.7, "status": "success"},
            {"path": "song.tracks[1].mixer_device.volume", "value": 0.6, "status": "success"},
            {"path": "song.tracks[2].name", "value": "Bass", "status": "success"},
        )
    )
    results = execute_batch(
        client,
        registry(volume_spec(), name_spec()),
        [
            {"id": "track.volume", "track": 0},
            {"id": "track.volume", "track": 1},
            {"id": "track.name", "track": 2},
        ],
    )

    assert client.names == ["batch"], "one call, and it is the batch one"
    assert len(client.calls) == 1
    ops, = client.only("batch")
    assert ops == [
        {"op": "get", "path": "song.tracks[0].mixer_device.volume"},
        {"op": "get", "path": "song.tracks[1].mixer_device.volume"},
        {"op": "get", "path": "song.tracks[2].name"},
    ]
    assert client.calls[0][2] == {"atomic": False}
    assert [r.value for r in results] == [0.7, 0.6, "Bass"]
    assert all(r.ok for r in results)


def test_batch_entries_keep_their_payload_when_status_is_stamped_on_it() -> None:
    """Measured against Live 12.4.5, and it must not regress.

    ``lom_batch`` stamps ``status: "success"`` onto the handler's own payload rather
    than nesting it under ``result``. Reading that as a §4 envelope drops every
    batched value in silence, which would have hit every intent tool that batches.
    """
    entry = {"path": "song.tracks[3].name", "value": "03 Kick", "type": "string",
             "status": "success"}
    client = FakeClient(batch=batch_reply(entry))
    results = execute_batch(client, registry(name_spec()), [{"id": "track.name", "track": 3}])

    assert results[0].ok is True
    assert results[0].value == "03 Kick"
    assert results[0].path == "song.tracks[3].name"


def test_batch_writes_carry_the_read_back_and_can_be_verified() -> None:
    """``verify`` is opt-in for a batch: ``read_back`` costs no extra round trip."""
    client = FakeClient(
        batch=batch_reply(
            {
                "path": "song.tracks[0].mixer_device.volume",
                "requested": 0.85,
                "before": 0.5,
                "after": 0.85,
                "clamped": False,
                "changed": True,
                "status": "success",
            },
            {
                "path": "song.tracks[1].mixer_device.volume",
                "requested": 1.0,
                "before": 0.5,
                "after": 0.92,
                "clamped": True,
                "changed": True,
                "status": "success",
            },
        )
    )
    results = execute_batch(
        client,
        registry(volume_spec()),
        [
            {"id": "track.volume", "track": 0, "value": 0.85},
            {"id": "track.volume", "track": 1, "value": 1.0},
        ],
        verify=True,
    )

    assert client.names == ["batch"], "read_back only inspects the reply it already has"
    assert results[0].verified is True
    assert results[1].verified is False
    assert results[1].clamped is True


def test_a_blocked_op_is_left_out_of_the_wire_message_and_keeps_its_position() -> None:
    """Verify failed batch operations are isolated and reported in results."""
    client = FakeClient(
        batch=batch_reply(
            {"path": "song.tracks[0].mixer_device.volume", "value": 0.7, "status": "success"},
            {"path": "song.tracks[2].name", "value": "Bass", "status": "success"},
        )
    )
    results = execute_batch(
        client,
        registry(volume_spec(), name_spec(), delete_clip_spec()),
        [
            {"id": "track.volume", "track": 0},
            {"id": "clip_slot.delete_clip", "track": 1, "slot": 0},
            {"id": "track.name", "track": 2},
        ],
    )

    ops, = client.only("batch")
    assert len(ops) == 2, "the destructive op never reached the wire"
    assert len(results) == 3
    assert results[0].ok is True and results[0].value == 0.7
    assert results[1].blocked is True and results[1].code == CODE_CONFIRM_REQUIRED
    assert results[2].ok is True and results[2].value == "Bass"


def test_a_batch_op_that_fails_validation_is_blocked_not_raised() -> None:
    """The one place the executor is deliberately more forgiving than :func:`execute`."""
    client = FakeClient(
        batch=batch_reply({"path": "song.tracks[0].name", "value": "Kick", "status": "success"})
    )
    results = execute_batch(
        client,
        registry(volume_spec(), name_spec()),
        [
            {"id": "track.volume", "track": 0, "value": 9.9},
            {"id": "track.name", "track": 0},
        ],
    )

    assert results[0].blocked is True
    assert results[0].code == CODE_BAD_REQUEST
    assert "above the allowed maximum" in str(results[0].message)
    assert results[1].ok is True
    ops, = client.only("batch")
    assert len(ops) == 1


def test_a_failing_op_inside_a_batch_keeps_its_structured_code() -> None:
    """Results come back in order, each shaped like its handler's or like an error."""
    client = FakeClient(
        batch=batch_reply(
            {"path": "song.tracks[0].name", "value": "Kick", "status": "success"},
            {"status": "error", "code": "index_out_of_range", "message": "no track 99"},
        )
    )
    results = execute_batch(
        client,
        registry(name_spec()),
        [{"id": "track.name", "track": 0}, {"id": "track.name", "track": 99}],
    )

    assert results[0].ok is True
    assert results[1].ok is False
    assert results[1].code == "index_out_of_range"
    assert results[1].path == "song.tracks[99].name", "the op knows its own path"


def test_a_short_reply_leaves_the_missing_ops_unknown() -> None:
    """Fewer results than ops is not "the rest succeeded"; it is "nobody knows"."""
    client = FakeClient(
        batch=batch_reply({"path": "song.tracks[0].name", "value": "Kick", "status": "success"})
    )
    results = execute_batch(
        client,
        registry(name_spec()),
        [{"id": "track.name", "track": 0}, {"id": "track.name", "track": 1}],
    )

    assert results[0].ok is True
    assert results[1].ok is False
    assert results[1].code == CODE_NO_RESULT
    assert "outcome is unknown" in str(results[1].message)


def test_a_batch_that_fails_as_a_whole_fails_every_op() -> None:
    client = FakeClient(
        batch={"status": "error", "code": "type_error", "message": "ops must be a list"}
    )
    results = execute_batch(
        client,
        registry(name_spec()),
        [{"id": "track.name", "track": 0}, {"id": "track.name", "track": 1}],
    )

    assert [r.ok for r in results] == [False, False]
    assert {r.code for r in results} == {"type_error"}


def test_atomic_is_passed_through_and_skipped_ops_keep_their_code() -> None:
    """``atomic`` only stops at the first error. Nothing is rolled back (§5.7)."""
    client = FakeClient(
        batch=batch_reply(
            {"status": "error", "code": "no_such_path", "message": "no such attribute"},
            {"status": "error", "code": "skipped", "message": "an earlier op failed"},
        )
    )
    results = execute_batch(
        client,
        registry(name_spec()),
        [{"id": "track.name", "track": 0}, {"id": "track.name", "track": 1}],
        atomic=True,
    )

    assert client.calls[0][2] == {"atomic": True}
    assert [r.code for r in results] == ["no_such_path", "skipped"]
    assert all(r.blocked is False for r in results), (
        "these ops did reach Live; only a client-side refusal is 'blocked'"
    )


def test_a_batch_op_without_an_id_is_a_caller_bug() -> None:
    client = FakeClient()
    with pytest.raises(ValueError, match="missing a string 'id'"):
        execute_batch(client, registry(name_spec()), [{"track": 0}])
    assert client.calls == []


def test_an_all_blocked_batch_sends_nothing_at_all() -> None:
    """No surviving op, no round trip, and every refusal is still reported."""
    client = FakeClient()
    results = execute_batch(
        client,
        registry(delete_clip_spec()),
        [
            {"id": "clip_slot.delete_clip", "track": 0, "slot": 0},
            {"id": "clip_slot.delete_clip", "track": 0, "slot": 1},
        ],
    )

    assert client.calls == []
    assert [r.code for r in results] == [CODE_CONFIRM_REQUIRED, CODE_CONFIRM_REQUIRED]


def test_batch_confirm_arms_a_destructive_op_and_is_not_sent_as_an_argument() -> None:
    """``confirm`` belongs to the executor; the script never sees it."""
    client = FakeClient(
        batch=batch_reply({"path": "song.tracks[1].clip_slots[2]", "method": "delete_clip",
                           "status": "success"})
    )
    results = execute_batch(
        client,
        registry(delete_clip_spec()),
        [{"id": "clip_slot.delete_clip", "track": 1, "slot": 2, "confirm": True}],
    )

    ops, = client.only("batch")
    assert ops == [
        {"op": "call", "path": "song.tracks[1].clip_slots[2]", "method": "delete_clip", "args": []}
    ]
    assert results[0].ok is True
    assert results[0].blocked is False


# --------------------------------------------------------------------------- #
# Result mapping in isolation
# --------------------------------------------------------------------------- #


def test_result_from_response_accepts_the_three_wire_shapes() -> None:
    """Envelope, bare error object, and already-unwrapped payload (§4, §5.7)."""
    envelope = Result.from_response(
        {"status": "success", "result": {"path": "song.tempo", "value": 124.0}},
        "song.tempo",
        "song.tempo",
    )
    assert envelope.ok is True and envelope.value == 124.0

    bare_error = Result.from_response(
        {"code": "bad_path", "message": "segment 'tracks[-1]' is not a name"},
        "track.name",
        "song.tracks[0].name",
    )
    assert bare_error.ok is False
    assert bare_error.code == "bad_path"
    assert bare_error.path == "song.tracks[0].name", "an error object echoes no path"

    unwrapped = Result.from_response(
        {"path": "song.tempo", "value": 124.0, "type": "float"}, "song.tempo", "song.tempo"
    )
    assert unwrapped.ok is True and unwrapped.value == 124.0


def test_a_missing_code_is_not_invented() -> None:
    """§4 says every error is structured. A hole is shown as a hole."""
    result = Result.from_response(
        {"status": "error", "message": "something went wrong"}, "track.name", "song.tracks[0].name"
    )
    assert result.ok is False
    assert result.code is None
