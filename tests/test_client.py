"""Transport tests for :mod:`ableton_maestro.client` with mock server.

No Ableton, ever. CI runs on Linux with no Live installed, so what stands in for
``live-remote-script/__init__.py`` here is :class:`FakeScript`: a thread, a loopback
socket on an ephemeral port, and the same frameless receive loop the real script runs
(docs/protocol.md §2). Nothing sleeps; the only waiting in the file is a deliberate
sub-second timeout in the two tests that are *about* timing out.

The weight of the file sits on §2 and §3, because that is where silent corruption
lives. A multi-byte character split across a ``recv`` boundary, or a discarded buffer
remainder, does not raise: it desynchronises the connection and every reply from then
on answers the *previous* request. Those two are therefore tested by construction
rather than by luck: the receive chunk size is driven down to one byte so the split is
guaranteed, and the fake answers a request it was never asked, so a client that threw
the remainder away has nothing to fall back on.
"""

from __future__ import annotations

import codecs
import json
import logging
import socket
import threading
from collections.abc import Callable
from typing import Any

import pytest

from ableton_maestro import client as client_module
from ableton_maestro.client import (
    READ_ONLY_HANDLERS,
    UNSPECIFIED_CODE,
    AbletonClient,
    AbletonCommandError,
    AbletonConnectionError,
    AbletonProtocolError,
    AbletonTimeoutError,
    is_read_only,
)

# A handler decides what goes back on the wire for one request. It returns raw byte
# chunks (raw, so a test can put two replies into a single write) or ``None`` to hang
# up without answering, which is the EOF case §8's retry rule is written for.
Handler = Callable[[dict[str, Any], int], list[bytes] | None]

# Timeouts for the clients under test. Short on purpose: a regression in the framing
# rules shows up as "no reply ever arrives", and the suite should say so in a second
# rather than in the protocol's real 10/20 s.
FAST_READ = 1.5
FAST_WRITE = 1.5


# --------------------------------------------------------------------------- #
# The fake script
# --------------------------------------------------------------------------- #


def encode(obj: Any) -> bytes:
    """Serialise a reply the way the script does: UTF-8, no ASCII escaping."""
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def success(result: Any) -> dict[str, Any]:
    """The §4 success envelope."""
    return {"status": "success", "result": result}


def failure(code: str, message: str, path: str | None = None) -> dict[str, Any]:
    """The §4 error envelope. ``path`` is present on the errors that have one."""
    body: dict[str, Any] = {"status": "error", "code": code, "message": message}
    if path is not None:
        body["path"] = path
    return body


def replies(*objects: Any) -> Handler:
    """A handler that answers the n-th request with the n-th object."""

    def _handler(request: dict[str, Any], seq: int) -> list[bytes] | None:
        if seq >= len(objects):
            return []
        return [encode(objects[seq])]

    return _handler


class FakeScript:
    """A stand-in for the Remote Script's socket server.

    It mirrors the script deliberately: loopback only, one connection served at a
    time, and the frameless read loop of docs/protocol.md §2: no delimiter, no length
    prefix, ``raw_decode`` until the buffer parses, remainder kept.

    Attributes:
        port: The ephemeral port it listens on.
        requests: Every request object it has parsed, in arrival order, across all
            connections. This is how a test proves a write was sent exactly once.
        connections: How many times a client has connected. A reconnect is the retry
            (§8), so this counter is what tells a retry from a resend.
    """

    def __init__(self, handler: Handler) -> None:
        self._handler = handler
        self.requests: list[dict[str, Any]] = []
        self.connections = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(8)
        self._listener.settimeout(0.05)
        self.port: int = self._listener.getsockname()[1]

        self._thread = threading.Thread(target=self._serve, name="fake-script", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ serving
    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _addr = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with self._lock:
                self.connections += 1
            try:
                self._converse(conn)
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def _converse(self, conn: socket.socket) -> None:
        """One connection, read the §2 way: incremental decode, remainder kept."""
        conn.settimeout(0.05)
        decoder = codecs.getincrementaldecoder("utf-8")()
        json_decoder = json.JSONDecoder()
        buffer = ""
        while not self._stop.is_set():
            try:
                chunk = conn.recv(4096)
            except TimeoutError:
                continue
            except OSError:
                return
            if not chunk:
                return
            buffer += decoder.decode(chunk)
            while True:
                stripped = buffer.lstrip()
                if not stripped:
                    buffer = ""
                    break
                try:
                    request, end = json_decoder.raw_decode(stripped)
                except ValueError:
                    buffer = stripped
                    break
                buffer = stripped[end:]
                if not self._answer(conn, request):
                    return

    def _answer(self, conn: socket.socket, request: dict[str, Any]) -> bool:
        """Run the handler for one request. False means: hang up on the client."""
        with self._lock:
            seq = len(self.requests)
            self.requests.append(request)
        chunks = self._handler(request, seq)
        if chunks is None:
            return False
        for chunk in chunks:
            try:
                conn.sendall(chunk)
            except OSError:
                return False
        return True

    # ------------------------------------------------------------------ control
    def stop(self) -> None:
        self._stop.set()
        try:
            self._listener.close()
        except OSError:
            pass
        self._thread.join(timeout=5.0)

    def handlers_seen(self) -> list[str]:
        """The ``type`` of every request, in order."""
        return [str(r.get("type")) for r in self.requests]


class _SendBlackHole:
    """A socket whose ``sendall`` times out; everything else is the real socket.

    There is no fast, portable way to make a genuine ``sendall`` block long enough to
  time out (it takes megabytes and a peer that refuses to read) so the write-timeout
    branch is reached by substituting the socket. ``shutdown``/``close`` still go to the
    real one, so a test can honestly assert that the client closed the connection.
    """

    def __init__(self, real: socket.socket) -> None:
        self.real = real
        self.sends = 0

    def settimeout(self, value: float | None) -> None:
        self.real.settimeout(value)

    def sendall(self, data: bytes) -> None:
        self.sends += 1
        raise TimeoutError("the send buffer never drained")

    def recv(self, size: int) -> bytes:
        return self.real.recv(size)

    def shutdown(self, how: int) -> None:
        self.real.shutdown(how)

    def close(self) -> None:
        self.real.close()


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def start_script() -> Any:
    """Factory for :class:`FakeScript` servers, all stopped at teardown."""
    started: list[FakeScript] = []

    def _start(handler: Handler) -> FakeScript:
        script = FakeScript(handler)
        started.append(script)
        return script

    yield _start
    for script in started:
        script.stop()


@pytest.fixture
def connect(start_script: Any) -> Any:
    """Factory for clients pointed at a fake script, all closed at teardown."""
    clients: list[AbletonClient] = []

    def _connect(script: FakeScript, **kwargs: Any) -> AbletonClient:
        kwargs.setdefault("read_timeout", FAST_READ)
        kwargs.setdefault("write_timeout", FAST_WRITE)
        client = AbletonClient("127.0.0.1", script.port, **kwargs)
        clients.append(client)
        return client

    yield _connect
    for client in clients:
        client.close()


def free_port() -> int:
    """A port nothing is listening on: for the "Live is not running" case."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return port


# --------------------------------------------------------------------------- #
# Round trip and message shape (§4)
# --------------------------------------------------------------------------- #


def test_request_shape_and_unwrapped_result(start_script: Any, connect: Any) -> None:
    """A call sends ``{type, params}`` and gets ``result`` back, unwrapped (§4)."""
    body = {"path": "song.tempo", "value": 124.0, "type": "float"}
    script = start_script(replies(success(body)))
    client = connect(script)

    assert client.get("song.tempo") == body
    assert script.requests == [{"type": "lom_get", "params": {"path": "song.tempo"}}]


def test_params_default_to_an_object(start_script: Any, connect: Any) -> None:
    """``params`` is always an object: the script calls ``params.get()`` (§4)."""
    script = start_script(replies(success({"pong": True, "script_version": "0.2.0"})))
    client = connect(script)

    assert client.ping()["pong"] is True
    assert script.requests[0]["params"] == {}


def test_params_must_be_a_mapping(start_script: Any, connect: Any) -> None:
    """A non-object ``params`` is refused here, not discovered inside Live."""
    script = start_script(replies(success({})))
    client = connect(script)

    with pytest.raises(TypeError, match="mapping"):
        client.send("lom_get", ["song.tempo"])  # type: ignore[arg-type]
    assert script.requests == []


def test_result_none_is_normalised_but_a_list_is_not(start_script: Any, connect: Any) -> None:
    """An absent result is harmless; a wrong-typed one is a contract break (§4)."""
    script = start_script(replies(success(None)))
    client = connect(script)

    assert client.send("lom_call", {"path": "song", "method": "undo"}) == {}


def test_handler_wrappers_send_the_documented_params(start_script: Any, connect: Any) -> None:
    """Each wrapper is a thin mapping onto §5: no interpretation of the payload."""
    script = start_script(replies(*[success({}) for _ in range(5)]))
    client = connect(script)

    client.set("song.tempo", 124.0)
    client.call("song", "undo")
    client.call("app.browser", "load_item", [{"__path__": "app.browser.drums.children[0]"}])
    client.describe("song.tracks[0]", depth=2)
    client.batch([{"op": "get", "path": "song.tempo"}], atomic=True)

    assert script.handlers_seen() == [
        "lom_set",
        "lom_call",
        "lom_call",
        "lom_describe",
        "lom_batch",
    ]
    assert script.requests[0]["params"] == {"path": "song.tempo", "value": 124.0}
    # No args key at all when none were given: the script's ``params.get("args")``
    # default is what decides, and sending an empty list would say something else.
    assert script.requests[1]["params"] == {"path": "song", "method": "undo"}
    assert script.requests[2]["params"]["args"] == [
        {"__path__": "app.browser.drums.children[0]"}
    ]
    assert script.requests[3]["params"] == {"path": "song.tracks[0]", "depth": 2}
    assert script.requests[4]["params"] == {
        "ops": [{"op": "get", "path": "song.tempo"}],
        "atomic": True,
    }


def test_batch_omits_atomic_when_false(start_script: Any, connect: Any) -> None:
    """``atomic`` defaults to False in the script; sending it is noise (§5.7)."""
    script = start_script(replies(success({"results": [], "ok_count": 0, "error_count": 0})))
    client = connect(script)

    client.batch([{"op": "get", "path": "song.tempo"}])
    assert script.requests[0]["params"] == {"ops": [{"op": "get", "path": "song.tempo"}]}


# --------------------------------------------------------------------------- #
# §2: there is no framing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7])
def test_multibyte_character_split_across_recv_survives(
    start_script: Any, connect: Any, monkeypatch: pytest.MonkeyPatch, chunk_size: int
) -> None:
    """A UTF-8 character straddling a ``recv`` boundary must not corrupt the buffer.

    Driving ``RECV_CHUNK`` down to a handful of bytes makes the split a certainty
    instead of a coincidence: every 2-, 3- and 4-byte character in the reply is torn
    across at least one boundary. A client that decoded each chunk on its own would
    raise ``UnicodeDecodeError`` here; only an incremental decoder gets the name back
    intact (§2).

    It also proves the module's own claim that ``RECV_CHUNK`` is cosmetic and the
    framing does not depend on it.
    """
    name = "Straße 日本語 🎹 café"
    monkeypatch.setattr(client_module, "RECV_CHUNK", chunk_size)
    script = start_script(
        replies(
            success(
                {
                    "path": "song.tracks[0].name",
                    "requested": name,
                    "before": "1-MIDI",
                    "after": name,
                    "clamped": False,
                    "changed": True,
                }
            )
        )
    )
    client = connect(script)

    result = client.set("song.tracks[0].name", name)

    assert result["after"] == name
    # And the request carried it out the same way: ensure_ascii=False, so the name
    # travels as UTF-8 bytes rather than as \u escapes.
    assert script.requests[0]["params"]["value"] == name


def test_two_replies_in_one_recv_are_both_delivered_in_order(
    start_script: Any, connect: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """The buffer remainder rule: keep the tail after ``raw_decode`` (§2).

    The fake writes *both* replies in a single ``sendall`` while answering the first
    request, and answers the second request with nothing at all. So the second reply
    can only come from the client's own buffer. A client that discarded the remainder
    has nothing left to hand back and stalls until the read timeout, which is the
    honest shape of the real bug, where the discarded reply desynchronises every
    exchange after it.

    The whitespace between the two objects is deliberate: the loop has to tolerate it,
    the way the script's does.
    """
    first = success({"path": "song.tracks[0].name", "value": "Kick", "type": "string"})
    second = success({"path": "song.tracks[1].name", "value": "Snare", "type": "string"})

    def handler(request: dict[str, Any], seq: int) -> list[bytes] | None:
        if seq == 0:
            return [encode(first) + b"\n" + encode(second)]
        return []

    script = start_script(handler)
    client = connect(script)

    caplog.set_level(logging.WARNING, logger="ableton_maestro.client")
    assert client.get("song.tracks[0].name")["value"] == "Kick"
    assert client.get("song.tracks[1].name")["value"] == "Snare"

    assert any("left over" in record.getMessage() for record in caplog.records), (
        "keeping the remainder is right, but staying quiet about it is not: "
        "under strict serialisation a leftover reply means the connection may "
        "already be desynchronised"
    )


def test_reply_arriving_in_many_writes_is_assembled(start_script: Any, connect: Any) -> None:
    """A reply arriving in many writes is assembled: "it parses now" is the boundary."""
    payload = encode(success({"path": "song.tempo", "value": 124.0, "type": "float"}))
    pieces = [payload[i : i + 4] for i in range(0, len(payload), 4)]

    def handler(request: dict[str, Any], seq: int) -> list[bytes] | None:
        return list(pieces)

    script = start_script(handler)
    client = connect(script)

    assert client.get("song.tempo")["value"] == 124.0


# --------------------------------------------------------------------------- #
# Errors (§4) and protocol breaks
# --------------------------------------------------------------------------- #


def test_structured_error_becomes_command_error_with_code_and_path(
    start_script: Any, connect: Any
) -> None:
    """Callers branch on ``code``, humans read ``message`` (§4)."""
    script = start_script(
        replies(
            failure(
                "no_such_path",
                "Track has no attribute 'volume'",
                path="song.tracks[0].volume",
            )
        )
    )
    client = connect(script)

    with pytest.raises(AbletonCommandError) as caught:
        client.get("song.tracks[0].volume")

    error = caught.value
    assert error.code == "no_such_path"
    assert error.path == "song.tracks[0].volume"
    assert error.handler == "lom_get"
    assert error.params == {"path": "song.tracks[0].volume"}
    assert error.message == "Track has no attribute 'volume'"
    assert error.response["status"] == "error"
    assert "no_such_path" in str(error)
    assert "song.tracks[0].volume" in str(error)


def test_structured_error_leaves_the_socket_open(start_script: Any, connect: Any) -> None:
    """A well-formed failure is an answer, not a broken connection (§3)."""
    script = start_script(
        replies(
            failure("not_settable", "group tracks have no arm state"),
            success({"path": "song.tempo", "value": 124.0, "type": "float"}),
        )
    )
    client = connect(script)

    with pytest.raises(AbletonCommandError):
        client.set("song.tracks[0].arm", True)
    assert client.connected is True

    # Same socket, no reconnect: the exchange after an error is an ordinary one.
    assert client.get("song.tempo")["value"] == 124.0
    assert script.connections == 1


def test_error_without_a_code_is_marked_unspecified(start_script: Any, connect: Any) -> None:
    """§4 promises a structured code. A missing one is shown, not invented."""
    script = start_script(replies({"status": "error", "message": "something went wrong"}))
    client = connect(script)

    with pytest.raises(AbletonCommandError) as caught:
        client.get("song.tempo")
    assert caught.value.code == UNSPECIFIED_CODE


def test_non_object_result_is_a_protocol_error_and_closes_the_socket(
    start_script: Any, connect: Any
) -> None:
    """§4: every Maestro result is an object, so it can grow a field safely."""
    script = start_script(replies({"status": "success", "result": [1, 2, 3]}))
    client = connect(script)

    with pytest.raises(AbletonProtocolError, match="not an object"):
        client.get("song.tracks")

    assert client.connected is False, (
        "a peer that just answered outside the contract is not one to keep a "
        "connection with (§3)"
    )


def test_non_object_reply_is_a_protocol_error_and_closes_the_socket(
    start_script: Any, connect: Any
) -> None:
    """§3 admits one reply shape: a JSON object. A bare list is not an envelope."""
    script = start_script(replies([{"status": "success"}]))
    client = connect(script)

    with pytest.raises(AbletonProtocolError, match="not a JSON object"):
        client.get("song.tempo")
    assert client.connected is False


def test_unexpected_status_is_a_protocol_error_and_closes_the_socket(
    start_script: Any, connect: Any
) -> None:
    """Branch on ``status``, and refuse anything that is neither value (§4)."""
    script = start_script(replies({"status": "maybe", "result": {}}))
    client = connect(script)

    with pytest.raises(AbletonProtocolError, match="Unexpected 'status'"):
        client.get("song.tempo")
    assert client.connected is False


# --------------------------------------------------------------------------- #
# §8: timeouts, retries, and the uncertainty that must not be resolved
# --------------------------------------------------------------------------- #


def test_read_timeout_closes_the_socket_and_did_not_land(
    start_script: Any, connect: Any
) -> None:
    """No reply inside the window: the socket goes, and a read cannot have landed."""

    def handler(request: dict[str, Any], seq: int) -> list[bytes] | None:
        return []  # accepted, parsed, and deliberately never answered

    script = start_script(handler)
    client = connect(script, read_timeout=0.2)

    with pytest.raises(AbletonTimeoutError) as caught:
        client.get("song.tempo")

    error = caught.value
    assert error.handler == "lom_get"
    assert error.timeout == pytest.approx(0.2)
    assert error.may_have_landed is False
    assert client.connected is False, (
        "a late reply on a reused socket would answer the next command (§3)"
    )
    assert len(script.requests) == 1, "a timed-out read must not be resent on the way out"


def test_write_timeout_closes_the_socket_and_may_have_landed(
    start_script: Any, connect: Any
) -> None:
    """§8: report the uncertainty, do not resolve it.

    A write that timed out may well have landed: repeating it can double a note list
    or re-fire a clip, so the client says ``may_have_landed`` and stops. The caller
    verifies with a read.
    """
    script = start_script(replies(success({})))
    client = connect(script, write_timeout=0.2)
    client.connect()
    stub = _SendBlackHole(client._sock)  # type: ignore[arg-type]
    client._sock = stub  # type: ignore[assignment]

    with pytest.raises(AbletonTimeoutError) as caught:
        client.set("song.tracks[0].mixer_device.volume", 0.85)

    error = caught.value
    assert error.handler == "lom_set"
    assert error.may_have_landed is True
    assert client.connected is False
    assert stub.sends == 1, "a write is never retried, not even after a send timeout"
    assert script.requests == [], "nothing reached the script, and it still counts as unknown"


def test_read_only_handler_retries_once_after_eof(start_script: Any, connect: Any) -> None:
    """A read may be retried exactly once, and the retry is a reconnect (§8)."""

    def handler(request: dict[str, Any], seq: int) -> list[bytes] | None:
        if seq == 0:
            return None  # Live hangs up mid-exchange
        return [encode(success({"path": "song.tempo", "value": 124.0, "type": "float"}))]

    script = start_script(handler)
    client = connect(script)

    assert client.get("song.tempo")["value"] == 124.0
    assert script.connections == 2
    assert script.handlers_seen() == ["lom_get", "lom_get"]


def test_read_only_retry_is_not_repeated_and_keeps_the_first_cause(
    start_script: Any, connect: Any
) -> None:
    """Exactly one retry, and the first failure's text survives into the message."""

    def handler(request: dict[str, Any], seq: int) -> list[bytes] | None:
        return None

    script = start_script(handler)
    client = connect(script)

    with pytest.raises(AbletonConnectionError) as caught:
        client.get("song.tempo")

    assert "retry failed as well" in str(caught.value)
    assert len(script.requests) == 2, "one attempt, one retry, and no more"


def test_write_handler_is_never_retried_after_eof(start_script: Any, connect: Any) -> None:
    """Verify writes that may have landed are not retried (prevents duplicate firing)."""

    def handler(request: dict[str, Any], seq: int) -> list[bytes] | None:
        return None

    script = start_script(handler)
    client = connect(script)

    with pytest.raises(AbletonConnectionError):
        client.set("song.tempo", 128.0)

    assert script.handlers_seen() == ["lom_set"]
    assert script.connections == 1
    assert client.connected is False


def test_timeout_classes_are_the_ones_in_section_8() -> None:
    """10 s for the read class, 20 s for the write class."""
    client = AbletonClient()

    assert client.timeout_for("lom_get") == 10.0
    assert client.timeout_for("lom_describe") == 10.0
    assert client.timeout_for("ping") == 10.0
    assert client.timeout_for("notes_get") == 10.0
    assert client.timeout_for("lom_set") == 20.0
    assert client.timeout_for("lom_call") == 20.0
    assert client.timeout_for("notes_set") == 20.0
    assert client.timeout_for("lom_batch") == 20.0


def test_browser_walk_stays_in_the_read_class() -> None:
    """The third row of §8's table (60 s) is deliberately *not* a client floor.

    §8's own prose settles it: the script gives ``browser_walk`` an 8 s
    main-thread budget and truncates its walk before that, reporting ``truncated``
    and ``truncated_by`` with real results. The invariant that matters is that the
    script always answers before the client gives up, and 8 < 10 holds it: a 60 s
    client floor would only sit waiting for an answer that already came. A caller
    who genuinely wants to wait longer passes ``timeout=`` for that one call.

    This test is here so that reinstating the floor is a decision, not a drift.
    """
    assert client_module.HANDLER_TIMEOUTS == {}
    assert AbletonClient().timeout_for("browser_walk") == 10.0


def test_a_handler_floor_raises_the_class_default_but_never_lowers_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The floor mechanism itself: waiting longer is safe, waiting less is not.

    Waiting longer than the class default never produces a wrong answer: it only
    delays a failure the caller can shorten. Waiting *less* would cut a handler off
    before the script's own budget fires, turning a structured ``live_error`` into
    a bare socket timeout, so a floor below the class default is ignored.
    """
    monkeypatch.setattr(
        client_module, "HANDLER_TIMEOUTS", {"browser_walk": 60.0, "lom_set": 5.0}
    )
    client = AbletonClient()

    assert client.timeout_for("browser_walk") == 60.0
    assert client.timeout_for("lom_set") == 20.0


def test_unknown_handlers_count_as_writes() -> None:
    """The safe direction: a handler we have never heard of might mutate the set."""
    assert is_read_only("lom_get") is True
    assert is_read_only("lom_set") is False
    assert is_read_only("something_new") is False

    client = AbletonClient()
    assert client.timeout_for("something_new") == 20.0
    assert AbletonTimeoutError("something_new", 1.0, "x").may_have_landed is True


def test_read_only_set_matches_the_protocol_table() -> None:
    """§8's first row, verbatim. Adding a handler means deciding which row it is in.

    ``enum_names`` was added to the script on 2026-08-31 and NOT to this row, and
    the cost was visible immediately: the first call after a Live restart hit the
    dead cached socket and was not retried, because a handler outside this set is
    treated as a write. Deciding the row is not paperwork.
    """
    assert set(READ_ONLY_HANDLERS) == {
        "ping",
        "script_info",
        "lom_get",
        "lom_describe",
        "notes_get",
        "automation_read",
        "browser_walk",
        "events_drain",
        "enum_names",
    }


def test_explicit_timeout_overrides_the_class_default(start_script: Any, connect: Any) -> None:
    """A caller who knows better may shorten the wait, per call."""

    def handler(request: dict[str, Any], seq: int) -> list[bytes] | None:
        return []

    script = start_script(handler)
    client = connect(script)

    with pytest.raises(AbletonTimeoutError) as caught:
        client.get("song.tempo", timeout=0.15)
    assert caught.value.timeout == pytest.approx(0.15)


# --------------------------------------------------------------------------- #
# §3: strictly serial
# --------------------------------------------------------------------------- #


def test_concurrent_threads_are_serialised_and_never_cross_replies(
    start_script: Any, connect: Any
) -> None:
    """One outstanding request per socket, whatever the callers do (§3).

    The fake echoes the requested path back as the value, so a crossed reply is
    detectable rather than merely suspected: if two requests ever went onto the wire
    without an answer in between, some thread would come back holding another thread's
    path: the permanent desynchronisation §3 describes.
    """
    def handler(request: dict[str, Any], seq: int) -> list[bytes] | None:
        path = request["params"]["path"]
        return [encode(success({"path": path, "value": path, "type": "string"}))]

    script = start_script(handler)
    client = connect(script)

    thread_count, per_thread = 6, 5
    failures: list[str] = []
    barrier = threading.Barrier(thread_count)

    def worker(index: int) -> None:
        path = f"song.tracks[{index}].name"
        barrier.wait(timeout=5.0)
        for _ in range(per_thread):
            try:
                got = client.get(path)
            except Exception as exc:  # noqa: BLE001 - the failure is the report
                failures.append(f"{path}: {exc!r}")
                return
            if got["value"] != path:
                failures.append(f"{path} received {got['value']!r}")
                return

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15.0)
        assert not thread.is_alive(), "a thread is stuck: the lock is not being released"

    assert failures == []
    assert len(script.requests) == thread_count * per_thread
    assert script.connections == 1, "they shared one socket, they did not each get one"


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


def test_context_manager_connects_and_closes(start_script: Any) -> None:
    script = FakeScript(replies(success({"pong": True})))
    try:
        with AbletonClient("127.0.0.1", script.port, read_timeout=FAST_READ) as client:
            assert client.connected is True
            assert client.ping()["pong"] is True
        assert client.connected is False
        client.close()  # idempotent, and never raises
    finally:
        script.stop()


def test_no_listener_explains_what_to_check() -> None:
    """The cold-start message is the whole diagnostic for "nothing answers"."""
    client = AbletonClient("127.0.0.1", free_port(), connect_timeout=1.0)

    with pytest.raises(AbletonConnectionError) as caught:
        client.ping()

    text = str(caught.value).lower()
    assert "control surface" in text
    assert "__pycache__" in text


def test_auto_connect_false_refuses_to_open_a_socket(start_script: Any, connect: Any) -> None:
    script = start_script(replies(success({})))
    client = connect(script, auto_connect=False)

    with pytest.raises(AbletonConnectionError, match="Not connected"):
        client.get("song.tempo")
    assert script.connections == 0


# --------------------------------------------------------------------------------
# A busy main thread is the one live_error worth trying again
# --------------------------------------------------------------------------------


def budget_expired(handler: str, seconds: float = 8.0) -> dict[str, Any]:
    """The envelope the script sends when Live's main thread did not get to it.

    ``timeout_seconds`` is the machine-readable half and the only reliable one:
    the message is prose, and prose is not a contract. The script sets this field
    on exactly one branch: the queue wait that gave up.
    """
    return {
        "status": "error",
        "code": "live_error",
        "message": (
            f"{handler} did not finish within {seconds:.1f}s on Live's main thread; the "
            "operation may still be running."
        ),
        "timeout_seconds": seconds,
        "handler": handler,
    }


def test_a_busy_main_thread_is_retried_once_for_a_read(
    start_script: Any, connect: Any
) -> None:
    """Verify harmless transient read errors can be safely retried.

    Measured 2026-09-01: opening a set holding four plugin instances kept Live's
    main thread past an 8 s ``lom_describe`` budget, and the identical call
    answered normally a moment later. Before this the caller saw a bare
    ``live_error`` and had to know to try again.
    """
    script = start_script(replies(budget_expired("lom_describe"), success({"class": "Reverb"})))
    client = connect(script)
    assert client.send("lom_describe", {"path": "song.tracks[0].devices[0]"}) == {
        "class": "Reverb"
    }
    assert len(script.requests) == 2, "the read should have been sent twice"


def test_a_busy_main_thread_is_not_retried_twice(start_script: Any, connect: Any) -> None:
    """Verify retry limit is enforced after a single retry."""
    script = start_script(
        replies(budget_expired("lom_describe"), budget_expired("lom_describe"))
    )
    client = connect(script)
    with pytest.raises(AbletonCommandError) as caught:
        client.send("lom_describe", {"path": "song"})
    assert caught.value.code == "live_error"
    assert len(script.requests) == 2


def test_a_busy_main_thread_is_never_retried_for_a_write(
    start_script: Any, connect: Any
) -> None:
    """Verify ambiguous write timeouts are treated as potentially applied.

    This is the same rule that governs a dropped connection, applied to the same
    uncertainty: a second write could take effect as well as the first.
    """
    script = start_script(replies(budget_expired("lom_set"), success({"after": 0.5})))
    client = connect(script)
    with pytest.raises(AbletonCommandError):
        client.send("lom_set", {"path": "song.tempo", "value": 120})
    assert len(script.requests) == 1, "a write must not be sent twice"


def test_an_ordinary_live_error_is_not_retried(start_script: Any, connect: Any) -> None:
    """A refusal is not a delay. Only the budget field marks the retryable case."""
    script = start_script(
        replies(
            failure("live_error", "Main track has no 'mute' property!"),
            success({"value": True}),
        )
    )
    client = connect(script)
    with pytest.raises(AbletonCommandError):
        client.send("lom_get", {"path": "song.master_track.mute"})
    assert len(script.requests) == 1
