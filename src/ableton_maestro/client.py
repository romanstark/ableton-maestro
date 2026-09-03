"""TCP transport client for the Ableton Maestro Remote Script.

Implements the wire protocol defined in docs/protocol.md for local loopback
communication with Ableton Live.

Design principles:
- Synchronous serial transport: Requests are serialized via a thread lock to
  match Ableton Live's single main-thread execution model (docs/protocol.md §3).
- Frameless JSON stream parsing: Decodes incoming UTF-8 bytes incrementally and
  preserves unparsed buffer data across receive boundaries (docs/protocol.md §2).
- Strict socket recycling: Closes and reopens the socket following timeouts or
  malformed replies to prevent desynchronization from delayed responses.
"""

from __future__ import annotations

import argparse
import codecs
import json
import logging
import socket
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any, Self

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "ERROR_CODES",
    "READ_ONLY_HANDLERS",
    "AbletonClient",
    "AbletonCommandError",
    "AbletonConnectionError",
    "AbletonError",
    "AbletonProtocolError",
    "AbletonTimeoutError",
    "is_read_only",
    "main",
]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Protocol constants (docs/protocol.md §1, §4, §8)
# --------------------------------------------------------------------------- #

#: Loopback binding. The communication channel is unauthenticated local TCP (docs/protocol.md §1).
DEFAULT_HOST = "127.0.0.1"

#: Default port for Ableton Maestro (docs/protocol.md §1).
DEFAULT_PORT = 9878

#: Size of socket receive chunks in bytes.
RECV_CHUNK = 8192

#: Timeout for establishing the TCP connection in seconds.
CONNECT_TIMEOUT = 5.0

#: Default timeout for read-only handlers in seconds (docs/protocol.md §8).
READ_TIMEOUT = 10.0

#: Default timeout for write handlers in seconds (docs/protocol.md §8).
WRITE_TIMEOUT = 20.0

#: Optional per-handler timeout overrides (docs/protocol.md §8).
HANDLER_TIMEOUTS: dict[str, float] = {}

#: Handlers that are safe to retry once upon connection loss (docs/protocol.md §8).
#: Write handlers are excluded to avoid duplicate modifications.
#: Note: Retrying events_drain may return fewer events as the initial attempt drains the ring
#: buffer.
READ_ONLY_HANDLERS: frozenset[str] = frozenset(
    {
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
)

#: Standard error codes returned by the Remote Script (docs/protocol.md §4).
ERROR_CODES: frozenset[str] = frozenset(
    {
        "unknown_handler",
        "bad_path",
        "no_such_path",
        "index_out_of_range",
        "not_settable",
        "method_not_allowed",
        "type_error",
        "live_error",
        "internal",
        "skipped",
    }
)

#: Fallback code used if the script returns status "error" without a specific code.
UNSPECIFIED_CODE = "unspecified"

#: Response field indicating that the main-thread execution budget expired in Live.
BUDGET_EXPIRED_FIELD = "timeout_seconds"


def _budget_expired(error: AbletonCommandError) -> bool:
    """Return True if error indicates temporary main-thread execution timeout."""
    return error.code == "live_error" and BUDGET_EXPIRED_FIELD in (error.response or {})


def is_read_only(handler: str) -> bool:
    """Return True if the handler is read-only and safe to retry."""
    return handler in READ_ONLY_HANDLERS


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class AbletonError(Exception):
    """Base exception for all errors raised by the Ableton transport client."""


class AbletonConnectionError(AbletonError):
    """Raised when the TCP connection cannot be established or drops."""


class AbletonTimeoutError(AbletonError):
    """Raised when no complete response is received within the timeout window.

    Attributes:
        handler: The handler name that was waiting for a reply.
        timeout: The timeout duration in seconds.
        may_have_landed: True if the operation was a write that might have completed in Live.
    """

    def __init__(self, handler: str, timeout: float, message: str) -> None:
        self.handler = handler
        self.timeout = timeout
        self.may_have_landed = not is_read_only(handler)
        super().__init__(message)


class AbletonProtocolError(AbletonError):
    """Raised when Remote Script reply violates expected wire protocol format."""


class AbletonCommandError(AbletonError):
    """Raised when the Remote Script executes a handler and returns status 'error'.

    Attributes:
        code: Structured error code (docs/protocol.md §4).
        message: Human-readable error description from Live.
        handler: The handler that was executed.
        params: Parameters sent with the request.
        path: Offending LOM path if provided.
        response: Complete response dictionary from the Remote Script.
    """

    def __init__(
        self,
        handler: str,
        params: Mapping[str, Any] | None,
        response: Mapping[str, Any],
    ) -> None:
        self.handler = handler
        self.params = dict(params or {})
        self.response = dict(response)
        self.code = str(response.get("code") or UNSPECIFIED_CODE)
        self.message = str(response.get("message") or "(no message supplied)")
        self.path = response.get("path")
        where = f" at {self.path}" if self.path else ""
        super().__init__(f"{handler} failed [{self.code}]{where}: {self.message}")


def _merge_connection_errors(
    first: AbletonConnectionError | None, current: AbletonConnectionError
) -> AbletonConnectionError:
    """Preserve the initial connection error message when a retry attempt also fails."""
    if first is None:
        return current
    return AbletonConnectionError(f"{first} | retry failed as well: {current}")


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class AbletonClient:
    """Synchronous TCP client communicating with the Ableton Maestro Remote Script.

    Example:
        >>> with AbletonClient() as live:  # doctest: +SKIP
        ...     tempo = live.get("song.tempo")["value"]

    All exchanges acquire an internal reentrant lock to serialize requests across threads,
    ensuring strictly serial transport over the underlying TCP socket.

    Args:
        host: Target host IP (default 127.0.0.1).
        port: Target port (default 9878).
        read_timeout: Timeout for read-only operations in seconds.
        write_timeout: Timeout for write operations in seconds.
        connect_timeout: Timeout for opening the socket in seconds.
        auto_connect: Automatically connect on first request and reconnect after disconnects.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        read_timeout: float = READ_TIMEOUT,
        write_timeout: float = WRITE_TIMEOUT,
        connect_timeout: float = CONNECT_TIMEOUT,
        auto_connect: bool = True,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.read_timeout = float(read_timeout)
        self.write_timeout = float(write_timeout)
        self.connect_timeout = float(connect_timeout)
        self.auto_connect = bool(auto_connect)

        self._sock: socket.socket | None = None
        self._lock = threading.RLock()
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._rx_buffer = ""

    # ------------------------------------------------------------- lifecycle
    @property
    def connected(self) -> bool:
        """Return True if a socket is currently open."""
        return self._sock is not None

    def connect(self) -> None:
        """Establish the TCP connection to the Remote Script.

        Raises:
            AbletonConnectionError: If the socket cannot connect to Live.
        """
        with self._lock:
            if self._sock is not None:
                return
            try:
                sock = socket.create_connection(
                    (self.host, self.port), timeout=self.connect_timeout
                )
            except OSError as exc:
                raise AbletonConnectionError(
                    f"No connection to {self.host}:{self.port} "
                    f"({exc.__class__.__name__}: {exc}). Is Ableton Live running, and is "
                    "the Ableton Maestro control surface selected under Preferences -> "
                    "Link, Tempo & MIDI? A changed Remote Script needs __pycache__ deleted "
                    "and Live restarted before it is loaded at all."
                ) from exc

            sock.settimeout(self.read_timeout)
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:  # pragma: no cover
                logger.debug("TCP_NODELAY not available on this socket")

            self._sock = sock
            self._reset_rx_state()
            logger.debug("connected to %s:%d", self.host, self.port)

    def close(self) -> None:
        """Close the socket connection and reset receive buffers."""
        with self._lock:
            sock, self._sock = self._sock, None
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
                logger.debug("connection closed")
            self._reset_rx_state()

    disconnect = close

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _reset_rx_state(self) -> None:
        """Reset the incremental UTF-8 decoder and receive buffer."""
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._rx_buffer = ""

    # ------------------------------------------------------------ one request
    def timeout_for(self, handler: str) -> float:
        """Return the default timeout duration in seconds for a specific handler."""
        base = self.read_timeout if is_read_only(handler) else self.write_timeout
        floor = HANDLER_TIMEOUTS.get(handler)
        return base if floor is None else max(base, floor)

    def send(
        self,
        handler: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send one request to the Remote Script and return its response.

        Blocks until the reply arrives or the timeout expires.

        Args:
            handler: Protocol handler name (e.g. 'lom_get', 'lom_set').
            params: Optional dictionary of parameters for the handler.
            timeout: Optional timeout override in seconds for this request.

        Returns:
            The unwrapped dictionary contained in the response's 'result' field.

        Raises:
            AbletonCommandError: If the Remote Script returns an error status.
            AbletonTimeoutError: If the request times out before a complete response arrives.
            AbletonConnectionError: If the socket is not connected or disconnects during transfer.
            AbletonProtocolError: If the reply cannot be parsed into a valid protocol response.
        """
        if params is not None and not isinstance(params, Mapping):
            raise TypeError("params must be a mapping; the script calls params.get().")

        effective_timeout = self.timeout_for(handler) if timeout is None else float(timeout)
        payload = json.dumps(
            {"type": handler, "params": dict(params or {})}, ensure_ascii=False
        ).encode("utf-8")

        retries_left = 1 if is_read_only(handler) else 0
        budget_retries_left = 1 if is_read_only(handler) else 0
        first_failure: AbletonConnectionError | None = None

        with self._lock:
            while True:
                if self._sock is None:
                    if not self.auto_connect and first_failure is None:
                        raise AbletonConnectionError(
                            "Not connected. Call connect(), or construct with auto_connect=True."
                        )
                    try:
                        self.connect()
                    except AbletonConnectionError as exc:
                        raise _merge_connection_errors(first_failure, exc) from exc

                try:
                    self._sendall(payload, handler, effective_timeout)
                    response = self._receive_json(effective_timeout, handler)
                except AbletonConnectionError as exc:
                    self.close()
                    if retries_left <= 0:
                        raise _merge_connection_errors(first_failure, exc) from exc
                    retries_left -= 1
                    first_failure = exc
                    logger.debug("connection lost; retrying read-only handler '%s'", handler)
                    continue
                except (AbletonTimeoutError, AbletonProtocolError):
                    self.close()
                    raise

                try:
                    return self._unwrap(handler, params, response)
                except AbletonCommandError as exc:
                    if budget_retries_left <= 0 or not _budget_expired(exc):
                        raise
                    budget_retries_left -= 1
                    logger.debug(
                        "Live main-thread budget expired; retrying read-only handler '%s'", handler
                    )
                    continue
                except AbletonProtocolError:
                    self.close()
                    raise

    def _sendall(self, payload: bytes, handler: str, timeout: float) -> None:
        """Transmit raw bytes to the TCP socket."""
        sock = self._sock
        if sock is None:  # pragma: no cover
            raise AbletonConnectionError("Not connected.")
        try:
            sock.settimeout(timeout)
            sock.sendall(payload)
        except TimeoutError as exc:
            raise AbletonTimeoutError(
                handler,
                timeout,
                f"Timed out sending '{handler}': Live is not accepting data.",
            ) from exc
        except OSError as exc:
            raise AbletonConnectionError(
                f"Send failed ({exc.__class__.__name__}: {exc}). "
                "Live most likely closed the connection."
            ) from exc

    def _receive_json(self, timeout: float, handler: str) -> dict[str, Any]:
        """Read data from the socket until a complete JSON object is parsed.

        Note: Over a frameless TCP stream, invalid JSON is indistinguishable from incomplete JSON.
        Malformed responses will therefore surface as AbletonTimeoutError rather than protocol
        errors.
        """
        sock = self._sock
        if sock is None:  # pragma: no cover
            raise AbletonConnectionError("Not connected.")

        deadline = time.monotonic() + timeout
        decoder = json.JSONDecoder()

        buffer = self._rx_buffer
        self._rx_buffer = ""

        while True:
            stripped = buffer.lstrip()
            if stripped:
                try:
                    value, index = decoder.raw_decode(stripped)
                except ValueError:
                    pass  # JSON is incomplete, continue reading from socket
                else:
                    rest = stripped[index:].lstrip()
                    if rest:
                        logger.warning(
                            "%d characters left over after the reply to '%s': "
                            "kept for the next read, but the connection may be desynchronised",
                            len(rest),
                            handler,
                        )
                        self._rx_buffer = rest
                    if not isinstance(value, dict):
                        raise AbletonProtocolError(
                            f"Reply to '{handler}' was not a JSON object but "
                            f"{type(value).__name__}: {value!r}"
                        )
                    return value

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AbletonTimeoutError(
                    handler,
                    timeout,
                    f"No complete reply to '{handler}' after {timeout:.1f}s "
                    f"({len(buffer)} characters buffered, still not valid JSON). "
                    "A larger client timeout only helps while Live is genuinely "
                    "still working; the script's own queue timeout cuts first.",
                )

            try:
                sock.settimeout(remaining)
                chunk = sock.recv(RECV_CHUNK)
            except TimeoutError as exc:
                raise AbletonTimeoutError(
                    handler,
                    timeout,
                    f"No reply to '{handler}' after {timeout:.1f}s.",
                ) from exc
            except OSError as exc:
                raise AbletonConnectionError(
                    f"Read failed ({exc.__class__.__name__}: {exc})."
                ) from exc

            if not chunk:
                raise AbletonConnectionError(
                    f"Live closed the connection during '{handler}' (EOF). The script "
                    "ends a client handler after an internal error. Live's Log.txt "
                    "carries the reason."
                )

            buffer += self._decoder.decode(chunk)

    @staticmethod
    def _unwrap(
        handler: str, params: Mapping[str, Any] | None, response: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Validate response status and return the result dictionary.

        Always branch on status == "error" rather than checking for the presence of the "result"
        key,
        as the Remote Script initializes and sends "result: {}" even on error paths.
        """
        status = response.get("status")
        if status == "error":
            raise AbletonCommandError(handler, params, response)
        if status != "success":
            raise AbletonProtocolError(
                f"Unexpected 'status' in the reply to '{handler}': {status!r} "
                f"(full response: {dict(response)!r})"
            )

        result = response.get("result", {})
        if result is None:
            return {}
        if not isinstance(result, dict):
            raise AbletonProtocolError(
                f"'result' for '{handler}' was {type(result).__name__}, not an object. "
                "Protocol §4: every Ableton Maestro result is an object, so that it can "
                "grow a field without breaking callers."
            )
        return dict(result)

    # ------------------------------------------------------- handler wrappers

    def ping(self, *, timeout: float | None = None) -> dict[str, Any]:
        """Send a liveness check to the Remote Script (docs/protocol.md §5.1).

        Returns:
            Dictionary containing 'pong', 'script_version', and 'uptime'.
        """
        return self.send("ping", timeout=timeout)

    def script_info(self, *, timeout: float | None = None) -> dict[str, Any]:
        """Script capabilities, versions, and Live version (docs/protocol.md §5.2)."""
        return self.send("script_info", timeout=timeout)

    def get(self, path: str, *, timeout: float | None = None) -> dict[str, Any]:
        """Read a property value at the given LOM path (docs/protocol.md §5.3).

        Returns:
            Dictionary containing 'path', 'value', 'type', and optional 'display' string.
        """
        return self.send("lom_get", {"path": path}, timeout=timeout)

    def set(self, path: str, value: Any, *, timeout: float | None = None) -> dict[str, Any]:
        """Write a property at the given LOM path and verify via read-back.

        See docs/protocol.md §5.4 for details.

        Returns:
            Dictionary containing 'path', 'requested', 'before', 'after', 'clamped', 'changed',
            and optional 'display' representation.
        """
        return self.send("lom_set", {"path": path, "value": value}, timeout=timeout)

    def call(
        self,
        path: str,
        method: str,
        args: Sequence[Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Invoke an allowlisted method on a Live object (docs/protocol.md §5.5).

        Args:
            path: Target LOM object path.
            method: Method name to call.
            args: Positional argument list.

        Returns:
            Dictionary containing 'path', 'method', and 'result'.
        """
        params: dict[str, Any] = {"path": path, "method": method}
        if args is not None:
            params["args"] = list(args)
        return self.send("lom_call", params, timeout=timeout)

    def describe(
        self, path: str, *, depth: int | None = None, timeout: float | None = None
    ) -> dict[str, Any]:
        """Introspect an object's properties, children, and methods.

        See docs/protocol.md §5.6 for details.
        """
        params: dict[str, Any] = {"path": path}
        if depth is not None:
            params["depth"] = int(depth)
        return self.send("lom_describe", params, timeout=timeout)

    def batch(
        self,
        ops: Sequence[Mapping[str, Any]],
        *,
        atomic: bool = False,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Execute multiple operations in a single round-trip exchange.

        See docs/protocol.md §5.7 for details.

        Args:
            ops: List of operation dictionaries.
            atomic: If True, execution stops at the first encountered error.

        Returns:
            Dictionary containing 'results', 'ok_count', and 'error_count'.
        """
        params: dict[str, Any] = {"ops": [dict(op) for op in ops]}
        if atomic:
            params["atomic"] = True
        return self.send("lom_batch", params, timeout=timeout)


# --------------------------------------------------------------------------- #
# CLI diagnostics
# --------------------------------------------------------------------------- #


def _parse_value(raw: str) -> Any:
    """Parse a CLI string into JSON, falling back to raw string on ValueError."""
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ableton_maestro.client",
        description="Diagnostic CLI tool for the Ableton Maestro Remote Script socket.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"default {DEFAULT_HOST}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"default {DEFAULT_PORT}")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="override timeout for this command in seconds (default: protocol §8)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable verbose transport logging"
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ping", help="send ping liveness check")
    sub.add_parser("info", help="retrieve script and Live versions")

    get_parser = sub.add_parser("get", help="read one LOM property")
    get_parser.add_argument("path", help="e.g. song.tempo or song.tracks[0].name")

    set_parser = sub.add_parser("set", help="write one LOM property with read-back verification")
    set_parser.add_argument("path", help="e.g. song.tempo")
    set_parser.add_argument("value", help="value in JSON format, or raw string")

    return parser


def _run(client: AbletonClient, args: argparse.Namespace) -> dict[str, Any]:
    timeout = args.timeout
    if args.command == "ping":
        started = time.monotonic()
        result = client.ping(timeout=timeout)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        print(f"round trip {elapsed_ms:.2f} ms", file=sys.stderr)
        return result
    if args.command == "info":
        return client.script_info(timeout=timeout)
    if args.command == "get":
        return client.get(args.path, timeout=timeout)
    if args.command == "set":
        return client.set(args.path, _parse_value(args.value), timeout=timeout)
    raise ValueError(f"unknown command: {args.command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI diagnostic entry point."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    client = AbletonClient(args.host, args.port)
    try:
        result = _run(client, args)
    except AbletonCommandError as exc:
        print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        return 1
    except AbletonTimeoutError as exc:
        print(f"timeout: {exc}", file=sys.stderr)
        if exc.may_have_landed:
            print(
                "This was a write. A timed-out write may still have landed. "
                "verify with a read, do not repeat it (protocol §8).",
                file=sys.stderr,
            )
        return 1
    except AbletonError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (TypeError, ValueError) as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("aborted", file=sys.stderr)
        return 130
    finally:
        client.close()

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
