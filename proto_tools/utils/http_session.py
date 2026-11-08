"""proto_tools/utils/http_session.py.

Shared HTTP session builder with retry logic.

Under the tool's ambient deadline (:mod:`proto_tools.utils.deadlines`), sessions from
``build_http_session`` end connects, response bodies, retries and ``Retry-After`` waits at it. A
``requests`` ``timeout=`` alone bounds each socket read, not the request, so it cannot do this.
Response headers, and urllib3's drain of an error body it is about to retry, remain bounded only
per socket read.

``request_with_retry`` below and ``proto_tools.proto._client._request_with_retry`` both retry on
``requests.exceptions.ConnectionError`` but aren't merged: the client's version also retries HTTP
status codes and needs the response in hand to read ``Retry-After``, which a generic callable
can't give it.
"""

import random
import time
from collections.abc import Callable, Iterator, Mapping
from importlib.metadata import PackageNotFoundError, version
from types import TracebackType
from typing import Any, TypeVar

import requests
from requests.adapters import HTTPAdapter
from typing_extensions import Self
from urllib3.connectionpool import ConnectionPool
from urllib3.exceptions import DecodeError, ProtocolError, ReadTimeoutError, SSLError
from urllib3.response import BaseHTTPResponse
from urllib3.util.retry import Retry
from urllib3.util.timeout import Timeout

from proto_tools.utils.base_config import client_identity
from proto_tools.utils.deadlines import DeadlineExceededError, expired, remaining, sleep_within_deadline

_RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

#: Added to every clamped socket timeout so the socket gives up just after the deadline, not before.
_DEADLINE_GRACE_SECONDS = 0.02
_BODY_CHUNK_BYTES = 64 * 1024
#: Keeps a server's error page from flooding the exception message.
_ERROR_BODY_CHARS = 1000

#: Spreads retries across callers hitting the same failure at once; not a security primitive.
_JITTER_FRACTION = 0.1

T = TypeVar("T")


class _DeadlineSignal(Exception):
    """Stop urllib3 at the deadline.

    Deliberately not an ``OSError``: requests' ``HTTPAdapter.send`` re-raises any ``OSError``
    (including ``TimeoutError``) as a retryable ``requests.ConnectionError``. ``DeadlineAdapter``
    turns this into :class:`DeadlineExceededError` instead.
    """


def _clamp(seconds: Any, cap: float) -> float:
    """Return ``seconds`` capped at ``cap``; an unset or infinite timeout becomes ``cap``."""
    return min(seconds, cap) if isinstance(seconds, int | float) else cap


class _DeadlineTimeout(Timeout):
    """A urllib3 ``Timeout`` re-clamped to the remaining deadline at the start of every attempt.

    urllib3 calls ``clone()`` once per attempt, internal retries included, and uses the clone.
    """

    def clone(self) -> Timeout:
        """Return a plain ``Timeout`` whose connect and read waits end at the deadline."""
        left = remaining()
        if left is None:
            return Timeout(connect=self._connect, read=self._read, total=self.total)
        if left <= 0:
            raise _DeadlineSignal
        cap = left + _DEADLINE_GRACE_SECONDS
        return Timeout(connect=_clamp(self._connect, cap), read=_clamp(self._read, cap))


class DeadlineRetry(Retry):
    """urllib3 ``Retry`` that stops at the tool's deadline.

    Without a deadline it behaves exactly like ``Retry``. With one, it never starts a retry after
    the deadline, and a wait (backoff or ``Retry-After``) that would pass it ends at the deadline. ``Retry.new()`` rebuilds
    retries as ``type(self)(**params)``, so the subclass survives every increment.
    """

    def increment(
        self,
        method: str | None = None,
        url: str | None = None,
        response: BaseHTTPResponse | None = None,
        error: Exception | None = None,
        _pool: ConnectionPool | None = None,
        _stacktrace: TracebackType | None = None,
    ) -> Self:
        """Count a retry, or stop when the deadline has passed."""
        if expired():
            raise _DeadlineSignal
        return super().increment(method, url, response, error, _pool, _stacktrace)

    def _check_wait(self, seconds: float | None) -> None:
        """Stop at the deadline, sleeping out the time left, when waiting ``seconds`` would pass it."""
        left = remaining()
        if left is not None and seconds and seconds >= left:
            time.sleep(max(left, 0.0))
            raise _DeadlineSignal

    def sleep_for_retry(self, response: Any = None) -> bool:
        """Honor ``Retry-After`` only when the wait fits before the deadline."""
        if response is not None:
            self._check_wait(self.get_retry_after(response))
        return super().sleep_for_retry(response)

    def sleep(self, response: Any = None) -> None:
        """Back off between attempts, or stop when the wait would pass the deadline."""
        retry_after = self.get_retry_after(response) if self.respect_retry_after_header and response else None
        # urllib3 backs off instead of honoring a Retry-After of 0, so check whichever wait it takes.
        self._check_wait(retry_after or self.get_backoff_time())
        super().sleep(response)


def _as_timeout(timeout: Any) -> _DeadlineTimeout:
    """Normalize a requests ``timeout`` (None, seconds, ``(connect, read)`` or ``Timeout``)."""
    if isinstance(timeout, Timeout):
        return _DeadlineTimeout(connect=timeout.connect_timeout, read=timeout.read_timeout)
    connect, read = timeout if isinstance(timeout, tuple) else (timeout, timeout)
    return _DeadlineTimeout(connect=connect, read=read)


def _timed_out(what: str) -> DeadlineExceededError:
    """Build the error raised when ``what`` runs into the deadline."""
    return DeadlineExceededError(f"Timed out {what}: the tool's timeout elapsed")


def _iter_raw(response: requests.Response, chunk_size: int, what: str) -> Iterator[bytes]:
    """Yield a streamed body as it arrives, giving up at the deadline.

    ``read1`` returns whatever bytes are available instead of blocking for a full chunk, and the
    socket timeout is re-clamped before each read, so a server sending one byte at a time cannot
    outlast the deadline. Errors are mapped as ``Response.iter_content`` maps them.
    """
    raw = response.raw
    sock = getattr(getattr(raw, "connection", None), "sock", None)
    base = sock.gettimeout() if sock is not None else None
    try:
        while True:
            left = remaining()
            if left is not None and left <= 0:
                raise _timed_out(what)
            if sock is not None and left is not None:
                sock.settimeout(_clamp(base, left + _DEADLINE_GRACE_SECONDS))
            chunk = raw.read1(chunk_size, decode_content=True)
            if not chunk:
                break
            yield chunk
    except DeadlineExceededError:
        response.close()
        raise
    except (ProtocolError, DecodeError, ReadTimeoutError, SSLError, OSError) as exc:
        response.close()
        if expired():
            raise _timed_out(what) from exc
        if isinstance(exc, ProtocolError):
            raise requests.exceptions.ChunkedEncodingError(exc) from exc
        if isinstance(exc, DecodeError):
            raise requests.exceptions.ContentDecodingError(exc) from exc
        if isinstance(exc, SSLError):
            raise requests.exceptions.SSLError(exc) from exc
        raise requests.exceptions.ConnectionError(exc) from exc
    raw.release_conn()


def iter_content_within_deadline(response: requests.Response, chunk_size: int) -> Iterator[bytes]:
    """Iterate a ``stream=True`` response body, giving up at the tool's deadline.

    ``Response.iter_content`` blocks until a whole chunk arrives and its socket timeout only
    bounds each read, so a slow server can hold it open indefinitely. Use this instead for
    streamed downloads.

    Args:
        response (requests.Response): Response from a ``stream=True`` request.
        chunk_size (int): Most bytes to yield at once.

    Yields:
        bytes: Decoded body chunks.

    Raises:
        DeadlineExceededError: When the deadline passes before the body ends.
    """
    if remaining() is None:
        yield from response.iter_content(chunk_size)
        return
    yield from _iter_raw(response, chunk_size, f"downloading {response.url}")


class DeadlineAdapter(HTTPAdapter):
    """``HTTPAdapter`` that ends a request's connect, body and retries at the tool's deadline.

    Without a deadline it behaves exactly like ``HTTPAdapter``. With one, it clamps connect and
    read timeouts to the time left, reads non-streamed bodies itself so a trickling server cannot
    outlast the deadline, and raises :class:`DeadlineExceededError` (never a retryable
    ``requests.ConnectionError``) once the deadline passes.
    """

    def send(
        self,
        request: requests.PreparedRequest,
        stream: bool = False,
        timeout: Any = None,
        verify: bool | str = True,
        cert: Any = None,
        proxies: Mapping[str, str] | None = None,
    ) -> requests.Response:
        """Send ``request`` within the deadline; see the class docstring."""
        if remaining() is None:
            return super().send(request, stream=stream, timeout=timeout, verify=verify, cert=cert, proxies=proxies)
        what = f"requesting {request.url}"
        if expired():
            raise _timed_out(what)
        try:
            response = super().send(
                request,
                stream=True,
                timeout=_as_timeout(timeout),  # type: ignore[arg-type]  # requests passes a urllib3 Timeout through
                verify=verify,
                cert=cert,
                proxies=proxies,
            )
        except _DeadlineSignal:
            raise _timed_out(what) from None
        except requests.RequestException as exc:
            if expired():
                raise _timed_out(what) from exc
            raise
        if not stream:
            response._content = b"".join(_iter_raw(response, _BODY_CHUNK_BYTES, what))
            response._content_consumed = True  # type: ignore[attr-defined]  # set in Response.__init__, unstubbed
        return response


def _proto_tools_version() -> str:
    """Return the installed proto-tools version, or ``unknown`` where there is no distribution.

    A Modal container puts proto-tools on the path by mounting the source tree rather than
    installing it, so it carries no dist-info and ``version()`` raises. The version only decorates
    a ``User-Agent``, so a request that would otherwise succeed must not fail for the want of it.
    """
    try:
        return version("proto-tools")
    except PackageNotFoundError:
        return "unknown"


def user_agent_for(identity: str | None = None) -> str:
    """Build a ``User-Agent`` naming proto-tools and the caller, so each caller is identifiable.

    Args:
        identity (str | None): Identifies the caller when the request originates elsewhere, such as
            a hosted process calling on a user's behalf. Derived from this process when omitted.

    Returns:
        str: A user agent naming proto-tools and the caller.
    """
    return f"proto-tools/{_proto_tools_version()} ({identity or client_identity()})"


def build_http_session(
    http_retries: int,
    backoff_seconds: float,
    user_agent: str,
    allowed_methods: list[str] | None = None,
    mount_http: bool = False,
    status_forcelist: list[int] | None = None,
) -> requests.Session:
    """Build a requests session whose retries and requests stay within the tool's deadline.

    Args:
        http_retries (int): Retries for connection errors and retryable statuses.
        backoff_seconds (float): urllib3 backoff factor between retries.
        user_agent (str): ``User-Agent`` header for every request.
        allowed_methods (list[str] | None): Methods that may be retried. Defaults to ``["GET"]``.
        mount_http (bool): Also mount the adapter for plain ``http://`` URLs.
        status_forcelist (list[int] | None): Statuses the adapter retries. Defaults to 429 and
            5xx; ``[]`` returns every response to the caller, for callers that retry themselves.
    """
    retry = DeadlineRetry(
        total=http_retries,
        connect=http_retries,
        backoff_factor=backoff_seconds,
        status_forcelist=_RETRY_STATUS_CODES if status_forcelist is None else status_forcelist,
        allowed_methods=allowed_methods or ["GET"],
    )
    adapter = DeadlineAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    if mount_http:
        session.mount("http://", adapter)
    session.headers.update({"User-Agent": user_agent})
    return session


def raise_for_status_with_body(response: requests.Response, operation: str) -> None:
    """Raise ``requests.HTTPError`` naming *operation* and quoting the start of the response body.

    ``response.raise_for_status()`` reports only the status line, but APIs explain a rejected
    query in the body, so the body is the part a caller needs to fix the request.

    Args:
        response (requests.Response): Response to check.
        operation (str): What the request was doing, such as ``"UniProt search"``.

    Raises:
        requests.HTTPError: If the response has an error status.
    """
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        body = str(getattr(response, "text", ""))[:_ERROR_BODY_CHARS]
        raise requests.HTTPError(f"{operation} failed: {error}; body={body!r}", response=response) from error


def request_with_retry(
    call: Callable[[], T],
    *,
    retries: int,
    backoff_seconds: float,
    retryable_exceptions: tuple[type[BaseException], ...] = (requests.exceptions.ConnectionError,),
) -> T:
    """Retry *call* on a transient failure, with the same backoff the session's own retries use.

    ``HTTPAdapter``'s ``Retry`` only covers what urllib3 catches inside its own connection pool.
    A server that closes a pooled keep-alive connection raises ``ConnectionResetError`` while
    urllib3 is *sending* the next request on it, which surfaces as
    ``requests.exceptions.ConnectionError`` above the adapter rather than through it. This retries
    that case explicitly instead.

    Args:
        call (Callable[[], T]): Zero-argument callable making the request; wraps any
            ``session.get``/``.post`` call.
        retries (int): Extra attempts after the first.
        backoff_seconds (float): Delay before the first retry, doubling (and jittered) after each
            attempt.
        retryable_exceptions (tuple[type[BaseException], ...]): Exceptions treated as transient.
            Defaults to the connection-reset case above.
    """
    try:
        return call()
    except retryable_exceptions:
        if retries <= 0:
            raise
        delay = backoff_seconds * (1 + random.uniform(-_JITTER_FRACTION, _JITTER_FRACTION))  # noqa: S311
        sleep_within_deadline(delay, "retrying a request")
        return request_with_retry(
            call,
            retries=retries - 1,
            backoff_seconds=backoff_seconds * 2,
            retryable_exceptions=retryable_exceptions,
        )
