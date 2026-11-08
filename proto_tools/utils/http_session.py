"""proto_tools/utils/http_session.py.

Shared HTTP session builder with retry logic.

``request_with_retry`` below and ``proto_tools.proto._client._request_with_retry`` both retry on
``requests.exceptions.ConnectionError`` but aren't merged: the client's version also retries HTTP
status codes and needs the response in hand to read ``Retry-After``, which a generic callable
can't give it.
"""

import random
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from typing import TypeVar

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from proto_tools.utils.base_config import client_identity

_RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

#: Keeps a server's error page from flooding the exception message.
_ERROR_BODY_CHARS = 1000

#: Spreads retries across callers hitting the same failure at once; not a security primitive.
_JITTER_FRACTION = 0.1

T = TypeVar("T")


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
    """Build a requests session with retry adapter.

    Args:
        http_retries (int): Retries for connection errors and retryable statuses.
        backoff_seconds (float): urllib3 backoff factor between retries.
        user_agent (str): ``User-Agent`` header for every request.
        allowed_methods (list[str] | None): Methods that may be retried. Defaults to ``["GET"]``.
        mount_http (bool): Also mount the adapter for plain ``http://`` URLs.
        status_forcelist (list[int] | None): Statuses the adapter retries. Defaults to 429 and
            5xx; ``[]`` returns every response to the caller, for callers that retry themselves.
    """
    retry = Retry(
        total=http_retries,
        connect=http_retries,
        backoff_factor=backoff_seconds,
        status_forcelist=_RETRY_STATUS_CODES if status_forcelist is None else status_forcelist,
        allowed_methods=allowed_methods or ["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
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
        time.sleep(max(delay, 0.0))
        return request_with_retry(
            call,
            retries=retries - 1,
            backoff_seconds=backoff_seconds * 2,
            retryable_exceptions=retryable_exceptions,
        )
