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
from typing import TypeVar

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

#: Spreads retries across callers hitting the same failure at once; not a security primitive.
_JITTER_FRACTION = 0.1

T = TypeVar("T")


def build_http_session(
    http_retries: int,
    backoff_seconds: float,
    user_agent: str,
    allowed_methods: list[str] | None = None,
    mount_http: bool = False,
) -> requests.Session:
    """Build a requests session with retry adapter."""
    retry = Retry(
        total=http_retries,
        connect=http_retries,
        backoff_factor=backoff_seconds,
        status_forcelist=_RETRY_STATUS_CODES,
        allowed_methods=allowed_methods or ["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    if mount_http:
        session.mount("http://", adapter)
    session.headers.update({"User-Agent": user_agent})
    return session


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
