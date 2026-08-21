"""proto_tools/utils/http_session.py.

Shared HTTP session builder with retry logic.
"""

import time
from collections.abc import Callable
from typing import TypeVar

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

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
) -> T:
    """Retry *call* on a dropped connection, with the same backoff the session's own retries use.

    ``HTTPAdapter``'s ``Retry`` only covers what urllib3 catches inside its own connection pool.
    A server that closes a pooled keep-alive connection between requests raises
    ``ConnectionResetError`` while urllib3 is *sending* the next request on it, which surfaces as
    ``requests.exceptions.ConnectionError`` above the adapter's retry logic rather than through it
    -- seen against ``rest.uniprot.org`` with ``http_retries=2`` already configured and still
    failing outright. Catching it here closes that gap without guessing at a given urllib3 version's
    exact retry coverage.

    Args:
        call (Callable[[], T]): Zero-argument callable making the request, so this has no opinion
            on the request's shape and can wrap any ``session.get``/``.post`` call.
        retries (int): Extra attempts after the first, matching ``http_retries`` on the tool's
            config.
        backoff_seconds (float): Seconds before the first retry, doubling after each attempt after
            that -- the same schedule ``Retry(backoff_factor=...)`` uses.
    """
    try:
        return call()
    except requests.exceptions.ConnectionError:
        if retries <= 0:
            raise
        time.sleep(backoff_seconds)
        return request_with_retry(call, retries=retries - 1, backoff_seconds=backoff_seconds * 2)
