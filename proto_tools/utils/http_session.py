"""proto_tools/utils/http_session.py

Shared HTTP session builder with retry logic."""

from __future__ import annotations

from typing import List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_RETRY_STATUS_CODES = [429, 500, 502, 503, 504]


def build_http_session(
    http_retries: int,
    backoff_seconds: float,
    user_agent: str,
    allowed_methods: Optional[List[str]] = None,
    mount_http: bool = False,
) -> requests.Session:
    """Build a requests session with retry adapter."""
    retry = Retry(
        total=http_retries,
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
