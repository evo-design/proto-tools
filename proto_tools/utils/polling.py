"""HTTP polling helper for submit-and-poll bio API patterns (Foldseek)."""

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


def poll_until_complete(
    session: requests.Session,
    status_url: str,
    *,
    poll_interval_seconds: float = 5.0,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    """Poll a JSON status endpoint until status reaches COMPLETE or ERROR.

    Returns the final response payload on COMPLETE; raises ValueError on
    ERROR; raises TimeoutError if timeout_seconds elapses.
    """
    deadline = time.monotonic() + timeout_seconds
    while True:
        response = session.get(status_url, timeout=15.0)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        status = payload.get("status")

        if status == "COMPLETE":
            return payload
        if status == "ERROR":
            raise ValueError(f"Job failed at {status_url}: {payload}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timeout after {timeout_seconds}s polling {status_url}; last status={status!r}")

        logger.debug("Polling %s — status=%s, sleeping %ss", status_url, status, poll_interval_seconds)
        time.sleep(poll_interval_seconds)
