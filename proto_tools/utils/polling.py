"""Poll biological APIs for the status of submitted jobs."""

import logging
import math
import time
from collections.abc import Callable
from typing import Any

import requests

from proto_tools.utils.deadlines import sleep_within_deadline

logger = logging.getLogger(__name__)


StatusExtractor = Callable[[requests.Response], "tuple[str, Any]"]


def _extract_json_status(response: requests.Response) -> "tuple[str, dict[str, Any]]":
    """Read the job status and payload from a JSON response.

    Args:
        response (requests.Response): Response from the status endpoint.

    Returns:
        tuple[str, dict[str, Any]]: ``(status_value, full_payload)``.

    Raises:
        ValueError: When the payload has no nonempty string ``status`` value.
    """
    payload: dict[str, Any] = response.json()
    status = payload.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError(f"Status payload from {response.url} missing string 'status' key: {payload!r}")
    return status, payload


def extract_text_status(response: requests.Response) -> "tuple[str, str]":
    """Read the job status from a plain-text response.

    Use for APIs that return status as plain text (e.g. iprscan5).

    Args:
        response (requests.Response): Response from the status endpoint.

    Returns:
        tuple[str, str]: ``(status, status)``. The status also serves as the payload.

    Raises:
        ValueError: When the body is empty.
    """
    text = response.text.strip()
    if not text:
        raise ValueError(f"Empty plain-text status body from {response.url}")
    return text, text


def poll_until_complete(
    session: requests.Session,
    status_url: str,
    *,
    poll_interval_seconds: float = 5.0,
    timeout_seconds: float | None = 600.0,
    success_states: frozenset[str] = frozenset({"COMPLETE"}),
    failure_states: frozenset[str] = frozenset({"ERROR"}),
    status_extractor: StatusExtractor = _extract_json_status,
) -> Any:
    """Poll a status endpoint until the job succeeds or fails.

    By default, read a JSON response with ``COMPLETE`` and ``ERROR`` status
    values, as used by Foldseek. For other APIs, such as iprscan5, supply
    a status extractor and the corresponding success and failure values.

    Args:
        session (requests.Session): HTTP session used to send polling requests.
        status_url (str): URL to query for the job status.
        poll_interval_seconds (float): Delay between polls.
        timeout_seconds (float | None): Time limit for polling, or ``None`` for no polling
            limit. Inside an in-process tool call, the tool's deadline also limits
            every request and sleep.
        success_states (frozenset[str]): Status values that indicate success.
        failure_states (frozenset[str]): Status values that indicate failure.
        status_extractor (StatusExtractor): Function that reads ``(status, payload)``
            from a response.

    Returns:
        Any: The payload returned by the extractor when the job succeeds.

    Raises:
        ValueError: When the status matches one of ``failure_states``.
        TimeoutError: When ``timeout_seconds`` elapses before a terminal
            state is reached.

    Note:
        HTTP 4xx errors and malformed URLs raise immediately. Connection errors,
        request timeouts, HTTP 5xx errors remaining after session retries, and
        JSON decoding errors are retried until the deadline.
    """
    deadline = time.monotonic() + (math.inf if timeout_seconds is None else timeout_seconds)
    while True:
        # Fetch the current status using the API's response format.
        try:
            response = session.get(status_url, timeout=15.0)
            response.raise_for_status()
            status, payload = status_extractor(response)
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else None
            # Raise client errors immediately and retry other HTTP errors until the deadline.
            if code is not None and 400 <= code < 500:
                raise
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timeout after {timeout_seconds}s polling {status_url}; last error={exc!r}"
                ) from exc
            logger.warning("HTTP %s polling %s — retrying after %ss", code, status_url, poll_interval_seconds)
            sleep_within_deadline(poll_interval_seconds, f"polling {status_url}")
            continue
        except (requests.exceptions.InvalidURL, requests.exceptions.MissingSchema, requests.exceptions.URLRequired):
            raise
        except requests.RequestException as exc:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timeout after {timeout_seconds}s polling {status_url}; last error={exc!r}"
                ) from exc
            logger.warning(
                "Transient error polling %s: %r — retrying after %ss", status_url, exc, poll_interval_seconds
            )
            sleep_within_deadline(poll_interval_seconds, f"polling {status_url}")
            continue

        # Return the payload on success, or raise if the job failed or took too long.
        if status in success_states:
            return payload
        if status in failure_states:
            raise ValueError(f"Job failed at {status_url}: {payload}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timeout after {timeout_seconds}s polling {status_url}; last status={status!r}")

        logger.debug("Polling %s — status=%s, sleeping %ss", status_url, status, poll_interval_seconds)
        sleep_within_deadline(poll_interval_seconds, f"polling {status_url}")
