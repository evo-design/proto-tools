"""HTTP polling helper for submit-and-poll bio API patterns."""

import logging
import time
from collections.abc import Callable
from typing import Any

import requests

logger = logging.getLogger(__name__)


StatusExtractor = Callable[[requests.Response], "tuple[str, Any]"]


def _extract_json_status(response: requests.Response) -> "tuple[str, dict[str, Any]]":
    """Default extractor: read ``payload['status']`` from a JSON body.

    Args:
        response (requests.Response): Status-endpoint response.

    Returns:
        tuple[str, dict[str, Any]]: ``(status_value, full_payload)``.

    Raises:
        ValueError: When the payload lacks a string ``status`` key.
    """
    payload: dict[str, Any] = response.json()
    status = payload.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError(f"Status payload from {response.url} missing string 'status' key: {payload!r}")
    return status, payload


def extract_text_status(response: requests.Response) -> "tuple[str, str]":
    """Plain-text extractor: read the body as the status sentinel.

    Use for APIs that return status as plain text (e.g. iprscan5).

    Args:
        response (requests.Response): Status-endpoint response.

    Returns:
        tuple[str, str]: ``(status, status)`` — same string twice since
            there's no separate payload to return.

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
    timeout_seconds: float = 600.0,
    success_states: frozenset[str] = frozenset({"COMPLETE"}),
    failure_states: frozenset[str] = frozenset({"ERROR"}),
    status_extractor: StatusExtractor = _extract_json_status,
) -> Any:
    """Poll a status endpoint until a terminal state is reached.

    Defaults match the Foldseek-shaped API (JSON body, ``COMPLETE`` /
    ``ERROR`` sentinels). Override the extractor + state sets for
    plain-text or differently-named protocols (e.g. iprscan5).

    Args:
        session (requests.Session): HTTP session for polling GETs.
        status_url (str): URL whose body advertises the job status.
        poll_interval_seconds (float): Delay between polls.
        timeout_seconds (float): Wall-clock cap.
        success_states (frozenset[str]): Terminal-success state values.
        failure_states (frozenset[str]): Terminal-failure state values.
        status_extractor (StatusExtractor): Maps a Response to
            ``(status, payload)``.

    Returns:
        Any: The extractor's payload on a success-state response.

    Raises:
        ValueError: When the status matches one of ``failure_states``.
        TimeoutError: When ``timeout_seconds`` elapses before a terminal
            state is reached.

    Note:
        4xx ``HTTPError`` and malformed-URL exceptions propagate
        immediately — retry can't recover. Transient errors
        (``ConnectionError`` / ``Timeout`` / post-retry 5xx /
        ``JSONDecodeError``) retry until the deadline.
    """
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            response = session.get(status_url, timeout=15.0)
            response.raise_for_status()
            status, payload = status_extractor(response)
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else None
            if code is not None and 400 <= code < 500:
                raise
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timeout after {timeout_seconds}s polling {status_url}; last error={exc!r}"
                ) from exc
            logger.warning("HTTP %s polling %s — retrying after %ss", code, status_url, poll_interval_seconds)
            time.sleep(poll_interval_seconds)
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
            time.sleep(poll_interval_seconds)
            continue

        if status in success_states:
            return payload
        if status in failure_states:
            raise ValueError(f"Job failed at {status_url}: {payload}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timeout after {timeout_seconds}s polling {status_url}; last status={status!r}")

        logger.debug("Polling %s — status=%s, sleeping %ss", status_url, status, poll_interval_seconds)
        time.sleep(poll_interval_seconds)
