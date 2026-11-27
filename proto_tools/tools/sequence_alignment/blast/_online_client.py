"""NCBI QBLAST submit-and-poll client that identifies its caller."""

import io
import logging
import re
import threading
import time
from typing import Any
from urllib.parse import urlencode

import requests

from proto_tools.utils import (
    DeadlineExceededError,
    build_http_session,
    deadline,
    poll_until_complete,
    remaining,
    request_with_retry,
    sleep_within_deadline,
    user_agent_for,
)

logger = logging.getLogger(__name__)

_URL = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
_HTTP_RETRIES = 2
_BACKOFF_SECONDS = 1.0
_REQUEST_TIMEOUT_SECONDS = 60
# NCBI asks for at least 10 seconds between submissions and at most one poll per search per minute.
_SUBMIT_SPACING_SECONDS = 10.0
_POLL_INTERVAL_SECONDS = 60.0
_REJECTION_RETRY_DELAYS_SECONDS = (60.0, 120.0)
_REJECTION = re.compile(r"Cannot accept request,\s*error code:\s*-99(?![\w.])")
_STATUS = re.compile(r"Status=\s*([A-Za-z]+)")

_submit_lock = threading.Lock()
_next_submit = 0.0


def search_xml(parameters: dict[str, Any], *, timeout: int | None, client_identity: str | None) -> str:
    """Submit a QBLAST search and return its XML result.

    ``timeout`` bounds the whole search, every request and wait included (submission spacing,
    ``-99`` cooldowns, polling), through a deadline (:mod:`proto_tools.utils.deadlines`).

    Args:
        parameters (dict[str, Any]): Uppercase NCBI search parameters.
        timeout (int | None): Total search budget in seconds; None waits indefinitely.
        client_identity (str | None): Caller named in the User-Agent; derived from this process when None.

    Returns:
        str: Raw BLAST XML.

    Raises:
        ValueError: When NCBI rejects the search, reports it failed, or answers with a non-XML page.
        DeadlineExceededError: When the search does not finish within ``timeout``.
    """
    with (
        deadline(timeout),
        build_http_session(
            http_retries=_HTTP_RETRIES,
            backoff_seconds=_BACKOFF_SECONDS,
            user_agent=user_agent_for(client_identity),
            allowed_methods=["GET", "POST"],
        ) as session,
    ):
        logger.update_status("Submitting search to NCBI BLAST")  # type: ignore[attr-defined]
        # Defaults match Biopython's qblast so existing searches return the same hits.
        rid = _submit(
            session,
            {
                "EXPECT": 10.0,
                "HITLIST_SIZE": 50,
                "ENTREZ_QUERY": "(none)",
                **parameters,
                "CMD": "Put",
                "tool": "proto-tools",
            },
        )
        logger.update_status("Waiting for NCBI BLAST results")  # type: ignore[attr-defined]
        sleep_within_deadline(_POLL_INTERVAL_SECONDS, "waiting for NCBI BLAST to run the search")
        query = urlencode({"CMD": "Get", "RID": rid, "FORMAT_TYPE": "XML", "ALIGNMENTS": 500, "DESCRIPTIONS": 500})
        xml: str = poll_until_complete(
            session,
            f"{_URL}?{query}",
            poll_interval_seconds=_POLL_INTERVAL_SECONDS,
            timeout_seconds=None,
            success_states=frozenset({"XML"}),
            failure_states=frozenset(),
            status_extractor=_qblast_status,
        )
    return xml


def _submit(session: requests.Session, parameters: dict[str, Any]) -> str:
    """POST a search and return its request ID, retrying NCBI's ``-99`` rejection after a cooldown."""
    from Bio.Blast import NCBIWWW

    global _next_submit  # noqa: PLW0603 -- process-wide submission spacing, guarded by _submit_lock
    for retry_delay in (*_REJECTION_RETRY_DELAYS_SECONDS, None):
        left = remaining()
        if not _submit_lock.acquire(timeout=-1 if left is None else max(left, 0.0)):
            raise DeadlineExceededError("Timed out waiting to submit an NCBI BLAST search: the tool's timeout elapsed")
        try:
            sleep_within_deadline(max(0.0, _next_submit - time.monotonic()), "spacing NCBI BLAST submissions")
            response = request_with_retry(
                lambda: session.post(_URL, data=parameters, timeout=_REQUEST_TIMEOUT_SECONDS),
                retries=_HTTP_RETRIES,
                backoff_seconds=_BACKOFF_SECONDS,
            )
            _next_submit = time.monotonic() + _SUBMIT_SPACING_SECONDS
        finally:
            _submit_lock.release()
        response.raise_for_status()
        if not _REJECTION.search(response.text):
            rid, _ = NCBIWWW._parse_qblast_ref_page(io.BytesIO(response.content))  # type: ignore[no-untyped-call]
            return str(rid)
        if retry_delay is None:
            break
        logger.warning("NCBI BLAST rejected the search (error code -99); retrying in %.0fs", retry_delay)
        sleep_within_deadline(retry_delay, "waiting out NCBI BLAST's -99 rejection")
    raise ValueError(
        f"NCBI BLAST rejected the search (error code -99) {len(_REJECTION_RETRY_DELAYS_SECONDS) + 1} times"
    )


def _qblast_status(response: requests.Response) -> tuple[str, str]:
    """Read a QBLAST poll page as ``("XML", page)`` when finished or ``("WAITING", page)`` while pending.

    A ``-99`` rejection of a poll counts as pending, so the search is polled again after the usual
    interval. Any other page (``FAILED``, ``UNKNOWN``, ``READY`` without XML, or unrecognized HTML)
    raises, so a failed search never reads as one with no hits.
    """
    page = response.text
    if page.lstrip().startswith(("<?xml", "<BlastOutput")):
        return "XML", page
    status = _STATUS.search(page)
    if not page.strip() or _REJECTION.search(page) or (status is not None and status.group(1).upper() == "WAITING"):
        return "WAITING", page
    reported = f"status {status.group(1).upper()}" if status is not None else "a non-XML page"
    raise ValueError(f"NCBI BLAST returned {reported} for {response.url}; no hits can be inferred")
