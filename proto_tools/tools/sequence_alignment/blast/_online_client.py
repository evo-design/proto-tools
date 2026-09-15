"""Identified NCBI BLAST transport without process-global Biopython patches."""

import io
import logging
import math
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

_URL = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
_USER_AGENT = "Proto-Tools/BLAST (+https://github.com/evo-design/proto-tools)"
_REQUEST_LOCK = threading.Lock()
_NEXT_REQUEST = 0.0
logger = logging.getLogger(__name__)


class _SearchState(str, Enum):
    """Internal QBLAST submit/poll terminal-state model."""

    SUBMIT = "submit"
    POLL = "poll"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"


class BlastSubmissionRejected(ValueError):
    """NCBI -99 recovery exhausted; callers must not automatically restart its budget."""


def _is_rejected(page: str) -> bool:
    """Recognize the observed application rejection, not arbitrary server errors."""
    return bool(re.search(r"Cannot accept request,\s*error code:\s*-99(?![\w.])", page))


def _classify_poll(page: str) -> tuple[_SearchState, str | None]:
    """Classify one QBLAST poll response without treating ambiguity as no hits."""
    status_match = re.search(r"Status=\s*([A-Za-z]+)", page)
    status = status_match.group(1).upper() if status_match else None
    if not page.strip() or status == "WAITING":
        return _SearchState.POLL, status
    if status is not None and status != "READY":
        return _SearchState.FAILURE, status
    if page.lstrip().startswith(("<?xml", "<BlastOutput")):
        return _SearchState.SUCCESS, status
    return _SearchState.FAILURE, status


def _publish_slot(path: Path, spacing: float) -> None:
    """Atomically retain a pacing reservation even if a requesting process dies."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(str(time.time() + spacing))
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


@contextmanager
def _shared_slot(deadline: float | None):
    """Optionally coordinate participating processes using an explicit shared path."""
    from filelock import FileLock

    location = os.environ.get("PROTO_BLAST_PACING_PATH")
    if not location:
        yield None
        return
    path = Path(location)
    if not path.is_absolute():
        raise ValueError("PROTO_BLAST_PACING_PATH must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    remaining = _remaining(deadline)
    with FileLock(str(path) + ".lock", timeout=remaining if remaining is not None else -1):
        if path.exists():
            next_epoch = float(path.read_text())
            if not math.isfinite(next_epoch):
                raise ValueError("Invalid BLAST pacing timestamp")
            _wait(max(0.0, next_epoch - time.time()), deadline)
        _publish_slot(path, 10.0)
        yield path


def _remaining(deadline: float | None) -> float | None:
    """Return remaining search budget, raising when exhausted."""
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Online BLAST search deadline exhausted")
    return remaining


def _wait(seconds: float, deadline: float | None) -> None:
    """Respect pacing without sleeping past a finite search deadline."""
    remaining = _remaining(deadline)
    if remaining is not None and seconds >= remaining:
        raise TimeoutError("Online BLAST deadline insufficient for next permitted request")
    if seconds > 0:
        time.sleep(seconds)


def _request(parameters: dict[str, Any], deadline: float | None) -> str:
    """POST with a truthful client identity and process-local request pacing."""
    global _NEXT_REQUEST  # noqa: PLW0603 - process-local pacing state protected by the lock
    remaining = _remaining(deadline)
    if not _REQUEST_LOCK.acquire(timeout=remaining if remaining is not None else -1):
        raise TimeoutError("Online BLAST deadline exhausted waiting for request slot")
    try:
        _wait(max(0.0, _NEXT_REQUEST - time.monotonic()), deadline)
        with _shared_slot(deadline) as pacing_path:
            spacing = 10.0
            request = Request(_URL, urlencode(parameters).encode(), headers={"User-Agent": _USER_AGENT})  # noqa: S310 - fixed HTTPS endpoint
            try:
                with urlopen(request, timeout=_remaining(deadline)) as response:  # noqa: S310 - fixed HTTPS endpoint
                    text = response.read().decode()
                if _is_rejected(text):
                    spacing = 60.0
                _remaining(deadline)
                return text
            finally:
                _NEXT_REQUEST = time.monotonic() + spacing
                if pacing_path is not None:
                    _publish_slot(pacing_path, spacing)
    finally:
        _REQUEST_LOCK.release()


def search_xml(parameters: dict[str, Any], timeout: int | None) -> str:
    """Submit and retrieve XML, preserving QBLAST search and output parameters.

    Args:
        parameters (dict[str, Any]): Uppercase NCBI search parameters.
        timeout (int | None): Total search budget in seconds; None waits indefinitely.

    Returns:
        str: Raw XML for the existing Biopython result parser.
    """
    from Bio.Blast import NCBIWWW

    from proto_tools.utils.remote_errors import is_retryable_remote_error

    deadline = None if timeout is None else time.monotonic() + timeout
    rejections = 0

    def wait_with_recovery_budget(seconds: float) -> None:
        try:
            _wait(seconds, deadline)
        except TimeoutError as exc:
            if not rejections:
                raise
            raise BlastSubmissionRejected("NCBI BLAST error code: -99; recovery deadline exhausted") from exc

    def request_with_recovery(parameters: dict[str, Any]) -> str:
        nonlocal rejections
        while True:
            try:
                page = _request(parameters, deadline)
            except Exception as exc:
                if not rejections or not is_retryable_remote_error(exc):
                    raise
                raise BlastSubmissionRejected(
                    "NCBI BLAST error code: -99; recovery interrupted or deadline exhausted"
                ) from exc
            if not _is_rejected(page) or re.search(r"^\s*RID\s*=\s*\S+\s*$", page, re.MULTILINE):
                return page
            if rejections >= 2:
                raise BlastSubmissionRejected("NCBI BLAST error code: -99; bounded recovery exhausted")
            delay = 60.0 * 2**rejections
            rejections += 1
            logger.warning("NCBI BLAST -99; retry %s/2 in %.0fs (%s)", rejections, delay, parameters["CMD"])
            wait_with_recovery_budget(delay)

    # Match the previous qblast defaults; never silently substitute databases.
    put = {
        "EXPECT": 10.0,
        "HITLIST_SIZE": 50,
        "ENTREZ_QUERY": "(none)",
        **parameters,
        "CMD": "Put",
        "tool": "proto-tools",
    }
    state = _SearchState.SUBMIT
    rid: str | None = None
    xml: str | None = None
    try:
        while state not in {_SearchState.SUCCESS, _SearchState.FAILURE}:
            if state is _SearchState.SUBMIT:
                page = request_with_recovery(put)
                # Reuse Biopython's established server-error/RID parser, not its
                # hardcoded HTTP client.
                rid, _ = NCBIWWW._parse_qblast_ref_page(io.BytesIO(page.encode()))
                state = _SearchState.POLL
                continue

            if state is _SearchState.POLL:
                if rid is None:  # defensive state-integrity guard
                    state = _SearchState.FAILURE
                    raise RuntimeError("NCBI BLAST poll state has no accepted request ID")
                wait_with_recovery_budget(60.0)
                get = {
                    "CMD": "Get",
                    "RID": rid,
                    "FORMAT_TYPE": "XML",
                    "ALIGNMENTS": 500,
                    "DESCRIPTIONS": 500,
                }
                page = request_with_recovery(get)
                state, reported_status = _classify_poll(page)
                if state is _SearchState.SUCCESS:
                    xml = page
                elif state is _SearchState.FAILURE:
                    if reported_status is not None and reported_status != "READY":
                        raise ValueError(f"NCBI BLAST search {rid} returned status {reported_status}")
                    raise ValueError(f"NCBI BLAST search {rid} returned a non-XML response; no hits can be inferred")
    except TimeoutError:
        state = _SearchState.TIMEOUT
        raise

    if state is not _SearchState.SUCCESS or xml is None:
        raise RuntimeError(f"NCBI BLAST search ended in unexpected state {state.value}")
    return xml
