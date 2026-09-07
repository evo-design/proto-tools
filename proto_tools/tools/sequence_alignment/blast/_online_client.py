"""Identified NCBI BLAST transport without process-global Biopython patches."""

import io
import re
import threading
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

_URL = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
_USER_AGENT = "Proto-Tools/BLAST (+https://github.com/evo-design/proto-tools)"
_REQUEST_LOCK = threading.Lock()
_NEXT_REQUEST = 0.0


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
        request = Request(_URL, urlencode(parameters).encode(), headers={"User-Agent": _USER_AGENT})  # noqa: S310 - fixed HTTPS endpoint
        try:
            with urlopen(request, timeout=_remaining(deadline)) as response:  # noqa: S310 - fixed HTTPS endpoint
                text = response.read().decode()
            _remaining(deadline)
            return text
        finally:
            _NEXT_REQUEST = time.monotonic() + 10.0
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

    deadline = None if timeout is None else time.monotonic() + timeout
    # Match the previous qblast defaults; never silently substitute databases.
    put = {
        "EXPECT": 10.0,
        "HITLIST_SIZE": 50,
        "ENTREZ_QUERY": "(none)",
        **parameters,
        "CMD": "Put",
        "tool": "proto-tools",
    }
    page = _request(put, deadline)
    # Reuse Biopython's established server-error/RID parser, not its hardcoded HTTP client.
    rid, _ = NCBIWWW._parse_qblast_ref_page(io.BytesIO(page.encode()))
    get = {"CMD": "Get", "RID": rid, "FORMAT_TYPE": "XML", "ALIGNMENTS": 500, "DESCRIPTIONS": 500}
    while True:
        _wait(60.0, deadline)
        xml = _request(get, deadline)
        status = re.search(r"Status=\s*([A-Za-z]+)", xml)
        if not xml.strip() or (status and status.group(1).upper() == "WAITING"):
            continue
        if status and status.group(1).upper() != "READY":
            raise ValueError(f"NCBI BLAST search {rid} returned status {status.group(1)}")
        if not xml.lstrip().startswith(("<?xml", "<BlastOutput")):
            raise ValueError(f"NCBI BLAST search {rid} returned a non-XML response; no hits can be inferred")
        return xml
