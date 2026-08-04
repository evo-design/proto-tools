"""proto_tools/tools/sequence_alignment/mmseqs2/msa_server.py.

The ColabFold MSA server exchange: submit sequences, poll a ticket, download alignments.

Which sequences to search and where the alignments land belongs to :mod:`remote_search`.

Based on ColabFold's MSA search implementation
(https://github.com/sokrypton/ColabFold @ 1f8fd1a, ``colabfold/colabfold.py``, ``run_mmseqs2``),
MIT licensed. Runs in the main process.
"""

import logging
import random
import tarfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import requests

from proto_tools.utils import base_config

logger = logging.getLogger(__name__)

MSA_SERVER_URL = "https://api.colabfold.com"


def _proto_tools_version() -> str:
    """Return the installed proto-tools version, or ``unknown`` where there is no distribution.

    A Modal container puts proto-tools on the path by mounting the source tree rather than
    installing it, so it carries no dist-info and ``version()`` raises. The version only decorates
    a ``User-Agent``, so a search that would otherwise succeed must not fail for the want of it.
    """
    try:
        return version("proto-tools")
    except PackageNotFoundError:
        return "unknown"


def user_agent_for(identity: str | None = None) -> str:
    """Build the ``User-Agent`` the server requires, naming proto-tools and the caller.

    Args:
        identity (str | None): Identifies the caller when the request originates elsewhere, such as
            a hosted process searching on a user's behalf. Derived from this machine when omitted.

    Returns:
        str: A user agent naming proto-tools and the caller.
    """
    return f"proto-tools/{_proto_tools_version()} ({identity or base_config.client_identity()})"


# Server-side statuses. Waiting helps only where the condition is expected to clear.
_TRANSIENT_STATUSES = frozenset({"UNKNOWN", "RATELIMIT", "PENDING", "RUNNING", "MAINTENANCE"})
_RATE_LIMITED_STATUSES = frozenset({"RATELIMIT", "MAINTENANCE"})

_REQUEST_TIMEOUT_SECONDS = 30.0
# Matches ColabFold's own client, which waits 5-10s between polls of a ticket.
_POLL_INTERVAL_SECONDS = 5.0
_RATE_LIMIT_INTERVAL_SECONDS = 10.0
# Spreads concurrent searches so a fanned-out batch does not resubmit in lockstep.
_MAX_JITTER_SECONDS = 5.0
# Network faults only; a status the server reported is handled by the poll loop.
_MAX_REQUEST_ATTEMPTS = 3


def _sleep_with_jitter(seconds: float) -> None:
    """Wait, spreading concurrent searches so they do not resubmit in lockstep."""
    time.sleep(seconds + random.uniform(0, _MAX_JITTER_SECONDS))  # noqa: S311 -- not for cryptographic use


def _request(method: str, url: str, user_agent: str, **kwargs: Any) -> requests.Response:
    """Make one HTTP request, retrying a network fault but not a reply.

    Args:
        method (str): HTTP method.
        url (str): Full URL.
        user_agent (str): Identifies the caller to the server.
        kwargs (Any): Passed to :func:`requests.request`.

    Returns:
        requests.Response: The server's reply.

    Raises:
        RuntimeError: If every attempt failed to reach the server.
    """
    headers = {"User-Agent": user_agent, **kwargs.pop("headers", {})}
    last: Exception | None = None
    for attempt in range(_MAX_REQUEST_ATTEMPTS):
        try:
            return requests.request(method, url, headers=headers, timeout=_REQUEST_TIMEOUT_SECONDS, **kwargs)
        except requests.RequestException as exc:  # noqa: PERF203 -- retry loop
            last = exc
            if attempt < _MAX_REQUEST_ATTEMPTS - 1:
                logger.warning("MSA server unreachable (%s), retrying", type(exc).__name__)
                _sleep_with_jitter(_POLL_INTERVAL_SECONDS * 2**attempt)
    raise RuntimeError(f"MSA server unreachable after {_MAX_REQUEST_ATTEMPTS} attempts: {last}") from last


def search_mode(use_env: bool, use_pairing: bool, pairing_strategy: str = "greedy") -> str:
    """Build the server's mode string for the requested search.

    The mode also names the directory the alignments are written to.

    Args:
        use_env (bool): Search the metagenomic database as well.
        use_pairing (bool): Pair alignments across a complex's chains.
        pairing_strategy (str): ``"greedy"`` pairs a species present in at least two chains,
            ``"complete"`` only one present in every chain. Ignored without pairing.

    Returns:
        str: Mode string, e.g. ``"env"``, ``"all"``, ``"pairgreedy-env"``, ``"paircomplete"``.
    """
    if use_pairing:
        mode = "paircomplete" if pairing_strategy == "complete" else "pairgreedy"
        return f"{mode}-env" if use_env else mode
    return "env" if use_env else "all"


def unique_sequences(sequences: list[str]) -> list[str]:
    """Deduplicate sequences, preserving first-occurrence order.

    The server answers once per distinct query, so identical chains are submitted once.

    Args:
        sequences (list[str]): Sequences as the caller supplied them.

    Returns:
        list[str]: The distinct sequences, first-occurrence order preserved.
    """
    return list(dict.fromkeys(sequences))


def _submit(sequences: list[str], mode: str, use_pairing: bool, user_agent: str) -> dict[str, Any]:
    """Hand the sequences to the server and take a ticket for them."""
    # The server expects a FASTA body whose identifiers are the arbitrary counter it echoes back.
    query = "".join(f">{index}\n{sequence}\n" for index, sequence in enumerate(unique_sequences(sequences), start=101))
    endpoint = "ticket/pair" if use_pairing else "ticket/msa"
    response = _request("POST", f"{MSA_SERVER_URL}/{endpoint}", user_agent, data={"q": query, "mode": mode})
    return _as_status(response)


def _poll(ticket_id: str, user_agent: str) -> dict[str, Any]:
    """Ask after a ticket already taken."""
    return _as_status(_request("GET", f"{MSA_SERVER_URL}/ticket/{ticket_id}", user_agent))


def _as_status(response: requests.Response) -> dict[str, Any]:
    """Read a reply as a status, treating an unreadable one as an error rather than raising."""
    try:
        return dict(response.json())
    except ValueError:
        return {"status": "ERROR", "detail": response.text[:200]}


def _download(ticket_id: str, destination: Path, user_agent: str) -> None:
    """Fetch a finished ticket's alignments and unpack them beside it."""
    response = _request("GET", f"{MSA_SERVER_URL}/result/download/{ticket_id}", user_agent, stream=True)
    archive = destination / "out.tar.gz"
    destination.mkdir(parents=True, exist_ok=True)
    with archive.open("wb") as handle:
        for block in response.iter_content(chunk_size=1 << 20):
            handle.write(block)
    with tarfile.open(archive) as tar:
        # filter="data" refuses members that would write outside the destination directory.
        tar.extractall(destination, filter="data")
    archive.unlink(missing_ok=True)


def run_remote_msa_search(
    sequences: list[str],
    prefix: str | Path,
    *,
    use_env: bool = False,
    use_pairing: bool = False,
    pairing_strategy: str = "greedy",
    client_identity: str | None = None,
    timeout: float | None = None,
) -> Path:
    """Search the ColabFold server for alignments and return the directory holding them.

    Waits while the server reports a transient condition (queued, running, rate limited, under
    maintenance) and fails immediately on any other.

    Args:
        sequences (list[str]): Query sequences.
        prefix (str | Path): Path prefix; results land in ``{prefix}_{mode}``.
        use_env (bool): Search the metagenomic database as well.
        use_pairing (bool): Pair alignments across a complex's chains.
        pairing_strategy (str): ``"greedy"`` or ``"complete"``; ignored without pairing.
        client_identity (str | None): Who asked, when the call came from elsewhere.
        timeout (float | None): Seconds to wait for this query; ``None`` waits indefinitely.

    Returns:
        Path: Directory holding the alignments.

    Raises:
        RuntimeError: If the server reported an error, or did not finish within ``timeout``.
    """
    mode = search_mode(use_env, use_pairing, pairing_strategy)
    destination = Path(f"{prefix}_{mode}")
    agent = user_agent_for(client_identity)

    status = _submit(sequences, mode, use_pairing, agent)
    started = time.monotonic()
    ticket_id: str | None = status.get("id")

    while (state := str(status.get("status", "ERROR"))) in _TRANSIENT_STATUSES:
        if timeout is not None and time.monotonic() - started > timeout:
            raise RuntimeError(f"MSA server did not finish within {timeout:.0f}s (last status {state})")
        _sleep_with_jitter(_RATE_LIMIT_INTERVAL_SECONDS if state in _RATE_LIMITED_STATUSES else _POLL_INTERVAL_SECONDS)
        # Rate limiting and maintenance refuse the work outright, so the search is resubmitted.
        if state in _RATE_LIMITED_STATUSES or ticket_id is None:
            status = _submit(sequences, mode, use_pairing, agent)
            ticket_id = status.get("id")
        else:
            status = _poll(ticket_id, agent)

    if state != "COMPLETE":
        detail = status.get("detail") or status.get("error") or ""
        raise RuntimeError(f"MSA server returned {state} for a {mode} search{f': {detail}' if detail else ''}")

    if ticket_id is None:
        raise RuntimeError(f"MSA server reported COMPLETE for a {mode} search without issuing a ticket")

    _download(ticket_id, destination, agent)
    logger.debug("Remote MSA search finished: %d sequence(s), mode %s", len(sequences), mode)
    return destination
