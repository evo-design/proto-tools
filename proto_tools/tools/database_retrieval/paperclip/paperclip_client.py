"""Minimal HTTP client for the hosted Paperclip literature service.

Paperclip runs entirely on GXL's servers and exposes each operation as a tool on its MCP
endpoint, so this module calls those tools directly with ``requests`` and needs no extra
dependency:

- Each operation is a JSON-RPC ``tools/call`` to ``{base_url}/mcp`` naming one Paperclip tool
  (``search``, ``lookup``, ``grep``, ``cat``) with typed arguments, authenticated with an
  ``X-API-Key`` or ``Authorization: Bearer`` header. The reply is formatted text.
- A search, a multi-hit lookup, and a corpus grep name a saved result id in that text; the
  structured hits are read from ``/api/cli/results/{result_id}``.
- Paper metadata and line-numbered text are files under ``/papers/{document_id}/``. ``cat``
  returns ``L<n>: text`` lines and stops at 50,000 characters, so long ranges are read in pages.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import requests

from proto_tools.utils import build_http_session, request_with_retry

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://paperclip.gxl.ai"
USER_AGENT = "proto-tools/paperclip-literature-v1"
_HTTP_RETRIES = 2
_BACKOFF_SECONDS = 1.0
#: Saved-result ids look like ``s_1a2b3c`` (search, lookup, grep), ``m_...`` (map), ``r_...``.
_RESULT_ID = re.compile(r"\b([smr]_[0-9a-f]{4,})\b")
#: One line of a paper as ``cat`` and in-paper ``grep`` return it.
_NUMBERED_LINE = re.compile(r"^L(\d+):")
#: ``cat`` appends this when a reply reaches its character limit, possibly mid-line.
_TRUNCATED = "[output truncated at"
#: Upper bound for "to the end of the file"; a reply that is not cut off has reached the end.
LAST_LINE = 10_000_000
#: Concurrent requests for the per-paper reads of one lookup.
_LOOKUP_WORKERS = 8
#: A lookup hit's identifier line: ``PMC11754230 · Frontiers in Immunology · 2025``.
_LOOKUP_ID_LINE = re.compile(r"^\s+(\S+) · ", re.MULTILINE)


class PaperclipError(RuntimeError):
    """Base class for Paperclip service failures."""


class PaperclipAuthError(PaperclipError):
    """No usable credentials, or the service rejected them."""


class PaperclipCommandError(PaperclipError):
    """The service ran the tool but reported a failure."""


@dataclass
class PaperclipToolResult:
    """Text output of one Paperclip tool call."""

    tool: str
    output: str
    result_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class PaperclipClient:
    """Authenticated client for the Paperclip MCP endpoint.

    Authenticates with ``api_key`` when given, else ``token`` as a bearer token; the tool
    config resolves both from ``PAPERCLIP_API_KEY`` and ``PAPERCLIP_TOKEN``. The
    ``PAPERCLIP_BASE_URL`` environment variable overrides the service origin.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        token: str | None = None,
        timeout_seconds: float = 120.0,
        session: requests.Session | None = None,
    ) -> None:
        """Resolve credentials and the service origin, then prepare a retrying HTTP session."""
        if api_key:
            self._auth_header = {"X-API-Key": api_key}
        elif token:
            self._auth_header = {"Authorization": f"Bearer {token}"}
        else:
            raise PaperclipAuthError(
                "Paperclip credentials are required: set PAPERCLIP_API_KEY (or PAPERCLIP_TOKEN), "
                "or pass paperclip_api_key in the tool config."
            )
        self.base_url = os.environ.get("PAPERCLIP_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._session = session or build_http_session(
            http_retries=_HTTP_RETRIES,
            backoff_seconds=_BACKOFF_SECONDS,
            user_agent=USER_AGENT,
            allowed_methods=["GET", "POST"],
        )
        self._call_id = 0

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": USER_AGENT,
            **self._auth_header,
        }

    def _raise_for_status(self, response: requests.Response, what: str) -> None:
        status = response.status_code
        if 200 <= status < 300:
            return
        detail = None
        with contextlib.suppress(ValueError, AttributeError):
            detail = response.json().get("detail")
        if status == 401:
            raise PaperclipAuthError(
                f"Paperclip rejected the credentials for {what} (HTTP 401): {detail or ''}".rstrip()
            )
        if status == 403:
            raise PaperclipAuthError(f"Paperclip denied access for {what} (HTTP 403): {detail or ''}".rstrip())
        if status == 404:
            raise PaperclipCommandError(
                f"Paperclip has no such resource for {what} (HTTP 404): {detail or ''}".rstrip()
            )
        if status == 429:
            raise PaperclipError(f"Paperclip rate limit reached for {what} (HTTP 429): {detail or ''}".rstrip())
        raise PaperclipError(f"Paperclip returned HTTP {status} for {what}: {response.text[:500]}")

    def call(self, tool: str, arguments: dict[str, Any]) -> PaperclipToolResult:
        """Call one Paperclip tool; arguments that are None or False are left out."""
        sent = {key: value for key, value in arguments.items() if value is not None and value is not False}
        what = f"{tool} {json.dumps(sent)}"
        self._call_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._call_id,
            "method": "tools/call",
            "params": {"name": tool, "arguments": sent},
        }
        response = request_with_retry(
            lambda: self._session.post(
                f"{self.base_url}/mcp", json=payload, headers=self._headers(), timeout=self.timeout_seconds
            ),
            retries=_HTTP_RETRIES,
            backoff_seconds=_BACKOFF_SECONDS,
        )
        self._raise_for_status(response, what)
        try:
            data = response.json()
        except ValueError as exc:
            raise PaperclipError(f"Paperclip returned non-JSON output for {what}") from exc
        if not isinstance(data, dict):
            raise PaperclipError(f"Paperclip returned an unexpected reply for {what}")
        error = data.get("error")
        if error is not None:
            message = error.get("message", "unknown error") if isinstance(error, dict) else str(error)
            raise PaperclipCommandError(f"Paperclip {what} failed: {message}")
        result = data.get("result") or {}
        output = "\n".join(
            block.get("text", "")
            for block in result.get("content", []) or []
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if result.get("isError"):
            raise PaperclipCommandError(f"Paperclip {what} failed: {output[:500]}")
        match = _RESULT_ID.search(output)
        return PaperclipToolResult(tool=tool, output=output, result_id=match.group(1) if match else None, raw=data)

    def fetch_result_papers(self, result_id: str) -> list[dict[str, Any]]:
        """Read the structured hit list saved under a result id."""
        url = f"{self.base_url}/api/cli/results/{result_id}"
        response = request_with_retry(
            lambda: self._session.get(url, headers=self._headers(), timeout=self.timeout_seconds),
            retries=_HTTP_RETRIES,
            backoff_seconds=_BACKOFF_SECONDS,
        )
        self._raise_for_status(response, f"result {result_id}")
        try:
            data = response.json()
        except ValueError as exc:
            raise PaperclipError(f"Paperclip returned non-JSON data for result {result_id}") from exc
        result_data = data.get("result_data") if isinstance(data, dict) else None
        if not isinstance(result_data, dict):
            return []
        papers = result_data.get("papers")
        if papers is None:
            papers = result_data.get("results")
        return [paper for paper in papers if isinstance(paper, dict)] if isinstance(papers, list) else []

    def search(
        self,
        query: str,
        *,
        sources: Sequence[str],
        limit: int,
        exact: bool = False,
        since: str | None = None,
        year: int | None = None,
        journal: str | None = None,
        author: str | None = None,
        article_type: str | None = None,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Search the selected corpora; return the saved result id and its structured hits."""
        result = self.call(
            "search",
            {
                "query": query,
                "source": ",".join(sources),
                "limit": limit,
                "exact": exact,
                "since": since,
                "year": None if year is None else str(year),
                "journal": journal,
                "author": author,
                "article_type": article_type,
            },
        )
        if result.result_id is None:
            return None, []
        return result.result_id, self.fetch_result_papers(result.result_id)

    def lookup(self, field: str, value: str, *, limit: int) -> tuple[str | None, list[dict[str, Any]]]:
        """Find papers by one metadata field; return the saved result id, if any, and the hits.

        Paperclip saves a result only for some lookups. Without one, each listed paper's
        ``meta.json`` stands in for its saved hit.
        """
        result = self.call("lookup", {"field": field, "value": value, "limit": limit})
        if result.result_id is not None:
            return result.result_id, self.fetch_result_papers(result.result_id)
        paths = [
            f"/papers/{document_id}/meta.json" for document_id in dict.fromkeys(_LOOKUP_ID_LINE.findall(result.output))
        ]
        if not paths:
            return None, []
        with ThreadPoolExecutor(max_workers=min(_LOOKUP_WORKERS, len(paths))) as pool:
            return None, [json.loads(text) for text in pool.map(self.read_file, paths)]

    def grep(
        self,
        pattern: str,
        path: str,
        *,
        ignore_case: bool = False,
        max_matches: int | None = None,
        context_lines: int = 0,
        exhaustive: bool = False,
    ) -> PaperclipToolResult:
        """Run a regular-expression search over one file or a whole corpus directory."""
        return self.call(
            "grep",
            {
                "pattern": pattern,
                "path": path,
                "ignore_case": ignore_case,
                "max_count": max_matches,
                "context": context_lines or None,
                "exhaustive": exhaustive,
            },
        )

    def read_file(self, path: str) -> str:
        """Return the text of a small file, such as a paper's ``meta.json``."""
        return self.call("cat", {"path": path}).output

    def read_lines(self, path: str, start: int, end: int) -> list[str]:
        """Return lines ``start`` through ``end`` (1-indexed, inclusive) as ``L<n>: text`` strings.

        Stops early at the end of the file, so ``end=LAST_LINE`` reads the rest of it. A reply cut
        off at Paperclip's character limit loses its last, possibly partial, line, and reading
        resumes from that line; a reply that is not cut off holds everything left in the range.
        """
        lines: list[str] = []
        next_line = start
        while next_line <= end:
            output = self.call("cat", {"path": path, "lines": f"{next_line}-{end}"}).output
            truncated = _TRUNCATED in output
            numbered = [
                (int(match.group(1)), raw)
                for raw in output.splitlines()
                if (match := _NUMBERED_LINE.match(raw)) is not None
            ]
            if truncated:
                numbered = numbered[:-1]
            if not numbered:
                if truncated:
                    raise PaperclipError(f"{path} line {next_line} is longer than one Paperclip read returns")
                break
            lines.extend(raw for _, raw in numbered)
            next_line = numbered[-1][0] + 1
            if not truncated:
                break
        return lines
