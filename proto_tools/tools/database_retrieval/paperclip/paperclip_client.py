"""Minimal HTTP client for the hosted Paperclip literature service.

Paperclip runs entirely on GXL's servers. The official ``gxl-paperclip`` package is a thin
client that formats a CLI-style command string and posts it to the service, so this module
speaks that same wire protocol directly with ``requests`` and needs no extra dependency:

- Commands go to the MCP endpoint as a JSON-RPC ``tools/call`` on the ``paperclip`` tool, with
  an ``X-API-Key`` or ``Authorization: Bearer`` header. The reply is a list of text blocks.
- A search reply names a saved result id; the structured hit list is then read from
  ``/api/cli/results/{result_id}``.
- Paper metadata and full text are files under ``/papers/{document_id}/`` in the service's
  virtual filesystem, read with ``cat`` and measured with ``wc``.

Only the commands the Paperclip tools use are wrapped here: ``search`` and ``lookup`` (which
save their hits on the server), ``grep`` over one paper or a whole corpus, and ``cat``, ``sed``,
and ``wc`` for reading paper files.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests

from proto_tools.utils import build_http_session, request_with_retry

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://paperclip.gxl.ai"
USER_AGENT = "proto-tools/paperclip-literature-v1"
_HTTP_RETRIES = 2
_BACKOFF_SECONDS = 1.0
#: Saved-result ids look like ``s_1a2b3c`` (search), ``m_...`` (map), ``r_...`` (results).
_RESULT_ID = re.compile(r"\b([smr]_[0-9a-f]{4,})\b")


class PaperclipError(RuntimeError):
    """Base class for Paperclip service failures."""


class PaperclipAuthError(PaperclipError):
    """No usable credentials, or the service rejected them."""


class PaperclipCommandError(PaperclipError):
    """The service ran the command but reported a failure."""


@dataclass
class PaperclipCommandResult:
    """Text output of one Paperclip command."""

    command: str
    output: str
    result_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class PaperclipClient:
    """Authenticated client for the Paperclip command endpoint.

    Credentials resolve in order: an explicit ``api_key``, an explicit ``bearer_token``, then
    the ``PAPERCLIP_API_KEY`` and ``PAPERCLIP_BEARER_TOKEN`` environment variables.
    ``PAPERCLIP_BASE_URL`` overrides the service origin when ``base_url`` is not given.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        bearer_token: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
        session: requests.Session | None = None,
    ) -> None:
        """Resolve credentials and the service origin, then prepare a retrying HTTP session."""
        api_key = (api_key or os.environ.get("PAPERCLIP_API_KEY") or "").strip()
        bearer_token = (bearer_token or os.environ.get("PAPERCLIP_BEARER_TOKEN") or "").strip()
        if api_key:
            self._auth_header = {"X-API-Key": api_key}
        elif bearer_token:
            self._auth_header = {"Authorization": f"Bearer {bearer_token}"}
        else:
            raise PaperclipAuthError(
                "Paperclip credentials are required: set PAPERCLIP_API_KEY (or PAPERCLIP_BEARER_TOKEN), "
                "or pass paperclip_api_key in the tool config."
            )
        self.base_url = (base_url or os.environ.get("PAPERCLIP_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._session = session or build_http_session(
            http_retries=_HTTP_RETRIES,
            backoff_seconds=_BACKOFF_SECONDS,
            user_agent=USER_AGENT,
        )
        self._call_id = 0

    @property
    def command_url(self) -> str:
        """JSON-RPC endpoint; the direct ``mcp.`` origin serves it at ``/papers``."""
        host = urlparse(self.base_url).hostname or ""
        return f"{self.base_url}/papers" if host.startswith("mcp.") else f"{self.base_url}/mcp"

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "User-Agent": USER_AGENT, **self._auth_header}

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

    def execute(self, command: str, args: Sequence[str] = ()) -> PaperclipCommandResult:
        """Run one Paperclip command and return its complete, un-rendered text output."""
        full_command = shlex.join([command, *args])
        self._call_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._call_id,
            "method": "tools/call",
            "params": {
                "name": "paperclip",
                "arguments": {
                    "command": full_command,
                    "description": full_command[:80],
                    # Return the whole output rather than the CLI's preview, without ANSI colour.
                    "skip_truncation": True,
                    "plain": True,
                    "no_color": True,
                },
            },
        }
        response = request_with_retry(
            lambda: self._session.post(
                self.command_url, json=payload, headers=self._headers(), timeout=self.timeout_seconds
            ),
            retries=_HTTP_RETRIES,
            backoff_seconds=_BACKOFF_SECONDS,
        )
        self._raise_for_status(response, f"command {full_command!r}")
        try:
            data = response.json()
        except ValueError as exc:
            raise PaperclipError(f"Paperclip returned non-JSON output for command {full_command!r}") from exc
        if not isinstance(data, dict):
            raise PaperclipError(f"Paperclip returned an unexpected reply for command {full_command!r}")
        error = data.get("error")
        if error is not None:
            message = error.get("message", "unknown error") if isinstance(error, dict) else str(error)
            raise PaperclipCommandError(f"Paperclip command {full_command!r} failed: {message}")
        result = data.get("result") or {}
        texts = [
            block.get("text", "")
            for block in result.get("content", []) or []
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        output = "\n".join(texts)
        if result.get("isError"):
            raise PaperclipCommandError(f"Paperclip command {full_command!r} failed: {output[:500]}")
        match = _RESULT_ID.search(output)
        return PaperclipCommandResult(
            command=full_command,
            output=output,
            result_id=match.group(1) if match else None,
            raw=data,
        )

    def fetch_result_papers(self, result_id: str) -> list[dict[str, Any]]:
        """Read the structured hit list saved under a search result id."""
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

    def _saved_result(self, command: str, args: Sequence[str]) -> tuple[str | None, list[dict[str, Any]]]:
        """Run a command that saves its hits on the server; return the result id and those hits."""
        result = self.execute(command, args)
        if result.result_id is None:
            if result.output.strip():
                logger.warning("Paperclip %s returned output without a saved result id: %.200s", command, result.output)
            return None, []
        return result.result_id, self.fetch_result_papers(result.result_id)

    def search(
        self,
        query: str,
        *,
        sources: Sequence[str],
        limit: int,
        ranking: str | None = None,
        exact: bool = False,
        since: str | None = None,
        year: int | None = None,
        journal: str | None = None,
        author: str | None = None,
        article_type: str | None = None,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Search the selected corpora; return the saved result id and its structured hits."""
        args: list[str] = []
        if exact:
            args.append("-e")
        args += [query, "-n", str(limit), "-s", ",".join(sources)]
        optional = {
            "--ranking": ranking,
            "--since": since,
            "--year": None if year is None else str(year),
            "--journal": journal,
            "--author": author,
            "--article-type": article_type,
        }
        for flag, value in optional.items():
            if value:
                args += [flag, value]
        return self._saved_result("search", args)

    def lookup(self, field: str, value: str, *, limit: int) -> tuple[str | None, list[dict[str, Any]]]:
        """Find papers by one metadata field; return the saved result id and its structured hits."""
        return self._saved_result("lookup", [field, value, "-n", str(limit)])

    def grep(
        self,
        pattern: str,
        path: str,
        *,
        ignore_case: bool = False,
        max_matches: int | None = None,
        context_lines: int = 0,
        exhaustive: bool = False,
    ) -> PaperclipCommandResult:
        """Run a regular-expression search over one file or a whole corpus directory."""
        args: list[str] = []
        if ignore_case:
            args.append("-i")
        if max_matches is not None:
            args += ["-m", str(max_matches)]
        if context_lines:
            args += ["-B", str(context_lines), "-A", str(context_lines)]
        if exhaustive:
            args.append("--exhaustive")
        return self.execute("grep", [*args, pattern, path])

    def read_file(self, path: str) -> str:
        """Return the complete text of a file in Paperclip's virtual filesystem."""
        return self.execute("cat", [path]).output

    def read_lines(self, path: str, start: int, end: int) -> str:
        """Return lines ``start`` through ``end`` (1-indexed, inclusive) of a file."""
        return self.execute("sed", ["-n", f"{start},{end}p", path]).output

    def byte_count(self, path: str) -> int | None:
        """Return the size in bytes that Paperclip reports for a file, or None if unparseable."""
        fields = self.execute("wc", [path]).output.split()
        if len(fields) < 3 or not fields[2].isdigit():
            return None
        return int(fields[2])
