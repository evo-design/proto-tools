"""HTTP client for Proto's hosted tool API.

Covers job submission, polling, log streaming, and output asset decoding.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import random
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

TOOLS_BASE_URL = "https://proto-tools.evodesign.org"

# A key sent over plaintext is a leaked key, so anything but loopback must be https.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

_CONNECT_TIMEOUT = 10.0
_READ_TIMEOUT = 60.0


class ProtoError(Exception):
    """Base for every error raised by this client."""


class ProtoAPIError(ProtoError):
    """The server returned a non-success status.

    Attributes:
        status_code (int): HTTP status returned.
        detail (Any): Parsed ``detail`` from the body, or the raw text.
    """

    def __init__(self, status_code: int, detail: Any) -> None:
        """Record the status and detail, and render them as the message."""
        super().__init__(f"[{status_code}] {_render_detail(detail)}")
        self.status_code = status_code
        self.detail = detail


class ProtoAuthError(ProtoAPIError):
    """401/403 — the API key is missing, invalid, or lacks access."""


def _render_detail(detail: Any) -> str:
    """Flatten a FastAPI ``detail`` into one line."""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        return str(detail.get("message") or detail.get("detail") or detail)
    if isinstance(detail, list) and detail:
        first = detail[0]
        if isinstance(first, dict):
            loc = ".".join(str(p) for p in first.get("loc", []) if p not in ("body", "config"))
            return f"{loc}: {first.get('msg', first)}" if loc else str(first.get("msg", first))
    return str(detail)


def _raise_for_status(response: requests.Response) -> None:
    """Translate an error response into the matching exception."""
    if response.ok:
        return
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    if response.status_code in (401, 403):
        raise ProtoAuthError(response.status_code, detail)
    raise ProtoAPIError(response.status_code, detail)


def _resolve_base_url(explicit: str | None) -> str:
    """Pick the tools base URL, refusing plaintext to anywhere but loopback."""
    url = (explicit or os.environ.get("PROTO_TOOLS_BASE_URL") or TOOLS_BASE_URL).rstrip("/")
    if url == TOOLS_BASE_URL:
        return url
    parsed = urlparse(url)
    if parsed.scheme != "https" and parsed.hostname not in _LOOPBACK_HOSTS:
        raise ValueError(f"Refusing to send an API key over plaintext to {url!r}; use https.")
    return url


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _origin_of(url: str) -> str:
    """Return ``scheme://host[:port]``, dropping the default port."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port
    netloc = host if port is None or port == _DEFAULT_PORTS.get(parsed.scheme) else f"{host}:{port}"
    return f"{parsed.scheme}://{netloc}"


def _decode_asset_bytes(ref: dict[str, Any], data: bytes) -> Any:
    """Decode asset bytes by MIME type: (gzipped) JSON to an object, text to str, else bytes."""
    mime_type = ref.get("mime_type") or ""
    if mime_type == "application/json+gzip":
        return json.loads(gzip.decompress(data).decode("utf-8"))
    if mime_type == "application/json" or mime_type.endswith("+json"):
        return json.loads(data.decode("utf-8"))
    if mime_type.startswith(("chemical/", "text/")):
        return data.decode("utf-8")
    return data


# Statuses worth another attempt: rate limiting and the transient 5xx family. A 4xx other
# than 429 reflects the request itself and will fail identically however often it is sent.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_RETRYABLE_EXCEPTIONS = (requests.ConnectionError, requests.Timeout)

_MAX_RETRIES = 2  # 3 attempts total
_BACKOFF_BASE = 0.5
_BACKOFF_MAX = 8.0
_RETRY_AFTER_MAX = 300.0
_JITTER = 0.1


def _retry_after_seconds(response: requests.Response) -> float | None:
    """Seconds the server asked us to wait, capped; ``None`` when it did not ask."""
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return min(float(raw), _RETRY_AFTER_MAX)
    except ValueError:
        return None  # HTTP-date form; fall back to our own backoff


def _backoff_delay(attempt: int, response: requests.Response | None) -> float:
    """Delay before the next attempt: the server's Retry-After if given, else jittered exponential."""
    if response is not None:
        asked = _retry_after_seconds(response)
        if asked is not None:
            return asked
    base: float = min(_BACKOFF_BASE * (2**attempt), _BACKOFF_MAX)
    # Jitter only spreads retries across callers; it is not a security primitive.
    return base * (1 + random.uniform(-_JITTER, _JITTER))  # noqa: S311


def _request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    **kwargs: Any,
) -> requests.Response:
    """Send a request, retrying transient failures with backoff.

    Only for requests that are safe to repeat — every GET, and a submit carrying an
    idempotency key. Without this a single blip mid-poll would surface as a
    ``ConnectionError``, and the tool wrapper's own retry would resubmit the whole job
    rather than re-poll the one already running.

    Not the same function as ``proto_tools.utils.http_session.request_with_retry``: this one also
    retries retryable HTTP status codes and reads ``Retry-After`` off the response, which needs
    the request-dispatch shape below rather than a generic callable.

    Args:
        session (requests.Session): Authenticated session to send on.
        method (str): HTTP method.
        url (str): Absolute URL.
        kwargs (Any): Passed through to ``requests``.

    Returns:
        requests.Response: The final response, successful or not; callers still
            translate its status.
    """
    last_exc: BaseException | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = session.request(method, url, **kwargs)
        except _RETRYABLE_EXCEPTIONS as exc:
            last_exc = exc
            if attempt == _MAX_RETRIES:
                raise
            time.sleep(_backoff_delay(attempt, None))
            continue
        if response.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES:
            delay = _backoff_delay(attempt, response)
            logger.debug("Retrying %s %s after HTTP %s in %.1fs", method, url, response.status_code, delay)
            response.close()
            time.sleep(delay)
            continue
        return response
    raise last_exc if last_exc is not None else RuntimeError("unreachable")


@dataclass
class JobStatus:
    """One poll of a submitted job."""

    status: str
    result: Any = None
    error: str | None = None


@dataclass
class LogRecord:
    """One NDJSON log line streamed from a running job."""

    type: str = "record"
    msg: str | None = None
    level: str = "info"
    stream: str = "stdout"
    update_status: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class _ToolsNamespace:
    """The ``/tools`` endpoints ``device="proto"`` dispatch needs."""

    def __init__(self, session: requests.Session, base_url: str) -> None:
        """Bind to an authenticated session and base URL."""
        self._session = session
        self._base = base_url

    def catalogue(self) -> list[dict[str, Any]]:
        """List every tool the service knows, hosted or not.

        The catalogue is public and this request is unauthenticated, which allows a
        caller to browse before configuring a key. Entries carry ``hosted`` and, when
        that is false, an ``unhosted_reason`` written to be read by a person.

        Returns:
            list[dict[str, Any]]: One entry per registered tool.
        """
        response = _request_with_retry(
            self._session, "GET", f"{self._base}/api/v1/tools", timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT)
        )
        _raise_for_status(response)
        return [dict(entry) for entry in response.json()]

    def submit(
        self,
        tool_key: str,
        inputs: dict[str, Any],
        config: dict[str, Any] | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> str:
        """Submit a job and return its id.

        Args:
            tool_key (str): Registry key of the tool to run.
            inputs (dict[str, Any]): Serialized tool input.
            config (dict[str, Any] | None): Serialized tool config.
            idempotency_key (str | None): Reuse to make a repeat submit return the same
                job instead of creating another; one is generated when omitted.

        Returns:
            str: The job id to poll.
        """
        response = _request_with_retry(
            self._session,
            "POST",
            f"{self._base}/api/v1/tools/{tool_key}/run",
            json={"inputs": inputs, "config": config or {}},
            headers={"Idempotency-Key": idempotency_key or str(uuid.uuid4())},
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
        )
        _raise_for_status(response)
        return str(response.json()["job_id"])

    def get(self, tool_key: str, job_id: str) -> JobStatus:
        """Fetch a job's current status.

        Args:
            tool_key (str): Registry key the job was submitted under.
            job_id (str): Job id returned by :meth:`submit`.

        Returns:
            JobStatus: Status, plus result or error once terminal.
        """
        response = _request_with_retry(
            self._session,
            "GET",
            f"{self._base}/api/v1/tools/{tool_key}/jobs/{job_id}",
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
        )
        _raise_for_status(response)
        body = response.json()
        return JobStatus(status=str(body.get("status")), result=body.get("result"), error=body.get("error"))

    def iter_job_logs(
        self,
        tool_key: str,
        job_id: str,
        *,
        follow: bool = True,
        level: str | None = None,
        stream: str | None = None,
    ) -> Iterator[LogRecord]:
        """Yield NDJSON log records as the job produces them.

        Args:
            tool_key (str): Registry key the job was submitted under.
            job_id (str): Job id to stream.
            follow (bool): Keep the connection open until the job ends.
            level (str | None): Minimum RFC 5424 severity to receive.
            stream (str | None): Restrict to one channel (``stdout``/``stderr``/``system``).

        Yields:
            LogRecord: One per line; a ``type="end"`` record terminates the stream.
        """
        params: dict[str, Any] = {"follow": str(follow).lower()}
        if level:
            params["level"] = level
        if stream:
            params["stream"] = stream
        with self._session.get(
            f"{self._base}/api/v1/tools/{tool_key}/jobs/{job_id}/logs",
            params=params,
            stream=True,
            timeout=(_CONNECT_TIMEOUT, None),
        ) as response:
            _raise_for_status(response)
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except ValueError:
                    continue
                yield LogRecord(
                    type=payload.get("type", "record"),
                    msg=payload.get("msg"),
                    level=payload.get("level", "info"),
                    stream=payload.get("stream", "stdout"),
                    update_status=bool(payload.get("update_status", False)),
                    extra=payload,
                )


class _AssetsNamespace:
    """Fetches the content behind an output asset ref."""

    def __init__(self, session: requests.Session, base_url: str) -> None:
        """Bind to an authenticated session and the origin its credentials belong to."""
        self._session = session
        self._origin = _origin_of(base_url)

    def decode(self, ref: dict[str, Any]) -> Any:
        """Fetch an asset and decode it by declared MIME type.

        The ref carries its own fetch URL. That URL redirects to object storage,
        and the redirect is followed by hand so the API key is never replayed to
        a third-party host.

        Args:
            ref (dict[str, Any]): An asset ref carrying ``url`` and ``mime_type``.

        Returns:
            Any: Parsed object for JSON, ``str`` for text and chemical types, else raw bytes.

        Raises:
            ValueError: If the ref carries no fetch URL.
            RuntimeError: If a redirect omits its ``Location``.
        """
        url = ref.get("url")
        if not url:
            raise ValueError(f"Asset ref {ref.get('id')!r} has no fetch URL and cannot be fetched.")
        return _decode_asset_bytes(ref, self._fetch(url))

    def _fetch(self, url: str) -> bytes:
        """Read the asset bytes, following one storage redirect without the API key."""
        response = _request_with_retry(
            self._session, "GET", url, allow_redirects=False, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT)
        )
        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("location")
            if not location:
                raise RuntimeError(f"Asset GET {url} redirect did not include a Location header")
            # Off our origin the signed URL is the credential; sending ours too would leak it.
            follow = self._session if _origin_of(location) == self._origin else requests
            redirected = follow.get(location, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT))
            redirected.raise_for_status()
            return bytes(redirected.content)
        _raise_for_status(response)
        return bytes(response.content)


class ProtoClient:
    """Authenticated client for Proto's hosted tool API.

    Attributes:
        tools (_ToolsNamespace): Submit, poll, and stream logs for tool jobs.
        assets (_AssetsNamespace): Resolve output asset refs.
    """

    def __init__(self, api_key: str | None = None, *, base_url: str | None = None) -> None:
        """Build a client, taking the key from ``PROTO_API_KEY`` when not passed.

        Args:
            api_key (str | None): API key; falls back to ``PROTO_API_KEY``.
            base_url (str | None): Override the tools API base URL.

        Raises:
            ValueError: If no key is available.
        """
        resolved = api_key if api_key is not None else os.environ.get("PROTO_API_KEY")
        if not resolved:
            raise ValueError("api_key must not be empty. Pass a valid key or set PROTO_API_KEY.")
        self._base = _resolve_base_url(base_url)
        session = requests.Session()
        session.headers.update({"X-API-Key": resolved, "Accept": "application/json"})
        self._session = session
        self.tools = _ToolsNamespace(session, self._base)
        self.assets = _AssetsNamespace(session, self._base)
