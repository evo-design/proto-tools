"""Shared client and models for the Genomic Intelligence hosted ``/v1`` API.

Every tool in this toolkit is a thin wrapper over one hosted endpoint. There is
no local model, no weights, and no ``standalone/`` environment: the tools issue
HTTPS requests to https://api.genomicintelligence.ai and shape the response.

Delivery mode is a per-request choice, not a property of a task. Omitting the
``Prefer`` header returns the result synchronously with ``200``; sending
``Prefer: respond-async`` returns ``202`` plus a job id to poll from
``GET /v1/tasks/jobs/{job_id}``. Every endpoint accepts both.

Coordinates in tool outputs are 0-based with exclusive ends, following the
genomics interval convention used elsewhere in ``sequence_scoring`` rather than
the 1-based residue numbering used across the rest of proto-tools.

Authentication is a bearer key in ``GI_API_KEY``. Request one at
https://genomicintelligence.ai.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import requests
from pydantic import BaseModel, ConfigDict, Field

from proto_tools.tools.sequence_scoring.shared_data_models import validate_dna_sequence
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    ConfigField,
    InputField,
    build_http_session,
    get_logger,
    poll_until_complete,
)

logger = get_logger(__name__)

GI_BASE_URL = "https://api.genomicintelligence.ai"
"""Default hosted API root."""

_API_VERSION = "v1"
_REQUEST_TIMEOUT_SECONDS = 300
_HTTP_RETRIES = 2
_BACKOFF_SECONDS = 1.0
_USER_AGENT = "proto-tools/genomic-intelligence-v1"
_POLL_TIMEOUT_SECONDS = 15.0

_MAX_BP = 500_000
"""Upper bound shared by every endpoint, published as ``maxLength``."""

GITask = Literal["promoter", "splice", "enhancer", "chromatin", "annotation", "expression"]

_MISSING_KEY_MESSAGE = (
    "Genomic Intelligence: GI_API_KEY is unset. These tools call the hosted API at "
    "https://api.genomicintelligence.ai and require a bearer key. Request one at "
    "https://genomicintelligence.ai, then run 'export GI_API_KEY=gi_...' or pass "
    "gi_api_key in the tool config."
)


# ============================================================================
# Errors
# ============================================================================


class GIAPIError(RuntimeError):
    """A non-2xx response from the Genomic Intelligence API.

    The API returns a uniform ``{"error": {"code", "message", "request_id",
    "details"}}`` envelope on every failure. ``code`` is drawn from a closed
    enum published in the OpenAPI document, so branch on it rather than on the
    HTTP status; ``details`` varies by code and is carried verbatim for display.

    Attributes:
        status (int): HTTP status code.
        code (str): Machine-readable error code from the published enum.
        message (str): Human-readable explanation from the service.
        request_id (str | None): Correlation id for support. Read from the
            envelope, falling back to the ``X-Request-Id`` response header.
        details (Any): Code-specific payload, carried verbatim.
    """

    def __init__(self, status: int, code: str, message: str, request_id: str | None, details: Any) -> None:
        """Build the error from a parsed error envelope."""
        self.status = status
        self.code = code
        self.message = message
        self.request_id = request_id
        self.details = details
        super().__init__(f"[{status} {code}] {message} (request_id={request_id or 'unset'})")


class GIResponseShapeError(RuntimeError):
    """A 2xx body whose nested fields contradict their documented types.

    Distinct from :class:`GIAPIError`, which reports what the service itself
    said went wrong. This is a well-formed response carrying a field the
    contract documents as an object or an array that arrived as something else.

    The envelope itself is checked in the client, where the ``Response`` is
    still in hand, so a malformed envelope raises :class:`GIAPIError` with the
    real status and correlation id. The per-task fields nested inside ``data``
    are checked in the parse helpers, which take plain dicts and have no
    response to name, hence the separate type.
    """


# ============================================================================
# Response shape
# ============================================================================


def as_object(value: Any, field: str) -> dict[str, Any]:
    """Read a response field documented as an object.

    :func:`require_envelope` guarantees ``data`` is a non-empty object. It
    deliberately does not police the per-task fields nested inside it, because
    it is shared by six predict endpoints and one workflow and must not encode
    any single task's schema, so the checking happens here.

    Absent or null is legitimate -- a task with no ``prediction`` omits it --
    and becomes ``{}``. A field that is *present with the wrong type* is a
    malformed response and is reported as one.

    Two failure modes pull in opposite directions here. The ``x or {}`` idiom
    this replaces handled null and absent but not a truthy wrong type: a
    ``summary`` arriving as a string passed ``or {}`` untouched and then raised
    ``AttributeError`` on ``.get``, which reads as a client bug. Substituting
    ``{}`` for it instead trades that traceback for something worse, a result
    object of zeros returned to the caller as a successful prediction and
    indistinguishable from a real prediction of nothing. Raising a typed error
    is neither.

    Args:
        value (Any): Raw field value from the response.
        field (str): Dotted path to the field, used in the error message.

    Returns:
        dict[str, Any]: The object, or ``{}`` when absent or null.

    Raises:
        GIResponseShapeError: If the field is present with a non-object type.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise GIResponseShapeError(f"{field} should be an object, got {type(value).__name__}")
    return value


def as_object_list(value: Any, field: str) -> list[dict[str, Any]]:
    """Same as :func:`as_object`, for a field documented as an array of objects.

    A truthy non-list (a bare string) is iterable, so ``or []`` let it through
    and the row comprehension iterated its characters; non-object elements fail
    the same way. Neither is dropped silently, because a result missing rows it
    should have had is exactly the quiet wrong answer this is here to prevent.

    Args:
        value (Any): Raw field value from the response.
        field (str): Dotted path to the field, used in the error message.

    Returns:
        list[dict[str, Any]]: The array, or ``[]`` when absent or null.

    Raises:
        GIResponseShapeError: If the field is present with a non-array type, or
            if any element is not an object.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise GIResponseShapeError(f"{field} should be an array, got {type(value).__name__}")
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise GIResponseShapeError(f"{field}[{index}] should be an object, got {type(item).__name__}")
    return value


# ============================================================================
# Config
# ============================================================================


class GIConfig(BaseConfig):
    """Shared configuration for every Genomic Intelligence tool.

    Attributes:
        gi_api_key (str | None): Bearer key for the hosted API. Defaults to the
            ``GI_API_KEY`` environment variable; an explicit value passed to the
            config overrides the env var.
        base_url (str): API root. Override only to target a non-production
            deployment.
        model (str | None): Model identifier. Leave unset: the service resolves
            the current default for the task. Enumerate the choices with
            ``GET /v1/tasks/{task}/models`` rather than pinning one here.
        respond_async (bool): Request ``202`` + polling instead of a
            synchronous ``200``. A per-request delivery choice available on
            every endpoint; useful for long inputs.
        poll_interval_seconds (float): Delay between job polls when
            ``respond_async`` is set.
        timeout_seconds (float): Wall-clock cap on the async wait.
    """

    gi_api_key: str | None = ConfigField(
        title="GI API Key",
        default_factory=lambda: os.environ.get("GI_API_KEY"),
        description="Bearer key for api.genomicintelligence.ai. Defaults to the GI_API_KEY env var if not set.",
        include_in_key=False,
    )
    base_url: str = ConfigField(
        title="Base URL",
        default=GI_BASE_URL,
        description="API root; override only to target a non-production deployment",
        include_in_key=False,
    )
    model: str | None = ConfigField(
        title="Model",
        default=None,
        description="Model id; leave unset to use the task's server-side default (GET /v1/tasks/{task}/models)",
    )
    respond_async: bool = ConfigField(
        title="Respond Async",
        default=False,
        description="Send Prefer: respond-async and poll the job instead of waiting for a synchronous 200",
        include_in_key=False,
    )
    poll_interval_seconds: float = ConfigField(
        title="Poll Interval (seconds)",
        default=5.0,
        ge=1.0,
        description="Delay between job polls when respond_async is set",
        include_in_key=False,
    )
    timeout_seconds: float = ConfigField(
        title="Timeout (seconds)",
        default=1800.0,
        ge=10.0,
        description="Maximum wall-clock time to wait for an async job",
        include_in_key=False,
    )


# ============================================================================
# Shared output pieces
# ============================================================================


class GIRequestMeta(BaseModel):
    """Provenance for one call, read from the response ``meta`` envelope.

    Attributes:
        model (str): Model the service actually used, after default resolution.
        request_id (str | None): Correlates this call in support requests.
        job_id (str | None): Identifies the computation itself.
        inference_time_ms (float | None): Server-side inference time.
        cold_start (bool | None): Whether the model was loaded for this call.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model: str = Field(title="Model", description="Model the service resolved and ran")
    request_id: str | None = Field(title="Request ID", default=None, description="Correlation id for support")
    job_id: str | None = Field(title="Job ID", default=None, description="Identifier of the computation")
    inference_time_ms: float | None = Field(
        title="Inference Time ms", default=None, description="Server-side inference time in milliseconds"
    )
    cold_start: bool | None = Field(
        title="Cold Start", default=None, description="Whether the model was loaded for this call"
    )


def build_request_meta(payload: dict[str, Any], data: dict[str, Any]) -> GIRequestMeta:
    """Assemble :class:`GIRequestMeta` from a ``{data, meta}`` response.

    Args:
        payload (dict[str, Any]): Full response body.
        data (dict[str, Any]): The ``data`` member, whose ``model`` field is the
            authoritative record of what ran.

    Returns:
        GIRequestMeta: Provenance for the call.

    Raises:
        GIResponseShapeError: If ``meta`` is present with a non-object type.
    """
    meta = as_object(payload.get("meta"), "meta")
    return GIRequestMeta(
        model=str(data.get("model") or meta.get("model") or ""),
        request_id=meta.get("request_id"),
        job_id=meta.get("job_id"),
        inference_time_ms=meta.get("inference_time_ms"),
        cold_start=meta.get("cold_start"),
    )


# ============================================================================
# Input
# ============================================================================


class GISequence(BaseToolInput):
    """A DNA sequence with an optional label, submitted as one API call.

    The label and the sequence are bundled rather than kept in parallel lists
    because the framework slices only the iterable input field when partitioning
    across workers; a separate ``names`` list would desync from the sequences it
    describes.

    Attributes:
        sequence (str): DNA sequence. Length bounds are per task and are checked
            against the endpoint's published values before any request.
        name (str): Label echoed back in the response, useful for correlating
            results in a batch.
    """

    sequence: str = InputField(title="Sequence", description="DNA sequence to score")
    name: str = InputField(default="sequence", title="Name", description="Label echoed back in the response")


def coerce_gi_sequences(value: Any) -> Any:
    """Normalize a ``list[GISequence]`` field's raw input.

    Use as a ``@field_validator(..., mode="before")`` body. Accepts a bare DNA
    string or a single item, wrapping either into a one-element list, then
    coerces each element from ``str`` / ``dict`` / :class:`GISequence`.

    Args:
        value (Any): Raw field value supplied by the caller.

    Returns:
        Any: A list for Pydantic to validate into ``list[GISequence]``.

    Raises:
        ValueError: If an element is not a str, dict, or GISequence.
    """
    if isinstance(value, (str, dict, GISequence)):
        value = [value]
    if not isinstance(value, list):
        return value
    coerced: list[Any] = []
    for item in value:
        if isinstance(item, (GISequence, dict)):
            coerced.append(item)
        elif isinstance(item, str):
            coerced.append(GISequence(sequence=item))
        else:
            raise ValueError(f"each sequence must be a str, dict, or GISequence, got {type(item).__name__}")
    return coerced


# ============================================================================
# Input validation
# ============================================================================


def validate_gi_sequence(sequence: str, *, min_bp: int, task: str) -> str:
    """Validate a DNA sequence against a task's published length bounds.

    The bounds are per-task floors published as ``minLength`` on each endpoint's
    own request schema. They are task properties, not model properties, so
    selecting a different ``model`` never moves them. Checking here turns a
    round-trip ``422`` into an immediate, local error.

    Args:
        sequence (str): DNA sequence. Whitespace is stripped first, matching
            how the API measures length, so a line-wrapped body is accepted.
        min_bp (int): The task's published floor in base pairs.
        task (str): Task name, used in the error message.

    Returns:
        str: The uppercased, validated sequence.

    Raises:
        ValueError: If the sequence is empty, contains non-nucleotide
            characters, or falls outside the task's published bounds.
    """
    # Strip whitespace before validating. The API strips newlines, spaces and
    # tabs and measures length against the stripped string, so a line-wrapped
    # FASTA body is legal input there. The shared validate_dna_sequence upper-
    # cases but does not strip, so passing a wrapped body straight through
    # reports it as an invalid nucleotide character and refuses a sequence the
    # service would have scored.
    cleaned = validate_dna_sequence("".join(sequence.split()))
    length = len(cleaned)
    if length < min_bp:
        raise ValueError(
            f"{task}: sequence is {length:,} bp; the endpoint's published minimum is {min_bp:,} bp. "
            f"This is a task floor, so selecting a different model will not lower it."
        )
    if length > _MAX_BP:
        raise ValueError(f"{task}: sequence is {length:,} bp; the endpoint's published maximum is {_MAX_BP:,} bp.")
    return cleaned


# ============================================================================
# Client
# ============================================================================


def resolve_api_key(config: GIConfig) -> str:
    """Return the bearer key, failing fast with an actionable message.

    Args:
        config (GIConfig): Tool configuration; ``gi_api_key`` already defaults
            from the ``GI_API_KEY`` environment variable.

    Returns:
        str: The bearer key.

    Raises:
        OSError: If no key is configured.
    """
    if config.gi_api_key:
        return config.gi_api_key
    raise OSError(_MISSING_KEY_MESSAGE)


def _build_session(config: GIConfig) -> requests.Session:
    """Build a retrying session carrying the bearer key."""
    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
        allowed_methods=["GET", "POST"],
    )
    session.headers.update(
        {
            "Authorization": f"Bearer {resolve_api_key(config)}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    )
    return session


def raise_for_gi_error(response: requests.Response) -> None:
    """Raise :class:`GIAPIError` when a response carries the error envelope.

    Args:
        response (requests.Response): Response to inspect.

    Raises:
        GIAPIError: On any non-2xx response.
    """
    if response.ok:
        return
    header_request_id = response.headers.get("X-Request-Id")
    try:
        body = response.json()
    except ValueError:
        raise GIAPIError(
            response.status_code,
            "http_error",
            f"non-JSON response body: {response.text[:200]!r}",
            header_request_id,
            None,
        ) from None
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        raise GIAPIError(response.status_code, "http_error", str(body)[:200], header_request_id, None)
    raise GIAPIError(
        response.status_code,
        str(error.get("code") or "unknown"),
        str(error.get("message") or ""),
        error.get("request_id") or header_request_id,
        error.get("details"),
    )


def _parse_json_body(response: requests.Response) -> Any:
    """Parse a 2xx body, reporting a non-JSON one as a named API error.

    ``response.json()`` raises ``requests.exceptions.JSONDecodeError``, which is
    a ``RequestException``; on the polling path ``poll_until_complete`` treats
    that as transient and retries it to the wall-clock deadline, so a 200 that
    is not JSON becomes a half-hour wait ending in ``TimeoutError``. Raising
    :class:`GIAPIError` instead stops immediately and names the body.

    Args:
        response (requests.Response): A response already known to be 2xx.

    Returns:
        Any: The decoded body.

    Raises:
        GIAPIError: If the body is not JSON.
    """
    try:
        return response.json()
    except ValueError:
        raise GIAPIError(
            response.status_code,
            "http_error",
            f"non-JSON response body: {response.text[:200]!r}",
            response.headers.get("X-Request-Id"),
            None,
        ) from None


def require_envelope(response: requests.Response) -> dict[str, Any]:
    """Parse a 2xx body and require the documented ``{data, meta}`` envelope.

    Every predict endpoint, the workflow endpoint and the job-result endpoint
    return content inside ``data``: a prediction payload, a ``job_id``, or a
    finished job's result. ``data`` must therefore be a non-empty object for all
    of them, and ``{"data": {}}`` is malformed for every one. Accepting it
    produced a result object of zeros with real-looking provenance beside it,
    which the caller cannot tell from a prediction of nothing.

    Not every endpoint is enveloped: ``GET /v1/tasks/{task}/models`` and
    ``/health`` are deliberately bare, so this is applied at the three call
    sites that read out of ``data`` rather than in :func:`raise_for_gi_error`.
    A ``202`` poll of a *running* job is not one of them -- its ``data`` carries
    optional progress and is legitimately empty -- so that path parses without
    enveloping.

    Args:
        response (requests.Response): A response already known to be 2xx.

    Returns:
        dict[str, Any]: The full payload.

    Raises:
        GIAPIError: If the body is not JSON, is not an object, or carries a
            ``data`` member that is missing, not an object, or empty.
    """
    body = _parse_json_body(response)
    if isinstance(body, dict):
        data = body.get("data")
        if isinstance(data, dict) and data:
            # Returned from inside the isinstance check so the decoded body is
            # a dict here by narrowing, not by assertion.
            return body
        if data is None:
            detail = "'data' was missing or null"
        elif not isinstance(data, dict):
            detail = f"'data' was {type(data).__name__}"
        else:
            detail = "'data' was an empty object"
    else:
        detail = f"body was {type(body).__name__}"
    raise GIAPIError(
        response.status_code,
        "http_error",
        f"expected a JSON object with a non-empty object 'data' member: {detail}",
        response.headers.get("X-Request-Id"),
        None,
    )


def _job_status(response: requests.Response) -> tuple[str, Any]:
    """Map a job-poll response to a polling state.

    The job endpoint uses the HTTP status as the discriminator: ``202`` while
    the job is running, ``200`` once the result is ready. Errors surface as the
    standard envelope and are raised rather than polled through.

    Args:
        response (requests.Response): One poll response.

    Returns:
        tuple[str, Any]: ``(state, payload)`` for ``poll_until_complete``.

    Raises:
        GIAPIError: On a non-2xx response, a non-JSON body, or a completed job
            whose payload is not a ``{data, meta}`` envelope.
    """
    if response.status_code == 202:
        # A running job is not enveloped: its ``data`` is legitimately empty
        # until there is a result. ``progress`` is advisory, logged only, and
        # read defensively so a surprise there cannot fail the job.
        payload = _parse_json_body(response)
        data = payload.get("data") if isinstance(payload, dict) else None
        progress = data.get("progress") if isinstance(data, dict) else None
        if isinstance(progress, dict) and progress:
            logger.info("Genomic Intelligence job progress: %s", progress)
        return "PENDING", payload
    raise_for_gi_error(response)
    return "COMPLETE", require_envelope(response)


def _post(
    session: requests.Session,
    config: GIConfig,
    path: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    """POST a request body and return the completed ``{data, meta}`` payload.

    Raises:
        GIAPIError: On a non-2xx response, or on a 2xx that is not a
            ``{data, meta}`` envelope. Both the synchronous result and the
            ``202`` acceptance read content out of ``data``, so both are
            enveloped here; the polls in between are enveloped by
            :func:`_job_status` when the job completes.
    """
    headers = {"Prefer": "respond-async"} if config.respond_async else {}
    response = session.post(
        f"{config.base_url.rstrip('/')}/{_API_VERSION}/{path.lstrip('/')}",
        json=body,
        headers=headers,
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    raise_for_gi_error(response)
    payload: dict[str, Any] = require_envelope(response)
    if response.status_code != 202:
        return payload

    job_id = payload["data"].get("job_id")
    if not job_id:
        raise GIAPIError(response.status_code, "unknown", "202 response carried no job_id", None, payload)
    logger.info("Genomic Intelligence job %s accepted; polling", job_id)
    result: dict[str, Any] = poll_until_complete(
        session,
        f"{config.base_url.rstrip('/')}/{_API_VERSION}/tasks/jobs/{job_id}",
        poll_interval_seconds=config.poll_interval_seconds,
        timeout_seconds=config.timeout_seconds,
        success_states=frozenset({"COMPLETE"}),
        failure_states=frozenset({"FAILED"}),
        status_extractor=_job_status,
    )
    return result


def call_predict(
    config: GIConfig,
    task: GITask,
    sequence: str,
    sequence_name: str,
    options: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call the predict operation for one task.

    Each task is its own published operation with its own request schema and its
    own closed ``options`` object, so option keys are never interchangeable
    between tasks: an unrecognised key is a ``422``, not a silent no-op.

    Args:
        config (GIConfig): Shared configuration.
        task (GITask): Task name, which selects the endpoint.
        sequence (str): Validated DNA sequence.
        sequence_name (str): Caller-supplied label echoed back in the response.
        options (dict[str, Any] | None): Task-specific options, omitted when
            empty.
        extra_body (dict[str, Any] | None): Additional top-level request fields
            (expression's ``tss_index``).

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: ``(data, full_payload)``.
            ``data`` is a non-empty object, guaranteed by
            :func:`require_envelope`.

    Raises:
        GIAPIError: On any non-2xx response, and on a 2xx whose body is not a
            ``{data, meta}`` envelope.
    """
    body: dict[str, Any] = {"sequence": sequence, "sequence_name": sequence_name}
    if config.model is not None:
        body["model"] = config.model
    if options:
        body["options"] = options
    if extra_body:
        body.update(extra_body)

    session = _build_session(config)
    try:
        payload = _post(session, config, f"tasks/{task}/predict", body)
    finally:
        session.close()
    data: dict[str, Any] = payload["data"]
    return data, payload


def call_workflow(
    config: GIConfig,
    sequence: str,
    sequence_name: str,
    options: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call ``POST /v1/workflows/find-genes-and-predict-expression``.

    Unlike the predict endpoints this one has a hard delivery rule: a
    synchronous request above 50,000 bp is refused with ``413 sync_too_large``,
    so the caller must opt into async for long inputs. It is also JSON-only,
    declaring no ``format`` parameter.

    Args:
        config (GIConfig): Shared configuration.
        sequence (str): Validated DNA sequence.
        sequence_name (str): Caller-supplied label.
        options (dict[str, Any]): Workflow options; the endpoint requires the
            member to be present even when empty.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: ``(data, full_payload)``.
            ``data`` is a non-empty object, guaranteed by
            :func:`require_envelope`.

    Raises:
        GIAPIError: On any non-2xx response, and on a 2xx whose body is not a
            ``{data, meta}`` envelope.
    """
    body: dict[str, Any] = {"sequence": sequence, "sequence_name": sequence_name, "options": options}
    if config.model is not None:
        body["model"] = config.model

    session = _build_session(config)
    try:
        payload = _post(session, config, "workflows/find-genes-and-predict-expression", body)
    finally:
        session.close()
    data: dict[str, Any] = payload["data"]
    return data, payload
