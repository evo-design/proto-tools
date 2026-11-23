"""Shared constants and base models for the PARADE UTR activity/stability tools."""

import ipaddress
import json
import re
import urllib.parse
from typing import Any, ClassVar, Literal, cast

from pydantic import ValidationError, field_validator
from pydantic_core import InitErrorDetails, core_schema

from proto_tools.utils import BaseConfig, BaseToolInput, ConfigField, InputField
from proto_tools.utils.device import RemoteDevice
from proto_tools.utils.sequence import return_invalid_nucleotide_chars
from proto_tools.utils.tool_io import Metrics, MetricSpec

# PARADE's queryable cell-line panel per UTR construct type -- the lines the upstream tutorial
# predicts on. Each code identifies a human cell line (see ``PARADE_CELL_LINE_NAMES``); the codes
# remain the canonical identifiers the model is conditioned on, while callers may select a line by
# either its code or its name. These are a subset of the model's trained condition channels: the
# vendored ``CELLTYPE_CODES_UTR3`` map also has ``c10`` (an extra, unnamed channel) which the
# checkpoint conditions on but the upstream pipeline never queries, so it is intentionally omitted
# from the queryable panel here. The model still conditions on all its trained channels via
# broadcast one-hot inputs.
ParadeConstructType = Literal["utr5", "utr3"]
ParadeCellType = Literal["c1", "c2", "c4", "c6", "c17", "c13"]
PARADE_CELL_TYPES: dict[str, tuple[ParadeCellType, ...]] = {
    "utr5": ("c1", "c2", "c4", "c6", "c17"),
    "utr3": ("c1", "c2", "c4", "c6", "c17", "c13"),
}

# Human-readable identities for the queryable panel codes. The upstream PARADE release and paper
# refer to these conditions only by anonymized code; naming them lets callers select a cell line by
# name (e.g. ``"HepG2"``) instead of the opaque code. Codes stay canonical -- a name is resolved to
# its code at the input boundary and the model is always addressed by code. ``c13`` is 3'UTR-only.
PARADE_CELL_LINE_NAMES: dict[ParadeCellType, str] = {
    "c1": "MDA-MB-231",
    "c2": "HepG2",
    "c4": "Jurkat",
    "c6": "SW480",
    "c17": "NALM6",
    "c13": "PA-1",
}

# One rendering of the panel, shared by the cell-selector descriptions and the resolver's error.
PARADE_CELL_PANEL = ", ".join(f"{code} {name}" for code, name in PARADE_CELL_LINE_NAMES.items())

# Case-insensitive lookup that accepts either a code (``"c2"``) or a cell-line name (``"HepG2"``).
_CELL_SELECTOR_TO_CODE: dict[str, ParadeCellType] = {
    **{code.lower(): code for code in PARADE_CELL_LINE_NAMES},
    **{name.lower(): code for code, name in PARADE_CELL_LINE_NAMES.items()},
}


def resolve_cell_type(value: str) -> ParadeCellType:
    """Resolve a PARADE cell selector to its canonical code.

    Accepts either a cell code (``"c2"``) or a cell-line name (``"HepG2"``), matched
    case-insensitively after stripping surrounding whitespace. A code is returned unchanged.
    Panel membership (which codes are valid for a given construct type) is enforced separately
    by the tool configs -- this only maps a name or code to a code.

    Args:
        value (str): A PARADE cell code or cell-line name.

    Returns:
        ParadeCellType: The canonical cell code.

    Raises:
        ValueError: If ``value`` is not a known code or cell-line name.
    """
    if not isinstance(value, str):
        raise ValueError(f"PARADE cell type must be a string, got {type(value).__name__}")
    code = _CELL_SELECTOR_TO_CODE.get(value.strip().lower())
    if code is None:
        raise ValueError(
            f"Unknown PARADE cell type {value!r}; expected a cell code or cell-line name: {PARADE_CELL_PANEL}"
        )
    return code


_VALID_INPUT_CHARS = "N"  # U is normalized to T; A/C/G/T come from DNA_NUCLEOTIDES.


def normalize_utr_sequence(sequence: str) -> str:
    """Uppercase, strip whitespace, map RNA ``U`` to ``T``, and validate A/C/G/T/N.

    PARADE encodes UTRs in the DNA alphabet (``T``, not ``U``). Accepting ``U`` and
    normalizing it lets callers pass RNA. ``N`` is accepted but, matching the upstream
    featurizer (which assigns 0.25 into an integer one-hot tensor and so truncates to 0),
    encodes as an all-zero column ``[0, 0, 0, 0]`` rather than a uniform 0.25 base.

    Args:
        sequence (str): A single UTR sequence.

    Returns:
        str: The normalized DNA-alphabet sequence.

    Raises:
        ValueError: If the sequence is empty or contains characters other than
            A, C, G, T, U, or N.
    """
    if not sequence or not sequence.strip():
        raise ValueError("Sequence cannot be empty")
    # ``split()`` drops every run of whitespace (spaces, tabs, newlines, CR); then map RNA U to DNA T.
    seq = "".join(sequence.upper().split()).replace("U", "T")
    invalid_chars = return_invalid_nucleotide_chars(seq, additional_valid_chars=_VALID_INPUT_CHARS)
    if invalid_chars:
        raise ValueError(f"Invalid nucleotide characters in sequence: {', '.join(sorted(invalid_chars))}")
    return seq


def require_https_checkpoint_url(value: str) -> str:
    """Validate a custom checkpoint URL, requiring HTTPS, a real host, and no embedded credentials.

    A PARADE checkpoint is unpickled after download, so a custom URL must be well-formed and
    fetched over a non-downgradeable transport. Malformed values (``https:relative``,
    ``https:///no-host``, ``https://``) are rejected here rather than after worker startup, and
    ``user:pass@`` userinfo is refused (URL-embedded credentials are unsupported — auth to
    Hugging Face flows through ``HF_TOKEN``). Empty passes through (the pinned URL is used).

    Error messages never echo the raw value, which may itself contain the credential.
    """
    # Leave an empty value alone so the pinned checkpoint URL is used
    if not value:
        return value
    # Reject whitespace, control characters and raw non-ASCII before parsing
    if any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("model_checkpoint link must not contain whitespace or control characters")
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError("model_checkpoint link must use ASCII with non-ASCII characters percent-encoded") from None
    # Parse the URL; reading the port raises on a non-numeric or out-of-range value
    try:
        parsed = urllib.parse.urlparse(value)
        parsed.port  # noqa: B018 - property access validates the port (ValueError if non-numeric/out of range)
    except ValueError:
        raise ValueError("model_checkpoint link is not a valid URL") from None
    # Require https, a host and no embedded user:pass credentials
    if parsed.scheme != "https":
        raise ValueError("model_checkpoint link must be an https:// URL")
    if not parsed.hostname:
        raise ValueError("model_checkpoint link must include a host (e.g. https://host/path)")
    if parsed.username or parsed.password:
        raise ValueError("model_checkpoint link must not embed credentials (user:pass@); use HF_TOKEN instead")
    host = parsed.hostname.rstrip(".")
    # Accept an IP literal; otherwise check every DNS label is well-formed
    try:
        ipaddress.ip_address(host)
    except ValueError:
        labels = host.split(".")
        if (
            len(host) > 253
            or any(not label or len(label) > 63 for label in labels)
            or any(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label) is None for label in labels)
        ):
            raise ValueError("model_checkpoint link host must be a valid IPv4, IPv6, or DNS hostname") from None
    return value


def _looks_like_checkpoint_url(value: str) -> bool:
    """True if ``value`` is a download link — ``http(s)://…`` or a schemeless ``host.tld/path``.

    A bare filename or local path (``model.ckpt``, ``sub/x.ckpt``, ``/oak/x.ckpt``, ``./x``) is not a
    link. A schemeless ``huggingface.co/user/model.ckpt`` is (and is normalized to ``https://`` on use).
    """
    if not value:
        return False
    scheme = urllib.parse.urlparse(value).scheme
    if scheme:
        return scheme in ("http", "https")
    head, _, rest = value.partition("/")
    if not rest or "." not in head:
        return False
    return all(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label) for label in head.split("."))


# Bundled model names: ``setup.sh`` fetches their checkpoint files, and the worker picks the one each
# operation scores. Any other ``model_checkpoint`` is your own ``.ckpt`` path or https link.
PARADE_BUNDLED_MODELS: frozenset[str] = frozenset({"parade"})


def local_checkpoint_on_remote(model_checkpoint: str, device: str) -> str | None:
    """Reject a local checkpoint path on a remote worker, where it doesn't exist.

    Args:
        model_checkpoint (str): A bundled model name, an https link, or a local path.
        device (str): The remote device the config is dispatched to.

    Returns:
        str | None: Why the config can't run on ``device``, or ``None`` if it can.
    """
    if model_checkpoint in PARADE_BUNDLED_MODELS or model_checkpoint.startswith("https://"):
        return None
    return (
        f"model_checkpoint points to a local file not available on device='{device}'. Use the bundled "
        "model 'parade' or an https link, or run locally with device='cuda' or device='cpu'."
    )


def normalize_checkpoint_override(value: str) -> str:
    """Validate a ``model_checkpoint``: a bundled name, an https link (schemeless allowed), or a local path.

    A link (``http(s)://…`` or ``host.tld/path``) is validated (HTTPS-only, real host, no embedded
    credentials) and returned with a lowercase ``https://`` prefix, which is how the worker tells a
    link from a path; a bundled name or local path passes through unchanged. A checkpoint is unpickled after
    download, so a link must use a non-downgradeable HTTPS transport.
    """
    if not value:
        raise ValueError("model_checkpoint must not be empty; use 'parade' or your own .ckpt path or https link")
    if not _looks_like_checkpoint_url(value):
        return value
    scheme = urllib.parse.urlparse(value).scheme  # urlparse lowercases the scheme
    normalized = f"{scheme}{value[len(scheme) :]}" if scheme else f"https://{value}"
    return require_https_checkpoint_url(normalized)


def _redact_checkpoint_url_input(value: Any, *, redact_all_strings: bool = False) -> Any:
    """Recursively replace checkpoint URL values inside a rejected model input."""
    if redact_all_strings and isinstance(value, (str, bytes, bytearray)):
        return "<redacted-checkpoint>"
    if isinstance(value, dict):
        return {
            ("<redacted-key>" if redact_all_strings and isinstance(key, str) else key): (
                "<redacted-checkpoint>"
                if key == "model_checkpoint"
                else _redact_checkpoint_url_input(item, redact_all_strings=redact_all_strings)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_checkpoint_url_input(item, redact_all_strings=redact_all_strings) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_checkpoint_url_input(item, redact_all_strings=redact_all_strings) for item in value)
    if isinstance(value, set):
        return {_redact_checkpoint_url_input(item, redact_all_strings=redact_all_strings) for item in value}
    return value


class _SafeCheckpointUrlConfigMixin:
    """Redact rejected checkpoint URLs from model-validation error representations."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> core_schema.CoreSchema:
        """Wrap model validation and remove raw checkpoint URLs from structured errors."""
        schema = handler(source)

        def validate_with_redaction(value: Any, validator: Any) -> Any:
            try:
                return validator(value)
            except ValidationError as error:
                # NOTE: pinned to pydantic 2.x error shape — errors(include_url=False) yields dicts
                # that from_exception_data accepts as InitErrorDetails. Re-test on a pydantic major bump.
                line_errors = cast(list[InitErrorDetails], error.errors(include_url=False))
                for line_error in line_errors:
                    error_input = line_error.get("input")
                    location = line_error.get("loc", ())
                    if "model_checkpoint" in location:
                        line_error["input"] = "<redacted-checkpoint>"
                    elif not location:
                        # Top-level error: input is the whole config (dict) or a bare scalar. A bare
                        # string might be a credentialed URL, so redact it — under a NEUTRAL label,
                        # since at this depth we can't tell it's a checkpoint_url (safety over clarity).
                        if isinstance(error_input, (str, bytes, bytearray)):
                            line_error["input"] = "<redacted>"
                        else:
                            line_error["input"] = _redact_checkpoint_url_input(error_input, redact_all_strings=True)
                    else:
                        # Per-field error elsewhere (e.g. cell_types): keep its input visible, only
                        # scrubbing any nested checkpoint_url so the offending value stays debuggable.
                        line_error["input"] = _redact_checkpoint_url_input(error_input, redact_all_strings=False)
                raise ValidationError.from_exception_data(cls.__name__, line_errors) from None

        return core_schema.no_info_wrap_validator_function(validate_with_redaction, schema)

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs: Any) -> Any:
        """Validate JSON, redacting the raw input (which may hold a signed URL) from parse errors.

        Parse with ``json.loads`` first because Pydantic's ``json_invalid`` error echoes the
        payload, which could expose credentials in a signed ``checkpoint_url``. Malformed
        JSON raises a redacted error that retains the parser's line, column, or byte
        offset. Valid JSON is passed unchanged to Pydantic to preserve JSON-mode validation.
        """
        try:
            json.loads(json_data)
        except json.JSONDecodeError as error:
            detail = f"Invalid JSON configuration: {error.msg} (line {error.lineno} column {error.colno})"
        except UnicodeDecodeError as error:
            detail = f"Invalid JSON configuration: not decodable as {error.encoding} (byte {error.start})"
        except ValueError:
            detail = "Invalid JSON configuration"
        else:
            return super().model_validate_json(json_data, **kwargs)  # type: ignore[misc]
        raise ValidationError.from_exception_data(
            cls.__name__,
            # pinned to pydantic 2.x error shape (InitErrorDetails: type/loc/input/ctx)
            [{"type": "json_invalid", "loc": (), "input": "<redacted-json-configuration>", "ctx": {"error": detail}}],
        ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevalidate a ``model_checkpoint`` assignment so the error is a plain, non-echoing ValueError.

        Assignment validation runs the field schema directly and bypasses the model
        wrap-validator above. Its ValidationError would expose the raw value, so this
        check raises a plain ValueError first to keep embedded credentials out of the error.
        """
        if name == "model_checkpoint" and isinstance(value, str):
            normalize_checkpoint_override(value)
        super().__setattr__(name, value)


class ParadeSequenceInput(BaseToolInput):
    """UTR sequences to score with a PARADE predictor.

    Attributes:
        sequences (list[str]): UTR sequence(s). A single string is normalized to a
            one-item list. ``U`` is mapped to ``T`` and ``N`` is allowed; mixed lengths
            are fine (the tool batches per length group).
    """

    sequences: list[str] = InputField(
        title="Sequences",
        description="UTR sequence(s) to score; RNA (U) is accepted and mapped to DNA (T).",
        min_length=1,
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value: Any) -> list[Any]:
        """Normalize a single UTR sequence to a one-item list."""
        if value is None:
            raise ValueError("sequences cannot be None")
        if isinstance(value, str):
            return [value]
        if not value:
            raise ValueError("sequences cannot be empty")
        return value  # type: ignore[no-any-return]

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, sequences: list[str]) -> list[str]:
        """Validate and normalize UTR sequences to the DNA alphabet."""
        return [normalize_utr_sequence(sequence) for sequence in sequences]

    def __len__(self) -> int:
        """Return the number of input sequences."""
        return len(self.sequences)


class ParadeCheckpointConfig(_SafeCheckpointUrlConfigMixin, BaseConfig):
    """Shared checkpoint-provisioning fields for PARADE predictors.

    Attributes:
        device (str): Device used for inference.
        model_checkpoint (str): ``"parade"``, the bundled checkpoints published with the paper
            (the worker picks the one for each construct type or stability), or your own
            checkpoint as a local ``.ckpt`` path or an ``https`` download link (a schemeless
            ``host.tld/path`` is accepted and normalized to ``https://``). A link also works on
            ``device="modal"``; a local path runs only locally.
        batch_size (int): Number of sequences to run per GPU batch.
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run PARADE inference on.",
        include_in_key=False,
    )
    model_checkpoint: str = ConfigField(
        title="Model / Checkpoint",
        default="parade",
        description="Bundled model name ('parade') or your own .ckpt path or https link.",
        reload_on_change=True,
        repr=False,
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=8,
        ge=1,
        description="Sequences per GPU forward pass; raise for throughput, lower if OOM",
        include_in_key=False,
    )

    @field_validator("model_checkpoint")
    @classmethod
    def _validate_checkpoint(cls, value: str) -> str:
        """Validate a bundled name, an https link (schemeless allowed), or a local path."""
        return normalize_checkpoint_override(value)

    def remote_unsupported_reason(self, device: RemoteDevice) -> str | None:
        """A local checkpoint path isn't present on a remote worker."""
        return local_checkpoint_on_remote(self.model_checkpoint, device)


class ParadeActivityMetrics(Metrics):
    """PARADE per-cell-type UTR activity predictions for one sequence.

    Values are the predicted activity mass-center for each requested PARADE cell code.
    Higher means higher predicted activity in that cell line. Metrics documented in
    ``metric_spec`` cover every code across the 5'UTR and 3'UTR panels.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        code: {
            "availability": "when requested",
            "type": "float",
            "min": None,
            "max": None,
            "description": f"Predicted PARADE UTR activity for {PARADE_CELL_LINE_NAMES[code]} (cell code {code}).",
            "better_values_are": "context-dependent",
        }
        for code in PARADE_CELL_LINE_NAMES
    }
