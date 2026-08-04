"""Shared constants and base models for the PARADE UTR activity/stability tools."""

import hashlib
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

# PARADE ships its trained checkpoints inside the public MIT-licensed GitHub
# repository. We provision them from raw.githubusercontent.com pinned to a commit
# so the exact artifact the paper published is what gets loaded.
PARADE_COMMIT = "f8e3e02688918e40d85eff0976a984810ee04372"
_PARADE_RAW_BASE = f"https://raw.githubusercontent.com/autosome-ru/parade/{PARADE_COMMIT}"

# Per-target checkpoint provisioning. ``url`` uses percent-encoded ``=`` because the
# upstream filenames embed ``epoch=..-step=..``. ``filename`` is the sanitized local
# cache name (no ``=``). ``md5`` is the checksum of the upstream artifact.
PARADE_CHECKPOINTS: dict[str, dict[str, str]] = {
    "utr5": {
        "filename": "parade-model-utr5-deltas-epoch9-step840.ckpt",
        "url": f"{_PARADE_RAW_BASE}/predictor/regression_multiple/saved_models/model-utr5-deltas-epoch%3D9-step%3D840.ckpt",
        "md5": "a48aeffc516e32f4d8780b855bbcd849",
    },
    "utr3": {
        "filename": "parade-model-utr3-deltas-epoch9-step1330.ckpt",
        "url": f"{_PARADE_RAW_BASE}/predictor/regression_multiple/saved_models/model-utr3-deltas-epoch%3D9-step%3D1330.ckpt",
        "md5": "399be95c4f6b9aa80c1c17452f31d558",
    },
    "stability": {
        "filename": "parade-stability-epoch24-step725.ckpt",
        "url": f"{_PARADE_RAW_BASE}/predictor/regression_stability/saved_models/stability-epoch%3D24-step%3D725.ckpt",
        "md5": "511c0b4d794f948708ab1e6fa866734b",
    },
}

# PARADE anonymized cell-line codes exposed (queryable) per UTR construct type -- the codes the
# upstream tutorial predicts on. These are a subset of the model's trained condition channels: the
# vendored ``CELLTYPE_CODES_UTR3`` map also has ``c10``, which the checkpoint conditions on but the
# upstream pipeline never queries, so it is intentionally omitted from the queryable panel here.
# The model still conditions on all of its trained channels via broadcast one-hot inputs.
ParadeConstructType = Literal["utr5", "utr3"]
ParadeCellType = Literal["c1", "c2", "c4", "c6", "c17", "c13"]
PARADE_CELL_TYPES: dict[str, tuple[ParadeCellType, ...]] = {
    "utr5": ("c1", "c2", "c4", "c6", "c17"),
    "utr3": ("c1", "c2", "c4", "c6", "c17", "c13"),
}

_VALID_INPUT_CHARS = "N"  # U is normalized to T; A/C/G/T come from DNA_NUCLEOTIDES.

_CUSTOM_CHECKPOINT_REJECTED = (
    "A custom checkpoint cannot be used with device='proto'. Loading a checkpoint executes "
    "the code inside it, and Proto's hosted service runs only the pinned checkpoints it can "
    "verify. Leave checkpoint empty to use the pinned one, run locally with device='cpu' or "
    "device='cuda', or use device='modal' to run it in a Modal workspace you own."
)


def normalize_utr_sequence(sequence: str) -> str:
    """Uppercase, strip whitespace, map RNA ``U`` to ``T``, and validate A/C/G/T/N.

    PARADE encodes UTRs in the DNA alphabet (``T``, not ``U``); accepting ``U`` and
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
    """Validate a custom checkpoint URL: HTTPS scheme, a real host, and no embedded credentials.

    A PARADE checkpoint is unpickled after download, so a custom URL must be well-formed and
    fetched over a non-downgradeable transport. Malformed values (``https:relative``,
    ``https:///no-host``, ``https://``) are rejected here rather than after worker startup, and
    ``user:pass@`` userinfo is refused (URL-embedded credentials are unsupported — auth to
    Hugging Face flows through ``HF_TOKEN``). Empty passes through (the pinned URL is used).

    Error messages never echo the raw value, which may itself contain the credential.
    """
    if not value:
        return value
    if any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("checkpoint_url must not contain whitespace or control characters")
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError("checkpoint_url must use ASCII with non-ASCII characters percent-encoded") from None
    try:
        parsed = urllib.parse.urlparse(value)
        parsed.port  # noqa: B018 - property access validates the port (ValueError if non-numeric/out of range)
    except ValueError:
        raise ValueError("checkpoint_url is not a valid URL") from None
    if parsed.scheme != "https":
        raise ValueError("checkpoint_url must be an https:// URL")
    if not parsed.hostname:
        raise ValueError("checkpoint_url must include a host (e.g. https://host/path)")
    if parsed.username or parsed.password:
        raise ValueError("checkpoint_url must not embed credentials (user:pass@); use HF_TOKEN instead")
    host = parsed.hostname.rstrip(".")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        labels = host.split(".")
        if (
            len(host) > 253
            or any(not label or len(label) > 63 for label in labels)
            or any(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label) is None for label in labels)
        ):
            raise ValueError("checkpoint_url host must be a valid IPv4, IPv6, or DNS hostname") from None
    return value


def _looks_like_checkpoint_url(value: str) -> bool:
    """True if ``value`` is a download link — ``http(s)://…`` or a schemeless ``host.tld/path``.

    A bare filename or local path (``model.ckpt``, ``sub/x.ckpt``, ``/oak/x.ckpt``, ``./x``) is not a
    link; a schemeless ``huggingface.co/user/model.ckpt`` is (and is normalized to ``https://`` on use).
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


def normalize_checkpoint_override(value: str) -> str:
    """Validate a checkpoint override: an https link (schemeless allowed) or a local path.

    A link (``http(s)://…`` or ``host.tld/path``) is normalized to ``https://`` and validated
    (HTTPS-only, real host, no embedded credentials); a local path passes through unchanged. A
    checkpoint is unpickled after download, so a link must use a non-downgradeable HTTPS transport.
    """
    if not value or not _looks_like_checkpoint_url(value):
        return value
    normalized = value if urllib.parse.urlparse(value).scheme else f"https://{value}"
    return require_https_checkpoint_url(normalized)


def _redact_checkpoint_url_input(value: Any, *, redact_all_strings: bool = False) -> Any:
    """Recursively replace checkpoint URL values inside a rejected model input."""
    if redact_all_strings and isinstance(value, (str, bytes, bytearray)):
        return "<redacted-checkpoint>"
    if isinstance(value, dict):
        return {
            ("<redacted-key>" if redact_all_strings and isinstance(key, str) else key): (
                "<redacted-checkpoint>"
                if key == "checkpoint"
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
                    if "checkpoint" in location:
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

        The pre-parse with ``json.loads`` is deliberate, not gratuitous: pydantic's own
        ``json_invalid`` error echoes the offending payload verbatim, which could leak a
        credentialed/signed ``checkpoint_url``. So we detect malformedness first and raise our own
        redacted error, then hand a valid payload to pydantic UNCHANGED (preserving JSON-mode
        validation semantics). The surfaced message keeps the parser's POSITION (line/column or
        byte offset) — which says *where* the JSON is broken without reproducing its contents.
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
        """Prevalidate a ``checkpoint`` assignment so the error is a plain, non-echoing ValueError.

        NOT redundant with ``validate_assignment``: assignment re-validation runs the field schema
        directly (never the model wrap-validator above), so its ValidationError would render the raw
        value. Raising a plain ValueError here first keeps any embedded credential out of the text.
        """
        if name == "checkpoint" and isinstance(value, str):
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
        checkpoint (str): Your own checkpoint to use instead of the pinned upstream one, given
            as a local ``.ckpt`` path or an ``https`` download link (a schemeless
            ``host.tld/path`` is accepted and normalized to ``https://``). Works on
            ``device="cpu"``, ``device="cuda"``, and ``device="modal"``, all of which run on
            hardware you control. Rejected on ``device="proto"``, because loading a checkpoint
            executes the code inside it and Proto's hosted service runs only checkpoints it can
            verify. Leave empty to use the pinned per-target checkpoint.
        batch_size (int): Number of sequences to run per GPU batch.
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run PARADE inference on.",
        include_in_key=False,
    )
    checkpoint: str = ConfigField(
        title="Checkpoint",
        default="",
        description="Your own .ckpt path or https link, used instead of the pinned checkpoint.",
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

    @field_validator("checkpoint")
    @classmethod
    def _validate_checkpoint(cls, value: str) -> str:
        """Validate a checkpoint override: an https link (schemeless allowed) or a local path."""
        return normalize_checkpoint_override(value)

    def remote_unsupported_reason(self, device: RemoteDevice) -> str | None:
        """Proto will not load a custom checkpoint; a deployment the caller owns may."""
        if device == "proto" and self.checkpoint:
            return _CUSTOM_CHECKPOINT_REJECTED
        return None


class ParadeActivityMetrics(Metrics):
    """PARADE per-cell-type UTR activity predictions for one sequence.

    Values are the predicted activity mass-center for each requested PARADE cell code;
    higher means higher predicted activity in that cell line. Metrics documented in
    ``metric_spec`` cover every code across the 5'UTR and 3'UTR panels.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        code: {
            "availability": "when requested",
            "type": "float",
            "min": None,
            "max": None,
            "description": f"Predicted PARADE UTR activity for cell code {code}.",
            "better_values_are": "context-dependent",
        }
        for code in ("c1", "c2", "c4", "c6", "c17", "c13")
    }


def resolve_checkpoint_source(target: str, checkpoint: str) -> tuple[str, str, str, str]:
    """Resolve ``(checkpoint_path, url, md5, filename)`` for the standalone dispatch.

    - ``checkpoint`` empty            -> pinned URL + built-in pinned checksum (no local path)
    - ``checkpoint`` is a link        -> download it (no checksum), under a link-specific cache name
    - ``checkpoint`` is a local path  -> pass the path through (no download)

    Args:
        target (str): One of ``"utr5"``, ``"utr3"``, or ``"stability"``.
        checkpoint (str): Optional override — empty, an https link, or a local ``.ckpt`` path.

    Returns:
        tuple[str, str, str, str]: ``(checkpoint_path, url, md5, filename)`` for the resolved checkpoint.
    """
    entry = PARADE_CHECKPOINTS[target]
    if not checkpoint:
        return "", entry["url"], entry["md5"], entry["filename"]
    if _looks_like_checkpoint_url(checkpoint):
        url = checkpoint if urllib.parse.urlparse(checkpoint).scheme else f"https://{checkpoint}"
        digest = hashlib.sha256(url.encode()).hexdigest()[:12]
        return "", url, "", f"parade-{target}-custom-{digest}.ckpt"
    return checkpoint, "", "", ""
