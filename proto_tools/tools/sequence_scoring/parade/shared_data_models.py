"""Shared constants and base models for the PARADE UTR activity/stability tools."""

from typing import Any, ClassVar, Literal

from pydantic import field_validator

from proto_tools.utils import BaseConfig, BaseToolInput, ConfigField, InputField
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

# PARADE anonymized cell-line codes, per UTR construct type. These are the exact
# condition tokens the published checkpoints were trained and queried with (see the
# ``CELLTYPE_CODES_*`` maps in the vendored ``utrdata_cl``); the model conditions on
# them through broadcast one-hot channels.
ParadeConstructType = Literal["utr5", "utr3"]
ParadeCellType = Literal["c1", "c2", "c4", "c6", "c17", "c13"]
PARADE_CELL_TYPES: dict[str, tuple[ParadeCellType, ...]] = {
    "utr5": ("c1", "c2", "c4", "c6", "c17"),
    "utr3": ("c1", "c2", "c4", "c6", "c17", "c13"),
}

_VALID_INPUT_CHARS = "N"  # U is normalized to T; A/C/G/T come from DNA_NUCLEOTIDES.


def normalize_utr_sequence(sequence: str) -> str:
    """Uppercase, strip whitespace, map RNA ``U`` to ``T``, and validate A/C/G/T/N.

    PARADE encodes UTRs in the DNA alphabet (``T``, not ``U``) and treats ``N`` as a
    uniform 0.25 base. Accepting ``U`` and normalizing it lets callers pass RNA.

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
    seq = sequence.upper().replace(" ", "").replace("\n", "").replace("U", "T")
    invalid_chars = return_invalid_nucleotide_chars(seq, additional_valid_chars=_VALID_INPUT_CHARS)
    if invalid_chars:
        raise ValueError(f"Invalid nucleotide characters in sequence: {', '.join(sorted(invalid_chars))}")
    return seq


class ParadeSequenceInput(BaseToolInput):
    """UTR sequences to score with a PARADE predictor.

    Attributes:
        sequences (list[str]): UTR sequence(s). A single string is normalized to a
            one-item list. ``U`` is mapped to ``T`` and ``N`` is allowed; all
            sequences in one call must share a length.
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


class ParadeCheckpointConfig(BaseConfig):
    """Shared checkpoint-provisioning fields for PARADE predictors.

    Attributes:
        device (str): Device used for inference.
        checkpoint_path (str): Optional local override path to a PARADE ``.ckpt``.
            Leave empty to download the pinned upstream checkpoint into the managed
            weights cache.
        checkpoint_url (str): Optional HTTPS override for the checkpoint download.
            Leave empty to use the pinned per-target URL.
        checkpoint_md5 (str): Optional MD5 override for the downloaded checkpoint.
            Leave empty to use the pinned per-target checksum.
        batch_size (int): Number of sequences to run per GPU batch.
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run PARADE inference on.",
        include_in_key=False,
    )
    checkpoint_path: str = ConfigField(
        title="Checkpoint Path",
        default="",
        description="Optional local PARADE .ckpt path; empty downloads the pinned upstream checkpoint.",
        reload_on_change=True,
    )
    checkpoint_url: str = ConfigField(
        title="Checkpoint URL",
        default="",
        description="Optional HTTPS override for the checkpoint; empty uses the pinned per-target URL.",
        reload_on_change=True,
    )
    checkpoint_md5: str = ConfigField(
        title="Checkpoint MD5",
        default="",
        description="Optional MD5 override; empty uses the pinned per-target checksum.",
        reload_on_change=True,
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of sequences to score simultaneously on GPU.",
        include_in_key=False,
    )

    def cloud_unsupported_reason(self) -> str | None:
        """A local checkpoint override isn't present on a hosted worker."""
        if self.checkpoint_path:
            return (
                "checkpoint_path points to a local file not available on device='cloud'. "
                "Leave it empty (the managed cache is used), or run locally with device='cpu'."
            )
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


def resolve_checkpoint_source(target: str, checkpoint_url: str, checkpoint_md5: str) -> tuple[str, str, str]:
    """Resolve the download URL, checksum, and cache filename for a PARADE target.

    Args:
        target (str): One of ``"utr5"``, ``"utr3"``, or ``"stability"``.
        checkpoint_url (str): Optional URL override; empty uses the pinned URL.
        checkpoint_md5 (str): Optional MD5 override; empty uses the pinned checksum.

    Returns:
        tuple[str, str, str]: ``(url, md5, filename)`` for the resolved checkpoint.
    """
    entry = PARADE_CHECKPOINTS[target]
    url = checkpoint_url or entry["url"]
    md5 = checkpoint_md5 or entry["md5"]
    return url, md5, entry["filename"]
