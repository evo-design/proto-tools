"""Malinois MPRA regulatory DNA activity scoring."""

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    DNA_NUCLEOTIDES,
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    GradientOutput,
    InputField,
    ToolInstance,
    return_invalid_nucleotide_chars,
)
from proto_tools.utils.tool_io import Metrics, MetricSpec

DEFAULT_MALINOIS_ARTIFACT_PATH = ""
DEFAULT_MALINOIS_DIR = ""
DEFAULT_MALINOIS_ARTIFACT_FILENAME = "MODELS-malinois_artifacts__20211113_021200__287348.tar.gz"
DEFAULT_MALINOIS_ARTIFACT_URL = (
    "https://zenodo.org/records/10698014/files/MODELS-malinois_artifacts__20211113_021200__287348.tar.gz?download=1"
)
DEFAULT_MALINOIS_ARTIFACT_MD5 = "375142a714e7df73c463b46113a65210"
MALINOIS_CELL_TYPES = ("K562", "HepG2", "SKNSH")
MalinoisCellType = Literal["K562", "HepG2", "SKNSH"]
MalinoisObjectiveDirection = Literal["max", "min"]
MalinoisLogits = list[list[list[float]]]
MalinoisGradientValue = list[list[list[float]]] | None


class MalinoisActivityMetrics(Metrics):
    """Raw Malinois activity predictions for one DNA insert.

    Metrics documented in ``metric_spec``:
        K562 (float): Raw Malinois activity prediction for K562 cells.
        HepG2 (float): Raw Malinois activity prediction for HepG2 cells.
        SKNSH (float): Raw Malinois activity prediction for SK-N-SH cells.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "K562": {
            "availability": "when requested",
            "type": "float",
            "min": None,
            "max": None,
            "description": "Raw Malinois activity prediction for K562 cells.",
        },
        "HepG2": {
            "availability": "when requested",
            "type": "float",
            "min": None,
            "max": None,
            "description": "Raw Malinois activity prediction for HepG2 cells.",
        },
        "SKNSH": {
            "availability": "when requested",
            "type": "float",
            "min": None,
            "max": None,
            "description": "Raw Malinois activity prediction for SK-N-SH cells.",
        },
    }


class MalinoisScoreInput(BaseToolInput):
    """Input for Malinois MPRA sequence scoring.

    Attributes:
        sequences (list[str]): DNA insert sequence(s) to score. A single string
            is normalized to a one-item list. Sequences must contain only A, C,
            G, and T; the configured ``seq_length`` is checked at run time.
    """

    sequences: list[str] = InputField(
        description="DNA insert sequence(s) to score with Malinois",
        min_length=1,
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value: Any) -> list[Any]:
        """Normalize a single DNA sequence to a one-item list."""
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
        """Validate and normalize DNA sequences."""
        normalized = []
        for sequence in sequences:
            if not sequence or not sequence.strip():
                raise ValueError("Sequence cannot be empty")
            seq = sequence.upper().replace(" ", "").replace("\n", "")
            invalid_chars = return_invalid_nucleotide_chars(seq)
            if invalid_chars:
                raise ValueError(f"Invalid nucleotide characters in sequence: {', '.join(sorted(invalid_chars))}")
            normalized.append(seq)
        return normalized

    def __len__(self) -> int:
        """Return the number of input sequences."""
        return len(self.sequences)


class MalinoisScoreResult(BaseModel):
    """Per-sequence Malinois scoring result.

    Attributes:
        sequence (str): DNA sequence that was scored.
        sequence_length (int): Length of the scored DNA sequence.
        scores (MalinoisActivityMetrics): Malinois predictions keyed by
            requested cell type name.
    """

    sequence: str = Field(description="DNA sequence scored by Malinois")
    sequence_length: int = Field(description="Length of the scored DNA sequence")
    scores: MalinoisActivityMetrics = Field(description="Malinois predictions keyed by cell type")


class MalinoisScoreOutput(BaseToolOutput):
    """Output from Malinois MPRA scoring.

    Attributes:
        results (list[MalinoisScoreResult]): Per-sequence Malinois predictions.
        cell_types (list[str]): Cell types included in each result's ``scores``.
        seq_length (int): Expected insert length used for MPRA flank padding.
    """

    results: list[MalinoisScoreResult] = Field(description="Per-sequence Malinois scoring results")
    cell_types: list[str] = Field(description="Cell types included in each score dictionary")
    seq_length: int = Field(description="Expected Malinois insert length")

    def __len__(self) -> int:
        """Return the number of per-sequence results."""
        return len(self.results)

    def __getitem__(self, index: int) -> MalinoisScoreResult:
        """Return a per-sequence result."""
        return self.results[index]

    def __iter__(self) -> Iterator[MalinoisScoreResult]:  # type: ignore[override]
        """Iterate over per-sequence results."""
        return iter(self.results)

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Path | str, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")
        data = {
            "results": [
                {
                    "sequence": result.sequence,
                    "sequence_length": result.sequence_length,
                    "scores": dict(result.scores.items()),
                }
                for result in self.results
            ],
            "cell_types": self.cell_types,
            "seq_length": self.seq_length,
        }
        if file_format == "json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            return
        if file_format == "csv":
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sequence_index", "sequence", "sequence_length", *self.cell_types])
                for idx, result in enumerate(self.results):
                    writer.writerow(
                        [
                            idx,
                            result.sequence,
                            result.sequence_length,
                            *[result.scores[cell_type] for cell_type in self.cell_types],
                        ]
                    )
            return
        raise ValueError(f"Unsupported format: {file_format}")


class MalinoisScoreConfig(BaseConfig):
    """Configuration for Malinois MPRA scoring.

    Attributes:
        cell_types (list[MalinoisCellType]): Cell-type outputs to return.
        seq_length (int): Expected insert length before MPRA flank padding.
        artifact_path (str): Optional local override path to the Malinois model
            artifact tarball. Leave empty to download ``artifact_url`` into the
            managed weights cache.
        artifact_url (str): HTTPS URL used to provision the Malinois artifact.
        artifact_md5 (str): Optional MD5 checksum for the downloaded artifact.
        malinois_dir (str): Optional local override directory containing
            unpacked artifact metadata. Leave empty to use the cache extraction
            directory.
        batch_size (int): Number of sequences to process in each GPU batch.
        device (str): Device used for inference.
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run Malinois inference on",
        hidden=True,
        include_in_key=False,
    )
    cell_types: list[MalinoisCellType] = ConfigField(
        title="Cell Types",
        default=["K562", "HepG2", "SKNSH"],
        description="Malinois cell-type outputs to return.",
    )
    seq_length: int = ConfigField(
        title="Sequence Length",
        default=200,
        ge=1,
        description="Expected DNA insert length before MPRA flank padding.",
        reload_on_change=True,
    )
    artifact_path: str = ConfigField(
        title="Artifact Path",
        default=DEFAULT_MALINOIS_ARTIFACT_PATH,
        description="Optional local artifact tarball path; empty uses the managed cache download.",
        reload_on_change=True,
        hidden=True,
    )
    artifact_url: str = ConfigField(
        title="Artifact URL",
        default=DEFAULT_MALINOIS_ARTIFACT_URL,
        description="HTTPS URL used to provision the Malinois artifact.",
        reload_on_change=True,
        hidden=True,
    )
    artifact_md5: str = ConfigField(
        title="Artifact MD5",
        default=DEFAULT_MALINOIS_ARTIFACT_MD5,
        description="Expected MD5 checksum for the downloaded Malinois artifact.",
        reload_on_change=True,
        hidden=True,
    )
    malinois_dir: str = ConfigField(
        title="Malinois Directory",
        default=DEFAULT_MALINOIS_DIR,
        description="Optional local Malinois metadata directory; empty uses the managed cache extraction.",
        reload_on_change=True,
        hidden=True,
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of sequences to score simultaneously on GPU.",
        advanced=True,
        include_in_key=False,
    )

    @field_validator("cell_types", mode="before")
    @classmethod
    def normalize_cell_types(cls, value: Any) -> list[Any]:
        """Normalize a single cell type to a list."""
        if isinstance(value, str):
            return [value]
        return value  # type: ignore[no-any-return]

    @model_validator(mode="after")
    def validate_cell_types(self) -> "MalinoisScoreConfig":
        """Validate that at least one unique cell type is requested."""
        if not self.cell_types:
            raise ValueError("cell_types cannot be empty")
        if len(set(self.cell_types)) != len(self.cell_types):
            raise ValueError("cell_types must be unique")
        return self


class MalinoisGradientLossTerm(BaseModel):
    """One differentiable Malinois objective term.

    Attributes:
        cell_type (MalinoisCellType): Malinois output to optimize.
        direction (MalinoisObjectiveDirection): ``"max"`` minimizes
            ``1 - sigmoid((raw - center) / scale)``; ``"min"`` minimizes
            ``sigmoid((raw - center) / scale)``.
        weight (float): Non-negative scalar applied before terms are summed.
        sigmoid_center (float): Raw Malinois score where the sigmoid is 0.5.
        sigmoid_scale (float): Positive scale for the raw score transform.
    """

    cell_type: MalinoisCellType = Field(default="K562", description="Malinois cell-type output to optimize.")
    direction: MalinoisObjectiveDirection = Field(default="max", description="Optimization direction.")
    weight: float = Field(default=1.0, ge=0.0, description="Non-negative term weight.")
    sigmoid_center: float = Field(default=4.0, description="Raw score at the midpoint of the sigmoid transform.")
    sigmoid_scale: float = Field(default=1.0, gt=0.0, description="Positive sigmoid scale.")


class MalinoisGradientInput(BaseToolInput):
    """Input for differentiable Malinois sequence optimization.

    Attributes:
        logits (MalinoisLogits): Batched relaxed DNA sequence logits with shape
            ``(B, L, 4)`` in ``A,C,G,T`` order. Use ``B=1`` for a single
            design candidate.
        temperature (float): Softmax temperature used to relax logits.
    """

    logits: MalinoisLogits = InputField(
        description="Batched relaxed DNA logits with shape (B, L, 4) in A,C,G,T order.",
        examples=[[[[0.0] * 4] * 200]],
    )
    temperature: float = InputField(
        default=1.0,
        gt=0.0,
        description="Softmax temperature used to convert logits into relaxed nucleotide probabilities.",
    )

    @field_validator("logits")
    @classmethod
    def validate_logits(cls, logits: MalinoisLogits) -> MalinoisLogits:
        """Ensure logits are a non-empty ``B x L x 4`` tensor."""
        if not logits:
            raise ValueError("logits must contain at least one batch item")
        expected_width = len(DNA_NUCLEOTIDES)
        seq_length: int | None = None
        for batch_idx, matrix in enumerate(logits):
            if not matrix:
                raise ValueError(f"logits batch {batch_idx} must contain at least one position")
            if not isinstance(matrix[0], list):
                raise ValueError("logits must have shape (B, L, 4); use batch size 1 for a single candidate")
            if seq_length is None:
                seq_length = len(matrix)
            elif len(matrix) != seq_length:
                raise ValueError(
                    f"all batched logits must have the same sequence length; "
                    f"batch 0 has {seq_length}, batch {batch_idx} has {len(matrix)}"
                )
            for row_idx, row in enumerate(matrix):
                if len(row) != expected_width:
                    raise ValueError(
                        f"logits batch {batch_idx} row {row_idx} must have {expected_width} columns, got {len(row)}"
                    )
        return logits


class MalinoisGradientConfig(BaseConfig):
    """Configuration for differentiable Malinois activity objectives.

    Attributes:
        loss_terms (list[MalinoisGradientLossTerm]): Per-cell objective terms
            summed into one scalar loss.
        seq_length (int): Expected insert length before Malinois flank padding.
        artifact_path (str): Optional local artifact tarball path.
        artifact_url (str): URL used to provision the Malinois artifact.
        artifact_md5 (str): Expected checksum for the downloaded artifact.
        malinois_dir (str): Optional extracted Malinois artifact directory.
        soft (float): Blend hard argmax one-hot (0) to softmax probabilities (1).
        hard (float): Straight-through hard-forward coefficient.
        compute_gradient (bool): Run backward pass and return gradient.
        device (str): Device used for inference and backpropagation.
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run Malinois inference on",
        hidden=True,
        include_in_key=False,
    )
    loss_terms: list[MalinoisGradientLossTerm] = ConfigField(
        default_factory=lambda: [MalinoisGradientLossTerm()],
        title="Loss Terms",
        description="Per-cell Malinois objectives to sum before backpropagation.",
    )
    seq_length: int = ConfigField(
        title="Sequence Length",
        default=200,
        ge=1,
        description="Expected DNA insert length before MPRA flank padding.",
        reload_on_change=True,
    )
    artifact_path: str = ConfigField(
        title="Artifact Path",
        default=DEFAULT_MALINOIS_ARTIFACT_PATH,
        description="Optional local artifact tarball path; empty uses the managed cache download.",
        reload_on_change=True,
        hidden=True,
    )
    artifact_url: str = ConfigField(
        title="Artifact URL",
        default=DEFAULT_MALINOIS_ARTIFACT_URL,
        description="HTTPS URL used to provision the Malinois artifact.",
        reload_on_change=True,
        hidden=True,
    )
    artifact_md5: str = ConfigField(
        title="Artifact MD5",
        default=DEFAULT_MALINOIS_ARTIFACT_MD5,
        description="Expected MD5 checksum for the downloaded Malinois artifact.",
        reload_on_change=True,
        hidden=True,
    )
    malinois_dir: str = ConfigField(
        title="Malinois Directory",
        default=DEFAULT_MALINOIS_DIR,
        description="Optional local Malinois metadata directory; empty uses the managed cache extraction.",
        reload_on_change=True,
        hidden=True,
    )
    soft: float = ConfigField(
        title="Soft Mixing",
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Blend hard argmax one-hot (0) to softmax probabilities (1).",
        advanced=True,
    )
    hard: float = ConfigField(
        title="Hard Mixing",
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Straight-through hard-forward coefficient.",
        advanced=True,
    )
    compute_gradient: bool = ConfigField(
        title="Compute Gradient",
        default=True,
        description="Run backward pass and return gradient; set False for forward-only scoring.",
        advanced=True,
    )

    @model_validator(mode="after")
    def validate_loss_terms(self) -> "MalinoisGradientConfig":
        """Validate that at least one loss term is active."""
        if not self.loss_terms:
            raise ValueError("loss_terms cannot be empty")
        return self


class MalinoisGradientSampleMetrics(Metrics):
    """Per-sample differentiable Malinois objective metrics.

    Raw cell-type predictions are stored as metrics when the corresponding cell
    type is present in the objective. Per-term transformation metadata is kept
    in ``loss_terms`` so metric values remain scalar and easy to index.

    Metrics documented in ``metric_spec``:
        loss (float): Weighted scalar objective value for one sample.
        K562 (float): Raw Malinois activity prediction for K562 cells.
        HepG2 (float): Raw Malinois activity prediction for HepG2 cells.
        SKNSH (float): Raw Malinois activity prediction for SK-N-SH cells.

    Attributes:
        loss_terms (list[dict[str, Any]]): Per-objective-term metadata,
            including direction, weight, sigmoid transform, and weighted score.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "loss": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": None,
            "description": "Weighted scalar Malinois objective value for one sample.",
        },
        "K562": {
            "availability": "when requested",
            "type": "float",
            "min": None,
            "max": None,
            "description": "Raw Malinois activity prediction for K562 cells.",
        },
        "HepG2": {
            "availability": "when requested",
            "type": "float",
            "min": None,
            "max": None,
            "description": "Raw Malinois activity prediction for HepG2 cells.",
        },
        "SKNSH": {
            "availability": "when requested",
            "type": "float",
            "min": None,
            "max": None,
            "description": "Raw Malinois activity prediction for SK-N-SH cells.",
        },
    }
    primary_metric: str | None = "loss"

    loss_terms: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-objective-term transform metadata for this sample.",
    )


def _build_gradient_sample_metrics(metrics: dict[str, Any]) -> list[MalinoisGradientSampleMetrics]:
    """Build per-sample metric containers from the standalone gradient payload."""
    raw_scores_by_sample = metrics.get("raw_scores")
    losses = metrics.get("losses")
    loss_terms_by_sample = metrics.get("loss_terms")
    if not isinstance(raw_scores_by_sample, list) or not isinstance(losses, list):
        return []

    sample_metrics = []
    for sample_idx, loss in enumerate(losses):
        raw_scores = raw_scores_by_sample[sample_idx] if sample_idx < len(raw_scores_by_sample) else {}
        loss_terms = (
            loss_terms_by_sample[sample_idx]
            if isinstance(loss_terms_by_sample, list) and sample_idx < len(loss_terms_by_sample)
            else []
        )
        metric_values: dict[str, float] = {"loss": float(loss)}
        if isinstance(raw_scores, dict):
            metric_values.update(
                {
                    cell_type: float(raw_scores[cell_type])
                    for cell_type in MALINOIS_CELL_TYPES
                    if cell_type in raw_scores
                }
            )
        sample_metrics.append(MalinoisGradientSampleMetrics.model_validate({**metric_values, "loss_terms": loss_terms}))
    return sample_metrics


class MalinoisGradientOutput(GradientOutput):
    """Differentiable Malinois activity output.

    Attributes:
        gradient (MalinoisGradientValue): Gradient tensor matching input DNA
            logits, or ``None`` when ``compute_gradient=False``.
        loss (float): Sum of per-sample weighted scalar objective values.
            Per-sample values are available in ``sample_metrics``.
        sample_metrics (list[MalinoisGradientSampleMetrics]): Per-sample metric
            containers with scalar loss and raw cell-type scores.
        metrics (dict[str, Any]): Legacy metadata bundle from the standalone
            worker, including raw scores, objective-term metadata, and runtime
            relaxation parameters.
        vocab (list[str]): DNA column ordering for logits and gradient.
    """

    gradient: MalinoisGradientValue = Field(
        default=None,
        description="Gradient w.r.t. input logits. None when compute_gradient=False.",
    )
    sample_metrics: list[MalinoisGradientSampleMetrics] = Field(
        default_factory=list,
        description="Per-sample Malinois objective metrics.",
    )

    @model_validator(mode="after")
    def populate_sample_metrics(self) -> "MalinoisGradientOutput":
        """Populate metric containers from the legacy metadata payload."""
        if not self.sample_metrics:
            self.sample_metrics = _build_gradient_sample_metrics(self.metrics)
        return self

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Export the gradient bundle as JSON."""
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        base = Path(export_path)
        json_path = base.parent / f"{base.name}.json"
        payload = self.model_dump(include={"gradient", "loss", "sample_metrics", "metrics", "vocab"})
        json_path.write_text(json.dumps(payload, indent=2))


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return MalinoisScoreInput(sequences=["A" * 200])


def example_gradient_input() -> Any:
    """Minimal valid input for differentiable Malinois examples."""
    return MalinoisGradientInput(logits=[[[0.0] * len(DNA_NUCLEOTIDES)] * 200])


@tool(
    key="malinois-score",
    label="Malinois MPRA Score",
    category="sequence_scoring",
    input_class=MalinoisScoreInput,
    config_class=MalinoisScoreConfig,
    output_class=MalinoisScoreOutput,
    metrics_class=MalinoisActivityMetrics,
    description="Score regulatory DNA activity using the Malinois MPRA model",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="sequences",
    iterable_output_field="results",
    cacheable=True,
)
def run_malinois_score(
    inputs: MalinoisScoreInput,
    config: MalinoisScoreConfig,
    instance: Any = None,
) -> MalinoisScoreOutput:
    """Score regulatory DNA sequences with Malinois.

    Args:
        inputs (MalinoisScoreInput): DNA insert sequences to score.
        config (MalinoisScoreConfig): Malinois runtime and model configuration.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        MalinoisScoreOutput: Per-sequence Malinois predictions keyed by cell type.
    """
    invalid_lengths = [len(sequence) for sequence in inputs.sequences if len(sequence) != config.seq_length]
    if invalid_lengths:
        raise ValueError(f"All Malinois sequences must have length {config.seq_length}; got {invalid_lengths[0]}")

    output_data = ToolInstance.dispatch(
        "malinois",
        {
            "operation": "score",
            "sequences": inputs.sequences,
            "cell_types": config.cell_types,
            "seq_length": config.seq_length,
            "artifact_path": config.artifact_path,
            "artifact_url": config.artifact_url,
            "artifact_md5": config.artifact_md5,
            "malinois_dir": config.malinois_dir,
            "batch_size": config.batch_size,
            "device": config.device,
            "verbose": config.verbose,
            "seed": config.seed,
        },
        instance=instance,
        config=config,
    )

    score_rows = output_data["scores"]
    if len(score_rows) != len(inputs.sequences):
        raise ValueError(f"Expected {len(inputs.sequences)} Malinois score rows, got {len(score_rows)}")

    return MalinoisScoreOutput(
        results=[
            MalinoisScoreResult(
                sequence=sequence,
                sequence_length=len(sequence),
                scores=MalinoisActivityMetrics.model_validate(
                    {cell_type: float(scores[cell_type]) for cell_type in config.cell_types}
                ),
            )
            for sequence, scores in zip(inputs.sequences, score_rows, strict=True)
        ],
        cell_types=list(config.cell_types),
        seq_length=config.seq_length,
    )


@tool(
    key="malinois-gradient",
    label="Malinois Gradient",
    category="sequence_scoring",
    input_class=MalinoisGradientInput,
    config_class=MalinoisGradientConfig,
    output_class=MalinoisGradientOutput,
    metrics_class=MalinoisGradientSampleMetrics,
    description="Compute differentiable Malinois MPRA activity losses and gradients for relaxed DNA logits",
    uses_gpu=True,
    example_input=example_gradient_input,
    cacheable=False,
)
def run_malinois_gradient(
    inputs: MalinoisGradientInput,
    config: MalinoisGradientConfig,
    instance: Any = None,
) -> MalinoisGradientOutput:
    """Compute Malinois activity gradients with respect to relaxed DNA logits."""
    invalid_lengths = [len(matrix) for matrix in inputs.logits if len(matrix) != config.seq_length]
    if invalid_lengths:
        raise ValueError(f"Malinois gradient logits must have length {config.seq_length}; got {invalid_lengths[0]}")

    result = ToolInstance.dispatch(
        "malinois",
        {
            "operation": "compute_gradient",
            "logits": inputs.logits,
            "temperature": inputs.temperature,
            "loss_terms": [term.model_dump() for term in config.loss_terms],
            "seq_length": config.seq_length,
            "artifact_path": config.artifact_path,
            "artifact_url": config.artifact_url,
            "artifact_md5": config.artifact_md5,
            "malinois_dir": config.malinois_dir,
            "soft": config.soft,
            "hard": config.hard,
            "compute_gradient": config.compute_gradient,
            "device": config.device,
            "verbose": config.verbose,
            "seed": config.seed,
        },
        instance=instance,
        config=config,
    )
    return MalinoisGradientOutput(**result)
