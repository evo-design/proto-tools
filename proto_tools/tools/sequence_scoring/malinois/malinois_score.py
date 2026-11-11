"""Malinois MPRA regulatory DNA activity scoring."""

import csv
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    DNA_NUCLEOTIDES,
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    BatchedNucleotideGradientInput,
    ConfigField,
    DifferentiableTerm,
    GradientConfig,
    GradientMetadata,
    GradientOutput,
    GradientResult,
    InputField,
    NucleotideSequenceLogits,
    TermCatalog,
    TermGradient,
    ToolInstance,
    resolve_terms,
    return_invalid_nucleotide_chars,
)
from proto_tools.utils.device import RemoteDevice
from proto_tools.utils.tool_io import Directionality, Metrics, MetricSpec

DEFAULT_MALINOIS_ARTIFACT_PATH = ""
DEFAULT_MALINOIS_DIR = ""
DEFAULT_MALINOIS_ARTIFACT_FILENAME = "MODELS-malinois_artifacts__20211113_021200__287348.tar.gz"
DEFAULT_MALINOIS_ARTIFACT_URL = (
    "https://zenodo.org/records/10698014/files/MODELS-malinois_artifacts__20211113_021200__287348.tar.gz?download=1"
)
DEFAULT_MALINOIS_ARTIFACT_MD5 = "375142a714e7df73c463b46113a65210"
MALINOIS_CELL_TYPES = ("K562", "HepG2", "SKNSH")
MalinoisCellType = Literal["K562", "HepG2", "SKNSH"]


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
            "better_values_are": "context-dependent",
        },
        "HepG2": {
            "availability": "when requested",
            "type": "float",
            "min": None,
            "max": None,
            "description": "Raw Malinois activity prediction for HepG2 cells.",
            "better_values_are": "context-dependent",
        },
        "SKNSH": {
            "availability": "when requested",
            "type": "float",
            "min": None,
            "max": None,
            "description": "Raw Malinois activity prediction for SK-N-SH cells.",
            "better_values_are": "context-dependent",
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
        title="Sequences",
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

    sequence: str = Field(title="Sequence", description="DNA sequence scored by Malinois")
    sequence_length: int = Field(title="Sequence Length", description="Length of the scored DNA sequence")
    scores: MalinoisActivityMetrics = Field(
        title="Activity Scores", description="Malinois predictions keyed by cell type"
    )


class MalinoisScoreOutput(BaseToolOutput):
    """Output from Malinois MPRA scoring.

    Attributes:
        results (list[MalinoisScoreResult]): Per-sequence Malinois predictions.
        cell_types (list[str]): Cell types included in each result's ``scores``.
        seq_length (int): Expected insert length used for MPRA flank padding.
    """

    results: list[MalinoisScoreResult] = Field(title="Results", description="Per-sequence Malinois scoring results")
    cell_types: list[str] = Field(title="Cell Types", description="Cell types included in each score dictionary")
    seq_length: int = Field(title="Insert Length", description="Expected Malinois insert length")

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
    )
    artifact_url: str = ConfigField(
        title="Artifact URL",
        default=DEFAULT_MALINOIS_ARTIFACT_URL,
        description="HTTPS URL used to provision the Malinois artifact.",
        reload_on_change=True,
    )
    artifact_md5: str = ConfigField(
        title="Artifact MD5",
        default=DEFAULT_MALINOIS_ARTIFACT_MD5,
        description="Expected MD5 checksum for the downloaded Malinois artifact.",
        reload_on_change=True,
    )
    malinois_dir: str = ConfigField(
        title="Malinois Directory",
        default=DEFAULT_MALINOIS_DIR,
        description="Optional local Malinois metadata directory; empty uses the managed cache extraction.",
        reload_on_change=True,
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=8,
        ge=1,
        description="Sequences per GPU forward pass; raise for throughput, lower if OOM",
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

    def remote_unsupported_reason(self, device: RemoteDevice) -> str | None:
        """Local artifact/metadata overrides aren't present on a hosted worker."""
        if self.artifact_path or self.malinois_dir:
            return f"artifact_path/malinois_dir point to local files not available on device='{device}'. Leave them empty (the managed cache is used), or run locally with device='cpu'."
        return None


class MalinoisActivityTerm(DifferentiableTerm):
    """One differentiable Malinois quantity: a single cell type's predicted MPRA activity.

    The value is the raw activity score in the model's own units, not a bounded transform of
    it. Whether more activity is wanted, how steeply to saturate it, and how it balances
    against another cell are goals a caller applies — ``SigmoidGoal(direction=..., center=...,
    scale=...)`` reproduces the bounded objective this tool used to compute internally.

    Attributes:
        id (Literal["activity"]): Fixed identifier for this kind of quantity. Every cell's term
            reports the same one; the ``cell_type`` selector is what tells two of them apart.
        cell_type (MalinoisCellType): Malinois cell-type output whose activity is differentiated.
    """

    #: Whether more activity is wanted depends on whether the cell is a target or an off-target.
    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "context-dependent"

    #: What this quantity measures, copied into each gradient's metadata.
    DESCRIPTION: ClassVar[str] = "Raw predicted Malinois MPRA activity for one cell line"

    id: Literal["activity"] = Field(
        default="activity",
        title="Term ID",
        description="Identifies this quantity as a Malinois MPRA activity prediction.",
    )
    cell_type: MalinoisCellType = Field(
        default="K562", title="Cell Type", description="Malinois cell-type output to differentiate."
    )


class MalinoisTermGradient(TermGradient):
    """One Malinois cell type's raw activity and its gradient.

    Attributes:
        term (MalinoisActivityTerm): The cell-type quantity this gradient answers.
    """

    term: MalinoisActivityTerm = Field(title="Term", description="Cell-type quantity answered.")


class MalinoisGradientResult(GradientResult):
    """One evaluated sample and every cell-type activity differentiated for it.

    Attributes:
        terms (list[MalinoisTermGradient]): One entry per requested cell, in the order requested.
    """

    terms: list[MalinoisTermGradient] = Field(  # type: ignore[assignment]  # narrows the element type
        default_factory=list,
        title="Terms",
        description="One gradient per requested cell-type quantity, in the order requested.",
    )


class MalinoisGradientInput(BatchedNucleotideGradientInput):
    """Input for differentiable Malinois sequence optimization.

    Attributes:
        inputs (list[NucleotideSequenceLogits]): One relaxed ``(L, 4)`` DNA state per
            candidate in ``A,C,G,T`` order. Every item must share one length, because the
            backend evaluates them in one vectorized pass.
        temperature (float | None): Softmax temperature shared by every item. The default ``1.0``
            reads each state as unnormalized and applies ``softmax(logits / T)``, returning
            logit-space gradients; ``None`` reads them as probabilities and returns
            probability-space gradients.
    """


class MalinoisGradientConfig(GradientConfig):
    """Configuration for differentiable Malinois activity objectives.

    Attributes:
        gradient_terms (list[MalinoisActivityTerm]): Cell types whose raw activity to
            differentiate, by name or as instances.
        seq_length (int): Expected insert length before Malinois flank padding.
        artifact_path (str): Optional local artifact tarball path.
        artifact_url (str): URL used to provision the Malinois artifact.
        artifact_md5 (str): Expected checksum for the downloaded artifact.
        malinois_dir (str): Optional extracted Malinois artifact directory.
        soft (float): Inherited blend from the raw input matrix (0) to softmax probabilities
            (1). Recommended to leave at 1.0 in almost all use cases; exposed for research
            and ColabDesign parity. See
            :class:`~proto_tools.utils.gradient.models.RelaxationConfig`.
        hard (float): Inherited straight-through coefficient; 1 forwards a hard one-hot.
        compute_gradient (bool): Run backward pass and return gradient.
        device (str): Device used for inference and backpropagation.
    """

    TERMS: ClassVar[TermCatalog] = {"activity": MalinoisActivityTerm}

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run Malinois inference on",
        include_in_key=False,
    )
    gradient_terms: list[MalinoisActivityTerm] = ConfigField(
        default_factory=lambda: [MalinoisActivityTerm()],
        min_length=1,
        title="Gradient Terms",
        description="Malinois cell types to differentiate, by name or as instances.",
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
    )
    artifact_url: str = ConfigField(
        title="Artifact URL",
        default=DEFAULT_MALINOIS_ARTIFACT_URL,
        description="HTTPS URL used to provision the Malinois artifact.",
        reload_on_change=True,
    )
    artifact_md5: str = ConfigField(
        title="Artifact MD5",
        default=DEFAULT_MALINOIS_ARTIFACT_MD5,
        description="Expected MD5 checksum for the downloaded Malinois artifact.",
        reload_on_change=True,
    )
    malinois_dir: str = ConfigField(
        title="Malinois Directory",
        default=DEFAULT_MALINOIS_DIR,
        description="Optional local Malinois metadata directory; empty uses the managed cache extraction.",
        reload_on_change=True,
    )
    compute_gradient: bool = ConfigField(
        title="Compute Gradient",
        default=True,
        description="Run backward pass and return gradient; set False for forward-only scoring.",
    )

    @field_validator("gradient_terms", mode="before")
    @classmethod
    def resolve_gradient_terms(cls, value: Any) -> Any:
        """Accept a term by name or class, leaving an already-built instance as it is."""
        if not isinstance(value, list):
            return value
        catalog = cls.supported_terms()
        return [
            resolve_terms([entry], catalog)[0]  # type: ignore[arg-type]
            if isinstance(entry, str | type)
            else entry
            for entry in value
        ]

    @model_validator(mode="after")
    def reject_repeated_terms(self) -> "MalinoisGradientConfig":
        """Each cell yields one quantity, so naming it twice is a mistake rather than a doubling."""
        cells = [term.cell_type for term in self.gradient_terms]
        repeated = sorted({cell for cell in cells if cells.count(cell) > 1})
        if repeated:
            raise ValueError(f"gradient_terms names cells {repeated} more than once; each cell yields one term")
        return self

    def remote_unsupported_reason(self, device: RemoteDevice) -> str | None:
        """Local artifact/metadata overrides aren't present on a hosted worker."""
        if self.artifact_path or self.malinois_dir:
            return f"artifact_path/malinois_dir point to local files not available on device='{device}'. Leave them empty (the managed cache is used), or run locally with device='cpu'."
        return None


class MalinoisGradientOutput(GradientOutput):
    """Differentiable Malinois activity output.

    Attributes:
        results (list[MalinoisGradientResult]): One per input sequence, in input order, each
            holding that sample's requested cell-type activities and their raw gradients
            (``None`` in forward-only mode).
        vocab (list[str]): DNA column ordering for logits and gradient.
    """

    results: list[MalinoisGradientResult] = Field(  # type: ignore[assignment]  # narrows the element type
        title="Results",
        description="One result per input sequence, in input order, each carrying its cell activities.",
    )

    @classmethod
    def from_batch(
        cls,
        payload: Mapping[str, Any],
        *,
        matrices: Sequence[Sequence[Sequence[float]]],
        temperature: float | None,
        soft: float = 1.0,
        hard: float = 0.0,
    ) -> "MalinoisGradientOutput":
        """Unbatch one worker payload into per-sample results.

        The worker reports one entry per cell, each holding that cell's raw activity and
        gradient for every sample, so this transposes cells x samples into one result per
        sequence.

        Args:
            payload (Mapping[str, Any]): Worker payload for the whole batch.
            matrices (Sequence[Sequence[Sequence[float]]]): The relaxed state each sample was
                differentiated at, in input order. Every cell's gradient for one sample is a
                derivative with respect to the same state, which is what lets them be combined.
            temperature (float | None): The input's temperature, which fixes the coordinate space.
            soft (float): Relaxation blend in force.
            hard (float): Straight-through coefficient in force.

        Returns:
            MalinoisGradientOutput: One result per sample, in input order.
        """
        entries = list(payload["terms"])
        batch_size = int(payload["metrics"]["batch_size"])
        results = [
            MalinoisGradientResult(
                terms=[
                    MalinoisTermGradient(
                        term=MalinoisActivityTerm(cell_type=entry["cell_type"]),
                        term_value=float(entry["values"][index]),
                        gradient=entry["gradients"][index] if entry["gradients"] else None,
                        metadata=GradientMetadata.from_input(
                            matrices[index],
                            vocab=payload["vocab"],
                            temperature=temperature,
                            soft=soft,
                            hard=hard,
                        ),
                    )
                    for entry in entries
                ]
            )
            for index in range(batch_size)
        ]
        return cls(results=results, vocab=payload["vocab"])

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Export the gradient bundle as JSON."""
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        base = Path(export_path)
        json_path = base.parent / f"{base.name}.json"
        payload = self.model_dump(include={"results", "vocab"})
        json_path.write_text(json.dumps(payload, indent=2))


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return MalinoisScoreInput(sequences=["A" * 200])


def example_gradient_input() -> Any:
    """Minimal valid input for differentiable Malinois examples."""
    return MalinoisGradientInput(inputs=[NucleotideSequenceLogits([[0.0] * len(DNA_NUCLEOTIDES)] * 200)])


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
    pin_visible_devices=True,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    max_chunk_size=32,
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
    description="Differentiate Malinois MPRA cell-type activity for relaxed DNA logits",
    uses_gpu=True,
    pin_visible_devices=True,
    example_input=example_gradient_input,
    cacheable=False,
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    max_chunk_size=32,
)
def run_malinois_gradient(
    inputs: MalinoisGradientInput,
    config: MalinoisGradientConfig,
    instance: Any = None,
) -> MalinoisGradientOutput:
    """Report each requested Malinois cell activity and its gradient w.r.t. the relaxed logits."""
    invalid_lengths = [len(matrix) for matrix in inputs.inputs if len(matrix) != config.seq_length]
    if invalid_lengths:
        raise ValueError(f"Malinois gradient logits must have length {config.seq_length}; got {invalid_lengths[0]}")

    result = ToolInstance.dispatch(
        "malinois",
        {
            "operation": "compute_gradient",
            "logits": [item.root for item in inputs.inputs],
            "temperature": inputs.temperature,
            "gradient_terms": [term.model_dump() for term in config.gradient_terms],
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
    return MalinoisGradientOutput.from_batch(
        result,
        matrices=[item.root for item in inputs.inputs],
        temperature=inputs.temperature,
        soft=config.soft,
        hard=config.hard,
    )
