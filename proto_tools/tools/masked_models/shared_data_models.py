"""proto_tools/tools/masked_models/shared_data_models.py.

Contains base schemas for embeddings, scoring, and sampling operations
shared across all masked/bidirectional protein language models.
"""

import csv
import json
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field, field_validator

from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
)
from proto_tools.utils.tool_io import Metrics, MetricSpec


# ============================================================================
# Embedding Schemas
# ============================================================================
class MaskedModelInput(BaseToolInput):
    """Base input for masked language model tools.

    Provides common input validation and normalization for protein
    sequences used across all masked protein language model tools (ESM2, ESM3).

    Attributes:
        sequences (list[str]): Protein sequence(s) to process. Can be
            provided as:

            - A single protein sequence string (e.g., ``"MVLSPADKTN"``)
            - A list of protein sequence strings (e.g., ``["MVLSP", "GGGS"]``)

            After validation, sequences are automatically normalized to a list format.
            Valid protein sequences contain only standard amino acid characters
            (20 standard amino acids + X (any amino acid) + * (stop codon)).
    """

    sequences: list[str] = InputField(
        description="Protein sequence(s) to process as string or list of strings. (will be normalized to List[str])",
        examples=[
            "MVLSP",
            ["MVLSP", "GGGS"],
        ],
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value: Any) -> list[str]:
        """Normalize sequences to a list.

        Accepts single string or list of strings.
        Normalizes to List[str] and validates non-empty sequences.
        """
        seqs = [value] if isinstance(value, str) else value

        for idx, seq in enumerate(seqs):
            if seq is None:
                raise ValueError(f"sequences[{idx}]: cannot be None")

        return seqs

    def __len__(self) -> int:
        """Get the total number of residues across all sequences."""
        return sum(len(seq) for seq in self.sequences)


class MaskedModelConfig(BaseConfig):
    """Base configuration for masked language model tools.

    Provides common configuration parameters shared across all masked protein
    language model tools, including batch processing, device management, and
    execution settings.

    Attributes:
        batch_size (int): Number of sequences to process simultaneously on GPU.
            Larger batches improve throughput but use more GPU memory.
        device (str): Device to run the model on (e.g., ``"cuda"``, ``"cpu"``).
    """

    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of sequences to process simultaneously on GPU",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on (e.g., 'cuda', 'cpu')",
        hidden=True,
        include_in_key=False,
    )


class Projection2D(BaseModel):
    """A single 2D point produced by dimensionality reduction (UMAP).

    Attributes:
        x (float): First reduced coordinate.
        y (float): Second reduced coordinate.
    """

    x: float = Field(description="First reduced coordinate")
    y: float = Field(description="Second reduced coordinate")


class SequenceEmbedding(BaseModel):
    """Per-sequence embedding data bundle.

    Bundles all per-sequence outputs (embedding, attention mask, logits, projection)
    into a single object so the caching/dedup system in tool_registry can correctly
    expand all parallel fields together via ``iterable_output_field="results"``.

    Follows the same pattern as ``MaskedModelScoringMetrics`` for scoring tools.

    Attributes:
        mean_embedding (list[float]): Mean-pooled embedding vector for one sequence.
        attention_mask (list[int]): Binary mask indicating valid positions (1) vs padding (0).
        logits (list[list[float]] | None): Optional per-position amino acid logits for one sequence.
        projection (Projection2D | None): Optional 2D coordinate from a UMAP projection
            of all embeddings in the same call. Populated when ``n_sequences >= 4``;
            ``None`` otherwise (single-point or 2-3-point UMAP is meaningless).
    """

    mean_embedding: list[float] = Field(
        description="Mean-pooled embedding vector (averaged over sequence length)",
    )
    attention_mask: list[int] = Field(
        description="Binary mask: 1 = valid position, 0 = padding",
    )
    logits: list[list[float]] | None = Field(
        default=None,
        description="Per-position amino acid logits (seq_len, vocab_size)",
    )
    projection: Projection2D | None = Field(
        default=None,
        description="2D UMAP projection of this sequence's embedding within the call's batch",
    )


class MaskedModelOutput(BaseToolOutput):
    """Base output for masked language model embedding tools.

    Contains per-sequence embedding results bundled as ``SequenceEmbedding``
    objects.

    Attributes:
        results (list[SequenceEmbedding]): Per-sequence embedding results, each containing
            a mean-pooled embedding vector, attention mask, and optional logits.
    """

    results: list[SequenceEmbedding] = Field(
        description="Per-sequence embedding results",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["csv", "json", "pt", "npy"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        if not self.results:
            warnings.warn(
                "No embeddings to export. The model output contains no embedding data.", UserWarning, stacklevel=2
            )
            return

        import numpy as np

        embeddings = [r.mean_embedding for r in self.results]
        data = np.array(embeddings)
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "csv":
            np.savetxt(path, data, delimiter=",")
        elif file_format == "json":
            with open(path, "w") as f:
                json.dump(embeddings, f)
        elif file_format == "npy":
            np.save(path, data)
        elif file_format == "pt":
            try:
                import torch

                torch.save(torch.tensor(embeddings), path)
            except ImportError:
                raise ImportError("PyTorch ('torch') is required for .pt export. Please install it.") from None
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Scoring Schemas
# ============================================================================
class MaskedModelScoringInput(BaseToolInput):
    """Input for masked model sequence scoring tools.

    Attributes:
        sequences (list[str]): Protein sequences to score. Can be provided as a single
            string or a list of strings.
    """

    sequences: list[str] = InputField(
        description="Protein sequences to score",
        examples=["MVLSPADKTNVKAAW", ["MVLSP", "GGGS"]],
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, v: Any) -> Any:
        """Convert single string to list of strings."""
        if isinstance(v, str):
            return [v]
        if not v:
            raise ValueError("sequences must not be empty")
        return v


class MaskedModelScoringConfig(BaseConfig):
    """Base configuration for masked model sequence scoring.

    Attributes:
        batch_size (int): Number of sequences to process simultaneously on GPU.
            Larger batches improve throughput but use more GPU memory; reduce
            if encountering out-of-memory errors.
        device (str): Device to run the model on.
        verbose (bool): Whether to print status messages.
    """

    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of sequences to process simultaneously on GPU",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        hidden=True,
        include_in_key=False,
    )


class MaskedModelScoringMetrics(Metrics):
    """Per-sequence scoring metrics for masked-model LM scorers (ESM2, ESM3, AbLang).

    All three masked-model scorers emit the same scalar set. Shared here rather
    than per-tool because the metric spec is identical across them.

    Metrics documented in ``metric_spec``:
        log_likelihood (float): Sum of per-position log-likelihoods. Always present.
        avg_log_likelihood (float): Mean per-position log-likelihood. Always present.
        perplexity (float): exp(-avg_log_likelihood). Always present. Range ``[1, ∞)``.

    Attributes:
        logits (list[list[float]] | None): Per-position logits array
            ``(seq_len, vocab_size)``. ``None`` unless ``return_logits=True``.
        vocab (list[str] | None): Token ordering for ``logits``.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "log_likelihood": {"availability": "always", "type": "float", "min": None, "max": 0.0},
        "avg_log_likelihood": {"availability": "always", "type": "float", "min": None, "max": 0.0},
        "perplexity": {"availability": "always", "type": "float", "min": 1.0, "max": None},
    }
    primary_metric: str | None = "perplexity"

    logits: list[list[float]] | None = Field(
        default=None,
        description="Per-position logits array as nested list (seq_len, vocab_size)",
    )
    vocab: list[str] | None = Field(
        default=None,
        description="Token ordering for logits: logits[:, j] corresponds to vocab[j]",
    )


class MaskedModelScoringOutput(BaseToolOutput):
    """Standardized output for masked model sequence scoring tools.

    Attributes:
        scores (list[MaskedModelScoringMetrics]): List of scoring outputs,
            one per input sequence. Each entry is a ``Metrics`` subclass with
            scalar metrics (accessed via ``score.perplexity`` or
            ``score["perplexity"]``) plus declared ``logits`` / ``vocab``
            fields that carry raw model outputs when requested.
    """

    scores: list[MaskedModelScoringMetrics] = Field(
        description="List of scoring outputs, one per input sequence",
    )

    @property
    def vocab(self) -> list[str] | None:
        """Token ordering for logits; derived from first score."""
        return self.scores[0].vocab if self.scores else None

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, index: int) -> MaskedModelScoringMetrics:
        return self.scores[index]

    def __iter__(self) -> Iterator[MaskedModelScoringMetrics]:  # type: ignore[override]
        return iter(self.scores)

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "json":

            def default(obj: Any) -> Any:
                if hasattr(obj, "tolist"):
                    return obj.tolist()
                return str(obj)

            data = []
            for s in self.scores:
                score_data: dict[str, Any] = dict(s.items())
                if s.logits is not None:
                    score_data["logits"] = s.logits
                if s.vocab is not None:
                    score_data["vocab"] = s.vocab
                data.append(score_data)

            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=default)

        elif file_format == "csv":
            if self.scores:
                fieldnames = list(self.scores[0].keys())
                with open(path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for s in self.scores:
                        writer.writerow(dict(s.items()))
        else:
            raise ValueError(f"Unsupported format: {file_format}")
