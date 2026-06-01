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
        title="Sequences",
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


class MaskedModelEmbeddingsConfig(BaseConfig):
    """Base configuration for masked language model embedding tools.

    Attributes:
        batch_size (int): Number of sequences to process simultaneously on GPU.
            Larger batches improve throughput but use more GPU memory.
        device (str): Device to run the model on.
    """

    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Sequences per GPU forward pass; raise for throughput, lower if OOM",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        include_in_key=False,
    )


class Projection2D(BaseModel):
    """A single 2D point produced by dimensionality reduction (UMAP).

    Attributes:
        x (float): First reduced coordinate.
        y (float): Second reduced coordinate.
    """

    x: float = Field(title="X", description="First reduced coordinate")
    y: float = Field(title="Y", description="Second reduced coordinate")


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
        title="Mean Embedding",
        description="Mean-pooled embedding vector (averaged over sequence length)",
    )
    attention_mask: list[int] = Field(
        title="Attention Mask",
        description="Binary mask: 1 = valid position, 0 = padding",
    )
    logits: list[list[float]] | None = Field(
        default=None,
        title="Logits",
        description="Per-position amino acid logits (seq_len, vocab_size)",
    )
    projection: Projection2D | None = Field(
        default=None,
        title="UMAP Projection",
        description="2D UMAP projection of this sequence's embedding within the call's batch",
    )


class MaskedModelEmbeddingsOutput(BaseToolOutput):
    """Base output for masked language model embedding tools.

    Contains per-sequence embedding results bundled as ``SequenceEmbedding``
    objects.

    Attributes:
        results (list[SequenceEmbedding]): Per-sequence embedding results, each containing
            a mean-pooled embedding vector, attention mask, and optional logits.
    """

    results: list[SequenceEmbedding] = Field(
        title="Results",
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
# Sampling Schemas
# ============================================================================
class MaskedModelSampleConfig(BaseConfig):
    """Base configuration for masked language model sampling tools.

    Attributes:
        batch_size (int): Number of sequences to process simultaneously on GPU.
            Larger batches improve throughput but use more GPU memory.
        device (str): Device to run the model on.
    """

    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Sequences per GPU forward pass; raise for throughput, lower if OOM",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        include_in_key=False,
    )


class MaskedModelSampleOutput(BaseToolOutput):
    """Base output for masked language model sampling tools.

    Attributes:
        sequences (list[str]): Sampled or restored protein sequences.
    """

    sequences: list[str] = Field(
        title="Sequences",
        description="Sampled/restored protein sequences",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["fasta", "txt", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "fasta"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "fasta":
            with open(path, "w") as f:
                f.writelines(f">seq_{i}\n{seq}\n" for i, seq in enumerate(self.sequences))
        elif file_format == "txt":
            with open(path, "w") as f:
                f.writelines(f"{seq}\n" for seq in self.sequences)
        elif file_format == "json":
            with open(path, "w") as f:
                json.dump(self.sequences, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Scoring Schemas
# ============================================================================
class MaskedModelScoringConfig(BaseConfig):
    """Base configuration for masked model sequence scoring.

    Attributes:
        batch_size (int): Number of sequences to process simultaneously on GPU.
            Larger batches improve throughput but use more GPU memory; reduce
            if encountering out-of-memory errors.
        device (str): Device to run the model on.
        verbose (int): Verbosity level (0=quiet, 1=info, 2=debug, 3=raw subprocess stderr).
            ``True`` is coerced to ``1`` and ``False`` to ``0``.
        return_logits (bool): Include per-position logits in the output (large; disable to
            save memory).
    """

    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Sequences per GPU forward pass; raise for throughput, lower if OOM",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        include_in_key=False,
    )
    return_logits: bool = ConfigField(
        title="Return Logits",
        default=False,
        description="Include per-position logits in the output (large; disable to save memory)",
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
        "log_likelihood": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": 0.0,
            "better_values_are": "higher",
        },
        "avg_log_likelihood": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": 0.0,
            "better_values_are": "higher",
        },
        "perplexity": {
            "availability": "always",
            "type": "float",
            "min": 1.0,
            "max": None,
            "better_values_are": "lower",
        },
    }
    primary_metric: str | None = Field(
        default="perplexity",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )

    logits: list[list[float]] | None = Field(
        default=None,
        title="Logits",
        description="Per-position logits array as nested list (seq_len, vocab_size)",
    )
    vocab: list[str] | None = Field(
        default=None,
        title="Vocabulary",
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
        title="Scores",
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
