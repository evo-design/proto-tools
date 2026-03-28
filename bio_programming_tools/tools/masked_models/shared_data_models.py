"""bio_programming_tools/tools/masked_models/shared_data_models.py

Contains base schemas for embeddings, scoring, and sampling operations
shared across all masked/bidirectional protein language models."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from bio_programming_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
)


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

    sequences: List[str] = InputField(
        description="Protein sequence(s) to process as string or list of strings. (will be normalized to List[str])",
        examples=[
            "MVLSP",
            ["MVLSP", "GGGS"],
        ],
    )

    @field_validator('sequences', mode='before')
    @classmethod
    def normalize_sequences(cls, value) -> List[str]:
        """Normalize sequences to a list.

        Accepts single string or list of strings.
        Normalizes to List[str] and validates non-empty sequences.
        """
        if isinstance(value, str):
            seqs = [value]
        else:
            seqs = value

        for seq in seqs:
            if seq is None:
                raise ValueError("Sequence cannot be None")

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


class SequenceEmbedding(BaseModel):
    """Per-sequence embedding data bundle.

    Bundles all per-sequence outputs (embedding, attention mask, logits) into a
    single object so the caching/dedup system in tool_registry can correctly
    expand all parallel fields together via ``iterable_output_field="results"``.

    Follows the same pattern as ``SequenceScores`` for scoring tools.

    Attributes:
        mean_embedding (list[float]): Mean-pooled embedding vector for one sequence.
        attention_mask (list[int]): Binary mask indicating valid positions (1) vs padding (0).
        logits (list[list[float]] | None): Optional per-position amino acid logits for one sequence.
    """

    mean_embedding: List[float] = Field(
        description="Mean-pooled embedding vector (averaged over sequence length)",
    )
    attention_mask: List[int] = Field(
        description="Binary mask: 1 = valid position, 0 = padding",
    )
    logits: Optional[List[List[float]]] = Field(
        default=None,
        description="Per-position amino acid logits (seq_len, vocab_size)",
    )


class MaskedModelOutput(BaseToolOutput):
    """Base output for masked language model embedding tools.

    Contains per-sequence embedding results bundled as ``SequenceEmbedding``
    objects.

    Attributes:
        results (list[SequenceEmbedding]): Per-sequence embedding results, each containing
            a mean-pooled embedding vector, attention mask, and optional logits.
    """

    results: List[SequenceEmbedding] = Field(
        description="Per-sequence embedding results",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    @property
    def output_format_options(self) -> List[str]:
        return ["csv", "json", "pt", "npy"]

    @property
    def output_format_default(self) -> str:
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str):
        if not self.results:
            import warnings
            warnings.warn(
                "No embeddings to export. The model output contains no embedding data.",
                UserWarning,
                stacklevel=2
            )
            return

        import numpy as np
        embeddings = [r.mean_embedding for r in self.results]
        data = np.array(embeddings)
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "csv":
            np.savetxt(path, data, delimiter=",")
        elif file_format == "json":
            import json
            with open(path, "w") as f:
                json.dump(embeddings, f)
        elif file_format == "npy":
            np.save(path, data)
        elif file_format == "pt":
            try:
                import torch
                torch.save(torch.tensor(embeddings), path)
            except ImportError:
                raise ImportError("PyTorch ('torch') is required for .pt export. Please install it.")
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

    sequences: List[str] = InputField(
        description="Protein sequences to score",
        examples=["MVLSPADKTNVKAAW", ["MVLSP", "GGGS"]],
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, v):
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


class SequenceScores(BaseModel):
    """Individual sequence score with flexible metrics dict.

    Represents scoring metrics for a single sequence. Metrics can be accessed
    via dict-style (score.metrics["perplexity"]) or attribute-style (score.perplexity).

    Attributes:
        metrics (dict[str, float]): Dictionary of scalar scoring metrics.
        logits (list[list[float]] | None): Optional per-position logits array.
        vocab (list[str] | None): Optional token ordering for logits; logits[:, j] corresponds to vocab[j].
    """

    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Dictionary of scalar scoring metrics",
    )
    logits: Optional[List[List[float]]] = Field(
        default=None,
        description="Per-position logits array as nested list (seq_len, vocab_size)",
    )
    vocab: Optional[List[str]] = Field(
        default=None,
        description="Token ordering for logits: logits[:, j] corresponds to vocab[j]",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to metrics."""
        metrics = object.__getattribute__(self, "metrics")
        if name in metrics:
            return metrics[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def add_metric(self, name: str, value: float) -> None:
        """Add a metric to the output."""
        self.metrics[name] = value

    def __iter__(self) -> Iterator[float]:
        return iter(self.metrics.values())

    def __len__(self) -> int:
        return len(self.metrics)

    def __getitem__(self, index: int) -> float:
        return list(self.metrics.values())[index]


class MaskedModelScoringOutput(BaseToolOutput):
    """Standardized output for masked model sequence scoring tools.

    Attributes:
        scores (list[SequenceScores]): List of scoring outputs, one per input
            sequence. Each entry contains metrics (log_likelihood,
            avg_log_likelihood, perplexity) and optional per-position logits.
    """

    scores: List[SequenceScores] = Field(
        description="List of scoring outputs, one per input sequence"
    )

    @property
    def vocab(self) -> Optional[List[str]]:
        """Token ordering for logits; derived from first score."""
        return self.scores[0].vocab if self.scores else None

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, index: int) -> SequenceScores:
        return self.scores[index]

    def __iter__(self) -> Iterator[SequenceScores]:
        return iter(self.scores)

    @property
    def output_format_options(self) -> List[str]:
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str):
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "json":
            import json

            def default(obj):
                if hasattr(obj, "tolist"):
                    return obj.tolist()
                return str(obj)

            data = []
            for s in self.scores:
                score_data = dict(s.metrics)
                if s.logits is not None:
                    score_data["logits"] = s.logits
                if s.vocab is not None:
                    score_data["vocab"] = s.vocab
                data.append(score_data)

            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=default)

        elif file_format == "csv":
            import csv

            if self.scores:
                fieldnames = list(self.scores[0].metrics.keys())
                with open(path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for s in self.scores:
                        writer.writerow(s.metrics)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
