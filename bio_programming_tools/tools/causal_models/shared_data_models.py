"""Shared data models for causal/autoregressive language model tools (Evo2, ProGen2).

Contains base schemas for scoring and sampling operations
shared across all causal/autoregressive language models.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from bio_programming_tools.utils.tool_io import BaseToolInput, BaseToolOutput
from bio_programming_tools.utils import BaseConfig, ConfigField


# ============================================================================
# Scoring Schemas
# ============================================================================
class CausalModelScoringInput(BaseToolInput):
    """Input for causal model sequence scoring tools.

    Attributes:
        sequences: Sequences to score. Can be provided as a single string
            or a list of strings.
    """

    sequences: List[str] = Field(
        description="Sequences to score",
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


class CausalModelScoringConfig(BaseConfig):
    """Base configuration for causal model sequence scoring.

    Attributes:
        batch_size: Number of sequences to process in each batch.
        device: Device to run the model on.
        verbose: Whether to print status messages.
    """

    batch_size: int = ConfigField(
        title="Batch Size",
        default=128,
        ge=1,
        description="Number of sequences to process in each batch",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        hidden=True,
    )
    verbose: bool = ConfigField(
        title="Verbose",
        default=False,
        description="Whether to print status messages",
        hidden=True,
    )


class SequenceScores(BaseModel):
    """Individual sequence score with flexible metrics dict.

    Represents scoring metrics for a single sequence. Metrics can be accessed
    via dict-style (score.metrics["perplexity"]) or attribute-style (score.perplexity).

    Attributes:
        metrics: Dictionary of scalar scoring metrics.
        logits: Optional per-position logits array.
        vocab: Optional token ordering for logits; logits[:, j] corresponds to vocab[j].
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


class CausalModelScoringOutput(BaseToolOutput):
    """Standardized output for causal model sequence scoring tools.

    Attributes:
        scores (List[SequenceScores]): List of scoring outputs, one per input
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


# ============================================================================
# Sampling Schemas
# ============================================================================
class CausalModelSampleInput(BaseToolInput):
    """Input for causal model sampling/generation tools.

    Attributes:
        sequences: Prompt sequences to condition generation on.
            Can be provided as a single string or a list of strings.
    """

    sequences: List[str] = Field(
        description="Prompt sequence(s) to condition generation on",
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


class CausalModelSampleConfig(BaseConfig):
    """Base configuration for causal model sampling/generation.

    Attributes:
        device: Device to run the model on.
        verbose: Whether to print status messages.
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        hidden=True,
    )
    verbose: bool = ConfigField(
        title="Verbose",
        default=False,
        description="Whether to print status messages",
        hidden=True,
    )


class CausalModelSampleOutput(BaseToolOutput):
    """Base output for causal model sampling/generation tools."""

    sequences: List[str] = Field(
        description="Generated sequences",
    )

    @property
    def output_format_options(self) -> List[str]:
        return ["fasta", "json"]

    @property
    def output_format_default(self) -> str:
        return "fasta"

    def _export_output(self, export_path: str | Path, file_format: str):
        path = Path(export_path)

        if file_format == "fasta":
            fasta_path = path.with_suffix(".fasta")
            with open(fasta_path, "w") as f:
                for i, seq in enumerate(self.sequences):
                    f.write(f">seq_{i}\n{seq}\n")
        elif file_format == "json":
            import json
            json_path = path.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump({"sequences": self.sequences}, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
