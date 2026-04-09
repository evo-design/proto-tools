"""Base schemas for scoring and sampling operations shared across causal language models."""

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
)


# ============================================================================
# Scoring Schemas
# ============================================================================
class CausalModelScoringInput(BaseToolInput):
    """Input for causal model sequence scoring tools.

    Attributes:
        sequences (list[str]): Sequences to score. Can be provided as a single string
            or a list of strings.
    """

    sequences: list[str] = InputField(
        description="Sequences to score",
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


class CausalModelScoringConfig(BaseConfig):
    """Base configuration for causal model sequence scoring.

    Attributes:
        batch_size (int): Number of sequences to process simultaneously on GPU.
            Larger batches improve throughput but use more GPU memory; reduce
            if encountering out-of-memory errors.
        device (str): Device to run the model on.
    """

    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of sequences to process simultaneously on GPU",
        advanced=True,
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
        per_position_metrics (dict[str, list[float | None]] | None): Optional per-position
            scoring metrics, keyed by metric name. Each value is a list of length
            equal to the input sequence, with ``None`` at positions where that
            metric is not available.
    """

    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Dictionary of scalar scoring metrics",
    )
    logits: list[list[float]] | None = Field(
        default=None,
        description="Per-position logits array as nested list (seq_len, vocab_size)",
    )
    vocab: list[str] | None = Field(
        default=None,
        description="Token ordering for logits: logits[:, j] corresponds to vocab[j]",
    )
    per_position_metrics: dict[str, list[float | None]] | None = Field(
        default=None,
        description="Per-position scoring metrics, keyed by metric name",
    )

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to metrics."""
        metrics = object.__getattribute__(self, "metrics")
        if name in metrics:
            return metrics[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def add_metric(self, name: str, value: float) -> None:
        """Add a metric to the output."""
        self.metrics[name] = value

    def __iter__(self) -> Iterator[float]:  # type: ignore[override]
        return iter(self.metrics.values())

    def __len__(self) -> int:
        return len(self.metrics)

    def __getitem__(self, index: int) -> float:
        return list(self.metrics.values())[index]


class CausalModelScoringOutput(BaseToolOutput):
    """Standardized output for causal model sequence scoring tools.

    Attributes:
        scores (list[SequenceScores]): List of scoring outputs, one per input
            sequence. Each entry contains metrics (log_likelihood,
            avg_log_likelihood, perplexity) and optional per-position logits.
    """

    scores: list[SequenceScores] = Field(description="List of scoring outputs, one per input sequence")

    @property
    def vocab(self) -> list[str] | None:
        """Token ordering for logits; derived from first score."""
        return self.scores[0].vocab if self.scores else None

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, index: int) -> SequenceScores:
        return self.scores[index]

    def __iter__(self) -> Iterator[SequenceScores]:  # type: ignore[override]
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

            data: list[dict[str, Any]] = []
            for s in self.scores:
                score_data: dict[str, Any] = dict(s.metrics)
                if s.logits is not None:
                    score_data["logits"] = s.logits
                if s.vocab is not None:
                    score_data["vocab"] = s.vocab
                if s.per_position_metrics is not None:
                    score_data["per_position_metrics"] = s.per_position_metrics
                data.append(score_data)

            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=default)

        elif file_format == "csv":
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
        prompts (list[str]): Prompt sequences to condition generation on.
            Can be provided as a single string or a list of strings.
    """

    prompts: list[str] = InputField(
        description="Prompt sequence(s) to condition generation on",
        examples=["MVLSPADKTNVKAAW", ["MVLSP", "GGGS"]],
    )

    @field_validator("prompts", mode="before")
    @classmethod
    def normalize_prompts(cls, v: Any) -> Any:
        """Convert single string to list of strings."""
        if isinstance(v, str):
            return [v]
        if not v:
            raise ValueError("prompts must not be empty")
        return v


class CausalModelSampleConfig(BaseConfig):
    """Base configuration for causal model sampling/generation.

    Attributes:
        prepend_prompt (bool): Whether to include the input prompt at the
            start of each generated sequence. When ``True``, output sequences
            begin with the prompt followed by newly generated tokens. When
            ``False``, only the newly generated tokens are returned.
        temperature (float): Sampling temperature controlling randomness.
            Higher values produce more diverse outputs; lower values produce
            more conservative, high-confidence sequences.
        top_p (float): Nucleus sampling threshold. Only tokens whose cumulative
            probability mass reaches this threshold are considered during
            sampling. Lower values restrict sampling to higher-probability
            tokens.
        batch_size (int): Number of prompts to process simultaneously on GPU.
            Larger batches improve throughput but use more GPU memory; reduce
            if encountering out-of-memory errors.
        device (str): Device to run the model on.
    """

    prepend_prompt: bool = ConfigField(
        title="Prepend Prompt",
        default=True,
        description="Include the input prompt at the start of each generated sequence",
    )
    temperature: float = ConfigField(
        title="Temperature",
        default=1.0,
        gt=0.0,
        description="Sampling temperature controlling randomness",
    )
    top_p: float = ConfigField(
        title="Top P",
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling threshold",
        advanced=True,
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of prompts to process simultaneously on GPU",
        advanced=True,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        hidden=True,
        include_in_key=False,
    )


class CausalModelSampleOutput(BaseToolOutput):
    """Base output for causal model sampling/generation tools."""

    sequences: list[str] = Field(
        description="Generated sequences",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["fasta", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "fasta"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path)

        if file_format == "fasta":
            fasta_path = path.with_suffix(".fasta")
            with open(fasta_path, "w") as f:
                f.writelines(f">seq_{i}\n{seq}\n" for i, seq in enumerate(self.sequences))
        elif file_format == "json":
            json_path = path.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump({"sequences": self.sequences}, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
