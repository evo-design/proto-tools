"""Base schemas for scoring and sampling operations shared across causal language models."""

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, field_validator

from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
)
from proto_tools.utils.tool_io import Metrics, MetricSpec


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
        title="Sequences",
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
        return_logits (bool): Whether to include per-position logits in the
            output. When ``True``, returns logits for each sequence. When
            ``False``, only returns metrics (saves memory and serialization
            time).
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


class CausalModelScoringMetrics(Metrics):
    """Per-sequence scoring metrics for causal-LM scorers (ProGen2/3, Evo1/2).

    All four causal-model scorers emit the same three scalar metrics. They may
    also emit per-position metrics (e.g. ``forward_log_likelihood_pp``,
    ``reverse_log_likelihood_pp``, ``log_likelihood_pp``) as list-valued extras;
    those keys vary per tool and are not declared in the shared ``metric_spec``.
    Per-position keys use the ``_pp`` suffix to avoid collisions with scalar
    metrics of the same stem (e.g. scalar ``log_likelihood`` vs per-position
    ``log_likelihood_pp``).

    Metrics documented in ``metric_spec``:
        log_likelihood (float): Sum of per-token log-likelihoods. Always present.
        avg_log_likelihood (float): Mean per-token log-likelihood. Always present.
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


class CausalModelScoringOutput(BaseToolOutput):
    """Standardized output for causal model sequence scoring tools.

    Attributes:
        scores (list[CausalModelScoringMetrics]): List of scoring outputs, one per
            input sequence. Each entry is a ``Metrics`` subclass with scalar
            metrics (``log_likelihood``, ``avg_log_likelihood``, ``perplexity``)
            and optional per-position ``_pp``-suffixed list extras; ``logits``
            and ``vocab`` are declared fields for raw model outputs.
    """

    scores: list[CausalModelScoringMetrics] = Field(
        title="Scores",
        description="List of scoring outputs, one per input sequence",
    )

    @property
    def vocab(self) -> list[str] | None:
        """Token ordering for logits; derived from first score."""
        return self.scores[0].vocab if self.scores else None

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, index: int) -> CausalModelScoringMetrics:
        return self.scores[index]

    def __iter__(self) -> Iterator[CausalModelScoringMetrics]:  # type: ignore[override]
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
                score_data: dict[str, Any] = dict(s.items())
                if s.logits is not None:
                    score_data["logits"] = s.logits
                if s.vocab is not None:
                    score_data["vocab"] = s.vocab
                data.append(score_data)

            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=default)

        elif file_format == "csv":
            # Only include scalar metrics in CSV; per-position (_pp) lists don't fit.
            if self.scores:
                scalar_keys = [k for k in self.scores[0] if not k.endswith("_pp")]
                with open(path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=scalar_keys)
                    writer.writeheader()
                    for s in self.scores:
                        writer.writerow({k: s[k] for k in scalar_keys})
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
        title="Prompts",
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
        description="Softmax temperature for sampling; lower is more deterministic",
    )
    top_p: float = ConfigField(
        title="Top P",
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling cutoff over per-position token probabilities",
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Prompts per GPU forward pass; raise for throughput, lower if OOM",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        include_in_key=False,
    )


class CausalModelSampleOutput(BaseToolOutput):
    """Base output for causal model sampling/generation tools.

    Attributes:
        sequences (list[str]): Generated sequences.
    """

    sequences: list[str] = Field(
        title="Sequences",
        description="Generated sequences",
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
        path = Path(export_path)

        if file_format == "fasta":
            fasta_path = path.with_suffix(".fasta")
            with open(fasta_path, "w") as f:
                f.writelines(f">seq_{i}\n{seq}\n" for i, seq in enumerate(self.sequences))
        elif file_format == "txt":
            txt_path = path.with_suffix(".txt")
            with open(txt_path, "w") as f:
                f.writelines(f"{seq}\n" for seq in self.sequences)
        elif file_format == "json":
            json_path = path.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump({"sequences": self.sequences}, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
