"""proto_tools/tools/gene_annotation/crispr_tracr/crispr_tracr.py.

This module provides a standardized interface for predicting tracrRNA sequences
from nucleotide CRISPR loci using the CRISPRtracrRNA tool from the Backofen Lab
(https://github.com/BackofenLab/CRISPRtracrRNA).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
    resolve_sequence_ids,
)


# ============================================================================
# Data Models
# ============================================================================
class TracrPrediction(BaseModel):
    """A single tracrRNA prediction for a sequence."""

    sequence_id: str = Field(description="ID of the input sequence")
    tracr_start: int | None = Field(default=None, description="Start position of predicted tracrRNA")
    tracr_end: int | None = Field(default=None, description="End position of predicted tracrRNA")
    tracr_hit: str | None = Field(default=None, description="tracrRNA hit description")
    interaction_energy: float | None = Field(
        default=None,
        description="IntaRNA interaction energy in kcal/mol, more negative = stronger (complete_run mode)",
    )
    anti_repeat_similarity_coverage_multiplication: float | None = Field(
        default=None,
        description="Anti-repeat similarity x coverage score",
    )
    intarna_anti_repeat_interaction: str | None = Field(
        default=None,
        description="IntaRNA anti-repeat interaction prediction",
    )

    @property
    def has_tracr(self) -> bool:
        """Whether a tracrRNA was predicted."""
        return self.tracr_start is not None


# Input:
class CrisprTracrInput(BaseToolInput):
    """Input for CRISPRtracrRNA prediction.

    Attributes:
        sequences (list[str]): Nucleotide sequence(s) to predict tracrRNA from.
            Each sequence should contain a CRISPR locus.
        sequence_ids (list[str] | None): Optional sequence identifiers.
    """

    sequences: list[str] = InputField(description="Nucleotide sequence(s) to predict tracrRNA from")
    sequence_ids: list[str] | None = InputField(
        default=None,
        description="Optional sequence identifiers (defaults to seq_0, seq_1, ...)",
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value: Any) -> list[str]:
        """Normalize a single sequence to a list."""
        if isinstance(value, str):
            return [value]
        return value  # type: ignore[no-any-return]


# Output:
class CrisprTracrOutput(BaseToolOutput):
    """Output from CRISPRtracrRNA prediction.

    Attributes:
        predictions (list[TracrPrediction]): Per-sequence tracrRNA predictions.
    """

    predictions: list[TracrPrediction] = Field(
        default_factory=list,
        description="Per-sequence tracrRNA predictions",
    )

    @property
    def num_with_tracr(self) -> int:
        """Number of sequences with a detected tracrRNA."""
        return sum(1 for p in self.predictions if p.has_tracr)

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        import pandas as pd

        path = Path(export_path).with_suffix(f".{file_format}")
        df = pd.DataFrame([p.model_dump() for p in self.predictions])
        if file_format == "csv":
            df.to_csv(path, index=False)
        elif file_format == "json":
            df.to_json(path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# Config:
class CrisprTracrConfig(BaseConfig):
    """Configuration for CRISPRtracrRNA prediction.

    Attributes:
        model_type (Literal['II', 'all']): Type of CRISPR model to use.
        run_type (Literal['complete_run', 'model_only']): Pipeline mode (complete_run or model_only).
        num_workers (int | None): Number of parallel workers.
    """

    model_type: Literal["II", "all"] = ConfigField(
        title="Model Type",
        default="II",
        description='CRISPR model type: "II" for type II only (faster), "all" for comprehensive',
    )
    run_type: Literal["complete_run", "model_only"] = ConfigField(
        title="Run Type",
        default="complete_run",
        description='Pipeline mode: "complete_run" for full analysis, "model_only" for fast scan',
    )
    num_workers: int | None = ConfigField(
        title="Number of Workers",
        default=None,
        description="Number of parallel workers (defaults to SLURM CPUs or 1)",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return CrisprTracrInput(sequences=["ATCGATCG"])


@tool(
    key="crispr-tracr",
    label="CRISPRtracrRNA Prediction",
    category="gene_annotation",
    input_class=CrisprTracrInput,
    config_class=CrisprTracrConfig,
    output_class=CrisprTracrOutput,
    description="Predict tracrRNA sequences from nucleotide CRISPR loci",
    example_input=example_input,
    iterable_input_field="sequences",
    iterable_output_field="predictions",
    cacheable=True,
)
def run_crispr_tracr(
    inputs: CrisprTracrInput,
    config: CrisprTracrConfig | None = None,
    instance: Any = None,
) -> CrisprTracrOutput:
    """Predict tracrRNA sequences from nucleotide CRISPR loci.

    Uses the CRISPRtracrRNA tool from the Backofen Lab to predict tracrRNA
    sequences associated with CRISPR loci. This is used as a Stage 3 filter
    in the Cas9 filtering pipeline to confirm that candidate sequences
    contain functional tracrRNA binding sites.

    Args:
        inputs (CrisprTracrInput): Validated input containing nucleotide sequences.
        config (CrisprTracrConfig | None): CRISPRtracrRNA configuration including model type.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        CrisprTracrOutput: Per-sequence tracrRNA predictions.

    Examples:
        >>> inputs = CrisprTracrInput(sequences=["ATCG..." * 1000])
        >>> config = CrisprTracrConfig(model_type="II")
        >>> result = run_crispr_tracr(inputs, config)
        >>> print(f"{result.num_with_tracr} sequences have tracrRNA predictions")
    """
    sequence_ids = resolve_sequence_ids(inputs.sequences, inputs.sequence_ids)

    num_workers = config.num_workers  # type: ignore[union-attr]
    if num_workers is None:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        num_workers = int(slurm_cpus) if slurm_cpus else 1

    input_data = {
        "sequences": inputs.sequences,
        "sequence_ids": sequence_ids,
        "config": {
            "model_type": config.model_type,  # type: ignore[union-attr]
            "run_type": config.run_type,  # type: ignore[union-attr]
            "num_workers": num_workers,
        },
    }

    input_data["device"] = "cpu"
    output_data = ToolInstance.dispatch(
        "crispr_tracr",
        input_data,
        instance=instance,
        config=config,
    )

    predictions = [TracrPrediction(**p) for p in output_data["predictions"]]

    return CrisprTracrOutput(
        metadata={
            "model_type": config.model_type,  # type: ignore[union-attr]
            "run_type": config.run_type,  # type: ignore[union-attr]
            "num_sequences": len(inputs.sequences),
        },
        predictions=predictions,
    )
