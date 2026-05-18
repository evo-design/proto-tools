"""proto_tools/tools/sequence_scoring/segmasker/segmasker.py.

Segmasker tool for detecting low-complexity regions in protein sequences.
"""

import os
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, field_validator, model_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)
from proto_tools.utils.tool_io import Metrics, MetricSpec


class SegmaskerInput(BaseToolInput):
    """Input object for Segmasker low-complexity region detection.

    This class defines the input parameters for detecting low-complexity regions
    in protein sequences using NCBI's segmasker tool.

    Attributes:
        sequences (list[str]): Protein sequence(s) to analyze for low-complexity
            regions. Can be provided as:

            - A single protein sequence string (automatically converted to list)
            - A list of protein sequence strings for batch processing

            Each sequence should contain standard amino acid characters. Empty
            sequences are handled gracefully (assigned 0.0 low-complexity fraction).
    """

    sequences: list[str] = InputField(description="Protein sequence(s) to analyze for low-complexity regions")

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value: Any) -> list[str]:
        """Normalize single string to list."""
        if isinstance(value, str):
            return [value]
        return value  # type: ignore[no-any-return]

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, sequences: list[str]) -> list[str]:
        """Validate sequences."""
        if not sequences:
            raise ValueError("At least one sequence is required")
        return sequences


class SegmaskerConfig(BaseConfig):
    """Configuration object for Segmasker low-complexity detection.

    Attributes:
        window (int): Sliding-window size for SEG complexity analysis. Larger
            windows are less sensitive to short low-complexity stretches.
        locut (float): Lower complexity cutoff. Regions scoring below this are
            classified as low-complexity.
        hicut (float): Upper complexity cutoff. Defines the transition between
            masked and unmasked regions. Must be >= ``locut``.
    """

    window: int = ConfigField(
        title="Window Size",
        default=12,
        ge=1,
        description="Sliding-window size for SEG; smaller = more sensitive to short low-complexity runs",
    )
    locut: float = ConfigField(
        title="Low-complexity Threshold",
        default=2.2,
        description="Lower SEG complexity cutoff; lower = stricter (only the most biased regions)",
    )
    hicut: float = ConfigField(
        title="High-complexity Threshold",
        default=2.5,
        description="Upper SEG complexity cutoff; must be >= locut",
    )

    @model_validator(mode="after")
    def _hicut_ge_locut(self) -> "SegmaskerConfig":
        if self.hicut < self.locut:
            raise ValueError(f"hicut ({self.hicut}) must be >= locut ({self.locut})")
        return self


class SegmaskerMetrics(Metrics):
    """Per-sequence low-complexity metrics emitted by Segmasker.

    Metrics documented in ``metric_spec``:
        low_complexity_fraction (float): Fraction of the sequence classified as
            low-complexity. Always present. Range ``[0.0, 1.0]``.
        low_complexity_count (int): Number of positions classified as
            low-complexity. Always present. Non-negative.
        sequence_length (int): Length of the input sequence in amino acids.
            Always present. Positive.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "low_complexity_fraction": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0},
        "low_complexity_count": {"availability": "always", "type": "int", "min": 0, "max": None},
        "sequence_length": {"availability": "always", "type": "int", "min": 1, "max": None},
    }
    primary_metric: str | None = "low_complexity_fraction"


class SegmaskerOutput(BaseToolOutput):
    """Output from Segmasker low-complexity region detection.

    Attributes:
        results (list[SegmaskerMetrics]): Per-sequence low-complexity metrics,
            index-aligned with ``inputs.sequences``.
    """

    results: list[SegmaskerMetrics] = Field(
        default_factory=list,
        description="Per-sequence low-complexity metrics",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "csv"

    def _export_output(
        self,
        export_path: str | os.PathLike,  # type: ignore[type-arg]
        file_format: str,
    ) -> None:
        import pandas as pd

        path = Path(export_path).with_suffix(f".{file_format}")

        df = pd.DataFrame([dict(r.items()) for r in self.results])

        if file_format == "csv":
            df.to_csv(path, index=False)

        elif file_format == "json":
            df.to_json(path, orient="records", indent=2)

        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return SegmaskerInput(sequences=["MKTL"])


@tool(
    key="segmasker-score",
    label="Segmasker Low-Complexity Detection",
    category="sequence_scoring",
    input_class=SegmaskerInput,
    config_class=SegmaskerConfig,
    output_class=SegmaskerOutput,
    metrics_class=SegmaskerMetrics,
    description="Detect low-complexity regions in protein sequences using NCBI segmasker",
    example_input=example_input,
    cacheable=True,
)
def run_segmasker(
    inputs: SegmaskerInput,
    config: SegmaskerConfig,
    instance: Any = None,
) -> SegmaskerOutput:
    """Detect low-complexity regions in protein sequences using NCBI segmasker.

    Uses NCBI's segmasker tool to identify compositionally biased and low-complexity
    regions in protein sequences. Low-complexity regions are often masked before
    homology searches to reduce false positive matches.

    Args:
        inputs (SegmaskerInput): Validated input containing one or more protein
            sequences to analyze.
        config (SegmaskerConfig): Validated segmasker configuration specifying
            window size and complexity thresholds.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        SegmaskerOutput: Structured output containing:
            - ``low_complexity_fractions``: Fraction of each sequence that is low-complexity
            - ``low_complexity_counts``: Number of low-complexity positions per sequence
            - ``sequence_lengths``: Length of each input sequence
            - ``metadata``: Execution metadata (num_sequences, window, locut, hicut)

    See Also:
        - BLAST+ suite: https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs
        - Segmasker documentation: https://www.ncbi.nlm.nih.gov/IEB/ToolBox/CPP_DOC/lxr/source/src/app/segmasker/

    Example:
        >>> inputs = SegmaskerInput(sequences=["AAAAAAAA", "MVLSPADKTN"])
        >>> config = SegmaskerConfig()
        >>> result = run_segmasker(inputs, config)
        >>> print(f"Low-complexity fractions: {result.low_complexity_fractions}")
    """
    input_data = {
        "sequences": inputs.sequences,
        "config": {
            "window": config.window,
            "locut": config.locut,
            "hicut": config.hicut,
        },
    }

    input_data["device"] = "cpu"
    output_data = ToolInstance.dispatch(
        "segmasker",
        input_data,
        instance=instance,
        config=config,
    )

    results = [
        SegmaskerMetrics(
            low_complexity_fraction=fraction,
            low_complexity_count=count,
            sequence_length=length,
        )
        for fraction, count, length in zip(
            output_data["fractions"],
            output_data["counts"],
            output_data["lengths"],
            strict=True,
        )
    ]

    return SegmaskerOutput(
        metadata={
            "num_sequences": len(inputs.sequences),
            "window": config.window,
            "locut": config.locut,
            "hicut": config.hicut,
        },
        results=results,
    )
