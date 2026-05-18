"""proto_tools/tools/gene_annotation/promoter_calculator/promoter_calculator.py.

Wrapper around the Salis Lab Promoter Calculator (Barrick Lab fork) for
predicting E. coli sigma70 promoter strength on DNA sequences.
"""

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
class PromoterPrediction(BaseModel):
    """A single predicted sigma70 promoter on one strand of an input sequence."""

    tss_name: str = Field(description="Promoter label, e.g. 'Fwd123' or 'Rev456'")
    tss: int = Field(description="Transcription start site position")
    strand: Literal["+", "-"] = Field(description="Strand, '+' or '-'")
    dG_total: float = Field(description="Predicted binding free energy (kcal/mol)")
    Tx_rate: float = Field(description="Predicted transcription initiation rate (au)")
    promoter_sequence: str = Field(description="DNA spanning the predicted promoter")
    length: int = Field(description="Length of the promoter sequence")
    UP_position: list[int] = Field(description="UP element [start, end]")
    hex35_position: list[int] = Field(description="-35 hexamer [start, end]")
    spacer_position: list[int] = Field(description="Spacer [start, end]")
    hex10_position: list[int] = Field(description="-10 hexamer [start, end]")
    disc_position: list[int] = Field(description="Discriminator [start, end]")


class PromoterCalculatorSequenceResult(BaseModel):
    """Promoter Calculator predictions for a single input sequence."""

    sequence_id: str = Field(description="ID of the input sequence")
    predictions: list[PromoterPrediction] = Field(
        default_factory=list,
        description="Predicted promoters across both strands",
    )

    @property
    def num_promoters(self) -> int:
        """Number of predicted promoters."""
        return len(self.predictions)

    @property
    def has_promoter(self) -> bool:
        """Whether any promoter was predicted."""
        return bool(self.predictions)


# Input:
class PromoterCalculatorInput(BaseToolInput):
    """Input for Salis Lab Promoter Calculator sigma70 promoter prediction.

    Attributes:
        sequences (list[str]): DNA sequences to scan for sigma70 promoters.
        sequence_ids (list[str] | None): Optional sequence identifiers.
    """

    sequences: list[str] = InputField(description="DNA sequences to scan for E. coli sigma70 (housekeeping) promoters")
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
class PromoterCalculatorOutput(BaseToolOutput):
    """Output from Salis Lab Promoter Calculator sigma70 promoter prediction.

    Attributes:
        results (list[PromoterCalculatorSequenceResult]): Per-sequence predictions.
    """

    results: list[PromoterCalculatorSequenceResult] = Field(
        default_factory=list,
        description="Per-sequence promoter predictions",
    )

    @property
    def num_sequences_with_promoter(self) -> int:
        """Number of sequences with at least one predicted promoter."""
        return sum(1 for r in self.results if r.has_promoter)

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        import pandas as pd

        path = Path(export_path).with_suffix(f".{file_format}")
        rows = [
            {
                "sequence_id": result.sequence_id,
                "tss_name": pred.tss_name,
                "tss": pred.tss,
                "strand": pred.strand,
                "dG_total": pred.dG_total,
                "Tx_rate": pred.Tx_rate,
                "promoter_sequence": pred.promoter_sequence,
                "length": pred.length,
                "UP_start": pred.UP_position[0],
                "UP_end": pred.UP_position[1],
                "hex35_start": pred.hex35_position[0],
                "hex35_end": pred.hex35_position[1],
                "spacer_start": pred.spacer_position[0],
                "spacer_end": pred.spacer_position[1],
                "hex10_start": pred.hex10_position[0],
                "hex10_end": pred.hex10_position[1],
                "disc_start": pred.disc_position[0],
                "disc_end": pred.disc_position[1],
            }
            for result in self.results
            for pred in result.predictions
        ]
        df = pd.DataFrame(rows)
        if file_format == "csv":
            df.to_csv(path, index=False)
        elif file_format == "json":
            df.to_json(path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# Config:
class PromoterCalculatorConfig(BaseConfig):
    """Configuration for Salis Lab Promoter Calculator.

    Attributes:
        threads (int): CPU threads for the calculator's internal parallelism.
            Default 1.
        circular (bool): Treat sequences as circular (examines the wraparound
            junction). Default False.
    """

    threads: int = ConfigField(
        title="Number of Threads",
        default=1,
        ge=1,
        description="CPU threads for promoter calculator parallelism; raise on multi-core hosts",
        include_in_key=False,
    )
    circular: bool = ConfigField(
        title="Circular Sequences",
        default=False,
        description="Examine the wraparound junction (use for plasmids/circular genomes)",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input: lacUV5 promoter padded with 20 nt of A on each side."""
    # 20-nt flanks are the empirical minimum for the calculator to actually fire on
    # this promoter; below that it returns no predictions (boundary effect).
    return PromoterCalculatorInput(sequences=["A" * 20 + "AAAATTGTGAGCGGATAACAATTTCACACAGGAAACAGCTATGACC" + "A" * 20])


@tool(
    key="promoter-calculator",
    label="Salis Lab Promoter Calculator",
    category="gene_annotation",
    input_class=PromoterCalculatorInput,
    config_class=PromoterCalculatorConfig,
    output_class=PromoterCalculatorOutput,
    description="Predict E. coli sigma70 promoter strength (dG and Tx rate) on DNA sequences",
    example_input=example_input,
    iterable_input_field="sequences",
    iterable_output_field="results",
    cacheable=True,
)
def run_promoter_calculator(
    inputs: PromoterCalculatorInput,
    config: PromoterCalculatorConfig,
    instance: Any = None,
) -> PromoterCalculatorOutput:
    """Predict E. coli sigma70 promoter strength on DNA sequences.

    Wraps the Salis Lab Promoter Calculator: a 346-parameter biophysical + ML
    model that scans both strands for canonical sigma70 elements (-35, spacer,
    -10, UP, discriminator) and predicts dG_total and Tx_rate per candidate.

    Args:
        inputs (PromoterCalculatorInput): DNA sequences to scan.
        config (PromoterCalculatorConfig): Calculator runtime parameters.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        PromoterCalculatorOutput: Per-sequence promoter predictions.

    Examples:
        >>> # lacUV5 promoter padded with 20 nt of A on each side
        >>> seq = "A" * 20 + "AAAATTGTGAGCGGATAACAATTTCACACAGGAAACAGCTATGACC" + "A" * 20
        >>> result = run_promoter_calculator(
        ...     PromoterCalculatorInput(sequences=[seq], sequence_ids=["lacUV5"]),
        ...     PromoterCalculatorConfig(),
        ... )
        >>> print(f"{result.num_sequences_with_promoter} sequences had promoters")
    """
    sequence_ids = resolve_sequence_ids(inputs.sequences, inputs.sequence_ids)

    input_data = {
        "sequences": inputs.sequences,
        "sequence_ids": sequence_ids,
        "config": {
            "threads": config.threads,
            "circular": config.circular,
        },
        "device": "cpu",
    }

    output_data = ToolInstance.dispatch(
        "promoter_calculator",
        input_data,
        instance=instance,
        config=config,
    )

    results = [
        PromoterCalculatorSequenceResult(
            sequence_id=result_dict["sequence_id"],
            predictions=[PromoterPrediction(**pred) for pred in result_dict["predictions"]],
        )
        for result_dict in output_data["results"]
    ]

    return PromoterCalculatorOutput(
        metadata={
            "threads": config.threads,
            "circular": config.circular,
            "num_sequences": len(inputs.sequences),
        },
        results=results,
    )
