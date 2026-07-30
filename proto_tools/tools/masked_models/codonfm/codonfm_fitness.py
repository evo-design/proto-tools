"""CodonFM (Encodon) coding-sequence fitness scoring."""

import csv
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from proto_tools.tools.masked_models.codonfm.shared_data_models import (
    CodonFMConfig,
    CodonSequenceInput,
    resolve_checkpoint_source,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseToolOutput, ToolInstance

# Input / Config are the shared codon schemas.
CodonFMFitnessInput = CodonSequenceInput
CodonFMFitnessConfig = CodonFMConfig


class CodonFMFitnessResult(BaseModel):
    """Per-sequence CodonFM fitness result.

    Attributes:
        sequence (str): Coding sequence that was scored (DNA alphabet).
        sequence_length (int): Length of the scored sequence in nucleotides.
        fitness (float): Mean per-token log-likelihood under Encodon; higher is more
            model-typical (a proxy for a well-formed, natural coding sequence).
    """

    sequence: str = Field(title="Sequence", description="Coding sequence scored by CodonFM")
    sequence_length: int = Field(title="Sequence Length", description="Length of the scored CDS in nucleotides")
    fitness: float = Field(title="Fitness", description="Mean per-token log-likelihood; higher is more model-typical")


class CodonFMFitnessOutput(BaseToolOutput):
    """Output from CodonFM fitness scoring.

    Attributes:
        results (list[CodonFMFitnessResult]): Per-sequence fitness predictions.
    """

    results: list[CodonFMFitnessResult] = Field(
        default_factory=list, title="Results", description="Per-sequence CodonFM fitness scoring results"
    )

    def __len__(self) -> int:
        """Return the number of per-sequence results."""
        return len(self.results)

    def __getitem__(self, index: int) -> CodonFMFitnessResult:
        """Return a per-sequence result."""
        return self.results[index]

    def __iter__(self) -> Iterator[CodonFMFitnessResult]:  # type: ignore[override]
        """Iterate over per-sequence results."""
        return iter(self.results)

    @property
    def output_format_options(self) -> list[str]:
        """Supported export formats."""
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        """Default export format."""
        return "json"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Export per-sequence fitness scores as JSON or CSV."""
        rows = [r.model_dump() for r in self.results]
        base = Path(export_path)
        if file_format == "json":
            (base.parent / f"{base.name}.json").write_text(json.dumps(rows, indent=2))
            return
        if file_format == "csv":
            with open(base.parent / f"{base.name}.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sequence", "sequence_length", "fitness"])
                writer.writeheader()
                writer.writerows(rows)
            return
        raise ValueError(f"Unsupported format: {file_format}")


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return CodonFMFitnessInput(sequences=["ATGGTGAGCAAGGGCGAGGAGCTGTTCACC"])


@tool(
    key="codonfm-fitness",
    label="CodonFM Fitness",
    category="masked_models",
    input_class=CodonFMFitnessInput,
    config_class=CodonFMFitnessConfig,
    output_class=CodonFMFitnessOutput,
    description="Score coding-sequence fitness with the upstream CodonFM/Encodon visible-token objective",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    cacheable=True,
)
def run_codonfm_fitness(
    inputs: CodonFMFitnessInput,
    config: CodonFMFitnessConfig,
    instance: Any = None,
) -> CodonFMFitnessOutput:
    """Score coding-sequence fitness with CodonFM (Encodon).

    Args:
        inputs (CodonFMFitnessInput): Coding sequences to score.
        config (CodonFMFitnessConfig): CodonFM runtime and checkpoint configuration.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        CodonFMFitnessOutput: Per-sequence mean-log-likelihood fitness scores.
    """
    safetensors_url, config_url, filename, subdir = resolve_checkpoint_source(config.model_checkpoint)

    output_data = ToolInstance.dispatch(
        "codonfm",
        {
            "operation": "fitness",
            "sequences": inputs.sequences,
            "safetensors_url": safetensors_url,
            "config_url": config_url,
            "safetensors_filename": filename,
            "cache_subdir": subdir,
            "batch_size": config.batch_size,
            "device": config.device,
            "verbose": config.verbose,
            "seed": config.seed,
        },
        instance=instance,
        config=config,
    )

    fitness = output_data["fitness"]
    if len(fitness) != len(inputs.sequences):
        raise ValueError(f"Expected {len(inputs.sequences)} CodonFM fitness scores, got {len(fitness)}")
    scores = [float(score) for score in fitness]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("CodonFM returned a non-finite fitness score")

    return CodonFMFitnessOutput(
        metadata={"model_checkpoint": config.model_checkpoint, "num_sequences": len(inputs.sequences)},
        results=[
            CodonFMFitnessResult(sequence=sequence, sequence_length=len(sequence), fitness=score)
            for sequence, score in zip(inputs.sequences, scores, strict=True)
        ]
    )
