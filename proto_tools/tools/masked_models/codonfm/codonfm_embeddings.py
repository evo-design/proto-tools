"""CodonFM (Encodon) CLS-embedding extraction."""

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

CodonFMEmbeddingsInput = CodonSequenceInput


class CodonFMEmbeddingsConfig(CodonFMConfig):
    """Configuration for CodonFM CLS-embedding extraction.

    Attributes:
        model_checkpoint (CodonFMCheckpoint): Encodon checkpoint to run.
        batch_size (int): Sequences processed per GPU forward pass.
    """


class CodonFMEmbeddingResult(BaseModel):
    """Per-sequence CodonFM embedding result.

    Attributes:
        sequence (str): Coding sequence that was embedded (DNA alphabet).
        sequence_length (int): Length of the sequence in nucleotides.
        embedding (list[float]): The final-layer CLS-token embedding vector (hidden size
            depends on the checkpoint: 1024 for 80M, 2048 for 600M/1B).
    """

    sequence: str = Field(title="Sequence", description="Coding sequence embedded by CodonFM")
    sequence_length: int = Field(title="Sequence Length", description="Length of the embedded CDS in nucleotides")
    embedding: list[float] = Field(title="Embedding", description="Final-layer CLS embedding vector")


class CodonFMEmbeddingsOutput(BaseToolOutput):
    """Output from CodonFM embedding extraction.

    Attributes:
        results (list[CodonFMEmbeddingResult]): Per-sequence CLS embeddings.
    """

    results: list[CodonFMEmbeddingResult] = Field(
        default_factory=list, title="Results", description="Per-sequence CodonFM CLS embeddings"
    )

    def __len__(self) -> int:
        """Return the number of per-sequence results."""
        return len(self.results)

    def __getitem__(self, index: int) -> CodonFMEmbeddingResult:
        """Return a per-sequence result."""
        return self.results[index]

    def __iter__(self) -> Iterator[CodonFMEmbeddingResult]:  # type: ignore[override]
        """Iterate over per-sequence results."""
        return iter(self.results)

    @property
    def output_format_options(self) -> list[str]:
        """Supported export formats."""
        return ["json"]

    @property
    def output_format_default(self) -> str:
        """Default export format."""
        return "json"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Export per-sequence CLS embeddings as JSON."""
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        base = Path(export_path)
        (base.parent / f"{base.name}.json").write_text(json.dumps([r.model_dump() for r in self.results], indent=2))


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return CodonFMEmbeddingsInput(sequences=["ATGGTGAGCAAGGGCGAGGAGCTGTTCACC"])


@tool(
    key="codonfm-embedding",
    label="CodonFM Embeddings",
    category="masked_models",
    input_class=CodonFMEmbeddingsInput,
    config_class=CodonFMEmbeddingsConfig,
    output_class=CodonFMEmbeddingsOutput,
    description="Extract final-layer CLS embeddings for coding sequences with the CodonFM/Encodon model",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    max_chunk_size=32,
    cacheable=True,
)
def run_codonfm_embeddings(
    inputs: CodonFMEmbeddingsInput,
    config: CodonFMEmbeddingsConfig,
    instance: Any = None,
) -> CodonFMEmbeddingsOutput:
    """Extract CLS embeddings for coding sequences with CodonFM (Encodon).

    Args:
        inputs (CodonFMEmbeddingsInput): Coding sequences to embed.
        config (CodonFMEmbeddingsConfig): CodonFM runtime and checkpoint configuration.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        CodonFMEmbeddingsOutput: Per-sequence CLS embedding vectors.
    """
    safetensors_url, config_url, filename, subdir = resolve_checkpoint_source(config.model_checkpoint)

    output_data = ToolInstance.dispatch(
        "codonfm",
        {
            "operation": "embeddings",
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

    embeddings = output_data["embeddings"]
    if len(embeddings) != len(inputs.sequences):
        raise ValueError(f"Expected {len(inputs.sequences)} CodonFM embeddings, got {len(embeddings)}")
    vectors = [[float(value) for value in vector] for vector in embeddings]
    if any(not vector for vector in vectors) or any(not math.isfinite(value) for vector in vectors for value in vector):
        raise ValueError("CodonFM returned an empty or non-finite embedding")
    widths = {len(vector) for vector in vectors}
    if len(widths) != 1:
        raise ValueError("CodonFM returned embeddings with inconsistent dimensions")

    return CodonFMEmbeddingsOutput(
        metadata={"model_checkpoint": config.model_checkpoint, "num_sequences": len(inputs.sequences)},
        results=[
            CodonFMEmbeddingResult(sequence=sequence, sequence_length=len(sequence), embedding=vector)
            for sequence, vector in zip(inputs.sequences, vectors, strict=True)
        ],
    )
