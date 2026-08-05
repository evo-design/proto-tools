"""CodonFM (Encodon) masked-codon sampling: resample a subset of codons from the model."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from proto_tools.tools.masked_models.codonfm.shared_data_models import (
    CodonFMCheckpoint,
    CodonSequenceInput,
    resolve_checkpoint_source,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, BaseToolOutput, ConfigField, ToolInstance

CodonFMSampleInput = CodonSequenceInput


class CodonFMSampleConfig(BaseConfig):
    """Configuration for CodonFM (Encodon) masked-codon sampling.

    Attributes:
        model_checkpoint (CodonFMCheckpoint): Encodon checkpoint to sample from.
        num_mutations (int | None): Exact number of codon positions to resample per sequence.
            The sampled codon can equal the original. When ``None``, ``mask_fraction`` is used.
        mask_fraction (float): Fraction of codons to resample when ``num_mutations`` is ``None``.
        temperature (float): Softmax temperature for codon sampling; higher is more diverse.
        device (str): Device used for CodonFM inference.
        batch_size (int): Number of (same-length) sequences per GPU forward pass.
    """

    model_checkpoint: CodonFMCheckpoint = ConfigField(
        title="Model Checkpoint",
        default="encodon_80m",
        description="Encodon checkpoint: encodon_80m | encodon_600m | encodon_1b | encodon_1b_cdwt.",
        reload_on_change=True,
    )
    num_mutations: int | None = ConfigField(
        title="Num Mutations",
        default=None,
        ge=1,
        description="Exact codons to resample per sequence; None uses mask_fraction.",
    )
    mask_fraction: float = ConfigField(
        title="Mask Fraction",
        default=0.15,
        gt=0.0,
        allow_inf_nan=False,
        le=1.0,
        description="Fraction of codons to resample when num_mutations is None.",
    )
    temperature: float = ConfigField(
        title="Temperature",
        default=1.0,
        gt=0.0,
        allow_inf_nan=False,
        description="Softmax temperature for codon sampling; higher is more diverse.",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="CUDA device to run CodonFM inference on.",
        include_in_key=False,
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of same-length sequences to sample per GPU batch.",
        include_in_key=False,
    )


class CodonFMSampleResult(BaseModel):
    """One resampled coding sequence.

    Attributes:
        sequence (str): Resampled coding sequence in the DNA alphabet.
    """

    sequence: str = Field(title="Sequence", description="Resampled coding sequence in the DNA alphabet")


class CodonFMSampleOutput(BaseToolOutput):
    """Output from CodonFM masked-codon sampling.

    Attributes:
        results (list[CodonFMSampleResult]): One resampled coding sequence per input.
    """

    results: list[CodonFMSampleResult] = Field(
        default_factory=list,
        title="Results",
        description="Resampled coding sequences, one per input and in input order",
    )

    @property
    def sequences(self) -> list[str]:
        """Return sampled sequence strings in input order."""
        return [result.sequence for result in self.results]

    def __len__(self) -> int:
        """Return the number of sampled sequences."""
        return len(self.results)

    def __getitem__(self, index: int) -> str:
        """Return one sampled sequence."""
        return self.results[index].sequence

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        """Iterate over sampled sequences."""
        return iter(self.sequences)

    @property
    def output_format_options(self) -> list[str]:
        """Supported export formats."""
        return ["json"]

    @property
    def output_format_default(self) -> str:
        """Default export format."""
        return "json"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Export the sampled sequences as JSON."""
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        base = Path(export_path)
        (base.parent / f"{base.name}.json").write_text(json.dumps(self.sequences, indent=2))


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return CodonFMSampleInput(sequences=["ATGGTGAGCAAGGGCGAGGAGCTGTTCACC"])


@tool(
    key="codonfm-sample",
    label="CodonFM Sampling",
    category="masked_models",
    input_class=CodonFMSampleInput,
    config_class=CodonFMSampleConfig,
    output_class=CodonFMSampleOutput,
    description="Resample a subset of codons in coding sequences with the CodonFM/Encodon model",
    uses_gpu=True,
    stochastic=True,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    max_chunk_size=32,
    cacheable=False,
)
def run_codonfm_sample(
    inputs: CodonFMSampleInput,
    config: CodonFMSampleConfig,
    instance: Any = None,
) -> CodonFMSampleOutput:
    """Resample masked codons in coding sequences with CodonFM (Encodon).

    A subset of codon positions (``num_mutations`` or ``mask_fraction``) is masked and refilled
    by sampling from the model's per-codon distribution over the 61 sense codons (stop codons are
    excluded, so a resampled codon is never a premature stop); sequence length is preserved.

    Args:
        inputs (CodonFMSampleInput): Coding sequences to resample.
        config (CodonFMSampleConfig): CodonFM sampling configuration.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        CodonFMSampleOutput: One resampled coding sequence per input.
    """
    safetensors_url, config_url, filename, subdir = resolve_checkpoint_source(config.model_checkpoint)
    if config.num_mutations is not None:
        shortest = min(len(sequence) // 3 for sequence in inputs.sequences)
        if config.num_mutations > shortest:
            raise ValueError(
                f"num_mutations ({config.num_mutations}) exceeds the shortest input sequence ({shortest} codons)"
            )

    output_data = ToolInstance.dispatch(
        "codonfm",
        {
            "operation": "sample",
            "sequences": inputs.sequences,
            "num_mutations": config.num_mutations,
            "mask_fraction": config.mask_fraction,
            "temperature": config.temperature,
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

    sampled = output_data["sequences"]
    if len(sampled) != len(inputs.sequences):
        raise ValueError(f"Expected {len(inputs.sequences)} CodonFM samples, got {len(sampled)}")
    sampled_sequences = [str(sequence) for sequence in sampled]
    for index, (source, sequence) in enumerate(zip(inputs.sequences, sampled_sequences, strict=True)):
        if len(sequence) != len(source) or len(sequence) % 3 or any(base not in "ACGT" for base in sequence):
            raise ValueError(f"CodonFM returned an invalid sampled sequence at index {index}")

    return CodonFMSampleOutput(
        metadata={"model_checkpoint": config.model_checkpoint, "num_sequences": len(inputs.sequences)},
        results=[CodonFMSampleResult(sequence=sequence) for sequence in sampled_sequences],
    )
