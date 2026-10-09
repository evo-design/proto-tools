"""SpliceAI2 splice site, junction, and transcript prediction from DNA sequence."""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from proto_tools.tools.rna_splicing.spliceai2.shared_data_models import SPLICEAI2_HF_URL, SpliceAI2Config
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
    return_invalid_dna_chars,
)
from proto_tools.utils.auth import require_hf_token

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
class SpliceAI2PredictInput(BaseToolInput):
    """Input for SpliceAI2 splice prediction.

    Attributes:
        sequences (list[str]): Sense-strand DNA sequence(s) to predict on. A
            single string is auto-wrapped into a list. Any length is accepted:
            the model's 65,536 bp of context per side is padded with unknown
            bases, so predictions cover every input position.
    """

    sequences: list[str] = InputField(
        title="Sequences",
        description="Sense-strand DNA sequence(s) to predict splice sites and transcripts for",
        examples=["ACGTACGTACGT", ["ACGT", "TTGGCCAA"]],
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value: Any) -> Any:
        """Convert a single string to a list and reject empty input."""
        if isinstance(value, str):
            return [value]
        if not value:
            raise ValueError("sequences must not be empty")
        return value

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, sequences: list[str]) -> list[str]:
        """Uppercase each sequence and require DNA bases (A/C/G/T/N)."""
        normalized = []
        for i, raw in enumerate(sequences):
            sequence = raw.strip().upper()
            if not sequence:
                raise ValueError(f"sequences[{i}] is empty")
            invalid = return_invalid_dna_chars(sequence, additional_valid_chars="N")
            if invalid:
                raise ValueError(f"sequences[{i}] has invalid nucleotide characters: {', '.join(sorted(invalid))}")
            normalized.append(sequence)
        return normalized


class SpliceAI2Junction(BaseModel):
    """A predicted splice junction (intron) between two candidate splice sites.

    Attributes:
        donor (int): 1-based position of the intron's first base (the donor site).
        acceptor (int): 1-based position of the intron's last base (the acceptor site).
        probability (float): Predicted junction usage (0-1).
    """

    donor: int = Field(title="Donor", description="1-based position of the intron's first base (donor site)")
    acceptor: int = Field(title="Acceptor", description="1-based position of the intron's last base (acceptor site)")
    probability: float = Field(title="Probability", description="Predicted junction usage (0-1)")


class SpliceAI2Transcript(BaseModel):
    """A predicted transcript, decoded as a path through the splice graph.

    Attributes:
        exons (list[tuple[int, int]]): Exons as 1-based inclusive ``(start, end)``
            intervals in the input sequence, in order.
    """

    exons: list[tuple[int, int]] = Field(
        title="Exons",
        description="Exons as 1-based inclusive (start, end) intervals, in order",
    )


class SpliceAI2Prediction(BaseModel):
    """SpliceAI2 predictions for one input sequence.

    Attributes:
        donor_probabilities (list[float]): Per-position splice donor usage (0-1),
            scored at the first base of each intron.
        acceptor_probabilities (list[float]): Per-position splice acceptor usage
            (0-1), scored at the last base of each intron.
        junctions (list[SpliceAI2Junction]): Junctions with usage >= 0.01 between
            candidate splice sites, ordered by donor then acceptor.
        transcripts (list[SpliceAI2Transcript]): Highest-scoring transcripts, best
            first. Empty when fewer than two candidate splice sites are found.
    """

    donor_probabilities: list[float] = Field(
        title="Donor Probabilities", description="Per-position splice donor usage (0-1)"
    )
    acceptor_probabilities: list[float] = Field(
        title="Acceptor Probabilities", description="Per-position splice acceptor usage (0-1)"
    )
    junctions: list[SpliceAI2Junction] = Field(
        title="Junctions", description="Junctions with usage >= 0.01 between candidate splice sites"
    )
    transcripts: list[SpliceAI2Transcript] = Field(
        title="Transcripts", description="Highest-scoring decoded transcripts, best first"
    )


class SpliceAI2PredictOutput(BaseToolOutput):
    """Output from SpliceAI2 splice prediction.

    Attributes:
        results (list[SpliceAI2Prediction]): One entry per input sequence, in input order.
    """

    results: list[SpliceAI2Prediction] = Field(
        title="Results",
        description="Splice site, junction, and transcript predictions, one per input sequence",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "npy"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "json":
            with open(path, "w") as handle:
                json.dump(self.model_dump(mode="json"), handle, indent=2)
            return

        if file_format == "npy":
            import numpy as np

            # One (L, 2) [donor, acceptor] array per sequence; ragged batches need an object array.
            arrays = [np.column_stack([r.donor_probabilities, r.acceptor_probabilities]) for r in self.results]
            uniform = len({a.shape for a in arrays}) <= 1
            np.save(path, np.stack(arrays) if uniform else np.array(arrays, dtype=object))
            return

        raise ValueError(f"Unsupported format: {file_format}")


class SpliceAI2PredictConfig(SpliceAI2Config):
    """Configuration for SpliceAI2 splice prediction.

    Attributes:
        num_transcripts (int): Number of highest-scoring transcripts to decode
            per sequence.
        assembly (SpliceAI2Assembly): Species assembly whose channel conditions
            the model. ``GRCh38`` (human) for human sequence.
        device (str): Device to run the model on (default ``cuda``).
    """

    num_transcripts: int = ConfigField(
        title="Number of Transcripts",
        default=3,
        ge=1,
        description="Number of highest-scoring transcripts to decode per sequence",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return SpliceAI2PredictInput(sequences=["ACGT" * 25])


@tool(
    key="spliceai2-predict",
    label="SpliceAI2 Splice and Transcript Prediction",
    category="rna_splicing",
    input_class=SpliceAI2PredictInput,
    config_class=SpliceAI2PredictConfig,
    output_class=SpliceAI2PredictOutput,
    description="Predict splice site and junction usage and decode transcripts from DNA with SpliceAI2",
    uses_gpu=True,
    gpu_only=True,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    max_chunk_size=16,
    cacheable=True,
)
def run_spliceai2_predict(
    inputs: SpliceAI2PredictInput,
    config: SpliceAI2PredictConfig,
    instance: Any = None,
) -> SpliceAI2PredictOutput:
    """Predict splice sites, junctions, and transcripts with SpliceAI2.

    Each sequence is read on its sense strand and padded with 65,536 unknown
    bases per side, the context SpliceAI2 crops away. Splice donor and acceptor
    usage is predicted at every position; sites above 0.01 usage become
    candidates for junction prediction, and transcripts are decoded as the
    highest-scoring paths through the resulting splice graph.

    Args:
        inputs (SpliceAI2PredictInput): Sense-strand DNA sequences.
        config (SpliceAI2PredictConfig): Transcript count, species assembly, and device.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        SpliceAI2PredictOutput: Per-sequence predictions, 1:1 with the inputs.

    Raises:
        OSError: If no HuggingFace token is available for the gated weights.

    See Also:
        - Model repository: https://github.com/Illumina/SpliceAI2
        - Weights: https://huggingface.co/illumina-ai/SpliceAI2
    """
    require_hf_token("SpliceAI2", SPLICEAI2_HF_URL)
    logger.debug("Using local venv for SpliceAI2 prediction")

    output_data = ToolInstance.dispatch(
        "spliceai2",
        {
            "operation": "predict",
            "sequences": inputs.sequences,
            "num_transcripts": config.num_transcripts,
            "assembly": config.assembly,
            "device": config.device,
        },
        instance=instance,
        config=config,
    )

    return SpliceAI2PredictOutput(
        results=[SpliceAI2Prediction(**item) for item in output_data["results"]],
        metadata={"assembly": config.assembly, "num_transcripts": config.num_transcripts},
    )
