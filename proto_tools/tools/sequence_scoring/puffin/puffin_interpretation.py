"""Puffin interpretable transcription initiation analysis with motif decomposition."""

import csv
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from proto_tools.tools.sequence_scoring.shared_data_models import validate_dna_sequence
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)

logger = logging.getLogger(__name__)

PUFFIN_PADDING = 325
PUFFIN_MIN_INPUT_LENGTH = 2 * PUFFIN_PADDING + 1
TARGET_SIGNALS = ("FANTOM_CAGE", "ENCODE_CAGE", "ENCODE_RAMPAGE", "GRO_CAP", "PRO_CAP")
MOTIF_NAMES = (
    "CREB",
    "ETS",
    "NFY",
    "NRF1",
    "SP",
    "TATA",
    "U1_snRNP",
    "YY1",
    "ZNF143",
)


# ============================================================================
# Data Models
# ============================================================================
class PuffinInterpretationInput(BaseToolInput):
    """Input for Puffin motif-level interpretation of transcription initiation.

    Each sequence must be at least 651 bp. The model uses 325 bp padding on each
    side of the region of interest, so the per-base output length equals
    ``len(sequence) - 650``.

    Attributes:
        sequences (list[str]): DNA sequence(s) at least 651 bp long. A single
            string is normalized to a one-item list. Only ``A``, ``C``, ``G``,
            ``T``, ``N`` are accepted.
    """

    sequences: list[str] = InputField(
        title="Sequences",
        description="DNA sequence(s) >= 651 bp. Per-base output length = len(seq) - 650",
        min_length=1,
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value: Any) -> list[Any]:
        """Normalize the plural sequence field to a list."""
        if value is None:
            raise ValueError("sequences cannot be None")
        if isinstance(value, str):
            return [value]
        if not value:
            raise ValueError("sequences cannot be empty")
        return value  # type: ignore[no-any-return]

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, sequences: list[str]) -> list[str]:
        """Validate nucleotide composition and minimum length."""
        validated = [validate_dna_sequence(sequence) for sequence in sequences]
        for index, sequence in enumerate(validated):
            if len(sequence) < PUFFIN_MIN_INPUT_LENGTH:
                raise ValueError(
                    f"sequences[{index}]: must be at least {PUFFIN_MIN_INPUT_LENGTH} bp "
                    f"(2 * {PUFFIN_PADDING} bp padding + 1 bp output); got {len(sequence)}"
                )
        return validated

    def __len__(self) -> int:
        """Number of input sequences."""
        return len(self.sequences)


class PuffinInterpretationResult(BaseModel):
    """Per-sequence Puffin interpretation result.

    All per-base arrays have length ``output_length`` (= ``sequence_length - 650``)
    and align with sequence coordinates ``[output_start, output_end)`` in the
    input sequence. ``prediction`` is the log-scale predicted signal for the
    selected ``target_signal`` and strand. The motif-keyed dicts use
    strand-suffixed names such as ``"YY1+"`` and ``"YY1-"`` (18 entries: 9
    motifs x 2 strands). The Long Inr motif used internally to compute the
    initiator-effect sum is not exposed per-motif.

    Attributes:
        sequence (str): Input DNA sequence that was scored.
        sequence_length (int): Length of the input sequence.
        output_length (int): Number of per-base output positions
            (= ``sequence_length - 650``).
        output_start (int): 0-based sequence coordinate of the first per-base
            output position (always ``325``).
        output_end (int): 0-based exclusive end of the per-base output span in
            the input sequence (= ``sequence_length - 325``).
        prediction (list[float]): Predicted transcription initiation signal for
            the selected ``target_signal`` and strand, length ``output_length``.
        motif_activations (dict[str, list[float]]): Per-base motif activation
            scores keyed by strand-suffixed motif name (e.g. ``"TATA+"``).
        motif_effects (dict[str, list[float]]): Per-base motif effect scores
            keyed by strand-suffixed motif name.
        sum_motif_effects (list[float]): Per-base sum of motif effects across
            non-initiator motifs.
        sum_initiator_effects (list[float]): Per-base sum of initiator-motif
            effects (centered to zero mean).
        sum_trinucleotide_effects (list[float]): Per-base sum of trinucleotide
            sequence effects (centered to zero mean).
        sum_total_effects (list[float]): Per-base sum of all sequence pattern
            effects (motif + initiator + trinucleotide).
        bp_contribution (list[float]): Per-base contribution score to
            transcription initiation at the target.
        bp_contribution_per_motif (dict[str, list[float]]): Per-base
            contribution to transcription, decomposed by motif name.
        bp_contribution_to_motif_activation (dict[str, list[float]]): Per-base
            contribution to motif-activation scores, decomposed by motif name.
    """

    sequence: str = Field(title="Sequence", description="DNA sequence originally provided to the tool")
    sequence_length: int = Field(title="Sequence Length", description="Length of the provided DNA sequence")
    output_length: int = Field(
        title="Output Length",
        description="Number of per-base output positions (= sequence_length - 650)",
    )
    output_start: int = Field(
        title="Output Start",
        description="0-based sequence coordinate of the first output position",
    )
    output_end: int = Field(
        title="Output End",
        description="0-based exclusive end of the output span in the input sequence",
    )
    prediction: list[float] = Field(
        title="Prediction",
        description="Predicted log-scale signal for the selected target and strand, length output_length",
    )
    motif_activations: dict[str, list[float]] = Field(
        title="Motif Activations",
        description="Per-base motif activation scores keyed by strand-suffixed motif name",
    )
    motif_effects: dict[str, list[float]] = Field(
        title="Motif Effects",
        description="Per-base motif effect scores keyed by strand-suffixed motif name",
    )
    sum_motif_effects: list[float] = Field(
        title="Sum Motif Effects", description="Per-base sum of non-initiator motif effects"
    )
    sum_initiator_effects: list[float] = Field(
        title="Sum Initiator Effects",
        description="Per-base sum of initiator-motif effects, centered",
    )
    sum_trinucleotide_effects: list[float] = Field(
        title="Sum Trinucleotide Effects",
        description="Per-base sum of trinucleotide sequence effects, centered",
    )
    sum_total_effects: list[float] = Field(
        title="Sum Total Effects",
        description="Per-base sum of motif, initiator, and trinucleotide effects",
    )
    bp_contribution: list[float] = Field(
        title="BP Contribution", description="Per-base contribution to transcription initiation"
    )
    bp_contribution_per_motif: dict[str, list[float]] = Field(
        title="BP Contribution per Motif",
        description="Per-base contribution to transcription, decomposed by motif",
    )
    bp_contribution_to_motif_activation: dict[str, list[float]] = Field(
        title="BP Contribution to Motif",
        description="Per-base contribution to motif-activation scores, decomposed by motif",
    )


class PuffinInterpretationConfig(BaseConfig):
    """Configuration for Puffin interpretation.

    Attributes:
        device (str): Device used for inference.
        target_signal (Literal['FANTOM_CAGE', 'ENCODE_CAGE', 'ENCODE_RAMPAGE', 'GRO_CAP', 'PRO_CAP']):
            Which transcription initiation assay's predictions to decompose into
            motif and basepair contributions.
        reverse_strand (bool): If ``True``, decompose the reverse-strand
            prediction for the chosen target instead of the forward strand.
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        include_in_key=False,
    )
    target_signal: Literal["FANTOM_CAGE", "ENCODE_CAGE", "ENCODE_RAMPAGE", "GRO_CAP", "PRO_CAP"] = ConfigField(
        title="Target Signal",
        default="FANTOM_CAGE",
        description="Which transcription initiation assay's predictions to decompose",
    )
    reverse_strand: bool = ConfigField(
        title="Reverse Strand",
        default=False,
        description="Decompose the reverse-strand prediction instead of the forward strand",
    )


class PuffinInterpretationOutput(BaseToolOutput):
    """Output from Puffin interpretation.

    Attributes:
        results (list[PuffinInterpretationResult]): Per-sequence interpretation
            results.
        target_signal (str): Target signal selected for interpretation.
        reverse_strand (bool): Whether the reverse-strand head was used.
        motif_names (list[str]): The 9 learned motif names (without strand
            suffix) exposed in the per-motif tracks. Strand-suffixed names
            appear as keys in each result's motif-keyed dicts.
    """

    results: list[PuffinInterpretationResult] = Field(
        title="Results", description="Per-sequence interpretation results"
    )
    target_signal: str = Field(
        title="Target Signal",
        description="Target transcription initiation signal that was interpreted",
    )
    reverse_strand: bool = Field(title="Reverse Strand", description="Whether the reverse-strand head was used")
    motif_names: list[str] = Field(
        title="Motif Names",
        description="The 9 learned motif names (without strand suffix)",
    )

    def __len__(self) -> int:
        """Number of per-sequence results."""
        return len(self.results)

    def __getitem__(self, index: int) -> PuffinInterpretationResult:
        """Per-sequence result by index."""
        return self.results[index]

    def __iter__(self) -> Iterator[PuffinInterpretationResult]:  # type: ignore[override]
        """Iterate per-sequence results."""
        return iter(self.results)

    @property
    def output_format_options(self) -> list[str]:
        """Supported export formats."""
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        """Default export format."""
        return "json"

    def _export_output(self, export_path: Path | str, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")
        _metadata_fields = {
            "tool_id",
            "execution_time",
            "timestamp",
            "success",
            "warnings",
            "errors",
            "metadata",
        }
        data = {k: v for k, v in self.model_dump().items() if k not in _metadata_fields}
        if file_format == "json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        elif file_format == "csv":
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "sequence_index",
                        "sequence_length",
                        "output_length",
                        "output_start",
                        "output_end",
                        "target_signal",
                        "reverse_strand",
                        "motif_names",
                        "prediction",
                        "motif_activations",
                        "motif_effects",
                        "sum_motif_effects",
                        "sum_initiator_effects",
                        "sum_trinucleotide_effects",
                        "sum_total_effects",
                        "bp_contribution",
                        "bp_contribution_per_motif",
                        "bp_contribution_to_motif_activation",
                    ]
                )
                for idx, result in enumerate(self.results):
                    writer.writerow(
                        [
                            idx,
                            result.sequence_length,
                            result.output_length,
                            result.output_start,
                            result.output_end,
                            self.target_signal,
                            self.reverse_strand,
                            json.dumps(self.motif_names),
                            json.dumps(result.prediction),
                            json.dumps(result.motif_activations),
                            json.dumps(result.motif_effects),
                            json.dumps(result.sum_motif_effects),
                            json.dumps(result.sum_initiator_effects),
                            json.dumps(result.sum_trinucleotide_effects),
                            json.dumps(result.sum_total_effects),
                            json.dumps(result.bp_contribution),
                            json.dumps(result.bp_contribution_per_motif),
                            json.dumps(result.bp_contribution_to_motif_activation),
                        ]
                    )
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> PuffinInterpretationInput:
    """Minimal valid input for testing and examples."""
    return PuffinInterpretationInput(sequences=["A" * 1650])


@tool(
    key="puffin-interpretation",
    label="Puffin Interpretation",
    category="sequence_scoring",
    input_class=PuffinInterpretationInput,
    config_class=PuffinInterpretationConfig,
    output_class=PuffinInterpretationOutput,
    description="Motif-level interpretation of transcription initiation with Puffin",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="sequences",
    iterable_output_field="results",
    cacheable=True,
)
def run_puffin_interpretation(
    inputs: PuffinInterpretationInput,
    config: PuffinInterpretationConfig,
    instance: Any = None,
) -> PuffinInterpretationOutput:
    """Decompose per-base transcription initiation into motif contributions.

    Args:
        inputs (PuffinInterpretationInput): Validated DNA sequence input.
        config (PuffinInterpretationConfig): Validated runtime configuration.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        PuffinInterpretationOutput: Per-sequence interpretation results with
            motif activations, motif effects, and basepair contribution scores.
    """
    logger.debug("Using local venv for Puffin interpretation")

    result = ToolInstance.dispatch(
        "puffin",
        {
            "operation": "interpret",
            "sequences": inputs.sequences,
            "target_signal": config.target_signal,
            "reverse_strand": config.reverse_strand,
            "device": config.device,
            "verbose": config.verbose,
            "seed": config.seed,
        },
        instance=instance,
        config=config,
    )

    per_sequence = result["results"]
    if len(per_sequence) != len(inputs.sequences):
        raise ValueError(f"Expected {len(inputs.sequences)} Puffin interpretations, got {len(per_sequence)}")

    results = [
        PuffinInterpretationResult(
            sequence=sequence,
            sequence_length=len(sequence),
            output_length=len(sequence) - 2 * PUFFIN_PADDING,
            output_start=PUFFIN_PADDING,
            output_end=len(sequence) - PUFFIN_PADDING,
            **entry,
        )
        for sequence, entry in zip(inputs.sequences, per_sequence, strict=True)
    ]
    return PuffinInterpretationOutput(
        results=results,
        target_signal=config.target_signal,
        reverse_strand=config.reverse_strand,
        motif_names=list(MOTIF_NAMES),
    )
