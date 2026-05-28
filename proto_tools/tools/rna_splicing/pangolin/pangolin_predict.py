"""Per-position tissue-specific splice-site probability prediction with Pangolin."""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from proto_tools.tools.rna_splicing.pangolin.shared_data_models import (
    PANGOLIN_FLANK,
    PangolinConfig,
    PangolinTissue,
    validate_dna,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseToolInput, BaseToolOutput, InputField, ToolInstance

logger = logging.getLogger(__name__)

# Minimum input length: 5000 bp flank on each side + at least one scored position.
MIN_PREDICT_LENGTH = 2 * PANGOLIN_FLANK + 1


# ============================================================================
# Data Models
# ============================================================================
class PangolinPredictInput(BaseToolInput):
    """Input for Pangolin splice-site probability prediction.

    Attributes:
        sequences (list[str]): DNA sequence(s) to score, each >= 10,001 bp
            (5,000 bp of flank on each side). Scores cover the central
            ``len - 10000`` positions; a single string is wrapped to a list.
    """

    sequences: list[str] = InputField(
        title="Sequences",
        description="DNA sequence(s) >= 10001 bp; predictions cover the central (len - 10000) positions",
        min_length=1,
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value: Any) -> list[Any]:
        """Normalize a single sequence string to a one-item list."""
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
        """Validate nucleotides and enforce the minimum length for each sequence."""
        validated: list[str] = []
        for i, raw in enumerate(sequences):
            seq = validate_dna(raw)
            if len(seq) < MIN_PREDICT_LENGTH:
                raise ValueError(
                    f"sequences[{i}] length {len(seq)} < minimum {MIN_PREDICT_LENGTH} "
                    f"(Pangolin needs {PANGOLIN_FLANK} bp of flank on each side)"
                )
            validated.append(seq)
        return validated


class PangolinPrediction(BaseModel):
    """Per-sequence Pangolin splice-site probability prediction.

    Attributes:
        scores (list[list[float]]): Per-position splice-site probability scores
            (the per-tissue P(splice) head) with shape
            ``[len(sequence) - 2 * PANGOLIN_FLANK][len(tissues)]``. Column order
            matches ``tissues``.
        tissues (list[PangolinTissue]): Tissue order of the score columns.
        output_start (int): Index in the input sequence of the first scored
            position (always ``PANGOLIN_FLANK``).
    """

    scores: list[list[float]] = Field(
        title="Scores",
        description="Per-position splice-site probability scores, shape [position][tissue]",
    )
    tissues: list[PangolinTissue] = Field(
        title="Tissues",
        description="Tissue order of the score columns",
    )
    output_start: int = Field(
        title="Output Start",
        description="Input-sequence index of the first scored position (= PANGOLIN_FLANK)",
    )


class PangolinPredictOutput(BaseToolOutput):
    """Output from Pangolin splice-site probability prediction.

    Attributes:
        results (list[PangolinPrediction]): Per-sequence predictions, 1:1 with the
            input sequences.
    """

    results: list[PangolinPrediction] = Field(
        title="Results",
        description="Per-sequence Pangolin predictions (1:1 with input sequences)",
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
            with open(path, "w") as f:
                json.dump(self.model_dump(mode="json"), f, indent=2)
        elif file_format == "npy":
            import numpy as np

            arrays = [np.asarray(r.scores, dtype=float) for r in self.results]
            if len({a.shape for a in arrays}) == 1:
                # Uniform lengths: a single regular (n_seq, n_pos, n_tissue) array.
                np.save(path, np.stack(arrays))
            else:
                # Ragged batch: fill the object array element-wise (np.array(dtype=object)
                # can broadcast and raise). Loading needs allow_pickle=True; prefer JSON.
                obj = np.empty(len(arrays), dtype=object)
                for idx, array in enumerate(arrays):
                    obj[idx] = array
                np.save(path, obj, allow_pickle=True)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


class PangolinPredictConfig(PangolinConfig):
    """Configuration for Pangolin splice-site prediction.

    Attributes:
        tissues (list[PangolinTissue]): Tissues whose splice predictions are
            ensembled. Defaults to all four Pangolin tissues.
    """


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples (10,004 bp sequence)."""
    return PangolinPredictInput(sequences=["ACGT" * 2501])


@tool(
    key="pangolin-predict",
    label="Pangolin Splice-Site Prediction",
    category="rna_splicing",
    input_class=PangolinPredictInput,
    config_class=PangolinPredictConfig,
    output_class=PangolinPredictOutput,
    description="Per-position tissue-specific splice-site probability prediction using Pangolin",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="sequences",
    iterable_output_field="results",
    cacheable=True,
)
def run_pangolin_predict(
    inputs: PangolinPredictInput,
    config: PangolinPredictConfig,
    instance: Any = None,
) -> PangolinPredictOutput:
    """Predict tissue-specific splice-site probability along DNA sequences with Pangolin.

    Pangolin is a SpliceAI-lineage deep-learning model that scores the
    splice-site probability per position across four tissues (heart, liver,
    brain, testis). Each input sequence is scored over its central
    ``len - 10000`` positions, using 5,000 bp of flanking context on each side.

    Args:
        inputs (PangolinPredictInput): Validated input sequences.
        config (PangolinPredictConfig): Pangolin configuration (tissues, device).
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        PangolinPredictOutput: Per-sequence predictions, 1:1 with the input
            sequences.

    See Also:
        - Pangolin paper: https://doi.org/10.1186/s13059-022-02664-4
        - Model repository: https://github.com/tkzeng/Pangolin
    """
    logger.debug("Using local venv for Pangolin prediction (tissues=%s)", config.tissues)

    input_data = {
        "operation": "predict",
        "sequences": inputs.sequences,
        "tissues": config.tissues,
        "device": config.device,
    }

    output_data = ToolInstance.dispatch("pangolin", input_data, instance=instance, config=config)

    results = [
        PangolinPrediction(scores=item["scores"], tissues=config.tissues, output_start=PANGOLIN_FLANK)
        for _sequence, item in zip(inputs.sequences, output_data["results"], strict=True)
    ]
    return PangolinPredictOutput(results=results, metadata={"tissues": config.tissues})
