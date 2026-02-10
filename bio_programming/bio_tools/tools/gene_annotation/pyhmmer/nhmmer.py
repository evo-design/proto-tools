"""PyHMMER nhmmer tool — search nucleotide sequences against nucleotide sequences."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import Field, field_validator

from bio_programming.bio_tools.tools.infra.tool_cache import tool_cache
from bio_programming.bio_tools.tools.tool_registry import tool

from .shared_data_models import PyHmmerConfig, PyHmmerInput, PyHmmerOutput, _build_dataframes


# ============================================================================
# Data Models
# ============================================================================
# Input:
class PyNhmmerInput(PyHmmerInput):
    """Input object for PyHMMER nhmmer (nucleotide sequences vs nucleotide sequences).

    Attributes:
        sequences (List[str]): Query nucleotide sequences.
            Inherited from ``PyHmmerInput``. Can be a single sequence string or
            a list of sequence strings.

        target_sequences (List[str]): Target nucleotide sequences to
            search against. Can be a single sequence string or a list of sequence
            strings.
    """

    target_sequences: List[str] = Field(
        description="Target nucleotide sequences as: single sequence string or list of sequence strings"
    )

    @field_validator("target_sequences", mode="before")
    @classmethod
    def normalize_target_sequences(cls, value) -> List[str]:
        """Normalize target sequences to list of strings."""
        return PyHmmerInput.normalize_sequences(value)

    @field_validator("target_sequences")
    @classmethod
    def validate_target_sequences(cls, sequences: List[str]) -> List[str]:
        """Validate target sequences are non-empty."""
        return PyHmmerInput.validate_sequences(sequences)


# Output:
PyNhmmerOutput = PyHmmerOutput

# Config:
PyNhmmerConfig = PyHmmerConfig


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="pyhmmer-nhmmer",
    label="PyHMMER NHMMER Search",
    input=PyNhmmerInput,
    config=PyNhmmerConfig,
    output=PyNhmmerOutput,
    description="Search nucleotide sequences against nucleotide database using PyHMMER",
)
@tool_cache("pyhmmer-nhmmer")
def run_pyhmmer_nhmmer(inputs: PyNhmmerInput, config: PyNhmmerConfig) -> PyNhmmerOutput:
    """Search nucleotide sequences against nucleotide database using PyHMMER.

    Args:
        inputs (PyNhmmerInput): Validated PyHMMER nhmmer input containing both
            query and target nucleotide sequences.
        config (PyNhmmerConfig): Validated PyHMMER configuration with search
            parameters including E-value thresholds and threading options.

    Returns:
        PyNhmmerOutput: Structured output with sequence-level and domain-level hits.
    """
    from bio_programming.bio_tools.tools.infra.env_manager import EnvManager

    venv_manager = EnvManager(model_name="pyhmmer")

    input_data = {
        "operation": "nhmmer",
        "sequences": inputs.sequences,
        "target_sequences": inputs.target_sequences,
        "num_threads": config.num_threads,
        "evalue_threshold": config.evalue_threshold,
        "score_threshold": config.score_threshold,
        "domain_evalue_threshold": config.domain_evalue_threshold,
        "domain_score_threshold": config.domain_score_threshold,
    }

    output_data = venv_manager.call_standalone_script_in_venv(
        script_path=Path(__file__).parent / "standalone" / "run.py",
        input_dict=input_data,
        device="cpu",
    )

    sequence_hits_df, domain_hits_df = _build_dataframes(
        output_data["sequence_hits"], output_data["domain_hits"]
    )

    return PyNhmmerOutput(
        metadata={
            "num_query_sequences": output_data.get("num_query_sequences", 0),
            "num_target_sequences": output_data.get("num_target_sequences", 0),
            "num_threads": config.num_threads,
            "evalue_threshold": config.evalue_threshold,
            "score_threshold": config.score_threshold,
            "domain_evalue_threshold": config.domain_evalue_threshold,
            "domain_score_threshold": config.domain_score_threshold,
        },
        sequence_hits_df=sequence_hits_df,
        domain_hits_df=domain_hits_df,
    )
