"""PyHMMER jackhmmer tool — iterative protein sequence search."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import Field, field_validator

from bio_programming_tools.tools.infra.tool_cache import tool_cache
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.tools.utils import ConfigField

from .shared_data_models import (
    PyHmmerConfig,
    PyHmmerInput,
    PyHmmerOutput,
    _build_dataframes,
)


# ============================================================================
# Data Models
# ============================================================================
# Input:
class PyJackhmmerInput(PyHmmerInput):
    """Input object for PyHMMER jackhmmer (protein sequences vs protein sequences).

    Attributes:
        sequences (List[str]): Query protein sequences.
            Inherited from ``PyHmmerInput``. Can be a single sequence string or
            a list of sequence strings.

        target_sequences (List[str]): Target protein sequences to
            search against. Can be a single sequence string or a list of sequence
            strings.
    """

    target_sequences: List[str] = Field(
        description="Target sequences as: single sequence string or list of sequence strings"
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


class PyJackhmmerConfig(PyHmmerConfig):
    """Configuration for PyHMMER jackhmmer search.

    Fields:
        num_threads: Number of CPU threads to use.
        evalue_threshold: Sequence-level E-value reporting threshold.
        score_threshold: Sequence-level score threshold.
        domain_evalue_threshold: Domain-level E-value reporting threshold.
        domain_score_threshold: Domain-level score threshold.
        max_iterations: Maximum number of jackhmmer iterations.
    """

    max_iterations: int = ConfigField(
        title="Maximum Iterations",
        default=5,
        ge=1,
        description="Maximum number of jackhmmer search iterations.",
        advanced=True,
    )


# Output:
PyJackhmmerOutput = PyHmmerOutput


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="pyhmmer-jackhmmer",
    label="PyHMMER JackHMMER Search",
    input=PyJackhmmerInput,
    config=PyJackhmmerConfig,
    output=PyJackhmmerOutput,
    description="Iteratively search protein sequences against protein database using PyHMMER",
)
@tool_cache("pyhmmer-jackhmmer")
def run_pyhmmer_jackhmmer(
    inputs: PyJackhmmerInput, config: PyJackhmmerConfig
) -> PyJackhmmerOutput:
    """Iteratively search protein sequences against protein database using PyHMMER.

    Args:
        inputs (PyJackhmmerInput): Validated jackhmmer input containing query and
            target protein sequences.
        config (PyJackhmmerConfig): Validated configuration including
            ``max_iterations`` and threshold settings.

    Returns:
        PyJackhmmerOutput: Structured output with sequence-level and domain-level hits.
    """
    from bio_programming_tools.tools.infra.env_manager import EnvManager

    venv_manager = EnvManager(model_name="pyhmmer")

    input_data = {
        "operation": "jackhmmer",
        "sequences": inputs.sequences,
        "target_sequences": inputs.target_sequences,
        "max_iterations": config.max_iterations,
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

    return PyJackhmmerOutput(
        metadata={
            "num_query_sequences": output_data.get("num_query_sequences", 0),
            "num_target_sequences": output_data.get("num_target_sequences", 0),
            "max_iterations": config.max_iterations,
            "iterations_per_query": output_data.get("iterations_per_query", []),
            "converged_per_query": output_data.get("converged_per_query", []),
            "num_threads": config.num_threads,
            "evalue_threshold": config.evalue_threshold,
            "score_threshold": config.score_threshold,
            "domain_evalue_threshold": config.domain_evalue_threshold,
            "domain_score_threshold": config.domain_score_threshold,
        },
        sequence_hits_df=sequence_hits_df,
        domain_hits_df=domain_hits_df,
    )
