"""PyHMMER phmmer tool — search protein sequences against protein sequences."""
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
class PyPhmmerInput(PyHmmerInput):
    """Input object for PyHMMER phmmer (protein sequences vs protein sequences).

    This class defines the input parameters for performing iterative protein
    sequence searches, where query sequences are compared directly against target
    sequences without requiring pre-built HMM profiles.

    Attributes:
        sequences (List[str]): Query protein sequences. Inherited from
            ``PyHmmerInput``. Can be a single sequence string or a list of sequence
            strings. These sequences will be used to build temporary HMM profiles
            on-the-fly.

        target_sequences (List[str]): Target protein sequences to
            search against. Can be a single sequence string or a list of sequence
            strings. The query sequences will be compared against these targets.
    """

    target_sequences: List[str] = Field(
        description="Target sequences as: single sequence string or list of sequence strings"
    )

    @field_validator("target_sequences", mode="before")
    @classmethod
    def normalize_target_sequences(cls, value) -> List[str]:
        """Normalize target sequences to list of strings."""
        # Reuse the same logic as sequences validation
        return PyHmmerInput.normalize_sequences(value)


# Output:
PyPhmmerOutput = PyHmmerOutput

# Config:
PyPhmmerConfig = PyHmmerConfig


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="pyhmmer-phmmer",
    label="PyHMMER PHMMER Search",
    input=PyPhmmerInput,
    config=PyPhmmerConfig,
    output=PyPhmmerOutput,
    description="Search protein sequences against protein database using PyHMMER",
)
@tool_cache("pyhmmer-phmmer")
def run_pyhmmer_phmmer(inputs: PyPhmmerInput, config: PyPhmmerConfig) -> PyPhmmerOutput:
    """Search protein sequences against protein database using PyHMMER.

    This function implements the phmmer algorithm, which performs iterative
    protein-protein searches by building temporary HMM profiles from query
    sequences on-the-fly. This is useful for finding homologous sequences without
    requiring pre-built HMM profiles.

    Args:
        inputs (PyPhmmerInput): Validated PyHMMER phmmer input containing both
            query and target protein sequences.
        config (PyPhmmerConfig): Validated PyHMMER configuration with search
            parameters including E-value thresholds and threading options.

    Returns:
        PyPhmmerOutput: Structured output containing:
            - ``sequence_hits_df``: DataFrame with sequence-level hits
            - ``domain_hits_df``: DataFrame with domain-level hits
            - ``num_sequence_hits``: Total number of sequence hits
            - ``num_domain_hits``: Total number of domain hits

    Raises:
        ValueError: If query or target sequences are empty or invalid.
        RuntimeError: If PyHMMER search execution fails.

    Examples:
        >>> # Search for similar sequences to of query proteins
        >>> inputs = PyPhmmerInput(
        ...     sequences=["MVLSPADKTNVKAAW"],
        ...     target_sequences=["MVLSPADKTN", "ATCGATCGAT", "MVLSPADKTNVK"]
        ... )
        >>> config = PyPhmmerConfig(
        ...     evalue_threshold=1.0,
        ...     domain_evalue_threshold=1.0
        ... )
        >>> result = run_pyhmmer_phmmer(inputs, config)
        >>> print(f"Found {result.num_sequence_hits} similar sequences")
        >>>
        >>> # Find all hits with >80% identity
        >>> if result.sequence_hits_df is not None:
        ...     # Calculate identity from alignment scores
        ...     high_identity = result.sequence_hits_df[
        ...         result.sequence_hits_df['evalue'] < 1e-10
        ...     ]
    """
    from bio_programming.bio_tools.tools.infra.env_manager import EnvManager

    venv_manager = EnvManager(model_name="pyhmmer")

    input_data = {
        "operation": "phmmer",
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

    # Convert results to DataFrames
    sequence_hits_df, domain_hits_df = _build_dataframes(
        output_data["sequence_hits"], output_data["domain_hits"]
    )

    return PyPhmmerOutput(
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
