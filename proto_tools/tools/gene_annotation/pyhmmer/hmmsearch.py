"""proto_tools/tools/gene_annotation/pyhmmer/hmmsearch.py

PyHMMER hmmsearch tool: search HMM profiles against protein sequences."""
from __future__ import annotations

from pathlib import Path

from pydantic import field_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import InputField, ToolInstance

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
class PyHmmsearchInput(PyHmmerInput):
    """Input object for PyHMMER hmmsearch (HMM profile vs protein sequences).

    This class defines the input parameters for searching one or more HMM profiles
    against protein sequences to identify matching protein families or domains.

    Attributes:
        sequences (list[str]): Target protein sequences to search.
            Inherited from ``PyHmmerInput``. Can be a single sequence string or
            a list of sequence strings.

        hmm (str | Path): Path to an HMM file containing one or more
            profile HMMs. The file should be in HMMER3 format (typically ``.hmm``
            extension). Can contain multiple HMM profiles; all will be searched
            against the target sequences.
    """

    hmm: str | Path = InputField(description="Path to HMM file")

    @field_validator("hmm", mode="before")
    @classmethod
    def validate_hmm_file(cls, value: str | Path) -> str:
        """Validate HMM file exists and return path as string."""
        hmm_path = Path(value)
        if not hmm_path.exists():
            raise ValueError(f"HMM file not found: {value}")
        if not hmm_path.is_file():
            raise ValueError(f"HMM path is not a file: {value}")
        return str(hmm_path)


# Output:
PyHmmsearchOutput = PyHmmerOutput

# Config:
PyHmmsearchConfig = PyHmmerConfig


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input():
    """Minimal valid input for testing and examples."""
    return PyHmmsearchInput(
        sequences=["MKTL"],
        hmm=str(Path(__file__).parent / "examples" / "example.hmm"),
    )


@tool(
    key="pyhmmer-hmmsearch",
    label="PyHMMER Profile Search",
    category="gene_annotation",
    input_class=PyHmmsearchInput,
    config_class=PyHmmsearchConfig,
    output_class=PyHmmsearchOutput,
    description="Search HMM profile(s) against sequences using PyHMMER",
    example_input=example_input,
    cacheable=True,
)
def run_pyhmmer_hmmsearch(inputs: PyHmmsearchInput, config: PyHmmsearchConfig | None = None, instance=None) -> PyHmmsearchOutput:
    """Search HMM profile(s) against protein sequences using PyHMMER.

    This function implements the hmmsearch algorithm, searching one or more HMM
    profiles against protein sequences to identify sequences that match the
    profile(s). This is useful for finding proteins belonging to specific families
    or containing particular domains.

    Args:
        inputs (PyHmmsearchInput): Validated PyHMMER hmmsearch input containing
            the HMM file path and target sequences.
        config (PyHmmsearchConfig | None): Validated PyHMMER configuration with search
            parameters including E-value thresholds and threading options.

    Returns:
        PyHmmsearchOutput: Structured output containing:
            - ``sequence_hits_df``: DataFrame with sequence-level hits
            - ``domain_hits_df``: DataFrame with domain-level hits
            - ``num_sequence_hits``: Total number of sequence hits
            - ``num_domain_hits``: Total number of domain hits

    Raises:
        FileNotFoundError: If the HMM file cannot be found.
        ValueError: If sequences are empty or invalid, or if HMM file is malformed.
        RuntimeError: If PyHMMER search execution fails.

    Examples:
        >>> # Search a kinase HMM against protein sequences
        >>> inputs = PyHmmsearchInput(
        ...     hmm="/path/to/kinase.hmm",
        ...     sequences=["MVLSPADKTN", "ATCGATCGAT"]
        ... )
        >>> config = PyHmmsearchConfig(
        ...     evalue_threshold=0.001,
        ...     domain_evalue_threshold=0.001
        ... )
        >>> result = run_pyhmmer_hmmsearch(inputs, config)
        >>> print(f"Found {result.num_sequence_hits} sequence hits")
        >>>
        >>> # Filter for high-scoring domains
        >>> if result.domain_hits_df is not None:
        ...     high_score = result.domain_hits_df[
        ...         result.domain_hits_df['domain_score'] > 50
        ...     ]
    """

    output_data = ToolInstance.dispatch(
        "pyhmmer",
        {
            "device": "cpu",
            "operation": "hmmsearch",
            "hmm_path": str(inputs.hmm),
            "sequences": inputs.sequences,
            "num_threads": config.num_threads,
            "evalue_threshold": config.evalue_threshold,
            "score_threshold": config.score_threshold,
            "domain_evalue_threshold": config.domain_evalue_threshold,
            "domain_score_threshold": config.domain_score_threshold,
        },
        instance=instance,
        config=config,
    )

    # Convert results to DataFrames
    sequence_hits_df, domain_hits_df = _build_dataframes(
        output_data["sequence_hits"], output_data["domain_hits"]
    )

    return PyHmmsearchOutput(
        metadata={
            "num_hmms": output_data.get("num_hmms", 0),
            "num_sequences": output_data.get("num_sequences", 0),
            "num_threads": config.num_threads,
            "evalue_threshold": config.evalue_threshold,
            "score_threshold": config.score_threshold,
            "domain_evalue_threshold": config.domain_evalue_threshold,
            "domain_score_threshold": config.domain_score_threshold,
        },
        sequence_hits_df=sequence_hits_df,
        domain_hits_df=domain_hits_df,
    )
