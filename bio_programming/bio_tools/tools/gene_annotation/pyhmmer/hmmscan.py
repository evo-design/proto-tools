"""PyHMMER hmmscan tool — search protein sequences against an HMM database."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator

from bio_programming.bio_tools.tools.infra.tool_cache import tool_cache
from bio_programming.bio_tools.tools.tool_registry import tool

from .shared_data_models import PyHmmerConfig, PyHmmerInput, PyHmmerOutput, _build_dataframes


# ============================================================================
# Data Models
# ============================================================================
# Input:
class PyHmmscanInput(PyHmmerInput):
    """Input object for PyHMMER hmmscan (protein sequences vs HMM database).

    This class defines the input parameters for searching protein sequences against
    an HMM database to identify domains and protein families within the sequences.

    Attributes:
        sequences (List[str]): Query protein sequences to search.
            Inherited from ``PyHmmerInput``. Can be a single sequence string or
            a list of sequence strings.

        hmm_db (str | Path): Path to an HMM database file containing
            multiple profile HMMs. The file should be in HMMER3 format and typically
            represents a comprehensive database like Pfam. All HMMs in the database
            will be searched against the query sequences.
    """

    hmm_db: str | Path = Field(description="Path to HMM database file")

    @field_validator("hmm_db", mode="before")
    @classmethod
    def validate_hmm_db_file(cls, value: str | Path) -> str:
        """Validate HMM database file exists and return path as string."""
        hmm_db_path = Path(value)
        if not hmm_db_path.exists():
            raise ValueError(f"HMM database file not found: {value}")
        if not hmm_db_path.is_file():
            raise ValueError(f"HMM database path is not a file: {value}")
        return str(hmm_db_path)


# Output:
PyHmmscanOutput = PyHmmerOutput

# Config:
PyHmmscanConfig = PyHmmerConfig


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="pyhmmer-hmmscan",
    label="PyHMMER Scan",
    input=PyHmmscanInput,
    config=PyHmmscanConfig,
    output=PyHmmscanOutput,
    description="Search sequences against HMM database using PyHMMER",
)
@tool_cache("pyhmmer-hmmscan")
def run_pyhmmer_hmmscan(inputs: PyHmmscanInput, config: PyHmmscanConfig) -> PyHmmscanOutput:
    """Search protein sequences against HMM database using PyHMMER.

    This function implements the hmmscan algorithm, searching protein sequences
    against an HMM database to identify domains and protein families within the
    query sequences. This is the reverse of hmmsearch and is useful for annotating
    proteins with known domain architectures.

    Args:
        inputs (PyHmmscanInput): Validated PyHMMER hmmscan input containing
            the HMM database path and query sequences.
        config (PyHmmscanConfig): Validated PyHMMER configuration with search
            parameters including E-value thresholds and threading options.

    Returns:
        PyHmmscanOutput: Structured output containing:
            - ``sequence_hits_df``: DataFrame with sequence-level hits
            - ``domain_hits_df``: DataFrame with domain-level hits
            - ``num_sequence_hits``: Total number of sequence hits
            - ``num_domain_hits``: Total number of domain hits

    Raises:
        FileNotFoundError: If the HMM database file cannot be found.
        ValueError: If sequences are empty or invalid, or if HMM database is malformed.
        RuntimeError: If PyHMMER search execution fails.

    Examples:
        >>> # Scan proteins against Pfam database
        >>> inputs = PyHmmscanInput(
        ...     hmm_db="/path/to/Pfam-A.hmm",
        ...     sequences=["MVLSPADKTN", "ATCGATCGAT"]
        ... )
        >>> config = PyHmmscanConfig(
        ...     evalue_threshold=1.0,
        ...     domain_evalue_threshold=1.0
        ... )
        >>> result = run_pyhmmer_hmmscan(inputs, config)
        >>> print(f"Found {result.num_domain_hits} domains")
        >>>
        >>> # Get domain architecture for each sequence
        >>> if result.domain_hits_df is not None:
        ...     for seq_name in result.domain_hits_df['target_name'].unique():
        ...         domains = result.domain_hits_df[
        ...             result.domain_hits_df['target_name'] == seq_name
        ...         ]['query_name'].tolist()
        ...         print(f"{seq_name}: {' + '.join(domains)}")
    """
    from bio_programming.bio_tools.tools.infra.env_manager import EnvManager

    venv_manager = EnvManager(model_name="pyhmmer")

    input_data = {
        "operation": "hmmscan",
        "hmm_db_path": str(inputs.hmm_db),
        "sequences": inputs.sequences,
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

    return PyHmmscanOutput(
        metadata={
            "num_hmms": output_data.get("num_hmms", 0),
            "num_queries": output_data.get("num_queries", 0),
            "num_threads": config.num_threads,
            "evalue_threshold": config.evalue_threshold,
            "score_threshold": config.score_threshold,
            "domain_evalue_threshold": config.domain_evalue_threshold,
            "domain_score_threshold": config.domain_score_threshold,
        },
        sequence_hits_df=sequence_hits_df,
        domain_hits_df=domain_hits_df,
    )
