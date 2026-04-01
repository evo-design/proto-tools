"""proto_tools/tools/gene_annotation/pyhmmer/hmmscan.py.

PyHMMER hmmscan tool: search protein sequences against an HMM database.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import field_validator

from proto_tools.tools.gene_annotation.pyhmmer.shared_data_models import (
    PyHmmerConfig,
    PyHmmerInput,
    PyHmmerOutput,
    _build_hit_models,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import InputField, ToolInstance


# ============================================================================
# Data Models
# ============================================================================
# Input:
class PyHmmscanInput(PyHmmerInput):
    """Input object for PyHMMER hmmscan (protein sequences vs HMM database).

    This class defines the input parameters for searching protein sequences against
    an HMM database to identify domains and protein families within the sequences.

    Attributes:
        sequences (list[str]): Query protein sequences to search.
            Inherited from ``PyHmmerInput``. Can be a single sequence string or
            a list of sequence strings.

        hmm_db (str | Path): Path to an HMM database file containing
            multiple profile HMMs. The file should be in HMMER3 format and typically
            represents a comprehensive database like Pfam. All HMMs in the database
            will be searched against the query sequences.
    """

    hmm_db: str | Path = InputField(description="Path to HMM database file")

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
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return PyHmmscanInput(
        sequences=["MKTL"],
        hmm_db=str(Path(__file__).parent / "examples" / "example.hmm"),
    )


@tool(
    key="pyhmmer-hmmscan",
    label="PyHMMER Scan",
    category="gene_annotation",
    input_class=PyHmmscanInput,
    config_class=PyHmmscanConfig,
    output_class=PyHmmscanOutput,
    description="Search sequences against HMM database using PyHMMER",
    example_input=example_input,
    cacheable=True,
)
def run_pyhmmer_hmmscan(
    inputs: PyHmmscanInput, config: PyHmmscanConfig | None = None, instance: Any = None
) -> PyHmmscanOutput:
    """Search protein sequences against HMM database using PyHMMER.

    This function implements the hmmscan algorithm, searching protein sequences
    against an HMM database to identify domains and protein families within the
    query sequences. This is the reverse of hmmsearch and is useful for annotating
    proteins with known domain architectures.

    Args:
        inputs (PyHmmscanInput): Validated PyHMMER hmmscan input containing
            the HMM database path and query sequences.
        config (PyHmmscanConfig | None): Validated PyHMMER configuration with search
            parameters including E-value thresholds and threading options.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        PyHmmscanOutput: Structured output containing:
            - ``sequence_hits``: List of sequence-level hits
            - ``domain_hits``: List of domain-level hits
            - ``num_sequence_hits``: Total number of sequence hits
            - ``num_domain_hits``: Total number of domain hits

    Raises:
        FileNotFoundError: If the HMM database file cannot be found.
        ValueError: If sequences are empty or invalid, or if HMM database is malformed.
        RuntimeError: If PyHMMER search execution fails.

    Examples:
        >>> # Scan proteins against Pfam database
        >>> inputs = PyHmmscanInput(hmm_db="/path/to/Pfam-A.hmm", sequences=["MVLSPADKTN", "ATCGATCGAT"])
        >>> config = PyHmmscanConfig(evalue_threshold=1.0, domain_evalue_threshold=1.0)
        >>> result = run_pyhmmer_hmmscan(inputs, config)
        >>> print(f"Found {result.num_domain_hits} domains")
        >>>
        >>> # Get domain architecture for each sequence
        >>> if result.domain_hits:
        ...     from itertools import groupby
        ...
        ...     for seq_name, hits in groupby(result.domain_hits, key=lambda h: h.target_name):
        ...         domains = [h.query_name for h in hits]
        ...         print(f"{seq_name}: {' + '.join(domains)}")
    """
    output_data = ToolInstance.dispatch(
        "pyhmmer",
        {
            "device": "cpu",
            "operation": "hmmscan",
            "hmm_db_path": str(inputs.hmm_db),
            "sequences": inputs.sequences,
            "num_threads": config.num_threads,  # type: ignore[union-attr]
            "evalue_threshold": config.evalue_threshold,  # type: ignore[union-attr]
            "score_threshold": config.score_threshold,  # type: ignore[union-attr]
            "domain_evalue_threshold": config.domain_evalue_threshold,  # type: ignore[union-attr]
            "domain_score_threshold": config.domain_score_threshold,  # type: ignore[union-attr]
        },
        instance=instance,
        config=config,
    )

    # Convert results to typed hit models
    sequence_hits, domain_hits = _build_hit_models(output_data["sequence_hits"], output_data["domain_hits"])

    return PyHmmscanOutput(
        metadata={
            "num_hmms": output_data.get("num_hmms", 0),
            "num_queries": output_data.get("num_queries", 0),
            "num_threads": config.num_threads,  # type: ignore[union-attr]
            "evalue_threshold": config.evalue_threshold,  # type: ignore[union-attr]
            "score_threshold": config.score_threshold,  # type: ignore[union-attr]
            "domain_evalue_threshold": config.domain_evalue_threshold,  # type: ignore[union-attr]
            "domain_score_threshold": config.domain_score_threshold,  # type: ignore[union-attr]
        },
        sequence_hits=sequence_hits,
        domain_hits=domain_hits,
    )
