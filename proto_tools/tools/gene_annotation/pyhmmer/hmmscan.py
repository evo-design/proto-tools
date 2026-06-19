"""proto_tools/tools/gene_annotation/pyhmmer/hmmscan.py.

PyHMMER hmmscan tool: search protein sequences against an HMM database.
"""

from pathlib import Path
from typing import Any, Literal

from proto_tools.tools.gene_annotation.pyhmmer.shared_data_models import (
    PyHmmerConfig,
    PyHmmerInput,
    PyHmmerOutput,
    _build_hit_models,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField, InputField, ToolInstance


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

    hmm_db: str | Path = InputField(title="HMM Database", description="Path to HMM database file")


# Output:
PyHmmscanOutput = PyHmmerOutput


# Config:
class PyHmmscanConfig(PyHmmerConfig):
    """Configuration for PyHMMER hmmscan.

    Adds ``bit_cutoffs`` — only meaningful for hmmsearch and hmmscan, which
    consume a pre-built HMM file that may carry curated GA/NC/TC cutoffs
    (Pfam HMMs always do). All other settings are inherited from
    :class:`PyHmmerConfig`.

    Attributes:
        num_threads (int): CPU threads (0 = auto). Inherited from ``PyHmmerConfig``.
        evalue_threshold (float): Sequence-level E-value cap to report.
            Inherited from ``PyHmmerConfig``.
        score_threshold (float | None): Sequence-level bit-score floor.
            Inherited from ``PyHmmerConfig``.
        domain_evalue_threshold (float): Per-domain E-value cap to report. Inherited from ``PyHmmerConfig``.
        domain_score_threshold (float | None): Per-domain bit-score floor. Inherited from ``PyHmmerConfig``.
        inclusion_evalue_threshold (float): Sequence-level E-value cap for
            inclusion. Inherited from ``PyHmmerConfig``.
        inclusion_domain_evalue_threshold (float): Per-domain E-value cap for
            inclusion. Inherited from ``PyHmmerConfig``.
        z_value (float | None): Effective database size.
            Inherited from ``PyHmmerConfig``.
        domain_z_value (float | None): Significant hit count.
            Inherited from ``PyHmmerConfig``.
        skip_filters (bool): Disable MSV/Vit/Fwd filters.
            Inherited from ``PyHmmerConfig``.
        bit_cutoffs (Literal['gathering', 'noise', 'trusted'] | None):
            Use the HMM's stored bit-score cutoff in place of E-value reporting.
            ``gathering`` is the Pfam-curated default for inclusion;
            ``noise`` is the most permissive; ``trusted``
            is the strictest. None = use E-value/score thresholds. Default: None.
            Pyhmmer raises ``MissingCutoffs`` if the HMM file lacks the requested
            cutoff line — set None for HMMs without curated thresholds.
    """

    bit_cutoffs: Literal["gathering", "noise", "trusted"] | None = ConfigField(
        title="HMM Bit-Score Cutoffs",
        default=None,
        description="HMM curated cutoff: 'gathering' (Pfam GA), 'noise' (permissive), 'trusted' (strictest)",
    )

    def cloud_unsupported_reason(self) -> str | None:
        """Reads a local HMM database file (``hmm_db``) that can't be staged to the hosted cloud."""
        return "needs a local HMM database file (hmm_db) that can't be staged to device='cloud'. Run locally with device='cpu'."


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
def run_pyhmmer_hmmscan(inputs: PyHmmscanInput, config: PyHmmscanConfig, instance: Any = None) -> PyHmmscanOutput:
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
            "num_threads": config.num_threads,
            "evalue_threshold": config.evalue_threshold,
            "score_threshold": config.score_threshold,
            "domain_evalue_threshold": config.domain_evalue_threshold,
            "domain_score_threshold": config.domain_score_threshold,
            "inclusion_evalue_threshold": config.inclusion_evalue_threshold,
            "inclusion_domain_evalue_threshold": config.inclusion_domain_evalue_threshold,
            "z_value": config.z_value,
            "domain_z_value": config.domain_z_value,
            "skip_filters": config.skip_filters,
            "bit_cutoffs": config.bit_cutoffs,
            "seed": config.seed,
        },
        instance=instance,
        config=config,
    )

    sequence_hits, domain_hits = _build_hit_models(output_data["sequence_hits"], output_data["domain_hits"])

    return PyHmmscanOutput(
        metadata={
            "num_hmms": output_data.get("num_hmms", 0),
            "num_queries": output_data.get("num_queries", 0),
            "evalue_threshold": config.evalue_threshold,
            "domain_evalue_threshold": config.domain_evalue_threshold,
            "bit_cutoffs": config.bit_cutoffs,
        },
        sequence_hits=sequence_hits,
        domain_hits=domain_hits,
    )
