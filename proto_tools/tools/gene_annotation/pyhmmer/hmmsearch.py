"""proto_tools/tools/gene_annotation/pyhmmer/hmmsearch.py.

PyHMMER hmmsearch tool: search HMM profiles against protein sequences.
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

    hmm: str | Path = InputField(title="HMM File", description="Path to HMM file")


# Output:
PyHmmsearchOutput = PyHmmerOutput


# Config:
class PyHmmsearchConfig(PyHmmerConfig):
    """Configuration for PyHMMER hmmsearch.

    Adds ``bit_cutoffs`` — only meaningful for hmmsearch and hmmscan, which
    consume a pre-built HMM file that may carry curated GA/NC/TC cutoffs
    (Pfam HMMs always do). All other knobs are inherited from
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
        """Reads a local HMM file (``hmm``) that can't be staged to the hosted cloud."""
        return "needs a local HMM file (hmm) that can't be staged to device='cloud'. Run locally with device='cpu'."


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
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
def run_pyhmmer_hmmsearch(
    inputs: PyHmmsearchInput, config: PyHmmsearchConfig, instance: Any = None
) -> PyHmmsearchOutput:
    """Search HMM profile(s) against protein sequences using PyHMMER.

    This function implements the hmmsearch algorithm, searching one or more HMM
    profiles against protein sequences to identify sequences that match the
    profile(s). This is useful for finding proteins belonging to specific families
    or containing particular domains.

    Args:
        inputs (PyHmmsearchInput): Validated PyHMMER hmmsearch input containing
            the HMM file path and target sequences.
        config (PyHmmsearchConfig): Validated PyHMMER configuration with search
            parameters including E-value thresholds and threading options.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        PyHmmsearchOutput: Structured output containing:
            - ``sequence_hits``: List of sequence-level hits
            - ``domain_hits``: List of domain-level hits
            - ``num_sequence_hits``: Total number of sequence hits
            - ``num_domain_hits``: Total number of domain hits

    Raises:
        FileNotFoundError: If the HMM file cannot be found.
        ValueError: If sequences are empty or invalid, or if HMM file is malformed.
        RuntimeError: If PyHMMER search execution fails.

    Examples:
        >>> # Search a kinase HMM against protein sequences
        >>> inputs = PyHmmsearchInput(hmm="/path/to/kinase.hmm", sequences=["MVLSPADKTN", "ATCGATCGAT"])
        >>> config = PyHmmsearchConfig(evalue_threshold=0.001, domain_evalue_threshold=0.001)
        >>> result = run_pyhmmer_hmmsearch(inputs, config)
        >>> print(f"Found {result.num_sequence_hits} sequence hits")
        >>>
        >>> # Filter for high-scoring domains
        >>> if result.domain_hits:
        ...     high_score = [hit for hit in result.domain_hits if hit.domain_score > 50]
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

    return PyHmmsearchOutput(
        metadata={
            "num_hmms": output_data.get("num_hmms", 0),
            "num_sequences": output_data.get("num_sequences", 0),
            "evalue_threshold": config.evalue_threshold,
            "domain_evalue_threshold": config.domain_evalue_threshold,
            "bit_cutoffs": config.bit_cutoffs,
        },
        sequence_hits=sequence_hits,
        domain_hits=domain_hits,
    )
