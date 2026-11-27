"""proto_tools/tools/gene_annotation/pyhmmer/phmmer.py.

PyHMMER phmmer tool: search protein sequences against protein sequences.
"""

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
class PyPhmmerInput(PyHmmerInput):
    """Input object for PyHMMER phmmer (protein sequences vs protein sequences).

    This class defines the input parameters for a single-pass protein search,
    where a temporary HMM is built from each query sequence and searched directly
    against the target sequences without requiring a pre-built HMM profile.

    Attributes:
        sequences (list[str]): Query protein sequences. Inherited from
            ``PyHmmerInput``. Can be a single sequence string or a list of sequence
            strings. These sequences will be used to build temporary HMM profiles
            on-the-fly.

        target_sequences (list[str]): Target protein sequences to
            search against. Can be a single sequence string or a list of sequence
            strings. The query sequences will be compared against these targets.
    """

    target_sequences: list[str] = InputField(
        title="Target Sequences",
        description="Target sequences as: single sequence string or list of sequence strings",
    )

    @field_validator("target_sequences", mode="before")
    @classmethod
    def normalize_target_sequences(cls, value: Any) -> list[str]:
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
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return PyPhmmerInput(
        sequences=["MKTL"],
        target_sequences=["MKTL", "ARND"],
    )


@tool(
    key="pyhmmer-phmmer",
    label="PyHMMER PHMMER Search",
    category="gene_annotation",
    input_class=PyPhmmerInput,
    config_class=PyPhmmerConfig,
    output_class=PyPhmmerOutput,
    description="Search protein sequences against protein database using PyHMMER",
    example_input=example_input,
    cacheable=True,
)
def run_pyhmmer_phmmer(inputs: PyPhmmerInput, config: PyPhmmerConfig, instance: Any = None) -> PyPhmmerOutput:
    """Search protein sequences against protein database using PyHMMER.

    This function implements the phmmer algorithm, a single-pass protein-protein
    search that builds a temporary HMM profile from each query sequence on-the-fly.
    This is useful for finding homologous sequences without requiring pre-built
    HMM profiles.

    Args:
        inputs (PyPhmmerInput): Validated PyHMMER phmmer input containing both
            query and target protein sequences.
        config (PyPhmmerConfig): Validated PyHMMER configuration with search
            parameters including E-value thresholds and threading options.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        PyPhmmerOutput: Structured output containing:
            - ``sequence_hits``: List of sequence-level hits
            - ``domain_hits``: List of domain-level hits
            - ``num_sequence_hits``: Total number of sequence hits
            - ``num_domain_hits``: Total number of domain hits

    Raises:
        ValueError: If query or target sequences are empty or invalid.
        RuntimeError: If PyHMMER search execution fails.

    Examples:
        >>> # Search for sequences similar to the query proteins
        >>> inputs = PyPhmmerInput(
        ...     sequences=["MVLSPADKTNVKAAW"], target_sequences=["MVLSPADKTN", "ATCGATCGAT", "MVLSPADKTNVK"]
        ... )
        >>> config = PyPhmmerConfig(evalue_threshold=1.0, domain_evalue_threshold=1.0)
        >>> result = run_pyhmmer_phmmer(inputs, config)
        >>> print(f"Found {result.num_sequence_hits} similar sequences")
        >>>
        >>> # Keep only highly significant hits
        >>> if result.sequence_hits:
        ...     significant = [hit for hit in result.sequence_hits if hit.evalue < 1e-10]
    """
    output_data = ToolInstance.dispatch(
        "pyhmmer",
        {
            "device": "cpu",
            "operation": "phmmer",
            "sequences": inputs.sequences,
            "target_sequences": inputs.target_sequences,
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
            "seed": config.seed,
        },
        instance=instance,
        config=config,
    )

    sequence_hits, domain_hits = _build_hit_models(output_data["sequence_hits"], output_data["domain_hits"])

    return PyPhmmerOutput(
        metadata={
            "num_query_sequences": output_data.get("num_query_sequences", 0),
            "num_target_sequences": output_data.get("num_target_sequences", 0),
            "evalue_threshold": config.evalue_threshold,
            "domain_evalue_threshold": config.domain_evalue_threshold,
        },
        sequence_hits=sequence_hits,
        domain_hits=domain_hits,
    )
