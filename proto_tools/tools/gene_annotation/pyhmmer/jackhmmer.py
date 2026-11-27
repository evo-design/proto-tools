"""proto_tools/tools/gene_annotation/pyhmmer/jackhmmer.py.

PyHMMER jackhmmer tool: iterative protein sequence search.
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
from proto_tools.utils import ConfigField, InputField, ToolInstance


# ============================================================================
# Data Models
# ============================================================================
# Input:
class PyJackhmmerInput(PyHmmerInput):
    """Input object for PyHMMER jackhmmer (protein sequences vs protein sequences).

    Attributes:
        sequences (list[str]): Query protein sequences.
            Inherited from ``PyHmmerInput``. Can be a single sequence string or
            a list of sequence strings.

        target_sequences (list[str]): Target protein sequences to
            search against. Can be a single sequence string or a list of sequence
            strings.
    """

    target_sequences: list[str] = InputField(
        title="Target Sequences",
        description="Target sequences as: single sequence string or list of sequence strings",
    )

    @field_validator("target_sequences", mode="before")
    @classmethod
    def normalize_target_sequences(cls, value: Any) -> list[str]:
        """Normalize target sequences to list of strings."""
        return PyHmmerInput.normalize_sequences(value)

    @field_validator("target_sequences")
    @classmethod
    def validate_target_sequences(cls, sequences: list[str]) -> list[str]:
        """Validate target sequences are non-empty."""
        return PyHmmerInput.validate_sequences(sequences)


class PyJackhmmerConfig(PyHmmerConfig):
    """Configuration for PyHMMER jackhmmer search.

    Inherits all reporting + inclusion + filter settings from :class:`PyHmmerConfig`.
    Adds ``max_iterations`` — the *defining* parameter for iterative search; surfaced
    as a primary parameter rather than buried in the advanced section.

    Attributes:
        num_threads (int): CPU threads (0 = auto). Inherited from ``PyHmmerConfig``.
        evalue_threshold (float): Sequence-level E-value cap to report.
            Inherited from ``PyHmmerConfig``.
        score_threshold (float | None): Sequence-level bit-score floor.
            Inherited from ``PyHmmerConfig``.
        domain_evalue_threshold (float): Per-domain E-value cap to report.
            Inherited from ``PyHmmerConfig``.
        domain_score_threshold (float | None): Per-domain bit-score floor.
            Inherited from ``PyHmmerConfig``.
        inclusion_evalue_threshold (float): Sequence-level inclusion E-value.
            Inherited from ``PyHmmerConfig``. Critical for jackhmmer — the
            included set seeds the next iteration's HMM.
        inclusion_domain_evalue_threshold (float): Per-domain inclusion E-value.
            Inherited from ``PyHmmerConfig``.
        z_value (float | None): Effective database size for E-value calc.
            Inherited from ``PyHmmerConfig``.
        domain_z_value (float | None): Significant hit count for domain E-value.
            Inherited from ``PyHmmerConfig``.
        skip_filters (bool): Disable MSV/Vit/Fwd filters.
            Inherited from ``PyHmmerConfig``.
        max_iterations (int): Maximum jackhmmer iterations; stops early on
            convergence. Default 5.
    """

    max_iterations: int = ConfigField(
        title="Maximum Iterations",
        default=5,
        ge=1,
        description="Max jackhmmer iterations; raise to 8-10 for distant homologs, 1-2 for fast preview",
    )


# Output:
PyJackhmmerOutput = PyHmmerOutput


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return PyJackhmmerInput(
        sequences=["MKTL"],
        target_sequences=["MKTL", "ARND"],
    )


@tool(
    key="pyhmmer-jackhmmer",
    label="PyHMMER JackHMMER Search",
    category="gene_annotation",
    input_class=PyJackhmmerInput,
    config_class=PyJackhmmerConfig,
    output_class=PyJackhmmerOutput,
    description="Iteratively search protein sequences against protein database using PyHMMER",
    example_input=example_input,
    cacheable=True,
)
def run_pyhmmer_jackhmmer(
    inputs: PyJackhmmerInput,
    config: PyJackhmmerConfig,
    instance: Any = None,
) -> PyJackhmmerOutput:
    """Iteratively search protein sequences against protein database using PyHMMER.

    Args:
        inputs (PyJackhmmerInput): Validated jackhmmer input containing query and
            target protein sequences.
        config (PyJackhmmerConfig): Validated configuration including
            ``max_iterations`` and threshold settings.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        PyJackhmmerOutput: Structured output with sequence-level and domain-level hits.
    """
    output_data = ToolInstance.dispatch(
        "pyhmmer",
        {
            "device": "cpu",
            "operation": "jackhmmer",
            "sequences": inputs.sequences,
            "target_sequences": inputs.target_sequences,
            "max_iterations": config.max_iterations,
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

    return PyJackhmmerOutput(
        metadata={
            "num_query_sequences": output_data.get("num_query_sequences", 0),
            "num_target_sequences": output_data.get("num_target_sequences", 0),
            "max_iterations": config.max_iterations,
            "iterations_per_query": output_data.get("iterations_per_query", []),
            "converged_per_query": output_data.get("converged_per_query", []),
            "evalue_threshold": config.evalue_threshold,
            "domain_evalue_threshold": config.domain_evalue_threshold,
        },
        sequence_hits=sequence_hits,
        domain_hits=domain_hits,
    )
