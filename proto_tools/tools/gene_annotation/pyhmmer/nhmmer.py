"""proto_tools/tools/gene_annotation/pyhmmer/nhmmer.py.

PyHMMER nhmmer tool: search nucleotide sequences against nucleotide sequences.
"""

from typing import Any, Literal

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
class PyNhmmerInput(PyHmmerInput):
    """Input object for PyHMMER nhmmer (nucleotide sequences vs nucleotide sequences).

    Attributes:
        sequences (list[str]): Query nucleotide sequences.
            Inherited from ``PyHmmerInput``. Can be a single sequence string or
            a list of sequence strings.

        target_sequences (list[str]): Target nucleotide sequences to
            search against. Can be a single sequence string or a list of sequence
            strings.
    """

    target_sequences: list[str] = InputField(
        description="Target nucleotide sequences as: single sequence string or list of sequence strings"
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


# Output:
PyNhmmerOutput = PyHmmerOutput


# Config:
class PyNhmmerConfig(PyHmmerConfig):
    """Configuration for PyHMMER nhmmer (long-targets nucleotide search).

    Adds ``strand`` — nhmmer is the only HMMER tool that runs on nucleotide
    targets, so it is the only place where forward / reverse-complement strand
    selection makes sense. All other knobs are inherited from
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
        strand (Literal['both', 'watson', 'crick']): Strand to search.
            ``both`` (default) runs the forward strand and its reverse complement;
            ``watson`` runs only the forward strand; ``crick``
            runs only the reverse complement.
    """

    strand: Literal["both", "watson", "crick"] = ConfigField(
        title="Strand",
        default="both",
        description="Strand: 'both' (~2x runtime), 'watson' (forward only), 'crick' (reverse only)",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return PyNhmmerInput(
        sequences=["ATCGATCG"],
        target_sequences=["ATCGATCG", "GCTAGCTA"],
    )


@tool(
    key="pyhmmer-nhmmer",
    label="PyHMMER NHMMER Search",
    category="gene_annotation",
    input_class=PyNhmmerInput,
    config_class=PyNhmmerConfig,
    output_class=PyNhmmerOutput,
    description="Search nucleotide sequences against nucleotide database using PyHMMER",
    example_input=example_input,
    cacheable=True,
)
def run_pyhmmer_nhmmer(inputs: PyNhmmerInput, config: PyNhmmerConfig, instance: Any = None) -> PyNhmmerOutput:
    """Search nucleotide sequences against nucleotide database using PyHMMER.

    Args:
        inputs (PyNhmmerInput): Validated PyHMMER nhmmer input containing both
            query and target nucleotide sequences.
        config (PyNhmmerConfig): Validated PyHMMER configuration with search
            parameters including E-value thresholds and threading options.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        PyNhmmerOutput: Structured output with sequence-level and domain-level hits.
    """
    output_data = ToolInstance.dispatch(
        "pyhmmer",
        {
            "device": "cpu",
            "operation": "nhmmer",
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
            "strand": config.strand,
            "seed": config.seed,
        },
        instance=instance,
        config=config,
    )

    sequence_hits, domain_hits = _build_hit_models(output_data["sequence_hits"], output_data["domain_hits"])

    return PyNhmmerOutput(
        metadata={
            "num_query_sequences": output_data.get("num_query_sequences", 0),
            "num_target_sequences": output_data.get("num_target_sequences", 0),
            "evalue_threshold": config.evalue_threshold,
            "domain_evalue_threshold": config.domain_evalue_threshold,
            "strand": config.strand,
        },
        sequence_hits=sequence_hits,
        domain_hits=domain_hits,
    )
