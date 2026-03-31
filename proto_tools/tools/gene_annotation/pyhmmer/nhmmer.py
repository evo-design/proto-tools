"""proto_tools/tools/gene_annotation/pyhmmer/nhmmer.py

PyHMMER nhmmer tool — search nucleotide sequences against nucleotide sequences."""
from __future__ import annotations

from typing import List

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

    target_sequences: List[str] = InputField(
        description="Target nucleotide sequences as: single sequence string or list of sequence strings"
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


# Output:
PyNhmmerOutput = PyHmmerOutput

# Config:
PyNhmmerConfig = PyHmmerConfig


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input():
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
def run_pyhmmer_nhmmer(inputs: PyNhmmerInput, config: PyNhmmerConfig | None = None, instance=None) -> PyNhmmerOutput:
    """Search nucleotide sequences against nucleotide database using PyHMMER.

    Args:
        inputs (PyNhmmerInput): Validated PyHMMER nhmmer input containing both
            query and target nucleotide sequences.
        config (PyNhmmerConfig | None): Validated PyHMMER configuration with search
            parameters including E-value thresholds and threading options.

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
        },
        instance=instance,
        config=config,
    )

    sequence_hits_df, domain_hits_df = _build_dataframes(
        output_data["sequence_hits"], output_data["domain_hits"]
    )

    return PyNhmmerOutput(
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
