"""proto_tools/tools/gene_annotation/pyhmmer/shared_data_models.py.

Shared data models and helpers for PyHMMER tools.
"""

import warnings
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
)

# ============================================================================
# Data Models
# ============================================================================


# Hit Models:
class SequenceHit(BaseModel):
    """A single sequence-level hit from a PyHMMER search.

    Attributes:
        query_name (str): Name of the query HMM.
        query_accession (str): Accession of the query HMM. ``"-"`` if unavailable.
        query_description (str): Description of the query HMM. ``"-"`` if unavailable.
        query_idx (int): Index of the query (0-indexed).
        target_name (str): Name of the target sequence.
        target_accession (str): Accession of the target sequence. ``"-"`` if unavailable.
        target_description (str): Description of the target sequence. ``"-"`` if unavailable.
        evalue (float): E-value of the hit.
        score (float): Bit score of the full sequence.
        bias (float): Bias correction for the sequence score.
        sum_score (float): Sum of domain scores.
        reported (bool): Whether the hit passes reporting thresholds.
        included (bool): Whether the hit passes inclusion thresholds.
        pvalue (float): P-value of the hit.
        num_domains (int): Number of domains found in the hit.
    """

    query_name: str = Field(description="Name of the query HMM")
    query_accession: str = Field(description="Accession of the query HMM. If not available, set to '-'.")
    query_description: str = Field(description="Description of the query HMM. If not available, set to '-'.")
    query_idx: int = Field(description="Index of the query (0-indexed)")
    target_name: str = Field(description="Name of the target sequence")
    target_accession: str = Field(description="Accession of the target sequence. If not available, set to '-'.")
    target_description: str = Field(description="Description of the target sequence. If not available, set to '-'.")
    evalue: float = Field(description="E-value of the hit")
    score: float = Field(description="Score of the full sequence")
    bias: float = Field(description="Bias of the full sequence")
    sum_score: float = Field(description="Sum score of the full sequence")
    reported: bool = Field(description="Whether the hit is reported")
    included: bool = Field(description="Whether the hit is included")
    pvalue: float = Field(description="p-value of the hit")
    num_domains: int = Field(description="Number of domains in the hit")


class DomainHit(BaseModel):
    """A single domain-level hit from a PyHMMER search.

    Attributes:
        query_name (str): Name of the query HMM.
        query_accession (str): Accession of the query HMM. ``"-"`` if unavailable.
        query_description (str): Description of the query HMM. ``"-"`` if unavailable.
        query_idx (int): Index of the query (0-indexed).
        target_name (str): Name of the target sequence.
        target_accession (str): Accession of the target sequence. ``"-"`` if unavailable.
        target_description (str): Description of the target sequence. ``"-"`` if unavailable.
        hmm_length (int): Length of the HMM profile.
        hmm_from (int): Start position of the domain match in the HMM (1-indexed).
        hmm_to (int): End position of the domain match in the HMM (1-indexed).
        target_from (int): Start position of the domain match in the target (1-indexed).
        target_to (int): End position of the domain match in the target (1-indexed).
        target_length (int): Length of the target sequence.
        c_evalue (float): Conditional E-value of the domain.
        i_evalue (float): Independent E-value of the domain.
        domain_score (float): Bit score of the domain.
        domain_bias (float): Bias correction of the domain score.
        domain_idx (int): Index of the domain within the hit (0-indexed).
        env_from (int): Envelope start position in the target (1-indexed).
        env_to (int): Envelope end position in the target (1-indexed).
        envelope_score (float): Bit score of the domain envelope.
        domain_included (bool): Whether the domain passes inclusion thresholds.
        domain_reported (bool): Whether the domain passes reporting thresholds.
        domain_pvalue (float): P-value of the domain.
    """

    query_name: str = Field(description="Name of the query HMM")
    query_accession: str = Field(description="Accession of the query HMM. If not available, set to '-'.")
    query_description: str = Field(description="Description of the query HMM. If not available, set to '-'.")
    query_idx: int = Field(description="Index of the query (0-indexed)")
    target_name: str = Field(description="Name of the target sequence")
    target_accession: str = Field(description="Accession of the target sequence. If not available, set to '-'.")
    target_description: str = Field(description="Description of the target sequence. If not available, set to '-'.")
    hmm_length: int = Field(description="Length of the HMM profile")
    hmm_from: int = Field(description="Start position of the domain match in the HMM (1-indexed)")
    hmm_to: int = Field(description="End position of the domain match in the HMM (1-indexed)")
    target_from: int = Field(description="Start position of the domain match in the target sequence (1-indexed)")
    target_to: int = Field(description="End position of the domain match in the target sequence (1-indexed)")
    target_length: int = Field(description="Length of the target sequence")
    c_evalue: float = Field(description="Conditional E-value of the domain")
    i_evalue: float = Field(description="Independent E-value of the domain")
    domain_score: float = Field(description="Bit score of the domain")
    domain_bias: float = Field(description="Bias correction of the domain score")
    domain_idx: int = Field(description="Index of the domain within the hit (0-indexed)")
    env_from: int = Field(description="Start position of the domain envelope in the target sequence (1-indexed)")
    env_to: int = Field(description="End position of the domain envelope in the target sequence (1-indexed)")
    envelope_score: float = Field(description="Bit score of the domain envelope")
    domain_included: bool = Field(description="Whether the domain passes inclusion thresholds")
    domain_reported: bool = Field(description="Whether the domain passes reporting thresholds")
    domain_pvalue: float = Field(description="P-value of the domain")


# Input:
class PyHmmerInput(BaseToolInput):
    """Base input object for PyHMMER tools.

    Provides common input fields and validation for protein sequences
    used across all PyHMMER search tools (hmmsearch, hmmscan, phmmer).

    Attributes:
        sequences (list[str]): Input protein sequences for searching.
            Can be provided as:

            - A single protein sequence string (amino acid sequence)
            - A list of protein sequence strings

            Sequences are automatically normalized to a list format internally.
            All sequences must be non-empty and contain valid characters.
    """

    sequences: list[str] = InputField(
        description="Input sequences as: single sequence string or list of sequence strings"
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value: Any) -> list[str]:
        """Normalize sequences to a list of strings.

        Handles input formats:
        - Single sequence string
        - List of sequence strings
        """
        if isinstance(value, str):
            # Single sequence
            return [value]
        if isinstance(value, list):
            # List of sequences
            return value
        raise ValueError(f"Unsupported sequences input type: {type(value)}")

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, sequences: list[str]) -> list[str]:
        """Validate that sequences are non-empty."""
        if not sequences:
            raise ValueError("At least one sequence is required")

        for i, seq in enumerate(sequences):
            if not seq or not seq.strip():
                raise ValueError(f"Sequence {i + 1} is empty")

        return sequences


# Output:
class PyHmmerOutput(BaseToolOutput):
    """Output from PyHMMER search operations.

    This class encapsulates the results of PyHMMER searches, providing both
    sequence-level and domain-level hits as structured typed lists.

    Attributes:
        sequence_hits (list[SequenceHit]): List of sequence-level hits from
            the search. Each SequenceHit contains:

            - ``query_name``: Name of the query HMM or sequence
            - ``query_accession``: Accession of the query (``"-"`` if unavailable)
            - ``query_description``: Description of the query (``"-"`` if unavailable)
            - ``query_idx``: Index of the query (0-indexed)
            - ``target_name``: Name of the target sequence
            - ``target_accession``: Accession of the target (``"-"`` if unavailable)
            - ``target_description``: Description of the target (``"-"`` if unavailable)
            - ``evalue``: E-value of the hit
            - ``score``: Bit score of the full sequence
            - ``bias``: Bias correction for the sequence score
            - ``sum_score``: Sum of domain scores
            - ``reported``: Whether the hit passes reporting thresholds
            - ``included``: Whether the hit passes inclusion thresholds
            - ``pvalue``: P-value of the hit
            - ``num_domains``: Number of domains found in the hit

            Empty list if no sequence hits are found.

        domain_hits (list[DomainHit]): List of domain-level hits from the search.
            Each DomainHit contains:

            - ``query_name``: Name of the query HMM or sequence
            - ``query_accession``: Accession of the query (``"-"`` if unavailable)
            - ``query_description``: Description of the query (``"-"`` if unavailable)
            - ``query_idx``: Index of the query (0-indexed)
            - ``target_name``: Name of the target sequence
            - ``target_accession``: Accession of the target (``"-"`` if unavailable)
            - ``target_description``: Description of the target (``"-"`` if unavailable)
            - ``hmm_length``: Length of the HMM profile
            - ``hmm_from``: Start position in HMM (1-indexed)
            - ``hmm_to``: End position in HMM (1-indexed)
            - ``target_from``: Start position in target sequence (1-indexed)
            - ``target_to``: End position in target sequence (1-indexed)
            - ``target_length``: Length of the target sequence
            - ``c_evalue``: Conditional E-value of the domain
            - ``i_evalue``: Independent E-value of the domain
            - ``domain_score``: Bit score of the domain
            - ``domain_bias``: Bias correction for the domain score
            - ``domain_idx``: Index of the domain within the hit (0-indexed)
            - ``env_from``: Envelope start position in target (1-indexed)
            - ``env_to``: Envelope end position in target (1-indexed)
            - ``envelope_score``: Bit score of the domain envelope
            - ``domain_included``: Whether domain passes inclusion thresholds
            - ``domain_reported``: Whether domain passes reporting thresholds
            - ``domain_pvalue``: P-value of the domain

            Empty list if no domain hits are found.

    Properties:
        num_sequence_hits: Total number of sequence-level hits found.
        num_domain_hits: Total number of domain-level hits found.
    """

    sequence_hits: list[SequenceHit] = Field(default_factory=list, description="Sequence-level hits from the search")
    domain_hits: list[DomainHit] = Field(default_factory=list, description="Domain-level hits from the search")

    @property
    def num_sequence_hits(self) -> int:
        """Return the number of unique sequence hits."""
        return len(self.sequence_hits)

    @property
    def num_domain_hits(self) -> int:
        """Return the number of domain-level hits."""
        return len(self.domain_hits)

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        import pandas as pd

        base_path = Path(export_path)

        # Check if there's any data to export
        has_seq_hits = len(self.sequence_hits) > 0
        has_dom_hits = len(self.domain_hits) > 0

        if not has_seq_hits and not has_dom_hits:
            warnings.warn(
                "No PyHMMER results to export. The search returned no hits.",
                UserWarning,
                stacklevel=2,
            )
            return

        seq_path = base_path.parent / f"{base_path.stem}_sequence_hits.{file_format}"
        dom_path = base_path.parent / f"{base_path.stem}_domain_hits.{file_format}"

        if file_format not in ("csv", "json"):
            raise ValueError(f"Unsupported format: {file_format}")

        if has_seq_hits:
            seq_df = pd.DataFrame([hit.model_dump() for hit in self.sequence_hits])
            if file_format == "csv":
                seq_df.to_csv(seq_path, index=False)
            else:
                seq_df.to_json(seq_path, orient="records", indent=2)

        if has_dom_hits:
            dom_df = pd.DataFrame([hit.model_dump() for hit in self.domain_hits])
            if file_format == "csv":
                dom_df.to_csv(dom_path, index=False)
            else:
                dom_df.to_json(dom_path, orient="records", indent=2)


# Config:
class PyHmmerConfig(BaseConfig):
    """Base configuration object for PyHMMER tools.

    This class provides common configuration parameters for all PyHMMER search
    operations, including threading, E-value thresholds, and score cutoffs.

    Attributes:
        num_threads (int): Number of CPU threads to use for parallel processing.
            Setting to ``0`` enables automatic detection of available cores.
            Higher values speed up searches on multi-core systems. Default: 0.

        evalue_threshold (float): E-value reporting threshold for sequence-level
            hits. Sequences with E-values at or below this threshold are reported.
            Lower values are more stringent. Common values:

            - ``10.0``: Default, permissive threshold
            - ``1.0``: Moderately stringent
            - ``0.001``: Very stringent

            Must be greater than 0. Default: 10.0.

        score_threshold (float | None): Score reporting threshold for
            sequence-level hits. If specified, this overrides the E-value threshold.
            Sequences with bit scores at or above this value are reported.
            Default: ``None`` (use E-value threshold).

        domain_evalue_threshold (float): E-value reporting threshold for
            domain-level hits within sequences. Domains with E-values at or below
            this threshold are reported. Must be greater than 0. Default: 10.0.

        domain_score_threshold (float | None): Score reporting threshold for
            domain-level hits. If specified, this overrides the domain E-value
            threshold. Domains with bit scores at or above this value are reported.
            Default: ``None`` (use domain E-value threshold).

    Note:
        When both E-value and score thresholds are specified, the score threshold
        takes precedence. This applies independently to both sequence-level and
        domain-level filtering.
    """

    num_threads: int = ConfigField(
        title="Number of Threads",
        default=0,
        ge=0,
        description="Number of CPU threads to use (0 for auto-detection)",
        hidden=True,
    )
    evalue_threshold: float = ConfigField(
        title="E-value Threshold",
        default=10.0,
        gt=0,
        description="E-value reporting threshold for sequence level hits",
    )
    score_threshold: float | None = ConfigField(
        title="Score Threshold",
        default=None,
        description="Score reporting threshold. (Overrides the E-value threshold)",
        advanced=True,
    )
    domain_evalue_threshold: float = ConfigField(
        title="Domain E-value Threshold",
        default=10.0,
        gt=0,
        description="E-value reporting threshold for domain level hits",
    )
    domain_score_threshold: float | None = ConfigField(
        title="Domain Score Threshold",
        default=None,
        description="Score reporting threshold for domain level hits. (Overrides the Domain E-value threshold)",
        advanced=True,
    )


# ============================================================================
# Helper Functions
# ============================================================================
def _build_hit_models(
    sequence_hits: list[dict[str, Any]],
    domain_hits: list[dict[str, Any]],
) -> tuple[list[SequenceHit], list[DomainHit]]:
    """Build lists of typed hit models from dicts returned by the standalone script.

    Args:
        sequence_hits (list[dict[str, Any]]): List of dicts with sequence-level hit data.
        domain_hits (list[dict[str, Any]]): List of dicts with domain-level hit data.

    Returns:
        tuple[list[SequenceHit], list[DomainHit]]: Tuple of (sequence_hit_models, domain_hit_models).
    """
    sequence_models = [SequenceHit(**hit) for hit in sequence_hits]
    domain_models = [DomainHit(**hit) for hit in domain_hits]

    return sequence_models, domain_models
