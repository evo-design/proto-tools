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

    query_name: str = Field(title="Query Name", description="Name of the query HMM")
    query_accession: str = Field(
        title="Query Accession", description="Accession of the query HMM. If not available, set to '-'."
    )
    query_description: str = Field(
        title="Query Description", description="Description of the query HMM. If not available, set to '-'."
    )
    query_idx: int = Field(title="Query Index", description="Index of the query (0-indexed)")
    target_name: str = Field(title="Target Name", description="Name of the target sequence")
    target_accession: str = Field(
        title="Target Accession", description="Accession of the target sequence. If not available, set to '-'."
    )
    target_description: str = Field(
        title="Target Description", description="Description of the target sequence. If not available, set to '-'."
    )
    evalue: float = Field(title="E-value", description="E-value of the hit")
    score: float = Field(title="Score", description="Score of the full sequence")
    bias: float = Field(title="Bias", description="Bias of the full sequence")
    sum_score: float = Field(title="Sum Score", description="Sum score of the full sequence")
    reported: bool = Field(title="Reported", description="Whether the hit is reported")
    included: bool = Field(title="Included", description="Whether the hit is included")
    pvalue: float = Field(title="P-value", description="p-value of the hit")
    num_domains: int = Field(title="Number of Domains", description="Number of domains in the hit")


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

    query_name: str = Field(title="Query Name", description="Name of the query HMM")
    query_accession: str = Field(
        title="Query Accession", description="Accession of the query HMM. If not available, set to '-'."
    )
    query_description: str = Field(
        title="Query Description", description="Description of the query HMM. If not available, set to '-'."
    )
    query_idx: int = Field(title="Query Index", description="Index of the query (0-indexed)")
    target_name: str = Field(title="Target Name", description="Name of the target sequence")
    target_accession: str = Field(
        title="Target Accession", description="Accession of the target sequence. If not available, set to '-'."
    )
    target_description: str = Field(
        title="Target Description", description="Description of the target sequence. If not available, set to '-'."
    )
    hmm_length: int = Field(title="HMM Length", description="Length of the HMM profile")
    hmm_from: int = Field(title="HMM From", description="Start position of the domain match in the HMM (1-indexed)")
    hmm_to: int = Field(title="HMM To", description="End position of the domain match in the HMM (1-indexed)")
    target_from: int = Field(
        title="Target From", description="Start position of the domain match in the target (1-indexed)"
    )
    target_to: int = Field(title="Target To", description="End position of the domain match in the target (1-indexed)")
    target_length: int = Field(title="Target Length", description="Length of the target sequence")
    c_evalue: float = Field(title="Conditional E-value", description="Conditional E-value of the domain")
    i_evalue: float = Field(title="Independent E-value", description="Independent E-value of the domain")
    domain_score: float = Field(title="Domain Score", description="Bit score of the domain")
    domain_bias: float = Field(title="Domain Bias", description="Bias correction of the domain score")
    domain_idx: int = Field(title="Domain Index", description="Index of the domain within the hit (0-indexed)")
    env_from: int = Field(
        title="Envelope From", description="Start position of the domain envelope in the target (1-indexed)"
    )
    env_to: int = Field(
        title="Envelope To", description="End position of the domain envelope in the target (1-indexed)"
    )
    envelope_score: float = Field(title="Envelope Score", description="Bit score of the domain envelope")
    domain_included: bool = Field(title="Domain Included", description="Whether the domain passes inclusion thresholds")
    domain_reported: bool = Field(title="Domain Reported", description="Whether the domain passes reporting thresholds")
    domain_pvalue: float = Field(title="Domain P-value", description="P-value of the domain")


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
        title="Sequences",
        description="Input sequences as: single sequence string or list of sequence strings",
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

    sequence_hits: list[SequenceHit] = Field(
        default_factory=list,
        title="Sequence Hits",
        description="Sequence-level hits from the search",
    )
    domain_hits: list[DomainHit] = Field(
        default_factory=list,
        title="Domain Hits",
        description="Domain-level hits from the search",
    )

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

    Common knobs for hmmsearch / hmmscan / phmmer / nhmmer / jackhmmer.
    Reporting thresholds control what appears in output; inclusion thresholds
    mark a hit as 'trusted' (jackhmmer uses included hits to seed the next
    iteration).

    Attributes:
        num_threads (int): CPU threads; 0 = auto-detect. Default 0.
        evalue_threshold (float): Sequence-level E-value cap to report.
            Default 10.0.
        score_threshold (float | None): Sequence-level bit-score floor.
            Overrides E-value when set. Default None.
        domain_evalue_threshold (float): Per-domain E-value cap to report.
            Default 10.0.
        domain_score_threshold (float | None): Per-domain bit-score floor.
            Overrides domain E-value when set. Default None.
        inclusion_evalue_threshold (float): Sequence-level inclusion E-value.
            Default 0.01.
        inclusion_domain_evalue_threshold (float): Per-domain inclusion E-value.
            Default 0.01.
        z_value (float | None): Effective database size for E-value calc.
            None = use the actual target count.
        domain_z_value (float | None): Significant hit count for domain E-value
            calc. None = use actual.
        skip_filters (bool): Disable MSV/Vit/Fwd heuristic filters.
            Slower but maximally sensitive. Default False.
    """

    num_threads: int = ConfigField(
        title="Number of Threads",
        default=0,
        ge=0,
        description="CPU threads. 0 auto-detects available cores",
        include_in_key=False,
    )
    evalue_threshold: float = ConfigField(
        title="E-value Threshold",
        default=10.0,
        gt=0,
        description="Sequence E-value cap. Lower = stricter; 0.001 for confident, 1e-10 for stringent",
    )
    score_threshold: float | None = ConfigField(
        title="Score Threshold",
        default=None,
        description="Sequence bit-score floor. Use for cross-DB-size comparisons; overrides E-value if set",
    )
    domain_evalue_threshold: float = ConfigField(
        title="Domain E-value Threshold",
        default=10.0,
        gt=0,
        description="Domain E-value cap. Tighten independently from sequence threshold",
    )
    domain_score_threshold: float | None = ConfigField(
        title="Domain Score Threshold",
        default=None,
        description="Domain bit-score floor. For cross-DB-size comparisons; overrides domain E-value",
    )
    inclusion_evalue_threshold: float = ConfigField(
        title="Inclusion E-value Threshold",
        default=0.01,
        gt=0,
        description="Inclusion E-value cap. Sets 'included' flag; seeds jackhmmer next iteration",
    )
    inclusion_domain_evalue_threshold: float = ConfigField(
        title="Inclusion Domain E-value",
        default=0.01,
        gt=0,
        description="Domain inclusion E-value cap. Sets included flag on domain hits",
    )
    z_value: float | None = ConfigField(
        title="Database Size (Z)",
        default=None,
        gt=0,
        description="Effective DB size for E-value calc. Set constant for cross-DB compare; None = actual count",
    )
    domain_z_value: float | None = ConfigField(
        title="Domain Database Size (Z)",
        default=None,
        gt=0,
        description="Significant hits for domain E-value calc. Set with Z for cross-DB; None = actual",
    )
    skip_filters: bool = ConfigField(
        title="Skip Heuristic Filters",
        default=False,
        description="Disable MSV/Vit/Fwd + bias filters. 10-100x slower but max sensitivity for distant homologs",
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
