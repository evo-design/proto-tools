"""Shared data models, constants, and helpers for PyHMMER tools."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
from pydantic import ConfigDict, Field, field_validator

from bio_programming_tools.tools.infra.tool_io import BaseToolInput, BaseToolOutput
from bio_programming_tools.tools.utils import BaseConfig, ConfigField

# ============================================================================
# Constants
# ============================================================================

# Standard column names for sequence-level hits
SEQUENCE_HIT_COLUMNS = {
    "query_name": ("Name of the query HMM", str),
    "query_accession": (
        "Accession of the query HMM. If not available, set to '-'.",
        str,
    ),
    "query_description": (
        "Description of the query HMM. If not available, set to '-'.",
        str,
    ),
    "query_idx": ("Index of the query (0-indexed)", int),
    "target_name": ("Name of the target sequence", str),
    "target_accession": (
        "Accession of the target sequence. If not available, set to '-'.",
        str,
    ),
    "target_description": (
        "Description of the target sequence. If not available, set to '-'.",
        str,
    ),
    "evalue": ("E-value of the hit", float),
    "score": ("Score of the full sequence", float),
    "bias": ("Bias of the full sequence", float),
    "sum_score": ("Sum score of the full sequence", float),
    "reported": ("Whether the hit is reported", bool),
    "included": ("Whether the hit is included", bool),
    "pvalue": ("p-value of the hit", float),
    "num_domains": ("Number of domains in the hit", int),
}

# Standard column names for domain-level hits
DOMAIN_HIT_COLUMNS = {
    "query_name": ("Name of the query HMM", str),
    "query_accession": (
        "Accession of the query HMM. If not available, set to '-'.",
        str,
    ),
    "query_description": (
        "Description of the query HMM. If not available, set to '-'.",
        str,
    ),
    "query_idx": ("Index of the query (0-indexed)", int),
    "target_name": ("Name of the target sequence", str),
    "target_accession": (
        "Accession of the target sequence. If not available, set to '-'.",
        str,
    ),
    "target_description": (
        "Description of the target sequence. If not available, set to '-'.",
        str,
    ),
    "hmm_length": ("Length of the HMM profile", int),
    "hmm_from": ("Start position of the domain match in the HMM (1-indexed)", int),
    "hmm_to": ("End position of the domain match in the HMM (1-indexed)", int),
    "target_from": (
        "Start position of the domain match in the target sequence (1-indexed)",
        int,
    ),
    "target_to": (
        "End position of the domain match in the target sequence (1-indexed)",
        int,
    ),
    "target_length": ("Length of the target sequence", int),
    "c_evalue": ("Conditional E-value of the domain", float),
    "i_evalue": ("Independent E-value of the domain", float),
    "domain_score": ("Bit score of the domain", float),
    "domain_bias": ("Bias correction of the domain score", float),
    "domain_idx": ("Index of the domain within the hit (0-indexed)", int),
    "env_from": (
        "Start position of the domain envelope in the target sequence (1-indexed)",
        int,
    ),
    "env_to": (
        "End position of the domain envelope in the target sequence (1-indexed)",
        int,
    ),
    "envelope_score": ("Bit score of the domain envelope", float),
    "domain_included": ("Whether the domain passes inclusion thresholds", bool),
    "domain_reported": ("Whether the domain passes reporting thresholds", bool),
    "domain_pvalue": ("P-value of the domain", float),
}


# ============================================================================
# Data Models
# ============================================================================

# Input:
class PyHmmerInput(BaseToolInput):
    """Base input object for PyHMMER tools.

    Provides common input fields and validation for protein sequences
    used across all PyHMMER search tools (hmmsearch, hmmscan, phmmer).

    Attributes:
        sequences (List[str]): Input protein sequences for searching.
            Can be provided as:

            - A single protein sequence string (amino acid sequence)
            - A list of protein sequence strings

            Sequences are automatically normalized to a list format internally.
            All sequences must be non-empty and contain valid characters.
    """

    sequences: List[str] = Field(
        description="Input sequences as: single sequence string or list of sequence strings"
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value) -> List[str]:
        """
        Normalize sequences to a list of strings.

        Handles input formats:
        - Single sequence string
        - List of sequence strings
        """
        if isinstance(value, str):
            # Single sequence
            return [value]
        elif isinstance(value, list):
            # List of sequences
            return value
        else:
            raise ValueError(f"Unsupported sequences input type: {type(value)}")

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, sequences: List[str]) -> List[str]:
        """Validate that sequences are non-empty."""
        if not sequences:
            raise ValueError("At least one sequence is required")

        for i, seq in enumerate(sequences):
            if not seq or not seq.strip():
                raise ValueError(f"Sequence {i+1} is empty")

        return sequences


# Output:
class PyHmmerOutput(BaseToolOutput):
    """Output from PyHMMER search operations.

    This class encapsulates the results of PyHMMER searches, providing both
    sequence-level and domain-level hits in structured DataFrames. The output
    format matches traditional HMMER tools.

    Attributes:
        sequence_hits_df (Optional[pd.DataFrame]): DataFrame containing
            sequence-level hits with the following columns:

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

            Returns ``None`` if no sequence hits are found.

        domain_hits_df (Optional[pd.DataFrame]): DataFrame containing domain-level
            hits with the following columns:

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

            Returns ``None`` if no domain hits are found.

    Properties:
        num_sequence_hits (int): Total number of sequence-level hits found.
        num_domain_hits (int): Total number of domain-level hits found.
    """
    sequence_hits_df: Optional[pd.DataFrame] = Field(
        default=None, description="DataFrame with per-sequence hits"
    )
    domain_hits_df: Optional[pd.DataFrame] = Field(
        default=None, description="DataFrame with per-domain hits"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow pandas DataFrame

    @property
    def num_sequence_hits(self) -> int:
        return len(self.sequence_hits_df) if self.sequence_hits_df is not None else 0

    @property
    def num_domain_hits(self) -> int:
        return len(self.domain_hits_df) if self.domain_hits_df is not None else 0

    @property
    def output_format_options(self) -> List[str]:
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str):
        import warnings

        base_path = Path(export_path)

        # Check if there's any data to export
        has_seq_hits = self.sequence_hits_df is not None and len(self.sequence_hits_df) > 0
        has_dom_hits = self.domain_hits_df is not None and len(self.domain_hits_df) > 0

        if not has_seq_hits and not has_dom_hits:
            warnings.warn(
                "No PyHMMER results to export. The search returned no hits.",
                UserWarning,
                stacklevel=2
            )
            return

        seq_path = base_path.parent / f"{base_path.stem}_sequence_hits.{file_format}"
        dom_path = base_path.parent / f"{base_path.stem}_domain_hits.{file_format}"

        if file_format == "csv":
            if has_seq_hits:
                self.sequence_hits_df.to_csv(seq_path, index=False)
            if has_dom_hits:
                self.domain_hits_df.to_csv(dom_path, index=False)

        elif file_format == "json":
            if has_seq_hits:
                self.sequence_hits_df.to_json(seq_path, orient="records", indent=2)
            if has_dom_hits:
                self.domain_hits_df.to_json(dom_path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


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

        score_threshold (Optional[float]): Score reporting threshold for
            sequence-level hits. If specified, this overrides the E-value threshold.
            Sequences with bit scores at or above this value are reported.
            Default: ``None`` (use E-value threshold).

        domain_evalue_threshold (float): E-value reporting threshold for
            domain-level hits within sequences. Domains with E-values at or below
            this threshold are reported. Must be greater than 0. Default: 10.0.

        domain_score_threshold (Optional[float]): Score reporting threshold for
            domain-level hits. If specified, this overrides the domain E-value
            threshold. Domains with bit scores at or above this value are reported.
            Default: ``None`` (use domain E-value threshold).

    Note:
        When both E-value and score thresholds are specified, the score threshold
        takes precedence. This applies independently to both sequence-level and
        domain-level filtering.
    """
    # TODO: Determine how this should be handled for the client.
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
    score_threshold: Optional[float] = ConfigField(
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
    domain_score_threshold: Optional[float] = ConfigField(
        title="Domain Score Threshold",
        default=None,
        description="Score reporting threshold for domain level hits. (Overrides the Domain E-value threshold)",
        advanced=True,
    )


# ============================================================================
# Helper Functions
# ============================================================================
def _build_dataframes(
    sequence_hits: List[dict],
    domain_hits: List[dict],
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Build DataFrames from lists of hit dicts returned by the standalone script.

    Args:
        sequence_hits: List of dicts with sequence-level hit data.
        domain_hits: List of dicts with domain-level hit data.

    Returns:
        Tuple of (sequence_hits_df, domain_hits_df). Either may be None if empty.
    """
    sequence_df = (
        pd.DataFrame(sequence_hits, columns=list(SEQUENCE_HIT_COLUMNS.keys()))
        if sequence_hits
        else None
    )
    domain_df = (
        pd.DataFrame(domain_hits, columns=list(DOMAIN_HIT_COLUMNS.keys()))
        if domain_hits
        else None
    )

    # Convert data types
    if sequence_df is not None:
        sequence_df = _convert_dtypes(sequence_df, is_domain=False)
    if domain_df is not None:
        domain_df = _convert_dtypes(domain_df, is_domain=True)

    return sequence_df, domain_df


def _convert_dtypes(df: pd.DataFrame, is_domain: bool = False) -> pd.DataFrame:
    """Convert DataFrame columns to appropriate data types."""
    ACTIVE_COLUMNS = DOMAIN_HIT_COLUMNS if is_domain else SEQUENCE_HIT_COLUMNS

    float_cols = [
        col_name
        for col_name, col_desc in ACTIVE_COLUMNS.items()
        if col_desc[1] == float
    ]
    int_cols = [
        col_name for col_name, col_desc in ACTIVE_COLUMNS.items() if col_desc[1] == int
    ]
    bool_cols = [
        col_name for col_name, col_desc in ACTIVE_COLUMNS.items() if col_desc[1] == bool
    ]

    # Convert numeric columns
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Keep as float to handle NaN values properly
            if df[col].notna().any():
                df[col] = df[col].round()

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    return df
