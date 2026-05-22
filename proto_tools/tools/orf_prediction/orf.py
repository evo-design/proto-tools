"""Standardized Open Reading Frame (ORF) representation as a Pydantic BaseModel."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proto_tools.utils import calculate_gc_content


class ORF(BaseModel):
    """Base representation of a predicted Open Reading Frame.

    A Pydantic model providing a standardized ORF representation used by
    ORF prediction tools (e.g., Orfipy, Prodigal). Stores sequence data,
    coordinates, and tool-specific metrics.

    Attributes:
        parent_id (str): Identifier of the parent/input sequence (e.g., "seq_0").
        orf_id (str): Unique ORF identifier within the parent sequence (e.g., "gene_1").
        strand (Literal["+", "-"]): Strand direction. "+" for forward, "-" for reverse.
        frame (Literal[1, 2, 3]): Reading frame (1, 2, or 3).
        amino_acid_sequence (str): Translated protein sequence.
        nucleotide_sequence (str): DNA sequence of the ORF.
        amino_acid_length (int): Length of protein in amino acids.
        nucleotide_length (int): Length of ORF in nucleotides.
        nucleotide_start (int): Start position in parent sequence (1-indexed, inclusive).
        nucleotide_end (int): End position in parent sequence (1-indexed, inclusive).
        metrics (dict[str, Any]): Dictionary of tool-specific metrics or metadata.

    Note:
        All ORF coordinates use 1-indexed, inclusive intervals (biology convention).
        To extract using Python slicing: sequence[nucleotide_start-1:nucleotide_end]
    """

    model_config = ConfigDict(extra="forbid")

    parent_id: str = Field(title="Parent ID", description="Identifier of the parent/input sequence")
    orf_id: str = Field(title="ORF ID", description="Unique ORF identifier within the parent sequence")
    strand: Literal["+", "-"] = Field(title="Strand", description="Strand direction ('+' or '-')")
    frame: Literal[1, 2, 3] = Field(title="Frame", description="Reading frame (1, 2, or 3)")
    amino_acid_sequence: str = Field(title="Amino Acid Sequence", description="Translated protein sequence")
    nucleotide_sequence: str = Field(title="Nucleotide Sequence", description="DNA sequence of the ORF")
    amino_acid_length: int = Field(title="Amino Acid Length", description="Length of protein in amino acids")
    nucleotide_length: int = Field(title="Nucleotide Length", description="Length of ORF in nucleotides")
    nucleotide_start: int = Field(title="Nucleotide Start", description="Start position (1-indexed, inclusive)")
    nucleotide_end: int = Field(title="Nucleotide End", description="End position (1-indexed, inclusive)")
    metrics: dict[str, Any] = Field(
        default_factory=dict, title="Metrics", description="Tool-specific metrics or metadata"
    )

    @model_validator(mode="after")
    def _validate_coordinates(self) -> ORF:
        """Validate ORF coordinate constraints (1-indexed, start < end)."""
        if self.nucleotide_start <= 0:
            msg = f"Invalid nucleotide_start {self.nucleotide_start}. Must be > 0 (1-indexed)."
            raise ValueError(msg)
        if self.nucleotide_end <= 0:
            msg = f"Invalid nucleotide_end {self.nucleotide_end}. Must be > 0 (1-indexed)."
            raise ValueError(msg)
        if self.nucleotide_start >= self.nucleotide_end:
            msg = f"Invalid coordinates: start ({self.nucleotide_start}) must be < end ({self.nucleotide_end})."
            raise ValueError(msg)
        return self

    # ============================================================================
    # Properties
    # ============================================================================

    @property
    def id(self) -> str:
        """Combined identifier: parent_id + orf_id (e.g., 'seq_0_gene_1')."""
        return f"{self.parent_id}_{self.orf_id}"

    @property
    def gc_content(self) -> float:
        """GC content percentage of the ORF.

        Returns from metrics if available, otherwise calculates from nucleotide_sequence.
        """
        if "gc_content" in self.metrics:
            return float(self.metrics["gc_content"])
        return calculate_gc_content(self.nucleotide_sequence)

    def __getattr__(self, name: str) -> Any:
        """Access metrics as attributes (e.g., orf.start_type for Prodigal metrics)."""
        # During Pydantic init, metrics may not be set yet
        metrics = self.__dict__.get("metrics")
        if metrics is not None and name in metrics:
            return metrics[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"ORF(id='{self.id}', start={self.nucleotide_start}, end={self.nucleotide_end}, strand='{self.strand}')"

    def __str__(self) -> str:
        return self.__repr__()

    # ============================================================================
    # Export
    # ============================================================================

    def to_flat_dict(self) -> dict[str, Any]:
        """Serialize to a flat dictionary with metrics expanded at the top level.

        Includes computed properties (``id``, ``gc_content``) and promotes metric
        keys to top-level entries for DataFrame-friendly export.

        Returns:
            dict[str, Any]: Flat dictionary suitable for CSV/DataFrame construction.
        """
        data = self.model_dump()
        data["id"] = self.id
        data["gc_content"] = self.gc_content
        # Promote metrics to top-level keys for DataFrame ease
        metrics = data.pop("metrics", {})
        for k, v in metrics.items():
            if k not in data:
                data[k] = v
        return data
