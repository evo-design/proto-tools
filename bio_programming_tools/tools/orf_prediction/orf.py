"""bio_programming_tools/tools/orf_prediction/orf.py"""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from bio_programming_tools.utils import calculate_gc_content


class ORF:
    """Base representation of a predicted Open Reading Frame.

    This class provides a standardized representation of an ORF that can be
    extended by specific ORF prediction tools (e.g., Orfipy, Prodigal).

    Attributes:
        parent_id: Identifier of the parent/input sequence (e.g., "seq_0").
        orf_id: Unique ORF identifier within the parent sequence (e.g., "gene_1").
        strand: Strand direction. "+" for forward, "-" for reverse.
        frame: Reading frame (1, 2, or 3).
        amino_acid_sequence: Translated protein sequence.
        nucleotide_sequence: DNA sequence of the ORF.
        amino_acid_length: Length of protein in amino acids.
        nucleotide_length: Length of ORF in nucleotides.
        nucleotide_start: Start position in parent sequence (1-indexed, inclusive).
        nucleotide_end: End position in parent sequence (1-indexed, inclusive).
        metrics: Dictionary of tool-specific metrics or metadata.
        id: Combined identifier (computed): parent_id + "_" + orf_id (e.g., "seq_0_gene_1").

    Note:
        All ORF coordinates use 1-indexed, inclusive intervals (biology convention).
        To extract using Python slicing: sequence[nucleotide_start-1:nucleotide_end]
    """

    def __init__(
        self,
        parent_id: str,
        orf_id: str,
        strand: Literal["+", "-"],
        frame: int,
        amino_acid_sequence: str,
        nucleotide_sequence: str,
        amino_acid_length: int,
        nucleotide_length: int,
        nucleotide_start: int,
        nucleotide_end: int,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.parent_id = parent_id
        self.orf_id = orf_id
        self.strand = strand
        self.frame = frame
        self.amino_acid_sequence = amino_acid_sequence
        self.nucleotide_sequence = nucleotide_sequence
        self.amino_acid_length = amino_acid_length
        self.nucleotide_length = nucleotide_length
        self.nucleotide_start = nucleotide_start
        self.nucleotide_end = nucleotide_end
        self.metrics = metrics if metrics is not None else {}

        # Validate the ORF fields
        self._validate()

    def __getattr__(self, name: str) -> Any:
        """Access metrics as attributes."""
        if name in self.metrics:
            return self.metrics[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def _validate(self) -> None:
        """Validate ORF fields after construction."""
        # Validate strand
        if self.strand not in ("+", "-"):
            raise ValueError(
                f"Invalid strand '{self.strand}'. Must be '+' or '-'."
            )

        # Validate frame
        if self.frame not in (1, 2, 3):
            raise ValueError(
                f"Invalid frame {self.frame}. Must be 1, 2, or 3."
            )

        # Validate coordinates
        if self.nucleotide_start <= 0:
            raise ValueError(
                f"Invalid nucleotide_start {self.nucleotide_start}. Must be > 0 (1-indexed)."
            )

        if self.nucleotide_end <= 0:
            raise ValueError(
                f"Invalid nucleotide_end {self.nucleotide_end}. Must be > 0 (1-indexed)."
            )

        if self.nucleotide_start >= self.nucleotide_end:
            raise ValueError(
                f"Invalid coordinates: start ({self.nucleotide_start}) must be < end ({self.nucleotide_end})."
            )

    @property
    def gc_content(self) -> float:
        """GC content percentage of the ORF.

        Returns from metrics if available, otherwise calculates from nucleotide_sequence.
        """
        if "gc_content" in self.metrics:
            return float(self.metrics["gc_content"])
        return calculate_gc_content(self.nucleotide_sequence)

    @property
    def id(self) -> str:
        """Combined identifier: parent_id + orf_id (e.g., 'seq_0_gene_1')."""
        return f"{self.parent_id}_{self.orf_id}"

    def __repr__(self) -> str:
        return (
            f"ORF(id='{self.id}', start={self.nucleotide_start}, "
            f"end={self.nucleotide_end}, strand='{self.strand}')"
        )

    def __str__(self) -> str:
        return self.__repr__()

    # ===============================
    # Pydantic Serialization
    # ===============================
    def model_dump(self) -> Dict[str, Any]:
        """Serialize ORF to a dictionary (for Pydantic models and DataFrames)."""
        return self._serialize_to_dict()

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """
        Tells Pydantic how to validate and serialize ORF objects.
        """
        return core_schema.no_info_after_validator_function(
            cls._validate_from_dict,
            core_schema.union_schema(
                [
                    # Allow creating from a dict (for deserialization)
                    core_schema.dict_schema(),
                    # Allow passing an existing ORF instance
                    core_schema.is_instance_schema(cls),
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize_to_dict,
                info_arg=False,
                return_schema=core_schema.dict_schema(),
            ),
        )

    @classmethod
    def _validate_from_dict(cls, value: Dict[str, Any] | ORF) -> ORF:
        """
        Create an ORF from a dictionary (used during deserialization).
        """
        if isinstance(value, cls):
            return value

        if not isinstance(value, dict):
            raise ValueError(f"Expected dict or ORF, got {type(value)}")

        # Known fields that go into __init__
        known_fields = {
            "parent_id",
            "orf_id",
            "strand",
            "frame",
            "amino_acid_sequence",
            "nucleotide_sequence",
            "amino_acid_length",
            "nucleotide_length",
            "nucleotide_start",
            "nucleotide_end",
            "metrics",
        }

        # Create copy and separate known fields from extra metrics
        data = value.copy()
        data.pop("id", None)  # Property, ignore

        init_args = {}
        extra_metrics = {}

        # If metrics already exists in data, start with it
        metrics = data.pop("metrics", {}) or {}
        extra_metrics.update(metrics)

        for k, v in data.items():
            if k in known_fields:
                init_args[k] = v
            else:
                extra_metrics[k] = v

        init_args["metrics"] = extra_metrics
        return cls(**init_args)

    def _serialize_to_dict(self) -> Dict[str, Any]:
        """
        Serialize ORF to a dictionary (for Pydantic models).
        """
        data = {
            "parent_id": self.parent_id,
            "orf_id": self.orf_id,
            "strand": self.strand,
            "frame": self.frame,
            "amino_acid_sequence": self.amino_acid_sequence,
            "nucleotide_sequence": self.nucleotide_sequence,
            "amino_acid_length": self.amino_acid_length,
            "nucleotide_length": self.nucleotide_length,
            "nucleotide_start": self.nucleotide_start,
            "nucleotide_end": self.nucleotide_end,
            "metrics": self.metrics,
            "id": self.id,
            "gc_content": self.gc_content,  # Add gc_content property to serialization
        }
        # Add metrics directly to the dict for DataFrame ease if they don't clash
        for k, v in self.metrics.items():
            if k not in data:
                data[k] = v
        return data
