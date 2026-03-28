"""bio_programming_tools/tools/sequence_scoring/alphagenome/shared_data_models.py

Shared data models, constants, and Literal types for AlphaGenome tools."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import Field, field_validator, model_validator

from bio_programming_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
)

# ============================================================================
# Constants
# ============================================================================

DEFAULT_ALPHAGENOME_MODEL_VERSION = "all_folds"
SUPPORTED_CONTEXT_LENGTHS = (1_048_576, 524_288, 131_072, 16_384)

OutputTypeName = Literal[
    "ATAC", "CAGE", "DNASE", "RNA_SEQ",
    "CHIP_HISTONE", "CHIP_TF",
    "SPLICE_SITES", "SPLICE_SITE_USAGE", "SPLICE_JUNCTIONS",
    "CONTACT_MAPS", "PROCAP",
]

VariantScorerName = Literal[
    "ATAC", "CONTACT_MAPS", "DNASE", "CHIP_TF", "CHIP_HISTONE",
    "CAGE", "PROCAP", "RNA_SEQ", "RNA_SEQ_ACTIVE",
    "SPLICE_SITES", "SPLICE_SITE_USAGE", "SPLICE_JUNCTIONS",
    "POLYADENYLATION",
    "ATAC_ACTIVE", "DNASE_ACTIVE", "CHIP_TF_ACTIVE",
    "CHIP_HISTONE_ACTIVE", "CAGE_ACTIVE", "PROCAP_ACTIVE",
]


# ============================================================================
# Data Models
# ============================================================================

# Input:
class AlphaGenomeInterval(BaseToolInput):
    """Base input for AlphaGenome tools that operate on genomic intervals.

    AlphaGenome's architecture requires input sequences whose length matches
    one of the supported context lengths:

    * **1,048,576 bp** (1 MB, recommended)
    * **524,288 bp** (500 KB)
    * **131,072 bp** (100 KB)
    * **16,384 bp** (16 KB)

    If the supplied interval does not match a supported length it is
    automatically resized at inference time (see ``standalone/inference.py``).

    Coordinates use 0-based indexing with an exclusive end (BED-style).

    Attributes:
        chromosome (str): Chromosome identifier, e.g. ``'chr1'``.
        interval_start (int): Interval start (0-based, inclusive).
        interval_end (int): Interval end (0-based, exclusive).
    """

    chromosome: str = InputField(description="Chromosome identifier, e.g. 'chr1'")
    interval_start: int = InputField(
        ge=0,
        description="Interval start (0-based, inclusive)",
    )
    interval_end: int = InputField(
        ge=1,
        description="Interval end (0-based, exclusive)",
    )

    @model_validator(mode="after")
    def validate_interval(self) -> AlphaGenomeInterval:
        """Validate interval start < end."""
        if self.interval_end <= self.interval_start:
            raise ValueError("end must be greater than start")
        return self


class AlphaGenomeVariant(AlphaGenomeInterval):
    """Input object for AlphaGenome variant-effect tools.

    Extends the base interval input with variant coordinates and alleles.

    Attributes:
        chromosome (str): Chromosome identifier, e.g. ``'chr1'``.
        interval_start (int): Interval start (0-based, inclusive).
        interval_end (int): Interval end (0-based, exclusive).
        variant_position (int): Variant genomic position (0-based).
        reference_bases (str): Reference allele, e.g. ``'A'`` or ``'AC'``.
        alternate_bases (str): Alternate allele, e.g. ``'G'`` or ``'GTT'``.
    """

    variant_position: int = InputField(
        ge=0,
        description="Variant genomic position (0-based)",
    )
    reference_bases: str = InputField(description="Reference allele, e.g. 'A' or 'AC'")
    alternate_bases: str = InputField(description="Alternate allele, e.g. 'G' or 'GTT'")

    @field_validator("reference_bases", "alternate_bases")
    @classmethod
    def validate_allele_bases(cls, bases: str) -> str:
        """Validate allele sequence characters."""
        normalized = bases.strip().upper()
        if not normalized:
            raise ValueError("Allele values cannot be empty")
        if not set(normalized) <= set("ACGTN"):
            raise ValueError("Allele values must only contain DNA bases A/C/G/T/N")
        return normalized

    @model_validator(mode="after")
    def validate_variant_position(self) -> AlphaGenomeVariant:
        """Ensure variant position is within the interval."""
        if not (self.interval_start <= self.variant_position < self.interval_end):
            raise ValueError("variant_position must be within [interval_start, interval_end)")
        return self


# Output:
class AlphaGenomePredictOutput(BaseToolOutput):
    """Structured output from AlphaGenome interval or variant prediction.

    Attributes:
        chromosome (str): Chromosome identifier.
        interval_start (int): Interval start (0-based).
        interval_end (int): Interval end (0-based, exclusive).
        requested_outputs (list[OutputTypeName]): Output types requested.
        result (dict[str, Any]): Serialized AlphaGenome prediction payload.
        variant (dict[str, Any] | None): Variant metadata (variant predictions only).
    """

    chromosome: str = Field(description="Chromosome identifier")
    interval_start: int = Field(description="Interval start (0-based)")
    interval_end: int = Field(description="Interval end (0-based, exclusive)")
    requested_outputs: List[OutputTypeName] = Field(description="Output types requested for this prediction")
    result: Dict[str, Any] = Field(description="Serialized AlphaGenome prediction payload")
    variant: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Variant metadata for variant-effect predictions",
    )

    @property
    def output_format_options(self) -> List[str]:
        return ["json", "npy"]

    @property
    def output_format_default(self) -> str:
        return "json"

    def _export_output(self, export_path: Union[Path, str], file_format: str):
        path = Path(export_path).with_suffix(f".{file_format}")
        payload = self.model_dump(mode="json")

        if file_format == "json":
            with open(path, "w") as handle:
                json.dump(payload, handle, indent=2)
            return

        if file_format == "npy":
            import numpy as np

            np.save(path, payload)
            return

        raise ValueError(f"Unsupported format: {file_format}")


class AlphaGenomeScoreOutput(BaseToolOutput):
    """Structured output from AlphaGenome scoring (variant, interval, or ISM).

    Scores are returned as a flat list of records (tidy DataFrame format),
    where each record represents one scorer-track(-gene) combination.

    Attributes:
        scores (list[dict[str, Any]]): Tidy score records. Each dict contains
            keys such as ``variant_id``, ``scored_interval``, ``gene_id``,
            ``gene_name``, ``output_type``, ``variant_scorer`` or
            ``interval_scorer``, ``track_name``, ``raw_score``, etc.
    """

    scores: List[Dict[str, Any]] = Field(
        description="Tidy score records (one per scorer-track-gene combination)"
    )

    @property
    def output_format_options(self) -> List[str]:
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        return "json"

    def _export_output(self, export_path: Union[Path, str], file_format: str):
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "json":
            with open(path, "w") as handle:
                json.dump(self.scores, handle, indent=2)
            return

        if file_format == "csv":
            if not self.scores:
                path.write_text("")
                return
            fieldnames = list(self.scores[0].keys())
            with open(path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.scores)
            return

        raise ValueError(f"Unsupported format: {file_format}")


# Config:
class AlphaGenomePredictConfig(BaseConfig):
    """Shared configuration for AlphaGenome prediction tools.

    Attributes:
        model_version (str): AlphaGenome Hugging Face model version.
        requested_outputs (list[OutputTypeName]): Output type names to request.
        ontology_terms (list[str] | None): Optional ontology term filters.
        organism (Literal['human', 'mouse']): Organism for predictions.
        device (str): Device to run inference on.
        timeout (int): Maximum execution time in seconds. AlphaGenome JAX
            compilation is slow on first run.
    """

    model_version: str = ConfigField(
        title="Model Version",
        default=DEFAULT_ALPHAGENOME_MODEL_VERSION,
        description="AlphaGenome Hugging Face model version",
        advanced=True,
        reload_on_change=True,
    )
    requested_outputs: List[OutputTypeName] = ConfigField(
        title="Requested Outputs",
        default=["RNA_SEQ"],
        description="Output type names to request from AlphaGenome",
    )
    ontology_terms: Optional[List[str]] = ConfigField(
        title="Ontology Terms",
        default=None,
        description="Optional ontology term filters",
        advanced=True,
    )
    organism: Literal["human", "mouse"] = ConfigField(
        title="Organism",
        default="human",
        description="Organism for AlphaGenome predictions",
        advanced=True,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run AlphaGenome inference on",
        hidden=True,
        include_in_key=False,
    )
    timeout: int = ConfigField(
        title="Timeout",
        default=1800,
        ge=1,
        description="Maximum execution time in seconds (AlphaGenome JAX compilation is slow on first run)",
        hidden=True,
    )

    @field_validator("requested_outputs")
    @classmethod
    def validate_requested_outputs(cls, outputs: List[str]) -> List[str]:
        """Uppercase and deduplicate requested output names."""
        seen: set[str] = set()
        normalized: List[str] = []
        for name in outputs:
            upper = name.strip().upper()
            if upper not in seen:
                seen.add(upper)
                normalized.append(upper)
        return normalized
