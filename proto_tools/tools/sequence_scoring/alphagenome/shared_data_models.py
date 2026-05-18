"""proto_tools/tools/sequence_scoring/alphagenome/shared_data_models.py.

Shared data models, constants, and Literal types for AlphaGenome tools.
"""

import csv
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationInfo, field_validator, model_validator

from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
)
from proto_tools.utils.compressed_array import decompress_result

# ============================================================================
# Constants
# ============================================================================

DEFAULT_ALPHAGENOME_MODEL_VERSION = "all_folds"
SUPPORTED_CONTEXT_LENGTHS = (1_048_576, 524_288, 131_072, 16_384)

OutputTypeName = Literal[
    "ATAC",
    "CAGE",
    "DNASE",
    "RNA_SEQ",
    "CHIP_HISTONE",
    "CHIP_TF",
    "SPLICE_SITES",
    "SPLICE_SITE_USAGE",
    "SPLICE_JUNCTIONS",
    "CONTACT_MAPS",
    "PROCAP",
]

VariantScorerName = Literal[
    "ATAC",
    "CONTACT_MAPS",
    "DNASE",
    "CHIP_TF",
    "CHIP_HISTONE",
    "CAGE",
    "PROCAP",
    "RNA_SEQ",
    "RNA_SEQ_ACTIVE",
    "SPLICE_SITES",
    "SPLICE_SITE_USAGE",
    "SPLICE_JUNCTIONS",
    "POLYADENYLATION",
    "ATAC_ACTIVE",
    "DNASE_ACTIVE",
    "CHIP_TF_ACTIVE",
    "CHIP_HISTONE_ACTIVE",
    "CAGE_ACTIVE",
    "PROCAP_ACTIVE",
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
    def validate_interval(self) -> "AlphaGenomeInterval":
        """Validate interval start < end."""
        if self.interval_end <= self.interval_start:
            raise ValueError(f"interval_end ({self.interval_end}) must be > interval_start ({self.interval_start})")
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
    def validate_allele_bases(cls, bases: str, info: ValidationInfo) -> str:
        """Validate allele sequence characters."""
        normalized = bases.strip().upper()
        if not normalized:
            raise ValueError(f"{info.field_name}: cannot be empty")
        invalid = sorted(set(normalized) - set("ACGTN"))
        if invalid:
            raise ValueError(
                f"{info.field_name}: must only contain DNA bases A/C/G/T/N; got invalid {invalid} in {bases!r}"
            )
        return normalized

    @model_validator(mode="after")
    def validate_variant_position(self) -> "AlphaGenomeVariant":
        """Ensure variant position is within the interval."""
        if not (self.interval_start <= self.variant_position < self.interval_end):
            raise ValueError(
                f"variant_position ({self.variant_position}) must be within "
                f"[interval_start={self.interval_start}, interval_end={self.interval_end})"
            )
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
    requested_outputs: list[OutputTypeName] = Field(description="Output types requested for this prediction")
    result: dict[str, Any] = Field(description="Serialized AlphaGenome prediction payload")
    variant: dict[str, Any] | None = Field(
        default=None,
        description="Variant metadata for variant-effect predictions",
    )

    def model_post_init(self, __context: Any) -> None:
        """Decompress any compressed arrays in the result dict.

        Compression is used as a wire format for subprocess IPC — large numpy
        arrays are sent as ``base85(zlib(tobytes()))`` dicts instead of nested
        Python lists, cutting transport time from minutes to seconds. This hook
        transparently decompresses them back to lists when the Pydantic model
        is constructed, so consumers always see the same ``list[list[float]]``
        structure regardless of whether compression was used on the wire.
        """
        super().model_post_init(__context)
        object.__setattr__(self, "result", decompress_result(self.result, to_list=True))

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "npy"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Path | str, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")
        payload = self.model_dump(mode="json")

        if file_format == "json":
            with open(path, "w") as handle:
                json.dump(payload, handle, indent=2)
            return

        if file_format == "npy":
            import numpy as np

            np.save(path, payload)  # type: ignore[arg-type]
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

    scores: list[dict[str, Any]] = Field(description="Tidy score records (one per scorer-track-gene combination)")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Path | str, file_format: str) -> None:
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
        timeout (int | None): Maximum execution time in seconds. AlphaGenome JAX
            compilation is slow on first run. ``None`` waits indefinitely.
    """

    model_version: str = ConfigField(
        title="Model Version",
        default=DEFAULT_ALPHAGENOME_MODEL_VERSION,
        description="AlphaGenome Hugging Face model version",
        reload_on_change=True,
    )
    requested_outputs: list[OutputTypeName] = ConfigField(
        title="Requested Outputs",
        default=["RNA_SEQ"],
        description="Output type names to request from AlphaGenome",
    )
    ontology_terms: list[str] | None = ConfigField(
        title="Ontology Terms",
        default=None,
        description="UBERON tissue/cell IDs to filter tracks (e.g. 'UBERON:0001167'); None = all",
    )
    organism: Literal["human", "mouse"] = ConfigField(
        title="Organism",
        default="human",
        description="Organism for AlphaGenome predictions",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run AlphaGenome inference on",
        include_in_key=False,
    )
    timeout: int | None = ConfigField(
        title="Timeout",
        default=1800,
        ge=1,
        description="Maximum execution time in seconds (JAX compilation is slow on first run). None = no cap.",
    )

    @field_validator("requested_outputs")
    @classmethod
    def validate_requested_outputs(cls, outputs: list[str]) -> list[str]:
        """Uppercase and deduplicate requested output names."""
        seen: set[str] = set()
        normalized: list[str] = []
        for name in outputs:
            upper = name.strip().upper()
            if upper not in seen:
                seen.add(upper)
                normalized.append(upper)
        return normalized
