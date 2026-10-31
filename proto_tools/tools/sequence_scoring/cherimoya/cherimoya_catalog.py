"""Search the experiments of a Cherimoya model collection."""

import csv
import json
import re
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationInfo, field_validator

from proto_tools.tools.sequence_scoring.cherimoya.catalog import (
    CHERIMOYA_COLLECTIONS,
    CherimoyaAssay,
    CherimoyaBiosampleClassification,
    CherimoyaCatalogEntry,
    CherimoyaCollectionKey,
    load_catalog,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, BaseToolInput, BaseToolOutput, ConfigField, InputField


class CherimoyaCatalogInput(BaseToolInput):
    """Filters for a Cherimoya collection; every filter left unset matches everything.

    Attributes:
        biosample (str | None): Case-insensitive regular expression matched against the
            biosample term name and summary, e.g. ``"K562"`` or ``"hepat|liver"``.
        target (str | None): Case-insensitive regular expression matched against the ChIP-seq
            target, e.g. ``"^CTCF$"`` or ``"GATA"``; matches nothing in an accessibility collection.
        assay (CherimoyaAssay | None): Restrict to ``"DNase-seq"``, ``"ATAC-seq"`` or ``"ChIP-seq"``.
        biosample_classification (list[CherimoyaBiosampleClassification] | None): Restrict
            to these ENCODE biosample classifications.
        accessions (list[str] | None): Restrict to these experiment accessions.
        defaults_only (bool): Return only the default experiment of each (assay, biosample),
            as Chorus committed it (CATv1 only; BATv1 has no defaults yet). Four of the 492 CATv1
            defaults are experiments ENCODE marks as perturbed; they are returned whatever
            ``include_perturbed`` says.
        include_perturbed (bool): Include experiments ENCODE marks as perturbed. Accessions
            named in ``accessions`` are always returned.
        min_count_pearson (float | None): Minimum fold 0 validation count Pearson.
        limit (int): Maximum number of experiments to return, best count Pearson first.
    """

    biosample: str | None = InputField(
        default=None,
        title="Biosample",
        description="Case-insensitive regex over biosample name and summary, e.g. 'K562' or 'hepat|liver'.",
    )
    target: str | None = InputField(
        default=None,
        title="Target",
        description="Case-insensitive regex over the ChIP-seq target, e.g. '^CTCF$' or 'GATA'.",
    )
    assay: CherimoyaAssay | None = InputField(
        default=None, title="Assay", description="DNase-seq, ATAC-seq or ChIP-seq"
    )
    biosample_classification: list[CherimoyaBiosampleClassification] | None = InputField(
        default=None,
        title="Biosample Classification",
        description="Restrict to these ENCODE biosample classifications.",
    )
    accessions: list[str] | None = InputField(
        default=None, title="Accessions", description="Restrict to these ENCODE experiment accessions."
    )
    defaults_only: bool = InputField(
        default=False,
        title="Defaults Only",
        description="Return only the default experiment of each (assay, biosample), perturbed or not.",
    )
    include_perturbed: bool = InputField(
        default=False,
        title="Include Perturbed",
        description="Include experiments ENCODE marks as perturbed; named accessions are always returned.",
    )
    min_count_pearson: float | None = InputField(
        default=None,
        ge=-1.0,
        le=1.0,
        title="Minimum Count Pearson",
        description="Minimum fold 0 validation Pearson of log total counts.",
    )
    limit: int = InputField(
        default=50, ge=1, title="Limit", description="Maximum experiments to return, best count Pearson first."
    )

    @field_validator("biosample", "target")
    @classmethod
    def validate_regex(cls, value: str | None, info: ValidationInfo) -> str | None:
        """Reject a biosample or target pattern that is not a valid regular expression."""
        if value is not None:
            try:
                re.compile(value)
            except re.error as error:
                raise ValueError(f"{info.field_name} is not a valid regular expression: {error}") from error
        return value


class CherimoyaCatalogConfig(BaseConfig):
    """Configuration for the Cherimoya catalog search.

    Attributes:
        collection (CherimoyaCollectionKey): Collection to search.
    """

    collection: CherimoyaCollectionKey = ConfigField(
        title="Collection",
        default="catv1",
        description="Model collection to search: catv1 (accessibility) or batv1 (ChIP-seq).",
    )


class CherimoyaCatalogOutput(BaseToolOutput):
    """Experiments matching a catalog search.

    Attributes:
        experiments (list[CherimoyaCatalogEntry]): Matches, best fold 0 count Pearson first.
        n_matched (int): Matches before ``limit`` was applied.
        collection (str): Collection searched.
        revision (str): Hugging Face revision the collection's checkpoints are pinned to.
    """

    experiments: list[CherimoyaCatalogEntry] = Field(title="Experiments", description="Matching experiments")
    n_matched: int = Field(title="Matched", description="Number of matches before the limit was applied")
    collection: str = Field(title="Collection", description="Collection searched")
    revision: str = Field(title="Revision", description="Hugging Face revision the checkpoints are pinned to")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "csv"

    def _export_output(self, export_path: Path | str, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")
        rows = [entry.model_dump() for entry in self.experiments]
        if file_format == "json":
            with open(path, "w") as f:
                json.dump({"experiments": rows, "n_matched": self.n_matched, "revision": self.revision}, f, indent=2)
            return
        if file_format == "csv":
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(CherimoyaCatalogEntry.model_fields))
                writer.writeheader()
                writer.writerows(rows)
            return
        raise ValueError(f"Unsupported format: {file_format}")


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return CherimoyaCatalogInput(biosample="K562", assay="DNase-seq")


@tool(
    key="cherimoya-catalog",
    label="Cherimoya Catalog",
    category="sequence_scoring",
    input_class=CherimoyaCatalogInput,
    config_class=CherimoyaCatalogConfig,
    output_class=CherimoyaCatalogOutput,
    description="Search the ENCODE accessibility (CATv1) or ChIP-seq (BATv1) experiments a Cherimoya collection has models for",
    example_input=example_input,
    local_only="cherimoya-catalog does not use a gpu and does not need an environment, so it can run in process",
    uses_gpu=False,
    cacheable=True,
)
def run_cherimoya_catalog(
    inputs: CherimoyaCatalogInput,
    config: CherimoyaCatalogConfig,
    instance: Any = None,  # noqa: ARG001 — required by tool interface
) -> CherimoyaCatalogOutput:
    """Search a Cherimoya collection for experiments by biosample, target, assay, and quality.

    Args:
        inputs (CherimoyaCatalogInput): Search filters.
        config (CherimoyaCatalogConfig): Collection to search.
        instance (Any): Unused; the catalog runs in process.

    Returns:
        CherimoyaCatalogOutput: Matching experiments, best fold 0 count Pearson first.
    """
    catalog = load_catalog(config.collection)
    pattern = re.compile(inputs.biosample, re.IGNORECASE) if inputs.biosample else None
    target_pattern = re.compile(inputs.target, re.IGNORECASE) if inputs.target else None
    wanted = set(inputs.accessions) if inputs.accessions else None
    classifications = set(inputs.biosample_classification) if inputs.biosample_classification else None
    unknown = sorted(wanted - set(catalog)) if wanted else []
    if unknown:
        raise ValueError(f"accessions {unknown} are not experiments in collection {config.collection!r}")
    hide_perturbed = not inputs.include_perturbed and wanted is None and not inputs.defaults_only

    matches = []
    for entry in catalog.values():
        if wanted is not None and entry.experiment_accession not in wanted:
            continue
        if inputs.assay is not None and entry.assay != inputs.assay:
            continue
        if classifications is not None and entry.biosample_classification not in classifications:
            continue
        if pattern is not None and not (pattern.search(entry.biosample) or pattern.search(entry.biosample_summary)):
            continue
        if target_pattern is not None and not target_pattern.search(entry.target):
            continue
        if inputs.defaults_only and not entry.is_default:
            continue
        if entry.perturbed and hide_perturbed:
            continue
        if inputs.min_count_pearson is not None and entry.count_pearson_fold_0 < inputs.min_count_pearson:
            continue
        matches.append(entry)

    matches.sort(key=lambda entry: (-entry.count_pearson_fold_0, entry.experiment_accession))
    return CherimoyaCatalogOutput(
        experiments=matches[: inputs.limit],
        n_matched=len(matches),
        collection=config.collection,
        revision=CHERIMOYA_COLLECTIONS[config.collection].revision,
    )
