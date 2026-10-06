"""proto_tools/tools/database_retrieval/rfam/rfam_regions.py.

Lists where an Rfam family is annotated in sequenced genomes, using the
family regions endpoint of the Rfam website.
"""

import csv
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from proto_tools.tools.database_retrieval.rfam.shared_data_models import (
    _BACKOFF_SECONDS,
    _HTTP_RETRIES,
    _USER_AGENT,
    RfamFamilyQuery,
    _family_url,
    _not_found,
    _rfam_get,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, BaseToolOutput, ConfigField, InputField, build_http_session

_HEADER_FAMILY = re.compile(r"^# Rfam regions for family (\S+) \((RF\d+)\)")
_HEADER_RELEASE = re.compile(r"^# file built using Rfam version (\S+)")
_HEADER_COUNT = re.compile(r"^# found (\d+) regions")
_REGION_COLUMNS = 7

# ============================================================================
# Data Models
# ============================================================================


class RfamRegion(BaseModel):
    """One annotated hit of an Rfam family in a sequence.

    Attributes:
        sequence_accession (str): Versioned accession of the sequence holding the hit.
        bit_score (float): Infernal bit score of the hit.
        start (int): First position of the hit (1-indexed, inclusive, start <= end).
        end (int): Last position of the hit (1-indexed, inclusive).
        strand (Literal['+', '-']): Strand of the hit on the sequence.
        description (str): Description of the sequence holding the hit.
        species (str): Species name of the sequence.
        taxid (int): NCBI taxonomy ID of the species.
    """

    sequence_accession: str = Field(title="Sequence Accession", description="Versioned accession, e.g. 'AM181176.4'")
    bit_score: float = Field(title="Bit Score", description="Infernal bit score of the hit")
    start: int = Field(title="Start", description="First position of the hit (1-indexed, inclusive)")
    end: int = Field(title="End", description="Last position of the hit (1-indexed, inclusive)")
    strand: Literal["+", "-"] = Field(title="Strand", description="Strand of the hit on the sequence")
    description: str = Field(title="Description", description="Description of the sequence holding the hit")
    species: str = Field(title="Species", description="Species name of the sequence")
    taxid: int = Field(title="Taxonomy ID", description="NCBI taxonomy ID of the species")


class RfamRegionsInput(RfamFamilyQuery):
    """Input for listing an Rfam family's annotated regions, optionally narrowed.

    Attributes:
        family (str): Rfam accession (e.g. 'RF01731') or family ID (e.g. 'TwoAYGGAY').
        taxid (int | None): Keep only hits in this NCBI taxonomy ID.
        species (str | None): Keep only hits whose species contains this text (case-insensitive).
        sequence_accession (str | None): Keep only hits on this sequence accession.
            The version suffix is optional ('AM181176' matches 'AM181176.4').
    """

    taxid: int | None = InputField(
        default=None,
        ge=1,
        title="Taxonomy ID",
        description="Keep only hits in this NCBI taxonomy ID (e.g. 216595)",
    )
    species: str | None = InputField(
        default=None,
        title="Species",
        description="Keep only hits whose species contains this text (case-insensitive)",
    )
    sequence_accession: str | None = InputField(
        default=None,
        title="Sequence Accession",
        description="Keep only hits on this sequence; version optional ('AM181176' or 'AM181176.4')",
    )

    @field_validator("species", "sequence_accession")
    @classmethod
    def _blank_is_unset(cls, value: str | None) -> str | None:
        """Treat a blank filter as no filter rather than one that matches everything by accident."""
        if value is None:
            return None
        return value.strip() or None


class RfamRegionsConfig(BaseConfig):
    """Configuration for listing Rfam family regions.

    Attributes:
        max_regions (int): Most regions returned after filtering; the rest are counted, not listed.
    """

    max_regions: int = ConfigField(
        default=500,
        ge=1,
        le=50000,
        title="Max Regions",
        description="Most regions returned after filtering; the rest are counted, not listed",
    )


class RfamRegionsOutput(BaseToolOutput):
    """Output from listing an Rfam family's annotated regions.

    Attributes:
        accession (str): Rfam accession of the family.
        rfam_id (str): Rfam family ID.
        rfam_release (str | None): Rfam release the regions were built from.
        total_regions (int): Regions the family has across all sequences.
        matched_regions (int): Regions left after the input filters.
        truncated (bool): Whether matched regions exceeded max_regions.
        regions (list[RfamRegion]): Matched regions, at most max_regions of them.
    """

    accession: str = Field(title="Accession", description="Rfam accession of the family")
    rfam_id: str = Field(title="Rfam ID", description="Rfam family ID")
    rfam_release: str | None = Field(default=None, title="Rfam Release", description="Rfam release of the regions")
    total_regions: int = Field(title="Total Regions", description="Regions the family has across all sequences")
    matched_regions: int = Field(title="Matched Regions", description="Regions left after the input filters")
    truncated: bool = Field(title="Truncated", description="Whether matched regions exceeded max_regions")
    regions: list[RfamRegion] = Field(
        default_factory=list, title="Regions", description="Matched regions, at most max_regions of them"
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "tsv", "csv"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Any, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")
        if file_format == "json":
            with path.open("w", encoding="utf-8") as f:
                json.dump(self.model_dump(mode="json"), f, indent=2)
            return
        if file_format in ("tsv", "csv"):
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(RfamRegion.model_fields),
                    delimiter="\t" if file_format == "tsv" else ",",
                )
                writer.writeheader()
                writer.writerows(r.model_dump() for r in self.regions)
            return
        raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Private Helpers
# ============================================================================


def _parse_regions(text: str, family: str) -> tuple[str, str, str | None, int, list[RfamRegion]]:
    """Parse the regions endpoint's text into (accession, id, release, total, regions)."""
    lines = text.splitlines()
    if not lines or not lines[0].startswith("#"):
        # Families with millions of hits get a plain-text refusal instead of a table.
        raise ValueError(
            f"Rfam did not list regions for {family!r}: {text.strip()[:300]} "
            "rfam-regions only lists families Rfam can return in a single response."
        )

    accession = rfam_id = ""
    release: str | None = None
    total: int | None = None
    regions: list[RfamRegion] = []
    for line in lines:
        if line.startswith("#"):
            if match := _HEADER_FAMILY.match(line):
                rfam_id, accession = match.groups()
            elif match := _HEADER_RELEASE.match(line):
                release = match.group(1)
            elif match := _HEADER_COUNT.match(line):
                total = int(match.group(1))
            continue
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != _REGION_COLUMNS:
            raise ValueError(f"Unexpected Rfam regions row ({len(fields)} columns): {line[:200]}")
        seq_acc, score, first, last, description, species, taxid = fields
        first_pos, last_pos = int(first), int(last)
        regions.append(
            RfamRegion(
                sequence_accession=seq_acc,
                bit_score=float(score),
                start=min(first_pos, last_pos),
                end=max(first_pos, last_pos),
                strand="+" if first_pos <= last_pos else "-",
                description=description,
                species=species,
                taxid=int(taxid),
            )
        )
    if not accession:
        raise ValueError(f"Rfam regions for {family!r} came back without a family header")
    return accession, rfam_id, release, total if total is not None else len(regions), regions


def _matches(region: RfamRegion, inputs: RfamRegionsInput) -> bool:
    """Apply the optional taxid / species / sequence filters to one region."""
    if inputs.taxid is not None and region.taxid != inputs.taxid:
        return False
    if inputs.species is not None and inputs.species.casefold() not in region.species.casefold():
        return False
    if inputs.sequence_accession is not None:
        wanted = inputs.sequence_accession
        if region.sequence_accession != wanted and region.sequence_accession.split(".")[0] != wanted:
            return False
    return True


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return RfamRegionsInput(family="RF01731", taxid=216595)


@tool(
    key="rfam-regions",
    local_only="rfam-regions does not use a gpu and does not need an environment, so it can run in process",
    label="Rfam Regions",
    category="database_retrieval",
    input_class=RfamRegionsInput,
    config_class=RfamRegionsConfig,
    output_class=RfamRegionsOutput,
    description="List where an Rfam RNA family is annotated in genomes, with coordinates and strand",
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_rfam_regions(
    inputs: RfamRegionsInput,
    config: RfamRegionsConfig,
    instance: Any = None,
) -> RfamRegionsOutput:
    """List the sequence regions an Rfam family is annotated in.

    Coordinates come back with start <= end and the strand stated separately,
    so a hit can be passed straight to ncbi-efetch's seq_start, seq_stop, and strand.

    Args:
        inputs (RfamRegionsInput): Family plus optional taxid, species, and sequence filters.
        config (RfamRegionsConfig): Cap on the number of regions returned.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        RfamRegionsOutput: Family identity, region counts, and the matched regions.
    """
    del instance

    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
    )
    try:
        response = _rfam_get(session, _family_url(inputs.family, "regions"), {"content-type": "text/plain"})
        if response is None:
            raise _not_found(inputs.family)
        accession, rfam_id, release, total, regions = _parse_regions(response.text, inputs.family)
    finally:
        session.close()

    matched = [r for r in regions if _matches(r, inputs)]
    return RfamRegionsOutput(
        accession=accession,
        rfam_id=rfam_id,
        rfam_release=release,
        total_regions=total,
        matched_regions=len(matched),
        truncated=len(matched) > config.max_regions,
        regions=matched[: config.max_regions],
    )
