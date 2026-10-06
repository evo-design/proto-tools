"""proto_tools/tools/database_retrieval/rfam/rfam_family.py.

Fetches an Rfam family's curation record, consensus secondary structure, and
seed alignment from the Rfam website.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import Field

from proto_tools.tools.database_retrieval.rfam.shared_data_models import (
    RfamFamilyQuery,
    _family_url,
    _not_found,
    _rfam_get,
    _rfam_session,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, BaseToolOutput, ConfigField

# ============================================================================
# Data Models
# ============================================================================


class RfamFamilyInput(RfamFamilyQuery):
    """Input for fetching an Rfam family.

    Attributes:
        family (str): Rfam accession (e.g. 'RF01731') or family ID (e.g. 'TwoAYGGAY').
    """


class RfamFamilyConfig(BaseConfig):
    """Configuration for fetching an Rfam family.

    Attributes:
        include_seed_alignment (bool): Return the full Stockholm seed alignment, not
            only its consensus lines.
    """

    include_seed_alignment: bool = ConfigField(
        default=False,
        title="Include Seed Alignment",
        description="Return the full Stockholm seed alignment, not only its consensus lines",
    )


class RfamFamilyOutput(BaseToolOutput):
    """Output from fetching an Rfam family.

    Attributes:
        accession (str): Rfam accession.
        rfam_id (str): Rfam family ID.
        description (str): One-line family description.
        rna_type (str): Rfam type annotation (e.g. 'Cis-reg', 'Gene; snRNA').
        comment (str | None): Curator comment, when the family has one.
        author (str): Family authors.
        seed_source (str): Where the seed alignment came from.
        structure_source (str | None): Where the consensus structure came from.
        num_seed (int): Sequences in the seed alignment.
        num_full (int): Annotated regions across all sequences.
        num_species (int): Species with at least one annotated region.
        clan_accession (str | None): Accession of the clan the family belongs to.
        clan_id (str | None): ID of the clan the family belongs to.
        gathering_cutoff (float): Bit-score threshold for family membership.
        trusted_cutoff (float): Lowest bit score of a true member.
        noise_cutoff (float): Highest bit score of a non-member.
        rfam_release (str): Rfam release the record comes from.
        rfam_release_date (str): Date of that release.
        consensus_structure (str): Consensus secondary structure (#=GC SS_cons), WUSS notation.
        consensus_sequence (str): Reference consensus sequence (#=GC RF), aligned with it.
        seed_alignment (str | None): Full Stockholm seed alignment, when requested.
    """

    accession: str = Field(title="Accession", description="Rfam accession")
    rfam_id: str = Field(title="Rfam ID", description="Rfam family ID")
    description: str = Field(title="Description", description="One-line family description")
    rna_type: str = Field(title="RNA Type", description="Rfam type annotation (e.g. 'Cis-reg', 'Gene; snRNA')")
    comment: str | None = Field(default=None, title="Comment", description="Curator comment, when present")
    author: str = Field(title="Author", description="Family authors")
    seed_source: str = Field(title="Seed Source", description="Where the seed alignment came from")
    structure_source: str | None = Field(
        default=None, title="Structure Source", description="Where the consensus structure came from"
    )
    num_seed: int = Field(title="Seed Sequences", description="Sequences in the seed alignment")
    num_full: int = Field(title="Full Regions", description="Annotated regions across all sequences")
    num_species: int = Field(title="Species", description="Species with at least one annotated region")
    clan_accession: str | None = Field(default=None, title="Clan Accession", description="Accession of the clan")
    clan_id: str | None = Field(default=None, title="Clan ID", description="ID of the clan")
    gathering_cutoff: float = Field(title="Gathering Cutoff", description="Bit-score threshold for membership")
    trusted_cutoff: float = Field(title="Trusted Cutoff", description="Lowest bit score of a true member")
    noise_cutoff: float = Field(title="Noise Cutoff", description="Highest bit score of a non-member")
    rfam_release: str = Field(title="Rfam Release", description="Rfam release the record comes from")
    rfam_release_date: str = Field(title="Rfam Release Date", description="Date of that release")
    consensus_structure: str = Field(
        title="Consensus Structure", description="Consensus secondary structure (#=GC SS_cons), WUSS notation"
    )
    consensus_sequence: str = Field(
        title="Consensus Sequence", description="Reference consensus sequence (#=GC RF), column-aligned"
    )
    seed_alignment: str | None = Field(
        default=None, title="Seed Alignment", description="Full Stockholm seed alignment, when requested"
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "sto"]

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
        if file_format == "sto":
            if self.seed_alignment is None:
                raise ValueError("No seed alignment to export; rerun with include_seed_alignment=True")
            path.write_text(self.seed_alignment.rstrip("\n") + "\n", encoding="utf-8")
            return
        raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Private Helpers
# ============================================================================


def _consensus_lines(stockholm: str) -> tuple[str, str]:
    """Join the #=GC SS_cons and #=GC RF lines across every block of a Stockholm alignment."""
    structure: list[str] = []
    sequence: list[str] = []
    for line in stockholm.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "#=GC":
            if parts[1] == "SS_cons":
                structure.append(parts[2])
            elif parts[1] == "RF":
                sequence.append(parts[2])
    if not structure:
        raise ValueError("Rfam seed alignment has no #=GC SS_cons line")
    return "".join(structure), "".join(sequence)


def _clean(value: Any) -> str | None:
    """Rfam ends curation fields with ';' and marks absent ones as null or ''; normalize both."""
    if value is None:
        return None
    return str(value).strip().rstrip(";").strip() or None


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return RfamFamilyInput(family="RF01731")


@tool(
    key="rfam-family",
    local_only="rfam-family does not use a gpu and does not need an environment, so it can run in process",
    label="Rfam Family",
    category="database_retrieval",
    input_class=RfamFamilyInput,
    config_class=RfamFamilyConfig,
    output_class=RfamFamilyOutput,
    description="Fetch an Rfam RNA family's curation record, consensus structure, and seed alignment",
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_rfam_family(
    inputs: RfamFamilyInput,
    config: RfamFamilyConfig,
    instance: Any = None,
) -> RfamFamilyOutput:
    """Fetch an Rfam family record and its consensus structure.

    The consensus structure and sequence are always read from the seed
    alignment; the alignment itself is returned only when requested.

    Args:
        inputs (RfamFamilyInput): Rfam accession or family ID.
        config (RfamFamilyConfig): Whether to return the full seed alignment.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        RfamFamilyOutput: Curation record, cutoffs, release, and consensus lines.
    """
    del instance

    session = _rfam_session(config)
    try:
        record_response = _rfam_get(session, _family_url(inputs.family), {"content-type": "application/json"})
        if record_response is None:
            raise _not_found(inputs.family)
        record = record_response.json()["rfam"]
        accession = record["acc"]
        alignment_response = _rfam_get(session, _family_url(accession, "alignment", "stockholm"))
        if alignment_response is None:
            raise ValueError(f"Rfam has no seed alignment for {accession}")
        stockholm = alignment_response.text
    finally:
        session.close()

    structure, sequence = _consensus_lines(stockholm)
    curation = record["curation"]
    cutoffs = record["cm"]["cutoffs"]
    clan = record.get("clan") or {}
    return RfamFamilyOutput(
        accession=accession,
        rfam_id=record["id"],
        description=record["description"],
        rna_type=_clean(curation["type"]) or "",
        comment=_clean(record.get("comment")),
        author=curation["author"],
        seed_source=_clean(curation["seed_source"]) or "",
        structure_source=_clean(curation.get("structure_source")),
        num_seed=curation["num_seed"],
        num_full=curation["num_full"],
        num_species=curation["num_species"],
        clan_accession=clan.get("acc"),
        clan_id=clan.get("id"),
        gathering_cutoff=cutoffs["gathering"],
        trusted_cutoff=cutoffs["trusted"],
        noise_cutoff=cutoffs["noise"],
        rfam_release=record["release"]["number"],
        rfam_release_date=record["release"]["date"],
        consensus_structure=structure,
        consensus_sequence=sequence,
        seed_alignment=stockholm if config.include_seed_alignment else None,
    )
