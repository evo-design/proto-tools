"""proto_tools/tools/database_retrieval/alphamissense/alphamissense_fetch.py.

Fetches per-residue, per-substitution AlphaMissense pathogenicity scores for
human proteins by UniProt accession, served as CSV via the AlphaFold
Protein Structure Database.
"""

import csv
import io
import json
import logging
from pathlib import Path
from typing import Any, Literal

import requests
from pydantic import BaseModel, Field

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    build_http_session,
)

logger = logging.getLogger(__name__)

_AFDB_FILES_BASE = "https://alphafold.ebi.ac.uk/files"
_REQUEST_TIMEOUT_SECONDS = 15
_HTTP_RETRIES = 2
_BACKOFF_SECONDS = 1.0
_USER_AGENT = "proto-tools/alphamissense-fetch-v1"

AlphaMissenseClass = Literal["likely_benign", "ambiguous", "likely_pathogenic"]
CoordinateSystem = Literal["uniprot", "hg19", "hg38"]

# UniProt-coordinate CSV uses abbreviated class codes; the genomic CSVs use the
# already-expanded names. Both map to the canonical AlphaMissenseClass values.
_AM_CLASS_MAP: dict[str, AlphaMissenseClass] = {
    "LBen": "likely_benign",
    "Amb": "ambiguous",
    "LPath": "likely_pathogenic",
    "likely_benign": "likely_benign",
    "ambiguous": "ambiguous",
    "likely_pathogenic": "likely_pathogenic",
}


# ============================================================================
# Data Models
# ============================================================================


class AlphaMissensePrediction(BaseModel):
    """One AlphaMissense pathogenicity prediction for a single substitution.

    In genomic mode (hg19/hg38), one protein variant may appear multiple
    times when it's reachable by more than one SNV.

    Attributes:
        position (int): 1-indexed residue position in the canonical UniProt sequence.
        wild_type_aa (str): Single-letter wild-type amino acid at this position.
        alt_aa (str): Single-letter alternate amino acid being scored.
        pathogenicity_score (float): AlphaMissense pathogenicity score (0.0-1.0).
            Higher values indicate the variant is more likely to be pathogenic.
        classification (AlphaMissenseClass): AlphaMissense class label
            ('likely_benign', 'ambiguous', or 'likely_pathogenic').
        chrom (str | None): Chromosome (e.g. 'chr17'); populated only for
            genomic-coordinate fetches.
        pos (int | None): 1-indexed genomic position; populated only for
            genomic-coordinate fetches.
        ref (str | None): Reference allele; populated only for genomic-
            coordinate fetches.
        alt (str | None): Alternate allele; populated only for genomic-
            coordinate fetches.
        transcript_id (str | None): GENCODE transcript identifier; populated
            only for genomic-coordinate fetches.
    """

    position: int = Field(description="1-indexed residue position", ge=1)
    wild_type_aa: str = Field(description="Wild-type amino acid (single letter)", min_length=1, max_length=1)
    alt_aa: str = Field(description="Alternate amino acid (single letter)", min_length=1, max_length=1)
    pathogenicity_score: float = Field(description="Pathogenicity score in [0, 1]", ge=0.0, le=1.0)
    classification: AlphaMissenseClass = Field(description="AlphaMissense classification")
    chrom: str | None = Field(default=None, description="Chromosome (genomic mode only)")
    pos: int | None = Field(default=None, description="1-indexed genomic position (genomic mode only)")
    ref: str | None = Field(default=None, description="Reference allele (genomic mode only)")
    alt: str | None = Field(default=None, description="Alternate allele (genomic mode only)")
    transcript_id: str | None = Field(default=None, description="GENCODE transcript ID (genomic mode only)")


class AlphaMissenseFetchInput(BaseToolInput):
    """Input for AlphaMissense fetch.

    Attributes:
        uniprot_id (str): UniProt accession (must be a human protein covered by
            AlphaMissense; e.g. 'P04637').
    """

    uniprot_id: str = InputField(description="UniProt accession (human; e.g. 'P04637')")


class AlphaMissenseFetchConfig(BaseConfig):
    """Configuration for AlphaMissense fetch.

    AlphaMissense is a static CSV at AFDB with no server-side filtering;
    filter ``output.predictions`` client-side.

    Attributes:
        coordinate_system (CoordinateSystem): Which AFDB CSV to fetch.
            ``"uniprot"`` (default) returns the full protein-coordinate
            saturation grid (~7,500 rows for TP53). ``"hg19"`` / ``"hg38"``
            return SNV-accessible substitutions in genomic coordinates
            (~2,500 rows for TP53) and populate ``chrom`` / ``pos`` / ``ref``
            / ``alt`` / ``transcript_id`` on each prediction; a protein
            variant reachable by multiple SNVs appears multiple times.
    """

    coordinate_system: CoordinateSystem = ConfigField(
        title="Coordinate System",
        default="uniprot",
        description="AFDB CSV variant: 'uniprot' (protein coords) or 'hg19'/'hg38' (genomic coords)",
    )


class AlphaMissenseFetchOutput(BaseToolOutput):
    """Output from AlphaMissense fetch.

    Attributes:
        uniprot_accession (str): UniProt accession that was looked up.
        predictions (list[AlphaMissensePrediction]): All per-substitution
            pathogenicity predictions in the source CSV (full saturation grid:
            ``sequence_length * 19`` for UniProt-coordinate fetches).
        num_predictions (int): Number of predictions in the source CSV.
        mean_pathogenicity (float | None): Mean pathogenicity score across all
            predictions; None when `predictions` is empty.
        source_url (str): URL of the AlphaMissense CSV that was fetched.
    """

    uniprot_accession: str = Field(description="UniProt accession looked up")
    predictions: list[AlphaMissensePrediction] = Field(
        default_factory=list, description="Per-substitution pathogenicity predictions"
    )
    num_predictions: int = Field(description="Number of predictions in the source CSV", ge=0)
    mean_pathogenicity: float | None = Field(
        default=None, description="Mean pathogenicity score across all predictions"
    )
    source_url: str = Field(description="URL of the AlphaMissense CSV fetched")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Any, file_format: str) -> None:
        if file_format == "json":
            path = Path(export_path).with_suffix(".json")
            with path.open("w", encoding="utf-8") as f:
                json.dump(self.model_dump(mode="json"), f, indent=2)
            return
        raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return AlphaMissenseFetchInput(uniprot_id="P04637")


@tool(
    key="alphamissense-fetch",
    label="AlphaMissense Fetch",
    category="database_retrieval",
    input_class=AlphaMissenseFetchInput,
    config_class=AlphaMissenseFetchConfig,
    output_class=AlphaMissenseFetchOutput,
    description=(
        "Fetch per-residue, per-substitution AlphaMissense pathogenicity scores for a "
        "human UniProt accession from the AlphaFold Protein Structure Database"
    ),
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_alphamissense_fetch(
    inputs: AlphaMissenseFetchInput,
    config: AlphaMissenseFetchConfig,
    instance: Any = None,
) -> AlphaMissenseFetchOutput:
    """Fetch AlphaMissense pathogenicity scores for a UniProt accession.

    AlphaMissense covers all reviewed human UniProt proteins. Non-human accessions
    raise ValueError. Returns the full saturation grid (UniProt coords) or the
    SNV-accessible subset (genomic coords); filter the output client-side as needed.

    Args:
        inputs (AlphaMissenseFetchInput): UniProt accession to look up.
        config (AlphaMissenseFetchConfig): `coordinate_system` selects which
            AFDB CSV variant to fetch.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        AlphaMissenseFetchOutput: Per-substitution predictions, count, mean
            pathogenicity score, and the source URL.
    """
    del instance

    accession = inputs.uniprot_id.strip().upper()
    csv_url = _csv_url_for(accession, config.coordinate_system)
    is_genomic = config.coordinate_system != "uniprot"

    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
    )

    try:
        rows = _fetch_csv(csv_url, session)
        if rows is None:
            raise ValueError(
                f"AlphaMissense has no predictions for accession '{accession}'. "
                "Coverage is human-only; check that the accession is a reviewed human protein."
            )

        predictions = [_parse_row(row, csv_url, is_genomic=is_genomic) for row in rows]
        mean = sum(p.pathogenicity_score for p in predictions) / len(predictions) if predictions else None

        return AlphaMissenseFetchOutput(
            uniprot_accession=accession,
            predictions=predictions,
            num_predictions=len(predictions),
            mean_pathogenicity=mean,
            source_url=csv_url,
        )
    finally:
        session.close()


# ============================================================================
# Private Helpers
# ============================================================================


def _csv_url_for(accession: str, coordinate_system: CoordinateSystem) -> str:
    """Build the AFDB CSV URL for the requested coordinate system."""
    suffix = {
        "uniprot": "aa-substitutions",
        "hg19": "hg19",
        "hg38": "hg38",
    }[coordinate_system]
    return f"{_AFDB_FILES_BASE}/AF-{accession}-F1-{suffix}.csv"


def _fetch_csv(url: str, session: requests.Session) -> list[dict[str, str]] | None:
    """Fetch and parse the AlphaMissense aa-substitutions CSV. Returns None on 404."""
    response = session.get(url, timeout=_REQUEST_TIMEOUT_SECONDS)
    if response.status_code == 404:
        logger.debug("AlphaMissense CSV not found at %s", url)
        return None
    response.raise_for_status()
    reader = csv.DictReader(io.StringIO(response.text))
    return list(reader)


def _parse_row(row: dict[str, str], source_url: str, *, is_genomic: bool = False) -> AlphaMissensePrediction:
    """Parse one CSV row into an AlphaMissensePrediction.

    Required CSV columns are accessed with bare `row[key]` so that a missing
    column raises KeyError -- a real schema regression in the upstream CSV
    that the @tool decorator should surface, not silently coerce to "".

    The UniProt-coordinate CSV has 3 columns; the hg19/hg38 CSVs have 10
    columns and additionally populate the genomic-coordinate fields.
    """
    variant = row["protein_variant"].strip()
    if len(variant) < 3 or not variant[1:-1].isdigit():
        raise ValueError(f"Malformed protein_variant '{variant}' in AlphaMissense CSV at {source_url}")
    am_class_raw = row["am_class"].strip()
    classification = _AM_CLASS_MAP.get(am_class_raw)
    if classification is None:
        raise ValueError(f"Unknown am_class '{am_class_raw}' in AlphaMissense CSV at {source_url}")
    fields: dict[str, Any] = {
        "position": int(variant[1:-1]),
        "wild_type_aa": variant[0],
        "alt_aa": variant[-1],
        "pathogenicity_score": float(row["am_pathogenicity"]),
        "classification": classification,
    }
    if is_genomic:
        fields["chrom"] = row["CHROM"]
        fields["pos"] = int(row["POS"])
        fields["ref"] = row["REF"]
        fields["alt"] = row["ALT"]
        fields["transcript_id"] = row["transcript_id"]
    return AlphaMissensePrediction(**fields)
