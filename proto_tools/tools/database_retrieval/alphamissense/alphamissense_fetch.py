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

AlphaMissenseClass = Literal["likely_benign", "ambiguous", "likely_pathogenic"]

_AM_CLASS_MAP: dict[str, AlphaMissenseClass] = {
    "LBen": "likely_benign",
    "Amb": "ambiguous",
    "LPath": "likely_pathogenic",
}


# ============================================================================
# Data Models
# ============================================================================


class AlphaMissensePrediction(BaseModel):
    """One AlphaMissense pathogenicity prediction for a single substitution.

    Attributes:
        position (int): 1-indexed residue position in the canonical UniProt sequence.
        wild_type_aa (str): Single-letter wild-type amino acid at this position.
        alt_aa (str): Single-letter alternate amino acid being scored.
        pathogenicity_score (float): AlphaMissense pathogenicity score (0.0-1.0).
            Higher values indicate the variant is more likely to be pathogenic.
        classification (AlphaMissenseClass): AlphaMissense class label
            ('likely_benign', 'ambiguous', or 'likely_pathogenic').
    """

    position: int = Field(description="1-indexed residue position", ge=1)
    wild_type_aa: str = Field(description="Wild-type amino acid (single letter)", min_length=1, max_length=1)
    alt_aa: str = Field(description="Alternate amino acid (single letter)", min_length=1, max_length=1)
    pathogenicity_score: float = Field(description="Pathogenicity score in [0, 1]", ge=0.0, le=1.0)
    classification: AlphaMissenseClass = Field(description="AlphaMissense classification")


class AlphaMissenseFetchInput(BaseToolInput):
    """Input for AlphaMissense fetch.

    Attributes:
        uniprot_id (str): UniProt accession (must be a human protein covered by
            AlphaMissense; e.g. 'P04637').
    """

    uniprot_id: str = InputField(description="UniProt accession (human; e.g. 'P04637')")


class AlphaMissenseFetchConfig(BaseConfig):
    """Configuration for AlphaMissense fetch.

    Attributes:
        positions (list[int] | None): If set, return only predictions whose 1-indexed
            position is in this list. None returns all positions.
        alt_residues (list[str] | None): If set, return only predictions whose alt
            amino acid (single letter) is in this list. None returns all alts.
        min_pathogenicity (float | None): If set, return only predictions with
            pathogenicity score at least this value (0.0-1.0).
        max_pathogenicity (float | None): If set, return only predictions with
            pathogenicity score at most this value (0.0-1.0).
        classification_filter (list[AlphaMissenseClass] | None): If set, return only
            predictions whose classification is in this list.
        request_timeout_seconds (int): HTTP timeout per request.
        http_retries (int): Number of retries for failed requests.
        backoff_seconds (float): Seconds to wait between retries (doubles after each
            attempt).
        user_agent (str): Identifier string sent to the AlphaFold DB API with each
            request.
    """

    positions: list[int] | None = ConfigField(
        title="Positions Filter",
        default=None,
        description="If set, return only predictions at these 1-indexed positions",
    )
    alt_residues: list[str] | None = ConfigField(
        title="Alt Residues Filter",
        default=None,
        description="If set, return only predictions to these alternate amino acids",
    )
    min_pathogenicity: float | None = ConfigField(
        title="Minimum Pathogenicity",
        default=None,
        ge=0.0,
        le=1.0,
        description="If set, drop predictions with score below this threshold",
    )
    max_pathogenicity: float | None = ConfigField(
        title="Maximum Pathogenicity",
        default=None,
        ge=0.0,
        le=1.0,
        description="If set, drop predictions with score above this threshold",
    )
    classification_filter: list[AlphaMissenseClass] | None = ConfigField(
        title="Classification Filter",
        default=None,
        description="If set, return only predictions matching these classifications",
    )
    request_timeout_seconds: int = ConfigField(
        title="Request Timeout",
        default=15,
        ge=1,
        description="HTTP timeout in seconds",
        advanced=True,
    )
    http_retries: int = ConfigField(
        title="HTTP Retries",
        default=2,
        ge=0,
        description="Retries for HTTP requests",
        advanced=True,
    )
    backoff_seconds: float = ConfigField(
        title="Backoff Seconds",
        default=1.0,
        ge=0.0,
        description="Seconds to wait between retries (doubles after each attempt)",
        advanced=True,
    )
    user_agent: str = ConfigField(
        title="User Agent",
        default="proto-tools/alphamissense-fetch-v1",
        description="Identifier string sent to the AlphaFold DB API with each request",
        advanced=True,
    )


class AlphaMissenseFetchOutput(BaseToolOutput):
    """Output from AlphaMissense fetch.

    Attributes:
        uniprot_accession (str): UniProt accession that was looked up.
        predictions (list[AlphaMissensePrediction]): Per-substitution pathogenicity
            predictions, after any filters in Config have been applied.
        num_total_predictions (int): Number of predictions in the source CSV before
            any filters were applied.
        num_returned (int): Number of predictions in `predictions` after filtering.
        mean_pathogenicity (float | None): Mean pathogenicity score across the
            returned predictions; None when `predictions` is empty.
        source_url (str): URL of the AlphaMissense CSV that was fetched.
    """

    uniprot_accession: str = Field(description="UniProt accession looked up")
    predictions: list[AlphaMissensePrediction] = Field(
        default_factory=list, description="Per-substitution pathogenicity predictions"
    )
    num_total_predictions: int = Field(description="Number of predictions in the source CSV before filtering", ge=0)
    num_returned: int = Field(description="Number of predictions returned after filtering", ge=0)
    mean_pathogenicity: float | None = Field(
        default=None, description="Mean pathogenicity score across returned predictions"
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
)
def run_alphamissense_fetch(
    inputs: AlphaMissenseFetchInput,
    config: AlphaMissenseFetchConfig,
    instance: Any = None,
) -> AlphaMissenseFetchOutput:
    """Fetch AlphaMissense pathogenicity scores for a UniProt accession.

    AlphaMissense covers all reviewed human UniProt proteins. Non-human accessions
    raise ValueError. Filters in Config are applied after the CSV is downloaded.

    Args:
        inputs (AlphaMissenseFetchInput): UniProt accession to look up.
        config (AlphaMissenseFetchConfig): Optional filters + HTTP retry settings.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        AlphaMissenseFetchOutput: Filtered per-substitution predictions, plus
            counts and the source URL.
    """
    del instance

    accession = inputs.uniprot_id.strip().upper()
    csv_url = f"{_AFDB_FILES_BASE}/AF-{accession}-F1-aa-substitutions.csv"

    session = build_http_session(
        http_retries=config.http_retries,
        backoff_seconds=config.backoff_seconds,
        user_agent=config.user_agent,
    )

    try:
        rows = _fetch_csv(csv_url, config, session)
        if rows is None:
            raise ValueError(
                f"AlphaMissense has no predictions for accession '{accession}'. "
                "Coverage is human-only; check that the accession is a reviewed human protein."
            )

        all_predictions = [_parse_row(row, csv_url) for row in rows]
        filtered = _apply_filters(all_predictions, config)
        mean = sum(p.pathogenicity_score for p in filtered) / len(filtered) if filtered else None

        return AlphaMissenseFetchOutput(
            uniprot_accession=accession,
            predictions=filtered,
            num_total_predictions=len(all_predictions),
            num_returned=len(filtered),
            mean_pathogenicity=mean,
            source_url=csv_url,
        )
    finally:
        session.close()


# ============================================================================
# Private Helpers
# ============================================================================


def _fetch_csv(
    url: str,
    config: AlphaMissenseFetchConfig,
    session: requests.Session,
) -> list[dict[str, str]] | None:
    """Fetch and parse the AlphaMissense aa-substitutions CSV. Returns None on 404."""
    response = session.get(url, timeout=config.request_timeout_seconds)
    if response.status_code == 404:
        logger.debug("AlphaMissense CSV not found at %s", url)
        return None
    response.raise_for_status()
    reader = csv.DictReader(io.StringIO(response.text))
    return list(reader)


def _parse_row(row: dict[str, str], source_url: str) -> AlphaMissensePrediction:
    """Parse one CSV row into an AlphaMissensePrediction.

    Required CSV columns are accessed with bare `row[key]` so that a missing
    column raises KeyError -- a real schema regression in the upstream CSV
    that the @tool decorator should surface, not silently coerce to "".
    """
    variant = row["protein_variant"].strip()
    if len(variant) < 3 or not variant[1:-1].isdigit():
        raise ValueError(f"Malformed protein_variant '{variant}' in AlphaMissense CSV at {source_url}")
    am_class_raw = row["am_class"].strip()
    classification = _AM_CLASS_MAP.get(am_class_raw)
    if classification is None:
        raise ValueError(f"Unknown am_class '{am_class_raw}' in AlphaMissense CSV at {source_url}")
    return AlphaMissensePrediction(
        position=int(variant[1:-1]),
        wild_type_aa=variant[0],
        alt_aa=variant[-1],
        pathogenicity_score=float(row["am_pathogenicity"]),
        classification=classification,
    )


def _apply_filters(
    predictions: list[AlphaMissensePrediction],
    config: AlphaMissenseFetchConfig,
) -> list[AlphaMissensePrediction]:
    """Apply position / alt / score / classification filters from Config."""
    positions = set(config.positions) if config.positions is not None else None
    alts = {a.upper() for a in config.alt_residues} if config.alt_residues is not None else None
    classes = set(config.classification_filter) if config.classification_filter is not None else None

    out = []
    for p in predictions:
        if positions is not None and p.position not in positions:
            continue
        if alts is not None and p.alt_aa.upper() not in alts:
            continue
        if config.min_pathogenicity is not None and p.pathogenicity_score < config.min_pathogenicity:
            continue
        if config.max_pathogenicity is not None and p.pathogenicity_score > config.max_pathogenicity:
            continue
        if classes is not None and p.classification not in classes:
            continue
        out.append(p)
    return out
