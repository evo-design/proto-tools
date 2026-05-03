"""proto_tools/tools/database_retrieval/alphafold_db/alphafold_db_fetch.py.

Fetches AlphaFold-predicted structures, per-residue pLDDT, and PAE matrices
from the AlphaFold Protein Structure Database by UniProt accession.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

import requests
from pydantic import Field

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

_AFDB_API_BASE = "https://alphafold.ebi.ac.uk/api/prediction"


# ============================================================================
# Data Models
# ============================================================================


class AlphaFoldDBFetchInput(BaseToolInput):
    """Input for AlphaFold DB fetch.

    Attributes:
        uniprot_id (str): UniProt accession to look up (e.g. 'P04637').
    """

    uniprot_id: str = InputField(description="UniProt accession (e.g. 'P04637')")


class AlphaFoldDBFetchConfig(BaseConfig):
    """Configuration for AlphaFold DB fetch.

    Attributes:
        structure_format (Literal["pdb", "cif"]): Structure file format to download.
        include_structure (bool): If True, download the structure file text.
        include_plddt (bool): If True, download the per-residue pLDDT JSON.
        include_pae (bool): If True, download the PAE (predicted aligned error)
            matrix. PAE files can be large for long proteins; default off.
        request_timeout_seconds (int): HTTP timeout per request.
        http_retries (int): Number of retries for failed requests.
        backoff_seconds (float): Seconds to wait between retries (doubles after each
            attempt).
        user_agent (str): Identifier string sent to the AlphaFold DB API with each
            request.
    """

    structure_format: Literal["pdb", "cif"] = ConfigField(
        title="Structure Format",
        default="pdb",
        description="Structure file format to download (pdb or mmCIF)",
    )
    include_structure: bool = ConfigField(
        title="Include Structure Text",
        default=True,
        description="Download the structure file text into the output",
    )
    include_plddt: bool = ConfigField(
        title="Include Per-Residue pLDDT",
        default=True,
        description="Download the per-residue pLDDT confidence array",
    )
    include_pae: bool = ConfigField(
        title="Include PAE Matrix",
        default=False,
        description="Download the PAE (predicted aligned error) matrix; large for long proteins",
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
        default="proto-tools/alphafold-db-fetch-v1",
        description="Identifier string sent to the AlphaFold DB API with each request",
        advanced=True,
    )


class AlphaFoldDBFetchOutput(BaseToolOutput):
    """Output from AlphaFold DB fetch.

    Attributes:
        uniprot_accession (str): Primary UniProt accession that was looked up.
        entry_id (str): AlphaFold entry identifier (e.g. 'AF-P04637-F1').
        gene (str | None): Gene symbol from the AlphaFold record.
        organism_scientific_name (str | None): Source organism scientific name.
        tax_id (int | None): NCBI taxonomy ID.
        sequence (str): Amino-acid sequence covered by the prediction.
        sequence_length (int): Length of the predicted sequence.
        sequence_start (int): 1-indexed start residue of the prediction (relative
            to the full UniProt sequence; >1 only for non-first fragments of very
            long proteins).
        sequence_end (int): 1-indexed inclusive end residue of the prediction.
        latest_version (int): Latest version of the AlphaFold DB prediction (this is
            the version of the served prediction; AlphaFold DB always serves the
            latest).
        model_created_date (str | None): ISO 8601 timestamp when this prediction was
            generated.
        mean_plddt (float | None): Mean per-residue pLDDT for the prediction
            (AlphaFold DB's globalMetricValue field).
        pdb_url (str): URL to the PDB structure file on AlphaFold DB.
        cif_url (str): URL to the mmCIF structure file on AlphaFold DB.
        pae_doc_url (str): URL to the PAE JSON document on AlphaFold DB.
        plddt_doc_url (str): URL to the per-residue pLDDT JSON document on
            AlphaFold DB.
        pae_image_url (str): URL to the rendered PAE PNG on AlphaFold DB.
        msa_url (str | None): URL to the MSA A3M used for prediction, when present.
        structure_format (str): Format of the structure file downloaded ('pdb' or
            'cif'); empty when structure was not requested.
        structure_text (str | None): Structure file contents in `structure_format`.
            None when `include_structure` is False.
        plddt_per_residue (list[float] | None): Per-residue pLDDT scores (0-100).
            None when `include_plddt` is False.
        pae_matrix (list[list[float]] | None): NxN predicted aligned error matrix
            in angstroms, indexed by aligned-pair (residue i, residue j). None when
            `include_pae` is False.
        source_url (str): AlphaFold DB API URL used for the metadata lookup.
        raw_entry (dict[str, Any]): Complete AlphaFold DB JSON record for advanced
            programmatic access.
    """

    uniprot_accession: str = Field(description="Primary UniProt accession")
    entry_id: str = Field(description="AlphaFold entry identifier (e.g. 'AF-P04637-F1')")
    gene: str | None = Field(default=None, description="Gene symbol")
    organism_scientific_name: str | None = Field(default=None, description="Source organism scientific name")
    tax_id: int | None = Field(default=None, description="NCBI taxonomy ID")
    sequence: str = Field(description="Amino-acid sequence covered by the prediction")
    sequence_length: int = Field(description="Length of the predicted sequence")
    sequence_start: int = Field(description="1-indexed start residue of the prediction")
    sequence_end: int = Field(description="1-indexed inclusive end residue of the prediction")
    latest_version: int = Field(
        description="Latest AlphaFold DB version of this prediction (always the served version)"
    )
    model_created_date: str | None = Field(default=None, description="ISO 8601 model creation timestamp")
    mean_plddt: float | None = Field(default=None, description="Mean per-residue pLDDT")
    pdb_url: str = Field(description="URL to PDB structure file")
    cif_url: str = Field(description="URL to mmCIF structure file")
    pae_doc_url: str = Field(description="URL to PAE JSON document")
    plddt_doc_url: str = Field(description="URL to per-residue pLDDT JSON document")
    pae_image_url: str = Field(description="URL to rendered PAE PNG")
    msa_url: str | None = Field(default=None, description="URL to MSA A3M file, when present")
    structure_format: str = Field(default="", description="Format of downloaded structure ('pdb' or 'cif')")
    structure_text: str | None = Field(default=None, description="Structure file contents")
    plddt_per_residue: list[float] | None = Field(default=None, description="Per-residue pLDDT (0-100)")
    pae_matrix: list[list[float]] | None = Field(default=None, description="NxN predicted aligned error matrix")
    source_url: str = Field(description="AlphaFold DB API URL used for lookup")
    raw_entry: dict[str, Any] = Field(
        default_factory=dict, description="Complete AlphaFold DB JSON record for advanced programmatic access"
    )

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
    return AlphaFoldDBFetchInput(uniprot_id="P04637")


@tool(
    key="alphafold-db-fetch",
    label="AlphaFold DB Fetch",
    category="database_retrieval",
    input_class=AlphaFoldDBFetchInput,
    config_class=AlphaFoldDBFetchConfig,
    output_class=AlphaFoldDBFetchOutput,
    description=(
        "Fetch predicted structure (PDB/mmCIF), per-residue pLDDT, and PAE "
        "matrix from the AlphaFold Protein Structure Database by UniProt "
        "accession"
    ),
    uses_gpu=False,
    example_input=example_input,
)
def run_alphafold_db_fetch(
    inputs: AlphaFoldDBFetchInput,
    config: AlphaFoldDBFetchConfig,
    instance: Any = None,
) -> AlphaFoldDBFetchOutput:
    """Fetch a prediction record from AlphaFold DB.

    Returns the prediction metadata, file URLs, and optionally the structure
    text, per-residue pLDDT array, and PAE matrix. AlphaFold DB returns
    multiple records when the protein has annotated alternative isoforms
    (common for human proteins; e.g. P04637 / TP53 returns 9 records, one
    per isoform) and when the canonical sequence is split into overlapping
    fragments (rare; only proteins longer than ~2,700 residues). In both
    cases the wrapper returns the first record (the canonical isoform's
    first fragment) and logs a warning when there is more than one.

    Args:
        inputs (AlphaFoldDBFetchInput): UniProt accession to look up.
        config (AlphaFoldDBFetchConfig): Format selection, payload toggles, and
            HTTP retry settings.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        AlphaFoldDBFetchOutput: Prediction metadata, URLs, and optionally
            structure text, per-residue pLDDT, and PAE matrix.
    """
    del instance

    accession = inputs.uniprot_id.strip().upper()
    api_url = f"{_AFDB_API_BASE}/{accession}"

    session = build_http_session(
        http_retries=config.http_retries,
        backoff_seconds=config.backoff_seconds,
        user_agent=config.user_agent,
    )

    try:
        entries = _fetch_prediction(api_url, config, session)
        if entries is None or not entries:
            raise ValueError(f"AlphaFold DB has no prediction for accession '{accession}'")

        entry = entries[0]
        if len(entries) > 1:
            logger.warning(
                "AlphaFold DB returned %d records for '%s' (alternative isoforms or "
                "additional fragments); using the canonical record %s",
                len(entries),
                accession,
                entry["modelEntityId"],
            )

        structure_format = config.structure_format
        structure_text: str | None = None
        if config.include_structure:
            structure_url = entry["pdbUrl"] if structure_format == "pdb" else entry["cifUrl"]
            structure_text = _fetch_text(structure_url, config, session)

        plddt_per_residue: list[float] | None = None
        if config.include_plddt:
            plddt_per_residue = _fetch_plddt(entry["plddtDocUrl"], config, session)

        pae_matrix: list[list[float]] | None = None
        if config.include_pae:
            pae_matrix = _fetch_pae(entry["paeDocUrl"], config, session)

        return AlphaFoldDBFetchOutput(
            uniprot_accession=entry.get("uniprotAccession", accession),
            entry_id=entry["modelEntityId"],
            gene=entry.get("gene"),
            organism_scientific_name=entry.get("organismScientificName"),
            tax_id=entry.get("taxId"),
            sequence=entry["sequence"],
            sequence_length=len(entry["sequence"]),
            sequence_start=entry["sequenceStart"],
            sequence_end=entry["sequenceEnd"],
            latest_version=entry["latestVersion"],
            model_created_date=entry.get("modelCreatedDate"),
            mean_plddt=entry.get("globalMetricValue"),
            pdb_url=entry["pdbUrl"],
            cif_url=entry["cifUrl"],
            pae_doc_url=entry["paeDocUrl"],
            plddt_doc_url=entry["plddtDocUrl"],
            pae_image_url=entry["paeImageUrl"],
            msa_url=entry.get("msaUrl"),
            structure_format=structure_format if config.include_structure else "",
            structure_text=structure_text,
            plddt_per_residue=plddt_per_residue,
            pae_matrix=pae_matrix,
            source_url=api_url,
            raw_entry=entry,
        )
    finally:
        session.close()


# ============================================================================
# Private Helpers
# ============================================================================


def _fetch_prediction(
    api_url: str,
    config: AlphaFoldDBFetchConfig,
    session: requests.Session,
) -> list[dict[str, Any]] | None:
    """Fetch prediction metadata list. Returns None on 404."""
    response = session.get(api_url, timeout=config.request_timeout_seconds)
    if response.status_code == 404:
        logger.debug("AlphaFold DB returned 404 for %s", api_url)
        return None
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected AlphaFold DB response shape at {api_url}: expected list")
    return payload


def _fetch_text(
    url: str,
    config: AlphaFoldDBFetchConfig,
    session: requests.Session,
) -> str:
    """Fetch a structure file as text."""
    response = session.get(url, timeout=config.request_timeout_seconds)
    response.raise_for_status()
    return response.text


def _fetch_plddt(
    url: str,
    config: AlphaFoldDBFetchConfig,
    session: requests.Session,
) -> list[float]:
    """Fetch the per-residue pLDDT confidence array."""
    response = session.get(url, timeout=config.request_timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    scores = payload.get("confidenceScore")
    if not isinstance(scores, list):
        raise ValueError(f"AlphaFold DB pLDDT JSON missing 'confidenceScore' list at {url}")
    return [float(value) for value in scores]


def _fetch_pae(
    url: str,
    config: AlphaFoldDBFetchConfig,
    session: requests.Session,
) -> list[list[float]]:
    """Fetch the PAE matrix as a 2D list of floats."""
    response = session.get(url, timeout=config.request_timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"AlphaFold DB PAE JSON empty or wrong shape at {url}")
    matrix = payload[0].get("predicted_aligned_error")
    if not isinstance(matrix, list):
        raise ValueError(f"AlphaFold DB PAE JSON missing 'predicted_aligned_error' at {url}")
    return [[float(value) for value in row] for row in matrix]
