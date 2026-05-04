"""proto_tools/tools/database_retrieval/alphafold_db/alphafold_db_fetch.py.

Fetches AlphaFold-predicted structures, per-residue pLDDT, and PAE matrices
from the AlphaFold Protein Structure Database by UniProt accession.
"""

import json
import logging
from pathlib import Path
from typing import Any, ClassVar, Literal

import requests
from pydantic import Field

from proto_tools.entities.structures.structure import BFactorType, Structure
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    build_http_session,
)
from proto_tools.utils.tool_io import Metrics, MetricSpec

logger = logging.getLogger(__name__)

_AFDB_API_BASE = "https://alphafold.ebi.ac.uk/api/prediction"
_REQUEST_TIMEOUT_SECONDS = 15
_HTTP_RETRIES = 2
_BACKOFF_SECONDS = 1.0
_USER_AGENT = "proto-tools/alphafold-db-fetch-v1"


# ============================================================================
# Data Models
# ============================================================================


class AlphaFoldDBFetchInput(BaseToolInput):
    """Input for AlphaFold DB fetch.

    Attributes:
        uniprot_id (str): UniProt accession to look up (e.g. 'P04637').
        isoform (int | None): Isoform number to select from the multi-record
            AFDB response. ``None`` (default) returns the canonical entry
            (``AF-{accession}-F1``); ``2`` selects ``AF-{accession}-2-F1``,
            etc. AFDB typically exposes isoforms 2-9 for human proteins.
            Raises ``ValueError`` if the requested isoform doesn't exist.
    """

    uniprot_id: str = InputField(description="UniProt accession (e.g. 'P04637')")
    isoform: int | None = InputField(
        default=None,
        ge=2,
        description="Isoform number to fetch (None = canonical record). For non-canonical isoforms only.",
        advanced=True,
    )


class AlphaFoldDBFetchConfig(BaseConfig):
    """Configuration for AlphaFold DB fetch.

    Attributes:
        structure_format (Literal["pdb", "cif"]): Structure file format to download.
        include_structure (bool): If True (default), download the structure file
            text and the per-residue pLDDT array, returning a parsed ``Structure``
            on the output. Set to False for metadata-only probes (URLs, mean pLDDT,
            gene, sequence) — saves ~100-500 KB per call, meaningful for batch sweeps.
        include_pae (bool): If True, also download the PAE (predicted aligned
            error) matrix and attach it to ``output.structure.metrics["pae_matrix"]``.
            Disabled by default — pAE files can be tens of MB for long proteins.
            No-op when ``include_structure=False``.
        include_msa (bool): If True, download the A3M MSA used as input to
            AlphaFold prediction. Disabled by default — A3M files can be
            hundreds of KB to several MB for highly conserved proteins.
    """

    structure_format: Literal["pdb", "cif"] = ConfigField(
        title="Structure Format",
        default="pdb",
        description="Structure file format to download (pdb or mmCIF)",
    )
    include_structure: bool = ConfigField(
        title="Include Structure",
        default=True,
        description="Fetch structure + per-residue pLDDT into output.structure; False for metadata-only probes",
        advanced=True,
    )
    include_pae: bool = ConfigField(
        title="Include PAE Matrix",
        default=False,
        description="Also fetch PAE into output.structure.metrics; ignored when include_structure=False",
        advanced=True,
    )
    include_msa: bool = ConfigField(
        title="Include MSA",
        default=False,
        description="Download the A3M MSA used as input to prediction; large for conserved proteins",
        advanced=True,
    )


class AlphaFoldDBMetrics(Metrics):
    """Per-structure metrics emitted by AlphaFold DB fetch.

    Attached to ``Structure.metrics`` when ``include_structure=True``. Uses the
    same ``avg_plddt`` / ``pae_matrix`` keys as the structure-prediction tools
    (ESMFold, AlphaFold2, AlphaFold3) so a downstream tool reading
    ``s.metrics["avg_plddt"]`` composes uniformly across DB-fetched and
    predicted structures.

    Metrics documented in ``metric_spec``:
        avg_plddt (float): Mean per-residue pLDDT score (0-100). Always present
            (sourced from AlphaFold DB's ``globalMetricValue`` field).
        plddt_per_residue (list[float]): Per-residue pLDDT scores (0-100), one
            entry per residue in the structure. Always present.
        pae_matrix (list[list[float]]): NxN predicted aligned error matrix in
            angstroms. Present when ``include_pae=True``.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "avg_plddt": {"availability": "always", "type": "float", "min": 0.0, "max": 100.0},
        "plddt_per_residue": {
            "availability": "always",
            "type": "list[float]",
            "min": 0.0,
            "max": 100.0,
        },
        "pae_matrix": {
            "availability": "when include_pae=True",
            "type": "list[list[float]]",
            "min": 0.0,
            "max": None,
        },
    }
    primary_metric: str | None = "avg_plddt"


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
            (AlphaFold DB's globalMetricValue field). Always populated from the
            metadata response, regardless of ``include_structure``; when
            ``include_structure=True`` it is also mirrored at
            ``structure.metrics["avg_plddt"]``.
        pdb_url (str): URL to the PDB structure file on AlphaFold DB.
        cif_url (str): URL to the mmCIF structure file on AlphaFold DB.
        pae_doc_url (str): URL to the PAE JSON document on AlphaFold DB.
        plddt_doc_url (str): URL to the per-residue pLDDT JSON document on
            AlphaFold DB.
        pae_image_url (str): URL to the rendered PAE PNG on AlphaFold DB.
        msa_url (str | None): URL to the MSA A3M used for prediction, when present.
        structure (Structure | None): Parsed AlphaFold structure (PDB or mmCIF
            body in ``structure_format``, ``b_factor_type=BFactorType.PLDDT``)
            with an :class:`AlphaFoldDBMetrics` ``metrics`` container carrying
            ``avg_plddt``, ``plddt_per_residue``, and (when ``include_pae=True``)
            ``pae_matrix``. None when ``include_structure=False``.
        msa_a3m (str | None): A3M-format MSA contents used as input to the
            AlphaFold prediction. None when ``include_msa`` is False or when the
            entry has no associated MSA URL.
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
    structure: Structure | None = Field(
        default=None,
        description="Parsed AlphaFold Structure (PLDDT B-factors, metrics with pLDDT and PAE); None when skipped",
    )
    msa_a3m: str | None = Field(default=None, description="A3M-format MSA contents used as input to prediction")
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
    cacheable=True,
)
def run_alphafold_db_fetch(
    inputs: AlphaFoldDBFetchInput,
    config: AlphaFoldDBFetchConfig,
    instance: Any = None,
) -> AlphaFoldDBFetchOutput:
    """Fetch a prediction record from AlphaFold DB.

    AFDB returns one record per isoform (TP53 has 9). Without ``isoform`` the
    canonical record (``AF-{accession}-F1``) is selected and a warning lists
    available alternatives.

    Args:
        inputs (AlphaFoldDBFetchInput): UniProt accession to look up.
        config (AlphaFoldDBFetchConfig): Format selection (pdb/cif) and
            artifact toggles (PAE, MSA).

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        AlphaFoldDBFetchOutput: Prediction metadata plus an optional parsed
            ``Structure`` (with per-residue pLDDT and optional PAE on
            ``structure.metrics``) and optionally the A3M MSA.
    """
    del instance

    accession = inputs.uniprot_id.strip().upper()
    api_url = f"{_AFDB_API_BASE}/{accession}"

    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
    )

    try:
        entries = _fetch_prediction(api_url, session)
        if entries is None or not entries:
            raise ValueError(f"AlphaFold DB has no prediction for accession '{accession}'")

        entry = _select_record(entries, accession, inputs.isoform)

        structure: Structure | None = None
        if config.include_structure:
            structure_url = entry["pdbUrl"] if config.structure_format == "pdb" else entry["cifUrl"]
            structure_text = _fetch_text(structure_url, session)
            plddt_per_residue = _fetch_plddt(entry["plddtDocUrl"], session)
            pae_matrix: list[list[float]] | None = None
            if config.include_pae:
                pae_matrix = _fetch_pae(entry["paeDocUrl"], session)

            metrics = AlphaFoldDBMetrics(
                avg_plddt=entry.get("globalMetricValue"),
                plddt_per_residue=plddt_per_residue,
                pae_matrix=pae_matrix,
            )
            structure = Structure(
                structure=structure_text,
                structure_format=config.structure_format,
                b_factor_type=BFactorType.PLDDT,
                metrics=metrics,
                source="alphafold-db-fetch",
            )

        msa_a3m: str | None = None
        if config.include_msa:
            msa_url = entry.get("msaUrl")
            if msa_url:
                msa_a3m = _fetch_text(msa_url, session)
            else:
                logger.debug("AlphaFold DB entry %s has no msaUrl; msa_a3m will be None", entry.get("modelEntityId"))

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
            structure=structure,
            msa_a3m=msa_a3m,
            source_url=api_url,
            raw_entry=entry,
        )
    finally:
        session.close()


# ============================================================================
# Private Helpers
# ============================================================================


def _select_record(
    entries: list[dict[str, Any]],
    accession: str,
    isoform: int | None,
) -> dict[str, Any]:
    """Pick the requested record from the multi-record AFDB response.

    Canonical entries have IDs like 'AF-{accession}-F1'; isoforms have
    'AF-{accession}-{isoform}-F1'. When isoform is None the canonical record
    (no isoform number in the ID) is returned. When set, the matching record
    is returned, or ValueError if it doesn't exist.
    """
    canonical_id = f"AF-{accession}-F1"
    if isoform is None:
        for entry in entries:
            if entry.get("modelEntityId") == canonical_id:
                if len(entries) > 1:
                    isoform_ids = sorted(e["modelEntityId"] for e in entries if e.get("modelEntityId") != canonical_id)
                    logger.warning(
                        "AlphaFold DB returned %d records for '%s'; using the canonical "
                        "record %s. Other isoforms available: %s",
                        len(entries),
                        accession,
                        canonical_id,
                        ", ".join(isoform_ids),
                    )
                return entry
        # Fallback: no entry matched the canonical pattern; use the first record.
        return entries[0]

    target_id = f"AF-{accession}-{isoform}-F1"
    for entry in entries:
        if entry.get("modelEntityId") == target_id:
            return entry
    available = sorted(e.get("modelEntityId", "?") for e in entries)
    raise ValueError(f"Isoform {isoform} not available for '{accession}'. Got {len(entries)} record(s): {available}")


def _fetch_prediction(api_url: str, session: requests.Session) -> list[dict[str, Any]] | None:
    """Fetch prediction metadata list. Returns None on 404."""
    response = session.get(api_url, timeout=_REQUEST_TIMEOUT_SECONDS)
    if response.status_code == 404:
        logger.debug("AlphaFold DB returned 404 for %s", api_url)
        return None
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected AlphaFold DB response shape at {api_url}: expected list")
    return payload


def _fetch_text(url: str, session: requests.Session) -> str:
    """Fetch a remote file as text (used for structure files and MSA A3M)."""
    response = session.get(url, timeout=_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def _fetch_plddt(url: str, session: requests.Session) -> list[float]:
    """Fetch the per-residue pLDDT confidence array."""
    response = session.get(url, timeout=_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    scores = payload.get("confidenceScore")
    if not isinstance(scores, list):
        raise ValueError(f"AlphaFold DB pLDDT JSON missing 'confidenceScore' list at {url}")
    return [float(value) for value in scores]


def _fetch_pae(url: str, session: requests.Session) -> list[list[float]]:
    """Fetch the PAE matrix as a 2D list of floats."""
    response = session.get(url, timeout=_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"AlphaFold DB PAE JSON empty or wrong shape at {url}")
    matrix = payload[0].get("predicted_aligned_error")
    if not isinstance(matrix, list):
        raise ValueError(f"AlphaFold DB PAE JSON missing 'predicted_aligned_error' at {url}")
    return [[float(value) for value in row] for row in matrix]
