"""proto_tools/tools/database_retrieval/alphafold_db/alphafold_db_fetch.py.

Fetches AlphaFold-predicted structures, per-residue pLDDT, and PAE matrices
from the AlphaFold Protein Structure Database by UniProt accession.
"""

import csv
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

    uniprot_id: str = InputField(title="UniProt Accession", description="UniProt accession (e.g. 'P04637')")
    isoform: int | None = InputField(
        default=None,
        ge=2,
        title="Isoform Number",
        description="Isoform number to fetch; None returns the canonical record. For non-canonical isoforms only.",
    )


class AlphaFoldDBFetchConfig(BaseConfig):
    """Configuration for AlphaFold DB fetch.

    Attributes:
        structure_format (Literal["pdb", "cif"]): Structure file format.
        include_structure (bool): If True (default), fetch the structure body
            and the per-residue pLDDT array, returning a parsed ``Structure``
            on the output. Set to False for metadata-only probes (URLs, mean
            pLDDT, gene, sequence) — saves ~100-500 KB per call, meaningful
            for batch sweeps.
        include_pae (bool): If True, also fetch the PAE (predicted aligned
            error) matrix and attach it to ``output.structure.metrics["pae"]``.
            Disabled by default — PAE files can be tens of MB for long proteins.
            No-op when ``include_structure=False``.
        include_msa (bool): If True, fetch the A3M MSA used as input to the
            AlphaFold prediction. Disabled by default — A3M files can be
            hundreds of KB to several MB for highly conserved proteins.
    """

    structure_format: Literal["pdb", "cif"] = ConfigField(
        title="Structure Format",
        default="pdb",
        description="Structure file format (pdb or mmCIF); ignored when include_structure=False",
    )
    include_structure: bool = ConfigField(
        title="Include Structure",
        default=True,
        description="Fetch the structure body and per-residue pLDDT; disable for metadata-only probes",
    )
    include_pae: bool = ConfigField(
        title="Include PAE Matrix",
        default=False,
        description="Also fetch the predicted aligned error matrix; tens of MB for long proteins",
    )
    include_msa: bool = ConfigField(
        title="Include MSA",
        default=False,
        description="Fetch the A3M MSA used for prediction; large for conserved proteins",
    )


class AlphaFoldDBMetrics(Metrics):
    """Per-structure metrics emitted by AlphaFold DB fetch.

    Attached to ``Structure.metrics`` when ``include_structure=True``. Uses the
    same ``avg_plddt`` / ``pae`` keys as the structure-prediction tools
    (ESMFold, AlphaFold2, AlphaFold3) so a downstream tool reading
    ``s.metrics["avg_plddt"]`` composes uniformly across DB-fetched and
    predicted structures.

    Metrics documented in ``metric_spec``:
        avg_plddt (float): Mean per-residue pLDDT score (0-100). Always present
            (sourced from AlphaFold DB's ``globalMetricValue`` field).
        plddt_per_residue (list[float]): Per-residue pLDDT scores (0-100), one
            entry per residue in the structure. Always present.
        pae (list[list[float]]): NxN predicted aligned error matrix in
            angstroms. Present when ``include_pae=True``.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "avg_plddt": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 100.0,
            "better_values_are": "higher",
        },
        "plddt_per_residue": {
            "availability": "always",
            "type": "list[float]",
            "min": 0.0,
            "max": 100.0,
            "better_values_are": "higher",
        },
        "pae": {
            "availability": "when include_pae=True",
            "type": "list[list[float]]",
            "min": 0.0,
            "max": None,
            "better_values_are": "lower",
        },
    }
    primary_metric: str | None = "avg_plddt"


class AlphaFoldDBFetchOutput(BaseToolOutput):
    """Output from AlphaFold DB fetch.

    CSV export writes the structure, MSA, and raw record as sibling sidecar
    files referenced by ``structure_path``, ``msa_path``, ``raw_path``
    columns; see the README for the layout. JSON export is unchanged.

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
        bcif_url (str | None): URL to the BinaryCIF structure file; ``None``
            on legacy entries that predate the bcif export.
        pae_doc_url (str): URL to the PAE JSON document on AlphaFold DB.
        plddt_doc_url (str): URL to the per-residue pLDDT JSON document on
            AlphaFold DB.
        pae_image_url (str): URL to the rendered PAE PNG on AlphaFold DB.
        msa_url (str | None): URL to the MSA A3M used for prediction, when present.
        am_annotations_url (str | None): AlphaMissense pathogenicity CSV URL
            (sequence coords); None for non-human or unscored entries.
        am_annotations_hg19_url (str | None): AlphaMissense annotations on GRCh37.
        am_annotations_hg38_url (str | None): AlphaMissense annotations on GRCh38.
        sequence_checksum (str | None): CRC64 checksum of the predicted sequence.
        structure (Structure | None): Parsed AlphaFold structure (PDB or mmCIF
            body in ``structure_format``, ``b_factor_type=BFactorType.PLDDT``)
            with an :class:`AlphaFoldDBMetrics` ``metrics`` container carrying
            ``avg_plddt``, ``plddt_per_residue``, and (when ``include_pae=True``)
            ``pae``. None when ``include_structure=False``.
        msa_a3m (str | None): A3M-format MSA contents used as input to the
            AlphaFold prediction. None when ``include_msa`` is False or when the
            entry has no associated MSA URL.
        source_url (str): AlphaFold DB API URL used for the metadata lookup.
        raw_entry (dict[str, Any]): Complete AlphaFold DB JSON record for advanced
            programmatic access.
    """

    uniprot_accession: str = Field(title="UniProt Accession", description="Primary UniProt accession")
    entry_id: str = Field(title="Entry ID", description="AlphaFold entry identifier (e.g. 'AF-P04637-F1')")
    gene: str | None = Field(default=None, title="Gene", description="Gene symbol")
    organism_scientific_name: str | None = Field(
        default=None, title="Organism", description="Source organism scientific name"
    )
    tax_id: int | None = Field(default=None, title="Taxonomy ID", description="NCBI taxonomy ID")
    sequence: str = Field(title="Sequence", description="Amino-acid sequence covered by the prediction")
    sequence_length: int = Field(title="Sequence Length", description="Length of the predicted sequence")
    sequence_start: int = Field(
        title="Sequence Start", description="1-indexed inclusive start residue of the prediction"
    )
    sequence_end: int = Field(title="Sequence End", description="1-indexed inclusive end residue of the prediction")
    latest_version: int = Field(
        title="Latest Version",
        description="Latest AlphaFold DB version of this prediction (always the served version)",
    )
    model_created_date: str | None = Field(
        default=None, title="Model Created Date", description="ISO 8601 model creation timestamp"
    )
    mean_plddt: float | None = Field(
        default=None, title="Mean pLDDT", description="Mean per-residue pLDDT confidence (0-100 scale)"
    )
    pdb_url: str = Field(title="PDB URL", description="URL to PDB structure file")
    cif_url: str = Field(title="mmCIF URL", description="URL to mmCIF structure file")
    bcif_url: str | None = Field(
        default=None, title="BinaryCIF URL", description="URL to BinaryCIF structure file (None on legacy entries)"
    )
    pae_doc_url: str = Field(title="PAE Document URL", description="URL to PAE JSON document")
    plddt_doc_url: str = Field(title="pLDDT Document URL", description="URL to per-residue pLDDT JSON document")
    pae_image_url: str = Field(title="PAE Image URL", description="URL to rendered PAE PNG")
    msa_url: str | None = Field(default=None, title="MSA URL", description="URL to MSA A3M file, when present")
    am_annotations_url: str | None = Field(
        default=None,
        title="AlphaMissense URL",
        description="URL to AlphaMissense pathogenicity CSV (sequence coords)",
    )
    am_annotations_hg19_url: str | None = Field(
        default=None, title="AlphaMissense hg19 URL", description="URL to AlphaMissense annotations on GRCh37"
    )
    am_annotations_hg38_url: str | None = Field(
        default=None, title="AlphaMissense hg38 URL", description="URL to AlphaMissense annotations on GRCh38"
    )
    sequence_checksum: str | None = Field(
        default=None,
        title="Sequence Checksum",
        description="CRC64 checksum of the predicted sequence (cache validation)",
    )
    structure: Structure | None = Field(
        default=None,
        title="Predicted Structure",
        description="Parsed AlphaFold Structure (PLDDT B-factors, metrics with pLDDT and PAE); None when skipped",
    )
    msa_a3m: str | None = Field(
        default=None, title="MSA A3M", description="A3M-format MSA contents used as input to prediction"
    )
    source_url: str = Field(title="Source URL", description="AlphaFold DB API URL used for lookup")
    raw_entry: dict[str, Any] = Field(
        default_factory=dict,
        title="Raw Entry",
        description="Complete AlphaFold DB JSON record for advanced programmatic access",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Any, file_format: str) -> None:
        base = Path(export_path)
        if file_format == "json":
            with base.with_suffix(".json").open("w", encoding="utf-8") as f:
                json.dump(self.model_dump(mode="json"), f, indent=2)
            return
        if file_format == "csv":
            self._write_csv_with_sidecars(base)
            return
        raise ValueError(f"Unsupported format: {file_format}")

    def _write_csv_with_sidecars(self, base_path: Path) -> None:
        """Write a metadata CSV plus blob sidecar files; see the class docstring."""
        csv_path = base_path.with_suffix(".csv")
        parent = csv_path.parent
        stem = csv_path.stem

        # __dict__.get avoids __getattr__'s ToolExecutionError on success=False outputs.
        structure_basename = ""
        structure_obj = self.__dict__.get("structure")
        if structure_obj is not None:
            ext = structure_obj.structure_format
            sidecar = parent / f"{stem}.{ext}"
            if ext == "pdb":
                structure_obj.write_pdb(sidecar)
            else:
                structure_obj.write_cif(sidecar)
            structure_basename = sidecar.name

        msa_basename = ""
        msa_text = self.__dict__.get("msa_a3m")
        if msa_text:
            sidecar = parent / f"{stem}.a3m"
            sidecar.write_text(msa_text, encoding="utf-8")
            msa_basename = sidecar.name

        raw_basename = ""
        raw = self.__dict__.get("raw_entry") or {}
        if raw:
            sidecar = parent / f"{stem}_raw.json"
            with sidecar.open("w", encoding="utf-8") as f:
                json.dump(raw, f, indent=2)
            raw_basename = sidecar.name

        row = self.model_dump(exclude={"structure", "msa_a3m", "raw_entry"})
        row["structure_path"] = structure_basename
        row["msa_path"] = msa_basename
        row["raw_path"] = raw_basename

        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)


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
    metrics_class=AlphaFoldDBMetrics,
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
            pae: list[list[float]] | None = None
            if config.include_pae:
                pae = _fetch_pae(entry["paeDocUrl"], session)

            metrics = AlphaFoldDBMetrics(
                avg_plddt=entry.get("globalMetricValue"),
                plddt_per_residue=plddt_per_residue,
                pae=pae,
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
            bcif_url=entry.get("bcifUrl"),
            pae_doc_url=entry["paeDocUrl"],
            plddt_doc_url=entry["plddtDocUrl"],
            pae_image_url=entry["paeImageUrl"],
            msa_url=entry.get("msaUrl"),
            am_annotations_url=entry.get("amAnnotationsUrl"),
            am_annotations_hg19_url=entry.get("amAnnotationsHg19Url"),
            am_annotations_hg38_url=entry.get("amAnnotationsHg38Url"),
            sequence_checksum=entry.get("sequenceChecksum"),
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
