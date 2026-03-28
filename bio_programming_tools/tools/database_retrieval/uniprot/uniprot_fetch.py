"""bio_programming_tools/tools/database_retrieval/uniprot/uniprot_fetch.py

Provides a single-API-call interface to the UniProt REST API for retrieving
protein entries by accession or searching by gene name and organism with
ranked result selection."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import requests
from pydantic import Field, model_validator

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    build_http_session,
)

logger = logging.getLogger(__name__)

_UNIPROT_BASE = "https://rest.uniprot.org"


# ============================================================================
# Data Models
# ============================================================================


class UniProtFetchInput(BaseToolInput):
    """Input for UniProt fetch operations.

    Attributes:
        uniprot_id (str | None): UniProt accession for direct entry lookup.
        target_name (str | None): Gene or protein name for search-based lookup.
        organism (str | None): Organism name for disambiguation during search.
        prefer_pdb_crossref (bool): When searching, prefer entries that have linked
            PDB structures.
        max_candidates (int): Maximum number of search results to evaluate when
            ranking.
    """

    uniprot_id: Optional[str] = InputField(
        default=None,
        description="UniProt accession for direct entry lookup",
    )
    target_name: Optional[str] = InputField(
        default=None,
        description="Gene or protein name for search",
    )
    organism: Optional[str] = InputField(
        default=None,
        description="Organism for search disambiguation",
    )
    prefer_pdb_crossref: bool = InputField(
        default=False,
        description="When searching, prefer entries that have linked PDB structures",
    )
    max_candidates: int = InputField(
        default=5,
        ge=1,
        le=25,
        description="Maximum number of search results to evaluate when ranking",
    )

    @model_validator(mode="after")
    def validate_lookup_params(self):
        """Require either uniprot_id or target_name+organism."""
        if not self.uniprot_id and not (self.target_name and self.organism):
            raise ValueError(
                "Provide either uniprot_id or both target_name and organism"
            )
        return self


class UniProtFetchOutput(BaseToolOutput):
    """Output from UniProt fetch tool.

    Attributes:
        accession (str): Primary UniProt accession.
        sequence (str | None): Protein sequence string.
        length (int | None): Sequence length.
        entry_type (str | None): Review status (e.g. 'UniProtKB reviewed (Swiss-Prot)' for
            curated entries).
        gene_names (list[str]): Extracted gene name symbols.
        pdb_crossrefs (list[str]): PDB structure IDs linked to this protein entry.
        source_url (str): UniProt entry URL.
        raw_entry (dict[str, Any]): Complete UniProt JSON record for advanced programmatic
            access.
    """

    accession: str = Field(description="Primary UniProt accession")
    sequence: Optional[str] = Field(default=None, description="Protein sequence")
    length: Optional[int] = Field(default=None, description="Sequence length")
    entry_type: Optional[str] = Field(
        default=None, description="Review status (e.g. 'UniProtKB reviewed (Swiss-Prot)' for curated entries)"
    )
    gene_names: List[str] = Field(
        default_factory=list, description="Gene name symbols"
    )
    pdb_crossrefs: List[str] = Field(
        default_factory=list, description="PDB structure IDs linked to this protein entry"
    )
    source_url: str = Field(description="UniProt entry URL")
    raw_entry: Dict[str, Any] = Field(
        default_factory=dict, description="Complete UniProt JSON record for advanced programmatic access"
    )

    @property
    def output_format_options(self) -> List[str]:
        return ["json"]

    @property
    def output_format_default(self) -> str:
        return "json"

    def _export_output(self, export_path, file_format: str):
        import json
        from pathlib import Path

        if file_format == "json":
            path = Path(export_path).with_suffix(".json")
            with path.open("w", encoding="utf-8") as f:
                json.dump(self.model_dump(mode="json"), f, indent=2)
            return
        raise ValueError(f"Unsupported format: {file_format}")


class UniProtFetchConfig(BaseConfig):
    """Configuration for UniProt fetch operations.

    Attributes:
        request_timeout_seconds (int): HTTP timeout per request.
        http_retries (int): Number of retries for failed requests.
        backoff_seconds (float): Seconds to wait between retries (doubles after each
            attempt).
        user_agent (str): Identifier string sent to database APIs with each request.
    """

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
        default="bio-programming-tools/uniprot-fetch-v1",
        description="Identifier string sent to database APIs with each request",
        advanced=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input():
    """Minimal valid input for testing and examples."""
    return UniProtFetchInput(uniprot_id="P04637")


@tool(
    key="uniprot-fetch",
    label="UniProt Fetch",
    category="database_retrieval",
    input_class=UniProtFetchInput,
    config_class=UniProtFetchConfig,
    output_class=UniProtFetchOutput,
    description="Fetch protein entries from UniProt by accession or search by name and organism",
    uses_gpu=False,
    example_input=example_input,
)
def run_uniprot_fetch(
    inputs: UniProtFetchInput,
    config: UniProtFetchConfig | None = None,
    instance=None,
) -> UniProtFetchOutput:
    """Fetch protein entries from UniProt.

    Supports direct entry lookup by accession or name+organism search with
    ranked result selection based on gene name match, reviewed status, and
    PDB cross-reference availability.

    Args:
        inputs (UniProtFetchInput): A UniProt fetch request with accession or name+organism.
        config (UniProtFetchConfig | None): HTTP timeout and retry settings.

    Returns:
        UniProtFetchOutput: Protein entry with sequence, gene names, and
            PDB cross-references.
    """
    del instance

    session = build_http_session(
        http_retries=config.http_retries,
        backoff_seconds=config.backoff_seconds,
        user_agent=config.user_agent,
    )

    try:
        if inputs.uniprot_id:
            entry = _fetch_entry(inputs.uniprot_id, config, session)
            if entry is None:
                raise ValueError(
                    f"UniProt ID '{inputs.uniprot_id}' not found"
                )
        else:
            entry = _search_entry(
                target_name=inputs.target_name,
                organism=inputs.organism,
                prefer_pdb_crossref=inputs.prefer_pdb_crossref,
                max_candidates=inputs.max_candidates,
                config=config,
                session=session,
            )
            if entry is None:
                raise ValueError(
                    f"No UniProt entry found for '{inputs.target_name}' "
                    f"in '{inputs.organism}'"
                )

        accession = entry.get("primaryAccession", inputs.uniprot_id or "")
        seq_data = entry.get("sequence", {})
        sequence = seq_data.get("value")
        length = seq_data.get("length") or (len(sequence) if sequence else None)
        entry_type = entry.get("entryType")
        gene_names = sorted(_extract_gene_names(entry))
        pdb_xrefs = _extract_pdb_crossrefs(entry)

        return UniProtFetchOutput(
            accession=accession,
            sequence=sequence,
            length=length,
            entry_type=entry_type,
            gene_names=gene_names,
            pdb_crossrefs=pdb_xrefs,
            source_url=f"{_UNIPROT_BASE}/uniprotkb/{accession}",
            raw_entry=entry,
        )
    finally:
        session.close()


# ============================================================================
# Private Helpers
# ============================================================================


def _fetch_entry(
    uniprot_id: str,
    config: UniProtFetchConfig,
    session: requests.Session,
) -> Optional[Dict[str, Any]]:
    """Fetch a UniProtKB entry by accession. Returns None on 404."""
    response = session.get(
        f"{_UNIPROT_BASE}/uniprotkb/{uniprot_id}.json",
        timeout=config.request_timeout_seconds,
    )
    if response.status_code == 404:
        logger.debug("UniProt ID '%s' not found", uniprot_id)
        return None
    response.raise_for_status()
    return response.json()


def _search_entry(
    target_name: str,
    organism: str,
    prefer_pdb_crossref: bool,
    max_candidates: int,
    config: UniProtFetchConfig,
    session: requests.Session,
) -> Optional[Dict[str, Any]]:
    """Search UniProt by name and organism and return best ranked entry."""
    all_results: List[Dict[str, Any]] = []
    seen_accessions: set[str] = set()

    queries = [
        f'gene_exact:{target_name} AND organism_name:"{organism}"',
        f'(gene_exact:{target_name} OR protein_name:{target_name}) AND organism_name:"{organism}"',
    ]

    for query in queries:
        params = {
            "query": query,
            "format": "json",
            "size": max_candidates,
        }
        response = session.get(
            f"{_UNIPROT_BASE}/uniprotkb/search",
            params=params,
            timeout=config.request_timeout_seconds,
        )
        if response.status_code >= 400:
            logger.warning(
                "UniProt search failed with status %d for query: %s",
                response.status_code,
                query,
            )
            continue

        data = response.json()
        for entry in data.get("results", []):
            accession = entry.get("primaryAccession")
            if accession and accession not in seen_accessions:
                seen_accessions.add(accession)
                all_results.append(entry)

        if all_results:
            break

    if not all_results:
        return None

    return max(
        all_results,
        key=lambda entry: _entry_priority(entry, target_name, prefer_pdb_crossref),
    )


def _entry_priority(
    entry: Dict[str, Any],
    target_name: str,
    prefer_pdb_crossref: bool,
) -> Tuple[int, int, int, int]:
    """Rank UniProt candidates for deterministic, biologically sensible selection."""
    target = target_name.strip().lower()
    gene_names = _extract_gene_names(entry)
    has_exact_gene = int(target in gene_names)
    has_pdb = int(bool(_extract_pdb_crossrefs(entry)))
    reviewed = int("reviewed" in str(entry.get("entryType", "")).lower())
    accession = str(entry.get("primaryAccession", ""))
    return (
        has_exact_gene,
        has_pdb if prefer_pdb_crossref else 0,
        reviewed,
        len(accession),
    )


def _extract_gene_names(entry: Dict[str, Any]) -> set[str]:
    """Extract normalized gene symbol candidates from a UniProt entry."""
    names: set[str] = set()
    genes = entry.get("genes", [])
    if not isinstance(genes, list):
        return names

    for gene_obj in genes:
        if not isinstance(gene_obj, dict):
            continue
        primary = gene_obj.get("geneName", {})
        if isinstance(primary, dict):
            value = primary.get("value")
            if isinstance(value, str) and value.strip():
                names.add(value.strip().lower())

    return names


def _extract_pdb_crossrefs(entry: Dict[str, Any]) -> List[str]:
    """Extract PDB cross references from UniProt entry JSON."""
    xrefs = entry.get("uniProtKBCrossReferences", [])
    pdb_ids = []
    for ref in xrefs:
        if ref.get("database") == "PDB" and ref.get("id"):
            pdb_ids.append(ref["id"])
    return pdb_ids
