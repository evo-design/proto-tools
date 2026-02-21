"""PDB fetch tool for retrieving structure metadata and chain sequences from RCSB PDB.

Provides entry metadata (title, method, resolution) and FASTA chain sequences
for any PDB accession via the RCSB PDB REST API.
"""

from __future__ import annotations

import logging
from io import StringIO
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests
from Bio import SeqIO
from Bio.Data.IUPACData import protein_letters
from pydantic import BaseModel, Field

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField
from bio_programming_tools.utils.http_session import build_http_session
from bio_programming_tools.utils.tool_io import BaseToolInput, BaseToolOutput

logger = logging.getLogger(__name__)

_PDB_ENTRY_BASE = "https://data.rcsb.org/rest/v1/core/entry"
_PDB_FASTA_BASE = "https://www.rcsb.org/fasta/entry"

_PROTEIN_ONLY_CHARS = set(protein_letters.upper()) - set("ATGCNU")


# ============================================================================
# Data Models
# ============================================================================


class PdbFetchInput(BaseToolInput):
    """Input for PDB fetch operations.

    Attributes:
        pdb_id: PDB accession (e.g. '1LBG').
        operation: Fetch entry metadata or FASTA chains.
    """

    pdb_id: str = Field(description="PDB accession (e.g. '1LBG')")
    operation: Literal["entry", "fasta"] = Field(
        default="entry",
        description="Fetch entry metadata or FASTA chains",
    )


class PdbChain(BaseModel):
    """Single chain from PDB FASTA.

    Attributes:
        chain_id: Chain identifier extracted from header.
        header: Full FASTA header line.
        sequence: Chain sequence string.
        is_protein: Whether this chain is a protein sequence.
    """

    chain_id: Optional[str] = Field(
        default=None, description="Chain identifier from header"
    )
    header: str = Field(description="FASTA header")
    sequence: str = Field(description="Chain sequence")
    is_protein: bool = Field(description="Whether this chain is a protein sequence")


class PdbFetchOutput(BaseToolOutput):
    """Output from PDB fetch tool.

    Attributes:
        pdb_id: PDB accession queried.
        title: Structure title (entry operation).
        method: Experimental method (entry operation).
        resolution: Resolution in angstroms (entry operation).
        chains: Parsed chain sequences (fasta operation).
        source_url: URL used for the request.
    """

    pdb_id: str = Field(description="PDB accession queried")
    title: Optional[str] = Field(default=None, description="Structure title")
    method: Optional[str] = Field(default=None, description="Experimental method")
    resolution: Optional[float] = Field(default=None, description="Resolution in angstroms")
    chains: List[PdbChain] = Field(default_factory=list, description="Parsed chain sequences")
    source_url: Optional[str] = Field(default=None, description="Request URL")

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


class PdbFetchConfig(BaseConfig):
    """Configuration for PDB fetch operations.

    Attributes:
        request_timeout_seconds: HTTP timeout per request.
        http_retries: Maximum HTTP retries.
        backoff_seconds: Retry backoff multiplier in seconds.
        user_agent: HTTP user-agent string.
    """

    request_timeout_seconds: int = ConfigField(
        title="Request Timeout",
        default=15,
        ge=1,
        description="HTTP timeout per request",
        advanced=True,
    )
    http_retries: int = ConfigField(
        title="HTTP Retries",
        default=3,
        ge=0,
        description="Max HTTP retries",
        advanced=True,
    )
    backoff_seconds: float = ConfigField(
        title="Backoff Seconds",
        default=1.0,
        ge=0.0,
        description="Retry backoff multiplier in seconds",
        advanced=True,
    )
    user_agent: str = ConfigField(
        title="User Agent",
        default="bio-programming-tools/pdb-fetch-v1",
        description="HTTP user-agent string",
        advanced=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================


@tool(
    key="pdb-fetch",
    label="PDB Fetch",
    category="database_retrieval",
    input=PdbFetchInput,
    config=PdbFetchConfig,
    output=PdbFetchOutput,
    description="Fetch structure metadata and chain sequences from RCSB PDB",
    uses_gpu=False,
)
def run_pdb_fetch(
    inputs: PdbFetchInput,
    config: PdbFetchConfig,
    instance=None,
) -> PdbFetchOutput:
    """Fetch structure metadata and chain sequences from RCSB PDB.

    Supports two operations:
    - ``entry``: returns title, experimental method, and resolution.
    - ``fasta``: returns parsed chain sequences with protein/nucleotide detection.

    Args:
        inputs: A single PDB fetch request.
        config: HTTP timeout and retry settings.

    Returns:
        PdbFetchOutput with the fetch result.
    """
    del instance

    session = build_http_session(
        http_retries=config.http_retries,
        backoff_seconds=config.backoff_seconds,
        user_agent=config.user_agent,
        mount_http=True,
    )
    pdb_id = inputs.pdb_id.upper()

    try:
        if inputs.operation == "entry":
            meta = _fetch_pdb_entry(pdb_id, config, session)
            if meta is None:
                return PdbFetchOutput(pdb_id=pdb_id)
            return PdbFetchOutput(
                pdb_id=pdb_id,
                title=meta.get("title"),
                method=meta.get("method"),
                resolution=meta.get("resolution"),
                source_url=f"{_PDB_ENTRY_BASE}/{pdb_id}",
            )

        # fasta
        raw_chains = _fetch_pdb_fasta(pdb_id, config, session)
        if raw_chains is None:
            return PdbFetchOutput(pdb_id=pdb_id)
        pdb_chains = [
            PdbChain(
                chain_id=_chain_id_from_header(header),
                header=header,
                sequence=sequence,
                is_protein=_is_protein_sequence(sequence),
            )
            for header, sequence in raw_chains
        ]
        return PdbFetchOutput(
            pdb_id=pdb_id,
            chains=pdb_chains,
            source_url=f"{_PDB_FASTA_BASE}/{pdb_id}",
        )
    finally:
        session.close()


# ============================================================================
# Private Helpers
# ============================================================================


def _request_pdb(
    session: requests.Session,
    url: str,
    config: PdbFetchConfig,
    source_label: str,
) -> Optional[requests.Response]:
    """Execute an HTTP GET, returning None on 404."""
    response = session.get(url, timeout=config.request_timeout_seconds)
    if response.status_code == 404:
        logger.debug("No record found at %s: %s", source_label, response.url)
        return None
    response.raise_for_status()
    return response


def _chain_id_from_header(header: str) -> Optional[str]:
    """Extract chain ID from a PDB FASTA header."""
    first_token = header.split("|")[0].strip()
    parts = first_token.split("_")
    if len(parts) >= 2:
        return parts[1]
    return None


def _fetch_pdb_entry(
    pdb_id: str,
    config: PdbFetchConfig,
    session: requests.Session,
) -> Optional[Dict[str, Any]]:
    """Fetch PDB entry metadata (title, method, resolution), or None on 404."""
    response = _request_pdb(
        session, f"{_PDB_ENTRY_BASE}/{pdb_id}", config, "pdb-entry"
    )
    if response is None:
        return None
    data = response.json()

    struct = data.get("struct", {})
    entry_info = data.get("rcsb_entry_info", {})
    exptl = data.get("exptl", [])

    title = struct.get("title")
    method = exptl[0].get("method") if exptl else None

    resolution = None
    for key in ("resolution_combined", "d_resolution_high", "em_resolution"):
        value = entry_info.get(key)
        if isinstance(value, list) and value:
            resolution = float(value[0])
            break
        if isinstance(value, (float, int)):
            resolution = float(value)
            break

    return {"title": title, "method": method, "resolution": resolution}


def _fetch_pdb_fasta(
    pdb_id: str,
    config: PdbFetchConfig,
    session: requests.Session,
) -> Optional[List[Tuple[str, str]]]:
    """Fetch PDB FASTA chains as (header, sequence) tuples, or None on 404."""
    response = _request_pdb(
        session, f"{_PDB_FASTA_BASE}/{pdb_id}", config, "pdb-fasta"
    )
    if response is None:
        return None
    text = response.text
    if not text or not text.strip():
        return []
    return [
        (record.description, str(record.seq))
        for record in SeqIO.parse(StringIO(text), "fasta")
    ]


def _is_protein_sequence(seq: str) -> bool:
    """Return True if sequence contains protein-specific amino acid characters."""
    upper = set(seq.upper()) - {"-", "*", "X", " "}
    if not upper:
        return False
    return bool(upper & _PROTEIN_ONLY_CHARS)
