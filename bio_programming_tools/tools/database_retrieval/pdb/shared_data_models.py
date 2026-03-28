"""bio_programming_tools/tools/database_retrieval/pdb/shared_data_models.py

Contains configuration, chain models, and private helpers used by
fetch_entry and fetch_fasta tool modules."""

from __future__ import annotations

import logging
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import requests
from Bio import SeqIO
from Bio.Data.IUPACData import protein_letters
from pydantic import BaseModel, Field

from bio_programming_tools.utils import BaseConfig, ConfigField

logger = logging.getLogger(__name__)

_PDB_ENTRY_BASE = "https://data.rcsb.org/rest/v1/core/entry"
_PDB_FASTA_BASE = "https://www.rcsb.org/fasta/entry"

_PROTEIN_ONLY_CHARS = set(protein_letters.upper()) - set("ATGCNU")


# ============================================================================
# Data Models
# ============================================================================


class PdbChain(BaseModel):
    """Single chain from PDB FASTA.

    Attributes:
        chain_id (str | None): Chain identifier extracted from header.
        header (str): Full FASTA header line.
        sequence (str): Chain sequence string.
        is_protein (bool): True if chain is protein, False if nucleic acid.
    """

    chain_id: Optional[str] = Field(
        default=None, description="Chain identifier from header"
    )
    header: str = Field(description="FASTA header")
    sequence: str = Field(description="Chain sequence")
    is_protein: bool = Field(description="True if chain is protein, False if nucleic acid")


class PdbFetchConfig(BaseConfig):
    """Configuration for PDB fetch operations.

    Attributes:
        request_timeout_seconds (int): HTTP timeout per request.
        http_retries (int): Maximum HTTP retries.
        backoff_seconds (float): Seconds to wait between retries (doubles after each
            attempt).
        user_agent (str): Identifier string sent to database APIs with each request.
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
        description="Seconds to wait between retries (doubles after each attempt)",
        advanced=True,
    )
    user_agent: str = ConfigField(
        title="User Agent",
        default="bio-programming-tools/pdb-fetch-v1",
        description="Identifier string sent to database APIs with each request",
        advanced=True,
    )


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
