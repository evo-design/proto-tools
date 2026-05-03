"""proto_tools/tools/database_retrieval/pdb/shared_data_models.py.

Contains configuration, chain models, and private helpers used by
fetch_entry and fetch_fasta tool modules.
"""

import logging
from io import StringIO
from typing import Any

import requests
from pydantic import BaseModel, Field

from proto_tools.utils import BaseConfig

logger = logging.getLogger(__name__)

_PDB_ENTRY_BASE = "https://data.rcsb.org/rest/v1/core/entry"
_PDB_FASTA_BASE = "https://www.rcsb.org/fasta/entry"
_REQUEST_TIMEOUT_SECONDS = 15
_HTTP_RETRIES = 3
_BACKOFF_SECONDS = 1.0
_USER_AGENT = "proto-tools/pdb-fetch-v1"

_PROTEIN_ONLY_CHARS_CACHE: set[str] | None = None


def _get_protein_only_chars() -> set[str]:
    """Return amino-acid letters that don't overlap nucleotides; lazy-imports biopython."""
    global _PROTEIN_ONLY_CHARS_CACHE  # noqa: PLW0603 -- module-level cache
    if _PROTEIN_ONLY_CHARS_CACHE is None:
        from Bio.Data.IUPACData import protein_letters

        _PROTEIN_ONLY_CHARS_CACHE = set(protein_letters.upper()) - set("ATGCNU")
    return _PROTEIN_ONLY_CHARS_CACHE


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

    chain_id: str | None = Field(default=None, description="Chain identifier from header")
    header: str = Field(description="FASTA header")
    sequence: str = Field(description="Chain sequence")
    is_protein: bool = Field(description="True if chain is protein, False if nucleic acid")


class PdbFetchConfig(BaseConfig):
    """Configuration for PDB fetch operations.

    PDB tools have no user-facing knobs — all behavior comes from the
    PDB ID supplied as input. This class is kept (rather than removed)
    so callers can still pass ``run_pdb_fetch_entry(input, config)``
    without breakage, but it has no fields.
    """


# ============================================================================
# Private Helpers
# ============================================================================


def _request_pdb(session: requests.Session, url: str, source_label: str) -> requests.Response | None:
    """Execute an HTTP GET, returning None on 404."""
    response = session.get(url, timeout=_REQUEST_TIMEOUT_SECONDS)
    if response.status_code == 404:
        logger.debug("No record found at %s: %s", source_label, response.url)
        return None
    response.raise_for_status()
    return response


def _chain_id_from_header(header: str) -> str | None:
    """Extract chain ID from a PDB FASTA header."""
    first_token = header.split("|")[0].strip()
    parts = first_token.split("_")
    if len(parts) >= 2:
        return parts[1]
    return None


def _fetch_pdb_entry(pdb_id: str, session: requests.Session) -> dict[str, Any] | None:
    """Fetch PDB entry metadata (title, method, resolution), or None on 404."""
    response = _request_pdb(session, f"{_PDB_ENTRY_BASE}/{pdb_id}", "pdb-entry")
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


def _fetch_pdb_fasta(pdb_id: str, session: requests.Session) -> list[tuple[str, str]] | None:
    """Fetch PDB FASTA chains as (header, sequence) tuples, or None on 404."""
    from Bio import SeqIO

    response = _request_pdb(session, f"{_PDB_FASTA_BASE}/{pdb_id}", "pdb-fasta")
    if response is None:
        return None
    text = response.text
    if not text or not text.strip():
        return []
    return [
        (record.description, str(record.seq))
        for record in SeqIO.parse(StringIO(text), "fasta")  # type: ignore[no-untyped-call]
    ]


def _is_protein_sequence(seq: str) -> bool:
    """Return True if sequence contains protein-specific amino acid characters."""
    upper = set(seq.upper()) - {"-", "*", "X", " "}
    if not upper:
        return False
    return bool(upper & _get_protein_only_chars())
