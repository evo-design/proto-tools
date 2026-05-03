"""proto_tools/tools/database_retrieval/ncbi/shared_data_models.py.

Contains configuration, FASTA record models, and private helpers used
by esearch, esummary, and efetch tool modules.
"""

import json
import logging
from io import StringIO
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

import requests
from pydantic import BaseModel, Field

from proto_tools.utils import BaseConfig, ConfigField

logger = logging.getLogger(__name__)

_NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_REQUEST_TIMEOUT_SECONDS = 15
_HTTP_RETRIES = 2
_BACKOFF_SECONDS = 1.0
_USER_AGENT = "proto-tools/ncbi-fetch-v1"


# ============================================================================
# Data Models
# ============================================================================


class NCBIFastaRecord(BaseModel):
    """A parsed FASTA record.

    Attributes:
        header (str): FASTA header line (without >).
        sequence (str): Sequence string with whitespace stripped.
        accession (str | None): Best-effort accession extracted from header.
    """

    header: str = Field(description="FASTA header line")
    sequence: str = Field(description="Sequence string")
    accession: str | None = Field(default=None, description="Accession extracted from header")


class NCBIFetchConfig(BaseConfig):
    """Configuration for NCBI Entrez operations.

    Attributes:
        ncbi_api_key (str | None): Optional NCBI API key. Without one NCBI
            limits to 3 requests/second; with one the limit is 10/second.
            See https://www.ncbi.nlm.nih.gov/account/settings/.
        ncbi_email (str | None): Optional contact email. NCBI recommends
            providing this so they can reach the operator if a script
            misbehaves.
    """

    ncbi_api_key: str | None = ConfigField(
        title="NCBI API Key",
        default=None,
        description="Optional NCBI API key (lifts rate limit from 3 to 10 req/s)",
        advanced=True,
        include_in_key=False,
    )
    ncbi_email: str | None = ConfigField(
        title="NCBI Email",
        default=None,
        description="Optional contact email; recommended by NCBI for traceability",
        advanced=True,
        include_in_key=False,
    )


# ============================================================================
# Private Helpers
# ============================================================================


def _ncbi_common_params(config: NCBIFetchConfig) -> dict[str, Any]:
    """Build common NCBI eutils parameters."""
    params: dict[str, Any] = {"tool": "proto_tools_ncbi_fetch"}
    if config.ncbi_email:
        params["email"] = config.ncbi_email
    if config.ncbi_api_key:
        params["api_key"] = config.ncbi_api_key
    return params


def _ncbi_esearch(
    db: str,
    term: str,
    max_results: int,
    config: NCBIFetchConfig,
    session: requests.Session,
    retstart: int = 0,
) -> list[str]:
    """Run NCBI esearch and return ID list."""
    params: dict[str, Any] = {
        "db": db,
        "term": term,
        "retmode": "json",
        "retmax": max_results,
    }
    if retstart:
        params["retstart"] = retstart
    params.update(_ncbi_common_params(config))

    response = session.get(
        f"{_NCBI_EUTILS_BASE}/esearch.fcgi",
        params=params,
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    if not _check_response(response, "ncbi-esearch"):
        return []
    data = json.loads(response.text, strict=False)
    return data.get("esearchresult", {}).get("idlist", [])  # type: ignore[no-any-return]


def _ncbi_esummary(
    db: str,
    identifier: str,
    config: NCBIFetchConfig,
    session: requests.Session,
) -> tuple[dict[str, Any], str] | None:
    """Run NCBI esummary and return (result_map, sanitized_url) or None."""
    params: dict[str, Any] = {
        "db": db,
        "id": identifier,
        "retmode": "json",
    }
    params.update(_ncbi_common_params(config))

    response = session.get(
        f"{_NCBI_EUTILS_BASE}/esummary.fcgi",
        params=params,
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    if not _check_response(response, "ncbi-esummary"):
        return None
    url = _sanitize_url(str(response.url))
    data = json.loads(response.text, strict=False)
    return data.get("result", {}), url


def _ncbi_efetch(
    db: str,
    identifier: str,
    rettype: str,
    config: NCBIFetchConfig,
    session: requests.Session,
    seq_start: int | None = None,
    seq_stop: int | None = None,
    strand: str | None = None,
) -> tuple[str, str] | None:
    """Run NCBI efetch and return (text, sanitized_url) or None."""
    params: dict[str, Any] = {
        "db": db,
        "id": identifier,
        "rettype": rettype,
        "retmode": "text",
    }
    if seq_start is not None:
        params["seq_start"] = seq_start
    if seq_stop is not None:
        params["seq_stop"] = seq_stop
    if strand is not None:
        params["strand"] = "2" if strand == "-" else "1"

    params.update(_ncbi_common_params(config))

    response = session.get(
        f"{_NCBI_EUTILS_BASE}/efetch.fcgi",
        params=params,
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    if not _check_response(response, "ncbi-efetch"):
        return None
    return response.text, _sanitize_url(str(response.url))


def _check_response(response: requests.Response, label: str) -> bool:
    """Check HTTP response. Return False on 404, raise on other errors."""
    if response.status_code == 404:
        logger.debug("No record found at %s: %s", label, response.url)
        return False
    response.raise_for_status()
    return True


def _sanitize_url(url: str) -> str:
    """Strip sensitive query parameters (api_key, email) from a URL."""
    parts = urlsplit(url)
    params = parse_qs(parts.query, keep_blank_values=True)
    for key in ("api_key", "email"):
        params.pop(key, None)
    clean_query = urlencode(params, doseq=True)
    return urlunsplit(parts._replace(query=clean_query))


def _parse_fasta_records(text: str) -> list[NCBIFastaRecord]:
    """Parse FASTA text into NCBIFastaRecord objects."""
    from Bio import SeqIO

    if not text or not text.strip():
        return []
    return [
        NCBIFastaRecord(
            header=record.description,
            sequence=str(record.seq),
            accession=_accession_from_header(record.description),
        )
        for record in SeqIO.parse(StringIO(text), "fasta")  # type: ignore[no-untyped-call]
    ]


def _accession_from_header(header: str) -> str | None:
    """Best-effort accession extraction from FASTA header."""
    tokens = header.split()
    if not tokens:
        return None

    first = tokens[0]
    if "|" in first:
        pieces = [p for p in first.split("|") if p]
        if len(pieces) >= 2:
            return pieces[1]
    return first
