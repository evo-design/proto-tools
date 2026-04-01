"""proto_tools/tools/database_retrieval/ncbi/shared_data_models.py.

Contains configuration, FASTA record models, and private helpers used
by esearch, esummary, and efetch tool modules.
"""

from __future__ import annotations

import json
import logging
from io import StringIO
from typing import Any

import requests
from Bio import SeqIO
from pydantic import BaseModel, Field

from proto_tools.utils import BaseConfig, ConfigField

logger = logging.getLogger(__name__)

_NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


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
        request_timeout_seconds (int): HTTP timeout per request.
        http_retries (int): Number of retries for failed requests.
        backoff_seconds (float): Seconds to wait between retries (doubles after each
            attempt).
        ncbi_api_key (str | None): Optional NCBI API key for higher rate limits.
        ncbi_email (str | None): Optional contact email for NCBI requests.
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
    ncbi_api_key: str | None = ConfigField(
        title="NCBI API Key",
        default=None,
        description="Optional NCBI API key",
        advanced=True,
    )
    ncbi_email: str | None = ConfigField(
        title="NCBI Email",
        default=None,
        description="Optional NCBI contact email",
        advanced=True,
    )
    user_agent: str = ConfigField(
        title="User Agent",
        default="proto-tools/ncbi-fetch-v1",
        description="Identifier string sent to database APIs with each request",
        advanced=True,
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
) -> list[str]:
    """Run NCBI esearch and return ID list."""
    params = {
        "db": db,
        "term": term,
        "retmode": "json",
        "retmax": max_results,
    }
    params.update(_ncbi_common_params(config))

    response = session.get(
        f"{_NCBI_EUTILS_BASE}/esearch.fcgi",
        params=params,  # type: ignore[arg-type]
        timeout=config.request_timeout_seconds,
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
        timeout=config.request_timeout_seconds,
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
        timeout=config.request_timeout_seconds,
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
    from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

    parts = urlsplit(url)
    params = parse_qs(parts.query, keep_blank_values=True)
    for key in ("api_key", "email"):
        params.pop(key, None)
    clean_query = urlencode(params, doseq=True)
    return urlunsplit(parts._replace(query=clean_query))


def _parse_fasta_records(text: str) -> list[NCBIFastaRecord]:
    """Parse FASTA text into NCBIFastaRecord objects."""
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
