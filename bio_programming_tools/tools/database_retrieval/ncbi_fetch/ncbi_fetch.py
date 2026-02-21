"""NCBI Entrez fetch tool for esearch, esummary, and efetch operations.

Provides a single-API-call interface to NCBI Entrez for searching and
retrieving biological sequences (protein, nucleotide, gene) in FASTA and
other formats.
"""

from __future__ import annotations

import logging
from io import StringIO
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests
from Bio import SeqIO
from pydantic import BaseModel, Field, model_validator

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField
from bio_programming_tools.utils.http_session import build_http_session
from bio_programming_tools.utils.tool_io import BaseToolInput, BaseToolOutput

logger = logging.getLogger(__name__)

_NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


# ============================================================================
# Data Models
# ============================================================================


class NCBIFetchInput(BaseToolInput):
    """Input for NCBI fetch operations.

    Attributes:
        db: NCBI database name (protein, nuccore, gene, etc.).
        operation: Entrez operation to perform.
        identifier: Accession or ID for efetch/esummary. Mutually exclusive
            with search_term.
        search_term: Entrez query for esearch. Mutually exclusive with
            identifier.
        rettype: Return type for efetch (fasta, fasta_cds_na, gb, etc.).
        seq_start: 1-indexed inclusive start for subsequence retrieval.
        seq_stop: 1-indexed inclusive stop for subsequence retrieval.
        strand: Strand for nucleotide retrieval (+ or -).
        max_results: Maximum results to return from esearch.
    """

    db: Literal["protein", "nuccore", "gene", "nucleotide"] = Field(
        description="NCBI database to query"
    )
    operation: Literal["esearch", "esummary", "efetch"] = Field(
        description="Entrez operation to perform"
    )
    identifier: Optional[str] = Field(
        default=None,
        description="Accession or ID for efetch/esummary",
    )
    search_term: Optional[str] = Field(
        default=None,
        description="Entrez search query for esearch",
    )
    rettype: str = Field(
        default="fasta",
        description="Return type for efetch (fasta, fasta_cds_na, gb, etc.)",
    )
    seq_start: Optional[int] = Field(
        default=None,
        ge=1,
        description="1-indexed inclusive start for subsequence",
    )
    seq_stop: Optional[int] = Field(
        default=None,
        ge=1,
        description="1-indexed inclusive stop for subsequence",
    )
    strand: Optional[Literal["+", "-"]] = Field(
        default=None,
        description="Strand for nucleotide retrieval",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum results from esearch",
    )

    @model_validator(mode="after")
    def validate_identifier_or_term(self):
        """Validate that the right fields are provided for each operation."""
        if self.operation == "esearch" and not self.search_term:
            raise ValueError("esearch requires search_term")
        if self.operation in ("efetch", "esummary") and not self.identifier:
            raise ValueError(f"{self.operation} requires identifier")
        return self


class NCBIFastaRecord(BaseModel):
    """A parsed FASTA record.

    Attributes:
        header: FASTA header line (without >).
        sequence: Sequence string with whitespace stripped.
        accession: Best-effort accession extracted from header.
    """

    header: str = Field(description="FASTA header line")
    sequence: str = Field(description="Sequence string")
    accession: Optional[str] = Field(
        default=None, description="Accession extracted from header"
    )


class NCBIFetchOutput(BaseToolOutput):
    """Output from NCBI fetch tool.

    Attributes:
        operation: The Entrez operation that was performed.
        db: The database queried.
        ids: ID list returned by esearch (empty for other operations).
        summary: Summary data returned by esummary (empty for other operations).
        fasta_records: Parsed FASTA records from efetch.
        raw_text: Raw response text from efetch (non-FASTA rettypes).
        source_url: Sanitized URL used for the request.
    """

    operation: str = Field(description="Entrez operation performed")
    db: str = Field(description="Database queried")
    ids: List[str] = Field(default_factory=list, description="esearch ID list")
    summary: Dict[str, Any] = Field(
        default_factory=dict, description="esummary result data"
    )
    fasta_records: List[NCBIFastaRecord] = Field(
        default_factory=list, description="Parsed FASTA records"
    )
    raw_text: Optional[str] = Field(
        default=None, description="Raw text for non-FASTA rettypes"
    )
    source_url: Optional[str] = Field(
        default=None, description="Sanitized request URL"
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


class NCBIFetchConfig(BaseConfig):
    """Configuration for NCBI fetch operations.

    Attributes:
        request_timeout_seconds: HTTP timeout per request.
        http_retries: Number of retries for failed requests.
        backoff_seconds: Base backoff time between retries.
        ncbi_api_key: Optional NCBI API key for higher rate limits.
        ncbi_email: Optional contact email for NCBI requests.
        user_agent: HTTP user-agent string.
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
        description="Base exponential backoff time",
        advanced=True,
    )
    ncbi_api_key: Optional[str] = ConfigField(
        title="NCBI API Key",
        default=None,
        description="Optional NCBI API key",
        advanced=True,
    )
    ncbi_email: Optional[str] = ConfigField(
        title="NCBI Email",
        default=None,
        description="Optional NCBI contact email",
        advanced=True,
    )
    user_agent: str = ConfigField(
        title="User Agent",
        default="bio-programming-tools/ncbi-fetch-v1",
        description="HTTP user-agent string",
        advanced=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================


@tool(
    key="ncbi-fetch",
    label="NCBI Entrez Fetch",
    category="database_retrieval",
    input=NCBIFetchInput,
    config=NCBIFetchConfig,
    output=NCBIFetchOutput,
    description="Search and fetch sequences from NCBI Entrez (esearch, esummary, efetch)",
    uses_gpu=False,
)
def run_ncbi_fetch(
    inputs: NCBIFetchInput,
    config: NCBIFetchConfig,
    instance=None,
) -> NCBIFetchOutput:
    """Search and fetch sequences from NCBI Entrez databases.

    Supports esearch (ID search), esummary (record summaries), and efetch
    (sequence/record retrieval) across protein, nuccore, and gene databases.

    Args:
        inputs: A single NCBI fetch request specifying db, operation, and
            identifier or search term.
        config: HTTP timeout, retry, and authentication settings.

    Returns:
        NCBIFetchOutput: Result containing IDs, summaries, or FASTA records
            depending on the operation.
    """
    del instance

    session = build_http_session(
        http_retries=config.http_retries,
        backoff_seconds=config.backoff_seconds,
        user_agent=config.user_agent,
        allowed_methods=["GET", "POST"],
    )

    try:
        if inputs.operation == "esearch":
            ids = _ncbi_esearch(
                db=inputs.db,
                term=inputs.search_term,
                max_results=inputs.max_results,
                config=config,
                session=session,
            )
            return NCBIFetchOutput(
                operation="esearch", db=inputs.db, ids=ids
            )

        if inputs.operation == "esummary":
            result = _ncbi_esummary(
                db=inputs.db,
                identifier=inputs.identifier,
                config=config,
                session=session,
            )
            if result is None:
                raise ValueError(
                    f"No record found for {inputs.db}:{inputs.identifier}"
                )
            summary, url = result
            return NCBIFetchOutput(
                operation="esummary",
                db=inputs.db,
                summary=summary,
                source_url=url,
            )

        # efetch
        result = _ncbi_efetch(
            db=inputs.db,
            identifier=inputs.identifier,
            rettype=inputs.rettype,
            config=config,
            session=session,
            seq_start=inputs.seq_start,
            seq_stop=inputs.seq_stop,
            strand=inputs.strand,
        )
        if result is None:
            raise ValueError(
                f"No record found for {inputs.db}:{inputs.identifier}"
            )
        text, url = result

        if inputs.rettype in ("fasta", "fasta_cds_na"):
            records = _parse_fasta_records(text)
            return NCBIFetchOutput(
                operation="efetch",
                db=inputs.db,
                fasta_records=records,
                source_url=url,
            )

        return NCBIFetchOutput(
            operation="efetch",
            db=inputs.db,
            raw_text=text,
            source_url=url,
        )
    finally:
        session.close()


# ============================================================================
# Private Helpers
# ============================================================================


def _ncbi_common_params(config: NCBIFetchConfig) -> Dict[str, Any]:
    """Build common NCBI eutils parameters."""
    params: Dict[str, Any] = {"tool": "bio_programming_tools_ncbi_fetch"}
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
) -> List[str]:
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
        params=params,
        timeout=config.request_timeout_seconds,
    )
    if not _check_response(response, "ncbi-esearch"):
        return []
    data = response.json()
    return data.get("esearchresult", {}).get("idlist", [])


def _ncbi_esummary(
    db: str,
    identifier: str,
    config: NCBIFetchConfig,
    session: requests.Session,
) -> Optional[Tuple[Dict[str, Any], str]]:
    """Run NCBI esummary and return (result_map, sanitized_url) or None."""
    params: Dict[str, Any] = {
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
    data = response.json()
    return data.get("result", {}), url


def _ncbi_efetch(
    db: str,
    identifier: str,
    rettype: str,
    config: NCBIFetchConfig,
    session: requests.Session,
    seq_start: Optional[int] = None,
    seq_stop: Optional[int] = None,
    strand: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """Run NCBI efetch and return (text, sanitized_url) or None."""
    params: Dict[str, Any] = {
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


def _parse_fasta_records(text: str) -> List[NCBIFastaRecord]:
    """Parse FASTA text into NCBIFastaRecord objects."""
    if not text or not text.strip():
        return []
    return [
        NCBIFastaRecord(
            header=record.description,
            sequence=str(record.seq),
            accession=_accession_from_header(record.description),
        )
        for record in SeqIO.parse(StringIO(text), "fasta")
    ]


def _accession_from_header(header: str) -> Optional[str]:
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
