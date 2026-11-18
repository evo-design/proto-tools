"""proto_tools/tools/database_retrieval/biorxiv/biorxiv_details.py.

bioRxiv and medRxiv preprint metadata by DOI: every posted version with its date and JATS XML
link, plus title, authors, category, license, abstract and the journal DOI once published.
"""

import json
import logging
import re
from functools import partial
from pathlib import Path
from typing import Any, Literal

import requests
from pydantic import BaseModel, ConfigDict, Field, field_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    build_http_session,
    raise_for_status_with_body,
    request_with_retry,
)
from proto_tools.utils.standalone_helpers_source.standalone_helpers import StatusCounter

logger = logging.getLogger(__name__)

_API_URL = "https://api.biorxiv.org/details"
_DETAILS_URL = _API_URL + "/{server}/{doi}"
_REQUEST_TIMEOUT_SECONDS = 30
_HTTP_RETRIES = 2
_BACKOFF_SECONDS = 1.0
_USER_AGENT = "proto-tools/biorxiv-details-v1"
_DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)

PreprintServer = Literal["auto", "biorxiv", "medrxiv"]
_AUTO_SERVERS = ("biorxiv", "medrxiv")


# ============================================================================
# Data Models
# ============================================================================


class BiorxivDetailsInput(BaseToolInput):
    """Preprint DOIs to look up.

    Attributes:
        dois (list[str]): bioRxiv or medRxiv DOIs (e.g. ``"10.1101/2022.12.09.519842"``). A
            ``https://doi.org/`` or ``doi:`` prefix is removed. A bare string is normalized to a
            one-item list.
    """

    dois: list[str] = InputField(
        min_length=1,
        title="DOIs",
        description="bioRxiv or medRxiv DOIs, e.g. '10.1101/2022.12.09.519842'",
    )

    @field_validator("dois", mode="before")
    @classmethod
    def _normalize_dois(cls, value: Any) -> Any:
        """Accept a bare string and strip DOI URL prefixes; reject anything that is not a DOI."""
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return value
        cleaned = []
        for item in value:
            doi = _DOI_PREFIX_RE.sub("", str(item).strip())
            if not doi.startswith("10."):
                raise ValueError(f"not a DOI: {item!r} (expected e.g. '10.1101/2022.12.09.519842')")
            cleaned.append(doi)
        return cleaned


class BiorxivDetailsConfig(BaseConfig):
    """Server to query.

    Attributes:
        server (PreprintServer): ``auto`` tries bioRxiv, then medRxiv when bioRxiv has no posts for
            the DOI; ``biorxiv`` or ``medrxiv`` queries only that server. Both servers issue
            ``10.1101`` and ``10.64898`` DOIs, so the prefix does not identify the server.
    """

    server: PreprintServer = ConfigField(
        default="auto",
        title="Server",
        description="'auto' tries bioRxiv then medRxiv; 'biorxiv' or 'medrxiv' queries only that one",
    )


class BiorxivVersion(BaseModel):
    """One posted version of a preprint.

    Attributes:
        version (int): Version number (1 is the first posting).
        date (str): Posting date (``YYYY-MM-DD``).
        type (str | None): Article type, e.g. ``new results``.
        jats_xml_url (str | None): URL of this version's JATS XML full text.
    """

    model_config = ConfigDict(extra="forbid")

    version: int = Field(title="Version", description="Version number")
    date: str = Field(title="Date", description="Posting date (YYYY-MM-DD)")
    type: str | None = Field(default=None, title="Type", description="Article type, e.g. 'new results'")
    jats_xml_url: str | None = Field(default=None, title="JATS XML URL", description="This version's JATS XML")


class BiorxivPreprint(BaseModel):
    """Metadata for one preprint, from its latest version.

    Attributes:
        doi (str): Requested preprint DOI.
        error (str | None): Why this item has no data; None when it was retrieved.
        server (str | None): Server that answered, as reported (``bioRxiv`` or ``medRxiv``).
        title (str | None): Title of the latest version.
        authors (list[str]): Authors (``"Surname, I."``) of the latest version.
        corresponding_author (str | None): Corresponding author.
        corresponding_institution (str | None): Corresponding author's institution.
        category (str | None): Subject category, e.g. ``bioengineering``.
        license (str | None): License code, e.g. ``cc_by``.
        abstract (str | None): Abstract of the latest version.
        published_doi (str | None): Journal DOI once the preprint is published, else None.
        versions (list[BiorxivVersion]): All posted versions, oldest first.
        source_url (str | None): bioRxiv API URL the record was read from.
    """

    model_config = ConfigDict(extra="forbid")

    doi: str = Field(title="DOI", description="Requested preprint DOI")
    error: str | None = Field(
        default=None, title="Error", description="Why this item has no data; None when it was retrieved"
    )
    server: str | None = Field(default=None, title="Server", description="Server that answered, as reported")
    title: str | None = Field(default=None, title="Title", description="Title of the latest version")
    authors: list[str] = Field(default_factory=list, title="Authors", description="Authors of the latest version")
    corresponding_author: str | None = Field(
        default=None, title="Corresponding Author", description="Corresponding author"
    )
    corresponding_institution: str | None = Field(
        default=None, title="Corresponding Institution", description="Corresponding author's institution"
    )
    category: str | None = Field(default=None, title="Category", description="Subject category")
    license: str | None = Field(default=None, title="License", description="License code, e.g. 'cc_by'")
    abstract: str | None = Field(default=None, title="Abstract", description="Abstract of the latest version")
    published_doi: str | None = Field(
        default=None, title="Published DOI", description="Journal DOI once published, else None"
    )
    versions: list[BiorxivVersion] = Field(
        default_factory=list, title="Versions", description="All posted versions, oldest first"
    )
    source_url: str | None = Field(
        default=None, title="Source URL", description="bioRxiv API URL the record was read from"
    )


class BiorxivDetailsOutput(BaseToolOutput):
    """Preprint records, one per input DOI.

    Attributes:
        preprints (list[BiorxivPreprint]): Records in input order; a DOI that is not found carries ``error``.
        source_url (str): bioRxiv API endpoint used.
    """

    preprints: list[BiorxivPreprint] = Field(
        default_factory=list, title="Preprints", description="Records in input order"
    )
    source_url: str = Field(title="Source URL", description="bioRxiv API endpoint used")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Any, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")
        if file_format == "json":
            with path.open("w", encoding="utf-8") as f:
                json.dump(self.model_dump(mode="json"), f, indent=2)
            return
        raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return BiorxivDetailsInput(dois=["10.1101/2022.12.09.519842"])


@tool(
    key="biorxiv-details",
    local_only="biorxiv-details does not use a gpu and does not need an environment, so it can run in process",
    label="bioRxiv Details",
    category="database_retrieval",
    input_class=BiorxivDetailsInput,
    config_class=BiorxivDetailsConfig,
    output_class=BiorxivDetailsOutput,
    description=(
        "Look up bioRxiv or medRxiv preprint metadata by DOI: versions and dates, authors, abstract, "
        "and the published journal DOI"
    ),
    uses_gpu=False,
    example_input=example_input,
    iterable_input_fields=["dois"],
    iterable_output_field="preprints",
    cacheable=True,
)
def run_biorxiv_details(
    inputs: BiorxivDetailsInput,
    config: BiorxivDetailsConfig,
    instance: Any = None,
) -> BiorxivDetailsOutput:
    """Fetch metadata and version history for each preprint DOI.

    Args:
        inputs (BiorxivDetailsInput): Preprint DOIs.
        config (BiorxivDetailsConfig): Server to query.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        BiorxivDetailsOutput: One record per DOI, in input order. A DOI that no queried server
            holds, or that the API rejects with an HTTP 4xx other than 429, gets a record with
            ``error`` set instead of failing the call.

    Raises:
        requests.HTTPError: On HTTP 429 or 5xx after retries.
    """
    del instance

    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
    )
    try:
        preprints = [
            _details(doi, config.server, session)
            for doi in StatusCounter(logger, "Fetching preprint %d/%d", inputs.dois)
        ]
        return BiorxivDetailsOutput(preprints=preprints, source_url=_API_URL)
    finally:
        session.close()


# ============================================================================
# Private Helpers
# ============================================================================


def _details(doi: str, server: str, session: requests.Session) -> BiorxivPreprint:
    """Look the DOI up on each candidate server in turn; return the first hit or an item error."""
    servers = _AUTO_SERVERS if server == "auto" else (server,)
    status = None
    for name in servers:
        url = _DETAILS_URL.format(server=name, doi=doi)
        response = request_with_retry(
            partial(session.get, url, timeout=_REQUEST_TIMEOUT_SECONDS),
            retries=_HTTP_RETRIES,
            backoff_seconds=_BACKOFF_SECONDS,
        )
        try:
            raise_for_status_with_body(response, f"{name} details request for DOI {doi}")
        except requests.HTTPError as error:
            if 400 <= response.status_code < 500 and response.status_code != 429:
                return BiorxivPreprint(doi=doi, error=str(error))
            raise
        payload = response.json()
        records = payload.get("collection") or []
        if records:
            return _preprint(doi, name, url, records)
        status = (payload.get("messages") or [{}])[0].get("status")
    tried = " or ".join(servers)
    return BiorxivPreprint(doi=doi, error=f"No preprint found for DOI {doi} on {tried} ({status})")


def _preprint(doi: str, server: str, url: str, records: list[dict[str, Any]]) -> BiorxivPreprint:
    """Fold one DOI's version records into a preprint, taking metadata from the latest version."""
    records = sorted(records, key=lambda record: int(record.get("version") or 0))
    latest = records[-1]
    published = str(latest.get("published") or "").strip()
    return BiorxivPreprint(
        doi=doi,
        server=str(latest.get("server") or server),
        title=str(latest.get("title") or "").strip(),
        authors=[name.strip() for name in str(latest.get("authors") or "").split(";") if name.strip()],
        corresponding_author=latest.get("author_corresponding") or None,
        corresponding_institution=latest.get("author_corresponding_institution") or None,
        category=latest.get("category") or None,
        license=latest.get("license") or None,
        abstract=latest.get("abstract") or None,
        published_doi=published if published and published != "NA" else None,
        versions=[
            BiorxivVersion(
                version=int(record["version"]),
                date=str(record["date"]),
                type=record.get("type") or None,
                jats_xml_url=record.get("jatsxml") or None,
            )
            for record in records
        ],
        source_url=url,
    )
