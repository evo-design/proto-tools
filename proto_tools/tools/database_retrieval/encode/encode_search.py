"""proto_tools/tools/database_retrieval/encode/encode_search.py.

Faceted search of the ENCODE portal: each query is a set of ENCODE facet filters (assay, target,
biosample, output type, assembly, file format) plus optional free text, and returns matching
processed files or experiments with their accessions, metadata and direct download URLs.
"""

import csv
import difflib
import hashlib
import json
import logging
import os
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal
from urllib.parse import urlencode

import requests
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    build_http_session,
    request_with_retry,
)
from proto_tools.utils.standalone_helpers_source.standalone_helpers import StatusCounter

logger = logging.getLogger(__name__)

_PORTAL_URL = "https://www.encodeproject.org"
_SEARCH_URL = f"{_PORTAL_URL}/search/"
_REQUEST_TIMEOUT_SECONDS = 60
_HTTP_RETRIES = 2
_BACKOFF_SECONDS = 1.0
_USER_AGENT = "proto-tools/encode-search-v1"
_DOWNLOAD_CHUNK_BYTES = 1 << 20
# Connect timeout, then the longest wait for the next chunk: a slow transfer that keeps sending data
# is never cut off, only one that stalls.
_DOWNLOAD_TIMEOUT_SECONDS = (30, 300)
_DEFAULT_MAX_DOWNLOAD_BYTES = 1_000_000_000
_RESERVED_PARAMS = frozenset({"type", "format", "limit", "field", "frame", "searchTerm"})
_RETURN_FIELDS = (
    "@id",
    "accession",
    "dataset",
    "href",
    "file_size",
    "md5sum",
    "output_type",
    "assembly",
    "file_format",
    "file_type",
    "biological_replicates",
    "preferred_default",
    "status",
    "assay_title",
    "date_released",
    "date_created",
    "description",
    "lab.title",
    "target.label",
    "biosample_ontology.term_name",
    "donors",
    "replicates.library.biosample.organism.scientific_name",
)
# Files carry no organism field, only donor links such as /human-donors/ENCDO.../. Human and mouse
# donors are one species each; fly and worm donors span several species, so those fall back to the
# assembly, and stay None when neither identifies the species.
_DONOR_ORGANISMS = {"human-donors": "Homo sapiens", "mouse-donors": "Mus musculus"}
_ASSEMBLY_ORGANISMS = {
    "GRCh38": "Homo sapiens",
    "hg19": "Homo sapiens",
    "GRCm39": "Mus musculus",
    "mm10": "Mus musculus",
    "mm9": "Mus musculus",
    "dm6": "Drosophila melanogaster",
    "dm3": "Drosophila melanogaster",
    "ce11": "Caenorhabditis elegans",
    "ce10": "Caenorhabditis elegans",
}


# ============================================================================
# Data Models
# ============================================================================


class EncodeSearchQuery(BaseModel):
    """One ENCODE portal search.

    A plain dict without ``filters``/``search_term`` keys is read as the ``filters`` mapping.

    Attributes:
        filters (dict[str, str | list[str]]): ENCODE facet filters, e.g. ``{"assay_title": "TF
            ChIP-seq", "target.label": "CTCF", "biosample_ontology.term_name": "K562",
            "output_type": "IDR thresholded peaks", "assembly": "GRCh38", "file_format": "bed"}``.
            Values must match the portal's facet terms exactly. A list matches any of its values,
            e.g. ``{"target.label": ["CTCF", "RAD21"]}``. A key ending in ``!`` negates the filter;
            with a list, it excludes every value.
        search_term (str | None): Free-text search (the portal's ``searchTerm``), e.g.
            ``"MAFK HepG2"``.
    """

    model_config = ConfigDict(extra="forbid")

    filters: dict[str, str | list[str]] = Field(
        default_factory=dict,
        title="Filters",
        description="ENCODE facet filters, e.g. {'target.label': 'CTCF'}; a list matches any of its values",
    )
    search_term: str | None = Field(default=None, title="Search Term", description="Free-text search term")

    @model_validator(mode="before")
    @classmethod
    def _bare_filters(cls, value: Any) -> Any:
        """Read a plain filter mapping as ``filters``."""
        if isinstance(value, dict) and not {"filters", "search_term"} & value.keys():
            return {"filters": value}
        return value

    @model_validator(mode="after")
    def _validate(self) -> "EncodeSearchQuery":
        """Strip values and reject empty queries and reserved parameters."""
        self.filters = {
            key.strip(): val.strip() if isinstance(val, str) else [item.strip() for item in val]
            for key, val in self.filters.items()
        }
        self.search_term = (self.search_term or "").strip() or None
        reserved = _RESERVED_PARAMS & self.filters.keys()
        if reserved:
            raise ValueError(f"filters must not set {sorted(reserved)}; use search_term or the config instead")
        if any(not key for key in self.filters) or any(not val for _key, val in self.filter_pairs()):
            raise ValueError("filter keys and values must not be empty")
        if any(isinstance(val, list) and not val for val in self.filters.values()):
            raise ValueError("a filter list needs at least one value")
        if not self.filters and self.search_term is None:
            raise ValueError("a query needs at least one filter or a search_term")
        return self

    def filter_pairs(self) -> list[tuple[str, str]]:
        """Return the filters as ``(key, value)`` pairs, one per value of a list filter."""
        return [(key, item) for key, val in self.filters.items() for item in ([val] if isinstance(val, str) else val)]


class EncodeSearchInput(BaseToolInput):
    """Searches to run against the ENCODE portal.

    Attributes:
        queries (list[EncodeSearchQuery]): One or more searches. A bare dict (a query or a plain
            filter mapping) is normalized to a one-item list.
    """

    queries: list[EncodeSearchQuery] = InputField(
        min_length=1, title="Queries", description="ENCODE searches; a bare dict is accepted"
    )

    @field_validator("queries", mode="before")
    @classmethod
    def _normalize_queries(cls, value: Any) -> Any:
        """Accept a single query or filter mapping."""
        if isinstance(value, (dict, EncodeSearchQuery)):
            return [value]
        return value


class EncodeSearchConfig(BaseConfig):
    """Item type, status filters and result size for ENCODE search.

    Attributes:
        item_type (Literal["File", "Experiment"]): Search processed and raw files, or experiments.
        released_only (bool): Restrict to ``status=released``, excluding archived, revoked and
            in-progress items. Ignored when the query sets its own ``status`` filter.
        default_only (bool): Restrict files to ``preferred_default=true``, the portal's default
            file per output type (e.g. the default IDR thresholded peaks). Files only.
        max_results (int): Maximum hits returned per query (1-1000).
    """

    item_type: Literal["File", "Experiment"] = ConfigField(
        default="File", title="Item Type", description="Search files or experiments"
    )
    released_only: bool = ConfigField(default=True, title="Released Only", description="Restrict to released items")
    default_only: bool = ConfigField(
        default=False, title="Default Only", description="Restrict files to the portal's preferred default files"
    )
    max_results: int = ConfigField(
        default=25, ge=1, le=1000, title="Max Results", description="Maximum hits returned per query"
    )


class EncodeSearchHit(BaseModel):
    """One ENCODE file or experiment returned by a search.

    Attributes:
        accession (str): File (``ENCFF...``) or experiment (``ENCSR...``) accession.
        dataset_accession (str | None): Accession of the experiment (or other dataset) the file
            belongs to; the accession itself for experiments.
        assay_title (str | None): Assay, e.g. ``TF ChIP-seq``.
        target (str | None): Assay target label, e.g. ``CTCF``.
        biosample (str | None): Biosample term name, e.g. ``K562``.
        organism (str | None): Source species, e.g. ``Homo sapiens``; comma-joined if several.
        output_type (str | None): File output type, e.g. ``IDR thresholded peaks``.
        assembly (str | None): Genome assembly; comma-joined for experiments with several.
        file_format (str | None): File format, e.g. ``bed``.
        file_type (str | None): Detailed file type, e.g. ``bed narrowPeak``.
        file_size (int | None): File size in bytes.
        md5sum (str | None): MD5 checksum of the file.
        biological_replicates (list[int]): Biological replicates the file is derived from.
        preferred_default (bool | None): Whether the file is the portal's default for its type.
        status (str | None): Portal status, e.g. ``released``.
        lab (str | None): Submitting lab or ``ENCODE Processing Pipeline``.
        date_released (str | None): Release date (experiments).
        date_created (str | None): Creation date (files), ``YYYY-MM-DD``.
        description (str | None): Experiment description.
        download_url (str | None): Direct download URL (files only).
        portal_url (str): Portal page for the item.
    """

    model_config = ConfigDict(extra="forbid")

    accession: str = Field(title="Accession", description="File or experiment accession")
    dataset_accession: str | None = Field(default=None, title="Dataset Accession", description="Parent experiment")
    assay_title: str | None = Field(default=None, title="Assay", description="Assay title")
    target: str | None = Field(default=None, title="Target", description="Assay target label")
    biosample: str | None = Field(default=None, title="Biosample", description="Biosample term name")
    organism: str | None = Field(
        default=None, title="Organism", description="Source species, e.g. Homo sapiens; None if not identifiable"
    )
    output_type: str | None = Field(default=None, title="Output Type", description="File output type")
    assembly: str | None = Field(default=None, title="Assembly", description="Genome assembly")
    file_format: str | None = Field(default=None, title="File Format", description="File format")
    file_type: str | None = Field(default=None, title="File Type", description="Detailed file type")
    file_size: int | None = Field(default=None, title="File Size", description="File size in bytes")
    md5sum: str | None = Field(default=None, title="MD5", description="File MD5 checksum")
    biological_replicates: list[int] = Field(
        default_factory=list, title="Biological Replicates", description="Source biological replicates"
    )
    preferred_default: bool | None = Field(
        default=None, title="Preferred Default", description="Portal default file for its type"
    )
    status: str | None = Field(default=None, title="Status", description="Portal status")
    lab: str | None = Field(default=None, title="Lab", description="Submitting lab")
    date_released: str | None = Field(default=None, title="Date Released", description="Release date")
    date_created: str | None = Field(default=None, title="Date Created", description="Creation date")
    description: str | None = Field(default=None, title="Description", description="Experiment description")
    download_url: str | None = Field(default=None, title="Download URL", description="Direct file download URL")
    portal_url: str = Field(title="Portal URL", description="Portal page for the item")


class EncodeSearchResult(BaseModel):
    """Hits for one query.

    Attributes:
        query (EncodeSearchQuery): The query as given.
        request_url (str): Full search URL sent, including type, status and default filters.
        total_count (int): Total matching items reported by the portal.
        hits (list[EncodeSearchHit]): Hits in the portal's order.
    """

    model_config = ConfigDict(extra="forbid")

    query: EncodeSearchQuery = Field(title="Query", description="Query as given")
    request_url: str = Field(title="Request URL", description="Full search URL sent")
    total_count: int = Field(title="Total Count", description="Total matching items")
    hits: list[EncodeSearchHit] = Field(default_factory=list, title="Hits", description="Hits in portal order")


class EncodeSearchOutput(BaseToolOutput):
    """Search results, one per input query.

    Attributes:
        results (list[EncodeSearchResult]): Results in input order.
        source_url (str): ENCODE search endpoint used.
    """

    results: list[EncodeSearchResult] = Field(
        default_factory=list, title="Results", description="Results in input order"
    )
    source_url: str = Field(title="Source URL", description="ENCODE search endpoint used")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "csv"]

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
        if file_format == "csv":
            fieldnames = ["query_index", "rank", *EncodeSearchHit.model_fields]
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for index, result in enumerate(self.results):
                    for rank, hit in enumerate(result.hits, start=1):
                        row = hit.model_dump()
                        row["biological_replicates"] = ";".join(str(rep) for rep in row["biological_replicates"])
                        writer.writerow({"query_index": index, "rank": rank, **row})
            return
        raise ValueError(f"Unsupported format: {file_format}")

    def download(
        self,
        directory: str | Path,
        accessions: list[str] | None = None,
        max_total_bytes: int | None = _DEFAULT_MAX_DOWNLOAD_BYTES,
    ) -> dict[str, Path]:
        """Download file hits from their ``download_url`` into ``directory``.

        Each file is checked against its ENCODE MD5. A file already in ``directory`` with the right
        MD5 is kept without downloading again, so repeat calls are cheap.

        Args:
            directory (str | Path): Folder to download into; created if missing.
            accessions (list[str] | None): File accessions to download; all file hits when None.
            max_total_bytes (int | None): Refuse, before downloading anything, when the files still to
                download add up to more than this; None for no limit. Defaults to 1 GB, which covers
                peak and quantification files; signal tracks, alignments and reads are often larger.

        Returns:
            dict[str, Path]: Local path of each file, keyed by accession, in hit order.

        Raises:
            ValueError: If a requested accession is not a file hit, there is nothing to download, or
                the total size exceeds ``max_total_bytes``.
            OSError: If a downloaded file does not match its ENCODE MD5.
        """
        hits = {hit.accession: hit for result in self.results for hit in result.hits if hit.download_url}
        if accessions is not None:
            missing = [accession for accession in accessions if accession not in hits]
            if missing:
                raise ValueError(f"Not file hits with a download URL in this output: {missing}")
            hits = {accession: hits[accession] for accession in dict.fromkeys(accessions)}
        if not hits:
            raise ValueError("No file hits to download; experiment searches have no download URLs")
        return _download_files(list(hits.values()), Path(directory), max_total_bytes)


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return EncodeSearchInput(
        queries=[
            EncodeSearchQuery(
                filters={
                    "assay_title": "TF ChIP-seq",
                    "target.label": "CTCF",
                    "biosample_ontology.term_name": "K562",
                    "output_type": "IDR thresholded peaks",
                    "assembly": "GRCh38",
                    "file_format": "bed",
                }
            )
        ]
    )


@tool(
    key="encode-search",
    local_only="encode-search does not use a gpu and does not need an environment, so it can run in process",
    label="ENCODE Search",
    category="database_retrieval",
    input_class=EncodeSearchInput,
    config_class=EncodeSearchConfig,
    output_class=EncodeSearchOutput,
    description=(
        "Search the ENCODE portal for experiments and processed files (peaks, signal, alignments) "
        "by assay, target, biosample and assembly; returns accessions, metadata and download URLs"
    ),
    uses_gpu=False,
    example_input=example_input,
    iterable_input_fields=["queries"],
    iterable_output_field="results",
    cacheable=True,
)
def run_encode_search(
    inputs: EncodeSearchInput,
    config: EncodeSearchConfig,
    instance: Any = None,
) -> EncodeSearchOutput:
    """Search the ENCODE portal and return matching files or experiments for each query.

    Args:
        inputs (EncodeSearchInput): Facet filters and optional free text per query.
        config (EncodeSearchConfig): Item type, released/default filters and result size.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        EncodeSearchOutput: One result per query, with download URLs for files.
    """
    del instance

    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
    )
    try:
        results = [
            _search(query, config, session) for query in StatusCounter(logger, "Running query %d/%d", inputs.queries)
        ]
        return EncodeSearchOutput(results=results, source_url=_SEARCH_URL)
    finally:
        session.close()


# ============================================================================
# Private Helpers
# ============================================================================


def _request_url(query: EncodeSearchQuery, config: EncodeSearchConfig) -> str:
    """Build the search URL for one query."""
    params: list[tuple[str, str]] = [("type", config.item_type)]
    if config.released_only and "status" not in query.filters:
        params.append(("status", "released"))
    if config.default_only and config.item_type == "File" and "preferred_default" not in query.filters:
        params.append(("preferred_default", "true"))
    params.extend(query.filter_pairs())
    if query.search_term:
        params.append(("searchTerm", query.search_term))
    params.extend([("format", "json"), ("limit", str(config.max_results))])
    params.extend(("field", field) for field in _RETURN_FIELDS)
    return f"{_SEARCH_URL}?{urlencode(params)}"


def _raise_for_status(response: Any) -> None:
    """Raise an HTTP error that includes a bounded ENCODE response body (e.g. bad facet names)."""
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        body = str(getattr(response, "text", ""))[:1000]
        raise requests.HTTPError(f"ENCODE search failed: {error}; body={body!r}", response=response) from error


def _fetch(session: requests.Session, url: str) -> dict[str, Any]:
    """GET one portal search; the portal's no-match answer (404 with an empty @graph) becomes total 0."""
    response = request_with_retry(
        lambda: session.get(url, headers={"Accept": "application/json"}, timeout=_REQUEST_TIMEOUT_SECONDS),
        retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
    )
    if response.status_code == 404 and "@graph" in response.text:
        return {"total": 0, "@graph": []}
    _raise_for_status(response)
    payload: dict[str, Any] = response.json()
    return payload


def _search(query: EncodeSearchQuery, config: EncodeSearchConfig, session: requests.Session) -> EncodeSearchResult:
    """Run one search request, parse its hits, and warn about filters that match nothing."""
    url = _request_url(query, config)
    payload = _fetch(session, url)
    items = payload.get("@graph", [])
    total = int(payload.get("total", len(items)))
    if total == 0:
        logger.info("ENCODE search returned no matches: %s", url)
    for message in _diagnose(query, config, session, no_hits=total == 0):
        warnings.warn(message, UserWarning, stacklevel=2)
    return EncodeSearchResult(
        query=query,
        request_url=url,
        total_count=total,
        hits=[_parse_hit(item) for item in items],
    )


def _count_url(config: EncodeSearchConfig, params: list[tuple[str, str]]) -> str:
    """Build a count-only search URL (no hits returned) for the item type and status filter."""
    base: list[tuple[str, str]] = [("type", config.item_type)]
    if config.released_only and "status" not in dict(params):
        base.append(("status", "released"))
    return f"{_SEARCH_URL}?{urlencode([*base, *params, ('format', 'json'), ('limit', '0')])}"


def _diagnose(
    query: EncodeSearchQuery, config: EncodeSearchConfig, session: requests.Session, no_hits: bool
) -> list[str]:
    """Explain filters whose value matches no item.

    The portal echoes unknown facet names and misspelled values back without complaint, so a typo
    looks like a real empty result. Each suspect value is counted on its own: every value when the
    search found nothing; otherwise negated values and values in lists, since a misspelled ``key!``
    excludes nothing and a misspelled list value adds nothing, with no effect on the hit count.
    """
    in_list = {key for key, val in query.filters.items() if isinstance(val, list)}
    suspects = [(key, value) for key, value in query.filter_pairs() if no_hits or key.endswith("!") or key in in_list]
    messages: list[str] = []
    facets: dict[str, list[str]] | None = None
    for key, value in suspects:
        field = key.removesuffix("!")
        if int(_fetch(session, _count_url(config, [(field, value)])).get("total", 0)) > 0:
            continue
        if facets is None:
            facets = {
                facet["field"]: [str(term.get("key")) for term in facet.get("terms") or []]
                for facet in _fetch(session, _count_url(config, [])).get("facets") or []
                if isinstance(facet, dict) and facet.get("field")
            }
        messages.append(_explain(key, field, value, facets, config, key in in_list))
    return messages


def _explain(
    key: str, field: str, value: str, facets: dict[str, list[str]], config: EncodeSearchConfig, in_list: bool
) -> str:
    """Describe a filter value that matches nothing, with likely corrections from the portal's facets."""
    scope = f"{'released ' if config.released_only else ''}{config.item_type}"
    if key.endswith("!"):
        message = f"Filter {key}={value!r} excludes nothing: no {scope} has {field}={value!r}"
    elif in_list:
        message = f"Filter {field}={value!r} in a list matches no {scope}, so it adds no hits"
    else:
        message = f"Filter {field}={value!r} matches no {scope}, so the search cannot return hits"
    if field in facets:
        close = _close_matches(value, facets[field])
        hint = f"did you mean {_either(close)}?" if close else "values must match the portal's terms exactly"
    else:
        close = _close_matches(field, list(facets))
        hint = f"{field!r} is not a {config.item_type} search facet"
        hint += f"; did you mean {_either(close)}?" if close else ""
    return f"{message} ({hint})"


def _parse_hit(item: dict[str, Any]) -> EncodeSearchHit:
    """Convert one ENCODE search item into a hit."""
    item_path = str(item.get("@id") or "")
    is_file = "/files/" in item_path
    assembly = item.get("assembly")
    href = item.get("href")
    return EncodeSearchHit(
        accession=item.get("accession") or _path_name(item_path) or "",
        dataset_accession=_path_name(item.get("dataset")) if is_file else item.get("accession"),
        assay_title=item.get("assay_title"),
        target=_linked_name(item.get("target"), "label"),
        biosample=_linked_name(item.get("biosample_ontology"), "term_name"),
        organism=_organism(item, assembly),
        output_type=item.get("output_type"),
        assembly=", ".join(assembly) if isinstance(assembly, list) else assembly,
        file_format=item.get("file_format"),
        file_type=item.get("file_type"),
        file_size=item.get("file_size"),
        md5sum=item.get("md5sum"),
        biological_replicates=list(item.get("biological_replicates") or []),
        preferred_default=item.get("preferred_default"),
        status=item.get("status"),
        lab=_linked_name(item.get("lab"), "title"),
        date_released=item.get("date_released"),
        date_created=str(item["date_created"])[:10] if item.get("date_created") else None,
        description=item.get("description"),
        download_url=f"{_PORTAL_URL}{href}" if href else None,
        portal_url=f"{_PORTAL_URL}{item_path}",
    )


def _organism(item: dict[str, Any], assembly: Any) -> str | None:
    """Return the item's species: from biosamples (experiments), else donor links or assembly (files)."""
    names = [
        name
        for replicate in item.get("replicates") or []
        if (
            name := (((replicate.get("library") or {}).get("biosample") or {}).get("organism") or {}).get(
                "scientific_name"
            )
        )
    ]
    if not names:
        donors = [_DONOR_ORGANISMS.get(str(donor).strip("/").split("/")[0]) for donor in item.get("donors") or []]
        assemblies = [
            _ASSEMBLY_ORGANISMS.get(str(name))
            for name in (assembly if isinstance(assembly, list) else [assembly])
            if name
        ]
        # Use a source only if it identifies every species; a partial answer would hide the rest.
        names = next((found for found in (donors, assemblies) if found and all(found)), [])
    return ", ".join(dict.fromkeys(str(name) for name in names)) or None


def _download_files(hits: list[EncodeSearchHit], directory: Path, max_total_bytes: int | None) -> dict[str, Path]:
    """Download hits into ``directory`` after checking the size budget; verify each MD5."""
    directory.mkdir(parents=True, exist_ok=True)
    targets = {hit.accession: directory / str(hit.download_url).rsplit("/", 1)[-1] for hit in hits}
    pending = [hit for hit in hits if not _matches_md5(targets[hit.accession], hit.md5sum)]
    total = sum(hit.file_size or 0 for hit in pending)
    if max_total_bytes is not None and total > max_total_bytes:
        sizes = ", ".join(f"{hit.accession} {(hit.file_size or 0) / 1e6:,.1f} MB" for hit in pending)
        raise ValueError(
            f"Downloading {len(pending)} files needs {total / 1e9:.2f} GB, over max_total_bytes "
            f"({max_total_bytes:,}): {sizes}. Raise max_total_bytes or pass fewer accessions."
        )
    if pending:
        session = build_http_session(
            http_retries=_HTTP_RETRIES, backoff_seconds=_BACKOFF_SECONDS, user_agent=_USER_AGENT
        )
        try:
            for hit in pending:
                _download_file(session, hit, targets[hit.accession])
        finally:
            session.close()
    return targets


def _download_file(session: requests.Session, hit: EncodeSearchHit, target: Path) -> None:
    """Stream one file into a scratch folder beside ``target``, verify its MD5, then move it into place."""
    logger.info("Downloading %s (%s bytes) to %s", hit.accession, hit.file_size, target)
    response = request_with_retry(
        lambda: session.get(str(hit.download_url), stream=True, timeout=_DOWNLOAD_TIMEOUT_SECONDS),
        retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
    )
    try:
        _raise_for_status(response)
        with TemporaryDirectory(dir=target.parent, prefix=".encode-download-") as scratch:
            partial = Path(scratch) / target.name
            digest = hashlib.md5()  # noqa: S324 -- matching ENCODE's published checksum, not security
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                    handle.write(chunk)
                    digest.update(chunk)
            if hit.md5sum and digest.hexdigest() != hit.md5sum:
                raise OSError(f"{hit.accession} MD5 {digest.hexdigest()} does not match ENCODE's {hit.md5sum}")
            os.replace(partial, target)
    finally:
        response.close()


def _matches_md5(path: Path, md5sum: str | None) -> bool:
    """Whether ``path`` exists and matches ``md5sum`` (any existing file, when ENCODE gives no MD5)."""
    if not path.is_file():
        return False
    if not md5sum:
        return True
    digest = hashlib.md5()  # noqa: S324 -- matching ENCODE's published checksum, not security
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest() == md5sum


def _close_matches(word: str, candidates: list[str]) -> list[str]:
    """Return up to three candidates resembling ``word``, ignoring case (portal matching is case-sensitive)."""
    by_lower = {candidate.lower(): candidate for candidate in candidates}
    return [by_lower[match] for match in difflib.get_close_matches(word.lower(), list(by_lower), n=3, cutoff=0.6)]


def _either(options: list[str]) -> str:
    """Join options as ``'a' or 'b'``."""
    return " or ".join(repr(option) for option in options)


def _linked_name(value: Any, key: str) -> str | None:
    """Return a name from an embedded object, or the last path segment of a link."""
    if isinstance(value, dict):
        name = value.get(key)
        return str(name) if name else _path_name(value.get("@id"))
    if isinstance(value, str):
        return _path_name(value)
    return None


def _path_name(path: Any) -> str | None:
    """Return the last segment of a portal path such as ``/experiments/ENCSR000BPJ/``."""
    if not isinstance(path, str) or not path.strip("/"):
        return None
    return path.strip("/").rsplit("/", 1)[-1]
