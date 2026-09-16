"""Retrieve literature through Paperclip without maintaining a local corpus."""

from __future__ import annotations

import hashlib
import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from proto_tools.entities.provenance import AccessLevel, RetrievedArtifact, SourceProvenance
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, BaseToolInput, BaseToolOutput, ConfigField, InputField, ToolInstance

PaperclipSource = Literal["pmc", "abstracts", "biorxiv", "medrxiv", "arxiv"]
_DOCUMENT_ID = re.compile(r"^(?:PMC\d+|bio_[A-Za-z0-9]+|med_[A-Za-z0-9]+|arx_[A-Za-z0-9]+)$")


class LiteratureAccessKind(str, Enum):
    """Granularity of literature content actually returned."""

    METADATA = "metadata"
    ABSTRACT = "abstract"
    FULL_TEXT = "full_text"
    INACCESSIBLE = "inaccessible"


class _PaperclipConfig(BaseConfig):
    """Shared Paperclip client settings.

    Attributes:
        paperclip_library_path (str): Directory containing the optional client.
        paperclip_base_url (str | None): Optional Paperclip service override.
        paperclip_api_key (str | None): Optional API key or login credentials.
        timeout_seconds (float): Maximum duration of one service operation.
    """

    paperclip_library_path: str = ConfigField(
        default_factory=lambda: os.environ.get("PAPERCLIP_LIBRARY_PATH", str(Path.home() / ".paperclip" / "lib")),
        title="Paperclip Library Path",
        description="Directory containing the optional gxl_paperclip Python client",
    )
    paperclip_base_url: str | None = ConfigField(
        default_factory=lambda: os.environ.get("PAPERCLIP_BASE_URL"),
        title="Paperclip Base URL",
        description="Optional Paperclip service override",
    )
    paperclip_api_key: str | None = ConfigField(
        default_factory=lambda: os.environ.get("PAPERCLIP_API_KEY"),
        title="Paperclip API Key",
        description="Optional API key; otherwise reuse credentials from paperclip login",
        include_in_key=False,
    )
    timeout_seconds: float = ConfigField(
        default=120.0,
        gt=0,
        le=600,
        title="Timeout Seconds",
        description="Maximum duration of one Paperclip service operation",
    )


class PaperclipSearchInput(BaseToolInput):
    """Biomedical literature query submitted to Paperclip.

    Attributes:
        query (str): Biomedical literature search query.
    """

    query: str = InputField(
        min_length=1,
        max_length=2000,
        title="Query",
        description="Biomedical literature search query",
    )


class PaperclipSearchConfig(_PaperclipConfig):
    """Paperclip corpus selection and bounded search settings.

    Attributes:
        paperclip_library_path (str): Directory containing the optional client.
        paperclip_base_url (str | None): Optional Paperclip service override.
        paperclip_api_key (str | None): Optional API key or login credentials.
        timeout_seconds (float): Maximum duration of one service operation.
        sources (list[PaperclipSource]): Literature corpora to search.
        limit (int): Maximum number of returned hits.
        exact (bool): Whether to use exact-text search.
        since (str | None): Optional date lower bound for supported corpora.
    """

    sources: list[PaperclipSource] = ConfigField(
        default_factory=lambda: ["pmc", "abstracts"],
        title="Sources",
        description="Paperclip literature corpora to search",
    )
    limit: int = ConfigField(default=10, ge=1, le=100, title="Result Limit", description="Maximum search hits")
    exact: bool = ConfigField(default=False, title="Exact Search", description="Use Paperclip exact-text search")
    since: str | None = ConfigField(
        default=None,
        title="Since",
        description="Optional Paperclip date lower bound for supported corpora",
    )

    @field_validator("sources")
    @classmethod
    def _deduplicate_sources(cls, value: list[PaperclipSource]) -> list[PaperclipSource]:
        if not value:
            raise ValueError("sources must contain at least one Paperclip corpus")
        return list(dict.fromkeys(value))


class PaperclipSearchHit(BaseModel):
    """One normalized, citable Paperclip search hit."""

    document_id: str = Field(title="Document ID", description="Stable Paperclip document identifier")
    title: str = Field(title="Title", description="Publication title")
    authors: str | None = Field(default=None, title="Authors", description="Publication author text")
    journal: str | None = Field(default=None, title="Journal", description="Journal or source title")
    publication_date: str | None = Field(default=None, title="Publication Date", description="Publication date text")
    year: int | None = Field(default=None, title="Year", description="Publication year")
    doi: str | None = Field(default=None, title="DOI", description="Digital object identifier")
    pmid: str | None = Field(default=None, title="PMID", description="PubMed identifier")
    pmcid: str | None = Field(default=None, title="PMCID", description="PubMed Central identifier")
    source: str = Field(title="Source", description="Paperclip source corpus")
    abstract: str | None = Field(default=None, title="Abstract", description="Publication abstract when available")
    matched_text: str | None = Field(default=None, title="Matched Text", description="Text fragment matched by search")
    relevance_score: float | None = Field(
        default=None, title="Relevance Score", description="Paperclip relevance score"
    )
    access_kind: LiteratureAccessKind = Field(title="Access Kind", description="Content granularity returned")
    provenance: SourceProvenance = Field(title="Provenance", description="Source and request provenance")
    raw_metadata: dict[str, Any] = Field(
        default_factory=dict, title="Raw Metadata", description="Unmodified Paperclip result", repr=False
    )


class _JsonOutput(BaseToolOutput):
    @property
    def output_format_options(self) -> list[str]:
        return ["json"]

    @property
    def output_format_default(self) -> str:
        return "json"

    def _export_output(self, export_path: Any, file_format: str) -> None:
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        Path(export_path).with_suffix(".json").write_text(
            json.dumps(self.model_dump(mode="json"), indent=2), encoding="utf-8"
        )


class PaperclipSearchOutput(_JsonOutput):
    """Structured Paperclip search results.

    Attributes:
        query (str): Submitted literature query.
        sources (list[str]): Searched Paperclip corpora.
        result_id (str | None): Paperclip search-result identifier.
        hits (list[PaperclipSearchHit]): Normalized literature matches.
    """

    query: str = Field(title="Query", description="Submitted literature query")
    sources: list[str] = Field(title="Sources", description="Searched Paperclip corpora")
    result_id: str | None = Field(default=None, title="Result ID", description="Paperclip result identifier")
    hits: list[PaperclipSearchHit] = Field(
        default_factory=list, title="Hits", description="Normalized literature matches"
    )


class PaperclipDocumentInput(BaseToolInput):
    """Stable Paperclip corpus document identifier.

    Attributes:
        document_id (str): Stable Paperclip document identifier.
    """

    document_id: str = InputField(
        title="Document ID",
        description="Paperclip corpus identifier, such as PMC11754230",
    )

    @field_validator("document_id")
    @classmethod
    def _safe_document_id(cls, value: str) -> str:
        if not _DOCUMENT_ID.fullmatch(value):
            raise ValueError("document_id must be a Paperclip paper identifier without a path")
        return value


class PaperclipDocumentConfig(_PaperclipConfig):
    """Settings for exact Paperclip document retrieval.

    Attributes:
        paperclip_library_path (str): Directory containing the optional client.
        paperclip_base_url (str | None): Optional Paperclip service override.
        paperclip_api_key (str | None): Optional API key or login credentials.
        timeout_seconds (float): Maximum duration of one service operation.
        include_full_text (bool): Retrieve line-addressable text when available.
    """

    include_full_text: bool = ConfigField(
        default=True,
        title="Include Full Text",
        description="Retrieve line-addressable full text when Paperclip makes it available",
    )


class PaperclipDocument(BaseModel):
    """Observed metadata and optionally available full text."""

    document_id: str = Field(title="Document ID", description="Stable Paperclip document identifier")
    metadata: dict[str, Any] = Field(title="Metadata", description="Paperclip publication metadata")
    content: str | None = Field(
        default=None, title="Content", description="Complete line-addressable full text", repr=False
    )
    content_preview: str | None = Field(
        default=None, title="Content Preview", description="Incomplete content returned by Paperclip", repr=False
    )
    expected_content_bytes: int | None = Field(
        default=None, title="Expected Content Bytes", description="Expected full-text byte count"
    )
    access_kind: LiteratureAccessKind = Field(title="Access Kind", description="Content granularity returned")
    provenance: SourceProvenance = Field(title="Provenance", description="Source and request provenance")
    metadata_artifact: RetrievedArtifact | None = Field(
        default=None,
        title="Metadata Artifact",
        description="Checksum and size descriptor for exact Paperclip metadata output",
    )
    artifact: RetrievedArtifact | None = Field(
        default=None,
        title="Artifact",
        description="Checksummed full-text artifact, or metadata artifact when full text was not returned",
    )


class PaperclipDocumentOutput(_JsonOutput):
    """One retrieved Paperclip literature document.

    Attributes:
        document (PaperclipDocument): Retrieved metadata and available content.
    """

    document: PaperclipDocument = Field(title="Document", description="Retrieved literature document")


def _shared_config(config: _PaperclipConfig) -> dict[str, Any]:
    """Serialize client settings for the managed standalone worker."""
    return {
        "paperclip_library_path": config.paperclip_library_path,
        "paperclip_base_url": config.paperclip_base_url,
        "paperclip_api_key": config.paperclip_api_key,
        "timeout_seconds": config.timeout_seconds,
    }


def _search_example() -> Any:
    return PaperclipSearchInput(query="CAR T cell binder design")


@tool(
    key="paperclip-literature-search",
    label="Paperclip Literature Search",
    category="database_retrieval",
    input_class=PaperclipSearchInput,
    config_class=PaperclipSearchConfig,
    output_class=PaperclipSearchOutput,
    description="Search Paperclip full-text and abstract literature corpora with provenance",
    uses_gpu=False,
    example_input=_search_example,
    cacheable=True,
)
def run_paperclip_search(
    inputs: PaperclipSearchInput, config: PaperclipSearchConfig, instance: Any = None
) -> PaperclipSearchOutput:
    """Search configured Paperclip literature corpora and normalize the hits."""
    result = ToolInstance.dispatch(
        "paperclip",
        {
            "operation": "search",
            "query": inputs.query,
            "config": {
                **_shared_config(config),
                "sources": list(config.sources),
                "limit": config.limit,
                "exact": config.exact,
                "since": config.since,
            },
        },
        instance=instance,
        config=config,
    )
    hits = [_normalize_hit(item) for item in result["papers"]]
    return PaperclipSearchOutput(
        query=inputs.query,
        sources=list(config.sources),
        result_id=result.get("result_id"),
        hits=hits,
    )


def _normalize_hit(item: dict[str, Any]) -> PaperclipSearchHit:
    document_id = str(item.get("document_id") or item.get("pmc_id") or "")
    if not document_id:
        raise ValueError("Paperclip search result did not include a stable document_id")
    abstract = item.get("abstract")
    source = str(item.get("source") or item.get("corpus") or "unknown")
    access_kind = LiteratureAccessKind.ABSTRACT if abstract else LiteratureAccessKind.METADATA
    pmid = item.get("pmid") or (item.get("citation_canonical_work") or {}).get("pmid")
    provenance = SourceProvenance(
        provider="Paperclip",
        source_url=f"https://paperclip.gxl.ai/citations/papers/{document_id}",
        identifier=document_id,
        version=str(item.get("pub_date") or item.get("citation_year") or "") or None,
        access_level=AccessLevel.OPEN_ACCESS if source != "abstracts" else AccessLevel.METADATA_ONLY,
        media_type="application/json",
        request_parameters={"operation": "search", "source": source},
    )
    return PaperclipSearchHit(
        document_id=document_id,
        title=str(item.get("title") or item.get("citation_title") or document_id),
        authors=item.get("authors"),
        journal=item.get("journal_title"),
        publication_date=item.get("pub_date"),
        year=item.get("pub_year"),
        doi=item.get("doi") or item.get("citation_doi"),
        pmid=str(pmid) if pmid else None,
        pmcid=item.get("pmc_id"),
        source=source,
        abstract=abstract,
        matched_text=item.get("chunk_text"),
        relevance_score=item.get("rerank_score") if item.get("rerank_score") is not None else item.get("score"),
        access_kind=access_kind,
        provenance=provenance,
        raw_metadata=item,
    )


def _document_example() -> Any:
    return PaperclipDocumentInput(document_id="PMC11754230")


@tool(
    key="paperclip-literature-document",
    label="Paperclip Literature Document",
    category="database_retrieval",
    input_class=PaperclipDocumentInput,
    config_class=PaperclipDocumentConfig,
    output_class=PaperclipDocumentOutput,
    description="Retrieve Paperclip metadata and available line-addressable full text",
    uses_gpu=False,
    example_input=_document_example,
    cacheable=True,
)
def run_paperclip_document(
    inputs: PaperclipDocumentInput, config: PaperclipDocumentConfig, instance: Any = None
) -> PaperclipDocumentOutput:
    """Retrieve exact Paperclip metadata and available line-addressable text."""
    result = ToolInstance.dispatch(
        "paperclip",
        {
            "operation": "document",
            "document_id": inputs.document_id,
            "config": {
                **_shared_config(config),
                "include_full_text": config.include_full_text,
            },
        },
        instance=instance,
        config=config,
    )
    metadata_output = result["metadata_output"]
    metadata = json.loads(metadata_output)
    metadata_content = metadata_output.encode("utf-8")
    metadata_digest = hashlib.sha256(metadata_content).hexdigest()
    provenance_parameters = {"operation": "document", "include_full_text": config.include_full_text}
    metadata_provenance = SourceProvenance(
        provider="Paperclip",
        source_url=f"https://paperclip.gxl.ai/citations/papers/{inputs.document_id}",
        identifier=inputs.document_id,
        version=str(metadata.get("pub_date") or metadata.get("year") or "") or None,
        content_sha256=metadata_digest,
        media_type="application/json",
        license=metadata.get("license"),
        access_level=AccessLevel.METADATA_ONLY,
        request_parameters=provenance_parameters,
    )
    metadata_artifact = RetrievedArtifact(
        name=f"Paperclip {inputs.document_id} metadata",
        content_sha256=metadata_digest,
        media_type="application/json",
        size_bytes=len(metadata_content),
        provenance=metadata_provenance,
    )
    content: str | None = None
    content_preview: str | None = None
    expected_content_bytes: int | None = None
    access_kind = LiteratureAccessKind.METADATA
    if config.include_full_text:
        content_output = result.get("content_output")
        if result.get("content_exit_code") == 0 and content_output and content_output.strip():
            fields = (result.get("wc_output") or "").split()
            expected_content_bytes = int(fields[2]) if result.get("wc_exit_code") == 0 and len(fields) >= 3 else None
            observed = content_output.encode("utf-8")
            if expected_content_bytes == len(observed):
                content = content_output
                access_kind = LiteratureAccessKind.FULL_TEXT
            else:
                content_preview = content_output
                access_kind = LiteratureAccessKind.INACCESSIBLE
        else:
            access_kind = LiteratureAccessKind.INACCESSIBLE
    if content is None:
        provenance = metadata_provenance
        artifact = metadata_artifact
    else:
        content_bytes = content.encode("utf-8")
        content_digest = hashlib.sha256(content_bytes).hexdigest()
        provenance = SourceProvenance(
            provider="Paperclip",
            source_url=f"https://paperclip.gxl.ai/citations/papers/{inputs.document_id}",
            identifier=inputs.document_id,
            version=str(metadata.get("pub_date") or metadata.get("year") or "") or None,
            content_sha256=content_digest,
            media_type="text/plain",
            license=metadata.get("license"),
            access_level=AccessLevel.OPEN_ACCESS,
            request_parameters=provenance_parameters,
        )
        artifact = RetrievedArtifact(
            name=f"Paperclip {inputs.document_id} full text",
            content_sha256=content_digest,
            media_type="text/plain",
            size_bytes=len(content_bytes),
            provenance=provenance,
        )
    return PaperclipDocumentOutput(
        document=PaperclipDocument(
            document_id=inputs.document_id,
            metadata=metadata,
            content=content,
            content_preview=content_preview,
            expected_content_bytes=expected_content_bytes,
            access_kind=access_kind,
            provenance=provenance,
            metadata_artifact=metadata_artifact,
            artifact=artifact,
        )
    )
