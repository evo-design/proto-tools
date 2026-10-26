"""Fetch standards-native design documents from a SynBioHub registry."""

import json
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

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
    request_with_retry,
)

SynBioHubFormat = Literal["sbol", "fasta", "genbank"]
_FORMAT_SUFFIX = {"sbol": "sbol", "fasta": "fasta", "genbank": "gb"}
_REQUEST_TIMEOUT_SECONDS = 30
_HTTP_RETRIES = 2
_BACKOFF_SECONDS = 1.0
_USER_AGENT = "proto-tools/synbiohub-fetch-v1"


class SynBioHubFetchInput(BaseToolInput):
    """Input identifying one SynBioHub top-level object.

    Attributes:
        object_uri (str): Absolute URI of a top-level object on the selected registry.
    """

    object_uri: str = InputField(
        title="Object URI",
        description="Absolute URI of a top-level object on the selected SynBioHub registry",
    )

    @field_validator("object_uri")
    @classmethod
    def _validate_object_uri(cls, value: str) -> str:
        uri = value.rstrip("/")
        parsed = urlsplit(uri)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("object_uri must be an absolute HTTP or HTTPS URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("object_uri must not contain credentials, query parameters, or fragments")
        return uri


class SynBioHubFetchConfig(BaseConfig):
    """Configuration for SynBioHub document retrieval.

    Attributes:
        registry_url (str): Registry origin allowed for object retrieval.
        formats (list[SynBioHubFormat]): Representations to retrieve.
        recursive_sbol (bool): Include child objects in SBOL output.
        synbiohub_token (str | None): Optional private-object access token.
    """

    registry_url: str = ConfigField(
        default="https://api.synbiohub.org",
        title="Registry URL",
        description="SynBioHub registry origin allowed for object retrieval",
    )
    formats: list[SynBioHubFormat] = ConfigField(
        default_factory=lambda: ["sbol", "fasta"],
        min_length=1,
        title="Formats",
        description="Standards-native representations to retrieve",
    )
    recursive_sbol: bool = ConfigField(
        default=True,
        title="Recursive SBOL",
        description="Include referenced child objects in the SBOL document",
    )
    synbiohub_token: str | None = ConfigField(
        title="SynBioHub Token",
        description="Optional private-object token.",
        credential="SYNBIOHUB_TOKEN",
    )

    @field_validator("registry_url")
    @classmethod
    def _validate_registry_url(cls, value: str) -> str:
        url = value.rstrip("/")
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"}:
            raise ValueError("registry_url must be an HTTP or HTTPS origin without a path")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("registry_url must not contain credentials, query parameters, or fragments")
        return url

    @field_validator("formats")
    @classmethod
    def _deduplicate_formats(cls, value: list[SynBioHubFormat]) -> list[SynBioHubFormat]:
        return list(dict.fromkeys(value))


class SynBioHubDocument(BaseModel):
    """One representation downloaded from SynBioHub.

    Attributes:
        format (SynBioHubFormat): Representation format.
        content (str): Decoded document content.
        source_url (str): Stable download endpoint.
    """

    model_config = ConfigDict(str_strip_whitespace=False)

    format: SynBioHubFormat = Field(title="Format", description="Downloaded representation format")
    content: str = Field(repr=False, title="Content", description="Decoded document content")
    source_url: str = Field(title="Source URL", description="Stable SynBioHub download endpoint")


class SynBioHubFetchOutput(BaseToolOutput):
    """Standards-native representations of one SynBioHub object.

    Attributes:
        object_uri (str): Canonical top-level object URI.
        registry_url (str): Registry origin used for retrieval.
        documents (list[SynBioHubDocument]): Downloaded SBOL and sequence documents.
    """

    object_uri: str = Field(title="Object URI", description="Canonical top-level object URI")
    registry_url: str = Field(title="Registry URL", description="Registry origin used for retrieval")
    documents: list[SynBioHubDocument] = Field(
        default_factory=list,
        title="Documents",
        description="Downloaded SBOL and sequence representations",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return supported aggregate output formats."""
        return ["json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Any, file_format: str) -> None:
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        Path(export_path).with_suffix(".json").write_text(
            json.dumps(self.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )


def example_input() -> Any:
    """Return minimal valid example input."""
    return SynBioHubFetchInput(object_uri="https://api.synbiohub.org/public/bsu/BO_9757/1")


@tool(
    key="synbiohub-fetch",
    local_only="synbiohub-fetch is an HTTP-only CPU tool and needs no isolated environment",
    label="SynBioHub Fetch",
    category="database_retrieval",
    input_class=SynBioHubFetchInput,
    config_class=SynBioHubFetchConfig,
    output_class=SynBioHubFetchOutput,
    description="Fetch SBOL, FASTA, and GenBank representations from a SynBioHub registry",
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_synbiohub_fetch(
    inputs: SynBioHubFetchInput,
    config: SynBioHubFetchConfig,
    instance: Any = None,
) -> SynBioHubFetchOutput:
    """Fetch the requested representations of one SynBioHub object.

    Args:
        inputs (SynBioHubFetchInput): Top-level object URI.
        config (SynBioHubFetchConfig): Registry, formats, recursion, and token settings.
        instance (Any): Optional ToolInstance; unused for this HTTP-only tool.

    Returns:
        SynBioHubFetchOutput: Downloaded standards-native documents.
    """
    del instance
    _validate_same_origin(inputs.object_uri, config.registry_url)
    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
    )
    headers = {"Accept": "text/plain"}
    if config.synbiohub_token:
        headers["X-authorization"] = config.synbiohub_token
    try:
        documents = [
            _fetch_document(
                inputs.object_uri,
                document_format,
                recursive_sbol=config.recursive_sbol,
                headers=headers,
                session=session,
            )
            for document_format in config.formats
        ]
        return SynBioHubFetchOutput(
            object_uri=inputs.object_uri,
            registry_url=config.registry_url,
            documents=documents,
        )
    finally:
        session.close()


def _validate_same_origin(object_uri: str, registry_url: str) -> None:
    """Reject cross-origin object URLs to bound caller-selected network access."""
    object_origin = urlsplit(object_uri)
    registry_origin = urlsplit(registry_url)
    if (object_origin.scheme, object_origin.netloc) != (registry_origin.scheme, registry_origin.netloc):
        raise ValueError("object_uri must use the same scheme and host as registry_url")


def _fetch_document(
    object_uri: str,
    document_format: SynBioHubFormat,
    *,
    recursive_sbol: bool,
    headers: dict[str, str],
    session: requests.Session,
) -> SynBioHubDocument:
    """Fetch one SynBioHub representation."""
    suffix = "sbolnr" if document_format == "sbol" and not recursive_sbol else _FORMAT_SUFFIX[document_format]
    source_url = f"{object_uri}/{suffix}"
    response = request_with_retry(
        lambda: session.get(
            source_url,
            headers=headers,
            timeout=_REQUEST_TIMEOUT_SECONDS,
            allow_redirects=False,
        ),
        retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
    )
    if 300 <= response.status_code < 400:
        raise ValueError(f"SynBioHub redirected {source_url}; verify the registry, object URI, and access token")
    response.raise_for_status()
    return SynBioHubDocument(
        format=document_format,
        content=response.content.decode(response.encoding or "utf-8"),
        source_url=source_url,
    )
