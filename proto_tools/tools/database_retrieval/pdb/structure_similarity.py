"""Search experimental PDB structures with RCSB's learned structure embedding.

This is intentionally distinct from Foldseek.  The RCSB Search API embeds a
query structure and retrieves structurally related experimental PDB chains or
assemblies using its learned global representation.  Custom structures are
uploaded to RCSB and therefore require an explicit acknowledgement.
"""

from __future__ import annotations

import csv
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import requests
from pydantic import BaseModel, ConfigDict, Field, model_validator

from proto_tools.entities import Structure
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    build_http_session,
)

_UPLOAD_URL = "https://user-upload.rcsb.org/v1/putMultipart"
_UPLOAD_DOWNLOAD_BASE = "https://user-upload.rcsb.org/v1/download/"
_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
_DOWNLOAD_URL = "https://files.rcsb.org/download/{name}"
_USER_AGENT = "proto-tools/pdb-structure-similarity-v1"
_HTTP_RETRIES = 1
_BACKOFF_SECONDS = 1.0

PdbStructureTarget = Literal["polymer_entity_instance", "assembly"]
PdbStructureSimilarityType = Literal["global", "local"]


def _raise_for_status(response: Any, operation: str) -> None:
    """Raise a retryable HTTP error with a bounded RCSB response body."""
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        body = str(getattr(response, "text", ""))[:1000]
        raise requests.HTTPError(f"RCSB {operation} failed: {error}; body={body!r}", response=response) from error


class PdbStructureSimilarityInput(BaseToolInput):
    """One public PDB reference or one custom query structure.

    ``structure`` accepts the same ``Structure``/path/raw-content inputs as
    other Proto structure tools.  Supplying it sends coordinates and embedded
    sequences to the public RCSB upload service; callers must set
    ``allow_external_upload=True`` in the config.  Public ``pdb_id`` queries do
    not upload local data.

    Attributes:
        structure (Structure | None): Custom query structure, mutually exclusive with ``pdb_id``.
        pdb_id (str | None): Existing public PDB entry ID.
        chain_id (str | None): Input chain/asym ID for a public PDB query.
        assembly_id (str | None): Input biological assembly ID for a public PDB query.
    """

    structure: Structure | None = InputField(
        default=None,
        title="Custom Query Structure",
        description="Custom structure to upload and search; mutually exclusive with pdb_id",
    )
    pdb_id: str | None = InputField(
        default=None,
        title="PDB ID",
        description="Existing public PDB entry ID; mutually exclusive with structure",
    )
    chain_id: str | None = InputField(
        default=None,
        title="Chain ID",
        description="Input asym/chain ID for a public PDB query",
    )
    assembly_id: str | None = InputField(
        default=None,
        title="Assembly ID",
        description="Input biological assembly ID for a public PDB query",
    )

    @model_validator(mode="after")
    def _validate_source(self) -> PdbStructureSimilarityInput:
        if (self.structure is None) == (self.pdb_id is None):
            raise ValueError("Provide exactly one of structure or pdb_id")
        if self.chain_id is not None and self.assembly_id is not None:
            raise ValueError("Provide at most one of chain_id or assembly_id")
        if self.structure is not None and (self.chain_id is not None or self.assembly_id is not None):
            raise ValueError("chain_id and assembly_id apply only to public pdb_id queries")
        if self.pdb_id is not None and self.chain_id is None and self.assembly_id is None:
            raise ValueError("A public pdb_id query requires chain_id or assembly_id")
        return self


class PdbStructureSimilarityConfig(BaseConfig):
    """RCSB embedding-search settings.

    Attributes:
        target_search_space (PdbStructureTarget): Experimental PDB chains or assemblies to search.
        similarity_type (PdbStructureSimilarityType): Global whole-shape or local substructure similarity.
        predicted_tm_score_cutoff (float): Minimum learned predicted TM-like score.
        number_of_candidates (int): Maximum nearest-neighbor candidates considered.
        allow_external_upload (bool): Acknowledge disclosure of a custom structure.
        timeout_seconds (float): Per-request HTTP timeout.
        http_retries (int): Extra transport attempts per upload/search request.
    """

    target_search_space: PdbStructureTarget = ConfigField(
        default="polymer_entity_instance",
        title="Target Search Space",
        description="Search experimental PDB chains or biological assemblies",
    )
    similarity_type: PdbStructureSimilarityType = ConfigField(
        default="global",
        title="Similarity Type",
        description="GLOBAL length-normalized similarity or LOCAL substructure similarity",
    )
    predicted_tm_score_cutoff: float = ConfigField(
        default=0.0,
        ge=0.0,
        le=1.0,
        title="Predicted TM-score Cutoff",
        description="Minimum RCSB embedding-model predicted TM score to return",
    )
    number_of_candidates: int = ConfigField(
        default=100,
        ge=1,
        le=15000,
        title="Number of Candidates",
        description="Maximum nearest-neighbor candidates considered by RCSB",
    )
    allow_external_upload: bool = ConfigField(
        default=False,
        title="Allow External Upload",
        description="Acknowledge that a custom structure is sent to RCSB's public upload service",
        include_in_key=False,
    )
    timeout_seconds: float = ConfigField(
        default=120.0,
        ge=10.0,
        title="HTTP Timeout",
        description="Timeout for each RCSB upload or search request",
        include_in_key=False,
    )
    http_retries: int = ConfigField(
        default=_HTTP_RETRIES,
        ge=0,
        le=3,
        title="HTTP Retries",
        description="Extra bounded transport attempts per RCSB request",
        include_in_key=False,
    )


class PdbStructureSimilarityHit(BaseModel):
    """One ranked RCSB structure-embedding search hit."""

    model_config = ConfigDict(extra="forbid")

    identifier: str = Field(description="RCSB chain or assembly identifier")
    score: float | None = Field(default=None, description="RCSB normalized ranking score")
    predicted_tm_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Structure service original score, trained to approximate maximum TM-score",
    )


class PdbStructureSimilarityOutput(BaseToolOutput):
    """Ranked experimental-PDB matches for one structure query.

    Attributes:
        hits (list[PdbStructureSimilarityHit]): Ranked experimental-PDB structure matches.
        total_count (int): Number of matches reported by RCSB.
        max_predicted_tm_score (float | None): Largest learned predicted TM-like hit score.
        source_url (str): RCSB Search API endpoint used for the query.
        query_was_uploaded (bool): Whether the call sent a custom structure to RCSB.
    """

    hits: list[PdbStructureSimilarityHit] = Field(
        default_factory=list,
        title="Hits",
        description="Ranked experimental-PDB structure matches",
    )
    total_count: int = Field(
        default=0,
        ge=0,
        title="Total Count",
        description="Number of matches returned by RCSB",
    )
    max_predicted_tm_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        title="Maximum Predicted TM-score",
        description="Largest learned predicted TM-like score among returned hits",
    )
    source_url: str = Field(
        title="Source URL",
        description="RCSB Search API endpoint used",
    )
    query_was_uploaded: bool = Field(
        default=False,
        title="Query Was Uploaded",
        description="Whether this call sent a custom structure to RCSB",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return supported export formats."""
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        """Return the default export format."""
        return "json"

    def _export_output(self, export_path: Any, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")
        if file_format == "json":
            with path.open("w", encoding="utf-8") as handle:
                json.dump(self.model_dump(mode="json"), handle, indent=2)
            return
        if file_format == "csv":
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "identifier",
                        "score",
                        "predicted_tm_score",
                    ],
                )
                writer.writeheader()
                writer.writerows(hit.model_dump() for hit in self.hits)
            return
        raise ValueError(f"Unsupported format: {file_format}")


class PdbStructureSimilarityBatchInput(BaseToolInput):
    """Aligned public or custom structure queries for bounded parallel search.

    Attributes:
        queries (list[PdbStructureSimilarityInput]): Public PDB references or
            custom structures to search.
        query_ids (list[str] | None): Optional unique identifiers aligned
            one-to-one with ``queries``.
    """

    queries: list[PdbStructureSimilarityInput] = InputField(
        min_length=1,
        title="Structure Queries",
        description="Public PDB references or custom structures to search",
    )
    query_ids: list[str] | None = InputField(
        default=None,
        title="Query Identifiers",
        description="Optional unique identifiers aligned one-to-one with queries",
    )

    @model_validator(mode="after")
    def _validate_alignment(self) -> PdbStructureSimilarityBatchInput:
        if self.query_ids is not None:
            if len(self.query_ids) != len(self.queries):
                raise ValueError("query_ids must align one-to-one with queries")
            if len(set(self.query_ids)) != len(self.query_ids):
                raise ValueError("query_ids must be unique")
        return self


class PdbStructureSimilarityBatchConfig(PdbStructureSimilarityConfig):
    """Similarity settings plus conservative RCSB request concurrency.

    Attributes:
        target_search_space (PdbStructureTarget): Experimental PDB chains or assemblies to search.
        similarity_type (PdbStructureSimilarityType): Global whole-shape or local substructure similarity.
        predicted_tm_score_cutoff (float): Minimum learned predicted TM-like score.
        number_of_candidates (int): Maximum nearest-neighbor candidates considered.
        allow_external_upload (bool): Acknowledge disclosure of a custom structure.
        timeout_seconds (float): Per-request HTTP timeout.
        http_retries (int): Extra bounded transport attempts per RCSB request.
        max_workers (int): Concurrent query workers, kept low for RCSB rate limits.
        fail_fast (bool): Raise on the first failed query instead of reporting it per query.
    """

    max_workers: int = ConfigField(
        default=2,
        ge=1,
        le=4,
        title="Maximum Workers",
        description="Concurrent query workers; keep low to respect RCSB rate limits",
        include_in_key=False,
    )
    fail_fast: bool = ConfigField(
        default=False,
        title="Fail Fast",
        description="Raise on the first failed query instead of returning an explicit per-query error",
    )


class PdbStructureSimilarityBatchResult(BaseModel):
    """One aligned batch result with explicit failure evidence."""

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(
        title="Query ID",
        description="Identifier of the query structure this result belongs to",
    )
    input_index: int = Field(
        ge=0,
        title="Input Index",
        description="Zero-based position of the query in the submitted batch",
    )
    output: PdbStructureSimilarityOutput | None = Field(
        default=None,
        title="Search Output",
        description="Structure-search result, or None when this query failed",
    )
    error_type: str | None = Field(
        default=None,
        title="Error Type",
        description="Exception class name when this query failed, otherwise None",
    )
    error_message: str | None = Field(
        default=None,
        title="Error Message",
        description="Failure detail when this query failed, otherwise None",
    )

    @model_validator(mode="after")
    def _validate_outcome(self) -> PdbStructureSimilarityBatchResult:
        if (self.output is None) == (self.error_type is None):
            raise ValueError("Provide exactly one of output or error_type")
        return self


class PdbStructureSimilarityBatchOutput(BaseToolOutput):
    """Input-aligned structure-search results and explicit failures.

    Attributes:
        results (list[PdbStructureSimilarityBatchResult]): One result per
            submitted query, in input order, each a success or a failure.
        successful_count (int): Number of queries that returned a
            structure-search result.
        failed_count (int): Number of queries that failed and carry no
            measurement.
        source_url (str): RCSB Search API endpoint used for every query.
    """

    results: list[PdbStructureSimilarityBatchResult] = Field(
        default_factory=list,
        title="Results",
        description="One result per submitted query, in input order, each a success or a failure",
    )
    successful_count: int = Field(
        default=0,
        ge=0,
        title="Successful Count",
        description="Number of queries that returned a structure-search result",
    )
    failed_count: int = Field(
        default=0,
        ge=0,
        title="Failed Count",
        description="Number of queries that failed and carry no measurement",
    )
    source_url: str = Field(
        title="Source URL",
        description="RCSB Search API endpoint used for every query",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return supported export formats."""
        return ["json"]

    @property
    def output_format_default(self) -> str:
        """Return the default export format."""
        return "json"

    def _export_output(self, export_path: Any, file_format: str) -> None:
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        path = Path(export_path).with_suffix(".json")
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")


def _original_structure_score(result: dict[str, Any]) -> float | None:
    """Extract the structure node's unnormalized/original score."""
    direct = result.get("original_score")
    if isinstance(direct, int | float):
        return float(direct)
    for service in result.get("services", []):
        if not isinstance(service, dict) or service.get("service_type") != "structure":
            continue
        for node in service.get("nodes", []):
            if isinstance(node, dict) and isinstance(node.get("original_score"), int | float):
                return float(node["original_score"])
    return None


def _upload_structure(session: Any, structure: Structure, timeout_seconds: float) -> str:
    with TemporaryDirectory(prefix="proto-rcsb-structure-") as temp_dir:
        path = Path(temp_dir) / "query.pdb"
        structure.write_pdb(path)
        with path.open("rb") as handle:
            response = session.post(
                _UPLOAD_URL,
                files={"file": (path.name, handle, "chemical/x-pdb")},
                data={"format": "pdb"},
                timeout=timeout_seconds,
            )
        _raise_for_status(response, "structure upload")
        payload = response.json()
    key = payload.get("key")
    if not isinstance(key, str) or not key:
        raise ValueError("RCSB upload response did not contain a structure key")
    return _UPLOAD_DOWNLOAD_BASE + key


def _query_parameters(
    inputs: PdbStructureSimilarityInput,
    config: PdbStructureSimilarityConfig,
    uploaded_url: str | None,
) -> dict[str, Any]:
    if uploaded_url is not None:
        value: dict[str, Any] = {"url": uploaded_url, "format": "bcif"}
    else:
        value = {"entry_id": inputs.pdb_id.upper() if inputs.pdb_id else None}
        if inputs.chain_id is not None:
            value["asym_id"] = inputs.chain_id
        else:
            value["assembly_id"] = inputs.assembly_id
    return {
        "value": value,
        "target_search_space": config.target_search_space,
        "similarity_type": config.similarity_type,
        "number_of_candidates": config.number_of_candidates,
        "ptmscore_cutoff": config.predicted_tm_score_cutoff,
    }


def _search_payload(
    parameters: dict[str, Any], config: PdbStructureSimilarityConfig, *, custom_query: bool
) -> dict[str, Any]:
    del custom_query
    # RCSB supports chain- and assembly-level return identifiers for both
    # deposited and URL-backed custom structure queries. Returning only entry
    # IDs for custom inputs discards the matched chain/assembly identity, which
    # callers need to resolve a hit back to a specific chain.
    return_type = "polymer_instance" if config.target_search_space == "polymer_entity_instance" else "assembly"
    return {
        "query": {"type": "terminal", "service": "structure", "parameters": parameters, "node_id": 0},
        "return_type": return_type,
        "request_info": {"query_id": uuid.uuid4().hex, "src": "ui"},
        "request_options": {
            "paginate": {"start": 0, "rows": config.number_of_candidates},
            "results_content_type": ["experimental"],
            "results_verbosity": "verbose",
        },
    }


def example_input() -> Any:
    """Minimal public-data example that does not upload local content."""
    return PdbStructureSimilarityInput(pdb_id="1YCR", chain_id="A")


def batch_example_input() -> Any:
    """Two aligned public-data queries that require no external upload."""
    return PdbStructureSimilarityBatchInput(
        query_ids=["p53", "mdm2"],
        queries=[
            PdbStructureSimilarityInput(pdb_id="1YCR", chain_id="B"),
            PdbStructureSimilarityInput(pdb_id="1YCR", chain_id="A"),
        ],
    )


@tool(
    key="pdb-structure-similarity",
    label="PDB Structure Similarity",
    category="database_retrieval",
    input_class=PdbStructureSimilarityInput,
    config_class=PdbStructureSimilarityConfig,
    output_class=PdbStructureSimilarityOutput,
    description=(
        "Compare protein structures against experimental PDB for prediction-independent structural novelty "
        "evidence using RCSB's learned structure embedding"
    ),
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_pdb_structure_similarity(
    inputs: PdbStructureSimilarityInput,
    config: PdbStructureSimilarityConfig,
    instance: Any = None,
) -> PdbStructureSimilarityOutput:
    """Search RCSB's experimental structure index.

    The returned ``predicted_tm_score`` is an independent learned similarity
    estimate, not a Foldseek score or proof of novelty.  Calibrate it against
    matched controls before using it as a hard decision boundary.
    """
    del instance
    if inputs.structure is not None and not config.allow_external_upload:
        raise ValueError(
            "Custom structure search uploads coordinates and embedded sequences to RCSB; "
            "set allow_external_upload=True only when that disclosure is permitted"
        )
    session = build_http_session(
        http_retries=config.http_retries,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
        # Both upload and search are POST requests. The shared session defaults
        # to retrying GET only, which previously made the configured retry count
        # ineffective for every operation in this tool.
        allowed_methods=["POST"],
        mount_http=True,
    )
    uploaded_url: str | None = None
    try:
        if inputs.structure is not None:
            uploaded_url = _upload_structure(session, inputs.structure, config.timeout_seconds)
        response = session.post(
            _SEARCH_URL,
            json=_search_payload(
                _query_parameters(inputs, config, uploaded_url),
                config,
                custom_query=uploaded_url is not None,
            ),
            timeout=config.timeout_seconds,
        )
        _raise_for_status(response, "structure search")
        payload = response.json()
    except requests.RequestException as exc:
        # The registry normally retries ConnectionError around the entire tool
        # call.  This tool already has per-request retries in its HTTP adapter;
        # converting the exhausted transport error prevents multiplicative
        # upload+search retries that can turn a bounded shortlist into hours of
        # repeated external calls.
        raise RuntimeError(
            "RCSB structure similarity exhausted its bounded HTTP retry budget "
            f"(timeout_seconds={config.timeout_seconds}, http_retries={config.http_retries}): {exc}"
        ) from exc
    finally:
        session.close()

    hits = [
        PdbStructureSimilarityHit(
            identifier=str(item["identifier"]),
            score=float(item["score"]) if isinstance(item.get("score"), int | float) else None,
            predicted_tm_score=_original_structure_score(item),
        )
        for item in payload.get("result_set", [])
        if isinstance(item, dict) and item.get("identifier") is not None
    ]
    measured = [hit.predicted_tm_score for hit in hits if hit.predicted_tm_score is not None]
    return PdbStructureSimilarityOutput(
        hits=hits,
        total_count=int(payload.get("total_count", len(hits))),
        max_predicted_tm_score=max(measured) if measured else None,
        source_url=_SEARCH_URL,
        query_was_uploaded=uploaded_url is not None,
    )


@tool(
    key="pdb-structure-similarity-batch",
    label="Batch PDB Structure Similarity",
    category="database_retrieval",
    input_class=PdbStructureSimilarityBatchInput,
    config_class=PdbStructureSimilarityBatchConfig,
    output_class=PdbStructureSimilarityBatchOutput,
    description=(
        "Batch portfolio comparison of protein structures against experimental PDB for prediction-independent "
        "structural novelty evidence, with bounded RCSB requests and explicit per-query failures"
    ),
    uses_gpu=False,
    example_input=batch_example_input,
    cacheable=True,
)
def run_pdb_structure_similarity_batch(
    inputs: PdbStructureSimilarityBatchInput,
    config: PdbStructureSimilarityBatchConfig,
    instance: Any = None,
) -> PdbStructureSimilarityBatchOutput:
    """Search a shortlist concurrently while preserving input alignment."""
    del instance
    single_config = PdbStructureSimilarityConfig.model_validate(config.model_dump(exclude={"max_workers", "fail_fast"}))
    query_ids = inputs.query_ids or [str(index) for index in range(len(inputs.queries))]

    def search(index: int) -> PdbStructureSimilarityBatchResult:
        try:
            output = run_pdb_structure_similarity(inputs.queries[index], single_config)
            if not isinstance(output, PdbStructureSimilarityOutput):
                raise TypeError("pdb-structure-similarity returned an unexpected output type")
        except Exception as error:
            if config.fail_fast:
                raise
            return PdbStructureSimilarityBatchResult(
                query_id=query_ids[index],
                input_index=index,
                error_type=type(error).__name__,
                error_message=str(error)[:1000],
            )
        return PdbStructureSimilarityBatchResult(
            query_id=query_ids[index],
            input_index=index,
            output=output,
        )

    with ThreadPoolExecutor(max_workers=min(config.max_workers, len(inputs.queries))) as executor:
        results = list(executor.map(search, range(len(inputs.queries))))
    successful_count = sum(result.output is not None for result in results)
    return PdbStructureSimilarityBatchOutput(
        results=results,
        successful_count=successful_count,
        failed_count=len(results) - successful_count,
        source_url=_SEARCH_URL,
        metadata={
            "query_count": len(results),
            "max_workers": config.max_workers,
            "input_order_preserved": True,
            "failed_queries_are_not_measurements": True,
        },
    )
