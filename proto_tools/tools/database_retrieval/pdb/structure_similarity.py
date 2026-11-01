"""Search experimental PDB structures with RCSB's learned structure embedding.

This is intentionally distinct from Foldseek.  The RCSB Search API embeds a
query structure and retrieves structurally related experimental PDB chains or
assemblies using its learned global representation.  Custom structures are
uploaded to RCSB's public upload service before the search.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import requests
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
_USER_AGENT = "proto-tools/pdb-structure-similarity-v1"
_REQUEST_TIMEOUT_SECONDS = 120
_HTTP_RETRIES = 3
_BACKOFF_SECONDS = 1.0

PdbStructureTarget = Literal["polymer_entity_instance", "assembly"]
PdbStructureSimilarityType = Literal["global", "local"]


class PdbStructureSimilarityQuery(BaseModel):
    """One public PDB reference or one custom query structure.

    ``structure`` accepts a ``Structure``, a file path, or raw PDB/CIF content.
    Supplying it sends coordinates and embedded sequences to the public RCSB
    upload service.  Public ``pdb_id`` queries do not upload local data.

    Attributes:
        structure (Structure | None): Custom query structure, mutually exclusive with ``pdb_id``.
        pdb_id (str | None): Existing public PDB entry ID.
        chain_id (str | None): Input chain/asym ID for a public PDB query.
        assembly_id (str | None): Input biological assembly ID for a public PDB query.
    """

    model_config = ConfigDict(extra="forbid")

    structure: Structure | None = Field(
        default=None,
        title="Custom Query Structure",
        description="Custom structure to upload and search; mutually exclusive with pdb_id",
    )
    pdb_id: str | None = Field(
        default=None,
        title="PDB ID",
        description="Existing public PDB entry ID; mutually exclusive with structure",
    )
    chain_id: str | None = Field(
        default=None,
        title="Chain ID",
        description="Input asym/chain ID for a public PDB query",
    )
    assembly_id: str | None = Field(
        default=None,
        title="Assembly ID",
        description="Input biological assembly ID for a public PDB query",
    )

    @model_validator(mode="after")
    def _validate_source(self) -> PdbStructureSimilarityQuery:
        if (self.structure is None) == (self.pdb_id is None):
            raise ValueError("Provide exactly one of structure or pdb_id")
        if self.chain_id is not None and self.assembly_id is not None:
            raise ValueError("Provide at most one of chain_id or assembly_id")
        if self.structure is not None and (self.chain_id is not None or self.assembly_id is not None):
            raise ValueError("chain_id and assembly_id apply only to public pdb_id queries")
        if self.pdb_id is not None and self.chain_id is None and self.assembly_id is None:
            raise ValueError("A public pdb_id query requires chain_id or assembly_id")
        return self


class PdbStructureSimilarityInput(BaseToolInput):
    """Structure queries to search against experimental PDB.

    Attributes:
        queries (list[PdbStructureSimilarityQuery]): Public PDB references or
            custom structures. A single query is normalized to a list.
    """

    queries: list[PdbStructureSimilarityQuery] = InputField(
        min_length=1,
        title="Structure Queries",
        description="Public PDB references or custom structures to search",
    )

    @field_validator("queries", mode="before")
    @classmethod
    def _normalize_queries(cls, value: Any) -> Any:
        """Accept a single query object or a list."""
        if isinstance(value, dict | PdbStructureSimilarityQuery):
            return [value]
        return value


class PdbStructureSimilarityConfig(BaseConfig):
    """RCSB embedding-search settings.

    Attributes:
        target_search_space (PdbStructureTarget): Experimental PDB chains or assemblies to search.
        similarity_type (PdbStructureSimilarityType): Global whole-shape or local substructure similarity.
        predicted_tm_score_cutoff (float): Minimum learned predicted TM-like score.
        number_of_candidates (int): Maximum nearest-neighbor candidates considered.
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


class PdbStructureSimilarityHit(BaseModel):
    """One ranked RCSB structure-embedding search hit."""

    model_config = ConfigDict(extra="forbid")

    identifier: str = Field(title="Identifier", description="RCSB chain or assembly identifier")
    score: float | None = Field(default=None, title="Score", description="RCSB normalized ranking score")
    predicted_tm_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        title="Predicted TM-score",
        description="Structure service original score, trained to approximate maximum TM-score",
    )


class PdbStructureSimilarityResult(BaseModel):
    """Ranked experimental-PDB matches for one query."""

    model_config = ConfigDict(extra="forbid")

    hits: list[PdbStructureSimilarityHit] = Field(
        default_factory=list,
        title="Hits",
        description="Ranked experimental-PDB structure matches",
    )
    total_count: int = Field(
        default=0,
        ge=0,
        title="Total Count",
        description="Number of matches reported by RCSB, including any beyond the returned hits",
    )


class PdbStructureSimilarityOutput(BaseToolOutput):
    """Structure-similarity results, one per query in input order.

    Attributes:
        results (list[PdbStructureSimilarityResult]): One result per query, in input order.
        source_url (str): RCSB Search API endpoint used for every query.
    """

    results: list[PdbStructureSimilarityResult] = Field(
        default_factory=list,
        title="Results",
        description="One result per query, in input order",
    )
    source_url: str = Field(
        title="Source URL",
        description="RCSB Search API endpoint used",
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
                writer = csv.DictWriter(handle, fieldnames=["query_index", "identifier", "score", "predicted_tm_score"])
                writer.writeheader()
                for index, result in enumerate(self.results):
                    writer.writerows({"query_index": index, **hit.model_dump()} for hit in result.hits)
            return
        raise ValueError(f"Unsupported format: {file_format}")


def _raise_for_status(response: Any, operation: str) -> None:
    """Raise an HTTP error that includes a bounded RCSB response body."""
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        body = str(getattr(response, "text", ""))[:1000]
        raise requests.HTTPError(f"RCSB {operation} failed: {error}; body={body!r}", response=response) from error


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


def _upload_structure(session: requests.Session, structure: Structure) -> str:
    """Upload a custom structure to RCSB and return the URL the search reads it from."""
    with TemporaryDirectory(prefix="proto-rcsb-structure-") as temp_dir:
        path = Path(temp_dir) / "query.pdb"
        structure.write_pdb(path)
        with path.open("rb") as handle:
            response = session.post(
                _UPLOAD_URL,
                files={"file": (path.name, handle, "chemical/x-pdb")},
                data={"format": "pdb"},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        _raise_for_status(response, "structure upload")
        payload = response.json()
    key = payload.get("key")
    if not isinstance(key, str) or not key:
        raise ValueError("RCSB upload response did not contain a structure key")
    return _UPLOAD_DOWNLOAD_BASE + key


def _search_payload(
    query: PdbStructureSimilarityQuery,
    config: PdbStructureSimilarityConfig,
    uploaded_url: str | None,
) -> dict[str, Any]:
    """Build the RCSB Search API request for one query."""
    if query.pdb_id is not None:
        value: dict[str, Any] = {"entry_id": query.pdb_id.upper()}
        if query.chain_id is not None:
            value["asym_id"] = query.chain_id
        else:
            value["assembly_id"] = query.assembly_id
    else:
        value = {"url": uploaded_url, "format": "bcif"}
    # Chain- and assembly-level return types keep each hit resolvable to a
    # specific chain or assembly, for public and uploaded queries alike.
    return_type = "polymer_instance" if config.target_search_space == "polymer_entity_instance" else "assembly"
    return {
        "query": {
            "type": "terminal",
            "service": "structure",
            "parameters": {
                "value": value,
                "target_search_space": config.target_search_space,
                "similarity_type": config.similarity_type,
                "number_of_candidates": config.number_of_candidates,
                "ptmscore_cutoff": config.predicted_tm_score_cutoff,
            },
        },
        "return_type": return_type,
        "request_options": {
            "paginate": {"start": 0, "rows": config.number_of_candidates},
            "results_content_type": ["experimental"],
            "results_verbosity": "verbose",
        },
    }


def _search(
    session: requests.Session,
    query: PdbStructureSimilarityQuery,
    config: PdbStructureSimilarityConfig,
) -> PdbStructureSimilarityResult:
    """Run one structure-similarity search, uploading the query first if it is custom."""
    uploaded_url = _upload_structure(session, query.structure) if query.structure is not None else None
    response = session.post(
        _SEARCH_URL,
        json=_search_payload(query, config, uploaded_url),
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    _raise_for_status(response, "structure search")
    payload = response.json()
    hits = [
        PdbStructureSimilarityHit(
            identifier=str(item["identifier"]),
            score=float(item["score"]) if isinstance(item.get("score"), int | float) else None,
            predicted_tm_score=_original_structure_score(item),
        )
        for item in payload.get("result_set", [])
        if isinstance(item, dict) and item.get("identifier") is not None
    ]
    return PdbStructureSimilarityResult(hits=hits, total_count=int(payload.get("total_count", len(hits))))


def example_input() -> Any:
    """Two public-data queries that upload no local content."""
    return PdbStructureSimilarityInput(
        queries=[
            PdbStructureSimilarityQuery(pdb_id="1YCR", chain_id="A"),
            PdbStructureSimilarityQuery(pdb_id="1YCR", chain_id="B"),
        ]
    )


@tool(
    key="pdb-structure-similarity",
    label="PDB Structure Similarity",
    category="database_retrieval",
    input_class=PdbStructureSimilarityInput,
    config_class=PdbStructureSimilarityConfig,
    output_class=PdbStructureSimilarityOutput,
    description="Search experimental PDB chains or assemblies for structures similar to a query, using RCSB's learned structure embedding",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_fields=["queries"],
    iterable_output_field="results",
    cacheable=True,
)
def run_pdb_structure_similarity(
    inputs: PdbStructureSimilarityInput,
    config: PdbStructureSimilarityConfig,
    instance: Any = None,
) -> PdbStructureSimilarityOutput:
    """Search RCSB's experimental structure index, one search per query.

    The returned ``predicted_tm_score`` is a learned similarity estimate, not a
    Foldseek score or an exact structural alignment.

    Args:
        inputs (PdbStructureSimilarityInput): Public PDB references or custom structures.
        config (PdbStructureSimilarityConfig): Search space, similarity type, and result limits.
        instance (Any): Unused; kept for tool API consistency.

    Returns:
        PdbStructureSimilarityOutput: One result per query, in input order.
    """
    del instance
    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
        # Upload and search are both POSTs; the shared session retries GET only by default.
        allowed_methods=["POST"],
    )
    try:
        results = [_search(session, query, config) for query in inputs.queries]
    finally:
        session.close()
    return PdbStructureSimilarityOutput(results=results, source_url=_SEARCH_URL)
