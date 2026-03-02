"""NCBI Entrez esearch tool — search databases by query term.

Wraps the NCBI E-utilities esearch endpoint for finding IDs across
protein, nuccore, and gene databases.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import Field

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils.http_session import build_http_session
from bio_programming_tools.utils.tool_io import BaseToolInput, BaseToolOutput

from .shared_data_models import NCBIFetchConfig, _ncbi_esearch

# ============================================================================
# Data Models
# ============================================================================


class NCBIEsearchInput(BaseToolInput):
    """Input for NCBI esearch.

    Attributes:
        db: NCBI database to query: 'protein', 'nuccore' (nucleotide core),
            or 'gene'.
        search_term: NCBI search query (e.g. 'lacI[Gene] AND Escherichia
            coli[Organism]').
        max_results: Maximum number of IDs to return from the search.
    """

    db: Literal["protein", "nuccore", "gene"] = Field(
        description="NCBI database to query: 'protein', 'nuccore' (nucleotide core), or 'gene'"
    )
    search_term: str = Field(
        description="NCBI search query (e.g. 'lacI[Gene] AND Escherichia coli[Organism]')",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of IDs to return from a search",
    )


class NCBIEsearchOutput(BaseToolOutput):
    """Output from NCBI esearch.

    Attributes:
        ids: List of NCBI IDs matching the search query.
    """

    ids: List[str] = Field(
        default_factory=list, description="List of NCBI IDs found by the search"
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


NCBIEsearchConfig = NCBIFetchConfig


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input():
    """Minimal valid input for testing and examples."""
    return NCBIEsearchInput(db="protein", search_term="insulin")


@tool(
    key="ncbi-esearch",
    label="NCBI Entrez ESearch",
    category="database_retrieval",
    input_class=NCBIEsearchInput,
    config_class=NCBIFetchConfig,
    output_class=NCBIEsearchOutput,
    description="Search NCBI Entrez databases by query term to find matching IDs",
    uses_gpu=False,
    example_input=example_input,
)
def run_ncbi_esearch(
    inputs: NCBIEsearchInput,
    config: NCBIFetchConfig | None = None,
    instance=None,
) -> NCBIEsearchOutput:
    """Search NCBI Entrez databases by query term.

    Args:
        inputs: Search parameters including database, query term, and
            max results.
        config: HTTP timeout, retry, and authentication settings.

    Returns:
        NCBIEsearchOutput containing matching NCBI IDs.
    """
    del instance

    session = build_http_session(
        http_retries=config.http_retries,
        backoff_seconds=config.backoff_seconds,
        user_agent=config.user_agent,
        allowed_methods=["GET", "POST"],
    )

    try:
        ids = _ncbi_esearch(
            db=inputs.db,
            term=inputs.search_term,
            max_results=inputs.max_results,
            config=config,
            session=session,
        )
        return NCBIEsearchOutput(ids=ids)
    finally:
        session.close()
