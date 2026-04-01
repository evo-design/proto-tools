"""proto_tools/tools/database_retrieval/ncbi/esearch.py.

Wraps the NCBI E-utilities esearch endpoint for finding IDs across
protein, nuccore, and gene databases.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from proto_tools.tools.database_retrieval.ncbi.shared_data_models import NCBIFetchConfig, _ncbi_esearch
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseToolInput,
    BaseToolOutput,
    InputField,
    build_http_session,
)

# ============================================================================
# Data Models
# ============================================================================


class NCBIEsearchInput(BaseToolInput):
    """Input for NCBI esearch.

    Attributes:
        db (Literal['protein', 'nuccore', 'gene']): NCBI database to query: 'protein', 'nuccore' (nucleotide core),
            or 'gene'.
        search_term (str): NCBI search query (e.g. 'lacI[Gene] AND Escherichia
            coli[Organism]').
        max_results (int): Maximum number of IDs to return from the search.
    """

    db: Literal["protein", "nuccore", "gene"] = InputField(
        description="NCBI database to query: 'protein', 'nuccore' (nucleotide core), or 'gene'"
    )
    search_term: str = InputField(
        description="NCBI search query (e.g. 'lacI[Gene] AND Escherichia coli[Organism]')",
    )
    max_results: int = InputField(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of IDs to return from a search",
    )


class NCBIEsearchOutput(BaseToolOutput):
    """Output from NCBI esearch.

    Attributes:
        ids (list[str]): List of NCBI IDs matching the search query.
    """

    ids: list[str] = Field(default_factory=list, description="List of NCBI IDs found by the search")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Any, file_format: str) -> None:
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


def example_input() -> Any:
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
    instance: Any = None,
) -> NCBIEsearchOutput:
    """Search NCBI Entrez databases by query term.

    Args:
        inputs (NCBIEsearchInput): Search parameters including database, query term, and
            max results.
        config (NCBIFetchConfig | None): HTTP timeout, retry, and authentication settings.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        NCBIEsearchOutput: NCBIEsearchOutput containing matching NCBI IDs.
    """
    del instance

    session = build_http_session(
        http_retries=config.http_retries,  # type: ignore[union-attr]
        backoff_seconds=config.backoff_seconds,  # type: ignore[union-attr]
        user_agent=config.user_agent,  # type: ignore[union-attr]
        allowed_methods=["GET", "POST"],
    )

    try:
        ids = _ncbi_esearch(
            db=inputs.db,
            term=inputs.search_term,
            max_results=inputs.max_results,
            config=config,  # type: ignore[arg-type]
            session=session,
        )
        return NCBIEsearchOutput(ids=ids)
    finally:
        session.close()
