"""proto_tools/tools/database_retrieval/ncbi/esearch.py.

Wraps the NCBI E-utilities esearch endpoint for finding IDs across
NCBI Entrez databases.
"""

import csv
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from proto_tools.tools.database_retrieval.ncbi.shared_data_models import (
    _BACKOFF_SECONDS,
    _HTTP_RETRIES,
    _USER_AGENT,
    NCBIDatabase,
    NCBIFetchConfig,
    _ncbi_esearch,
)
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
        db (NCBIDatabase): NCBI database to query (e.g. 'protein', 'nuccore',
            'gene', 'pubmed', 'taxonomy', 'structure').
        search_term (str): NCBI search query.
        max_results (int): Max IDs returned (NCBI ``retmax``).
        retstart (int): 0-indexed offset of the first hit (NCBI ``retstart``).
        sort (str | None): Sort key (db-dependent — e.g. 'relevance' /
            'pub_date' / 'most_recent' on pubmed).
        field (str | None): Restrict the search term to a single index field
            (db-dependent — e.g. 'title' / 'author' on pubmed).
        datetype (Literal['mdat','pdat','edat'] | None): Date axis for
            mindate/maxdate/reldate (modification / publication / Entrez).
        mindate (str | None): Lower date bound, ``YYYY/MM/DD`` (also
            ``YYYY/MM`` and ``YYYY``); requires datetype.
        maxdate (str | None): Upper date bound; requires datetype.
        reldate (int | None): Restrict to records dated within the last N
            days; requires datetype.
    """

    db: NCBIDatabase = InputField(
        title="Database",
        description="NCBI database to query (e.g. 'protein', 'nuccore', 'gene', 'pubmed', 'taxonomy')",
    )
    search_term: str = InputField(
        title="Search Term",
        description="NCBI search query (e.g. 'lacI[Gene] AND Escherichia coli[Organism]')",
    )
    max_results: int = InputField(
        default=20,
        ge=1,
        le=10000,
        title="Max Results",
        description="Max IDs to return (NCBI retmax; default 20)",
    )
    retstart: int = InputField(
        default=0,
        ge=0,
        title="Start Offset",
        description="0-indexed offset of the first hit (NCBI retstart). For pagination.",
    )
    sort: str | None = InputField(
        default=None,
        title="Sort Key",
        description="Db-specific sort key (e.g. 'relevance' or 'pub_date' on pubmed)",
    )
    field: str | None = InputField(
        default=None,
        title="Search Field",
        description="Restrict search term to a single field (e.g. 'title' or 'author' on pubmed)",
    )
    datetype: Literal["mdat", "pdat", "edat"] | None = InputField(
        default=None,
        title="Date Type",
        description="Date axis for mindate/maxdate/reldate (one of mdat, pdat, edat)",
    )
    mindate: str | None = InputField(
        default=None,
        title="Min Date",
        description="Lower date bound (YYYY/MM/DD, YYYY/MM, or YYYY); requires datetype",
    )
    maxdate: str | None = InputField(
        default=None,
        title="Max Date",
        description="Upper date bound (YYYY/MM/DD, YYYY/MM, or YYYY); requires datetype",
    )
    reldate: int | None = InputField(
        default=None,
        ge=1,
        title="Relative Date (Days)",
        description="Records dated within the last N days; requires datetype",
    )

    @model_validator(mode="after")
    def _check_date_filters_have_datetype(self) -> "NCBIEsearchInput":
        """Reject mindate/maxdate/reldate without datetype (NCBI silently ignores them)."""
        if any(v is not None for v in (self.mindate, self.maxdate, self.reldate)) and self.datetype is None:
            raise ValueError("mindate / maxdate / reldate require datetype to be set ('mdat', 'pdat', or 'edat')")
        return self


class NCBIEsearchOutput(BaseToolOutput):
    """Output from NCBI esearch.

    Attributes:
        ids (list[str]): List of NCBI IDs matching the search query.
    """

    ids: list[str] = Field(default_factory=list, title="IDs", description="List of NCBI IDs found by the search")

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
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id"])
                writer.writerows([[i] for i in self.ids])
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
    cacheable=True,
)
def run_ncbi_esearch(
    inputs: NCBIEsearchInput,
    config: NCBIFetchConfig,
    instance: Any = None,
) -> NCBIEsearchOutput:
    """Search NCBI Entrez databases by query term.

    Args:
        inputs (NCBIEsearchInput): Search parameters including database, query term, and
            max results.
        config (NCBIFetchConfig): NCBI API key and email settings.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        NCBIEsearchOutput: Matching NCBI record IDs.
    """
    del instance

    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
        allowed_methods=["GET", "POST"],
    )

    try:
        ids = _ncbi_esearch(
            db=inputs.db,
            term=inputs.search_term,
            max_results=inputs.max_results,
            config=config,
            session=session,
            retstart=inputs.retstart,
            sort=inputs.sort,
            field=inputs.field,
            datetype=inputs.datetype,
            mindate=inputs.mindate,
            maxdate=inputs.maxdate,
            reldate=inputs.reldate,
        )
        return NCBIEsearchOutput(ids=ids)
    finally:
        session.close()
