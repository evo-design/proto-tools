"""proto_tools/tools/database_retrieval/ncbi/esummary.py.

Wraps the NCBI E-utilities esummary endpoint for fetching record
metadata from NCBI Entrez databases.
"""

import csv
import json
from pathlib import Path
from typing import Any

from pydantic import Field

from proto_tools.tools.database_retrieval.ncbi.shared_data_models import (
    _BACKOFF_SECONDS,
    _HTTP_RETRIES,
    _USER_AGENT,
    NCBIDatabase,
    NCBIFetchConfig,
    _ncbi_esummary,
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


class NCBIEsummaryInput(BaseToolInput):
    """Input for NCBI esummary.

    Attributes:
        db (NCBIDatabase): NCBI database to query (e.g. 'protein', 'nuccore',
            'gene', 'pubmed', 'taxonomy', 'structure'). See ``NCBIDatabase``
            for the full set of supported databases.
        identifier (str): Accession or NCBI ID to summarize (e.g. 'NP_000537.3',
            '7157').
    """

    db: NCBIDatabase = InputField(
        title="Database",
        description="NCBI database to query (e.g. 'protein', 'nuccore', 'gene', 'pubmed', 'taxonomy')",
    )
    identifier: str = InputField(
        title="Identifier",
        description="Accession or NCBI ID for esummary (e.g. 'NP_000537.3', '7157')",
    )


class NCBIEsummaryOutput(BaseToolOutput):
    """Output from NCBI esummary.

    Attributes:
        summary (dict[str, Any]): Record summary data returned by esummary.
        source_url (str): Sanitized URL used for the request.
    """

    summary: dict[str, Any] = Field(
        default_factory=dict, title="Summary", description="Record summary data returned by esummary"
    )
    source_url: str = Field(title="Source URL", description="Sanitized request URL")

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
            # esummary's `summary` is a {uid -> record} dict whose record
            # fields differ per db; flatten to one row per uid with nested
            # values JSON-encoded.
            records = [v for k, v in self.summary.items() if k != "uids" and isinstance(v, dict)]
            rows: list[dict[str, Any]] = [
                {
                    k: (json.dumps(v, separators=(",", ":")) if isinstance(v, (dict, list)) else v)
                    for k, v in rec.items()
                }
                for rec in records
            ]
            with path.open("w", encoding="utf-8", newline="") as f:
                if not rows:
                    return
                fieldnames: list[str] = []
                seen: set[str] = set()
                for r in rows:
                    for k in r:
                        if k not in seen:
                            seen.add(k)
                            fieldnames.append(k)
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow({k: r.get(k, "") for k in fieldnames})
            return
        raise ValueError(f"Unsupported format: {file_format}")


NCBIEsummaryConfig = NCBIFetchConfig


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return NCBIEsummaryInput(db="protein", identifier="NP_000537.3")


@tool(
    key="ncbi-esummary",
    label="NCBI Entrez ESummary",
    category="database_retrieval",
    input_class=NCBIEsummaryInput,
    config_class=NCBIFetchConfig,
    output_class=NCBIEsummaryOutput,
    description="Retrieve record summary metadata from NCBI Entrez by ID",
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_ncbi_esummary(
    inputs: NCBIEsummaryInput,
    config: NCBIFetchConfig,
    instance: Any = None,
) -> NCBIEsummaryOutput:
    """Retrieve record summary metadata from NCBI Entrez.

    Args:
        inputs (NCBIEsummaryInput): Database and identifier to summarize.
        config (NCBIFetchConfig): NCBI API key and email settings.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        NCBIEsummaryOutput: NCBIEsummaryOutput containing the record summary.
    """
    del instance

    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
        allowed_methods=["GET", "POST"],
    )

    try:
        result = _ncbi_esummary(
            db=inputs.db,
            identifier=inputs.identifier,
            config=config,
            session=session,
        )
        if result is None:
            raise ValueError(f"No record found for {inputs.db}:{inputs.identifier}")
        summary, url = result
        return NCBIEsummaryOutput(summary=summary, source_url=url)
    finally:
        session.close()
