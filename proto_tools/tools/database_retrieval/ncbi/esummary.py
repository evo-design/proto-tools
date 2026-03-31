"""proto_tools/tools/database_retrieval/ncbi/esummary.py

Wraps the NCBI E-utilities esummary endpoint for fetching metadata
about protein, nuccore, and gene records."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import Field

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseToolInput,
    BaseToolOutput,
    InputField,
    build_http_session,
)

from .shared_data_models import NCBIFetchConfig, _ncbi_esummary

# ============================================================================
# Data Models
# ============================================================================


class NCBIEsummaryInput(BaseToolInput):
    """Input for NCBI esummary.

    Attributes:
        db (Literal['protein', 'nuccore', 'gene']): NCBI database to query: 'protein', 'nuccore' (nucleotide core),
            or 'gene'.
        identifier (str): Accession or NCBI ID to summarize (e.g. 'NP_000537.3',
            '7157').
    """

    db: Literal["protein", "nuccore", "gene"] = InputField(
        description="NCBI database to query: 'protein', 'nuccore' (nucleotide core), or 'gene'"
    )
    identifier: str = InputField(
        description="Accession or NCBI ID for esummary (e.g. 'NP_000537.3', '7157')",
    )


class NCBIEsummaryOutput(BaseToolOutput):
    """Output from NCBI esummary.

    Attributes:
        summary (dict[str, Any]): Record summary data returned by esummary.
        source_url (str): Sanitized URL used for the request.
    """

    summary: Dict[str, Any] = Field(
        default_factory=dict, description="Record summary data returned by esummary"
    )
    source_url: str = Field(description="Sanitized request URL")

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


NCBIEsummaryConfig = NCBIFetchConfig


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input():
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
)
def run_ncbi_esummary(
    inputs: NCBIEsummaryInput,
    config: NCBIFetchConfig | None = None,
    instance=None,
) -> NCBIEsummaryOutput:
    """Retrieve record summary metadata from NCBI Entrez.

    Args:
        inputs (NCBIEsummaryInput): Database and identifier to summarize.
        config (NCBIFetchConfig | None): HTTP timeout, retry, and authentication settings.

    Returns:
        NCBIEsummaryOutput: NCBIEsummaryOutput containing the record summary.
    """
    del instance

    session = build_http_session(
        http_retries=config.http_retries,
        backoff_seconds=config.backoff_seconds,
        user_agent=config.user_agent,
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
            raise ValueError(
                f"No record found for {inputs.db}:{inputs.identifier}"
            )
        summary, url = result
        return NCBIEsummaryOutput(summary=summary, source_url=url)
    finally:
        session.close()
