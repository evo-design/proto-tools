"""proto_tools/tools/database_retrieval/ncbi/efetch.py.

Wraps the NCBI E-utilities efetch endpoint for fetching sequences and
records from protein, nuccore, and gene databases.
"""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from proto_tools.tools.database_retrieval.ncbi.shared_data_models import (
    NCBIFastaRecord,
    NCBIFetchConfig,
    _ncbi_efetch,
    _parse_fasta_records,
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


class NCBIEfetchInput(BaseToolInput):
    """Input for NCBI efetch.

    Attributes:
        db (Literal['protein', 'nuccore', 'gene']): NCBI database to query: 'protein', 'nuccore' (nucleotide core),
            or 'gene'.
        identifier (str): Accession or NCBI ID to fetch (e.g. 'NP_000537.3').
        return_format (Literal['fasta', 'fasta_cds_na']): NCBI rettype: 'fasta' for sequences or
            'fasta_cds_na' for coding DNA sequences.
        seq_start (int | None): Start position for subsequence extraction (1-indexed,
            inclusive).
        seq_stop (int | None): Stop position for subsequence extraction (1-indexed,
            inclusive).
        strand (Literal['+', '-'] | None): Strand for nucleotide retrieval (+ or -).
    """

    db: Literal["protein", "nuccore", "gene"] = InputField(
        description="NCBI database to query: 'protein', 'nuccore' (nucleotide core), or 'gene'"
    )
    identifier: str = InputField(
        description="Accession or NCBI ID for efetch (e.g. 'NP_000537.3', '7157')",
    )
    return_format: Literal["fasta", "fasta_cds_na"] = InputField(
        default="fasta",
        description="NCBI rettype: 'fasta' for sequences, 'fasta_cds_na' for CDS",
    )
    seq_start: int | None = InputField(
        default=None,
        ge=1,
        description="Start position for subsequence extraction (1-indexed, inclusive)",
    )
    seq_stop: int | None = InputField(
        default=None,
        ge=1,
        description="Stop position for subsequence extraction (1-indexed, inclusive)",
    )
    strand: Literal["+", "-"] | None = InputField(
        default=None,
        description="Strand for nucleotide retrieval",
    )


class NCBIEfetchOutput(BaseToolOutput):
    """Output from NCBI efetch.

    Attributes:
        fasta_records (list[NCBIFastaRecord]): Parsed FASTA records from efetch.
        source_url (str): Sanitized URL used for the request.
    """

    fasta_records: list[NCBIFastaRecord] = Field(default_factory=list, description="Parsed FASTA records")
    source_url: str = Field(description="Sanitized request URL")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Any, file_format: str) -> None:
        if file_format == "json":
            path = Path(export_path).with_suffix(".json")
            with path.open("w", encoding="utf-8") as f:
                json.dump(self.model_dump(mode="json"), f, indent=2)
            return
        raise ValueError(f"Unsupported format: {file_format}")


NCBIEfetchConfig = NCBIFetchConfig


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return NCBIEfetchInput(db="protein", identifier="NP_000537.3")


@tool(
    key="ncbi-efetch",
    label="NCBI Entrez EFetch",
    category="database_retrieval",
    input_class=NCBIEfetchInput,
    config_class=NCBIFetchConfig,
    output_class=NCBIEfetchOutput,
    description="Fetch sequences and records from NCBI Entrez by accession or ID",
    uses_gpu=False,
    example_input=example_input,
)
def run_ncbi_efetch(
    inputs: NCBIEfetchInput,
    config: NCBIFetchConfig,
    instance: Any = None,
) -> NCBIEfetchOutput:
    """Fetch sequences from NCBI Entrez databases.

    Retrieves FASTA records by accession or ID with optional subsequence
    extraction.

    Args:
        inputs (NCBIEfetchInput): Database, identifier, format, and optional coordinate
            parameters.
        config (NCBIFetchConfig): HTTP timeout, retry, and authentication settings.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        NCBIEfetchOutput: NCBIEfetchOutput containing parsed FASTA records.
    """
    del instance

    session = build_http_session(
        http_retries=config.http_retries,
        backoff_seconds=config.backoff_seconds,
        user_agent=config.user_agent,
        allowed_methods=["GET", "POST"],
    )

    try:
        result = _ncbi_efetch(
            db=inputs.db,
            identifier=inputs.identifier,
            rettype=inputs.return_format,
            config=config,
            session=session,
            seq_start=inputs.seq_start,
            seq_stop=inputs.seq_stop,
            strand=inputs.strand,
        )
        if result is None:
            raise ValueError(f"No record found for {inputs.db}:{inputs.identifier}")
        text, url = result
        records = _parse_fasta_records(text)
        return NCBIEfetchOutput(fasta_records=records, source_url=url)
    finally:
        session.close()
