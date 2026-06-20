"""proto_tools/tools/database_retrieval/ncbi/efetch.py.

Wraps the NCBI E-utilities efetch endpoint for fetching records from
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
    NCBIFastaRecord,
    NCBIFetchConfig,
    NCBISequenceDatabase,
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
    """Input for NCBI efetch (sequence dbs only; use esummary for metadata).

    Attributes:
        db (NCBISequenceDatabase): Sequence database to query.
        identifier (str): Accession or NCBI ID to fetch (e.g. 'NP_000537.3').
        return_format (Literal['fasta', 'fasta_cds_na']): NCBI rettype.
            'fasta_cds_na' is nuccore-only.
        seq_start (int | None): Subsequence start (1-indexed, inclusive).
        seq_stop (int | None): Subsequence stop (1-indexed, inclusive).
        strand (Literal['+', '-'] | None): Strand for nucleotide retrieval.
    """

    db: NCBISequenceDatabase = InputField(
        title="Database",
        description="NCBI sequence database to query (protein, nuccore, or nucleotide)",
    )
    identifier: str = InputField(
        title="Identifier",
        description="Accession or NCBI ID for efetch (e.g. 'NP_000537.3', '7157')",
    )
    return_format: Literal["fasta", "fasta_cds_na"] = InputField(
        default="fasta",
        title="Return Format",
        description="NCBI rettype: 'fasta' (protein or nuccore) or 'fasta_cds_na' (CDS, nuccore-only)",
    )
    seq_start: int | None = InputField(
        default=None,
        ge=1,
        title="Sequence Start",
        description="Start position for subsequence extraction (1-indexed, inclusive)",
    )
    seq_stop: int | None = InputField(
        default=None,
        ge=1,
        title="Sequence Stop",
        description="Stop position for subsequence extraction (1-indexed, inclusive)",
    )
    strand: Literal["+", "-"] | None = InputField(
        default=None,
        title="Strand",
        description="Strand for nucleotide retrieval (nuccore/nucleotide only)",
    )

    @model_validator(mode="after")
    def _validate_format_db(self) -> "NCBIEfetchInput":
        """Reject incompatible db / return_format combos before NCBI silently returns empty."""
        if self.return_format == "fasta_cds_na" and self.db == "protein":
            raise ValueError("return_format='fasta_cds_na' requires db='nuccore' or 'nucleotide' (CDS extraction)")
        return self


class NCBIEfetchOutput(BaseToolOutput):
    """Output from NCBI efetch.

    Attributes:
        fasta_records (list[NCBIFastaRecord]): Parsed FASTA records from efetch.
        source_url (str): Sanitized URL used for the request.
    """

    fasta_records: list[NCBIFastaRecord] = Field(
        default_factory=list, title="FASTA Records", description="Parsed FASTA records"
    )
    source_url: str = Field(title="Source URL", description="Sanitized request URL")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "csv", "fasta"]

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
            rows = [r.model_dump() for r in self.fasta_records]
            with path.open("w", encoding="utf-8", newline="") as f:
                if not rows:
                    return
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            return
        if file_format == "fasta":
            with path.open("w", encoding="utf-8") as f:
                for r in self.fasta_records:
                    f.write(f">{r.header}\n{r.sequence}\n")
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
    description="Fetch FASTA records from NCBI sequence dbs (protein/nuccore) by accession or ID",
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
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
        config (NCBIFetchConfig): NCBI API key and email settings.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        NCBIEfetchOutput: Parsed FASTA records and the sanitized request URL.
    """
    del instance

    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
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
