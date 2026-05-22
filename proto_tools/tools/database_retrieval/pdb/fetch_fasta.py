"""proto_tools/tools/database_retrieval/pdb/fetch_fasta.py.

Wraps the RCSB PDB FASTA endpoint for fetching chain sequences with
automatic protein/nucleic acid classification.
"""

import csv
import json
from pathlib import Path
from typing import Any

from pydantic import Field

from proto_tools.tools.database_retrieval.pdb.shared_data_models import (
    _BACKOFF_SECONDS,
    _HTTP_RETRIES,
    _PDB_FASTA_BASE,
    _USER_AGENT,
    PdbChain,
    PdbFetchConfig,
    _chain_ids_from_header,
    _fetch_pdb_fasta,
    _is_protein_sequence,
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


class PdbFetchFastaInput(BaseToolInput):
    """Input for PDB FASTA chain fetch.

    Attributes:
        pdb_id (str): PDB accession (e.g. '1LBG').
    """

    pdb_id: str = InputField(title="PDB ID", description="PDB accession (e.g. '1LBG')")


class PdbFetchFastaOutput(BaseToolOutput):
    """Output from PDB FASTA chain fetch.

    Attributes:
        chains (list[PdbChain]): Parsed chain sequences with protein/nucleotide classification.
        source_url (str | None): URL used for the request.
    """

    chains: list[PdbChain] = Field(
        default_factory=list,
        title="Chains",
        description="Parsed chain sequences",
    )
    source_url: str | None = Field(default=None, title="Source URL", description="Request URL")

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
            rows = []
            for c in self.chains:
                d = c.model_dump()
                d["chain_ids"] = ";".join(d["chain_ids"])
                rows.append(d)
            with path.open("w", encoding="utf-8", newline="") as f:
                if not rows:
                    return
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            return
        if file_format == "fasta":
            with path.open("w", encoding="utf-8") as f:
                for c in self.chains:
                    f.write(f">{c.header}\n{c.sequence}\n")
            return
        raise ValueError(f"Unsupported format: {file_format}")


PdbFetchFastaConfig = PdbFetchConfig


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return PdbFetchFastaInput(pdb_id="1LBG")


@tool(
    key="pdb-fetch-fasta",
    label="PDB Fetch FASTA",
    category="database_retrieval",
    input_class=PdbFetchFastaInput,
    config_class=PdbFetchConfig,
    output_class=PdbFetchFastaOutput,
    description="Fetch chain sequences from RCSB PDB with protein/nucleotide classification",
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_pdb_fetch_fasta(
    inputs: PdbFetchFastaInput,
    config: PdbFetchConfig,
    instance: Any = None,
) -> PdbFetchFastaOutput:
    """Fetch chain sequences from RCSB PDB.

    Returns parsed chain sequences with automatic protein/nucleic acid
    classification based on amino acid composition.

    Args:
        inputs (PdbFetchFastaInput): PDB accession to look up.
        config (PdbFetchConfig): Empty placeholder (PDB fetch has no user knobs).
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        PdbFetchFastaOutput: PdbFetchFastaOutput with chain sequences.
    """
    del config, instance

    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
        mount_http=True,
    )
    pdb_id = inputs.pdb_id.upper()

    try:
        raw_chains = _fetch_pdb_fasta(pdb_id, session)
        if raw_chains is None:
            return PdbFetchFastaOutput()
        pdb_chains = [
            PdbChain(
                chain_ids=_chain_ids_from_header(header),
                header=header,
                sequence=sequence,
                is_protein=_is_protein_sequence(sequence),
            )
            for header, sequence in raw_chains
        ]
        return PdbFetchFastaOutput(
            chains=pdb_chains,
            source_url=f"{_PDB_FASTA_BASE}/{pdb_id}",
        )
    finally:
        session.close()
