"""PDB fetch FASTA tool — retrieve chain sequences from RCSB PDB.

Wraps the RCSB PDB FASTA endpoint for fetching chain sequences with
automatic protein/nucleic acid classification.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils.http_session import build_http_session
from bio_programming_tools.utils.tool_io import BaseToolInput, BaseToolOutput

from .shared_data_models import (
    PdbChain,
    PdbFetchConfig,
    _PDB_FASTA_BASE,
    _chain_id_from_header,
    _fetch_pdb_fasta,
    _is_protein_sequence,
)

# ============================================================================
# Data Models
# ============================================================================


class PdbFetchFastaInput(BaseToolInput):
    """Input for PDB FASTA chain fetch.

    Attributes:
        pdb_id: PDB accession (e.g. '1LBG').
    """

    pdb_id: str = Field(description="PDB accession (e.g. '1LBG')")


class PdbFetchFastaOutput(BaseToolOutput):
    """Output from PDB FASTA chain fetch.

    Attributes:
        chains: Parsed chain sequences with protein/nucleotide classification.
        source_url: URL used for the request.
    """

    chains: List[PdbChain] = Field(default_factory=list, description="Parsed chain sequences")
    source_url: Optional[str] = Field(default=None, description="Request URL")

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


PdbFetchFastaConfig = PdbFetchConfig


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input():
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
)
def run_pdb_fetch_fasta(
    inputs: PdbFetchFastaInput,
    config: PdbFetchConfig | None = None,
    instance=None,
) -> PdbFetchFastaOutput:
    """Fetch chain sequences from RCSB PDB.

    Returns parsed chain sequences with automatic protein/nucleic acid
    classification based on amino acid composition.

    Args:
        inputs: PDB accession to look up.
        config: HTTP timeout and retry settings.

    Returns:
        PdbFetchFastaOutput with chain sequences.
    """
    del instance

    session = build_http_session(
        http_retries=config.http_retries,
        backoff_seconds=config.backoff_seconds,
        user_agent=config.user_agent,
        mount_http=True,
    )
    pdb_id = inputs.pdb_id.upper()

    try:
        raw_chains = _fetch_pdb_fasta(pdb_id, config, session)
        if raw_chains is None:
            return PdbFetchFastaOutput()
        pdb_chains = [
            PdbChain(
                chain_id=_chain_id_from_header(header),
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
