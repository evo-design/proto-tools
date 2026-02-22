"""PDB fetch entry tool — retrieve structure metadata from RCSB PDB.

Wraps the RCSB PDB REST API core entry endpoint for fetching title,
experimental method, and resolution for a PDB accession.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils.http_session import build_http_session
from bio_programming_tools.utils.tool_io import BaseToolInput, BaseToolOutput

from .shared_data_models import (
    PdbFetchConfig,
    _PDB_ENTRY_BASE,
    _fetch_pdb_entry,
)

# ============================================================================
# Data Models
# ============================================================================


class PdbFetchEntryInput(BaseToolInput):
    """Input for PDB entry metadata fetch.

    Attributes:
        pdb_id: PDB accession (e.g. '1LBG').
    """

    pdb_id: str = Field(description="PDB accession (e.g. '1LBG')")


class PdbFetchEntryOutput(BaseToolOutput):
    """Output from PDB entry metadata fetch.

    Attributes:
        title: Structure title.
        method: Experimental method.
        resolution: Resolution in angstroms.
        source_url: URL used for the request.
    """

    title: Optional[str] = Field(default=None, description="Structure title")
    method: Optional[str] = Field(default=None, description="Experimental method")
    resolution: Optional[float] = Field(default=None, description="Resolution in angstroms")
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


PdbFetchEntryConfig = PdbFetchConfig


# ============================================================================
# Tool Implementation
# ============================================================================


@tool(
    key="pdb-fetch-entry",
    label="PDB Fetch Entry",
    category="database_retrieval",
    input=PdbFetchEntryInput,
    config=PdbFetchConfig,
    output=PdbFetchEntryOutput,
    description="Fetch structure metadata (title, method, resolution) from RCSB PDB",
    uses_gpu=False,
)
def run_pdb_fetch_entry(
    inputs: PdbFetchEntryInput,
    config: PdbFetchConfig,
    instance=None,
) -> PdbFetchEntryOutput:
    """Fetch structure metadata from RCSB PDB.

    Returns title, experimental method, and resolution for a PDB accession.

    Args:
        inputs: PDB accession to look up.
        config: HTTP timeout and retry settings.

    Returns:
        PdbFetchEntryOutput with structure metadata.
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
        meta = _fetch_pdb_entry(pdb_id, config, session)
        if meta is None:
            return PdbFetchEntryOutput()
        return PdbFetchEntryOutput(
            title=meta.get("title"),
            method=meta.get("method"),
            resolution=meta.get("resolution"),
            source_url=f"{_PDB_ENTRY_BASE}/{pdb_id}",
        )
    finally:
        session.close()
