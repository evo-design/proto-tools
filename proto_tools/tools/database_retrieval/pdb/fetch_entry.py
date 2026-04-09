"""proto_tools/tools/database_retrieval/pdb/fetch_entry.py.

Wraps the RCSB PDB REST API core entry endpoint for fetching title,
experimental method, and resolution for a PDB accession.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import Field

from proto_tools.tools.database_retrieval.pdb.shared_data_models import (
    _PDB_ENTRY_BASE,
    PdbFetchConfig,
    _fetch_pdb_entry,
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


class PdbFetchEntryInput(BaseToolInput):
    """Input for PDB entry metadata fetch.

    Attributes:
        pdb_id (str): PDB accession (e.g. '1LBG').
    """

    pdb_id: str = InputField(description="PDB accession (e.g. '1LBG')")


class PdbFetchEntryOutput(BaseToolOutput):
    """Output from PDB entry metadata fetch.

    Attributes:
        title (str | None): Structure title.
        method (str | None): Experimental method.
        resolution (float | None): Resolution in angstroms.
        source_url (str | None): URL used for the request.
    """

    title: str | None = Field(default=None, description="Structure title")
    method: str | None = Field(default=None, description="Experimental method")
    resolution: float | None = Field(default=None, description="Resolution in angstroms")
    source_url: str | None = Field(default=None, description="Request URL")

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


PdbFetchEntryConfig = PdbFetchConfig


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return PdbFetchEntryInput(pdb_id="1LBG")


@tool(
    key="pdb-fetch-entry",
    label="PDB Fetch Entry",
    category="database_retrieval",
    input_class=PdbFetchEntryInput,
    config_class=PdbFetchConfig,
    output_class=PdbFetchEntryOutput,
    description="Fetch structure metadata (title, method, resolution) from RCSB PDB",
    uses_gpu=False,
    example_input=example_input,
)
def run_pdb_fetch_entry(
    inputs: PdbFetchEntryInput,
    config: PdbFetchConfig,
    instance: Any = None,
) -> PdbFetchEntryOutput:
    """Fetch structure metadata from RCSB PDB.

    Returns title, experimental method, and resolution for a PDB accession.

    Args:
        inputs (PdbFetchEntryInput): PDB accession to look up.
        config (PdbFetchConfig): HTTP timeout and retry settings.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        PdbFetchEntryOutput: PdbFetchEntryOutput with structure metadata.
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
