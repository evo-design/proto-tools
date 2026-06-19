"""proto_tools/tools/database_retrieval/pdb/fetch_entry.py.

Wraps the RCSB PDB REST API core entry endpoint for fetching title,
experimental method, and resolution for a PDB accession.
"""

import csv
import json
from pathlib import Path
from typing import Any

from pydantic import Field

from proto_tools.tools.database_retrieval.pdb.shared_data_models import (
    _BACKOFF_SECONDS,
    _HTTP_RETRIES,
    _PDB_ENTRY_BASE,
    _USER_AGENT,
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

    pdb_id: str = InputField(title="PDB ID", description="PDB accession (e.g. '1LBG')")


class PdbFetchEntryOutput(BaseToolOutput):
    """Output from PDB entry metadata fetch.

    Attributes:
        title (str | None): Structure title.
        method (str | None): Experimental method.
        resolution (float | None): Resolution in angstroms.
        source_url (str | None): URL used for the request.
    """

    title: str | None = Field(default=None, title="Title", description="Human-readable RCSB entry title")
    method: str | None = Field(
        default=None,
        title="Experimental Method",
        description="PDB experimental method (e.g. X-RAY DIFFRACTION, SOLUTION NMR, ELECTRON MICROSCOPY)",
    )
    resolution: float | None = Field(
        default=None, title="Resolution", description="Resolution in Å (None for NMR or fiber diffraction)"
    )
    source_url: str | None = Field(default=None, title="Source URL", description="Request URL")

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
            row = self.model_dump()
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
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
    cacheable=True,
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
        config (PdbFetchConfig): Empty placeholder (PDB fetch has no user settings).
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        PdbFetchEntryOutput: PdbFetchEntryOutput with structure metadata.
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
        meta = _fetch_pdb_entry(pdb_id, session)
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
