"""Wraps Foldseek multimer (complex) structural search — remote and local.

Distinct from `foldseek-search`: input is a multi-chain PDB assembly, and the
remote mode dispatches to https://search.foldseek.com/foldmulti via the same
ticket-based API but with mode prefixed ``complex-`` (e.g. ``complex-3diaa``).

Cite: Kim et al., *Nature Methods* 2025, DOI 10.1038/s41592-025-02593-7.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from proto_tools.tools.structure_alignment.foldseek.foldseek_search import (
    _BACKOFF_SECONDS,
    _FOLDSEEK_BASE,
    _HTTP_RETRIES,
    _LOCAL_DB_PSEUDONAME,
    _RESULT_DOWNLOAD_TIMEOUT_SECONDS,
    FoldseekDatabase,
    FoldseekHit,
    FoldseekMode,
    FoldseekSearchMode,
    _parse_m8_archive,
    _parse_m8_text,
    _submit,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
    build_http_session,
    poll_until_complete,
)

logger = logging.getLogger(__name__)

# Distinct user-agent so server-side analytics can separate single-chain vs multimer traffic.
_USER_AGENT = "proto-tools/foldseek-multimer-search-v1"


# ============================================================================
# Data Models
# ============================================================================


FoldseekMultimerHit = FoldseekHit
"""Same shape as FoldseekHit — multimer search returns the standard 12-column M8."""


class FoldseekMultimerSearchInput(BaseToolInput):
    """Input for Foldseek multimer (complex) search.

    Attributes:
        structure_text (str): Multi-chain PDB-format text of the query complex.
            Use ``alphafold-db-fetch`` (with `multimer` ID) or `pdb-fetch-entry`
            for an experimental complex.
    """

    structure_text: str = InputField(
        title="Structure Text",
        description="Multi-chain PDB-format text of the query complex",
    )


class FoldseekMultimerSearchConfig(BaseConfig):
    """Configuration for Foldseek multimer search (remote or local).

    Attributes:
        search_mode (FoldseekSearchMode): 'remote' (default; hits the public
            Foldseek-Multimer endpoint at ``search.foldseek.com/foldmulti``)
            or 'local' (runs ``foldseek easy-multimersearch`` locally).
        databases (list[FoldseekDatabase]): Remote-only — server-hosted
            multimer-aware databases. Default ['pdb100'].
        mode (FoldseekMode): Remote-only — alignment mode. Wire-level the
            mode is prefixed ``complex-`` automatically.
        poll_interval_seconds (float): Remote-only — delay between status polls.
        timeout_seconds (float): Remote-only — max wall-clock time.
        local_db (str | None): Local-only (required) — path to a local
            multimer-aware Foldseek DB.
        evalue (float): Local-only — E-value cutoff (lower = stricter).
        sensitivity (float): Local-only — prefilter sensitivity (1.0-9.5;
            higher = slower + more sensitive).
        max_seqs (int): Local-only — max prefilter targets per query.
        alignment_type (Literal[0, 1, 2, 3]): Local-only — alignment scoring
            method (0=3Di, 1=TMalign, 2=3Di+AA, 3=LoL).
        tmscore_threshold (float): Local-only — keep alignments with TM-score
            above this (0-1). 0.0 keeps all.
        lddt_threshold (float): Local-only — keep alignments with LDDT above
            this (0-1). 0.0 keeps all.
        num_threads (int): Local-only — CPU threads.
    """

    search_mode: FoldseekSearchMode = ConfigField(
        title="Search Mode",
        default="remote",
        description="'remote' (search.foldseek.com/foldmulti) or 'local' (foldseek easy-multimersearch)",
    )
    databases: list[FoldseekDatabase] = ConfigField(
        title="Databases",
        default_factory=lambda: ["pdb100"],
        description="Remote-only — multimer-aware databases to search",
    )
    mode: FoldseekMode = ConfigField(
        title="Alignment Mode",
        default="3diaa",
        description="Remote-only — alignment mode (wire-encoded as 'complex-{mode}' for multimer)",
    )
    poll_interval_seconds: float = ConfigField(
        title="Poll Interval (seconds)",
        default=5.0,
        ge=1.0,
        description="Remote-only — delay between status polls",
        include_in_key=False,
    )
    timeout_seconds: float = ConfigField(
        title="Timeout (seconds)",
        default=600.0,
        ge=10.0,
        description="Remote-only — max wall-clock time",
        include_in_key=False,
    )
    local_db: str | None = ConfigField(
        title="Local Foldseek Database",
        default=None,
        description="Path to a local Foldseek multimer DB or a directory of multi-chain PDBs (required)",
    )
    evalue: float = ConfigField(
        title="E-value Threshold",
        default=10.0,
        ge=0.0,
        description="E-value cutoff. Lower = stricter (1e-3 for confident homologs; default 10.0 reports all)",
    )
    sensitivity: float = ConfigField(
        title="Sensitivity",
        default=4.0,
        ge=1.0,
        le=9.5,
        description="Prefilter sensitivity (1.0-9.5). Lower = faster, higher = more sensitive (default 4.0)",
    )
    max_seqs: int = ConfigField(
        title="Max Sequences",
        default=300,
        ge=1,
        description="Max prefilter targets per query (multimer default 300; raise for more hits)",
    )
    alignment_type: Literal[0, 1, 2, 3] = ConfigField(
        title="Alignment Type",
        default=2,
        description="Alignment scoring: 0=3Di SW, 1=TMalign, 2=3Di+AA (default), 3=LoL",
    )
    tmscore_threshold: float = ConfigField(
        title="TM-score Threshold",
        default=0.0,
        ge=0.0,
        le=1.0,
        description="TM-score floor for hits (0-1). 0.0 keeps all; 0.5 ≈ same fold",
    )
    lddt_threshold: float = ConfigField(
        title="LDDT Threshold",
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LDDT floor for hits (0-1). 0.0 keeps all; 0.7+ = high local accuracy",
    )
    num_threads: int = ConfigField(
        title="Threads (local)", default=4, ge=1, description="CPU threads for local search", include_in_key=False
    )

    _REMOTE_ONLY_DEFAULTS = {  # noqa: RUF012
        "databases": ["pdb100"],
        "mode": "3diaa",
        "poll_interval_seconds": 5.0,
        "timeout_seconds": 600.0,
    }
    _LOCAL_ONLY_DEFAULTS = {  # noqa: RUF012
        "local_db": None,
        "evalue": 10.0,
        "sensitivity": 4.0,
        "max_seqs": 300,
        "alignment_type": 2,
        "tmscore_threshold": 0.0,
        "lddt_threshold": 0.0,
        "num_threads": 4,
    }

    @model_validator(mode="after")
    def validate_mode_requirements(self) -> "FoldseekMultimerSearchConfig":
        """Hard-error on missing required local fields; soft-warn on cross-mode misuse."""
        if self.search_mode == "local" and not self.local_db:
            raise ValueError("local_db is required when search_mode='local'")

        ignored_table = self._LOCAL_ONLY_DEFAULTS if self.search_mode == "remote" else self._REMOTE_ONLY_DEFAULTS
        kind = "local-only" if self.search_mode == "remote" else "remote-only"
        for name, default in ignored_table.items():
            if getattr(self, name) != default:
                logger.warning("Config field '%s' is %s and is ignored in %s mode.", name, kind, self.search_mode)
        return self


class FoldseekMultimerSearchOutput(BaseToolOutput):
    """Output from Foldseek multimer search.

    Attributes:
        ticket_id (str): Remote job ticket ID; empty in local mode.
        hits (list[FoldseekMultimerHit]): Multimer alignment hits.
        num_hits (int): ``len(hits)``.
        databases_queried (list[str]): Databases included in this search;
            in local mode contains the single local DB path.
        result_url (str): Remote result-archive URL; empty in local mode.
    """

    ticket_id: str = Field(
        title="Ticket ID",
        description="Foldseek job ticket ID (remote only; empty in local mode)",
    )
    hits: list[FoldseekMultimerHit] = Field(default_factory=list, title="Hits", description="Multimer alignment hits")
    num_hits: int = Field(title="Number of Hits", description="Total number of hits", ge=0)
    databases_queried: list[str] = Field(title="Databases Queried", description="Databases included in the search")
    result_url: str = Field(title="Result URL", description="Foldseek result archive URL (remote only)")

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


# ============================================================================
# Tool Implementation
# ============================================================================


# Shared 2-chain HIV-1 protease (1HSG) fixture; foldseek rejects too-short
# multimers.
_EXAMPLE_PDB_PATH = str(Path(__file__).parents[1] / "example_multimer_input_fixture.pdb")


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    pdb_text = Path(_EXAMPLE_PDB_PATH).read_text()
    return FoldseekMultimerSearchInput(structure_text=pdb_text)


@tool(
    key="foldseek-multimer-search",
    label="Foldseek Multimer Search",
    category="structure_alignment",
    input_class=FoldseekMultimerSearchInput,
    config_class=FoldseekMultimerSearchConfig,
    output_class=FoldseekMultimerSearchOutput,
    description="Search Foldseek multimer (complex) structural homology — remote (server) or local (CLI)",
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_foldseek_multimer_search(
    inputs: FoldseekMultimerSearchInput,
    config: FoldseekMultimerSearchConfig,
    instance: Any = None,
) -> FoldseekMultimerSearchOutput:
    """Run a Foldseek multimer (complex) structural search.

    Dispatches to the public Foldseek-Multimer server (remote) or the local
    Foldseek CLI's ``easy-multimersearch`` based on ``config.search_mode``.

    Args:
        inputs (FoldseekMultimerSearchInput): Multi-chain query PDB.
        config (FoldseekMultimerSearchConfig): Search-mode + per-mode options.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        FoldseekMultimerSearchOutput: Multimer alignment hits, plus
            ticket_id / result_url when running in remote mode.
    """
    if config.search_mode == "local":
        return _local_multimer_search(inputs, config, instance=instance)
    return _remote_multimer_search(inputs, config)


def _remote_multimer_search(
    inputs: FoldseekMultimerSearchInput,
    config: FoldseekMultimerSearchConfig,
) -> FoldseekMultimerSearchOutput:
    """Remote multimer search via search.foldseek.com (mode 'complex-{base}')."""
    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
        allowed_methods=["GET", "POST"],
    )
    try:
        # Multimer wire mode is 'complex-{base}' (per MultimerSearch.vue); _submit is shared with single-chain.
        wire_mode = f"complex-{config.mode}"
        ticket_id = _submit(inputs.structure_text, list(config.databases), wire_mode, session)
        poll_until_complete(
            session,
            f"{_FOLDSEEK_BASE}/api/ticket/{ticket_id}",
            poll_interval_seconds=config.poll_interval_seconds,
            timeout_seconds=config.timeout_seconds,
        )
        result_url = f"{_FOLDSEEK_BASE}/api/result/download/{ticket_id}"
        archive_response = session.get(result_url, timeout=_RESULT_DOWNLOAD_TIMEOUT_SECONDS)
        archive_response.raise_for_status()
        hits = _parse_m8_archive(archive_response.content)
        return FoldseekMultimerSearchOutput(
            ticket_id=ticket_id,
            hits=hits,
            num_hits=len(hits),
            databases_queried=list(config.databases),
            result_url=result_url,
        )
    finally:
        session.close()


def _local_multimer_search(
    inputs: FoldseekMultimerSearchInput,
    config: FoldseekMultimerSearchConfig,
    instance: Any = None,
) -> FoldseekMultimerSearchOutput:
    """Local multimer search via `foldseek easy-multimersearch`."""
    if config.local_db is None:
        raise RuntimeError("Internal: local_db is None in local-mode call")
    output_data = ToolInstance.dispatch(
        "foldseek",
        {
            "operation": "easy_multimersearch",
            "structure_text": inputs.structure_text,
            "local_db": config.local_db,
            "evalue": config.evalue,
            "sensitivity": config.sensitivity,
            "max_seqs": config.max_seqs,
            "alignment_type": config.alignment_type,
            "tmscore_threshold": config.tmscore_threshold,
            "lddt_threshold": config.lddt_threshold,
            "num_threads": config.num_threads,
        },
        instance=instance,
        config=config,
    )
    hits = _parse_m8_text(output_data.get("stdout", ""), database=_LOCAL_DB_PSEUDONAME)
    return FoldseekMultimerSearchOutput(
        ticket_id="",
        hits=hits,
        num_hits=len(hits),
        databases_queried=[config.local_db],
        result_url="",
    )
