"""Wraps Foldseek structural search — both the public server and the local CLI.

Remote mode (default): submit a query PDB to https://search.foldseek.com, poll
for completion, download the result archive, return parsed alignment hits
across one or more reference databases (PDB100, AlphaFold DB).

Local mode: dispatch ``foldseek easy-search`` against a user-provided Foldseek
database via the standalone env binary.
"""

import io
import json
import logging
import tarfile
from pathlib import Path
from typing import Any, Literal

import requests
from pydantic import BaseModel, ConfigDict, Field, model_validator

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

_FOLDSEEK_BASE = "https://search.foldseek.com"
_REQUEST_TIMEOUT_SECONDS = 30
_RESULT_DOWNLOAD_TIMEOUT_SECONDS = 120
_HTTP_RETRIES = 2
_BACKOFF_SECONDS = 1.0
_USER_AGENT = "proto-tools/foldseek-search-v1"
_LOCAL_DB_PSEUDONAME = "local_db"  # placeholder for FoldseekHit.database in local mode

FoldseekDatabase = Literal[
    "pdb100",
    "afdb50",
    "afdb-swissprot",
    "afdb-proteome",
    "mgnify_esm30",
    "gmgcl_id",
    "BFVD",
    "cath50",
    "bfmd",
]
FoldseekMode = Literal["3diaa", "tmalign", "lolalign"]
FoldseekSearchMode = Literal["remote", "local"]


# ============================================================================
# Data Models
# ============================================================================


class FoldseekHit(BaseModel):
    """One Foldseek alignment hit (one row from a result M8 file).

    Attributes:
        database (str): Source database the hit came from (e.g. 'pdb100').
        target_id (str): Database-specific target identifier (e.g. '1tup_A',
            'AF-P04637-F1').
        sequence_identity (float): Sequence identity over the aligned region,
            as a fraction in [0, 1].
        alignment_length (int): Length of the aligned region in residues.
        mismatches (int): Number of mismatched columns.
        gap_openings (int): Number of gap-opening events.
        query_start (int): 1-indexed start position in the query.
        query_end (int): 1-indexed end position in the query.
        target_start (int): 1-indexed start position in the target.
        target_end (int): 1-indexed end position in the target.
        evalue (float): Expectation value.
        bit_score (float): Bit score.
    """

    model_config = ConfigDict(extra="forbid")

    database: str = Field(title="Database", description="Source database the hit came from")
    target_id: str = Field(title="Target ID", description="Database-specific target identifier")
    sequence_identity: float = Field(
        title="Sequence Identity",
        description="Sequence identity over aligned region (0-1)",
        ge=0.0,
        le=1.0,
    )
    alignment_length: int = Field(title="Alignment Length", description="Aligned region length in residues", ge=0)
    mismatches: int = Field(title="Mismatches", description="Mismatched columns", ge=0)
    gap_openings: int = Field(title="Gap Openings", description="Gap-opening events", ge=0)
    query_start: int = Field(title="Query Start", description="Query start (1-indexed)")
    query_end: int = Field(title="Query End", description="Query end (1-indexed)")
    target_start: int = Field(title="Target Start", description="Target start (1-indexed)")
    target_end: int = Field(title="Target End", description="Target end (1-indexed)")
    evalue: float = Field(
        title="E-value",
        description="Foldseek e-value; smaller values are more significant",
        ge=0.0,
    )
    bit_score: float = Field(title="Bit Score", description="Foldseek alignment bit score; higher is better")


class FoldseekSearchInput(BaseToolInput):
    """Input for Foldseek structural search.

    Attributes:
        structure_text (str): PDB-format text of the query structure. Use
            ``alphafold-db-fetch`` or ``pdb-fetch-entry`` upstream to obtain it.
    """

    structure_text: str = InputField(title="Structure Text", description="PDB-format text of the query structure")


class FoldseekSearchConfig(BaseConfig):
    """Configuration for Foldseek search (remote or local).

    Attributes:
        search_mode (FoldseekSearchMode): 'remote' (default) hits the public
            Foldseek server; 'local' runs the Foldseek CLI against a local DB.
        databases (list[FoldseekDatabase]): Remote-only — server-hosted
            databases to search.
        mode (FoldseekMode): Remote-only — alignment mode; '3diaa' (default)
            is fast 3Di+AA local; 'tmalign' is global; 'lolalign' is local LoL.
        poll_interval_seconds (float): Remote-only — delay between status polls.
        timeout_seconds (float): Remote-only — max wall-clock time for the search.
        local_db (str | None): Local-only (required) — path to a local Foldseek DB.
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
        description="'remote' (search.foldseek.com) or 'local' (local Foldseek CLI)",
    )
    databases: list[FoldseekDatabase] = ConfigField(
        title="Databases",
        default_factory=lambda: ["pdb100", "afdb50"],
        description="Remote-only — server-hosted reference databases to search",
    )
    mode: FoldseekMode = ConfigField(
        title="Alignment Mode",
        default="3diaa",
        description="Remote-only — '3diaa' (fast 3Di+AA local), 'tmalign' (global), or 'lolalign' (local, slow)",
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
        description="Remote-only — max wall-clock time for the search",
        include_in_key=False,
    )
    local_db: str | None = ConfigField(
        title="Local Foldseek Database",
        default=None,
        description="Path to a local Foldseek DB or a directory of PDB files (required for local mode)",
    )
    evalue: float = ConfigField(
        title="E-value Threshold",
        default=10.0,
        ge=0.0,
        description="E-value cutoff. Lower = stricter (1e-3 for confident homologs; default 10.0 reports all)",
    )
    sensitivity: float = ConfigField(
        title="Sensitivity",
        default=9.5,
        ge=1.0,
        le=9.5,
        description="Prefilter sensitivity (1.0-9.5). Lower = faster, higher = more sensitive (default 9.5)",
    )
    max_seqs: int = ConfigField(
        title="Max Sequences",
        default=1000,
        ge=1,
        description="Max prefilter targets per query. Raise to surface more hits at cost of runtime",
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
        "databases": ["pdb100", "afdb50"],
        "mode": "3diaa",
        "poll_interval_seconds": 5.0,
        "timeout_seconds": 600.0,
    }
    _LOCAL_ONLY_DEFAULTS = {  # noqa: RUF012
        "local_db": None,
        "evalue": 10.0,
        "sensitivity": 9.5,
        "max_seqs": 1000,
        "alignment_type": 2,
        "tmscore_threshold": 0.0,
        "lddt_threshold": 0.0,
        "num_threads": 4,
    }

    @model_validator(mode="after")
    def validate_mode_requirements(self) -> "FoldseekSearchConfig":
        """Hard-error on missing required local fields; soft-warn on cross-mode misuse."""
        if self.search_mode == "local" and not self.local_db:
            raise ValueError("local_db is required when search_mode='local'")

        ignored_table = self._LOCAL_ONLY_DEFAULTS if self.search_mode == "remote" else self._REMOTE_ONLY_DEFAULTS
        kind = "local-only" if self.search_mode == "remote" else "remote-only"
        for name, default in ignored_table.items():
            if getattr(self, name) != default:
                logger.warning("Config field '%s' is %s and is ignored in %s mode.", name, kind, self.search_mode)
        return self


class FoldseekSearchOutput(BaseToolOutput):
    """Output from Foldseek structural search.

    Attributes:
        ticket_id (str): Remote job ticket ID (re-fetchable for ~24h);
            empty in local mode.
        hits (list[FoldseekHit]): All alignment hits across the queried
            databases, in the order Foldseek returned them.
        num_hits (int): ``len(hits)``.
        databases_queried (list[str]): Databases included in this search;
            in local mode contains the single local DB path.
        result_url (str): Remote result-archive URL; empty in local mode.
    """

    ticket_id: str = Field(
        title="Ticket ID",
        description="Foldseek job ticket ID (remote only; empty in local mode)",
    )
    hits: list[FoldseekHit] = Field(default_factory=list, title="Hits", description="All alignment hits")
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


# Shared 65-residue fixture; foldseek rejects too-short structures.
_EXAMPLE_PDB_PATH = str(Path(__file__).parents[1] / "example_input_fixture.pdb")


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    pdb_text = Path(_EXAMPLE_PDB_PATH).read_text()
    return FoldseekSearchInput(structure_text=pdb_text)


@tool(
    key="foldseek-search",
    label="Foldseek Search",
    category="structure_alignment",
    input_class=FoldseekSearchInput,
    config_class=FoldseekSearchConfig,
    output_class=FoldseekSearchOutput,
    description="Search Foldseek structural homology against PDB100/AlphaFold DB (remote) or a local DB (local)",
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_foldseek_search(
    inputs: FoldseekSearchInput,
    config: FoldseekSearchConfig,
    instance: Any = None,
) -> FoldseekSearchOutput:
    """Run a Foldseek structural search.

    Dispatches to the public server (remote) or the local Foldseek CLI based
    on ``config.search_mode``.

    Args:
        inputs (FoldseekSearchInput): Query structure as PDB-format text.
        config (FoldseekSearchConfig): Search-mode + per-mode options.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        FoldseekSearchOutput: Alignment hits, plus ticket_id / result_url
            when running in remote mode.
    """
    if config.search_mode == "local":
        return _local_search(inputs, config, instance=instance)
    return _remote_search(inputs, config)


def _remote_search(inputs: FoldseekSearchInput, config: FoldseekSearchConfig) -> FoldseekSearchOutput:
    """Remote mode — submit, poll, download archive, parse M8 per database."""
    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
        allowed_methods=["GET", "POST"],
    )
    try:
        ticket_id = _submit(inputs.structure_text, list(config.databases), config.mode, session)
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
        return FoldseekSearchOutput(
            ticket_id=ticket_id,
            hits=hits,
            num_hits=len(hits),
            databases_queried=list(config.databases),
            result_url=result_url,
        )
    finally:
        session.close()


def _local_search(
    inputs: FoldseekSearchInput,
    config: FoldseekSearchConfig,
    instance: Any = None,
) -> FoldseekSearchOutput:
    """Local mode — dispatch `foldseek easy-search` via the standalone env."""
    if config.local_db is None:
        raise RuntimeError("Internal: local_db is None in local-mode call")
    output_data = ToolInstance.dispatch(
        "foldseek",
        {
            "operation": "easy_search",
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
    return FoldseekSearchOutput(
        ticket_id="",
        hits=hits,
        num_hits=len(hits),
        databases_queried=[config.local_db],
        result_url="",
    )


# ============================================================================
# Private Helpers
# ============================================================================


def _submit(
    structure_text: str,
    databases: list[str],
    mode: str,
    session: requests.Session,
) -> str:
    """Submit a structure to Foldseek; return the job ticket ID."""
    files = {"q": ("query.pdb", structure_text, "chemical/x-pdb")}
    data = [("database[]", db) for db in databases] + [("mode", mode)]
    response = session.post(
        f"{_FOLDSEEK_BASE}/api/ticket",
        files=files,
        data=data,
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    ticket_id = payload.get("id")
    if not isinstance(ticket_id, str) or not ticket_id:
        raise ValueError(f"Foldseek submit returned no ticket ID: {payload}")
    return ticket_id


def _parse_m8_text(text: str, database: str) -> list[FoldseekHit]:
    """Parse standard 12-column BLAST M8 text into FoldseekHit objects.

    Rows with fewer than 12 columns are silently skipped. Sequence identity in
    column 3 is encoded as a percentage (0-100); normalize to a fraction in
    [0, 1] for downstream consumers.
    """
    hits: list[FoldseekHit] = []
    for line in text.splitlines():
        row = line.split("\t")
        if len(row) < 12:
            continue
        hits.append(
            FoldseekHit(
                database=database,
                target_id=row[1],
                sequence_identity=float(row[2]) / 100.0,
                alignment_length=int(row[3]),
                mismatches=int(row[4]),
                gap_openings=int(row[5]),
                query_start=int(row[6]),
                query_end=int(row[7]),
                target_start=int(row[8]),
                target_end=int(row[9]),
                evalue=float(row[10]),
                bit_score=float(row[11]),
            )
        )
    return hits


def _parse_m8_archive(archive_bytes: bytes) -> list[FoldseekHit]:
    """Extract per-database M8 files from a Foldseek result tarball and parse them.

    The archive contains one file per queried database, named ``alis_{db}.m8``.
    Raises ValueError if the bytes aren't a valid gzipped tar (e.g. the server
    returned an HTML error page).
    """
    hits: list[FoldseekHit] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile() or not member.name.endswith(".m8"):
                    continue
                database = Path(member.name).stem.removeprefix("alis_")
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                text = extracted.read().decode("utf-8", errors="replace")
                hits.extend(_parse_m8_text(text, database))
    except (tarfile.TarError, EOFError) as e:
        raise ValueError(f"Foldseek result is not a valid tar.gz archive (server error?): {e}") from e
    return hits
