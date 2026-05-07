"""Wraps FoldMason multiple structure alignment — both the public server and the local CLI.

Remote mode (default): submit a list of PDB structures to
https://search.foldseek.com/api/ticket/foldmason, poll for completion, fetch
the result JSON, and return the AA + 3Di MSAs reconstructed from the per-row
entries plus the Newick guide tree.

Local mode: dispatch ``foldmason easy-msa`` against the standalone-env binary.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

import requests
from pydantic import Field, field_validator, model_validator

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

_FOLDMASON_BASE = "https://search.foldseek.com"
_REQUEST_TIMEOUT_SECONDS = 60
_RESULT_DOWNLOAD_TIMEOUT_SECONDS = 180
_HTTP_RETRIES = 2
_BACKOFF_SECONDS = 1.0
_USER_AGENT = "proto-tools/foldmason-msa-v1"

FoldmasonSearchMode = Literal["remote", "local"]


# ============================================================================
# Data Models
# ============================================================================


class FoldmasonMSAInput(BaseToolInput):
    """Input for FoldMason multiple structure alignment.

    Attributes:
        structures (list[str]): PDB-format text strings to align (≥2).
        structure_ids (list[str] | None): Optional IDs per structure (default:
            ``'structure_0'``, ``'structure_1'``, ...). Length must match
            ``structures``. IDs become the FASTA record headers and Newick
            leaf labels in the output.
    """

    structures: list[str] = InputField(
        description="PDB-format text strings to align (must provide at least 2)",
        min_length=2,
    )
    structure_ids: list[str] | None = InputField(
        default=None,
        description="Optional IDs per structure (default: 'structure_0', 'structure_1', ...)",
    )

    @field_validator("structure_ids")
    @classmethod
    def _ids_are_safe_filenames(cls, ids: list[str] | None) -> list[str] | None:
        """Reject IDs containing path separators or `..` — they're written to disk as `{id}.pdb`."""
        if ids is None:
            return None
        for sid in ids:
            if not sid or "/" in sid or "\\" in sid or sid in {".", ".."}:
                raise ValueError(f"structure_id {sid!r} is not a safe filename")
        return ids

    @model_validator(mode="after")
    def _ids_match_structures_length(self) -> "FoldmasonMSAInput":
        """structure_ids, when supplied, must have the same length as structures."""
        if self.structure_ids is not None and len(self.structure_ids) != len(self.structures):
            raise ValueError(
                f"structure_ids length ({len(self.structure_ids)}) must match structures length ({len(self.structures)})"
            )
        return self


class FoldmasonMSAConfig(BaseConfig):
    """Configuration for FoldMason MSA (remote or local).

    Attributes:
        search_mode (FoldmasonSearchMode): 'remote' (default) hits the public
            FoldMason server; 'local' runs the FoldMason CLI.
        poll_interval_seconds (float): Remote-only — delay between status polls.
        timeout_seconds (float): Remote-only — max wall-clock time.
        gap_open (int): Local-only — gap open cost.
        gap_extend (int): Local-only — gap extension cost.
        refine_iters (int): Local-only — number of alignment-refinement
            iterations. 0 = no refinement.
        precluster (bool): Local-only — pre-cluster structures before MSA
            construction. Recommended for large datasets (>1k structures).
        guide_tree_newick (str | None): Local-only — Newick guide tree to use
            instead of computing one. Leaf labels must match ``structure_ids``.
        num_threads (int): Local-only — CPU threads.
    """

    search_mode: FoldmasonSearchMode = ConfigField(
        title="Search Mode",
        default="remote",
        description="'remote' (search.foldseek.com/foldmason) or 'local' (foldmason easy-msa)",
    )
    poll_interval_seconds: float = ConfigField(
        title="Poll Interval (seconds)",
        default=5.0,
        ge=1.0,
        description="Remote-only — delay between status polls",
        advanced=True,
        include_in_key=False,
        depends_on={"search_mode": ["remote"]},
    )
    timeout_seconds: float = ConfigField(
        title="Timeout (seconds)",
        default=600.0,
        ge=10.0,
        description="Remote-only — max wall-clock time for the alignment",
        advanced=True,
        include_in_key=False,
        depends_on={"search_mode": ["remote"]},
    )
    gap_open: int = ConfigField(
        title="Gap Open Cost",
        default=25,
        ge=0,
        description="Local-only — gap open cost",
        advanced=True,
        depends_on={"search_mode": ["local"]},
        hidden=True,
    )
    gap_extend: int = ConfigField(
        title="Gap Extend Cost",
        default=2,
        ge=0,
        description="Local-only — gap extension cost",
        advanced=True,
        depends_on={"search_mode": ["local"]},
        hidden=True,
    )
    refine_iters: int = ConfigField(
        title="Refine Iterations",
        default=0,
        ge=0,
        description="Local-only — number of alignment-refinement iterations",
        advanced=True,
        depends_on={"search_mode": ["local"]},
        hidden=True,
    )
    precluster: bool = ConfigField(
        title="Pre-cluster",
        default=False,
        description="Local-only — pre-cluster structures before MSA (recommended for large datasets)",
        advanced=True,
        depends_on={"search_mode": ["local"]},
        hidden=True,
    )
    guide_tree_newick: str | None = ConfigField(
        title="Guide Tree (Newick)",
        default=None,
        description="Local-only — Newick guide tree to use instead of computing one; leaf labels must match structure_ids",
        advanced=True,
        depends_on={"search_mode": ["local"]},
        hidden=True,
    )
    num_threads: int = ConfigField(
        title="Threads (local)",
        default=4,
        ge=1,
        description="Local-only — CPU threads",
        advanced=True,
        include_in_key=False,
        depends_on={"search_mode": ["local"]},
        hidden=True,
    )

    _REMOTE_ONLY_DEFAULTS = {  # noqa: RUF012
        "poll_interval_seconds": 5.0,
        "timeout_seconds": 600.0,
    }
    _LOCAL_ONLY_DEFAULTS = {  # noqa: RUF012
        "gap_open": 25,
        "gap_extend": 2,
        "refine_iters": 0,
        "precluster": False,
        "guide_tree_newick": None,
        "num_threads": 4,
    }

    @model_validator(mode="after")
    def validate_mode_requirements(self) -> "FoldmasonMSAConfig":
        """Soft-warn on cross-mode misuse."""
        ignored_table = self._LOCAL_ONLY_DEFAULTS if self.search_mode == "remote" else self._REMOTE_ONLY_DEFAULTS
        kind = "local-only" if self.search_mode == "remote" else "remote-only"
        for name, default in ignored_table.items():
            if getattr(self, name) != default:
                logger.warning("Config field '%s' is %s and is ignored in %s mode.", name, kind, self.search_mode)
        return self


class FoldmasonMSAOutput(BaseToolOutput):
    """Output from FoldMason multiple structure alignment.

    Attributes:
        ticket_id (str): Remote job ticket ID; empty in local mode.
        aa_msa_fasta (str): Amino-acid alphabet MSA in FASTA format.
        three_di_msa_fasta (str): 3Di alphabet MSA in FASTA format.
        newick_tree (str): Newick guide tree.
        num_sequences (int): Number of sequences in the alignment.
        alignment_length (int): Number of MSA columns.
        result_url (str): Remote result-archive URL; empty in local mode.
    """

    ticket_id: str = Field(description="FoldMason job ticket ID (remote only; empty in local mode)")
    aa_msa_fasta: str = Field(description="Amino-acid MSA in FASTA format")
    three_di_msa_fasta: str = Field(description="3Di-alphabet MSA in FASTA format")
    newick_tree: str = Field(description="Newick guide tree")
    num_sequences: int = Field(description="Number of aligned sequences", ge=0)
    alignment_length: int = Field(description="Number of MSA columns", ge=0)
    result_url: str = Field(description="FoldMason result archive URL (remote only)")

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


# Shared 65-residue fixture; foldmason rejects too-short structures.
_EXAMPLE_PDB_PATH = str(Path(__file__).parents[1] / "example_input_fixture.pdb")


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    pdb_text = Path(_EXAMPLE_PDB_PATH).read_text()
    return FoldmasonMSAInput(structures=[pdb_text, pdb_text])


@tool(
    key="foldmason-msa",
    label="FoldMason MSA",
    category="structure_alignment",
    input_class=FoldmasonMSAInput,
    config_class=FoldmasonMSAConfig,
    output_class=FoldmasonMSAOutput,
    description="Multiple structure alignment via FoldMason — remote (server) or local (CLI)",
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_foldmason_msa(
    inputs: FoldmasonMSAInput,
    config: FoldmasonMSAConfig,
    instance: Any = None,
) -> FoldmasonMSAOutput:
    """Run a FoldMason multiple structure alignment.

    Dispatches to the public FoldMason server (remote) or the local FoldMason
    CLI's ``easy-msa`` based on ``config.search_mode``.

    Args:
        inputs (FoldmasonMSAInput): Structures to align + optional IDs.
        config (FoldmasonMSAConfig): Search-mode + per-mode options.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        FoldmasonMSAOutput: AA + 3Di MSAs, Newick tree, and ticket / archive
            URL when running in remote mode.
    """
    if config.search_mode == "local":
        return _local_msa(inputs, config, instance=instance)
    return _remote_msa(inputs, config)


def _remote_msa(inputs: FoldmasonMSAInput, config: FoldmasonMSAConfig) -> FoldmasonMSAOutput:
    """Remote mode — submit, poll, fetch result JSON, build MSAs + tree."""
    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
        allowed_methods=["GET", "POST"],
    )
    try:
        ids = inputs.structure_ids or [f"structure_{i}" for i in range(len(inputs.structures))]
        ticket_id = _submit_foldmason(inputs.structures, ids, session)
        poll_until_complete(
            session,
            f"{_FOLDMASON_BASE}/api/ticket/{ticket_id}",
            poll_interval_seconds=config.poll_interval_seconds,
            timeout_seconds=config.timeout_seconds,
        )
        result_url = f"{_FOLDMASON_BASE}/api/result/foldmason/{ticket_id}"
        result_response = session.get(result_url, timeout=_RESULT_DOWNLOAD_TIMEOUT_SECONDS)
        result_response.raise_for_status()
        aa_fasta, three_di_fasta, newick, num_sequences, alignment_length = _parse_msa_response_json(
            result_response.json()
        )
        return FoldmasonMSAOutput(
            ticket_id=ticket_id,
            aa_msa_fasta=aa_fasta,
            three_di_msa_fasta=three_di_fasta,
            newick_tree=newick,
            num_sequences=num_sequences,
            alignment_length=alignment_length,
            result_url=result_url,
        )
    finally:
        session.close()


def _local_msa(
    inputs: FoldmasonMSAInput,
    config: FoldmasonMSAConfig,
    instance: Any = None,
) -> FoldmasonMSAOutput:
    """Local mode — dispatch `foldmason easy-msa` via the standalone env."""
    ids = inputs.structure_ids or [f"structure_{i}" for i in range(len(inputs.structures))]
    output_data = ToolInstance.dispatch(
        "foldmason",
        {
            "operation": "easy_msa",
            "structures": inputs.structures,
            "structure_ids": ids,
            "gap_open": config.gap_open,
            "gap_extend": config.gap_extend,
            "refine_iters": config.refine_iters,
            "precluster": config.precluster,
            "guide_tree_newick": config.guide_tree_newick,
            "num_threads": config.num_threads,
        },
        instance=instance,
        config=config,
    )
    aa_fasta = output_data["aa_msa_fasta"]
    num_sequences, alignment_length = _msa_dimensions(aa_fasta)
    return FoldmasonMSAOutput(
        ticket_id="",
        aa_msa_fasta=aa_fasta,
        three_di_msa_fasta=output_data["three_di_msa_fasta"],
        newick_tree=output_data["newick_tree"],
        num_sequences=num_sequences,
        alignment_length=alignment_length,
        result_url="",
    )


# ============================================================================
# Private Helpers
# ============================================================================


def _submit_foldmason(
    structures: list[str],
    structure_ids: list[str],
    session: requests.Session,
) -> str:
    """Submit a list of structures to FoldMason; return the job ticket ID.

    Multipart wire format (per ``FoldMasonSearch.vue``): one ``fileNames[]`` +
    one ``queries[]`` per structure. Filenames omit the ``.pdb`` extension so
    the server preserves user-supplied IDs verbatim as entry names.
    """
    files: list[tuple[str, tuple[Any, str | bytes, str]]] = []
    for sid, text in zip(structure_ids, structures, strict=True):
        files.append(("fileNames[]", (None, sid, "text/plain")))
        files.append(("queries[]", (sid, text, "chemical/x-pdb")))
    response = session.post(
        f"{_FOLDMASON_BASE}/api/ticket/foldmason",
        files=files,
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    ticket_id = payload.get("id")
    if not isinstance(ticket_id, str) or not ticket_id:
        raise ValueError(f"FoldMason submit returned no ticket ID: {payload}")
    return ticket_id


def _parse_msa_response_json(payload: dict[str, Any]) -> tuple[str, str, str, int, int]:
    """Build AA FASTA, 3Di FASTA, Newick tree, and dimensions from the server JSON.

    Returns:
        tuple[str, str, str, int, int]: ``(aa_fasta, three_di_fasta, newick, num_sequences, alignment_length)``.
    """
    entries = payload["entries"]
    aa_fasta = "".join(f">{e['name']}\n{e['aa']}\n" for e in entries)
    three_di_fasta = "".join(f">{e['name']}\n{e['ss']}\n" for e in entries)
    return aa_fasta, three_di_fasta, payload["tree"], len(entries), len(entries[0]["aa"]) if entries else 0


def _msa_dimensions(aa_fasta: str) -> tuple[int, int]:
    """Return ``(num_sequences, alignment_length)`` from an AA-FASTA MSA."""
    sequence_chunks: list[list[str]] = []
    for line in aa_fasta.splitlines():
        if line.startswith(">"):
            sequence_chunks.append([])
        elif sequence_chunks:
            sequence_chunks[-1].append(line.strip())
    if not sequence_chunks:
        return (0, 0)
    return (len(sequence_chunks), sum(len(s) for s in sequence_chunks[0]))
