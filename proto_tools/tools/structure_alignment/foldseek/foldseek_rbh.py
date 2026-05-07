"""Wraps `foldseek easy-rbh` for reciprocal-best-hit structural search.

Local-only — the public Foldseek server does not expose RBH (verified against
``GET https://search.foldseek.com/api/databases``). RBH returns only mutual
best hits between a query and a target DB, useful for structural orthology
calls. The output schema is the same 12-column M8 produced by
``foldseek-search``, so ``FoldseekHit`` and ``_parse_m8_text`` are reused.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from proto_tools.tools.structure_alignment.foldseek.foldseek_search import (
    _LOCAL_DB_PSEUDONAME,
    FoldseekHit,
    _parse_m8_text,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


class FoldseekRBHInput(BaseToolInput):
    """Input for Foldseek reciprocal-best-hits search.

    Attributes:
        structure_text (str): PDB-format text of the single-chain query
            structure. Use ``alphafold-db-fetch`` or ``pdb-fetch-entry``
            upstream to obtain it.
    """

    structure_text: str = InputField(description="PDB-format text of the query structure")


class FoldseekRBHConfig(BaseConfig):
    """Configuration for Foldseek easy-rbh.

    Attributes:
        local_db (str | None): Path to the target — either a prebuilt Foldseek
            DB (e.g. ``/data/pdb100``) or a directory of PDB files (Foldseek
            auto-builds a temporary DB). Required.
        evalue (float): E-value cutoff (lower = stricter).
        sensitivity (float): Prefilter sensitivity (1.0-9.5; higher = slower
            + more sensitive). Default 4.0 matches foldseek's
            ``setStructureRbhDefaults`` (which, unlike the search workflow,
            does not bump sensitivity to 9.5).
        max_seqs (int): Max prefilter targets per query.
        alignment_type (Literal[0, 1, 2, 3]): Alignment scoring method (0=3Di,
            1=TMalign, 2=3Di+AA, 3=LoL). Note: foldseek's RBH workflow only
            branches on TMalign (1) and 3Di+AA (2); 0 falls through to the
            same alignment branch as 2.
        cov (float): Minimum aligned-residue coverage for an RBH pair (0-1).
            0.0 keeps all.
        cov_mode (Literal[0, 1, 2]): How `cov` is measured: 0=bidirectional,
            1=target-only, 2=query-only.
        tmscore_threshold (float): Keep RBH pairs with TM-score above this
            (0-1). 0.0 keeps all.
        lddt_threshold (float): Keep RBH pairs with LDDT above this (0-1).
            0.0 keeps all.
        num_threads (int): CPU threads.
    """

    local_db: str | None = ConfigField(
        title="Local Foldseek Target",
        default=None,
        description="Path to a local Foldseek target DB or directory of PDBs (auto-createdb)",
    )
    evalue: float = ConfigField(
        title="E-value Threshold",
        default=10.0,
        ge=0.0,
        description="E-value cutoff for inner alignment. Lower = stricter (default 10.0 reports all)",
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
        default=1000,
        ge=1,
        description="Max prefilter targets per query (raise for more candidates)",
    )
    alignment_type: Literal[0, 1, 2, 3] = ConfigField(
        title="Alignment Type",
        default=2,
        description="Alignment scoring: 0=3Di SW, 1=TMalign, 2=3Di+AA (default), 3=LoL. RBH treats 0 same as 2",
        advanced=True,
    )
    cov: float = ConfigField(
        title="Coverage Threshold",
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Min aligned-residue coverage for an RBH pair (0-1). 0.0 keeps all",
        advanced=True,
    )
    cov_mode: Literal[0, 1, 2] = ConfigField(
        title="Coverage Mode",
        default=0,
        description="Coverage mode: 0=bidirectional, 1=target-only, 2=query-only",
        advanced=True,
    )
    tmscore_threshold: float = ConfigField(
        title="TM-score Threshold",
        default=0.0,
        ge=0.0,
        le=1.0,
        description="TM-score floor for RBH pairs (0-1). 0.0 keeps all",
        advanced=True,
    )
    lddt_threshold: float = ConfigField(
        title="LDDT Threshold",
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LDDT floor for RBH pairs (0-1). 0.0 keeps all",
        advanced=True,
    )
    num_threads: int = ConfigField(
        title="Threads",
        default=4,
        ge=1,
        description="CPU threads",
        advanced=True,
        include_in_key=False,
    )

    @model_validator(mode="after")
    def _local_db_required(self) -> "FoldseekRBHConfig":
        """foldseek-rbh has no remote analog; ``local_db`` must always be supplied."""
        if not self.local_db:
            raise ValueError("local_db is required for foldseek-rbh (no remote mode exists)")
        return self


class FoldseekRBHOutput(BaseToolOutput):
    """Output from Foldseek reciprocal-best-hits search.

    Attributes:
        hits (list[FoldseekHit]): Mutual best-hit alignments. Each hit is a
            standard 12-column M8 row, identical schema to ``foldseek-search``.
        num_hits (int): ``len(hits)``.
        target_db (str): The target DB path that was queried.
    """

    hits: list[FoldseekHit] = Field(default_factory=list, description="Reciprocal best-hit alignments")
    num_hits: int = Field(description="Total number of hits", ge=0)
    target_db: str = Field(description="Target DB path that was queried")

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
    return FoldseekRBHInput(structure_text=pdb_text)


@tool(
    key="foldseek-rbh",
    label="Foldseek Reciprocal Best Hits",
    category="structure_alignment",
    input_class=FoldseekRBHInput,
    config_class=FoldseekRBHConfig,
    output_class=FoldseekRBHOutput,
    description="Find reciprocal best-hit structural alignments between a query and a target DB using Foldseek easy-rbh",
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_foldseek_rbh(
    inputs: FoldseekRBHInput,
    config: FoldseekRBHConfig,
    instance: Any = None,
) -> FoldseekRBHOutput:
    """Run a Foldseek reciprocal-best-hits search via the local CLI.

    Args:
        inputs (FoldseekRBHInput): Single-chain query PDB.
        config (FoldseekRBHConfig): Target DB + alignment parameters.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        FoldseekRBHOutput: Mutual best-hit alignments.
    """
    assert config.local_db is not None, "guaranteed by FoldseekRBHConfig._local_db_required"  # noqa: S101
    output_data = ToolInstance.dispatch(
        "foldseek",
        {
            "operation": "easy_rbh",
            "structure_text": inputs.structure_text,
            "local_db": config.local_db,
            "evalue": config.evalue,
            "sensitivity": config.sensitivity,
            "max_seqs": config.max_seqs,
            "alignment_type": config.alignment_type,
            "cov": config.cov,
            "cov_mode": config.cov_mode,
            "tmscore_threshold": config.tmscore_threshold,
            "lddt_threshold": config.lddt_threshold,
            "num_threads": config.num_threads,
        },
        instance=instance,
        config=config,
    )
    hits = _parse_m8_text(output_data.get("stdout", ""), database=_LOCAL_DB_PSEUDONAME)
    return FoldseekRBHOutput(
        hits=hits,
        num_hits=len(hits),
        target_db=config.local_db,
    )
