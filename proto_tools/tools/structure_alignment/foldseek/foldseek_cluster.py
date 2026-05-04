"""Wraps `foldseek easy-cluster` for structural clustering of user-provided structures.

Local-only — no remote analog. The public Foldseek server only does query-vs-DB
search; clustering an arbitrary user set requires the local CLI.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class FoldseekCluster(BaseModel):
    """One Foldseek structural cluster.

    Attributes:
        representative_id (str): ID of the cluster representative.
        member_ids (list[str]): IDs of all members (includes the representative).
    """

    model_config = ConfigDict(extra="forbid")

    representative_id: str = Field(description="ID of the cluster representative")
    member_ids: list[str] = Field(description="IDs of all members (includes the representative)")


class FoldseekClusterInput(BaseToolInput):
    """Input for Foldseek structural clustering.

    Attributes:
        structures (list[str]): PDB-format text strings to cluster (≥2).
        structure_ids (list[str] | None): Optional IDs per structure (default:
            ``'structure_0'``, ``'structure_1'``, ...). Length must match
            ``structures``.
    """

    structures: list[str] = InputField(
        description="PDB-format text strings to cluster (must provide at least 2)",
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
    def _ids_match_structures_length(self) -> "FoldseekClusterInput":
        """structure_ids, when supplied, must have the same length as structures."""
        if self.structure_ids is not None and len(self.structure_ids) != len(self.structures):
            raise ValueError(
                f"structure_ids length ({len(self.structure_ids)}) must match structures length ({len(self.structures)})"
            )
        return self


class FoldseekClusterConfig(BaseConfig):
    """Configuration for Foldseek easy-cluster.

    Attributes:
        min_seq_id (float): Sequence-identity threshold (0-1). Default 0.0
            because Foldseek clusters by 3Di structural similarity, not seq id.
        cov (float): Coverage threshold (0-1) for the alignment.
        cov_mode (Literal[0, 1, 2]): Foldseek coverage mode (0: bidirectional,
            1: target, 2: query).
        evalue (float): E-value cutoff for cluster-membership alignments
            (lower = stricter; default 0.01 matches the foldseek cluster
            workflow's runtime default).
        alignment_type (Literal[0, 1, 2, 3]): Alignment scoring method (0=3Di,
            1=TMalign, 2=3Di+AA, 3=LoL).
        tmscore_threshold (float): Keep cluster-membership alignments with
            TM-score above this (0-1). 0.0 keeps all.
        lddt_threshold (float): Keep cluster-membership alignments with LDDT
            above this (0-1). 0.0 keeps all.
        num_threads (int): CPU threads.
    """

    min_seq_id: float = ConfigField(
        title="Min Sequence Identity",
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sequence-identity floor (0-1). 0.0 lets 3Di structural similarity dominate",
        advanced=True,
    )
    cov: float = ConfigField(
        title="Coverage Threshold",
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Min aligned-residue coverage for cluster membership (0-1)",
    )
    cov_mode: Literal[0, 1, 2] = ConfigField(
        title="Coverage Mode",
        default=0,
        description="Coverage mode: 0=bidirectional, 1=target-only, 2=query-only",
    )
    evalue: float = ConfigField(
        title="E-value Threshold",
        default=0.01,
        ge=0.0,
        description="E-value cutoff for cluster-membership alignments. Lower = stricter (default 0.01)",
        advanced=True,
    )
    alignment_type: Literal[0, 1, 2, 3] = ConfigField(
        title="Alignment Type",
        default=2,
        description="Alignment scoring: 0=3Di SW, 1=TMalign, 2=3Di+AA (default), 3=LoL",
        advanced=True,
    )
    tmscore_threshold: float = ConfigField(
        title="TM-score Threshold",
        default=0.0,
        ge=0.0,
        le=1.0,
        description="TM-score floor for cluster-membership alignments (0-1). 0.0 keeps all",
        advanced=True,
    )
    lddt_threshold: float = ConfigField(
        title="LDDT Threshold",
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LDDT floor for cluster-membership alignments (0-1). 0.0 keeps all",
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


class FoldseekClusterOutput(BaseToolOutput):
    """Output from Foldseek structural clustering.

    Attributes:
        clusters (list[FoldseekCluster]): One entry per cluster, each holding
            a representative and its members.
        num_clusters (int): ``len(clusters)``.
        num_structures (int): Total number of input structures clustered.
    """

    clusters: list[FoldseekCluster] = Field(default_factory=list, description="One entry per cluster")
    num_clusters: int = Field(description="Total number of clusters", ge=0)
    num_structures: int = Field(description="Total number of input structures", ge=0)

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


_EXAMPLE_PDB = """\
HEADER    EXAMPLE STRUCTURE
ATOM      1  N   MET A   1      11.104  13.207  10.300  1.00 20.00           N
ATOM      2  CA  MET A   1      11.804  14.247  11.040  1.00 20.00           C
ATOM      3  C   MET A   1      13.304  14.011  10.940  1.00 20.00           C
ATOM      4  O   MET A   1      13.804  13.001  10.440  1.00 20.00           O
END
"""


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return FoldseekClusterInput(structures=[_EXAMPLE_PDB, _EXAMPLE_PDB])


@tool(
    key="foldseek-cluster",
    label="Foldseek Cluster",
    category="structure_alignment",
    input_class=FoldseekClusterInput,
    config_class=FoldseekClusterConfig,
    output_class=FoldseekClusterOutput,
    description="Cluster a set of protein structures by structural similarity using Foldseek easy-cluster",
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_foldseek_cluster(
    inputs: FoldseekClusterInput,
    config: FoldseekClusterConfig,
    instance: Any = None,
) -> FoldseekClusterOutput:
    """Cluster structures with Foldseek easy-cluster.

    Args:
        inputs (FoldseekClusterInput): PDB structures + optional IDs.
        config (FoldseekClusterConfig): Clustering thresholds + threads.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        FoldseekClusterOutput: Clusters with their representatives + members.
    """
    ids = inputs.structure_ids or [f"structure_{i}" for i in range(len(inputs.structures))]

    output_data = ToolInstance.dispatch(
        "foldseek",
        {
            "operation": "easy_cluster",
            "structures": inputs.structures,
            "structure_ids": ids,
            "min_seq_id": config.min_seq_id,
            "cov": config.cov,
            "cov_mode": config.cov_mode,
            "evalue": config.evalue,
            "alignment_type": config.alignment_type,
            "tmscore_threshold": config.tmscore_threshold,
            "lddt_threshold": config.lddt_threshold,
            "num_threads": config.num_threads,
        },
        instance=instance,
        config=config,
    )
    clusters = _parse_cluster_tsv(output_data.get("clusters_tsv", ""))
    return FoldseekClusterOutput(
        clusters=clusters,
        num_clusters=len(clusters),
        num_structures=len(inputs.structures),
    )


# ============================================================================
# Private Helpers
# ============================================================================


def _parse_cluster_tsv(tsv_text: str) -> list[FoldseekCluster]:
    """Parse Foldseek's 2-column cluster TSV (representative<TAB>member) into FoldseekCluster objects.

    Each line is one (representative, member) pair; representatives appear as
    members of their own clusters. Empty lines are skipped.
    """
    members_by_rep: dict[str, list[str]] = {}
    for line in tsv_text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        rep, member = parts[0], parts[1]
        members_by_rep.setdefault(rep, []).append(member)
    return [FoldseekCluster(representative_id=rep, member_ids=members) for rep, members in members_by_rep.items()]
