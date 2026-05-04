"""Wraps `foldseek easy-multimercluster` for structural clustering of multi-chain assemblies.

Local-only — no remote analog. The public Foldseek server only does query-vs-DB
search; clustering an arbitrary user set of multimers requires the local CLI.
"""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from proto_tools.tools.structure_alignment.foldseek.foldseek_cluster import (
    FoldseekCluster,
    _parse_cluster_tsv,
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


class FoldseekMultimerClusterInput(BaseToolInput):
    """Input for Foldseek multimer (complex) structural clustering.

    Foldseek emits cluster member IDs as ``{multimer_id}_{chain}``. To keep
    those IDs round-trippable, ``structure_ids`` must not contain ``_`` — chain
    IDs in PDBs are single ASCII characters (column 22) and cannot contain
    ``_``, so the only place ``_`` can leak in is the user-supplied multimer ID.
    The wrapper does not pre-validate this; member IDs in the output reflect
    whatever Foldseek produced.

    Attributes:
        structures (list[str]): Multi-chain PDB-format text strings to cluster
            (≥2).
        structure_ids (list[str] | None): Optional IDs per structure (default:
            ``'multimer_0'``, ``'multimer_1'``, ...). Length must match
            ``structures``. See note above re: ``_`` in IDs.
    """

    structures: list[str] = InputField(
        description="Multi-chain PDB-format text strings to cluster (must provide at least 2)",
        min_length=2,
    )
    structure_ids: list[str] | None = InputField(
        default=None,
        description="Optional IDs per structure (default: 'multimer_0', 'multimer_1', ...)",
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
    def _ids_match_structures_length(self) -> "FoldseekMultimerClusterInput":
        """structure_ids, when supplied, must have the same length as structures."""
        if self.structure_ids is not None and len(self.structure_ids) != len(self.structures):
            raise ValueError(
                f"structure_ids length ({len(self.structure_ids)}) must match structures length ({len(self.structures)})"
            )
        return self


class FoldseekMultimerClusterConfig(BaseConfig):
    """Configuration for Foldseek easy-multimercluster.

    Attributes:
        multimer_tm_threshold (float): Multimer-level TM-score threshold (0-1).
            Two multimers cluster only if their multimer TM-score exceeds this.
        chain_tm_threshold (float): Per-chain TM-score threshold (0-1) used to
            filter chain-pair alignments before assembling the multimer score.
        interface_lddt_threshold (float): Interface lDDT threshold (0-1) for
            chain-pair alignments.
        num_threads (int): CPU threads.
    """

    multimer_tm_threshold: float = ConfigField(
        title="Multimer TM-score Threshold",
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Multimer-level TM-score threshold (0-1)",
    )
    chain_tm_threshold: float = ConfigField(
        title="Chain TM-score Threshold",
        default=0.001,
        ge=0.0,
        le=1.0,
        description="Per-chain TM-score threshold (0-1) for chain-pair filtering",
    )
    interface_lddt_threshold: float = ConfigField(
        title="Interface lDDT Threshold",
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Interface lDDT threshold (0-1) for chain-pair alignments",
    )
    num_threads: int = ConfigField(
        title="Threads",
        default=4,
        ge=1,
        description="CPU threads",
        advanced=True,
        include_in_key=False,
    )


class FoldseekMultimerClusterOutput(BaseToolOutput):
    """Output from Foldseek multimer structural clustering.

    Attributes:
        clusters (list[FoldseekCluster]): One entry per cluster, each holding
            a representative multimer and its members. Member IDs may include
            ``{multimer_id}_{chain}`` suffixes per Foldseek's chain-aware schema.
        num_clusters (int): ``len(clusters)``.
        num_multimers (int): Total number of input multimers clustered.
        rep_seq_fasta (str): Representative-multimer FASTA produced by Foldseek
            (with ``#multimer_id`` group separators between chains).
    """

    clusters: list[FoldseekCluster] = Field(default_factory=list, description="One entry per cluster")
    num_clusters: int = Field(description="Total number of clusters", ge=0)
    num_multimers: int = Field(description="Total number of input multimers", ge=0)
    rep_seq_fasta: str = Field(description="Representative-multimer FASTA from Foldseek")

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
HEADER    EXAMPLE COMPLEX
ATOM      1  CA  MET A   1      11.000  13.000  10.000  1.00 20.00           C
ATOM      2  CA  ALA A   2      12.000  14.000  11.000  1.00 20.00           C
ATOM      3  CA  MET B   1      21.000  23.000  20.000  1.00 20.00           C
ATOM      4  CA  ALA B   2      22.000  24.000  21.000  1.00 20.00           C
END
"""


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return FoldseekMultimerClusterInput(structures=[_EXAMPLE_PDB, _EXAMPLE_PDB])


@tool(
    key="foldseek-multimercluster",
    label="Foldseek Multimer Cluster",
    category="structure_alignment",
    input_class=FoldseekMultimerClusterInput,
    config_class=FoldseekMultimerClusterConfig,
    output_class=FoldseekMultimerClusterOutput,
    description="Cluster a set of protein complexes by multimer-level structural similarity using Foldseek easy-multimercluster",
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_foldseek_multimercluster(
    inputs: FoldseekMultimerClusterInput,
    config: FoldseekMultimerClusterConfig,
    instance: Any = None,
) -> FoldseekMultimerClusterOutput:
    """Cluster multimers with Foldseek easy-multimercluster.

    Args:
        inputs (FoldseekMultimerClusterInput): Multi-chain PDB structures + optional IDs.
        config (FoldseekMultimerClusterConfig): Multimer/chain/interface thresholds + threads.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        FoldseekMultimerClusterOutput: Clusters with representatives + members,
            plus the representative-multimer FASTA.
    """
    ids = inputs.structure_ids or [f"multimer_{i}" for i in range(len(inputs.structures))]

    output_data = ToolInstance.dispatch(
        "foldseek",
        {
            "operation": "easy_multimercluster",
            "structures": inputs.structures,
            "structure_ids": ids,
            "multimer_tm_threshold": config.multimer_tm_threshold,
            "chain_tm_threshold": config.chain_tm_threshold,
            "interface_lddt_threshold": config.interface_lddt_threshold,
            "num_threads": config.num_threads,
        },
        instance=instance,
        config=config,
    )
    clusters = _parse_cluster_tsv(output_data.get("clusters_tsv", ""))
    return FoldseekMultimerClusterOutput(
        clusters=clusters,
        num_clusters=len(clusters),
        num_multimers=len(inputs.structures),
        rep_seq_fasta=output_data.get("rep_seq_fasta", ""),
    )
