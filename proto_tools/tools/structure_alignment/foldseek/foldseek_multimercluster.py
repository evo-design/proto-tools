"""Wraps `foldseek easy-multimercluster` for structural clustering of multi-chain assemblies.

Local-only — no remote analog. The public Foldseek server only does query-vs-DB
search; clustering an arbitrary user set of multimers requires the local CLI.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from proto_tools.entities import Structure
from proto_tools.entities.structures.utils import detect_structure_format
from proto_tools.tools.structure_alignment.foldseek.foldseek_cluster import (
    _STRUCTURE_EXTENSIONS,
    FoldseekCluster,
    _coerce_structure_items_to_text,
    _parse_cluster_tsv,
    _resolve_directory_structures,
    _validate_resolved_input,
)
from proto_tools.tools.structure_alignment.foldseek.foldseek_search import _require_linux_x86_64_for_gpu
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

    IDs must not contain ``_``: Foldseek emits cluster member IDs as
    ``{multimer_id}_{chain}``, so silent ID mangling is rejected up front
    (whether user-supplied or filename-derived).

    Attributes:
        structures (list[Structure | str] | str | Path | None): Multi-chain items
            to cluster (≥2) — a list of Structure objects / file paths / PDB·mmCIF
            text, or a directory path (filename stems become ``structure_ids``).
        structure_ids (list[str] | None): Optional IDs for the list form (default
            ``multimer-0``, ...); derived from filename stems for a directory. No ``_``.
    """

    structures: list[Structure | str] | str | Path | None = InputField(
        default=None,
        title="Structures",
        description="Multimers to cluster (≥2): a list of Structure/path/PDB/mmCIF-text items, or a directory path",
        min_length=2,
    )
    structure_ids: list[str] | None = InputField(
        default=None,
        title="Structure IDs",
        description="Optional IDs (only with the list form; default: 'multimer-0', ...). Must not contain '_'.",
    )

    @model_validator(mode="before")
    @classmethod
    def _resolve(cls, data: Any) -> Any:
        data = _coerce_structure_items_to_text(data)
        return _resolve_directory_structures(data, extensions=_STRUCTURE_EXTENSIONS)

    @model_validator(mode="after")
    def _check(self) -> "FoldseekMultimerClusterInput":
        _validate_resolved_input(
            self.structures,
            self.structure_ids,
            reject_underscore=True,
            reject_fasta=True,
        )
        return self


class FoldseekMultimerClusterConfig(BaseConfig):
    """Configuration for Foldseek easy-multimercluster.

    Attributes:
        multimer_tm_threshold (float): Maps to ``--multimer-tm-threshold``.
            Multimer-level TM-score (0-1) above which two multimers cluster
            together.
        chain_tm_threshold (float): Maps to ``--chain-tm-threshold``. Per-chain
            TM-score (0-1) used to filter chain-pair alignments before
            assembling the multimer score.
        interface_lddt_threshold (float): Maps to ``--interface-lddt-threshold``.
            Interface lDDT (0-1) for chain-pair alignments.
        alignment_type (Literal[0, 1, 2, 3]): Alignment scoring method (0=3Di,
            1=TMalign, 2=3Di+AA, 3=LoL).
        tmscore_threshold (float): Keep chain-pair alignments with TM-score
            above this (0-1). 0.0 keeps all.
        lddt_threshold (float): Keep chain-pair alignments with LDDT above
            this (0-1). 0.0 keeps all.
        num_threads (int): CPU threads.
        use_gpu (bool): Run with --gpu 1 on a Linux x86_64 NVIDIA GPU host (driver >= 525.60.13).
    """

    multimer_tm_threshold: float = ConfigField(
        title="Multimer TM-score Threshold",
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Multimer TM-score (0-1) for cluster merging; lower = more permissive",
    )
    chain_tm_threshold: float = ConfigField(
        title="Chain TM-score Threshold",
        default=0.001,
        ge=0.0,
        le=1.0,
        description="Per-chain TM-score (0-1) above which chain-pair alignments contribute to clustering",
    )
    interface_lddt_threshold: float = ConfigField(
        title="Interface lDDT Threshold",
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Interface lDDT (0-1); lower = more permissive cluster merging",
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
        description="TM-score floor for chain-pair alignments (0-1). 0.0 keeps all",
    )
    lddt_threshold: float = ConfigField(
        title="LDDT Threshold",
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LDDT floor for chain-pair alignments (0-1). 0.0 keeps all",
    )
    num_threads: int = ConfigField(title="Threads", default=4, ge=1, description="CPU threads", include_in_key=False)
    use_gpu: bool = ConfigField(
        title="Use GPU",
        default=False,
        description="Run `--gpu 1` on a Linux x86_64 NVIDIA GPU host (driver >= 525.60.13); CPU otherwise.",
    )

    @model_validator(mode="after")
    def _validate_use_gpu(self) -> "FoldseekMultimerClusterConfig":
        """Reject use_gpu on hosts without the Linux x86_64 GPU build."""
        _require_linux_x86_64_for_gpu(self.use_gpu)
        return self

    @property
    def gpus_per_instance(self) -> int:
        """Number of GPUs the configured run uses (1 if GPU, else 0)."""
        return 1 if self.use_gpu else 0


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

    clusters: list[FoldseekCluster] = Field(default_factory=list, title="Clusters", description="One entry per cluster")
    num_clusters: int = Field(title="Number of Clusters", description="Total number of clusters", ge=0)
    num_multimers: int = Field(title="Number of Multimers", description="Total number of input multimers", ge=0)
    rep_seq_fasta: str = Field(title="Representative FASTA", description="Representative-multimer FASTA from Foldseek")

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


# Shared 2-chain HIV-1 protease (1HSG) fixture; foldseek rejects too-short multimers.
_EXAMPLE_PDB_PATH = str(Path(__file__).parents[1] / "example_multimer_input_fixture.pdb")


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    pdb_text = Path(_EXAMPLE_PDB_PATH).read_text()
    return FoldseekMultimerClusterInput(structures=[pdb_text, pdb_text])


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
        inputs (FoldseekMultimerClusterInput): Multi-chain structures
            (PDB or mmCIF) + optional IDs.
        config (FoldseekMultimerClusterConfig): Multimer/chain/interface
            thresholds + threads.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        FoldseekMultimerClusterOutput: Clusters with representatives + members,
            plus the representative-multimer FASTA.
    """
    assert isinstance(inputs.structures, list)  # noqa: S101 — directory paths are resolved to a list by the validator
    structures = [s if isinstance(s, str) else s.structure_pdb for s in inputs.structures]
    ids = inputs.structure_ids or [f"multimer-{i}" for i in range(len(structures))]
    formats = [detect_structure_format(s) for s in structures]

    output_data = ToolInstance.dispatch(
        "foldseek",
        {
            "operation": "easy_multimercluster",
            "structures": structures,
            "structure_ids": ids,
            "structure_formats": formats,
            "multimer_tm_threshold": config.multimer_tm_threshold,
            "chain_tm_threshold": config.chain_tm_threshold,
            "interface_lddt_threshold": config.interface_lddt_threshold,
            "alignment_type": config.alignment_type,
            "tmscore_threshold": config.tmscore_threshold,
            "lddt_threshold": config.lddt_threshold,
            "num_threads": config.num_threads,
            "use_gpu": config.use_gpu,
            "device": "cuda" if config.use_gpu else "cpu",
        },
        instance=instance,
        config=config,
    )
    clusters = _parse_cluster_tsv(output_data.get("clusters_tsv", ""))
    return FoldseekMultimerClusterOutput(
        clusters=clusters,
        num_clusters=len(clusters),
        num_multimers=len(structures),
        rep_seq_fasta=output_data.get("rep_seq_fasta", ""),
    )
