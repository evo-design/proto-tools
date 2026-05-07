"""Wraps `foldseek easy-cluster` for structural clustering of user-provided structures.

Local-only — no remote analog. The public Foldseek server only does query-vs-DB
search; clustering an arbitrary user set requires the local CLI.
"""

import gzip
import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proto_tools.entities.structures.utils import detect_structure_format
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

# gz variants before plain variants so `.pdb.gz` matches before `.pdb` in `str.endswith`.
_STRUCTURE_EXTENSIONS: tuple[str, ...] = (".pdb.gz", ".cif.gz", ".mmcif.gz", ".pdb", ".cif", ".mmcif")
_FASTA_EXTENSIONS: tuple[str, ...] = (".fasta.gz", ".fa.gz", ".faa.gz", ".fasta", ".fa", ".faa")
_SUPPORTED_EXTENSIONS: tuple[str, ...] = _STRUCTURE_EXTENSIONS + _FASTA_EXTENSIONS


def _detect_input_format(text: str) -> str:
    """Return ``"fasta"``, ``"pdb"``, or ``"cif"`` for an input text string.

    FASTA is identified by a leading ``>`` (after stripping whitespace);
    everything else is delegated to :func:`detect_structure_format`.
    """
    if text.lstrip().startswith(">"):
        return "fasta"
    return detect_structure_format(text)


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

    Inputs may be PDB, mmCIF, or FASTA — format is auto-detected per string.
    A single call must use one mode (all FASTA OR all PDB/mmCIF). FASTA inputs
    go through Foldseek's built-in ProstT5 model to predict 3Di sequences.

    Attributes:
        structures (list[str] | None): PDB, mmCIF, or single-record FASTA text
            strings (≥2; format auto-detected). Mutually exclusive with
            ``structures_dir``.
        structures_dir (str | None): Directory of ``.pdb``/``.cif``/``.mmcif``/
            ``.fasta``/``.fa``/``.faa`` files (incl. ``.gz``; ≥2). Filename
            stems become ``structure_ids``. Mutually exclusive with
            ``structures``.
        structure_ids (list[str] | None): Optional IDs (only with
            ``structures``; default ``'structure_0'``, ``'structure_1'``, ...).
    """

    structures: list[str] | None = InputField(
        default=None,
        description="PDB, mmCIF, or single-record FASTA text strings (≥2). Mutually exclusive with structures_dir.",
        min_length=2,
    )
    structures_dir: str | None = InputField(
        default=None,
        description="Directory of structure (.pdb/.cif) or FASTA (.fasta/.fa) files, incl. .gz; ≥2.",
    )
    structure_ids: list[str] | None = InputField(
        default=None,
        description="Optional IDs (only with `structures`; default: 'structure_0', 'structure_1', ...).",
    )

    @model_validator(mode="before")
    @classmethod
    def _resolve(cls, data: Any) -> Any:
        return _resolve_structures_dir_in_data(data)

    @model_validator(mode="after")
    def _check(self) -> "FoldseekClusterInput":
        _validate_resolved_input(self.structures, self.structure_ids)
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
        prostt5_weights_dir (str | None): Path to ProstT5 model weights for
            FASTA inputs. If None, weights are auto-provisioned under
            ``resolve_weights_dir("foldseek")/prostt5/weights`` on first FASTA
            call (honors ``PROTO_FOLDSEEK_WEIGHTS_DIR`` / ``PROTO_MODEL_CACHE``).
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
    prostt5_weights_dir: str | None = ConfigField(
        title="ProstT5 Weights Directory",
        default=None,
        description="Path to ProstT5 weights for FASTA inputs (auto-provisioned if None).",
        advanced=True,
        include_in_key=False,
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


# Shared 65-residue fixture; foldseek rejects too-short structures.
_EXAMPLE_PDB_PATH = str(Path(__file__).parents[1] / "example_input_fixture.pdb")


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    pdb_text = Path(_EXAMPLE_PDB_PATH).read_text()
    return FoldseekClusterInput(structures=[pdb_text, pdb_text])


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
        inputs (FoldseekClusterInput): Structures (PDB, mmCIF, or FASTA) + optional IDs.
        config (FoldseekClusterConfig): Clustering thresholds + threads.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        FoldseekClusterOutput: Clusters with their representatives + members.
    """
    assert inputs.structures is not None  # noqa: S101 — guaranteed by model validator
    structures = inputs.structures
    ids = inputs.structure_ids or [f"structure_{i}" for i in range(len(structures))]
    formats = [_detect_input_format(s) for s in structures]

    output_data = ToolInstance.dispatch(
        "foldseek",
        {
            "operation": "easy_cluster",
            "structures": structures,
            "structure_ids": ids,
            "structure_formats": formats,
            "min_seq_id": config.min_seq_id,
            "cov": config.cov,
            "cov_mode": config.cov_mode,
            "evalue": config.evalue,
            "alignment_type": config.alignment_type,
            "tmscore_threshold": config.tmscore_threshold,
            "lddt_threshold": config.lddt_threshold,
            "prostt5_weights_dir": config.prostt5_weights_dir,
            "num_threads": config.num_threads,
        },
        instance=instance,
        config=config,
    )
    clusters = _parse_cluster_tsv(output_data.get("clusters_tsv", ""))
    return FoldseekClusterOutput(
        clusters=clusters,
        num_clusters=len(clusters),
        num_structures=len(structures),
    )


# ============================================================================
# Private Helpers
# ============================================================================


def _read_structures_dir(
    dir_path: str, extensions: tuple[str, ...] = _SUPPORTED_EXTENSIONS
) -> tuple[list[str], list[str]]:
    """Enumerate ``extensions``-matching files (sorted by filename); return (texts, stems)."""
    path = Path(dir_path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"structures_dir {dir_path!r} is not an existing directory")

    matches: list[tuple[Path, str]] = []
    for child in sorted(path.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_file():
            continue
        name_lower = child.name.lower()
        for ext in extensions:
            if name_lower.endswith(ext):
                matches.append((child, child.name[: -len(ext)]))
                break

    if len(matches) < 2:
        raise ValueError(
            f"structures_dir {str(path)!r} must contain at least 2 structure files "
            f"(extensions: {extensions}); found {len(matches)}"
        )

    structures, ids = [], []
    for file_path, stem in matches:
        if file_path.name.lower().endswith(".gz"):
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                structures.append(f.read())
        else:
            structures.append(file_path.read_text(encoding="utf-8"))
        ids.append(stem)
    return structures, ids


def _resolve_structures_dir_in_data(data: Any, *, extensions: tuple[str, ...] = _SUPPORTED_EXTENSIONS) -> Any:
    """``mode="before"`` helper: enforce mutex, read ``structures_dir`` into ``structures`` + ``structure_ids``."""
    if not isinstance(data, dict):
        return data
    has_structures = data.get("structures") is not None
    has_dir = data.get("structures_dir") is not None
    if has_structures and has_dir:
        raise ValueError("Provide exactly one of `structures` or `structures_dir`, not both.")
    if not has_dir:
        return data
    if data.get("structure_ids") is not None:
        raise ValueError(
            "`structure_ids` may not be combined with `structures_dir`; IDs are derived from filename stems."
        )
    structures, ids = _read_structures_dir(data["structures_dir"], extensions=extensions)
    return {**data, "structures": structures, "structure_ids": ids, "structures_dir": None}


def _validate_resolved_input(
    structures: list[str] | None,
    structure_ids: list[str] | None,
    *,
    reject_underscore: bool = False,
    reject_fasta: bool = False,
) -> None:
    """``mode="after"`` helper: require structures, validate IDs, enforce uniform format."""
    if structures is None:
        raise ValueError("Provide exactly one of `structures` or `structures_dir`.")

    fasta_count = sum(1 for s in structures if _detect_input_format(s) == "fasta")
    if 0 < fasta_count < len(structures):
        raise ValueError("Cannot mix FASTA with PDB/mmCIF inputs in a single call; all inputs must be the same kind.")
    if reject_fasta and fasta_count:
        raise ValueError("FASTA input is not supported here; pass PDB or mmCIF text/files.")

    if structure_ids is None:
        return
    seen: set[str] = set()
    for sid in structure_ids:
        if not sid or "/" in sid or "\\" in sid or sid in {".", ".."}:
            raise ValueError(f"structure_id {sid!r} is not a safe filename")
        if reject_underscore and "_" in sid:
            raise ValueError(
                f"structure_id {sid!r} contains '_', which collides with Foldseek's "
                f"'{{multimer_id}}_{{chain}}' output schema. Rename to avoid '_'."
            )
        if sid in seen:
            raise ValueError(
                f"structure_id {sid!r} is duplicated; each ID must be unique "
                f"(filename stems collide if a directory contains both `<stem>.pdb` and `<stem>.cif`)"
            )
        seen.add(sid)
    if len(structure_ids) != len(structures):
        raise ValueError(
            f"structure_ids length ({len(structure_ids)}) must match structures length ({len(structures)})"
        )


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
