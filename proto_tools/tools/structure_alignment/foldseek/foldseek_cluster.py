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

from proto_tools.entities import Structure
from proto_tools.entities.structures.utils import detect_structure_format
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

    representative_id: str = Field(title="Representative ID", description="ID of the cluster representative")
    member_ids: list[str] = Field(title="Member IDs", description="IDs of all members (includes the representative)")


class FoldseekClusterInput(BaseToolInput):
    """Input for Foldseek structural clustering.

    Inputs may be PDB, mmCIF, or FASTA — format is auto-detected per string.
    A single call must use one mode (all FASTA OR all PDB/mmCIF). FASTA inputs
    go through Foldseek's built-in ProstT5 model to predict 3Di sequences.

    Attributes:
        structures (list[Structure | str] | str | Path | None): Items to cluster
            (≥2) — a list of Structure objects / file paths / PDB·mmCIF·FASTA text,
            or a directory path (filename stems become ``structure_ids``).
        structure_ids (list[str] | None): Optional IDs for the list form (default
            ``structure_0``, ...); derived from filename stems for a directory.
    """

    structures: list[Structure | str] | str | Path | None = InputField(
        default=None,
        title="Structures",
        description="Items to cluster (≥2): a list of Structure/path/PDB/mmCIF/FASTA-text items, or a directory path",
        min_length=2,
    )
    structure_ids: list[str] | None = InputField(
        default=None,
        title="Structure IDs",
        description="Optional IDs (only with the list form; default: 'structure_0', 'structure_1', ...).",
    )

    @model_validator(mode="before")
    @classmethod
    def _resolve(cls, data: Any) -> Any:
        data = _coerce_structure_items_to_text(data)
        return _resolve_directory_structures(data)

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
        use_gpu (bool): Run with --gpu 1 on a Linux x86_64 NVIDIA GPU host (driver >= 525.60.13).
    """

    min_seq_id: float = ConfigField(
        title="Min Sequence Identity",
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sequence-identity floor (0-1). 0.0 lets 3Di structural similarity dominate",
    )
    cov: float = ConfigField(
        title="Coverage Threshold",
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Min aligned-residue coverage for cluster membership (0-1)",
    )
    cov_mode: Literal[0, 1, 2] = ConfigField(
        title="Coverage Mode", default=0, description="Coverage mode: 0=bidirectional, 1=target-only, 2=query-only"
    )
    evalue: float = ConfigField(
        title="E-value Threshold",
        default=0.01,
        ge=0.0,
        description="E-value cutoff for cluster-membership alignments. Lower = stricter (default 0.01)",
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
        description="TM-score floor for cluster-membership alignments (0-1). 0.0 keeps all",
    )
    lddt_threshold: float = ConfigField(
        title="LDDT Threshold",
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LDDT floor for cluster-membership alignments (0-1). 0.0 keeps all",
    )
    prostt5_weights_dir: str | None = ConfigField(
        title="ProstT5 Weights Directory",
        default=None,
        description="Path to ProstT5 weights for FASTA inputs (auto-provisioned if None).",
        include_in_key=False,
    )
    num_threads: int = ConfigField(title="Threads", default=4, ge=1, description="CPU threads", include_in_key=False)
    use_gpu: bool = ConfigField(
        title="Use GPU",
        default=False,
        description="Run `--gpu 1` on a Linux x86_64 NVIDIA GPU host (driver >= 525.60.13); CPU otherwise.",
    )

    @model_validator(mode="after")
    def _validate_use_gpu(self) -> "FoldseekClusterConfig":
        """Reject use_gpu on hosts without the Linux x86_64 GPU build."""
        _require_linux_x86_64_for_gpu(self.use_gpu)
        return self

    @property
    def gpus_per_instance(self) -> int:
        """Number of GPUs the configured run uses (1 if GPU, else 0)."""
        return 1 if self.use_gpu else 0


class FoldseekClusterOutput(BaseToolOutput):
    """Output from Foldseek structural clustering.

    Attributes:
        clusters (list[FoldseekCluster]): One entry per cluster, each holding
            a representative and its members.
        num_clusters (int): ``len(clusters)``.
        num_structures (int): Total number of input structures clustered.
    """

    clusters: list[FoldseekCluster] = Field(default_factory=list, title="Clusters", description="One entry per cluster")
    num_clusters: int = Field(title="Number of Clusters", description="Total number of clusters", ge=0)
    num_structures: int = Field(title="Number of Structures", description="Total number of input structures", ge=0)

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
    assert isinstance(inputs.structures, list)  # noqa: S101 — directory paths are resolved to a list by the validator
    structures = [s if isinstance(s, str) else s.structure_pdb for s in inputs.structures]
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
            "use_gpu": config.use_gpu,
            "device": "cuda" if config.use_gpu else "cpu",
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
        raise ValueError(f"structures path {dir_path!r} is not an existing directory")

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
            f"structures directory {str(path)!r} must contain at least 2 structure files "
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


def _coerce_structure_items_to_text(data: Any) -> Any:
    """``mode="before"`` helper: per-item, coerce ``Structure`` / ``Path`` entries in ``structures`` to text.

    Structure objects emit their PDB representation. Paths to FASTA files are read
    as text (so FASTA's leading-``>`` detection still works downstream); paths to
    structure files load via :func:`Structure.from_file` and emit PDB. Strings
    pass through unchanged (format auto-detected per-string at standalone dispatch).
    """
    if not isinstance(data, dict):
        return data
    items = data.get("structures")
    if not isinstance(items, list):
        return data
    coerced: list[Any] = []
    for item in items:
        if isinstance(item, Structure):
            coerced.append(item.structure_pdb)
        elif isinstance(item, Path):
            lower = item.name.lower()
            if any(lower.endswith(ext) for ext in _FASTA_EXTENSIONS):
                # Read FASTA (and gz) text verbatim so the leading-`>` check downstream still classifies it.
                if lower.endswith(".gz"):
                    with gzip.open(item, "rt") as f:
                        coerced.append(f.read())
                else:
                    coerced.append(item.read_text())
            else:
                coerced.append(Structure.from_file(item).structure_pdb)
        else:
            coerced.append(item)
    return {**data, "structures": coerced}


def _resolve_directory_structures(data: Any, *, extensions: tuple[str, ...] = _SUPPORTED_EXTENSIONS) -> Any:
    """``mode="before"`` helper: read a directory-path ``structures`` (``str`` / ``Path``) into ``structures`` + ``structure_ids`` (filename stems); a list / ``None`` is left untouched."""
    if not isinstance(data, dict):
        return data
    value = data.get("structures")
    if not isinstance(value, (str, Path)):  # list / None → nothing to resolve
        return data
    if data.get("structure_ids") is not None:
        raise ValueError(
            "`structure_ids` may not be combined with a directory path; IDs are derived from filename stems."
        )
    structures, ids = _read_structures_dir(str(value), extensions=extensions)
    return {**data, "structures": structures, "structure_ids": ids}


def _validate_resolved_input(
    structures: list[Structure | str] | str | Path | None,
    structure_ids: list[str] | None,
    *,
    reject_underscore: bool = False,
    reject_fasta: bool = False,
) -> None:
    """``mode="after"`` helper: require structures (non-list ⇒ none given), validate IDs, enforce uniform format."""
    if not isinstance(structures, list):
        raise ValueError("`structures` is required: provide a list of items or a directory path.")

    texts = [s if isinstance(s, str) else s.structure_pdb for s in structures]
    fasta_count = sum(1 for s in texts if _detect_input_format(s) == "fasta")
    if 0 < fasta_count < len(texts):
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
