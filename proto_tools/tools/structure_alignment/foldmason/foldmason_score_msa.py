"""Wraps `foldmason msa2lddt` for scoring a precomputed MSA against a structure set.

Local-only — no remote analog. Computes a per-MSA average LDDT score plus
per-column scores by aligning each row's structure with the alignment, useful
as a quality metric for any structural-MSA pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

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


class FoldmasonScoreMSAInput(BaseToolInput):
    """Input for FoldMason MSA-LDDT scoring.

    Attributes:
        structures (list[str]): PDB-format text strings (≥2). Order must match
            the rows of ``aa_msa_fasta``.
        structure_ids (list[str] | None): Optional IDs per structure (default:
            ``'structure_0'``, ...). Must match the FASTA record headers in
            ``aa_msa_fasta`` so msa2lddt can resolve each row to its structure.
        aa_msa_fasta (str): Amino-acid MSA in FASTA, typically from `foldmason-msa`.
    """

    structures: list[str] = InputField(
        description="PDB-format text strings whose order matches the rows of `aa_msa_fasta` (≥2)", min_length=2
    )
    structure_ids: list[str] | None = InputField(
        default=None,
        description="Optional IDs per structure (default: 'structure_0', ...); must match FASTA headers in `aa_msa_fasta`",
    )
    aa_msa_fasta: str = InputField(
        description="Amino-acid MSA in FASTA. Row order and IDs must match `structures` / `structure_ids`"
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
    def _ids_match_structures_length(self) -> "FoldmasonScoreMSAInput":
        """structure_ids, when supplied, must have the same length as structures."""
        if self.structure_ids is not None and len(self.structure_ids) != len(self.structures):
            raise ValueError(
                f"structure_ids length ({len(self.structure_ids)}) must match structures length ({len(self.structures)})"
            )
        return self


class FoldmasonScoreMSAConfig(BaseConfig):
    """Configuration for FoldMason msa2lddt.

    Attributes:
        pair_threshold (float): Minimum fraction of pair sub-alignments with
            LDDT information required to score a column (0-1). 0.0 (default)
            keeps all columns.
        only_scoring_cols (bool): If True, normalise the average LDDT by the
            number of scoring columns rather than total alignment length.
        guide_tree_newick (str | None): Newick guide tree to score against;
            leaf labels must match ``structure_ids``. None lets foldmason
            recompute the tree internally.
        num_threads (int): CPU threads.
    """

    pair_threshold: float = ConfigField(
        title="Pair Threshold",
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of pair sub-alignments with LDDT info to score a column (0-1)",
    )
    only_scoring_cols: bool = ConfigField(
        title="Only Scoring Columns",
        default=False,
        description="Normalise average LDDT by scoring-column count rather than alignment length",
    )
    guide_tree_newick: str | None = ConfigField(
        title="Guide Tree (Newick)",
        default=None,
        description="Newick guide tree to score against; leaf labels must match structure_ids",
    )
    num_threads: int = ConfigField(title="Threads", default=4, ge=1, description="CPU threads", include_in_key=False)


class FoldmasonScoreMSAOutput(BaseToolOutput):
    """Output from FoldMason MSA-LDDT scoring.

    Attributes:
        average_lddt (float): Average MSA LDDT score (0-1) across all scored
            columns.
        columns_considered (int): Number of columns that had enough pairwise
            information to be scored.
        alignment_length (int): Total number of MSA columns.
        column_scores (list[float]): Per-column LDDT scores, length equal to
            ``alignment_length``.
    """

    average_lddt: float = Field(description="Average MSA LDDT score (0-1)", ge=0.0, le=1.0)
    columns_considered: int = Field(description="Number of MSA columns scored", ge=0)
    alignment_length: int = Field(description="Total MSA columns", ge=0)
    column_scores: list[float] = Field(default_factory=list, description="Per-column LDDT scores")

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


# Shared 65-residue fixture; the AA MSA is the fixture's own sequence aligned
# to itself (two identical copies, so no gaps).
_EXAMPLE_PDB_PATH = str(Path(__file__).parents[1] / "example_input_fixture.pdb")
_EXAMPLE_AA_SEQUENCE = "MRKKLDLKKFVEDKNQEYAARALGLSQKLIEEVLKRGLPVYVETNKDGNIKVYITQDGITQPFPP"
_EXAMPLE_MSA = f">structure_0\n{_EXAMPLE_AA_SEQUENCE}\n>structure_1\n{_EXAMPLE_AA_SEQUENCE}\n"


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    pdb_text = Path(_EXAMPLE_PDB_PATH).read_text()
    return FoldmasonScoreMSAInput(
        structures=[pdb_text, pdb_text],
        aa_msa_fasta=_EXAMPLE_MSA,
    )


@tool(
    key="foldmason-score-msa",
    label="FoldMason Score MSA",
    category="structure_alignment",
    input_class=FoldmasonScoreMSAInput,
    config_class=FoldmasonScoreMSAConfig,
    output_class=FoldmasonScoreMSAOutput,
    description="Score a structural MSA with average + per-column LDDT using FoldMason msa2lddt",
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_foldmason_score_msa(
    inputs: FoldmasonScoreMSAInput,
    config: FoldmasonScoreMSAConfig,
    instance: Any = None,
) -> FoldmasonScoreMSAOutput:
    """Score a structural MSA with FoldMason msa2lddt.

    Args:
        inputs (FoldmasonScoreMSAInput): Structures + AA-FASTA MSA + optional IDs.
        config (FoldmasonScoreMSAConfig): Scoring thresholds + threads.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        FoldmasonScoreMSAOutput: Average + per-column LDDT.
    """
    ids = inputs.structure_ids or [f"structure_{i}" for i in range(len(inputs.structures))]

    output_data = ToolInstance.dispatch(
        "foldmason",
        {
            "operation": "msa2lddt",
            "structures": inputs.structures,
            "structure_ids": ids,
            "aa_msa_fasta": inputs.aa_msa_fasta,
            "pair_threshold": config.pair_threshold,
            "only_scoring_cols": config.only_scoring_cols,
            "guide_tree_newick": config.guide_tree_newick,
            "num_threads": config.num_threads,
        },
        instance=instance,
        config=config,
    )
    return FoldmasonScoreMSAOutput(
        average_lddt=output_data["average_lddt"],
        columns_considered=output_data["columns_considered"],
        alignment_length=output_data["alignment_length"],
        column_scores=output_data["column_scores"],
    )
