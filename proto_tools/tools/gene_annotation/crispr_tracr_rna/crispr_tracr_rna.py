"""Standardized interface for the CRISPRtracrRNA pipeline.

Wraps the CRISPRtracrRNA tool from the Backofen Lab
(https://github.com/BackofenLab/CRISPRtracrRNA) for predicting tracrRNA
sequences and validating them via a multi-evidence ranking pipeline.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)


# ============================================================================
# Data Models
# ============================================================================
class CrisprTracrRNAPrediction(BaseModel):
    """A single tracrRNA prediction with all upstream evidence fields.

    Mirrors upstream CRISPRtracrRNA's ``complete_run`` CSV columns plus the
    multi-evidence ``score`` from ``candidate_ranking.py``. Every field is
    nullable: ``model_run`` mode only populates candidate-detection fields;
    ``complete_run`` may populate any subject to pipeline-stage hits. Flag
    fields are raw strings (typically ``"True"``/``"False"``) to mirror
    upstream's CSV verbatim.
    """

    # Identity
    sequence_id: str = Field(title="Sequence ID", description="ID of the input sequence")
    accession_number: str | None = Field(
        default=None, title="Accession Number", description="Upstream's accession_number column."
    )
    crispr_array_index: int | None = Field(
        default=None, title="CRISPR Array Index", description="Index of the matched CRISPR array."
    )
    crispr_array_category: str | None = Field(
        default=None, title="CRISPR Array Category", description="CRISPR array category from CRISPRidentify."
    )

    # CRISPR array (from CRISPRidentify)
    crispr_array_score: float | None = Field(
        default=None, title="CRISPR Array Score", description="CRISPRidentify array-detection confidence."
    )
    crispr_array_start: int | None = Field(
        default=None, title="CRISPR Array Start", description="Start position of the CRISPR array."
    )
    crispr_array_end: int | None = Field(
        default=None, title="CRISPR Array End", description="End position of the CRISPR array."
    )
    crispr_array_repeat_consensus: str | None = Field(
        default=None, title="Repeat Consensus", description="Consensus repeat sequence."
    )
    crispr_array_orientation: str | None = Field(
        default=None, title="Array Orientation", description="Predicted array orientation."
    )
    crispr_orientation_flag: str | None = Field(
        default=None, title="Orientation Flag", description="Orientation-confidence flag."
    )

    # Anti-repeat (from fasta36 + vmatch + clustalo + blast)
    anti_repeat_sequence: str | None = Field(
        default=None, title="Anti-repeat Sequence", description="Anti-repeat sequence."
    )
    anti_repeat_start: int | None = Field(
        default=None, title="Anti-repeat Start", description="Anti-repeat start position."
    )
    anti_repeat_end: int | None = Field(default=None, title="Anti-repeat End", description="Anti-repeat end position.")
    anti_repeat_direction: str | None = Field(
        default=None, title="Anti-repeat Direction", description="Anti-repeat strand/direction."
    )
    anti_repeat_relative_location: str | None = Field(
        default=None,
        title="Anti-repeat Location",
        description="Anti-repeat location relative to the CRISPR array.",
    )
    anti_repeat_distance_from_crispr_array: int | None = Field(
        default=None,
        title="Anti-repeat Distance",
        description="Distance from the anti-repeat to the CRISPR array.",
    )
    anti_repeat_similarity: float | None = Field(
        default=None, title="Anti-repeat Similarity", description="Anti-repeat sequence similarity (0-1)."
    )
    anti_repeat_coverage: float | None = Field(
        default=None, title="Anti-repeat Coverage", description="Anti-repeat alignment coverage (0-1)."
    )
    anti_repeat_similarity_coverage_multiplication: float | None = Field(
        default=None, title="Similarity x Coverage", description="Similarity multiplied by coverage."
    )
    anti_repeat_upstream: str | None = Field(
        default=None, title="Anti-repeat Upstream", description="Sequence upstream of the anti-repeat."
    )

    # tracrRNA (from covariance-model search)
    tracr_rna_taken_flag: str | None = Field(
        default=None, title="tracrRNA Taken Flag", description="Flag for whether the tracrRNA was selected."
    )
    tracr_rna_tail_sequence: str | None = Field(
        default=None, title="tracrRNA Tail", description="3' tail sequence of the tracrRNA."
    )
    tracr_rna_global_window_sequence: str | None = Field(
        default=None, title="tracrRNA Window", description="Global window sequence around the tracrRNA."
    )
    tracr_rna_sequence: str | None = Field(
        default=None, title="tracrRNA Sequence", description="Predicted tracrRNA sequence."
    )

    # Interaction (from IntaRNA)
    intarna_anti_repeat_interaction_interval: str | None = Field(
        default=None, title="IntaRNA Interval", description="Interval of the predicted RNA-RNA interaction."
    )
    intarna_anti_repeat_interaction: str | None = Field(
        default=None, title="IntaRNA Interaction", description="IntaRNA anti-repeat interaction structure."
    )
    interaction_energy: float | None = Field(
        default=None,
        title="Interaction Energy",
        description="IntaRNA interaction energy in kcal/mol; more negative values indicate a stronger interaction.",
    )
    poli_u_signal_coordinates: str | None = Field(
        default=None,
        title="Poly-U Signal Coordinates",
        description="Coordinates of the poly-U transcription terminator signal.",
    )

    # Terminator (from erpin)
    terminator_all_locations: str | None = Field(
        default=None, title="Terminator Locations", description="All erpin terminator hit locations."
    )
    terminator_all_scores: str | None = Field(
        default=None, title="Terminator Scores", description="All erpin terminator hit scores."
    )
    best_terminator_location: str | None = Field(
        default=None, title="Best Terminator Location", description="Best erpin terminator location."
    )
    best_terminator_score: float | None = Field(
        default=None, title="Best Terminator Score", description="Best erpin terminator score."
    )
    terminator_presence_flag: str | None = Field(
        default=None, title="Terminator Flag", description="Flag for terminator presence."
    )

    # Tail (covariance-model tail hit)
    tail_model_hit_location: str | None = Field(
        default=None, title="Tail Hit Location", description="Tail model hit location."
    )
    tail_model_hit_score: float | None = Field(
        default=None, title="Tail Hit Score", description="Tail model hit score."
    )
    tail_presence_flag: str | None = Field(
        default=None, title="Tail Presence Flag", description="Flag for tail presence."
    )

    # Cas (from CRISPRcasIdentifier)
    closest_corresponding_cas_interval: str | None = Field(
        default=None, title="Closest Cas Interval", description="Coordinates of the closest cas-effector cassette."
    )
    distance_to_cas: int | None = Field(
        default=None, title="Distance to Cas", description="Distance from the tracrRNA to the cas cassette."
    )

    # Multi-evidence ranking (from candidate_ranking.py)
    score: float | None = Field(
        default=None,
        title="Ranking Score",
        description="Multi-evidence ranking score (weighted sum across all evidence fields).",
    )

    # model_run only
    start: int | None = Field(default=None, title="Start", description="cmsearch hit start position (model_run mode).")
    end: int | None = Field(default=None, title="End", description="cmsearch hit end position (model_run mode).")
    e_value: float | None = Field(default=None, title="E-value", description="cmsearch hit e-value (model_run mode).")
    best_e_value: float | None = Field(
        default=None, title="Best E-value", description="cmsearch best-hit e-value (model_run mode)."
    )
    hit_sequence: str | None = Field(
        default=None, title="Hit Sequence", description="cmsearch hit sequence (model_run mode)."
    )

    @property
    def has_tracr(self) -> bool:
        """Whether a tracrRNA was predicted (any candidate-detection field populated)."""
        return (
            self.tracr_rna_sequence is not None or self.anti_repeat_start is not None or self.hit_sequence is not None
        )


# Input:
class CrisprTracrRNAInput(BaseToolInput):
    """Input for CRISPRtracrRNA prediction.

    Attributes:
        sequences (list[str]): Nucleotide sequence(s) to predict tracrRNA from.
            Each sequence should contain a CRISPR locus. Labeled positionally
            (``seq_0``, ``seq_1``, ...); results are returned in input order.
    """

    sequences: list[str] = InputField(
        title="Sequences",
        description="Nucleotide sequence(s) to predict tracrRNA from",
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value: Any) -> list[str]:
        """Normalize a single sequence to a list."""
        if isinstance(value, str):
            return [value]
        return value  # type: ignore[no-any-return]


class CrisprTracrRNASequenceResult(BaseModel):
    """All tracrRNA candidates for one input sequence, score-descending.

    Attributes:
        sequence_id (str): ID of the input sequence.
        candidates (list[CrisprTracrRNAPrediction]): All candidate hits for
            this sequence, top-ranked first; empty when upstream found nothing.
    """

    sequence_id: str = Field(title="Sequence ID", description="ID of the input sequence")
    candidates: list[CrisprTracrRNAPrediction] = Field(
        default_factory=list,
        title="tracrRNA Candidates",
        description="All candidate hits for this sequence, top-ranked first.",
    )

    @property
    def has_tracr(self) -> bool:
        """Whether any candidate has a tracrRNA detection."""
        return any(c.has_tracr for c in self.candidates)

    @property
    def top_candidate(self) -> CrisprTracrRNAPrediction | None:
        """The highest-scoring candidate, or None when no hits."""
        return self.candidates[0] if self.candidates else None


# Output:
class CrisprTracrRNAOutput(BaseToolOutput):
    """Output from CRISPRtracrRNA prediction.

    Attributes:
        results (list[CrisprTracrRNASequenceResult]): One result per input
            sequence, each carrying all candidate hits upstream produced for
            that sequence (top-ranked first).
    """

    results: list[CrisprTracrRNASequenceResult] = Field(
        default_factory=list,
        title="Results",
        description="Per-input sequence results, each with all candidate hits top-ranked first.",
    )

    @property
    def num_with_tracr(self) -> int:
        """Number of input sequences for which a tracrRNA was detected."""
        return sum(1 for r in self.results if r.has_tracr)

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        import pandas as pd

        # One row per candidate; sequences with no candidates emit a sequence_id-only row.
        rows: list[dict[str, Any]] = []
        for result in self.results:
            if not result.candidates:
                rows.append({"sequence_id": result.sequence_id})
            else:
                rows.extend(c.model_dump() for c in result.candidates)
        path = Path(export_path).with_suffix(f".{file_format}")
        df = pd.DataFrame(rows)
        if file_format == "csv":
            df.to_csv(path, index=False)
        elif file_format == "json":
            df.to_json(path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# Config:
class CrisprTracrRNAConfig(BaseConfig):
    """Configuration for CRISPRtracrRNA prediction.

    Mirrors the upstream CRISPRtracrRNA.py argparse surface. Defaults match
    upstream verbatim. The ten ranking weights only take effect in
    ``run_type="complete_run"``.

    Attributes:
        model_type (Literal['II', 'all']): CRISPR model type.
        run_type (Literal['complete_run', 'model_run']): Pipeline mode.
        num_workers (int | None): Parallel workers across input sequences (defaults to 1).
        anti_repeat_similarity_threshold (float): Minimum anti-repeat ↔ repeat similarity (0-1).
        anti_repeat_coverage_threshold (float): Minimum anti-repeat alignment coverage (0-1).
        weight_crispr_array_score (float): Ranking weight for CRISPR array confidence.
        weight_anti_repeat_sim (float): Ranking weight for anti-repeat similarity.
        weight_anti_repeat_coverage (float): Ranking weight for anti-repeat coverage.
        weight_anti_sim_coverage (float): Ranking weight for similarity x coverage.
        weight_interaction_score (float): Ranking weight for IntaRNA interaction energy.
        weight_model_hit_score (float): Ranking weight for the covariance-model tail hit.
        weight_terminator_hit_score (float): Ranking weight for erpin terminator score.
        weight_consistency_orientation (float): Ranking weight for orientation consistency.
        weight_consistency_anti_repeat_tail (float): Ranking weight for anti-repeat ↔ tail consistency.
        weight_consistency_tail_terminator (float): Ranking weight for tail ↔ terminator consistency.
        perform_type_v_anti_repeat_analysis (bool): Type V (Cas12) anti-repeat search.
    """

    model_type: Literal["II", "all"] = ConfigField(
        title="Model Type",
        default="II",
        description="CRISPR model: 'II' (type II only, fast, default) or 'all' (type II + type V cluster models)",
    )
    run_type: Literal["complete_run", "model_run"] = ConfigField(
        title="Run Type",
        default="complete_run",
        description="Pipeline mode: 'complete_run' (full ranking, default) or 'model_run' (cmsearch scan only, fast)",
    )
    num_workers: int | None = ConfigField(
        title="Number of Workers",
        default=None,
        description="Parallel workers across input sequences. None or 0 defaults to 1; capped at len(sequences).",
        include_in_key=False,
    )
    anti_repeat_similarity_threshold: float = ConfigField(
        title="Anti-repeat Similarity",
        default=0.7,
        description="Anti-repeat ↔ repeat similarity floor (0-1). Lower for divergent CRISPR families",
    )
    anti_repeat_coverage_threshold: float = ConfigField(
        title="Anti-repeat Coverage",
        default=0.6,
        description="Anti-repeat alignment coverage floor (0-1). Lower for partial anti-repeats",
    )
    weight_crispr_array_score: float = ConfigField(
        title="Weight: CRISPR Array Score",
        default=0.5,
        description="Multi-evidence ranking weight for CRISPRidentify array-detection confidence.",
    )
    weight_anti_repeat_sim: float = ConfigField(
        title="Weight: Anti-repeat Similarity",
        default=0.5,
        description="Multi-evidence ranking weight for anti-repeat sequence similarity.",
    )
    weight_anti_repeat_coverage: float = ConfigField(
        title="Weight: Anti-repeat Coverage",
        default=0.5,
        description="Multi-evidence ranking weight for anti-repeat alignment coverage.",
    )
    weight_anti_sim_coverage: float = ConfigField(
        title="Weight: Sim x Coverage",
        default=0.5,
        description="Multi-evidence ranking weight for the similarity x coverage product.",
    )
    weight_interaction_score: float = ConfigField(
        title="Weight: IntaRNA Score",
        default=0.6,
        description="Multi-evidence ranking weight for the IntaRNA RNA-RNA interaction energy.",
    )
    weight_model_hit_score: float = ConfigField(
        title="Weight: Tail Hit Score",
        default=0.9,
        description="Multi-evidence ranking weight for the covariance-model tail hit score.",
    )
    weight_terminator_hit_score: float = ConfigField(
        title="Weight: Terminator Hit Score",
        default=0.9,
        description="Multi-evidence ranking weight for erpin terminator presence/score.",
    )
    weight_consistency_orientation: float = ConfigField(
        title="Weight: Orientation",
        default=0.1,
        description="Multi-evidence ranking weight for repeat / anti-repeat orientation consistency.",
    )
    weight_consistency_anti_repeat_tail: float = ConfigField(
        title="Weight: Anti-repeat-Tail",
        default=0.1,
        description="Multi-evidence ranking weight for anti-repeat ↔ tail positional consistency.",
    )
    weight_consistency_tail_terminator: float = ConfigField(
        title="Weight: Tail-Terminator",
        default=0.1,
        description="Multi-evidence ranking weight for tail ↔ terminator positional consistency.",
    )
    perform_type_v_anti_repeat_analysis: bool = ConfigField(
        title="Type V Anti-repeat Analysis",
        default=False,
        description="Search Type V (Cas12) anti-repeat locations. Niche; off by default",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return CrisprTracrRNAInput(sequences=["ATCGATCG"])


@tool(
    key="crispr-tracr-rna",
    label="CRISPRtracrRNA Prediction",
    category="gene_annotation",
    input_class=CrisprTracrRNAInput,
    config_class=CrisprTracrRNAConfig,
    output_class=CrisprTracrRNAOutput,
    description="Predict tracrRNA sequences from nucleotide CRISPR loci",
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    cacheable=True,
)
def run_crispr_tracr_rna(
    inputs: CrisprTracrRNAInput,
    config: CrisprTracrRNAConfig,
    instance: Any = None,
) -> CrisprTracrRNAOutput:
    """Predict tracrRNA sequences from nucleotide CRISPR loci.

    Uses the CRISPRtracrRNA tool from the Backofen Lab to predict tracrRNA
    sequences associated with CRISPR loci, confirming and characterizing
    Type II (Cas9) and tracrRNA-bearing Type V (Cas12) CRISPR-Cas systems.

    Args:
        inputs (CrisprTracrRNAInput): Validated input containing nucleotide sequences.
        config (CrisprTracrRNAConfig): CRISPRtracrRNA configuration including model type.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        CrisprTracrRNAOutput: One ``CrisprTracrRNASequenceResult`` per input sequence,
            each carrying every candidate hit upstream produced (top-ranked first).

    Examples:
        >>> inputs = CrisprTracrRNAInput(sequences=["ATCG..." * 1000])
        >>> config = CrisprTracrRNAConfig(model_type="II")
        >>> result = run_crispr_tracr_rna(inputs, config)
        >>> top = result.results[0].top_candidate
        >>> print(f"{result.num_with_tracr} sequences have a tracrRNA hit")
    """
    sequence_ids = [f"seq_{i}" for i in range(len(inputs.sequences))]

    num_workers = config.num_workers or 1

    input_data = {
        "sequences": inputs.sequences,
        "sequence_ids": sequence_ids,
        "config": {
            "model_type": config.model_type,
            "run_type": config.run_type,
            "num_workers": num_workers,
            "anti_repeat_similarity_threshold": config.anti_repeat_similarity_threshold,
            "anti_repeat_coverage_threshold": config.anti_repeat_coverage_threshold,
            "weight_crispr_array_score": config.weight_crispr_array_score,
            "weight_anti_repeat_sim": config.weight_anti_repeat_sim,
            "weight_anti_repeat_coverage": config.weight_anti_repeat_coverage,
            "weight_anti_sim_coverage": config.weight_anti_sim_coverage,
            "weight_interaction_score": config.weight_interaction_score,
            "weight_model_hit_score": config.weight_model_hit_score,
            "weight_terminator_hit_score": config.weight_terminator_hit_score,
            "weight_consistency_orientation": config.weight_consistency_orientation,
            "weight_consistency_anti_repeat_tail": config.weight_consistency_anti_repeat_tail,
            "weight_consistency_tail_terminator": config.weight_consistency_tail_terminator,
            "perform_type_v_anti_repeat_analysis": config.perform_type_v_anti_repeat_analysis,
        },
    }

    input_data["device"] = "cpu"
    output_data = ToolInstance.dispatch(
        "crispr_tracr_rna",
        input_data,
        instance=instance,
        config=config,
    )

    results = [
        CrisprTracrRNASequenceResult(
            sequence_id=r["sequence_id"],
            candidates=[CrisprTracrRNAPrediction(**c) for c in r["candidates"]],
        )
        for r in output_data["results"]
    ]

    return CrisprTracrRNAOutput(
        metadata={
            "model_type": config.model_type,
            "run_type": config.run_type,
            "num_sequences": len(inputs.sequences),
        },
        results=results,
    )
