"""SpliceAI variant delta-score annotation."""

import csv
import json
import logging
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)
from proto_tools.utils.tool_io import Metrics, MetricSpec, MissingAssetError

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================
DEFAULT_ANNOTATION = "grch38"
# SpliceAI's -D flag accepts 0..4999; the model sees 5000 bp of context per side.
MAX_SPLICEAI_DISTANCE = 4999


# ============================================================================
# Data Models
# ============================================================================
class SpliceAIVariant(BaseToolInput):
    """A single genetic variant to score for splice-altering effects.

    ``position`` is 1-based (VCF), unlike the sibling ``AlphaGenomeVariant`` (0-based).

    Attributes:
        chromosome (str): Chromosome identifier, matching the reference FASTA and
            annotation (e.g. ``'chr1'`` or ``'1'`` — be consistent across all three).
        position (int): Variant position, 1-based (VCF convention).
        ref (str): Reference allele, e.g. ``'A'`` or ``'AC'`` (DNA bases A/C/G/T/N).
        alt (str): Alternate allele, e.g. ``'G'`` or ``'GTT'`` (DNA bases A/C/G/T/N).
    """

    chromosome: str = InputField(
        title="Chromosome",
        description="Chromosome identifier, e.g. 'chr1' (must match reference FASTA and annotation)",
    )
    position: int = InputField(
        ge=1,
        title="Position",
        description="Variant position, 1-based (VCF convention)",
    )
    ref: str = InputField(
        title="Reference Allele",
        description="Reference allele (e.g. 'A', 'AC'); must match the reference genome base at this position",
    )
    alt: str = InputField(title="Alternate Allele", description="Alternate allele (e.g. 'G', 'GTT')")

    @field_validator("ref", "alt")
    @classmethod
    def validate_allele_bases(cls, bases: str, info: ValidationInfo) -> str:
        """Uppercase and validate allele sequence characters (DNA A/C/G/T/N)."""
        normalized = bases.strip().upper()
        if not normalized:
            raise ValueError(f"{info.field_name}: cannot be empty")
        invalid = sorted(set(normalized) - set("ACGTN"))
        if invalid:
            raise ValueError(
                f"{info.field_name}: must only contain DNA bases A/C/G/T/N; got invalid {invalid} in {bases!r}"
            )
        return normalized


class SpliceAIScoreInput(BaseToolInput):
    """Input for SpliceAI variant scoring.

    Attributes:
        variants (list[SpliceAIVariant]): Variants to score. A single variant is
            auto-wrapped into a list.
    """

    variants: list[SpliceAIVariant] = InputField(
        title="Variants",
        description="Variants to score for splice-altering effects",
    )

    @field_validator("variants", mode="before")
    @classmethod
    def normalize_variants(cls, value: Any) -> list[Any]:
        """Normalize a single variant to a list and reject empty input."""
        if value is None:
            raise ValueError("variants cannot be None")
        if not isinstance(value, list):
            value = [value]
        if not value:
            raise ValueError("variants cannot be empty")
        return value  # type: ignore[no-any-return]


class SpliceAIGeneScore(BaseModel):
    """SpliceAI delta scores and positions for one variant against one gene.

    All scores and positions are ``None`` for complex MNV variants (multi-base
    ref and alt), which SpliceAI does not score.

    Attributes:
        allele (str): Alternate allele these scores correspond to.
        symbol (str): Gene symbol the variant was scored against.
        ds_ag (float | None): Delta score, acceptor gain (0-1).
        ds_al (float | None): Delta score, acceptor loss (0-1).
        ds_dg (float | None): Delta score, donor gain (0-1).
        ds_dl (float | None): Delta score, donor loss (0-1).
        dp_ag (int | None): Delta position, acceptor gain (bp relative to the variant).
        dp_al (int | None): Delta position, acceptor loss (bp relative to the variant).
        dp_dg (int | None): Delta position, donor gain (bp relative to the variant).
        dp_dl (int | None): Delta position, donor loss (bp relative to the variant).
    """

    allele: str = Field(title="Allele", description="Alternate allele these scores correspond to")
    symbol: str = Field(title="Gene Symbol", description="Gene symbol the variant was scored against")
    ds_ag: float | None = Field(title="DS Acceptor Gain", description="Delta score for acceptor gain (0-1)")
    ds_al: float | None = Field(title="DS Acceptor Loss", description="Delta score for acceptor loss (0-1)")
    ds_dg: float | None = Field(title="DS Donor Gain", description="Delta score for donor gain (0-1)")
    ds_dl: float | None = Field(title="DS Donor Loss", description="Delta score for donor loss (0-1)")
    dp_ag: int | None = Field(
        title="DP Acceptor Gain", description="Delta position for acceptor gain (bp from variant)"
    )
    dp_al: int | None = Field(
        title="DP Acceptor Loss", description="Delta position for acceptor loss (bp from variant)"
    )
    dp_dg: int | None = Field(title="DP Donor Gain", description="Delta position for donor gain (bp from variant)")
    dp_dl: int | None = Field(title="DP Donor Loss", description="Delta position for donor loss (bp from variant)")


class SpliceAIScoreMetrics(Metrics):
    """Per-variant SpliceAI scoring metric.

    Metrics documented in ``metric_spec``:
        max_delta_score (float): Max of the four delta scores; the headline
            SpliceAI score (thresholds >=0.2 / >=0.5 / >=0.8). Absent for
            variants with no gene overlap and for unscored complex MNVs.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "max_delta_score": {
            "description": "Maximum delta score (acceptor/donor gain/loss) across overlapping genes",
            "availability": "present for scored variants overlapping an annotated gene",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "context-dependent",
        },
    }
    primary_metric: str | None = Field(
        default="max_delta_score",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )


class SpliceAIVariantResult(BaseModel):
    """SpliceAI scores for one variant.

    Attributes:
        chromosome (str): Variant chromosome.
        position (int): Variant position (1-based).
        ref (str): Reference allele.
        alt (str): Alternate allele.
        scores (list[SpliceAIGeneScore]): One record per gene the variant
            overlaps (empty if it overlaps no annotated gene).
        metrics (SpliceAIScoreMetrics): Per-variant scalar metric (max delta score).
    """

    chromosome: str = Field(title="Chromosome", description="Variant chromosome")
    position: int = Field(title="Position", description="Variant position (1-based)")
    ref: str = Field(title="Reference Allele", description="Reference allele")
    alt: str = Field(title="Alternate Allele", description="Alternate allele")
    scores: list[SpliceAIGeneScore] = Field(
        title="Gene Scores",
        description="Per-gene SpliceAI delta scores (empty if no gene overlap)",
    )
    metrics: SpliceAIScoreMetrics = Field(
        title="Splice-Effect Metrics",
        description="Per-variant scalar metric (max delta score)",
    )


class SpliceAIScoreOutput(BaseToolOutput):
    """Output from SpliceAI variant scoring.

    Attributes:
        results (list[SpliceAIVariantResult]): Per-variant scores, 1:1 with the
            input variants and in the same order.
    """

    results: list[SpliceAIVariantResult] = Field(
        title="Results",
        description="Per-variant SpliceAI scores (1:1 with input variants)",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "csv", "vcf"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "json":
            with open(path, "w") as handle:
                json.dump(self.model_dump(mode="json"), handle, indent=2)
            return

        if file_format == "csv":
            rows = [
                {
                    "chromosome": r.chromosome,
                    "position": r.position,
                    "ref": r.ref,
                    "alt": r.alt,
                    "allele": s.allele,
                    "symbol": s.symbol,
                    "ds_ag": s.ds_ag,
                    "ds_al": s.ds_al,
                    "ds_dg": s.ds_dg,
                    "ds_dl": s.ds_dl,
                    "dp_ag": s.dp_ag,
                    "dp_al": s.dp_al,
                    "dp_dg": s.dp_dg,
                    "dp_dl": s.dp_dl,
                }
                for r in self.results
                for s in r.scores
            ]
            if not rows:
                path.write_text("")
                return
            with open(path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            return

        if file_format == "vcf":
            path.write_text(self._to_vcf())
            return

        raise ValueError(f"Unsupported format: {file_format}")

    def _to_vcf(self) -> str:
        """Render results as a minimal VCF carrying the standard SpliceAI INFO field."""
        lines = [
            "##fileformat=VCFv4.2",
            '##INFO=<ID=SpliceAI,Number=.,Type=String,Description="SpliceAI variant '
            "annotation. These include delta scores (DS) and delta positions (DP) for "
            "acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). "
            'Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL">',
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
        ]
        for r in self.results:
            info = "."
            if r.scores:
                annotations = []
                for s in r.scores:
                    ds = [f"{v:.2f}" if v is not None else "." for v in (s.ds_ag, s.ds_al, s.ds_dg, s.ds_dl)]
                    dp = [str(v) if v is not None else "." for v in (s.dp_ag, s.dp_al, s.dp_dg, s.dp_dl)]
                    annotations.append("|".join([s.allele, s.symbol, *ds, *dp]))
                info = "SpliceAI=" + ",".join(annotations)
            lines.append(f"{r.chromosome}\t{r.position}\t.\t{r.ref}\t{r.alt}\t.\t.\t{info}")
        return "\n".join(lines) + "\n"


class SpliceAIScoreConfig(BaseConfig):
    """Configuration for SpliceAI variant scoring.

    Attributes:
        reference_fasta (str | None): Path (or AssetRef) to the reference genome
            FASTA. Required at call time — SpliceAI extracts the wild-type
            sequence around each variant from this genome. ``None`` raises a
            ``MissingAssetError`` so un-provisioned hosts skip cleanly.
        annotation (str): Gene annotation source: ``'grch37'`` or ``'grch38'``
            (GENCODE files bundled with SpliceAI) or a path to a custom
            tab-separated annotation file.
        max_distance (int): Maximum distance (bp) between the variant and a
            gained/lost splice site to report (the SpliceAI ``-D`` flag).
        mask (bool): Mask scores for annotated acceptor/donor gain and
            unannotated acceptor/donor loss (the SpliceAI ``-M`` flag).
        device (str): Device to run inference on. SpliceAI (TensorFlow)
            auto-falls-back to CPU when no GPU is visible.
    """

    reference_fasta: str | None = ConfigField(
        title="Reference FASTA",
        default=None,
        description="Path (or AssetRef) to the reference genome FASTA; required at call time",
        reload_on_change=True,
    )
    annotation: str = ConfigField(
        title="Annotation",
        default=DEFAULT_ANNOTATION,
        description="'grch37'/'grch38' (bundled GENCODE) or path to a custom gene annotation file",
        reload_on_change=True,
    )
    max_distance: int = ConfigField(
        title="Max Distance",
        default=50,
        ge=0,
        le=MAX_SPLICEAI_DISTANCE,
        description="Max distance (bp) between variant and gained/lost splice site (the -D flag)",
    )
    mask: bool = ConfigField(
        title="Mask",
        default=False,
        description="Zero out scores for annotated-site gains and unannotated-site losses (SpliceAI -M flag)",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run inference on (e.g. 'cpu', 'cuda', 'cuda:0')",
        include_in_key=False,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return SpliceAIScoreInput(variants=[SpliceAIVariant(chromosome="chr1", position=100, ref="A", alt="C")])


@tool(
    key="spliceai-score",
    label="SpliceAI Variant Scoring",
    category="rna_splicing",
    input_class=SpliceAIScoreInput,
    config_class=SpliceAIScoreConfig,
    output_class=SpliceAIScoreOutput,
    description="Score variants for splice-altering effects (delta scores/positions) with SpliceAI",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["variants"],
    iterable_output_field="results",
    cacheable=True,
    metrics_class=SpliceAIScoreMetrics,
)
def run_spliceai_score(
    inputs: SpliceAIScoreInput,
    config: SpliceAIScoreConfig,
    instance: Any = None,
) -> SpliceAIScoreOutput:
    """Score genetic variants for splice-altering effects using SpliceAI.

    Args:
        inputs (SpliceAIScoreInput): Variants to score.
        config (SpliceAIScoreConfig): Reference genome, annotation, distance, masking, and device.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        SpliceAIScoreOutput: Per-variant results (1:1 with inputs), each with per-gene scores and a max-delta metric.

    Raises:
        MissingAssetError: If ``config.reference_fasta`` is None or missing (the test layer converts this to a skip).
    """
    if config.reference_fasta is None or not Path(config.reference_fasta).expanduser().exists():
        raise MissingAssetError(
            "spliceai",
            "reference",
            f"reference_fasta not provided or not found: {config.reference_fasta!r}. "
            "SpliceAI requires a reference genome FASTA (set SpliceAIScoreConfig.reference_fasta).",
        )

    logger.debug("Using local venv for SpliceAI variant scoring")

    dispatch_result = ToolInstance.dispatch(
        "spliceai",
        {
            "operation": "score",
            "variants": [
                {"chromosome": v.chromosome, "position": v.position, "ref": v.ref, "alt": v.alt}
                for v in inputs.variants
            ],
            "reference_fasta": config.reference_fasta,
            "annotation": config.annotation,
            "max_distance": config.max_distance,
            "mask": int(config.mask),
            "device": config.device,
        },
        instance=instance,
        config=config,
    )

    results: list[SpliceAIVariantResult] = []
    for variant, gene_dicts in zip(inputs.variants, dispatch_result["results"], strict=True):
        scores = [SpliceAIGeneScore(**gene) for gene in gene_dicts]
        ds_values = [v for s in scores for v in (s.ds_ag, s.ds_al, s.ds_dg, s.ds_dl) if v is not None]
        max_delta = max(ds_values) if ds_values else None
        results.append(
            SpliceAIVariantResult(
                chromosome=variant.chromosome,
                position=variant.position,
                ref=variant.ref,
                alt=variant.alt,
                scores=scores,
                metrics=SpliceAIScoreMetrics(max_delta_score=max_delta),
            )
        )
    return SpliceAIScoreOutput(results=results)
