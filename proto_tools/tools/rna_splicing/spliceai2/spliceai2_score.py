"""SpliceAI2 variant splice-effect scoring from a reference genome."""

import csv
import json
import logging
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator

from proto_tools.tools.rna_splicing.reference_genome import ReferenceGenomeConfig
from proto_tools.tools.rna_splicing.spliceai.spliceai_score import SpliceAIVariant
from proto_tools.tools.rna_splicing.spliceai2.shared_data_models import SPLICEAI2_HF_URL, SpliceAI2Config
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseToolInput, BaseToolOutput, InputField, ToolInstance
from proto_tools.utils.auth import require_hf_token
from proto_tools.utils.tool_io import Metrics, MetricSpec

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================
# Upstream reports the ten strongest changes per effect type.
TOP_K = 10

# Effect type -> upstream column prefix, in upstream column order.
_SITE_EFFECTS: dict[str, str] = {
    "donor_gain": "donor_gain",
    "donor_loss": "donor_loss",
    "acceptor_gain": "acceptor_gain",
    "acceptor_loss": "acceptor_loss",
}
_JUNCTION_EFFECTS: dict[str, str] = {"junction_gain": "jxn_gain", "junction_loss": "jxn_loss"}

# Upstream field suffixes per effect kind, mapped to this tool's field names.
_SITE_FIELDS: dict[str, str] = {
    "delta_score": "delta_score",
    "ref_score": "ref_score",
    "alt_score": "alt_score",
    "dist": "distance",
}
_JUNCTION_FIELDS: dict[str, str] = {
    "delta_score": "delta_score",
    "ref_score": "ref_score",
    "alt_score": "alt_score",
    "donor_dist": "donor_distance",
    "acceptor_dist": "acceptor_distance",
}


def upstream_columns() -> list[str]:
    """The 260 per-change columns of upstream's ``.spliceai2`` output, in upstream order."""
    columns = [f"{p}_{f}_{k}" for p in _SITE_EFFECTS.values() for f in _SITE_FIELDS for k in range(TOP_K)]
    columns += [f"{p}_{f}_{k}" for p in _JUNCTION_EFFECTS.values() for f in _JUNCTION_FIELDS for k in range(TOP_K)]
    return columns


# ============================================================================
# Data Models
# ============================================================================
class SpliceAI2Variant(SpliceAIVariant):
    """A genetic variant and the strand of the gene it is scored against.

    Attributes:
        chromosome (str): Chromosome identifier, matching the reference FASTA
            (e.g. ``'chr1'`` or ``'1'``).
        position (int): Variant position, 1-based (VCF convention).
        ref (str): Reference allele, e.g. ``'A'`` or ``'AC'`` (DNA bases A/C/G/T/N).
        alt (str): Alternate allele, e.g. ``'G'`` or ``'GTT'`` (DNA bases A/C/G/T/N).
        strand (Literal['+', '-']): Strand of the gene the variant is scored
            against. Score both strands when it is unknown.
    """

    strand: Literal["+", "-"] = InputField(
        title="Strand",
        description="Gene strand ('+' or '-'); score both strands if unknown",
    )


class SpliceAI2ScoreInput(BaseToolInput):
    """Input for SpliceAI2 variant scoring.

    Attributes:
        variants (list[SpliceAI2Variant]): Variants to score. A single variant is
            auto-wrapped into a list.
    """

    variants: list[SpliceAI2Variant] = InputField(
        title="Variants",
        description="Variants (with gene strand) to score for splice-altering effects",
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


class SpliceAI2SiteChange(BaseModel):
    """One predicted change in splice donor or acceptor usage.

    Attributes:
        delta_score (float): Absolute change in splice site usage (0-1).
        ref_score (float): Predicted usage with the reference allele (0-1).
        alt_score (float): Predicted usage with the alternate allele (0-1).
        distance (int): Signed distance (bp, along the gene) from the variant to the site.
    """

    delta_score: float = Field(title="Delta Score", description="Absolute change in splice site usage (0-1)")
    ref_score: float = Field(title="Reference Score", description="Splice site usage with the reference allele")
    alt_score: float = Field(title="Alternate Score", description="Splice site usage with the alternate allele")
    distance: int = Field(title="Distance", description="Signed distance (bp) from the variant to the splice site")


class SpliceAI2JunctionChange(BaseModel):
    """One predicted change in splice junction usage.

    Attributes:
        delta_score (float): Absolute change in junction usage (0-1).
        ref_score (float): Predicted usage with the reference allele (0-1).
        alt_score (float): Predicted usage with the alternate allele (0-1).
        donor_distance (int): Signed distance (bp) from the variant to the junction donor.
        acceptor_distance (int): Signed distance (bp) from the variant to the junction acceptor.
    """

    delta_score: float = Field(title="Delta Score", description="Absolute change in junction usage (0-1)")
    ref_score: float = Field(title="Reference Score", description="Junction usage with the reference allele")
    alt_score: float = Field(title="Alternate Score", description="Junction usage with the alternate allele")
    donor_distance: int = Field(
        title="Donor Distance", description="Signed distance (bp) from the variant to the donor"
    )
    acceptor_distance: int = Field(
        title="Acceptor Distance", description="Signed distance (bp) from the variant to the acceptor"
    )


class SpliceAI2ScoreMetrics(Metrics):
    """Per-variant SpliceAI2 scoring metric.

    Metrics documented in ``metric_spec``:
        summary_score (float): Largest donor/acceptor gain or loss; the headline
            SpliceAI2 score (thresholds 0.1 / 0.25 / 0.5).
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "summary_score": {
            "description": "Maximum delta score across donor/acceptor gain and loss",
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "context-dependent",
        },
    }
    primary_metric: str | None = Field(
        default="summary_score",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )


class SpliceAI2VariantResult(BaseModel):
    """SpliceAI2 splice-effect predictions for one variant.

    Attributes:
        chromosome (str): Variant chromosome.
        position (int): Variant position (1-based).
        ref (str): Reference allele.
        alt (str): Alternate allele.
        strand (Literal['+', '-']): Gene strand the variant was scored against.
        donor_gain (list[SpliceAI2SiteChange]): Ten strongest new or strengthened donors.
        donor_loss (list[SpliceAI2SiteChange]): Ten strongest lost or weakened donors.
        acceptor_gain (list[SpliceAI2SiteChange]): Ten strongest new or strengthened acceptors.
        acceptor_loss (list[SpliceAI2SiteChange]): Ten strongest lost or weakened acceptors.
        junction_gain (list[SpliceAI2JunctionChange]): Ten strongest new or strengthened junctions.
        junction_loss (list[SpliceAI2JunctionChange]): Ten strongest lost or weakened junctions.
        metrics (SpliceAI2ScoreMetrics): Per-variant summary score.
    """

    chromosome: str = Field(title="Chromosome", description="Variant chromosome")
    position: int = Field(title="Position", description="Variant position (1-based)")
    ref: str = Field(title="Reference Allele", description="Reference allele")
    alt: str = Field(title="Alternate Allele", description="Alternate allele")
    strand: Literal["+", "-"] = Field(title="Strand", description="Gene strand the variant was scored against")
    donor_gain: list[SpliceAI2SiteChange] = Field(
        title="Donor Gain", description="Strongest donor gains, largest first"
    )
    donor_loss: list[SpliceAI2SiteChange] = Field(
        title="Donor Loss", description="Strongest donor losses, largest first"
    )
    acceptor_gain: list[SpliceAI2SiteChange] = Field(
        title="Acceptor Gain", description="Strongest acceptor gains, largest first"
    )
    acceptor_loss: list[SpliceAI2SiteChange] = Field(
        title="Acceptor Loss", description="Strongest acceptor losses, largest first"
    )
    junction_gain: list[SpliceAI2JunctionChange] = Field(
        title="Junction Gain", description="Strongest junction gains, largest first"
    )
    junction_loss: list[SpliceAI2JunctionChange] = Field(
        title="Junction Loss", description="Strongest junction losses, largest first"
    )
    metrics: SpliceAI2ScoreMetrics = Field(title="Metrics", description="Per-variant summary score")

    @property
    def summary_score(self) -> float:
        """Largest donor/acceptor gain or loss (upstream ``spliceai2_summary_score``)."""
        return float(self.metrics.get("summary_score"))

    def to_row(self) -> dict[str, Any]:
        """Flatten into one row of upstream's ``.spliceai2`` table."""
        row: dict[str, Any] = {
            "chrom": self.chromosome,
            "pos": self.position,
            "ref": self.ref,
            "alt": self.alt,
            "strand": self.strand,
        }
        for name, prefix in _SITE_EFFECTS.items():
            changes: list[SpliceAI2SiteChange] = getattr(self, name)
            for upstream_field, field in _SITE_FIELDS.items():
                for k, change in enumerate(changes):
                    row[f"{prefix}_{upstream_field}_{k}"] = getattr(change, field)
        for name, prefix in _JUNCTION_EFFECTS.items():
            junctions: list[SpliceAI2JunctionChange] = getattr(self, name)
            for upstream_field, field in _JUNCTION_FIELDS.items():
                for k, junction in enumerate(junctions):
                    row[f"{prefix}_{upstream_field}_{k}"] = getattr(junction, field)
        row["spliceai2_summary_score"] = self.summary_score
        return row


class SpliceAI2ScoreOutput(BaseToolOutput):
    """Output from SpliceAI2 variant scoring.

    Attributes:
        results (list[SpliceAI2VariantResult]): Per-variant predictions, 1:1 with
            the input variants and in the same order.
    """

    results: list[SpliceAI2VariantResult] = Field(
        title="Results",
        description="Per-variant SpliceAI2 predictions (1:1 with input variants)",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "tsv"]

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

        if file_format == "tsv":
            # Same columns as upstream's `.spliceai2` file, so existing downstream scripts read it.
            fieldnames = ["chrom", "pos", "ref", "alt", "strand", *upstream_columns(), "spliceai2_summary_score"]
            with open(path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
                writer.writerows(result.to_row() for result in self.results)
            return

        raise ValueError(f"Unsupported format: {file_format}")


class SpliceAI2ScoreConfig(SpliceAI2Config, ReferenceGenomeConfig):
    """Configuration for SpliceAI2 variant scoring.

    The input window (196,608 bp, reporting ±32,768 bp around the variant), the
    1,024 candidate splice sites shared by both alleles, and the two-model
    ensemble are fixed at the values the model was trained and evaluated with.

    Attributes:
        reference_fasta (ProvisionedGenome | str | None): The reference genome
            the sequence around each variant is read from. Either a provisioned
            assembly name (``'grch38'``/``'grch37'``, or ``'hg38'``/``'hg19'``),
            downloaded on first use, or a path to a FASTA on this machine.
            Required at call time; ``None`` raises a ``MissingAssetError``.
        assembly (SpliceAI2Assembly): Species assembly whose channel conditions
            the model. ``GRCh38`` (human) for human sequence, including GRCh37
            coordinates.
        device (str): Device to run the model on (default ``cuda``).
    """


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input: HBB IVS1-1 G>A (c.92+1G>A), a canonical donor loss on GRCh38."""
    return SpliceAI2ScoreInput(
        variants=[SpliceAI2Variant(chromosome="11", position=5226929, ref="C", alt="T", strand="-")]
    )


def _site_changes(row: dict[str, Any], prefix: str) -> list[SpliceAI2SiteChange]:
    """Collect one site effect type's top-k changes from an upstream-keyed row."""
    return [
        SpliceAI2SiteChange(**{field: row[f"{prefix}_{upstream}_{k}"] for upstream, field in _SITE_FIELDS.items()})
        for k in range(TOP_K)
    ]


def _junction_changes(row: dict[str, Any], prefix: str) -> list[SpliceAI2JunctionChange]:
    """Collect one junction effect type's top-k changes from an upstream-keyed row."""
    return [
        SpliceAI2JunctionChange(
            **{field: row[f"{prefix}_{upstream}_{k}"] for upstream, field in _JUNCTION_FIELDS.items()}
        )
        for k in range(TOP_K)
    ]


@tool(
    key="spliceai2-score",
    label="SpliceAI2 Variant Scoring",
    category="rna_splicing",
    input_class=SpliceAI2ScoreInput,
    config_class=SpliceAI2ScoreConfig,
    output_class=SpliceAI2ScoreOutput,
    description="Score variants for changes in splice site and junction usage with SpliceAI2",
    uses_gpu=True,
    gpu_only=True,
    example_input=example_input,
    iterable_input_fields=["variants"],
    iterable_output_field="results",
    max_chunk_size=64,
    cacheable=True,
    metrics_class=SpliceAI2ScoreMetrics,
)
def run_spliceai2_score(
    inputs: SpliceAI2ScoreInput,
    config: SpliceAI2ScoreConfig,
    instance: Any = None,
) -> SpliceAI2ScoreOutput:
    """Score genetic variants for splice-altering effects using SpliceAI2.

    Each variant is scored by comparing the reference and alternate alleles in a
    196,608 bp window read from the reference genome, on the given gene strand.
    SpliceAI2 reports the ten strongest donor, acceptor, and junction gains and
    losses within ±32,768 bp, plus a summary score (the largest site effect).

    Args:
        inputs (SpliceAI2ScoreInput): Variants to score, each with its gene strand.
        config (SpliceAI2ScoreConfig): Reference genome, species assembly, and device.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        SpliceAI2ScoreOutput: Per-variant predictions, 1:1 with the inputs.

    Raises:
        MissingAssetError: If ``config.reference_fasta`` is None or missing (the test layer converts this to a skip).
        OSError: If no HuggingFace token is available for the gated weights.

    See Also:
        - Model repository: https://github.com/Illumina/SpliceAI2
        - Weights: https://huggingface.co/illumina-ai/SpliceAI2
    """
    require_hf_token("SpliceAI2", SPLICEAI2_HF_URL)
    # Resolved here, so a named assembly is provisioned before the worker starts.
    resolved_fasta = config.require_reference_fasta("spliceai2")

    logger.debug("Using local venv for SpliceAI2 variant scoring")

    output_data = ToolInstance.dispatch(
        "spliceai2",
        {
            "operation": "score",
            "variants": [
                {"chromosome": v.chromosome, "position": v.position, "ref": v.ref, "alt": v.alt, "strand": v.strand}
                for v in inputs.variants
            ],
            "reference_fasta": str(resolved_fasta),
            "assembly": config.assembly,
            "device": config.device,
        },
        instance=instance,
        config=config,
    )

    results = [
        SpliceAI2VariantResult(
            chromosome=variant.chromosome,
            position=variant.position,
            ref=variant.ref,
            alt=variant.alt,
            strand=variant.strand,
            donor_gain=_site_changes(row, "donor_gain"),
            donor_loss=_site_changes(row, "donor_loss"),
            acceptor_gain=_site_changes(row, "acceptor_gain"),
            acceptor_loss=_site_changes(row, "acceptor_loss"),
            junction_gain=_junction_changes(row, "jxn_gain"),
            junction_loss=_junction_changes(row, "jxn_loss"),
            metrics=SpliceAI2ScoreMetrics(summary_score=row["spliceai2_summary_score"]),
        )
        for variant, row in zip(inputs.variants, output_data["results"], strict=True)
    ]
    return SpliceAI2ScoreOutput(
        results=results,
        metadata={"reference_fasta": config.reference_fasta, "assembly": config.assembly},
    )
