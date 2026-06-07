"""Variant splice-effect scoring (gain/loss) with Pangolin."""

import csv
import json
import logging
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from proto_tools.tools.rna_splicing.pangolin.shared_data_models import (
    PANGOLIN_FLANK,
    PangolinConfig,
    validate_dna,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseToolInput, BaseToolOutput, ConfigField, InputField, ToolInstance
from proto_tools.utils.tool_io import Metrics, MetricSpec

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
class PangolinVariant(BaseToolInput):
    """A variant and the reference sequence window it sits in.

    Sequence-centric, so no genome FASTA is needed: coordinates are 0-based
    within ``sequence``, which must give ``PANGOLIN_FLANK`` (5,000) bp of flank on
    each side (``PANGOLIN_FLANK + distance`` to fill the full reporting window,
    else it is clipped). Only SNVs and simple indels are supported.

    Attributes:
        sequence (str): Reference DNA window containing the variant.
        variant_position (int): 0-based index of the variant in ``sequence``.
        reference_bases (str): Reference allele, e.g. ``'A'`` or ``'AC'``.
        alternate_bases (str): Alternate allele, e.g. ``'G'`` or ``'GTT'``.
        strand (Literal['+', '-']): Strand to score on. Defaults to ``'+'``.
    """

    sequence: str = InputField(
        title="Sequence",
        description="Reference DNA window with >= 5000 bp of flank on each side of the variant",
    )
    variant_position: int = InputField(
        ge=0,
        title="Variant Position",
        description="0-based index of the variant within `sequence`",
    )
    reference_bases: str = InputField(title="Reference Bases", description="Reference allele, e.g. 'A' or 'AC'")
    alternate_bases: str = InputField(title="Alternate Bases", description="Alternate allele, e.g. 'G' or 'GTT'")
    strand: Literal["+", "-"] = InputField(
        default="+",
        title="Strand",
        description="Strand to score on ('+' or '-')",
    )

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, sequence: str) -> str:
        """Validate and normalize the reference window."""
        return validate_dna(sequence)

    @field_validator("reference_bases", "alternate_bases")
    @classmethod
    def validate_allele_bases(cls, bases: str, info: ValidationInfo) -> str:
        """Validate allele sequence characters (A/C/G/T/N, non-empty)."""
        normalized = bases.strip().upper()
        if not normalized:
            raise ValueError(f"{info.field_name}: cannot be empty")
        invalid = sorted(set(normalized) - set("ACGTN"))
        if invalid:
            raise ValueError(
                f"{info.field_name}: must only contain DNA bases A/C/G/T/N; got invalid {invalid} in {bases!r}"
            )
        if not set(normalized) & set("ACGT"):
            raise ValueError(f"{info.field_name}: must contain at least one A/C/G/T base; got {bases!r}")
        return normalized

    @model_validator(mode="after")
    def validate_variant(self) -> "PangolinVariant":
        """Check the variant format, allele/window match, and flanking context."""
        ref_len = len(self.reference_bases)
        alt_len = len(self.alternate_bases)
        if ref_len != 1 and alt_len != 1 and ref_len != alt_len:
            raise ValueError(
                "unsupported variant format: Pangolin scores SNVs and simple indels only "
                "(reference_bases or alternate_bases must be length 1, or both equal length); "
                f"got reference_bases length {ref_len}, alternate_bases length {alt_len}"
            )
        end = self.variant_position + ref_len
        if end > len(self.sequence):
            raise ValueError(
                f"variant_position + len(reference_bases) ({end}) exceeds sequence length ({len(self.sequence)})"
            )
        observed = self.sequence[self.variant_position : end]
        if observed != self.reference_bases:
            raise ValueError(
                f"reference_bases {self.reference_bases!r} does not match `sequence` at position "
                f"{self.variant_position} (found {observed!r})"
            )
        if self.variant_position < PANGOLIN_FLANK:
            raise ValueError(f"variant_position ({self.variant_position}) needs >= {PANGOLIN_FLANK} bp of 5' flank")
        if len(self.sequence) - end < PANGOLIN_FLANK:
            raise ValueError(f"variant needs >= {PANGOLIN_FLANK} bp of 3' flank (have {len(self.sequence) - end})")
        return self


class PangolinVariantMetrics(Metrics):
    """Scalar splice-effect metrics for a scored variant.

    Attributes:
        primary_metric (str | None): Defaults to ``"max_gain"``.

    Metrics documented in ``metric_spec``:
        max_gain (float): Largest splice gain across the window and tissues.
        max_loss (float): Largest splice loss (most negative) across the window and tissues.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "max_gain": {
            "availability": "always",
            "type": "float",
            "min": -1.0,
            "max": 1.0,
            "better_values_are": "context-dependent",
        },
        "max_loss": {
            "availability": "always",
            "type": "float",
            "min": -1.0,
            "max": 1.0,
            "better_values_are": "context-dependent",
        },
    }
    primary_metric: str | None = Field(
        default="max_gain",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )


class PangolinVariantEffect(BaseModel):
    """Pangolin splice-effect scores for one variant.

    Attributes:
        loss_scores (list[float]): Per-position splice-loss scores over the window.
        gain_scores (list[float]): Per-position splice-gain scores over the window.
        increase_position (int): Offset in bp from the variant of the largest increase.
        increase_score (float): Score at ``increase_position``.
        decrease_position (int): Offset in bp from the variant of the largest decrease.
        decrease_score (float): Score at ``decrease_position``.
        metrics (PangolinVariantMetrics): Scalar splice-effect summary metrics.
    """

    loss_scores: list[float] = Field(
        title="Loss Scores",
        description="Per-position splice-loss scores over the reporting window",
    )
    gain_scores: list[float] = Field(
        title="Gain Scores",
        description="Per-position splice-gain scores over the reporting window",
    )
    increase_position: int = Field(
        title="Increase Position",
        description="Position (bp, relative to variant) of the largest increase",
    )
    increase_score: float = Field(title="Increase Score", description="Largest splice-score increase")
    decrease_position: int = Field(
        title="Decrease Position",
        description="Position (bp, relative to variant) of the largest decrease",
    )
    decrease_score: float = Field(title="Decrease Score", description="Largest splice-score decrease")
    metrics: PangolinVariantMetrics = Field(
        title="Splice-Effect Metrics", description="Scalar splice-effect summary metrics"
    )


class PangolinScoreVariantsInput(BaseToolInput):
    """Input for Pangolin variant splice-effect scoring.

    Attributes:
        variants (list[PangolinVariant]): Variants to score. A single variant is
            auto-wrapped into a list.
    """

    variants: list[PangolinVariant] = InputField(
        title="Variants",
        description="Variants (each with its reference sequence window) to score",
    )

    @field_validator("variants", mode="before")
    @classmethod
    def normalize_variants(cls, value: Any) -> list[Any]:
        """Normalize a single variant to a one-item list."""
        if value is None:
            raise ValueError("variants cannot be None")
        if not isinstance(value, list):
            value = [value]
        if not value:
            raise ValueError("variants cannot be empty")
        return value  # type: ignore[no-any-return]


class PangolinScoreVariantsOutput(BaseToolOutput):
    """Output from Pangolin variant splice-effect scoring.

    Attributes:
        results (list[PangolinVariantEffect]): Per-variant splice-effect scores,
            1:1 with the input variants.
    """

    results: list[PangolinVariantEffect] = Field(
        title="Results",
        description="Per-variant splice-effect scores (1:1 with input variants)",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "json":
            with open(path, "w") as f:
                json.dump(self.model_dump(mode="json"), f, indent=2)
        elif file_format == "csv":
            fieldnames = [
                "increase_position",
                "increase_score",
                "decrease_position",
                "decrease_score",
                "max_gain",
                "max_loss",
            ]
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self.results:
                    writer.writerow(
                        {
                            "increase_position": r.increase_position,
                            "increase_score": r.increase_score,
                            "decrease_position": r.decrease_position,
                            "decrease_score": r.decrease_score,
                            "max_gain": r.metrics["max_gain"],
                            "max_loss": r.metrics["max_loss"],
                        }
                    )
        else:
            raise ValueError(f"Unsupported format: {file_format}")


class PangolinScoreVariantsConfig(PangolinConfig):
    """Configuration for Pangolin variant splice-effect scoring.

    Attributes:
        tissues (list[PangolinTissue]): Tissues whose splice predictions are
            ensembled. Defaults to all four Pangolin tissues.
        distance (int): Number of bp on each side of the variant included in the
            reporting window. Defaults to 50 (matching the Pangolin CLI).
    """

    distance: int = ConfigField(
        title="Distance",
        default=50,
        ge=1,
        description="bp on each side of the variant to report splice scores for",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input: one SNV in a 10,200 bp window."""
    sequence = "ACGT" * 2550
    position = 5100
    reference = sequence[position]
    alternate = "C" if reference != "C" else "G"
    return PangolinScoreVariantsInput(
        variants=[
            PangolinVariant(
                sequence=sequence,
                variant_position=position,
                reference_bases=reference,
                alternate_bases=alternate,
            )
        ]
    )


@tool(
    key="pangolin-score-variants",
    label="Pangolin Variant Splice Scoring",
    category="rna_splicing",
    input_class=PangolinScoreVariantsInput,
    config_class=PangolinScoreVariantsConfig,
    output_class=PangolinScoreVariantsOutput,
    description="Score the splicing effect (gain/loss) of variants using Pangolin",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["variants"],
    iterable_output_field="results",
    cacheable=True,
    metrics_class=PangolinVariantMetrics,
)
def run_pangolin_score_variants(
    inputs: PangolinScoreVariantsInput,
    config: PangolinScoreVariantsConfig,
    instance: Any = None,
) -> PangolinScoreVariantsOutput:
    """Score the splicing effect (gain/loss) of variants with Pangolin.

    For each variant, Pangolin compares predicted splice-site probability between
    the reference and alternate sequence over a ``± distance`` window, reducing
    across the selected tissues to a per-position splice gain (largest increase)
    and loss (largest decrease).

    Args:
        inputs (PangolinScoreVariantsInput): Validated variants with reference
            windows.
        config (PangolinScoreVariantsConfig): Pangolin configuration (tissues,
            reporting distance, device).
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        PangolinScoreVariantsOutput: Per-variant splice-effect scores, 1:1 with
            the input variants.

    See Also:
        - Pangolin paper: https://doi.org/10.1186/s13059-022-02664-4
        - Model repository: https://github.com/tkzeng/Pangolin

    Note:
        Annotation-based score masking (the CLI ``--mask`` option) is not
        supported because it requires exon annotations; raw gain/loss is returned.
    """
    logger.debug("Using local venv for Pangolin variant scoring (tissues=%s)", config.tissues)

    # Upstream skips deletions wider than the reporting window (len(ref) > 2*distance).
    for v in inputs.variants:
        if len(v.reference_bases) > 2 * config.distance:
            raise ValueError(
                f"reference_bases length {len(v.reference_bases)} exceeds 2*distance ({2 * config.distance}); "
                "increase `distance` or this deletion cannot be scored"
            )

    input_data = {
        "operation": "score_variants",
        "variants": [
            {
                "sequence": v.sequence,
                "variant_position": v.variant_position,
                "reference_bases": v.reference_bases,
                "alternate_bases": v.alternate_bases,
                "strand": v.strand,
            }
            for v in inputs.variants
        ],
        "tissues": config.tissues,
        "distance": config.distance,
        "device": config.device,
    }

    output_data = ToolInstance.dispatch("pangolin", input_data, instance=instance, config=config)

    results = [
        PangolinVariantEffect(
            loss_scores=item["loss_scores"],
            gain_scores=item["gain_scores"],
            increase_position=item["increase_position"],
            increase_score=item["increase_score"],
            decrease_position=item["decrease_position"],
            decrease_score=item["decrease_score"],
            metrics=PangolinVariantMetrics(max_gain=item["increase_score"], max_loss=item["decrease_score"]),
        )
        for _variant, item in zip(inputs.variants, output_data["results"], strict=True)
    ]
    return PangolinScoreVariantsOutput(
        results=results,
        metadata={"tissues": config.tissues, "distance": config.distance},
    )
