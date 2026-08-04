"""SpliceAI variant delta-score annotation."""

import csv
import json
import logging
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from proto_tools.databases.assets import dataset_file, is_registered_dataset
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)
from proto_tools.utils.device import RemoteDevice
from proto_tools.utils.tool_io import Metrics, MetricSpec, MissingAssetError

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================
DEFAULT_ANNOTATION = "grch38"
# SpliceAI's -D flag accepts 0..4999; the model sees 5000 bp of context per side.
MAX_SPLICEAI_DISTANCE = 4999

# Assemblies with a registered dataset, so naming one is all a caller needs to do. The names double
# as SpliceAI's bundled annotation keys, which is what lets `annotation` default to the assembly.
ProvisionedGenome = Literal["grch37", "grch38"]

# The FASTA within each registered genome dataset.
_GENOME_FASTA: dict[str, str] = {
    "grch37": "Homo_sapiens.GRCh37.dna.primary_assembly.fa",
    "grch38": "Homo_sapiens.GRCh38.dna.primary_assembly.fa",
}

# UCSC spellings of the same assemblies. Callers think in these; the bundled annotations do not.
_GENOME_ALIASES: dict[str, str] = {"hg19": "grch37", "hg38": "grch38"}


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
        reference_fasta (ProvisionedGenome | str | None): The reference genome
            SpliceAI reads the wild-type sequence around each variant from.
            Either a provisioned assembly name (``'grch38'``/``'grch37'``, or
            their UCSC spellings ``'hg38'``/``'hg19'``), downloaded on first use
            and the only form a remote worker can resolve, or a path to a FASTA
            on this machine. Required at call time; ``None`` raises a
            ``MissingAssetError`` so un-provisioned hosts skip cleanly.
        annotation (str): Gene annotation source: ``'grch37'`` or ``'grch38'``
            (GENCODE files bundled with SpliceAI) or a path to a custom
            tab-separated annotation file. Defaults to the named assembly when
            ``reference_fasta`` gives one, since the two describe one build.
        max_distance (int): Maximum distance (bp) between the variant and a
            gained/lost splice site to report (the SpliceAI ``-D`` flag).
        mask (bool): Mask scores for annotated acceptor/donor gain and
            unannotated acceptor/donor loss (the SpliceAI ``-M`` flag).
        device (str): Device to run inference on. SpliceAI (TensorFlow)
            auto-falls-back to CPU when no GPU is visible.
    """

    reference_fasta: ProvisionedGenome | str | None = ConfigField(
        title="Reference FASTA",
        default=None,
        description="Assembly name ('grch38'/'grch37', provisioned on demand) or a local FASTA path",
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

    @field_validator("reference_fasta", mode="before")
    @classmethod
    def _normalize_assembly_alias(cls, value: Any) -> Any:
        """Accept the UCSC spelling of a provisioned assembly, e.g. ``hg38`` for ``grch38``."""
        if isinstance(value, str):
            return _GENOME_ALIASES.get(value.strip().lower(), value)
        return value

    @field_validator("reference_fasta")
    @classmethod
    def _validate_genome_or_path(cls, value: str | None) -> str | None:
        """Require a value that is not a provisioned assembly to be a FASTA on this machine.

        Catches a typo where it is made rather than several minutes later, and keeps the two forms
        of the field distinguishable: anything not registered is a path, so it is local-only.

        Safe to check the filesystem here only because :meth:`remote_unsupported_reason` refuses a
        path on a remote device. Were that guard dropped, a container rebuilding this config from
        its transport dict would fail here on a path that never existed on its side.
        """
        if value is None or is_registered_dataset(value):
            return value
        if not Path(value).expanduser().exists():
            raise ValueError(
                f"reference_fasta: {value!r} is neither a provisioned assembly "
                f"({', '.join(sorted(_GENOME_FASTA))}) nor an existing file on this machine"
            )
        return value

    @model_validator(mode="after")
    def _derive_annotation_from_assembly(self) -> "SpliceAIScoreConfig":
        """Default the annotation to the named assembly, so the two cannot silently disagree.

        A variant's coordinates mean a different place in each assembly, so an annotation from one
        build against a genome from another is wrong rather than merely inconsistent. Naming a
        provisioned assembly settles both; an explicitly set ``annotation`` still wins.
        """
        if "annotation" not in self.model_fields_set and self.reference_fasta in _GENOME_FASTA:
            self.annotation = self.reference_fasta
        return self

    def remote_unsupported_reason(self, device: RemoteDevice) -> str | None:
        """A genome given as a path lives on the caller's machine, so only a named assembly travels."""
        if self.reference_fasta is not None and not is_registered_dataset(self.reference_fasta):
            return (
                f"reference_fasta={self.reference_fasta!r} is a local path, which can't be staged to "
                f"device='{device}'. Name a provisioned assembly ({', '.join(sorted(_GENOME_FASTA))}) "
                f"instead, or run locally with device='cpu'."
            )
        return None

    @classmethod
    def minimal(cls, **kwargs: Any) -> "SpliceAIScoreConfig":
        """Cheap-mode defaults: name the default assembly, since the tool cannot run without one.

        Naming it costs nothing where it is already staged, and provisions it where it is not —
        which is what lets parametrized infrastructure run this tool at all rather than reporting
        it as an unprovisioned asset it has no way to satisfy.
        """
        kwargs.setdefault("reference_fasta", DEFAULT_ANNOTATION)
        return super().minimal(**kwargs)  # type: ignore[return-value]

    def resolved_reference_fasta(self) -> Path | None:
        """Return the FASTA to read, provisioning the assembly on first use.

        Returns:
            Path | None: The resolved FASTA, or ``None`` when nothing was configured. The path may
                not exist when provisioning was unavailable; the caller reports that as a
                ``MissingAssetError`` rather than treating it as a failure.
        """
        if self.reference_fasta is None:
            return None
        if is_registered_dataset(self.reference_fasta):
            return dataset_file(self.reference_fasta, _GENOME_FASTA[self.reference_fasta])
        return Path(self.reference_fasta).expanduser()


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input: a BRCA1 splice-region variant, 5 nt into an intron (GRCh38)."""
    # A real locus rather than a placeholder, so the example returns scores instead of the empty
    # result a position outside any gene gives. GRCh38 coordinates, matching the default assembly.
    return SpliceAIScoreInput(variants=[SpliceAIVariant(chromosome="17", position=43051122, ref="A", alt="G")])


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
    max_chunk_size=64,
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
    # Resolved once here, so a named assembly is provisioned before the worker starts and the
    # standalone only ever sees a concrete path.
    resolved_fasta = config.resolved_reference_fasta()
    if resolved_fasta is None or not resolved_fasta.exists():
        raise MissingAssetError(
            "spliceai",
            "reference",
            f"reference_fasta not provided or not found: {config.reference_fasta!r}. "
            f"SpliceAI requires a reference genome FASTA — name a provisioned assembly "
            f"({', '.join(sorted(_GENOME_FASTA))}) or set SpliceAIScoreConfig.reference_fasta to a local path.",
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
            # The resolved path, not the configured name: the standalone caches its Annotator on
            # this value, and a name and its path would otherwise be two keys for one genome —
            # each miss rebuilding an Annotator over a multi-gigabyte FASTA.
            "reference_fasta": str(resolved_fasta),
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
