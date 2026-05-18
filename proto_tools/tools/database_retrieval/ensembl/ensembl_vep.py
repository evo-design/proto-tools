"""proto_tools/tools/database_retrieval/ensembl/ensembl_vep.py.

Wraps Ensembl's Variant Effect Predictor REST endpoint (HGVS form):
submits a notation, returns per-transcript consequence predictions.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from proto_tools.tools.database_retrieval.ensembl.shared_data_models import (
    EnsemblAssembly,
    EnsemblSpecies,
    base_url_for,
    build_session,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
)

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 30

# Snake-case field → VEP URL param. AlphaMissense / REVEL / CADD / Conservation /
# Phenotypes / Blosum62 / MaxEntScan must be sent CamelCase; lowercase is silently ignored.
# Split by source: ROOT fields live on EnsemblVEPConfig directly (species/assembly-restricted
# or non-bool); ANNOTATIONS fields live on the nested EnsemblVEPAnnotationConfig.
_VEP_ROOT_PARAM_MAP: dict[str, str] = {
    "mane": "mane",
    "sift": "sift",
    "polyphen": "polyphen",
    "appris": "appris",
    "tsl": "tsl",
    "ccds": "ccds",
    "distance": "distance",
    "pick": "pick",
    "per_gene": "per_gene",
    "alphamissense": "AlphaMissense",
    "revel": "REVEL",
    "cadd": "CADD",
}
_VEP_ANNOTATION_PARAM_MAP: dict[str, str] = {
    "canonical": "canonical",
    "hgvs": "hgvs",
    "protein": "protein",
    "domains": "domains",
    "numbers": "numbers",
    "variant_class": "variant_class",
    "uniprot": "uniprot",
    "xref_refseq": "xref_refseq",
    "mirna": "mirna",
    "pubmed": "pubmed",
    "conservation": "Conservation",
    "phenotypes": "Phenotypes",
    "blosum62": "Blosum62",
    "max_ent_scan": "MaxEntScan",
}


# ============================================================================
# Data Models
# ============================================================================


class EnsemblVEPConsequence(BaseModel):
    """One variant-effect prediction record.

    Attributes:
        input (str): HGVS string the API echoed back.
        most_severe_consequence (str): Highest-severity consequence term
            (Sequence Ontology: ``intron_variant``, ``missense_variant``,
            ``stop_gained``, …).
        seq_region_name (str | None): Chromosome / contig.
        start (int | None): 1-indexed inclusive genomic start.
        end (int | None): 1-indexed inclusive genomic end.
        strand (int | None): +1 or -1.
        allele_string (str | None): Reference/alternate alleles (e.g. 'G/C').
        transcript_consequences (list[dict[str, Any]]): Per-transcript
            consequence records — kept as raw dicts because the field set
            (sift_prediction, polyphen_prediction, codons, amino_acids, …)
            varies by consequence type and plugin configuration.
        colocated_variants (list[dict[str, Any]]): Co-located known variants
            (rsIDs, frequencies, clinical significance) — also kept raw.
    """

    model_config = ConfigDict(extra="ignore")

    input: str
    most_severe_consequence: str
    seq_region_name: str | None = None
    start: int | None = None
    end: int | None = None
    strand: int | None = None
    allele_string: str | None = None
    transcript_consequences: list[dict[str, Any]] = Field(default_factory=list)
    colocated_variants: list[dict[str, Any]] = Field(default_factory=list)


class EnsemblVEPInput(BaseToolInput):
    """Input for Ensembl VEP.

    Attributes:
        hgvs (str): HGVS notation. Genomic (e.g. ``9:g.22125504G>C``),
            coding (``ENST00000357654:c.5074G>A``), or protein
            (``ENSP00000418960:p.Tyr124Cys``) forms all work.
    """

    hgvs: str = InputField(description="HGVS notation (genomic / coding / protein)")

    @model_validator(mode="after")
    def validate_hgvs(self) -> "EnsemblVEPInput":
        """Reject empty / whitespace-only HGVS."""
        if not self.hgvs.strip():
            raise ValueError("hgvs must be a non-empty HGVS notation")
        return self


class EnsemblVEPAnnotationConfig(BaseConfig):
    """Optional VEP annotations to include in the response.

    Each field is a species-agnostic toggle. Species-restricted toggles
    (MANE, AlphaMissense, REVEL, CADD, APPRIS, TSL, CCDS) live on the
    parent :class:`EnsemblVEPConfig` so their ``depends_on`` form-hiding
    can reference ``species`` / ``assembly`` directly.

    Attributes:
        canonical (bool): Mark canonical Ensembl transcripts in each
            consequence record.
        hgvs (bool): Include HGVS notation per consequence.
        protein (bool): Include Ensembl protein identifiers.
        domains (bool): List overlapping protein domain names.
        numbers (bool): Include affected exon/intron numbers.
        variant_class (bool): Include Sequence Ontology variant class.
        uniprot (bool): Include UniProt accession for each transcript.
        xref_refseq (bool): Include RefSeq cross-reference IDs.
        mirna (bool): Include overlapping miRNA target sites.
        pubmed (bool): Include PubMed citation IDs.
        conservation (bool): Include conservation scores from EPO alignments.
        phenotypes (bool): Include overlapping phenotype/disease annotations.
        blosum62 (bool): Include BLOSUM62 substitution score for missense.
        max_ent_scan (bool): Include MaxEntScan splice-site scores.
    """

    canonical: bool = ConfigField(
        title="Canonical Transcripts",
        default=False,
        description="Mark canonical Ensembl transcripts in each consequence record",
    )
    hgvs: bool = ConfigField(title="HGVS Notation", default=False, description="Include HGVS notation per consequence")
    protein: bool = ConfigField(title="Protein IDs", default=False, description="Include Ensembl protein identifiers")
    domains: bool = ConfigField(
        title="Protein Domains", default=False, description="List overlapping protein domain names"
    )
    numbers: bool = ConfigField(
        title="Exon/Intron Numbers", default=False, description="Include affected exon/intron numbers per consequence"
    )
    variant_class: bool = ConfigField(
        title="Variant Class", default=False, description="Include Sequence Ontology variant classification"
    )
    uniprot: bool = ConfigField(
        title="UniProt Accessions", default=False, description="Include UniProt accession for each transcript"
    )
    xref_refseq: bool = ConfigField(
        title="RefSeq Xrefs", default=False, description="Include RefSeq cross-reference IDs per transcript"
    )
    mirna: bool = ConfigField(
        title="miRNA Targets", default=False, description="Include overlapping miRNA target sites"
    )
    pubmed: bool = ConfigField(
        title="PubMed Citations", default=False, description="Include PubMed citation IDs for each variant"
    )
    conservation: bool = ConfigField(
        title="Conservation Scores", default=False, description="Conservation scores from EPO multi-species alignments"
    )
    phenotypes: bool = ConfigField(
        title="Phenotype Annotations",
        default=False,
        description="Overlapping phenotype/disease annotations from ClinVar/dbSNP/etc.",
    )
    blosum62: bool = ConfigField(
        title="BLOSUM62 Score", default=False, description="BLOSUM62 substitution score for missense changes"
    )
    max_ent_scan: bool = ConfigField(
        title="MaxEntScan Splice Scores", default=False, description="MaxEntScan splice donor/acceptor scores"
    )


class EnsemblVEPConfig(BaseConfig):
    """Configuration for Ensembl VEP.

    Species-agnostic annotation toggles (canonical, hgvs, protein, …) are
    grouped under :class:`EnsemblVEPAnnotationConfig` and accessed via
    ``config.annotations.<field>``. Species-restricted ones stay at the
    root so their ``depends_on`` form-hiding can reference ``species`` /
    ``assembly`` directly.

    Attributes:
        species (EnsemblSpecies): Species slug. Default ``homo_sapiens``.
        assembly (EnsemblAssembly): Genome assembly. ``GRCh38`` (default)
            or ``GRCh37``.
        annotations (EnsemblVEPAnnotationConfig): Collapsible group of
            species-agnostic annotation toggles.
        sift (Literal['b', 'p', 's'] | None): SIFT pathogenicity output —
            ``b`` (both prediction + score), ``p`` (prediction only),
            ``s`` (score only); ``None`` falls back to API default.
        polyphen (Literal['b', 'p', 's'] | None): PolyPhen output level;
            same value semantics as ``sift``.
        mane (bool): Include MANE Select annotations (GRCh38 only).
        alphamissense (bool): AlphaMissense missense pathogenicity scores
            (human only).
        revel (bool): REVEL ensemble pathogenicity scores (human only).
        cadd (bool): CADD deleteriousness scores (human only).
        appris (bool): Include APPRIS principal isoform tag (human/mouse
            only).
        tsl (bool): Include transcript support level (human/mouse only).
        ccds (bool): Include CCDS identifier per transcript (human/mouse
            only).
        distance (int | None): Up/downstream distance (bp) used to assign
            consequence terms. ``None`` keeps the API default (5000).
        pick (bool): Return only one consequence per variant — Ensembl's
            PICK heuristic (canonical, longest CDS, …).
        per_gene (bool): Return one consequence per gene (less aggressive
            than ``pick``); incompatible with ``pick``.
    """

    species: EnsemblSpecies = ConfigField(title="Species", default="homo_sapiens", description="Species slug for VEP")
    assembly: EnsemblAssembly = ConfigField(
        title="Assembly", default="GRCh38", description="Genome assembly; GRCh37 routes to grch37.rest.ensembl.org"
    )
    annotations: EnsemblVEPAnnotationConfig = ConfigField(
        title="Annotations",
        default_factory=EnsemblVEPAnnotationConfig,
        description="Optional VEP annotations to include in the response",
    )
    sift: Literal["b", "p", "s"] | None = ConfigField(
        title="SIFT Output",
        default=None,
        description="SIFT output level: b=both, p=prediction, s=score; None = API default",
    )
    polyphen: Literal["b", "p", "s"] | None = ConfigField(
        title="PolyPhen Output",
        default=None,
        description="PolyPhen output level: b=both, p=prediction, s=score; None = API default",
    )
    mane: bool = ConfigField(
        title="MANE Select", default=False, description="Include MANE Select annotations (GRCh38 only)"
    )
    alphamissense: bool = ConfigField(
        title="AlphaMissense Scores",
        default=False,
        description="AlphaMissense missense pathogenicity scores (human only)",
    )
    revel: bool = ConfigField(
        title="REVEL Scores", default=False, description="REVEL ensemble pathogenicity scores (human only)"
    )
    cadd: bool = ConfigField(title="CADD Scores", default=False, description="CADD deleteriousness scores (human only)")
    appris: bool = ConfigField(
        title="APPRIS Tag", default=False, description="APPRIS principal-isoform tag per transcript (human/mouse only)"
    )
    tsl: bool = ConfigField(
        title="Transcript Support Level",
        default=False,
        description="Transcript support level (TSL) per transcript (human/mouse only)",
    )
    ccds: bool = ConfigField(
        title="CCDS IDs",
        default=False,
        description="Include CCDS identifier per transcript (human/mouse only — CCDS is a human/mouse project)",
    )
    distance: int | None = ConfigField(
        title="Up/Downstream Distance (bp)",
        default=None,
        ge=0,
        description="Bases up/downstream considered for consequence terms (None = API default 5000)",
    )
    pick: bool = ConfigField(
        title="Pick One Consequence",
        default=False,
        description="Collapse output to one consequence per variant via Ensembl PICK heuristic",
    )
    per_gene: bool = ConfigField(
        title="One Consequence per Gene",
        default=False,
        description="Collapse output to one consequence per gene; incompatible with pick",
    )

    @model_validator(mode="after")
    def _check_pick_per_gene(self) -> "EnsemblVEPConfig":
        """Reject ``pick`` and ``per_gene`` set together (Ensembl rejects this)."""
        if self.pick and self.per_gene:
            raise ValueError("Set either 'pick' or 'per_gene', not both")
        return self


class EnsemblVEPOutput(BaseToolOutput):
    """Output from Ensembl VEP.

    Attributes:
        consequences (list[EnsemblVEPConsequence]): One record per VEP
            input (Ensembl returns a list even for a single HGVS).
        num_consequences (int): ``len(consequences)``.
        source_url (str): Final URL hit.
        raw_payload (list[dict[str, Any]]): Raw API JSON.
    """

    consequences: list[EnsemblVEPConsequence] = Field(
        default_factory=list, description="One record per VEP input (Ensembl returns a list even for a single HGVS)"
    )
    source_url: str = Field(description="Final Ensembl REST URL that was hit")
    raw_payload: list[dict[str, Any]] = Field(default_factory=list, description="Raw API JSON")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_consequences(self) -> int:
        """``len(self.consequences)`` — derived so the count can't drift."""
        return len(self.consequences)

    @property
    def output_format_options(self) -> list[str]:
        """Return supported output formats."""
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Any, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")
        if file_format == "json":
            with path.open("w", encoding="utf-8") as f:
                json.dump(self.model_dump(mode="json"), f, indent=2)
            return
        if file_format == "csv":
            # One row per consequence; nested transcript_consequences and
            # colocated_variants are JSON-encoded into single cells.
            rows = []
            for c in self.consequences:
                d = c.model_dump()
                d["transcript_consequences"] = json.dumps(d["transcript_consequences"], separators=(",", ":"))
                d["colocated_variants"] = json.dumps(d["colocated_variants"], separators=(",", ":"))
                rows.append(d)
            with path.open("w", encoding="utf-8", newline="") as f:
                if not rows:
                    return
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            return
        raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return EnsemblVEPInput(hgvs="9:g.22125504G>C")


@tool(
    key="ensembl-vep",
    label="Ensembl VEP",
    category="database_retrieval",
    input_class=EnsemblVEPInput,
    config_class=EnsemblVEPConfig,
    output_class=EnsemblVEPOutput,
    description=(
        "Predict variant consequences from an HGVS notation via Ensembl's Variant Effect Predictor REST endpoint"
    ),
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_ensembl_vep(
    inputs: EnsemblVEPInput,
    config: EnsemblVEPConfig,
    instance: Any = None,
) -> EnsemblVEPOutput:
    """Submit an HGVS notation to Ensembl VEP and parse the consequence list.

    Args:
        inputs (EnsemblVEPInput): HGVS notation (genomic / coding / protein).
        config (EnsemblVEPConfig): Species + assembly.
        instance (Any): Optional ToolInstance; unused for HTTP-only tools.

    Returns:
        EnsemblVEPOutput: One ``EnsemblVEPConsequence`` per record returned
            by the API (typically one for a single-HGVS query).
    """
    del instance

    base = base_url_for(config.assembly)
    url = f"{base}/vep/{config.species}/hgvs/{quote(inputs.hgvs.strip(), safe='')}"
    params = _build_vep_params(config)

    session = build_session("ensembl-vep")
    try:
        response = session.get(
            url,
            params=params,
            headers={"Accept": "application/json"},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError(
                f"Ensembl VEP returned non-JSON for {inputs.hgvs!r} at {response.url}; "
                f"body[:200]={response.text[:200]!r}"
            ) from exc
        if not isinstance(payload, list):
            raise ValueError(f"Ensembl VEP returned non-list payload: {type(payload).__name__}")
        consequences: list[EnsemblVEPConsequence] = []
        for i, c in enumerate(payload):
            if not isinstance(c, dict):
                raise ValueError(f"Ensembl VEP payload[{i}] is non-dict: {type(c).__name__}")
            consequences.append(EnsemblVEPConsequence.model_validate(c))
        return EnsemblVEPOutput(
            consequences=consequences,
            source_url=response.url,
            raw_payload=payload,
        )
    finally:
        session.close()


# ============================================================================
# Private Helpers
# ============================================================================


def _build_vep_params(config: EnsemblVEPConfig) -> dict[str, str]:
    """Translate set flags into VEP query params (bool→"1"; str/int→str; None omitted)."""
    params: dict[str, str] = {}
    for source, mapping in (
        (config, _VEP_ROOT_PARAM_MAP),
        (config.annotations, _VEP_ANNOTATION_PARAM_MAP),
    ):
        for field, api_name in mapping.items():
            value = getattr(source, field)
            if value is True:
                params[api_name] = "1"
            elif isinstance(value, str) and value:
                params[api_name] = value
            elif isinstance(value, int) and not isinstance(value, bool):
                params[api_name] = str(value)
    return params
