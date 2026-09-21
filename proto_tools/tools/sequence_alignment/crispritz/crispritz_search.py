"""CRISPRitz mismatch- and bulge-aware CRISPR off-target search."""

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, BaseToolInput, BaseToolOutput, ConfigField, InputField, ToolInstance

CRISPRITZ_VERSION = "2.8.1"
CRISPRITZ_COMMIT = "1b4cfca5059bb4fc3230b583badafe64494e2cce"
IUPAC_DNA = set("ACGTRYSWKMBDHVN")


class CRISPRitzSearchInput(BaseToolInput):
    """Guide protospacers and a local reference genome to search.

    ``guides`` contains protospacers only: do not append the PAM or CRISPRitz's
    placeholder ``N`` bases. All guides in one call must have the same length.
    RNA ``U`` is accepted and normalized to DNA ``T``. ``reference_fasta`` may
    name one FASTA file, including a multi-record NCBI ``.fna``, or a directory
    recursively containing ``.fa``, ``.fasta``, or ``.fna`` files. Compressed
    FASTA is not accepted. The reference is read-only and may be the genome file
    returned inside an ``ncbi-reference-bundle`` artifact.

    Attributes:
        guides (list[str]): PAM-free guide protospacers to search.
        reference_fasta (str): Local uncompressed reference FASTA path.
        reference_id (str | None): Optional versioned reference identifier.
    """

    guides: list[str] = InputField(
        title="Guide Protospacers",
        description="PAM-free guide DNA/RNA protospacers; all entries must have the same length",
        min_length=1,
    )
    reference_fasta: str = InputField(
        title="Reference FASTA",
        description="Local uncompressed FASTA file or directory of .fa/.fasta/.fna files",
    )
    reference_id: str | None = InputField(
        default=None,
        title="Reference Identifier",
        description="Optional versioned assembly or accession identifier retained as provenance",
    )

    @field_validator("guides", mode="before")
    @classmethod
    def normalize_guide_list(cls, value: Any) -> list[Any]:
        """Normalize one guide string to a one-item batch."""
        return [value] if isinstance(value, str) else value

    @field_validator("guides")
    @classmethod
    def validate_guides(cls, guides: list[str]) -> list[str]:
        """Normalize guide alphabet and enforce a uniform protospacer length."""
        normalized = [guide.strip().upper().replace("U", "T") for guide in guides]
        for index, guide in enumerate(normalized):
            if not guide:
                raise ValueError(f"guides[{index}] cannot be empty")
            invalid = sorted(set(guide) - set("ACGT"))
            if invalid:
                raise ValueError(f"guides[{index}] contains non-canonical bases: {invalid}")
        lengths = {len(guide) for guide in normalized}
        if len(lengths) != 1:
            raise ValueError("all guides in one CRISPRitz search must have the same length")
        return normalized

    def __len__(self) -> int:
        return len(self.guides)


class CRISPRitzSearchConfig(BaseConfig):
    """CRISPRitz search and index parameters.

    ``pam`` is written in nuclease orientation and excludes the protospacer.
    Use ``pam_side='three_prime'`` for SpCas9 ``NGG`` and
    ``pam_side='five_prime'`` for Cas12a ``TTTV``. CRISPRitz searches both
    genomic strands. Mismatch and bulge limits are inclusive maxima.

    Attributes:
        pam (str): IUPAC PAM motif without protospacer bases.
        pam_side (Literal["three_prime", "five_prime"]): PAM position relative to the protospacer.
        nuclease (str): Nuclease identity retained as provenance.
        max_mismatches (int): Inclusive substitution limit per hit.
        max_dna_bulges (int): Inclusive DNA-bulge limit per hit.
        max_rna_bulges (int): Inclusive RNA-bulge limit per hit.
        max_total_edits (int | None): Optional inclusive limit on all edits.
        threads (int): CPU threads used for indexing and search.
        index_cache_dir (str | None): Optional persistent CRISPRitz index cache.
    """

    pam: str = ConfigField(default="NGG", title="PAM", description="IUPAC PAM motif without protospacer bases")
    pam_side: Literal["three_prime", "five_prime"] = ConfigField(
        default="three_prime", title="PAM Side", description="PAM position relative to the protospacer"
    )
    nuclease: str = ConfigField(
        default="SpCas9", title="Nuclease", description="Descriptive nuclease identity retained in provenance"
    )
    max_mismatches: int = ConfigField(
        default=4, ge=0, title="Maximum Mismatches", description="Inclusive substitution limit per hit"
    )
    max_dna_bulges: int = ConfigField(
        default=1, ge=0, title="Maximum DNA Bulges", description="Inclusive DNA bulge-length limit per hit"
    )
    max_rna_bulges: int = ConfigField(
        default=1, ge=0, title="Maximum RNA Bulges", description="Inclusive RNA bulge-length limit per hit"
    )
    max_total_edits: int | None = ConfigField(
        default=None,
        ge=0,
        title="Maximum Total Edits",
        description="Optional inclusive cap on mismatches plus bulge length",
    )
    threads: int = ConfigField(default=4, ge=1, title="Threads", description="CPU threads for indexing and search")
    index_cache_dir: str | None = ConfigField(
        default=None,
        title="Index Cache Directory",
        description="Persistent checksum-keyed CRISPRitz index cache; defaults under PROTO_HOME",
    )

    @field_validator("pam")
    @classmethod
    def validate_pam(cls, pam: str) -> str:
        """Normalize and validate the IUPAC PAM."""
        normalized = pam.strip().upper().replace("U", "T")
        if not normalized:
            raise ValueError("pam cannot be empty")
        invalid = sorted(set(normalized) - IUPAC_DNA)
        if invalid:
            raise ValueError(f"pam contains invalid IUPAC bases: {invalid}")
        return normalized


class CRISPRitzReferenceFile(BaseModel):
    """One FASTA source included in the indexed reference."""

    path: str = Field(title="Path", description="Resolved source FASTA path")
    sha256: str = Field(title="SHA256", description="SHA256 checksum of the source FASTA bytes")
    size_bytes: int = Field(title="Size Bytes", description="Source FASTA size in bytes")


class CRISPRitzHit(BaseModel):
    """One CRISPRitz target or off-target alignment."""

    guide: str = Field(title="Guide", description="Normalized PAM-free guide protospacer")
    bulge_type: Literal["none", "dna", "rna", "dna_rna"] = Field(
        title="Bulge Type", description="Normalized alignment indel class"
    )
    crispritz_bulge_type: str = Field(
        title="CRISPRitz Bulge Type", description="Unmodified upstream X, DNA, RNA, or RNA,DNA label"
    )
    guide_alignment: str = Field(title="Guide Alignment", description="CRISPRitz crRNA alignment")
    target_alignment: str = Field(title="Target Alignment", description="CRISPRitz genomic alignment")
    chromosome: str = Field(title="Chromosome", description="Reference FASTA record identifier")
    position: int = Field(title="Position", description="Zero-based CRISPRitz genomic start coordinate")
    cluster_position: int = Field(title="Cluster Position", description="Zero-based CRISPRitz cluster coordinate")
    strand: Literal["+", "-"] = Field(title="Strand", description="Genomic target strand")
    mismatches: int = Field(title="Mismatches", description="Substitution count excluding bulges")
    bulge_size: int = Field(title="Bulge Size", description="Bulge length")
    total_edits: int = Field(title="Total Edits", description="Mismatches plus bulge length")


class CRISPRitzSearchOutput(BaseToolOutput):
    """Structured mismatch- and bulge-aware off-target enumeration.

    Attributes:
        hits (list[CRISPRitzHit]): Enumerated target and off-target sites.
        guides (list[str]): Normalized queried protospacers.
        reference_id (str | None): Optional reference accession.
        reference_sha256 (str): Combined reference FASTA checksum.
        reference_files (list[CRISPRitzReferenceFile]): Checksummed reference files.
        index_key (str): Content- and parameter-addressed index key.
        index_reused (bool): Whether a completed index was reused.
        crispritz_version (str): Pinned CRISPRitz version.
        crispritz_commit (str): Verified CRISPRitz source revision.
        nuclease (str): Caller-supplied nuclease identity.
        pam (str): IUPAC PAM motif.
        pam_side (Literal["three_prime", "five_prime"]): PAM orientation.
        max_mismatches (int): Search substitution limit.
        max_dna_bulges (int): Search DNA-bulge limit.
        max_rna_bulges (int): Search RNA-bulge limit.
        max_total_edits (int | None): Optional total-edit limit.
    """

    hits: list[CRISPRitzHit] = Field(default_factory=list, title="Hits", description="Enumerated target sites")
    guides: list[str] = Field(title="Guides", description="Normalized PAM-free queried protospacers")
    reference_id: str | None = Field(default=None, title="Reference Identifier", description="Reference accession")
    reference_sha256: str = Field(title="Reference SHA256", description="Combined reference FASTA checksum")
    reference_files: list[CRISPRitzReferenceFile] = Field(
        title="Reference Files", description="Checksummed source FASTA files"
    )
    index_key: str = Field(title="Index Key", description="Content- and parameter-addressed index key")
    index_reused: bool = Field(title="Index Reused", description="Whether an existing completed index was reused")
    crispritz_version: str = Field(title="CRISPRitz Version", description="Pinned upstream version")
    crispritz_commit: str = Field(title="CRISPRitz Commit", description="Verified upstream source revision")
    nuclease: str = Field(title="Nuclease", description="Caller-supplied nuclease identity")
    pam: str = Field(title="PAM", description="IUPAC PAM motif")
    pam_side: Literal["three_prime", "five_prime"] = Field(title="PAM Side", description="PAM orientation")
    max_mismatches: int = Field(title="Maximum Mismatches", description="Search substitution limit")
    max_dna_bulges: int = Field(title="Maximum DNA Bulges", description="Search DNA-bulge limit")
    max_rna_bulges: int = Field(title="Maximum RNA Bulges", description="Search RNA-bulge limit")
    max_total_edits: int | None = Field(title="Maximum Total Edits", description="Optional total-edit limit")

    def __len__(self) -> int:
        return len(self.hits)

    def __iter__(self) -> Iterator[CRISPRitzHit]:  # type: ignore[override]
        return iter(self.hits)

    @property
    def output_format_options(self) -> list[str]:
        """Return supported export formats."""
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        """Return the default export format."""
        return "json"

    def _export_output(self, export_path: Path | str, file_format: str) -> None:
        """Export full JSON provenance or a flat hit CSV."""
        path = Path(export_path).with_suffix(f".{file_format}")
        if file_format == "json":
            path.write_text(json.dumps(self.model_dump(mode="json"), indent=2))
            return
        if file_format == "csv":
            with path.open("w", newline="") as handle:
                fieldnames = list(CRISPRitzHit.model_fields)
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(hit.model_dump() for hit in self.hits)
            return
        raise ValueError(f"Unsupported format: {file_format}")


def example_input() -> CRISPRitzSearchInput:
    """Return a documented local-reference example."""
    return CRISPRitzSearchInput(
        guides=["GACTGCATTGCATGACTGCA"],
        reference_fasta=str(Path(__file__).parent / "example_reference.fasta"),
        reference_id="synthetic-crispritz-example-v1",
    )


@tool(
    key="crispritz-search",
    local_only=(
        "crispritz-search reads a caller-supplied local reference and maintains a local checksum-keyed index; "
        "run it locally with device='cpu'"
    ),
    label="CRISPRitz Off-Target Search",
    category="sequence_alignment",
    input_class=CRISPRitzSearchInput,
    config_class=CRISPRitzSearchConfig,
    output_class=CRISPRitzSearchOutput,
    description="Enumerate CRISPR targets with explicit mismatches and DNA/RNA bulges",
    uses_gpu=False,
    example_input=example_input,
    cacheable=False,
)
def run_crispritz_search(
    inputs: CRISPRitzSearchInput, config: CRISPRitzSearchConfig, instance: Any = None
) -> CRISPRitzSearchOutput:
    """Run one native batched CRISPRitz search against a local reference."""
    reference_path = Path(inputs.reference_fasta).expanduser()
    if not reference_path.exists():
        raise FileNotFoundError(f"reference_fasta does not exist: {reference_path}")
    payload = ToolInstance.dispatch(
        "crispritz",
        {
            "guides": inputs.guides,
            "reference_fasta": inputs.reference_fasta,
            "pam": config.pam,
            "pam_side": config.pam_side,
            "max_mismatches": config.max_mismatches,
            "max_dna_bulges": config.max_dna_bulges,
            "max_rna_bulges": config.max_rna_bulges,
            "max_total_edits": config.max_total_edits,
            "threads": config.threads,
            "index_cache_dir": config.index_cache_dir,
            "device": "cpu",
        },
        instance=instance,
        config=config,
    )
    return CRISPRitzSearchOutput(
        hits=[CRISPRitzHit.model_validate(hit) for hit in payload["hits"]],
        guides=inputs.guides,
        reference_id=inputs.reference_id,
        reference_sha256=payload["reference_sha256"],
        reference_files=[CRISPRitzReferenceFile.model_validate(item) for item in payload["reference_files"]],
        index_key=payload["index_key"],
        index_reused=payload["index_reused"],
        crispritz_version=CRISPRITZ_VERSION,
        crispritz_commit=CRISPRITZ_COMMIT,
        nuclease=config.nuclease,
        pam=config.pam,
        pam_side=config.pam_side,
        max_mismatches=config.max_mismatches,
        max_dna_bulges=config.max_dna_bulges,
        max_rna_bulges=config.max_rna_bulges,
        max_total_edits=config.max_total_edits,
        metadata={"coordinate_system": "zero_based_crispritz_native"},
    )
