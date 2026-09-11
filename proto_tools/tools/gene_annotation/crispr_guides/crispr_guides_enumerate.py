"""Deterministic PAM-aware CRISPR guide enumeration."""

import csv
import hashlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, BaseToolInput, BaseToolOutput, ConfigField, InputField

logger = logging.getLogger(__name__)

IUPAC_DNA = frozenset("ACGTRYSWKMBDHVN")
IUPAC_BASES = {
    "A": frozenset("A"),
    "C": frozenset("C"),
    "G": frozenset("G"),
    "T": frozenset("T"),
    "R": frozenset("AG"),
    "Y": frozenset("CT"),
    "S": frozenset("CG"),
    "W": frozenset("AT"),
    "K": frozenset("GT"),
    "M": frozenset("AC"),
    "B": frozenset("CGT"),
    "D": frozenset("AGT"),
    "H": frozenset("ACT"),
    "V": frozenset("ACG"),
    "N": frozenset("ACGT"),
}
COMPLEMENT = str.maketrans("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN")


class CRISPRGuideRegion(BaseModel):
    """One reference-oriented sequence interval to scan.

    ``sequence`` must be written in the reference plus-strand orientation.
    ``start`` is the 1-indexed inclusive coordinate of its first base. The end
    coordinate is derived as ``start + len(sequence) - 1``.

    Attributes:
        sequence_id (str): Reference record or contig identifier.
        sequence (str): DNA interval in reference plus-strand orientation.
        start (int): 1-indexed inclusive coordinate of the first base.
        reference_id (str | None): Optional versioned assembly identifier.
        reference_sha256 (str | None): Optional checksum of the source reference artifact.
        expected_sequence_sha256 (str | None): Optional checksum of the normalized interval sequence.
    """

    sequence_id: str = Field(title="Sequence Identifier", description="Reference record or contig identifier")
    sequence: str = Field(title="Sequence", description="DNA interval in reference plus-strand orientation")
    start: int = Field(default=1, ge=1, title="Start", description="1-indexed inclusive first-base coordinate")
    reference_id: str | None = Field(
        default=None, title="Reference Identifier", description="Optional versioned assembly identifier"
    )
    reference_sha256: str | None = Field(
        default=None, title="Reference SHA256", description="Optional checksum of the source reference artifact"
    )
    expected_sequence_sha256: str | None = Field(
        default=None, title="Expected Sequence SHA256", description="Optional normalized-sequence checksum"
    )

    @field_validator("sequence")
    @classmethod
    def normalize_sequence(cls, value: str) -> str:
        """Normalize DNA and reject symbols outside the IUPAC alphabet."""
        normalized = "".join(value.split()).upper().replace("U", "T")
        if not normalized:
            raise ValueError("sequence cannot be empty")
        invalid = sorted(set(normalized) - IUPAC_DNA)
        if invalid:
            raise ValueError(f"sequence contains invalid IUPAC DNA bases: {invalid}")
        return normalized

    @field_validator("reference_sha256", "expected_sequence_sha256")
    @classmethod
    def normalize_checksum(cls, value: str | None) -> str | None:
        """Normalize and validate an optional SHA256 digest."""
        if value is None:
            return None
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
            raise ValueError("SHA256 values must contain exactly 64 hexadecimal characters")
        return normalized

    @model_validator(mode="after")
    def verify_sequence_checksum(self) -> "CRISPRGuideRegion":
        """Reject a sequence that does not match its expected digest."""
        observed = hashlib.sha256(self.sequence.encode()).hexdigest()
        if self.expected_sequence_sha256 is not None and observed != self.expected_sequence_sha256:
            raise ValueError("sequence does not match expected_sequence_sha256")
        return self


class CRISPRGuideEnumerationInput(BaseToolInput):
    """Reference intervals to scan for PAM-compatible protospacers.

    Attributes:
        regions (list[CRISPRGuideRegion]): Independently scanned reference intervals.
    """

    regions: list[CRISPRGuideRegion] = InputField(
        title="Regions", description="Reference-oriented DNA intervals to scan", min_length=1
    )

    @field_validator("regions", mode="before")
    @classmethod
    def normalize_regions(cls, value: Any) -> list[Any]:
        """Normalize one region object or dictionary to a one-item batch."""
        return value if isinstance(value, list) else [value]

    def __len__(self) -> int:
        return len(self.regions)


class CRISPRGuideEnumerationConfig(BaseConfig):
    """PAM grammar, guide length, strands, and returned context.

    ``pam`` is written in nuclease orientation. For a three-prime PAM the
    oriented target is ``protospacer + PAM``; for a five-prime PAM it is
    ``PAM + protospacer``. The default 4-nt and 3-nt exterior flanks produce a
    30-nt Rule Set 3 context for a 20-nt SpCas9 protospacer and 3-nt PAM.

    Attributes:
        pam (str): IUPAC PAM motif without protospacer placeholder bases.
        pam_side (Literal['three_prime', 'five_prime']): PAM orientation.
        protospacer_length (int): Number of PAM-free guide bases.
        strands (Literal['both', 'plus', 'minus']): Reference PAM strands to scan.
        context_5p (int): Exterior 5-prime context bases to return.
        context_3p (int): Exterior 3-prime context bases to return.
        nuclease (str): Descriptive nuclease identity retained as provenance.
    """

    pam: str = ConfigField(
        default="NGG", title="PAM", description="IUPAC PAM motif without protospacer placeholder bases"
    )
    pam_side: Literal["three_prime", "five_prime"] = ConfigField(
        default="three_prime", title="PAM Side", description="PAM position relative to the protospacer"
    )
    protospacer_length: int = ConfigField(
        default=20, ge=1, title="Protospacer Length", description="Number of PAM-free guide bases"
    )
    strands: Literal["both", "plus", "minus"] = ConfigField(
        default="both", title="Strands", description="Reference PAM-bearing strands to scan"
    )
    context_5p: int = ConfigField(
        default=4, ge=0, title="5-prime Context", description="Exterior 5-prime context bases to return"
    )
    context_3p: int = ConfigField(
        default=3, ge=0, title="3-prime Context", description="Exterior 3-prime context bases to return"
    )
    nuclease: str = ConfigField(
        default="SpCas9", title="Nuclease", description="Descriptive nuclease identity retained as provenance"
    )

    @field_validator("pam")
    @classmethod
    def normalize_pam(cls, value: str) -> str:
        """Normalize and validate an IUPAC PAM motif."""
        normalized = value.strip().upper().replace("U", "T")
        if not normalized:
            raise ValueError("pam cannot be empty")
        invalid = sorted(set(normalized) - IUPAC_DNA)
        if invalid:
            raise ValueError(f"pam contains invalid IUPAC DNA bases: {invalid}")
        return normalized


class CRISPRGuideCandidate(BaseModel):
    """One PAM-compatible protospacer with reference coordinates.

    Coordinates are 1-indexed and inclusive. ``pam_strand`` identifies the
    reference strand containing the oriented protospacer/PAM pattern;
    ``target_strand`` is the complementary strand bound by the guide.

    Attributes:
        protospacer (str): PAM-free sequence in nuclease orientation.
        pam (str): Matched PAM sequence in nuclease orientation.
        pam_side (Literal['three_prime', 'five_prime']): PAM orientation.
        pam_strand (Literal['+', '-']): Reference strand containing the PAM.
        target_strand (Literal['+', '-']): Complementary strand bound by the guide.
        protospacer_start (int): Genomic protospacer start coordinate.
        protospacer_end (int): Genomic protospacer end coordinate.
        pam_start (int): Genomic PAM start coordinate.
        pam_end (int): Genomic PAM end coordinate.
        context_sequence (str): Oriented exterior-context, protospacer, and PAM sequence.
        context_start (int): Genomic start coordinate of returned context.
        context_end (int): Genomic end coordinate of returned context.
        context_complete (bool): Whether both requested exterior flanks were available.
        context_is_canonical (bool): Whether returned context contains only A/C/G/T.
        context_5p_length (int): Available exterior 5-prime context length.
        context_3p_length (int): Available exterior 3-prime context length.
    """

    protospacer: str = Field(title="Protospacer", description="PAM-free sequence in nuclease orientation")
    pam: str = Field(title="PAM", description="Matched PAM sequence in nuclease orientation")
    pam_side: Literal["three_prime", "five_prime"] = Field(title="PAM Side", description="PAM orientation")
    pam_strand: Literal["+", "-"] = Field(title="PAM Strand", description="Reference strand containing the PAM")
    target_strand: Literal["+", "-"] = Field(
        title="Target Strand", description="Complementary reference strand bound by the guide"
    )
    protospacer_start: int = Field(title="Protospacer Start", description="1-indexed inclusive genomic start")
    protospacer_end: int = Field(title="Protospacer End", description="1-indexed inclusive genomic end")
    pam_start: int = Field(title="PAM Start", description="1-indexed inclusive genomic start")
    pam_end: int = Field(title="PAM End", description="1-indexed inclusive genomic end")
    context_sequence: str = Field(title="Context Sequence", description="Oriented target with exterior context")
    context_start: int = Field(title="Context Start", description="1-indexed inclusive genomic context start")
    context_end: int = Field(title="Context End", description="1-indexed inclusive genomic context end")
    context_complete: bool = Field(title="Context Complete", description="Whether requested flanks were available")
    context_is_canonical: bool = Field(
        title="Context Is Canonical", description="Whether returned context contains only A/C/G/T"
    )
    context_5p_length: int = Field(title="5-prime Context Length", description="Available exterior 5-prime bases")
    context_3p_length: int = Field(title="3-prime Context Length", description="Available exterior 3-prime bases")


class CRISPRGuideRegionResult(BaseModel):
    """Guide candidates and provenance for one input interval.

    Attributes:
        sequence_id (str): Reference record or contig identifier.
        region_start (int): 1-indexed inclusive interval start.
        region_end (int): 1-indexed inclusive interval end.
        sequence_sha256 (str): SHA256 of the normalized interval sequence.
        reference_id (str | None): Optional versioned assembly identifier.
        reference_sha256 (str | None): Optional source-reference checksum.
        candidates (list[CRISPRGuideCandidate]): Deterministically ordered guide candidates.
        ambiguous_windows_skipped (int): Windows excluded for non-ACGT protospacer or PAM bases.
    """

    sequence_id: str = Field(title="Sequence Identifier", description="Reference record or contig identifier")
    region_start: int = Field(title="Region Start", description="1-indexed inclusive interval start")
    region_end: int = Field(title="Region End", description="1-indexed inclusive interval end")
    sequence_sha256: str = Field(title="Sequence SHA256", description="SHA256 of the normalized interval sequence")
    reference_id: str | None = Field(default=None, title="Reference Identifier", description="Assembly identifier")
    reference_sha256: str | None = Field(
        default=None, title="Reference SHA256", description="Source-reference checksum"
    )
    candidates: list[CRISPRGuideCandidate] = Field(
        default_factory=list, title="Candidates", description="PAM-compatible protospacers"
    )
    ambiguous_windows_skipped: int = Field(
        default=0, title="Ambiguous Windows Skipped", description="Non-canonical windows excluded"
    )


class CRISPRGuideEnumerationOutput(BaseToolOutput):
    """Per-region deterministic guide enumeration results.

    Attributes:
        results (list[CRISPRGuideRegionResult]): One result per input region, in input order.
        pam (str): Normalized IUPAC PAM grammar.
        pam_side (Literal['three_prime', 'five_prime']): PAM orientation.
        protospacer_length (int): PAM-free guide length.
        strands (Literal['both', 'plus', 'minus']): Scanned PAM-bearing strands.
        context_5p (int): Requested exterior 5-prime context.
        context_3p (int): Requested exterior 3-prime context.
        nuclease (str): Descriptive nuclease identity.
    """

    results: list[CRISPRGuideRegionResult] = Field(title="Results", description="One result per input region")
    pam: str = Field(title="PAM", description="Normalized IUPAC PAM grammar")
    pam_side: Literal["three_prime", "five_prime"] = Field(title="PAM Side", description="PAM orientation")
    protospacer_length: int = Field(title="Protospacer Length", description="PAM-free guide length")
    strands: Literal["both", "plus", "minus"] = Field(title="Strands", description="Scanned PAM strands")
    context_5p: int = Field(title="5-prime Context", description="Requested exterior 5-prime context")
    context_3p: int = Field(title="3-prime Context", description="Requested exterior 3-prime context")
    nuclease: str = Field(title="Nuclease", description="Descriptive nuclease identity")

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[CRISPRGuideRegionResult]:  # type: ignore[override]
        return iter(self.results)

    @property
    def output_format_options(self) -> list[str]:
        """Return supported export formats."""
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        """Return the default export format."""
        return "json"

    def _export_output(self, export_path: Path | str, file_format: str) -> None:
        """Export complete JSON or one flat candidate row per CSV record."""
        path = Path(export_path).with_suffix(f".{file_format}")
        if file_format == "json":
            path.write_text(json.dumps(self.model_dump(mode="json"), indent=2))
            return
        if file_format == "csv":
            fieldnames = ["sequence_id", *CRISPRGuideCandidate.model_fields]
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for result in self.results:
                    writer.writerows(
                        {"sequence_id": result.sequence_id, **candidate.model_dump()} for candidate in result.candidates
                    )
            return
        raise ValueError(f"Unsupported format: {file_format}")


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def _matches_pam(sequence: str, pam: str) -> bool:
    return len(sequence) == len(pam) and all(
        base in IUPAC_BASES[code] for base, code in zip(sequence, pam, strict=True)
    )


def _reference_bounds(region: CRISPRGuideRegion, start: int, end: int, strand: str) -> tuple[int, int]:
    """Convert a zero-based half-open oriented interval to reference coordinates."""
    if strand == "+":
        return region.start + start, region.start + end - 1
    return region.start + len(region.sequence) - end, region.start + len(region.sequence) - start - 1


def _enumerate_strand(
    region: CRISPRGuideRegion, config: CRISPRGuideEnumerationConfig, strand: Literal["+", "-"]
) -> tuple[list[CRISPRGuideCandidate], int]:
    sequence = region.sequence if strand == "+" else _reverse_complement(region.sequence)
    guide_length = config.protospacer_length
    pam_length = len(config.pam)
    target_length = guide_length + pam_length
    candidates = []
    ambiguous_skipped = 0
    for target_start in range(max(0, len(sequence) - target_length + 1)):
        if config.pam_side == "three_prime":
            protospacer_slice = (target_start, target_start + guide_length)
            pam_slice = (protospacer_slice[1], protospacer_slice[1] + pam_length)
        else:
            pam_slice = (target_start, target_start + pam_length)
            protospacer_slice = (pam_slice[1], pam_slice[1] + guide_length)
        protospacer = sequence[slice(*protospacer_slice)]
        pam_sequence = sequence[slice(*pam_slice)]
        if set(pam_sequence) - set("ACGT"):
            ambiguous_skipped += 1
            continue
        if not _matches_pam(pam_sequence, config.pam):
            continue
        if set(protospacer) - set("ACGT"):
            ambiguous_skipped += 1
            continue

        requested_start = target_start - config.context_5p
        requested_end = target_start + target_length + config.context_3p
        context_start = max(0, requested_start)
        context_end = min(len(sequence), requested_end)
        genomic_context = _reference_bounds(region, context_start, context_end, strand)
        genomic_protospacer = _reference_bounds(region, *protospacer_slice, strand)
        genomic_pam = _reference_bounds(region, *pam_slice, strand)
        candidates.append(
            CRISPRGuideCandidate(
                protospacer=protospacer,
                pam=pam_sequence,
                pam_side=config.pam_side,
                pam_strand=strand,
                target_strand="-" if strand == "+" else "+",
                protospacer_start=genomic_protospacer[0],
                protospacer_end=genomic_protospacer[1],
                pam_start=genomic_pam[0],
                pam_end=genomic_pam[1],
                context_sequence=sequence[context_start:context_end],
                context_start=genomic_context[0],
                context_end=genomic_context[1],
                context_complete=requested_start >= 0 and requested_end <= len(sequence),
                context_is_canonical=not bool(set(sequence[context_start:context_end]) - set("ACGT")),
                context_5p_length=target_start - context_start,
                context_3p_length=context_end - (target_start + target_length),
            )
        )
    return candidates, ambiguous_skipped


def example_input() -> CRISPRGuideEnumerationInput:
    """Return a minimal sequence interval containing one SpCas9 target."""
    return CRISPRGuideEnumerationInput(
        regions=[CRISPRGuideRegion(sequence_id="synthetic", sequence="AAAAGACTGCATTGCATGACTGCATGGAAA")]
    )


@tool(
    key="crispr-guides-enumerate",
    local_only="crispr-guides-enumerate is a deterministic in-process sequence operation",
    label="CRISPR Guide Enumeration",
    category="gene_annotation",
    input_class=CRISPRGuideEnumerationInput,
    config_class=CRISPRGuideEnumerationConfig,
    output_class=CRISPRGuideEnumerationOutput,
    description="Enumerate PAM-compatible guide protospacers on both strands of reference-oriented DNA intervals",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_fields=["regions"],
    iterable_output_field="results",
)
def run_crispr_guides_enumerate(
    inputs: CRISPRGuideEnumerationInput,
    config: CRISPRGuideEnumerationConfig,
    instance: Any = None,  # noqa: ARG001 -- required by the tool interface
) -> CRISPRGuideEnumerationOutput:
    """Enumerate guides without scoring or selecting candidates.

    Args:
        inputs (CRISPRGuideEnumerationInput): Reference-oriented DNA intervals.
        config (CRISPRGuideEnumerationConfig): PAM, guide length, strand, and context settings.
        instance (Any): Reserved tool execution instance.

    Returns:
        CRISPRGuideEnumerationOutput: One provenance-bearing candidate bundle per interval.
    """
    results = []
    requested_strands: tuple[Literal["+", "-"], ...]
    if config.strands == "both":
        requested_strands = ("+", "-")
    elif config.strands == "plus":
        requested_strands = ("+",)
    else:
        requested_strands = ("-",)

    for region in inputs.regions:
        candidates = []
        ambiguous_skipped = 0
        for strand in requested_strands:
            strand_candidates, strand_ambiguous = _enumerate_strand(region, config, strand)
            candidates.extend(strand_candidates)
            ambiguous_skipped += strand_ambiguous
        candidates.sort(key=lambda item: (item.protospacer_start, item.protospacer_end, item.pam_strand))
        results.append(
            CRISPRGuideRegionResult(
                sequence_id=region.sequence_id,
                region_start=region.start,
                region_end=region.start + len(region.sequence) - 1,
                sequence_sha256=hashlib.sha256(region.sequence.encode()).hexdigest(),
                reference_id=region.reference_id,
                reference_sha256=region.reference_sha256,
                candidates=candidates,
                ambiguous_windows_skipped=ambiguous_skipped,
            )
        )

    logger.debug("Enumerated CRISPR guides for %d reference intervals", len(inputs.regions))
    return CRISPRGuideEnumerationOutput(
        results=results,
        pam=config.pam,
        pam_side=config.pam_side,
        protospacer_length=config.protospacer_length,
        strands=config.strands,
        context_5p=config.context_5p,
        context_3p=config.context_3p,
        nuclease=config.nuclease,
        metadata={"coordinate_system": "1-indexed-inclusive", "selection_applied": False, "scoring_applied": False},
    )
