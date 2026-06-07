"""miRanda microRNA target-site prediction in RNA/DNA sequences."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
    resolve_sequence_ids,
)

_EXAMPLES_DIR = Path(__file__).parent / "examples"


# ============================================================================
# Data Models
# ============================================================================
class MirandaTargetSite(BaseModel):
    """A single predicted microRNA target site on a target sequence."""

    mirna_id: str = Field(title="microRNA ID", description="Identifier of the microRNA query that hit this site")
    score: float = Field(title="Score", description="Smith-Waterman complementarity alignment score")
    energy: float = Field(title="Energy", description="ViennaRNA duplex free energy (kcal/mol); 0.0 if energy disabled")
    target_start: int = Field(title="Target Start", description="1-indexed inclusive start on the target sequence")
    target_end: int = Field(title="Target End", description="1-indexed inclusive end on the target sequence")
    mirna_start: int = Field(title="microRNA Start", description="1-indexed inclusive start on the microRNA query")
    mirna_end: int = Field(title="microRNA End", description="1-indexed inclusive end on the microRNA query")
    alignment_length: int = Field(title="Alignment Length", description="Length of the aligned region")
    identity: float = Field(title="Identity (%)", description="Percent Watson-Crick identity over the alignment")
    similarity: float = Field(
        title="Similarity (%)", description="Percent similarity (matches plus G:U wobble) over the alignment"
    )
    mirna_alignment: str = Field(default="", title="microRNA Alignment", description="microRNA strand (3'->5')")
    pairing: str = Field(default="", title="Pairing", description="Pairing annotation (`|` match, `:` G:U wobble)")
    target_alignment: str = Field(default="", title="Target Alignment", description="Target strand (5'->3')")


class MirandaSequenceResult(BaseModel):
    """All predicted target sites found in a single target sequence."""

    target_id: str = Field(title="Target ID", description="Identifier of the target sequence")
    target_sequence: str = Field(title="Target Sequence", description="The input target sequence")
    target_sites: list[MirandaTargetSite] = Field(
        default_factory=list,
        title="Target Sites",
        description="Predicted microRNA target sites in this sequence, sorted by score descending",
    )

    @property
    def num_sites(self) -> int:
        """Number of predicted target sites in this sequence."""
        return len(self.target_sites)

    @property
    def has_sites(self) -> bool:
        """Whether any target sites were predicted."""
        return len(self.target_sites) > 0


# Input:
class MirandaInput(BaseToolInput):
    """Target sequences to scan and the microRNA queries to scan them with.

    Attributes:
        target_sequences (list[str]): RNA/DNA target sequences (mRNA, 3'UTR, genomic) to scan.
        mirna_queries (list[str]): microRNA query sequences, applied to every target.
        mirna_ids (list[str] | None): Optional microRNA labels (default seq_0, seq_1, ...).
    """

    target_sequences: list[str] = InputField(
        title="Target Sequences",
        description="RNA/DNA target sequences (mRNA, 3'UTR, genomic) to scan for target sites",
    )
    mirna_queries: list[str] = InputField(
        title="microRNA Queries",
        description="microRNA query sequences to scan with; applied to every target sequence",
    )
    mirna_ids: list[str] | None = InputField(
        default=None,
        title="microRNA IDs",
        description="Optional microRNA identifiers surfaced on each site (defaults to seq_0, seq_1, ...)",
    )

    @field_validator("target_sequences", "mirna_queries", mode="before")
    @classmethod
    def _normalize_sequences(cls, value: Any) -> Any:
        """Normalize a single sequence string to a one-element list."""
        if isinstance(value, str):
            return [value]
        return value

    @model_validator(mode="after")
    def _validate_lengths(self) -> "MirandaInput":
        """Reject empty sequence lists and mismatched id lists."""
        if not self.target_sequences:
            raise ValueError("miranda-scan: `target_sequences` cannot be empty")
        if not self.mirna_queries:
            raise ValueError("miranda-scan: `mirna_queries` cannot be empty")
        if self.mirna_ids is not None and len(self.mirna_ids) != len(self.mirna_queries):
            raise ValueError(
                f"miranda-scan: `mirna_ids` ({len(self.mirna_ids)}) must match "
                f"`mirna_queries` ({len(self.mirna_queries)})"
            )
        return self


# Output:
class MirandaOutput(BaseToolOutput):
    """Per-target prediction results, one per input target sequence in input order.

    Attributes:
        results (list[MirandaSequenceResult]): Per-target results, in input order.
    """

    results: list[MirandaSequenceResult] = Field(
        default_factory=list,
        title="Results",
        description="Per-target prediction results, one per input target sequence",
    )

    def __len__(self) -> int:
        """Number of target results."""
        return len(self.results)

    def __getitem__(self, idx: int) -> MirandaSequenceResult:
        """Get a target result by index."""
        return self.results[idx]

    def __iter__(self) -> Iterator[MirandaSequenceResult]:  # type: ignore[override]
        """Iterate over the per-target results."""
        return iter(self.results)

    @property
    def total_sites(self) -> int:
        """Total number of target sites across all target sequences."""
        return sum(r.num_sites for r in self.results)

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

        path = Path(export_path).with_suffix(f".{file_format}")
        columns = [
            "target_id",
            "mirna_id",
            "target_start",
            "target_end",
            "mirna_start",
            "mirna_end",
            "score",
            "energy",
            "alignment_length",
            "identity",
            "similarity",
        ]
        rows = [
            {
                "target_id": result.target_id,
                "mirna_id": site.mirna_id,
                "target_start": site.target_start,
                "target_end": site.target_end,
                "mirna_start": site.mirna_start,
                "mirna_end": site.mirna_end,
                "score": site.score,
                "energy": site.energy,
                "alignment_length": site.alignment_length,
                "identity": site.identity,
                "similarity": site.similarity,
            }
            for result in self.results
            for site in result.target_sites
        ]
        df = pd.DataFrame(rows, columns=columns)
        if file_format == "csv":
            df.to_csv(path, index=False)
        elif file_format == "json":
            df.to_json(path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# Config:
class MirandaConfig(BaseConfig):
    """Scoring thresholds and alignment parameters for miRanda.

    This build parses the score/energy/gap thresholds as integers (only ``scale`` is
    fractional). Lower ``score_threshold`` / raise ``energy_threshold`` toward 0 for
    higher sensitivity.

    Attributes:
        score_threshold (int): Minimum alignment score to report a site (``-sc``).
        energy_threshold (int): Maximum duplex free energy in kcal/mol; negative (``-en``).
        scale (float): Contrast scaling on the microRNA 5' seed region (``-scale``).
        gap_open (int): Penalty for opening an alignment gap; negative (``-go``).
        gap_extend (int): Per-position penalty for extending a gap; negative (``-ge``).
        strict (bool): Apply strict miRNA:target duplex base-pairing heuristics; ``-strict`` when True.
        compute_energy (bool): Compute ViennaRNA free energy; ``-noenergy`` when False.
        trim (int): Trim targets to this many nucleotides; 0 disables (``-trim``).
    """

    score_threshold: int = ConfigField(
        title="Score Threshold",
        default=50,
        ge=0,
        description="Minimum alignment score to report a site; lower for higher sensitivity",
    )
    energy_threshold: int = ConfigField(
        title="Energy Threshold",
        default=-20,
        le=0,
        description="Max duplex free energy (kcal/mol) to report; negative, raise toward 0 for more sensitivity",
    )
    scale: float = ConfigField(
        title="5' Scaling",
        default=4.0,
        description="Contrast scaling on 5' seed match/mismatch scores; higher emphasizes seed complementarity",
    )
    gap_open: int = ConfigField(
        title="Gap Open Penalty",
        default=-8,
        le=0,
        description="Penalty for opening a gap in the alignment; more negative discourages gaps",
    )
    gap_extend: int = ConfigField(
        title="Gap Extend Penalty",
        default=-2,
        le=0,
        description="Per-position penalty for extending an open gap; more negative discourages long gaps",
    )
    strict: bool = ConfigField(
        title="Strict Heuristics",
        default=False,
        description="Strict miRNA:target 5'-seed heuristics (`-strict`); enable for fewer, higher-confidence sites",
    )
    compute_energy: bool = ConfigField(
        title="Compute Energy",
        default=True,
        description="Compute ViennaRNA duplex free energy; disable (`-noenergy`) for a faster score-only scan",
    )
    trim: int = ConfigField(
        title="Trim Length",
        default=0,
        ge=0,
        description="Trim target sequences to this many nucleotides (0 = no trimming)",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def _read_fasta(path: Path) -> tuple[list[str], list[str]]:
    """Read a FASTA file into parallel (ids, sequences) lists."""
    ids: list[str] = []
    seqs: list[str] = []
    current: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if ids:  # flush the previous record (empty string if it had no sequence lines)
                seqs.append("".join(current))
                current = []
            header = line[1:].split()
            ids.append(header[0] if header else f"seq_{len(ids)}")
        elif line.strip():
            current.append(line.strip())
    if ids:
        seqs.append("".join(current))
    return ids, seqs


def example_input() -> Any:
    """Minimal valid input: miR-bantam scanned against the Drosophila hid 3'UTR."""
    _, mirna_seqs = _read_fasta(_EXAMPLES_DIR / "bantam_mirna.fasta")
    _, target_seqs = _read_fasta(_EXAMPLES_DIR / "hid_utr.fasta")
    return MirandaInput(
        target_sequences=target_seqs,
        mirna_queries=mirna_seqs,
        mirna_ids=["miR-bantam"],
    )


@tool(
    key="miranda-scan",
    label="miRanda Target Scan",
    category="gene_annotation",
    input_class=MirandaInput,
    config_class=MirandaConfig,
    output_class=MirandaOutput,
    description=(
        "Predict microRNA target sites in RNA/DNA sequences using miRanda "
        "(complementarity alignment plus ViennaRNA thermodynamics)."
    ),
    example_input=example_input,
    iterable_input_field="target_sequences",
    iterable_output_field="results",
    cacheable=True,
)
def run_miranda_scan(inputs: MirandaInput, config: MirandaConfig, instance: Any = None) -> MirandaOutput:
    """Scan microRNA queries against target sequences and return per-target sites.

    Args:
        inputs (MirandaInput): Target sequences and microRNA queries to scan with.
        config (MirandaConfig): Scoring thresholds and alignment parameters.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        MirandaOutput: Per-target results, one per input target sequence in input order.

    Examples:
        >>> inputs = MirandaInput(target_sequences=["ACGT..."], mirna_queries=["UGAGAUC..."])
        >>> result = run_miranda_scan(inputs, MirandaConfig())
        >>> print(f"{result.total_sites} target sites found")
    """
    mirna_ids = resolve_sequence_ids(inputs.mirna_queries, inputs.mirna_ids)
    # Target ids are positional: the iterable framework may dedup/cache-strip
    # target_sequences, so a parallel user-supplied id list cannot stay aligned.
    target_ids = resolve_sequence_ids(inputs.target_sequences, None)

    output_data = ToolInstance.dispatch(
        "miranda",
        {
            "device": "cpu",
            "mirna_sequences": list(inputs.mirna_queries),
            "target_sequences": list(inputs.target_sequences),
            "score_threshold": config.score_threshold,
            "energy_threshold": config.energy_threshold,
            "scale": config.scale,
            "gap_open": config.gap_open,
            "gap_extend": config.gap_extend,
            "strict": config.strict,
            "compute_energy": config.compute_energy,
            "trim": config.trim,
        },
        instance=instance,
        config=config,
    )

    sites_by_target = _parse_miranda_output(output_data["stdout"], mirna_ids)
    results = [
        MirandaSequenceResult(
            target_id=target_ids[idx],
            target_sequence=inputs.target_sequences[idx],
            target_sites=sites_by_target.get(idx, []),
        )
        for idx in range(len(inputs.target_sequences))
    ]

    return MirandaOutput(
        metadata={
            "num_targets": len(inputs.target_sequences),
            "num_mirnas": len(inputs.mirna_queries),
            "score_threshold": config.score_threshold,
            "energy_threshold": config.energy_threshold,
            "scale": config.scale,
            "gap_open": config.gap_open,
            "gap_extend": config.gap_extend,
            "strict": config.strict,
            "compute_energy": config.compute_energy,
            "trim": config.trim,
        },
        results=results,
    )


# ============================================================================
# Output Parsing
# ============================================================================
def _parse_miranda_output(stdout: str, mirna_ids: list[str]) -> dict[int, list[MirandaTargetSite]]:
    """Parse miRanda stdout into target sites grouped by target index.

    Each hit is a self-identifying tab-delimited ``>`` line preceded by a
    ``Query:``/pairing/``Ref:`` block; internal ids ``q{i}``/``t{j}`` map back to
    input order. ``mirna_ids`` gives user-facing names by query order. Returns
    ``{target_index: [sites sorted by score desc]}``.
    """
    sites_by_target: dict[int, list[MirandaTargetSite]] = {}
    pending: dict[str, str] = {}
    lines = stdout.splitlines()

    for i, line in enumerate(lines):
        if "Query:" in line and "3'" in line and line.rstrip().endswith("5'"):
            content = line.split("3'", 1)[1].rsplit("5'", 1)[0]
            pending = {
                "mirna_alignment": content.strip(),
                # The pairing line sits under the sequence; [16:] skips miRanda's fixed
                # left margin ("   Query:    3' ") so it aligns with the alignment chars.
                "pairing": lines[i + 1][16:].rstrip() if i + 1 < len(lines) else "",
            }
            continue
        if "Ref:" in line and "5'" in line and line.rstrip().endswith("3'"):
            content = line.split("5'", 1)[1].rsplit("3'", 1)[0]
            pending["target_alignment"] = content.strip()
            continue

        if not line.startswith(">") or line.startswith(">>"):
            continue

        # `>` hit fields: query_id, ref_id, score, energy, qpos, rpos, len, identity%, similarity%.
        fields = line.split("\t")
        if len(fields) < 9:
            raise RuntimeError(f"miranda-scan: malformed hit line (expected >=9 tab fields): {line!r}")
        mirna_idx = _internal_index(fields[0][1:], "q")
        target_idx = _internal_index(fields[1], "t")
        if mirna_idx is None or target_idx is None:
            raise RuntimeError(f"miranda-scan: unrecognized internal id in hit line: {line!r}")
        mirna_start, mirna_end = (int(x) for x in fields[4].split())
        target_start, target_end = (int(x) for x in fields[5].split())
        site = MirandaTargetSite(
            mirna_id=mirna_ids[mirna_idx],
            score=float(fields[2]),
            energy=float(fields[3]),
            mirna_start=mirna_start,
            mirna_end=mirna_end,
            target_start=target_start,
            target_end=target_end,
            alignment_length=int(fields[6]),
            identity=float(fields[7].rstrip("%")),
            similarity=float(fields[8].rstrip("%")),
            mirna_alignment=pending.get("mirna_alignment", ""),
            pairing=pending.get("pairing", ""),
            target_alignment=pending.get("target_alignment", ""),
        )
        sites_by_target.setdefault(target_idx, []).append(site)
        pending = {}

    for sites in sites_by_target.values():
        sites.sort(key=lambda s: (-s.score, s.target_start))
    return sites_by_target


def _internal_index(token: str, prefix: str) -> int | None:
    """Parse the integer index from an internal FASTA id like ``q12`` -> 12."""
    if not token.startswith(prefix):
        return None
    try:
        return int(token[len(prefix) :])
    except ValueError:
        return None
