"""MEME Suite FIMO motif scanning: find occurrences of known motifs in sequences."""

import warnings
from pathlib import Path
from typing import Any

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


class FimoMatch(BaseModel):
    """A single motif occurrence found by FIMO.

    Mirrors one row of FIMO's ``fimo.tsv`` output.

    Attributes:
        motif_id (str): Identifier of the matched motif (first token of the MEME
            ``MOTIF`` line; the motif accession when present).
        motif_alt_id (str): Alternate motif name (second token of the ``MOTIF``
            line); ``"-"`` when the motif has no alternate name.
        sequence_name (str): Name of the target sequence containing the match.
        start (int): Match start in the target sequence (1-indexed, inclusive).
        stop (int): Match end in the target sequence (1-indexed, inclusive).
        strand (str): Strand of the match, ``"+"`` or ``"-"``.
        score (float): Log-odds score of the match.
        pvalue (float): P-value of the match.
        qvalue (float): Benjamini-Hochberg q-value (false discovery rate).
        matched_sequence (str): Subsequence at the hit, on the matched strand.
    """

    motif_id: str = Field(title="Motif ID", description="Matched motif identifier (motif accession when present)")
    motif_alt_id: str = Field(title="Motif Alt ID", description="Alternate motif name; '-' if none")
    sequence_name: str = Field(title="Sequence Name", description="Name of the target sequence containing the match")
    start: int = Field(title="Start", description="Match start in the target sequence (1-indexed, inclusive)")
    stop: int = Field(title="Stop", description="Match end in the target sequence (1-indexed, inclusive)")
    strand: str = Field(title="Strand", description="Strand of the match ('+' or '-')")
    score: float = Field(title="Score", description="Log-odds score of the match")
    pvalue: float = Field(title="P-value", description="P-value of the match")
    qvalue: float = Field(title="Q-value", description="Benjamini-Hochberg q-value (FDR) of the match")
    matched_sequence: str = Field(title="Matched Sequence", description="Subsequence at the hit on the matched strand")


# Input:
class MEMEFimoScanInput(BaseToolInput):
    """Input for FIMO motif scanning.

    Attributes:
        sequences (list[str]): Target sequences to scan for motif occurrences.
            A single sequence string is normalized to a one-element list.
        motifs (str | Path): Path to a MEME-format motif file (``.meme``) of
            position weight matrices, e.g. exported from JASPAR. ``AssetRef`` supported.
    """

    sequences: list[str] = InputField(
        title="Sequences",
        description="Target sequences to scan, as a single string or a list of strings",
    )
    motifs: str | Path = InputField(
        title="Motif File",
        description="Path to a MEME-format (.meme) motif PWM file, e.g. from JASPAR. AssetRef supported",
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def _normalize_sequences(cls, value: Any) -> Any:
        """Normalize a single sequence string to a one-element list."""
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("sequences")
    @classmethod
    def _validate_sequences(cls, sequences: list[str]) -> list[str]:
        """Reject empty input or blank sequences."""
        if not sequences:
            raise ValueError("At least one sequence is required")
        for i, seq in enumerate(sequences):
            if not seq or not seq.strip():
                raise ValueError(f"Sequence {i + 1} is empty")
        return sequences


# Config:
class MEMEFimoScanConfig(BaseConfig):
    """Configuration for FIMO motif scanning.

    Exposes the two FIMO knobs end users commonly set; defaults reproduce FIMO's
    command-line behavior for nucleotide motifs.

    Attributes:
        threshold (float): Report only matches with a p-value at or below this
            cutoff (FIMO ``--thresh``). Default 1e-4.
        both_strands (bool): Scan both the given and reverse-complement strands.
            Set False for single-strand scans (FIMO ``--norc`` disables the reverse
            strand). Automatically ignored for protein / non-complementable motifs,
            which are always scanned forward-only. Default True.
    """

    threshold: float = ConfigField(
        title="P-value Threshold",
        default=1e-4,
        gt=0,
        le=1,
        description="Report matches with p-value <= this cutoff; lower is stricter (FIMO --thresh)",
    )
    both_strands: bool = ConfigField(
        title="Scan Both Strands",
        default=True,
        description="Scan both strands for nucleotide motifs; set False for single-strand. Ignored for protein",
    )


# Output:
class MEMEFimoScanOutput(BaseToolOutput):
    """FIMO scan results: motif occurrences across the target sequences.

    Attributes:
        matches (list[FimoMatch]): Every motif occurrence passing the p-value
            threshold, across all target sequences and motifs. Empty if none.

    Properties:
        num_matches: Total number of motif occurrences found.
    """

    matches: list[FimoMatch] = Field(
        default_factory=list,
        title="Matches",
        description="Motif occurrences passing the p-value threshold",
    )

    @property
    def num_matches(self) -> int:
        """Return the number of motif occurrences found."""
        return len(self.matches)

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

        if file_format not in ("csv", "json"):
            raise ValueError(f"Unsupported format: {file_format}")
        if not self.matches:
            warnings.warn("No FIMO matches to export. The scan returned no occurrences.", UserWarning, stacklevel=2)
            return
        df = pd.DataFrame([m.model_dump() for m in self.matches])
        path = Path(export_path)
        if file_format == "csv":
            df.to_csv(path, index=False)
        else:
            df.to_json(path, orient="records", indent=2)


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return MEMEFimoScanInput(
        sequences=["GTTGAGCTGGTCAACAAGTTGAGCTGGTCAAC"],
        motifs=str(Path(__file__).parent / "examples" / "example.meme"),
    )


@tool(
    key="meme-fimo-scan",
    label="MEME FIMO Motif Scan",
    category="gene_annotation",
    input_class=MEMEFimoScanInput,
    config_class=MEMEFimoScanConfig,
    output_class=MEMEFimoScanOutput,
    description="Scan sequences for occurrences of known motifs (PWMs) using MEME Suite FIMO",
    example_input=example_input,
    cacheable=True,
)
def run_meme_fimo_scan(
    inputs: MEMEFimoScanInput, config: MEMEFimoScanConfig, instance: Any = None
) -> MEMEFimoScanOutput:
    """Scan sequences for occurrences of known motifs using MEME Suite FIMO.

    FIMO (Find Individual Motif Occurrences) scans each target sequence against
    every position-weight-matrix motif in the supplied MEME file and reports each
    match with a log-odds score, p-value, and q-value. Backed by ``pymemesuite``,
    so no host MEME installation is required.

    Args:
        inputs (MEMEFimoScanInput): Target sequences and the MEME-format motif file.
        config (MEMEFimoScanConfig): Scan parameters (p-value threshold, strand mode).
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        MEMEFimoScanOutput: ``matches`` — every occurrence passing ``threshold``,
            each carrying the motif id, sequence name, 1-indexed coordinates,
            strand, score, p-value, and q-value.

    Examples:
        >>> inputs = MEMEFimoScanInput(sequences=["GTTGAGCTGGTCAAC"], motifs="/path/to/jaspar.meme")
        >>> result = run_meme_fimo_scan(inputs, MEMEFimoScanConfig(threshold=1e-4))
        >>> print(f"Found {result.num_matches} motif occurrence(s)")
        >>> strongest = min(result.matches, key=lambda m: m.pvalue) if result.matches else None
    """
    output_data = ToolInstance.dispatch(
        "meme",
        {
            "device": "cpu",
            "operation": "fimo_scan",
            "sequences": inputs.sequences,
            "motifs_path": str(inputs.motifs),
            "threshold": config.threshold,
            "both_strands": config.both_strands,
        },
        instance=instance,
        config=config,
    )

    matches = [FimoMatch(**m) for m in output_data["matches"]]

    return MEMEFimoScanOutput(
        metadata={
            "num_sequences": output_data.get("num_sequences", len(inputs.sequences)),
            "num_motifs": output_data.get("num_motifs", 0),
            "threshold": config.threshold,
            "both_strands": config.both_strands,
        },
        matches=matches,
    )
