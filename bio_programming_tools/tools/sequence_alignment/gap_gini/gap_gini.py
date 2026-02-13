"""
Alignment gap distribution Gini score computation.

This module computes a Gini coefficient on the gap run-length distribution
of pairwise alignments, used to detect truncation artifacts where gaps
are concentrated in one region rather than distributed evenly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from pydantic import Field, field_validator

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, BaseToolInput, BaseToolOutput


# ============================================================================
# Internal Utilities
# ============================================================================
def _gini(x: np.ndarray) -> float:
    """Compute the Gini coefficient of an array of values.

    Args:
        x: Array of non-negative values.

    Returns:
        Gini coefficient in [0, 1]. 0 means perfect equality, 1 means
        maximum inequality.
    """
    if len(x) == 0 or np.mean(x) == 0:
        return 0.0
    diffsum = 0.0
    for i in range(len(x) - 1):
        diffsum += np.sum(np.abs(x[i] - x[i + 1 :]))
    return float(diffsum / (len(x) ** 2 * np.mean(x)))


def _gap_runs(seq: str) -> List[int]:
    """Compute run lengths of consecutive gap characters in a sequence.

    Non-gap characters contribute a run length of 1. Consecutive gap
    characters ('-') are counted together as a single run.

    Args:
        seq: Aligned sequence string (may contain '-' gap characters).

    Returns:
        List of run lengths.
    """
    if not seq:
        return []

    runs = []
    prev = None
    run_len = 1

    for ind, char in enumerate(seq):
        if prev is None:
            prev = char
            continue

        if char == "-" and prev == "-":
            run_len += 1
        elif char != "-" and prev == "-":
            runs.append(run_len)
            run_len = 1
        else:
            runs.append(1)

        if ind == len(seq) - 1:
            runs.append(run_len)

        prev = char

    return runs


def _gap_gini_single(alignment: str) -> float:
    """Compute gap Gini score for a single pairwise alignment.

    Parses a FASTA-formatted pairwise alignment (2 sequences), computes
    gap run lengths for each sequence, and returns the maximum Gini
    coefficient across both sequences.

    Args:
        alignment: FASTA-formatted string containing exactly 2 aligned sequences.

    Returns:
        Maximum Gini coefficient of gap run distributions across both sequences.

    Raises:
        ValueError: If the alignment does not contain exactly 2 sequences.
    """
    sequences = re.findall(
        r"^[^>].*?(?=(?:^>|\Z))", alignment, re.MULTILINE | re.DOTALL
    )
    if len(sequences) != 2:
        raise ValueError(
            f"Expected 2 sequences in pairwise alignment, got {len(sequences)}"
        )
    al1, al2 = [seq.replace("\n", "") for seq in sequences]

    al1_runs = np.array(_gap_runs(al1))
    al2_runs = np.array(_gap_runs(al2))

    gini1 = _gini(al1_runs) if len(al1_runs) > 0 else 0.0
    gini2 = _gini(al2_runs) if len(al2_runs) > 0 else 0.0

    return max(gini1, gini2)


# ============================================================================
# Data Models
# ============================================================================
# Input:
class GapGiniInput(BaseToolInput):
    """Input for gap Gini score computation.

    Attributes:
        alignments (List[str]): List of FASTA-formatted pairwise alignment strings.
            Each string must contain exactly 2 aligned sequences (with gap characters).
            These typically come from MAFFT pairwise alignment output.
    """

    alignments: List[str] = Field(
        description="List of FASTA-formatted pairwise alignment strings (2 sequences each)"
    )

    @field_validator("alignments", mode="before")
    @classmethod
    def normalize_alignments(cls, value) -> List[str]:
        """Normalize a single alignment string to a list."""
        if isinstance(value, str):
            return [value]
        return value


# Output:
class GapGiniOutput(BaseToolOutput):
    """Output from gap Gini score computation.

    Attributes:
        gini_scores (List[float]): Per-alignment Gini scores. Each value is
            the maximum Gini coefficient across the two aligned sequences.
            Lower values (< 0.1) indicate evenly distributed gaps (good).
            Higher values indicate concentrated gaps (potential truncation).
    """

    gini_scores: List[float] = Field(
        default_factory=list,
        description="Per-alignment Gini scores (max across both sequences)",
    )

    @property
    def output_format_options(self) -> List[str]:
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str):
        path = Path(export_path).with_suffix(f".{file_format}")
        df = pd.DataFrame(
            {"alignment_index": range(len(self.gini_scores)), "gini_score": self.gini_scores}
        )
        if file_format == "csv":
            df.to_csv(path, index=False)
        elif file_format == "json":
            df.to_json(path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# Config:
class GapGiniConfig(BaseConfig):
    """Configuration for gap Gini score computation.

    This tool requires no configuration parameters — the Gini coefficient
    is computed directly from the alignment gap structure.
    """


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="gap-gini",
    label="Alignment Gap Gini Score",
    input=GapGiniInput,
    config=GapGiniConfig,
    output=GapGiniOutput,
    description="Compute gap distribution Gini score for pairwise alignments",
)
def run_gap_gini(inputs: GapGiniInput, config: GapGiniConfig) -> GapGiniOutput:
    """Compute gap distribution Gini scores for pairwise alignments.

    For each pairwise alignment, computes the Gini coefficient of the gap
    run-length distribution. This metric detects truncation artifacts where
    gaps are concentrated in one region rather than distributed evenly.

    A Gini score close to 0 indicates evenly distributed gaps (typical of
    genuine sequence divergence), while a score close to 1 indicates highly
    concentrated gaps (typical of truncation artifacts).

    Args:
        inputs (GapGiniInput): Validated input containing FASTA-formatted
            pairwise alignments.
        config (GapGiniConfig): Configuration (no parameters needed).

    Returns:
        GapGiniOutput: Per-alignment Gini scores.

    Examples:
        >>> inputs = GapGiniInput(
        ...     alignments=[">seq1\\nACGT--ACGT\\n>seq2\\nACGTACACGT\\n"]
        ... )
        >>> config = GapGiniConfig()
        >>> result = run_gap_gini(inputs, config)
        >>> print(result.gini_scores)
    """
    gini_scores = []
    for alignment in inputs.alignments:
        score = _gap_gini_single(alignment)
        gini_scores.append(score)

    return GapGiniOutput(
        metadata={"num_alignments": len(inputs.alignments)},
        gini_scores=gini_scores,
    )
