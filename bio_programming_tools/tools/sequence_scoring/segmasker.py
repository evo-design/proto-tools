"""
Segmasker tool for detecting low-complexity regions in protein sequences.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from io import StringIO
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from Bio import SeqIO
from pydantic import ConfigDict, Field, field_validator

from bio_programming_tools.tools.infra.tool_cache import tool_cache
from bio_programming_tools.tools.infra.tool_io import BaseToolInput, BaseToolOutput
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.tools.utils import BaseConfig, ConfigField


class SegmaskerInput(BaseToolInput):
    """Input object for Segmasker low-complexity region detection.

    This class defines the input parameters for detecting low-complexity regions
    in protein sequences using NCBI's segmasker tool.

    Attributes:
        sequences (List[str]): Protein sequence(s) to analyze for low-complexity
            regions. Can be provided as:

            - A single protein sequence string (automatically converted to list)
            - A list of protein sequence strings for batch processing

            Each sequence should contain standard amino acid characters. Empty
            sequences are handled gracefully (assigned 0.0 low-complexity fraction).
    """

    sequences: List[str] = Field(
        description="Protein sequence(s) to analyze for low-complexity regions"
    )

    @field_validator('sequences', mode='before')
    @classmethod
    def normalize_sequences(cls, value) -> List[str]:
        """Normalize single string to list."""
        if isinstance(value, str):
            return [value]
        return value

    @field_validator('sequences')
    @classmethod
    def validate_sequences(cls, sequences: List[str]) -> List[str]:
        """Validate sequences."""
        if not sequences:
            raise ValueError("At least one sequence is required")
        return sequences


class SegmaskerConfig(BaseConfig):
    """Configuration object for Segmasker low-complexity detection.

    This class defines configuration parameters for running NCBI's segmasker tool
    to identify low-complexity regions in protein sequences. Low-complexity regions
    are compositionally biased sequences that may interfere with homology searches.

    Attributes:
        segmasker_path (str): Path to the NCBI segmasker executable. If segmasker
            is in your PATH, use the default value. Otherwise, provide the full
            path to the executable. Default: ``"segmasker"``.

        window (int): Window size for analyzing sequence complexity. The algorithm
            examines the sequence in sliding windows of this size. Larger windows
            are less sensitive to short low-complexity regions. Typical range: 12-20.
            Must be at least 1. Default: 15.

        locut (float): Low-complexity cutoff threshold. Regions with complexity
            scores below this threshold are classified as low-complexity. Lower
            values identify only the most extreme low-complexity regions. Typical
            range: 1.4-2.2. Default: 1.8.

        hicut (float): High-complexity cutoff threshold. Regions with complexity
            scores above this threshold are classified as high-complexity. Used to
            define the transition between masked and unmasked regions. Should be
            greater than ``locut``. Typical range: 2.5-3.8. Default: 3.4.
    """

    segmasker_path: str = ConfigField(
        title="Segmasker Path",
        default="segmasker",
        description="Path to NCBI segmasker executable",
        hidden=True,
    )
    window: int = ConfigField(
        title="Window Size",
        default=15,
        ge=1,
        description="Window size for low-complexity detection",
        advanced=True,
    )
    locut: float = ConfigField(
        title="Low-complexity Threshold",
        default=1.8,
        description="Threshold below which a region is considered low-complexity",
        advanced=True,
    )
    hicut: float = ConfigField(
        title="High-complexity Threshold",
        default=3.4,
        description="Threshold above which a region is considered high-complexity",
        advanced=True,
    )


class SegmaskerOutput(BaseToolOutput):
    """Output from Segmasker low-complexity region detection.

    This class encapsulates the results of segmasker analysis, providing
    low-complexity statistics for each input sequence.

    Attributes:
        low_complexity_fractions (List[float]): Fraction of each sequence classified
            as low-complexity. Range: 0.0-1.0 where:

            - ``0.0``: No low-complexity regions detected
            - ``0.1-0.3``: Moderate low-complexity content
            - ``> 0.5``: High low-complexity content

            Length matches the number of input sequences.

        low_complexity_counts (List[int]): Number of positions classified as
            low-complexity in each sequence. Equals ``low_complexity_fraction * length``.

        sequence_lengths (List[int]): Length of each input sequence in amino acids.

        results_df (Optional[pd.DataFrame]): Detailed results as a pandas DataFrame
            with columns:

            - ``sequence_id``: Identifier (e.g., ``"seq_0"``, ``"seq_1"``)
            - ``length``: Sequence length in amino acids
            - ``lowercase_count``: Number of low-complexity positions
            - ``low_complexity_fraction``: Fraction of low-complexity positions (0.0-1.0)

            Returns ``None`` if processing fails.
    """

    low_complexity_fractions: List[float] = Field(
        description="Fraction of low-complexity regions for each sequence (0.0-1.0)"
    )
    low_complexity_counts: List[int] = Field(
        description="Number of low-complexity positions for each sequence"
    )
    sequence_lengths: List[int] = Field(description="Length of each input sequence")
    results_df: Optional[pd.DataFrame] = Field(
        default=None, description="DataFrame with detailed results"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def output_format_options(self) -> List[str]:
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        return "csv"

    def _export_output(self, export_path: Union[str, os.PathLike], file_format: str):
        path = Path(export_path).with_suffix(f".{file_format}")

        if self.results_df is None:
            # Create a minimal DF with summary stats if full results are missing
            df = pd.DataFrame({
                "sequence_length": self.sequence_lengths,
                "low_complexity_count": self.low_complexity_counts,
                "low_complexity_fraction": self.low_complexity_fractions
            })
        else:
            df = self.results_df

        if file_format == "csv":
            df.to_csv(path, index=False)

        elif file_format == "json":
            df.to_json(path, orient="records", indent=2)

        else:
            raise ValueError(f"Unsupported format: {file_format}")


@tool(
    key="segmasker",
    label="Segmasker Low-Complexity Detection",
    input=SegmaskerInput,
    config=SegmaskerConfig,
    output=SegmaskerOutput,
    description="Detect low-complexity regions in protein sequences using NCBI segmasker",
)
@tool_cache("segmasker")
def run_segmasker(
    inputs: SegmaskerInput,
    config: SegmaskerConfig
) -> SegmaskerOutput:
    """Detect low-complexity regions in protein sequences using NCBI segmasker.

    Uses NCBI's segmasker tool to identify compositionally biased and low-complexity
    regions in protein sequences. Low-complexity regions are often masked before
    homology searches to reduce false positive matches.

    Args:
        inputs (SegmaskerInput): Validated input containing one or more protein
            sequences to analyze.
        config (SegmaskerConfig): Validated segmasker configuration specifying
            window size and complexity thresholds.

    Returns:
        SegmaskerOutput: Structured output containing:
            - ``low_complexity_fractions``: Fraction of each sequence that is low-complexity
            - ``low_complexity_counts``: Number of low-complexity positions per sequence
            - ``sequence_lengths``: Length of each input sequence
            - ``results_df``: Detailed results DataFrame

    See Also:
        - BLAST+ suite: https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs
        - Segmasker documentation: https://www.ncbi.nlm.nih.gov/IEB/ToolBox/CPP_DOC/lxr/source/src/app/segmasker/

    Example:
        >>> inputs = SegmaskerInput(
        ...     sequences=["AAAAAAAA", "MVLSPADKTN"]
        ... )
        >>> config = SegmaskerConfig()
        >>> result = run_segmasker(inputs, config)
        >>> print(f"Low-complexity fractions: {result.low_complexity_fractions}")
    """

    # Check segmasker installation
    try:
        check_result = subprocess.run(
            [config.segmasker_path, "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if check_result.returncode != 0:
            raise FileNotFoundError
    except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        raise RuntimeError(
            "NCBI segmasker not found or not working properly. Install with: conda install -c bioconda blast"
        )

    # Get sequence lengths
    seq_lengths = [len(s) for s in inputs.sequences]

    # Handle all empty sequences
    if all(length == 0 for length in seq_lengths):
        return SegmaskerOutput(
            metadata={"num_sequences": len(inputs.sequences), "all_empty": True},
            low_complexity_fractions=[0.0] * len(inputs.sequences),
            low_complexity_counts=[0] * len(inputs.sequences),
            sequence_lengths=seq_lengths,
        )

    # Use faster temp directory if available
    tmp_dir = os.getenv("TMPDIR") or ("/dev/shm" if os.path.exists("/dev/shm") else None)

    # Create temporary FASTA with all sequences
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", dir=tmp_dir
    ) as tmp_file:
        for seq_idx, seq_str in enumerate(inputs.sequences):
            # Use placeholder for empty sequences
            seq_to_write = seq_str if len(seq_str) > 0 else "A"
            tmp_file.write(f">seq_{seq_idx}\n{seq_to_write}\n")
        # Ensure data is written before subprocess reads it
        tmp_file.flush()
        tmp_path = tmp_file.name

        try:
            # Run segmasker
            cmd = [
                config.segmasker_path,
                "-in",
                tmp_path,
                "-outfmt",
                "fasta",
                "-window",
                str(config.window),
                "-locut",
                str(config.locut),
                "-hicut",
                str(config.hicut),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                raise RuntimeError(f"Segmasker failed: {result.stderr}")

            # Parse output
            seq_records = list(SeqIO.parse(StringIO(result.stdout), "fasta"))

            if len(seq_records) != len(inputs.sequences):
                raise RuntimeError(
                    f"Segmasker returned {len(seq_records)} results but expected {len(inputs.sequences)}"
                )

            # Calculate fractions and counts
            fractions = []
            counts = []
            results_data = []

            for seq_idx, (original_seq, record) in enumerate(
                zip(inputs.sequences, seq_records)
            ):
                # Handle empty sequences
                if len(original_seq) == 0:
                    fractions.append(0.0)
                    counts.append(0)
                    results_data.append(
                        {
                            "sequence_id": f"seq_{seq_idx}",
                            "length": 0,
                            "lowercase_count": 0,
                            "low_complexity_fraction": 0.0,
                        }
                    )
                    continue

                masked_seq = str(record.seq)
                lowercase_count = sum(1 for c in masked_seq if c.islower())
                fraction = lowercase_count / len(original_seq)

                fractions.append(fraction)
                counts.append(lowercase_count)
                results_data.append(
                    {
                        "sequence_id": f"seq_{seq_idx}",
                        "length": len(original_seq),
                        "lowercase_count": lowercase_count,
                        "low_complexity_fraction": fraction,
                    }
                )

            results_df = pd.DataFrame(results_data)

            return SegmaskerOutput(
                metadata={
                    "num_sequences": len(inputs.sequences),
                    "window": config.window,
                    "locut": config.locut,
                    "hicut": config.hicut,
                },
                low_complexity_fractions=fractions,
                low_complexity_counts=counts,
                sequence_lengths=seq_lengths,
                results_df=results_df,
            )

        except subprocess.TimeoutExpired:
            raise RuntimeError("Segmasker execution timed out after 60 seconds")

        except Exception as e:
            raise RuntimeError(f"Error running segmasker: {str(e)}")
