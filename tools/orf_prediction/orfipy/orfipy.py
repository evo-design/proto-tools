"""
ORF (Open Reading Frame) prediction using Orfipy.

This module provides a standardized interface for ORF prediction using Orfipy,
supporting general ORF prediction and analysis of results.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
from pydantic import ConfigDict, Field, computed_field, field_validator

from bio_programming.bio_tools.tools.utils import BaseConfig, ConfigField
from bio_programming.bio_tools.tools.orf_prediction.orf import ORF
from bio_programming.bio_tools.tools.infra.tool_cache import tool_cache_iterable
from bio_programming.bio_tools.tools.infra.tool_io import BaseToolInput, BaseToolOutput
from bio_programming.bio_tools.tools.tool_registry import tool
from bio_programming.bio_tools.tools.utils import resolve_sequence_ids


# ============================================================================
# Data Models
# ============================================================================
class OrfipyInput(BaseToolInput):
    """Input object for Orfipy ORF (Open Reading Frame) prediction.

    This class defines the input parameters for predicting open reading frames
    in DNA sequences using Orfipy, a fast ORF prediction tool.

    Attributes:
        sequences (List[str]): DNA sequence(s) to analyze for open
            reading frames. Can be provided as:

            - A single DNA sequence string (e.g., ``"ATGTACTATTCAT...TGA"``)
            - A list of DNA sequence strings for batch processing

            Sequences are automatically normalized to uppercase and filtered to
            contain only valid DNA nucleotides (A, T, C, G).
        sequence_ids (Optional[List[str]]): Optional list of sequence identifiers.
            If not provided, sequences are assigned sequential IDs (seq_0, seq_1, ...).
            These IDs are used as ``parent_id`` in the output ORFs.
    """

    sequences: List[str] = Field(
        description="DNA sequence(s) to analyze for open reading frames"
    )
    sequence_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional sequence identifiers (defaults to seq_0, seq_1, ...)",
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value) -> List[str]:
        """Normalize sequences to a list."""
        if isinstance(value, str):
            return [value]
        return value


class OrfipyConfig(BaseConfig):
    """Configuration object for Orfipy ORF prediction.

    This class defines all configuration parameters for running Orfipy to predict
    open reading frames in DNA sequences. Orfipy identifies potential coding
    regions based on start/stop codons and length filters.

    Attributes:
        threads (int): Number of CPU threads to use for processing each sequence.
            Since processing is batched per-sequence, this controls intra-sequence
            parallelism. Must be at least 1. Default: 4.

        start_codons (str): Comma-separated list of start codons to recognize.
            Common options:

            - ``"ATG"``: Standard start codon only (most stringent)
            - ``"ATG,GTG,TTG"``: Include alternative bacterial start codons (default)
            - ``"ATG,CTG,GTG,TTG"``: Extended set for some bacteria

            Default: ``"ATG,GTG,TTG"``.

        stop_codons (str): Comma-separated list of stop codons to recognize.
            Standard genetic code uses:

            - ``"TAA"``: Ochre
            - ``"TAG"``: Amber
            - ``"TGA"``: Opal

            Default: ``"TAA,TAG,TGA"`` (all three standard stop codons).

        strand (str): Which strand(s) to scan for ORFs. Options:

            - ``"f"``: Forward strand only
            - ``"r"``: Reverse strand only
            - ``"b"``: Both strands (default)

            Default: ``"b"``.

        min_len (int): Minimum ORF length in nucleotides (not including stop codon
            unless ``include_stop=True``). ORFs shorter than this are filtered out.
            Common values:

            - ``0``: No minimum (default, accept all ORFs)
            - ``300``: ~100 amino acids, typical for small proteins
            - ``900``: ~300 amino acids, substantial proteins only

            Must be at least 0. Default: 0.

        max_len (int): Maximum ORF length in nucleotides. ORFs longer than this
            are filtered out. Useful for excluding very long ORFs that may span
            multiple genes. Must be at least 1. Default: 10000.

        include_stop (bool): Whether to include the stop codon in the reported
            ORF nucleotide sequence. If ``True``, the stop codon is included in
            both the nucleotide sequence and length calculations. If ``False``,
            the stop codon is excluded. Default: ``True``.

        translation_table (Optional[int]): Optional NCBI genetic code translation
            table number (1-33). Common options:

            - ``None``: Use standard genetic code (defaults to ``1``)
            - ``1``: Standard genetic code
            - ``11``: Bacterial, archaeal, and plant plastid code
            - ``2``: Vertebrate mitochondrial code
            - ``4``: Mold, protozoan, and coelenterate mitochondrial code

            See NCBI documentation for complete list. Range: 1-33. Default: ``None``.

    Note:
        The ``translation_table`` parameter should be converted to a Literal type
        with descriptive string values. TODO
    """

    threads: int = ConfigField(
        title="Number of Threads",
        default=4,
        ge=1,
        description="Number of CPU threads to use",
        hidden=True,
    )
    # TODO: This should be a multi-select. Can we do that?
    start_codons: str = ConfigField(
        title="Start Codons",
        default="ATG,GTG,TTG",
        description="Comma-separated list of start codons",
    )
    # TODO: This should be a multi-select. Can we do that?
    stop_codons: str = ConfigField(
        title="Stop Codons",
        default="TAA,TAG,TGA",
        description="Comma-separated list of stop codons",
    )
    strand: Literal["f", "r", "b"] = ConfigField(
        title="Strand",
        default="b",
        description="Which strand(s) to scan: 'f' (forward), 'r' (reverse), or 'b' (both)",
    )
    min_len: int = ConfigField(
        title="Minimum Length",
        default=0,
        ge=0,
        description="Minimum ORF length in nucleotides",
    )
    max_len: int = ConfigField(
        title="Maximum Length",
        default=10000,
        ge=1,
        description="Maximum ORF length in nucleotides",
    )
    include_stop: bool = ConfigField(
        title="Include Stop",
        default=True,
        description="Whether to include the stop codon in the reported ORF",
    )
    # TODO: This should be a literal with string values that get translated to ints internally
    translation_table: Optional[int] = ConfigField(
        title="Translation Table",
        default=None,
        ge=1,
        le=33,
        description="Optional NCBI translation table (1-33)",
        advanced=True,
    )
    model_config = ConfigDict(extra="ignore")

    @field_validator("strand")
    @classmethod
    def validate_strand(cls, v: str) -> str:
        """Validate strand parameter"""
        valid_strands = {"f", "r", "b"}
        if v not in valid_strands:
            raise ValueError(
                f"Invalid strand '{v}'. Must be one of: {', '.join(valid_strands)}"
            )
        return v


class OrfipyOutput(BaseToolOutput):
    """Output from Orfipy ORF prediction.

    This class encapsulates the results of Orfipy ORF prediction as in-memory
    objects for downstream analysis.

    Attributes:
        predicted_orfs (List[List[ORF]]): List of ORF results per input sequence.
            This is the source of truth for all predicted ORFs. Each inner list
            contains the ORFs found in a single input sequence.

        num_orfs (int): Total number of ORFs predicted across all input sequences.
            Computed property derived from predicted_orfs.

        results_df (pd.DataFrame): Parsed results as a pandas DataFrame
            with the following columns:

            - ``parent_id``: ID of the parent sequence
            - ``orf_id``: Unique ORF identifier within the parent
            - ``amino_acid_sequence``: Translated protein sequence
            - ``nucleotide_sequence``: DNA sequence of the ORF
            - ``amino_acid_length``: Length of protein in amino acids
            - ``nucleotide_length``: Length of ORF in nucleotides
            - ``nucleotide_start``: Start position in parent sequence (1-indexed, inclusive)
            - ``nucleotide_end``: End position in parent sequence (1-indexed, inclusive)
            - ``strand``: Strand direction (``"+"`` for forward, ``"-"`` for reverse)
            - ``frame``: Reading frame (1, 2, or 3)

            Computed property derived from predicted_orfs. Returns empty
            DataFrame if no ORFs were found.
    """

    predicted_orfs: List[List[ORF]] = Field(
        default_factory=list,
        description="List of ORF results per input sequence",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True  # Required for pd.DataFrame in computed_field
    )

    @computed_field
    @property
    def num_orfs(self) -> int:
        """Total number of ORFs predicted across all input sequences."""
        return sum(len(result) for result in self.predicted_orfs)

    @computed_field
    @property
    def num_orfs_per_sequence(self) -> List[int]:
        """Number of ORFs predicted for each input sequence."""
        return [len(result) for result in self.predicted_orfs]

    @computed_field
    @property
    def results_df(self) -> pd.DataFrame:
        """All ORF results as a pandas DataFrame."""
        all_orfs = [orf.model_dump() for sr in self.predicted_orfs for orf in sr]
        return pd.DataFrame(all_orfs)

    @property
    def output_format_options(self) -> List[str]:
        return ["csv", "json", "faa", "fna"]

    @property
    def output_format_default(self) -> str:
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str):
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "csv":
            self.results_df.to_csv(path, index=False)

        elif file_format == "json":
            self.results_df.to_json(path, orient="records", indent=2)

        elif file_format in ["faa", "fna"]:
            with open(path, "w") as f:
                for seq_results in self.predicted_orfs:
                    for orf in seq_results:
                        header = f">{orf.parent_id}_{orf.orf_id} [{orf.nucleotide_start}-{orf.nucleotide_end}]({orf.strand}) frame:{orf.frame}"
                        seq = (
                            orf.amino_acid_sequence
                            if file_format == "faa"
                            else orf.nucleotide_sequence
                        )
                        f.write(f"{header}\n{seq}\n")
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================
@tool_cache_iterable(
    input_iterable_field="sequences",
    output_iterable_field="predicted_orfs",
    tool_name="orfipy-prediction",
)
@tool(
    key="orfipy-prediction",
    label="Orfipy ORF Prediction",
    input=OrfipyInput,
    config=OrfipyConfig,
    output=OrfipyOutput,
    description="ORF (Open Reading Frame) prediction using Orfipy",
)
def run_orfipy_prediction(inputs: OrfipyInput, config: OrfipyConfig) -> OrfipyOutput:
    """Predict open reading frames (ORFs) in DNA sequences using Orfipy.

    Uses Orfipy, a fast ORF prediction tool, to identify potential coding regions.
    Processing is batched but caching is handled per-sequence via @tool_cache_iterable.

    Args:
        inputs (OrfipyInput): Validated input containing one or more DNA sequences
            for ORF prediction.
        config (OrfipyConfig): Validated Orfipy configuration specifying start/stop
            codons, length filters, strand selection, and threading options.

    Returns:
        OrfipyOutput: Structured output containing sequence results.
            Aggregated fields ``num_orfs`` and ``results_df`` are computed properties
            derived from ``predicted_orfs`` on access.

    See Also:
        - Orfipy GitHub: https://github.com/urmi-21/orfipy
        - Orfipy Publication: https://doi.org/10.1093/bioinformatics/btab090

    Examples:
        >>> # Basic ORF prediction (in-memory)
        >>> inputs = OrfipyInput(
        ...     sequences=["ATGTACTATTCATTAA"]
        ... )
        >>> config = OrfipyConfig(
        ...     min_len=12
        ... )
        >>> result = run_orfipy_prediction(inputs, config)
        >>> print(f"Found {result.num_orfs} ORFs")

    Note:
        - Caching is performed per-sequence (based on sequence content).
        - Threads are applied per-sequence during execution.
    """
    from bio_programming.bio_tools.tools.infra.env_manager import EnvManager

    sequence_ids = resolve_sequence_ids(inputs.sequences, inputs.sequence_ids)

    venv_manager = EnvManager(model_name="orfipy")

    input_data = {
        "sequences": inputs.sequences,
        "sequence_ids": sequence_ids,
        "config": {
            "threads": config.threads,
            "start_codons": config.start_codons,
            "stop_codons": config.stop_codons,
            "strand": config.strand,
            "min_len": config.min_len,
            "max_len": config.max_len,
            "include_stop": config.include_stop,
            "translation_table": config.translation_table,
        },
    }

    output_data = venv_manager.call_standalone_script_in_venv(
        script_path=Path(__file__).parent / "standalone" / "run.py",
        input_dict=input_data,
        device="cpu",
        verbose=False,
    )

    # Reconstruct ORF objects from returned dicts
    predicted_orfs = []
    for seq_orfs in output_data["predicted_orfs"]:
        orfs = [ORF(**orf_dict) for orf_dict in seq_orfs]
        predicted_orfs.append(orfs)

    return OrfipyOutput(
        metadata={
            "threads": config.threads,
            "min_len": config.min_len,
            "max_len": config.max_len,
            "strand": config.strand,
        },
        predicted_orfs=predicted_orfs,
    )
