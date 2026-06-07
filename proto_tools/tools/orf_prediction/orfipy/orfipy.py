"""Standardized interface for ORF prediction using Orfipy."""

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, computed_field, field_validator, model_validator

from proto_tools.tools.orf_prediction.orf import ORF
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
# Codon & Translation Table Types
# ============================================================================
StartCodon = Literal["ATG", "GTG", "TTG", "CTG"]
StopCodon = Literal["TAA", "TAG", "TGA"]

OrfipyTranslationTable = Literal[
    "standard",
    "vertebrate_mitochondrial",
    "yeast_mitochondrial",
    "mold_protozoan_mitochondrial",
    "invertebrate_mitochondrial",
    "ciliate_nuclear",
    "echinoderm_mitochondrial",
    "euplotid_nuclear",
    "bacterial",
    "alternative_yeast_nuclear",
    "ascidian_mitochondrial",
    "alternative_flatworm_mitochondrial",
    "chlorophycean_mitochondrial",
    "trematode_mitochondrial",
    "scenedesmus_mitochondrial",
    "thraustochytrium_mitochondrial",
    "rhabdopleuridae_mitochondrial",
    "candidate_division_sr1",
    "pachysolen_nuclear",
    "karyorelict_nuclear",
    "condylostoma_nuclear",
    "mesodinium_nuclear",
    "peritrich_nuclear",
]

# NOTE: Values are orfipy's internal sequential keys (1-23), NOT NCBI transl_table
# numbers. Orfipy's translation_tables_dict uses contiguous numbering that diverges
# from NCBI starting at key 7 (NCBI skips tables 7, 8, 15, 17-20, but orfipy
# numbers them sequentially). See: orfipy/translation_tables.py
logger = logging.getLogger(__name__)

ORFIPY_TRANSLATION_TABLE_MAP: dict[str, int] = {
    "standard": 1,
    "vertebrate_mitochondrial": 2,
    "yeast_mitochondrial": 3,
    "mold_protozoan_mitochondrial": 4,
    "invertebrate_mitochondrial": 5,
    "ciliate_nuclear": 6,
    "echinoderm_mitochondrial": 7,  # NCBI 9
    "euplotid_nuclear": 8,  # NCBI 10
    "bacterial": 9,  # NCBI 11
    "alternative_yeast_nuclear": 10,  # NCBI 12
    "ascidian_mitochondrial": 11,  # NCBI 13
    "alternative_flatworm_mitochondrial": 12,  # NCBI 14
    "chlorophycean_mitochondrial": 13,  # NCBI 16
    "trematode_mitochondrial": 14,  # NCBI 21
    "scenedesmus_mitochondrial": 15,  # NCBI 22
    "thraustochytrium_mitochondrial": 16,  # NCBI 23
    "rhabdopleuridae_mitochondrial": 17,  # NCBI 24
    "candidate_division_sr1": 18,  # NCBI 25
    "pachysolen_nuclear": 19,  # NCBI 26
    "karyorelict_nuclear": 20,  # NCBI 27
    "condylostoma_nuclear": 21,  # NCBI 28
    "mesodinium_nuclear": 22,  # NCBI 29
    "peritrich_nuclear": 23,  # NCBI 30
}


# ============================================================================
# Data Models
# ============================================================================
class OrfipyInput(BaseToolInput):
    """Input object for Orfipy ORF (Open Reading Frame) prediction.

    This class defines the input parameters for predicting open reading frames
    in DNA sequences using Orfipy, a fast ORF prediction tool.

    Attributes:
        sequences (list[str]): DNA sequence(s) to analyze for open
            reading frames. Can be provided as:

            - A single DNA sequence string (e.g., ``"ATGTACTATTCAT...TGA"``)
            - A list of DNA sequence strings for batch processing

            Sequences are automatically normalized to uppercase and filtered to
            contain only valid DNA nucleotides (A, T, C, G). Each sequence is
            labeled positionally (``seq_0``, ``seq_1``, ...) as ``parent_id`` in
            the output ORFs; results are returned in input order.
    """

    sequences: list[str] = InputField(
        title="Sequences",
        description="DNA sequence(s) to analyze for open reading frames",
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value: Any) -> list[str]:
        """Normalize sequences to a list."""
        if isinstance(value, str):
            return [value]
        return value  # type: ignore[no-any-return]


class OrfipyConfig(BaseConfig):
    """Configuration object for Orfipy ORF prediction.

    This class defines all configuration parameters for running Orfipy to predict
    open reading frames in DNA sequences. Orfipy identifies potential coding
    regions based on start/stop codons and length filters.

    Attributes:
        threads (int): Number of CPU threads to use for processing each sequence.
            Since processing is batched per-sequence, this controls intra-sequence
            parallelism. Must be at least 1. Default: 4.

        start_codons (list[StartCodon]): Start codons to recognize for ORF
            prediction. Multi-select from:

            - ``"ATG"``: Standard start codon
            - ``"GTG"``: Alternative bacterial start codon
            - ``"TTG"``: Alternative bacterial start codon
            - ``"CTG"``: Alternative bacterial start codon

            Default: ``["ATG", "GTG", "TTG"]``.

        stop_codons (list[StopCodon]): Stop codons to recognize for ORF
            prediction. Multi-select from:

            - ``"TAA"``: Ochre stop codon
            - ``"TAG"``: Amber stop codon
            - ``"TGA"``: Opal stop codon

            Default: ``["TAA", "TAG", "TGA"]``.

        strand (Literal['f', 'r', 'b']): Which strand(s) to scan for ORFs. Options:

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

        max_len (int): Maximum ORF length in nucleotides. ORFs longer than this are silently filtered
            out by orfipy; raise (e.g. ``1_000_000_000``) for genome-scale inputs. Default: 10000.

        include_stop (bool): Whether to include the stop codon in the reported
            ORF nucleotide sequence. If ``True``, the stop codon is included in
            both the nucleotide sequence and length calculations. If ``False``,
            the stop codon is excluded. Default: ``True``.

        ignore_case (bool): Treat lowercase (soft-masked) nucleotides as
            ORF-eligible. Default: ``False``.

        partial_3 (bool): Report ORFs missing a stop codon at the 3' end of
            the sequence. Default: ``False``.

        partial_5 (bool): Report ORFs missing a start codon at the 5' end of
            the sequence. Default: ``False``.

        between_stops (bool): Report ORFs spanning stop-to-stop (start codons
            ignored). Default: ``False``.

        translation_table (OrfipyTranslationTable | None): NCBI genetic code for
            translation. ``None`` uses the standard genetic code (table 1).
            Only tables supported by orfipy's built-in translation table dict
            are available (NCBI tables 1-6, 9-14, 16, 21-30). Common options:

            - ``"standard"``: Standard genetic code (NCBI table 1)
            - ``"bacterial"``: Bacterial, archaeal, and plant plastid (NCBI table 11)
            - ``"vertebrate_mitochondrial"``: Vertebrate mitochondrial (NCBI table 2)
            - ``"mold_protozoan_mitochondrial"``: Mold/protozoan mitochondrial (NCBI table 4)

            See ``ORFIPY_TRANSLATION_TABLE_MAP`` for the complete list. Default: ``None``.
    """

    threads: int = ConfigField(
        title="Number of Threads",
        default=4,
        ge=1,
        description="CPU threads passed to orfipy --procs",
    )
    start_codons: list[StartCodon] = ConfigField(
        title="Start Codons",
        default=["ATG", "GTG", "TTG"],
        min_length=1,
        description="Start codons to recognize for ORF prediction",
    )
    stop_codons: list[StopCodon] = ConfigField(
        title="Stop Codons",
        default=["TAA", "TAG", "TGA"],
        min_length=1,
        description="Stop codons to recognize for ORF prediction",
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
        description="Min ORF length in nt; 0 keeps all ORFs",
    )
    max_len: int = ConfigField(
        title="Maximum Length",
        default=10000,
        ge=1,
        description="Max ORF length in nt (caps payloads); raise to 1_000_000_000 for genome-scale",
    )
    include_stop: bool = ConfigField(
        title="Include Stop",
        default=True,
        description="Include the stop codon in the reported ORF nucleotide sequence and length",
    )
    ignore_case: bool = ConfigField(
        title="Ignore Case",
        default=False,
        description="Treat lowercase (soft-masked) nucleotides as ORF-eligible",
    )
    partial_3: bool = ConfigField(
        title="Allow 3' Partial ORFs",
        default=False,
        description="Report ORFs missing a stop codon at the sequence end",
    )
    partial_5: bool = ConfigField(
        title="Allow 5' Partial ORFs",
        default=False,
        description="Report ORFs missing a start codon at the sequence start",
    )
    between_stops: bool = ConfigField(
        title="Between Stops",
        default=False,
        description="Report ORFs spanning stop-to-stop (ignores start codons; implies partial_3 + partial_5)",
    )
    translation_table: OrfipyTranslationTable | None = ConfigField(
        title="Translation Table",
        default=None,
        description="NCBI genetic code for translation (None = standard genetic code)",
    )
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_length_range(self) -> "OrfipyConfig":
        """Reject inverted length ranges (orfipy silently returns 0 ORFs)."""
        if self.min_len > self.max_len:
            raise ValueError(f"min_len ({self.min_len}) must be <= max_len ({self.max_len})")
        return self


class OrfipyOutput(BaseToolOutput):
    """Output from Orfipy ORF prediction.

    This class encapsulates the results of Orfipy ORF prediction as in-memory
    objects for downstream analysis.

    Attributes:
        predicted_orfs (list[list[ORF]]): List of ORF results per input sequence.
            This is the source of truth for all predicted ORFs. Each inner list
            contains the ORFs found in a single input sequence.

        num_orfs: Total number of ORFs predicted across all input sequences.
            Computed property derived from predicted_orfs.

        num_orfs_per_sequence: Number of ORFs predicted for each input sequence.
            Computed property derived from predicted_orfs.
    """

    predicted_orfs: list[list[ORF]] = Field(
        default_factory=list,
        title="Predicted ORFs",
        description="List of ORF results per input sequence",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_orfs(self) -> int:
        """Total number of ORFs predicted across all input sequences."""
        return sum(len(result) for result in self.predicted_orfs)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_orfs_per_sequence(self) -> list[int]:
        """Number of ORFs predicted for each input sequence."""
        return [len(result) for result in self.predicted_orfs]

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["csv", "json", "faa", "fna"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format in ("csv", "json"):
            import pandas as pd

            all_orfs = [orf.to_flat_dict() for sr in self.predicted_orfs for orf in sr]
            df = pd.DataFrame(all_orfs)
            if file_format == "csv":
                df.to_csv(path, index=False)
            else:
                df.to_json(path, orient="records", indent=2)

        elif file_format in ["faa", "fna"]:
            with open(path, "w") as f:
                for seq_results in self.predicted_orfs:
                    for orf in seq_results:
                        header = f">{orf.parent_id}_{orf.orf_id} [{orf.nucleotide_start}-{orf.nucleotide_end}]({orf.strand}) frame:{orf.frame}"
                        seq = orf.amino_acid_sequence if file_format == "faa" else orf.nucleotide_sequence
                        f.write(f"{header}\n{seq}\n")
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return OrfipyInput(sequences=["ATGTACTATTCATTAA"])


@tool(
    key="orfipy-prediction",
    label="Orfipy ORF Prediction",
    category="orf_prediction",
    input_class=OrfipyInput,
    config_class=OrfipyConfig,
    output_class=OrfipyOutput,
    description="ORF (Open Reading Frame) prediction using Orfipy",
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="predicted_orfs",
    cacheable=True,
)
def run_orfipy_prediction(inputs: OrfipyInput, config: OrfipyConfig, instance: Any = None) -> OrfipyOutput:
    """Predict open reading frames (ORFs) in DNA sequences using Orfipy.

    Uses Orfipy, a fast ORF prediction tool, to identify potential coding regions.
    Processing is batched but caching is handled per-sequence via cacheable=True.

    Args:
        inputs (OrfipyInput): Validated input containing one or more DNA sequences
            for ORF prediction.
        config (OrfipyConfig): Validated Orfipy configuration specifying start/stop
            codons, length filters, strand selection, and threading options.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        OrfipyOutput: Structured output containing sequence results.
            Aggregated fields ``num_orfs`` and ``num_orfs_per_sequence`` are computed
            properties derived from ``predicted_orfs`` on access.

    See Also:
        - Orfipy GitHub: https://github.com/urmi-21/orfipy
        - Orfipy Publication: https://doi.org/10.1093/bioinformatics/btab090

    Examples:
        >>> # Basic ORF prediction (in-memory)
        >>> inputs = OrfipyInput(sequences=["ATGTACTATTCATTAA"])
        >>> config = OrfipyConfig(min_len=12)
        >>> result = run_orfipy_prediction(inputs, config)
        >>> print(f"Found {result.num_orfs} ORFs")

    Note:
        - Caching is performed per-sequence (based on sequence content).
        - Threads are applied per-sequence during execution.
    """
    sequence_ids = [f"seq_{i}" for i in range(len(inputs.sequences))]

    long_inputs = [sid for sid, seq in zip(sequence_ids, inputs.sequences, strict=True) if len(seq) > config.max_len]
    if long_inputs:
        logger.warning(
            "orfipy: %d input sequence(s) exceed max_len=%d nt; ORFs longer than max_len will be filtered out. "
            "Inputs: %s",
            len(long_inputs),
            config.max_len,
            long_inputs[:5] + (["..."] if len(long_inputs) > 5 else []),
        )

    output_data = ToolInstance.dispatch(
        "orfipy",
        {
            "device": "cpu",
            "sequences": inputs.sequences,
            "sequence_ids": sequence_ids,
            "config": {
                "threads": config.threads,
                "start_codons": ",".join(config.start_codons),
                "stop_codons": ",".join(config.stop_codons),
                "strand": config.strand,
                "min_len": config.min_len,
                "max_len": config.max_len,
                "include_stop": config.include_stop,
                "ignore_case": config.ignore_case,
                "partial_3": config.partial_3,
                "partial_5": config.partial_5,
                "between_stops": config.between_stops,
                "translation_table": (
                    ORFIPY_TRANSLATION_TABLE_MAP[config.translation_table]
                    if config.translation_table is not None
                    else None
                ),
            },
        },
        instance=instance,
        config=config,
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
