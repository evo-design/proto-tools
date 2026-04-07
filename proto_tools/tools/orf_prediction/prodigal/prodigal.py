"""Prodigal ORF prediction for prokaryotic genomes.

This module provides standardized interfaces for ORF (Open Reading Frame) prediction
using Prodigal for prokaryotic ORF prediction and analysis of results.
"""

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, computed_field, field_validator, model_validator

from proto_tools.tools.orf_prediction.orf import ORF
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
    return_invalid_dna_chars,
)

# ============================================================================
# Translation Table Mapping
# ============================================================================
TranslationTable = Literal[
    "standard",
    "vertebrate_mitochondrial",
    "yeast_mitochondrial",
    "mycoplasma",
    "invertebrate_mitochondrial",
    "ciliate_nuclear",
    "echinoderm_mitochondrial",
    "euplotid_nuclear",
    "bacterial",
    "alternative_yeast_nuclear",
    "ascidian_mitochondrial",
    "alternative_flatworm_mitochondrial",
    "blepharisma_nuclear",
    "chlorophycean_mitochondrial",
    "trematode_mitochondrial",
    "scenedesmus_mitochondrial",
    "thraustochytrium_mitochondrial",
    "rhabdopleuridae_mitochondrial",
    "candidate_division_sr1",
]

TRANSLATION_TABLE_MAP: dict[str, int] = {
    "standard": 1,
    "vertebrate_mitochondrial": 2,
    "yeast_mitochondrial": 3,
    "mycoplasma": 4,
    "invertebrate_mitochondrial": 5,
    "ciliate_nuclear": 6,
    "echinoderm_mitochondrial": 9,
    "euplotid_nuclear": 10,
    "bacterial": 11,
    "alternative_yeast_nuclear": 12,
    "ascidian_mitochondrial": 13,
    "alternative_flatworm_mitochondrial": 14,
    "blepharisma_nuclear": 15,
    "chlorophycean_mitochondrial": 16,
    "trematode_mitochondrial": 21,
    "scenedesmus_mitochondrial": 22,
    "thraustochytrium_mitochondrial": 23,
    "rhabdopleuridae_mitochondrial": 24,
    "candidate_division_sr1": 25,
}


# ============================================================================
# Data Models
# ============================================================================
class ProdigalInput(BaseToolInput):
    """Input object for Prodigal ORF (Open Reading Frame) prediction.

    This class defines the input parameters for predicting genes in prokaryotic
    DNA sequences using Prodigal, a fast and reliable gene prediction tool.

    Attributes:
        input_sequences (list[str]): DNA sequence(s) to analyze for
            genes and open reading frames. Can be provided as:

            - A single DNA sequence string (e.g., ``"ATGCGTAAATAA"``)
            - A list of DNA sequence strings for batch processing

            Sequences are automatically normalized to uppercase and validated to
            contain only valid DNA nucleotides including IUPAC ambiguity codes.
            Empty sequences are not allowed.
    """

    input_sequences: list[str] = InputField(description="DNA sequence(s) to analyze for open reading frames")

    @model_validator(mode="before")
    @classmethod
    def normalize_sequences(cls, data: Any) -> None:
        """Normalize input_sequences from string to list."""
        if isinstance(data.get("input_sequences"), str):
            data["input_sequences"] = [data["input_sequences"]]
        return data  # type: ignore[no-any-return]

    @field_validator("input_sequences")
    @classmethod
    def validate_sequences(cls, sequences: Any) -> None:
        """Validate DNA sequences."""
        if not sequences:
            raise ValueError("At least one sequence is required")

        validated = []
        for seq in sequences:
            # Convert to uppercase first
            seq_upper = seq.upper()
            # Check for invalid characters (allow IUPAC ambiguity codes)
            invalid_chars = return_invalid_dna_chars(seq_upper)
            if invalid_chars:
                raise ValueError(f"Invalid DNA characters in sequence: {', '.join(sorted(invalid_chars))}")
            validated.append(seq_upper)

        return validated  # type: ignore[return-value]


class ProdigalConfig(BaseConfig):
    """Configuration object for Prodigal ORF prediction.

    This class defines all configuration parameters for running Prodigal, a fast
    and reliable protein-coding gene prediction tool specifically designed for
    prokaryotic genomes (bacteria and archaea).

    Attributes:
        meta_mode (bool): Use meta mode for gene prediction. Options:

            - ``True``: Meta mode for short sequences, contigs, or metagenomic data
            (default). Uses pre-trained parameters and works well on incomplete
            genomes or mixed samples.
            - ``False``: Single-genome mode. Trains on the input sequence, requiring
            at least 100kb of sequence for reliable training. Use for complete or
            near-complete genomes.

            Default: ``True``.

        translation_table (TranslationTable): NCBI genetic code for translation.
            Only used in single-genome mode (``meta_mode=False``). In meta mode,
            pre-trained metagenomic models use their own built-in translation
            tables and this parameter is ignored. Common options:

            - ``"bacterial"``: Bacterial, archaeal, and plant plastid code (default)
            - ``"mycoplasma"``: Mycoplasma/Spiroplasma code
            - ``"candidate_division_sr1"``: Candidate division SR1 and Gracilibacteria

            See ``TRANSLATION_TABLE_MAP`` for the complete list. Default: ``"bacterial"``.

        closed_ends (bool): Prevent genes from running off sequence edges. Options:

            - ``False``: Allow partial genes at sequence ends (default). Use for
            contigs, draft genomes, or any incomplete sequences.
            - ``True``: Force all genes to be complete within the sequence. Only
            use for complete circular genomes (chromosomes or plasmids).

            Default: ``False``.

        num_threads (int): Number of CPU threads for parallel processing of multiple
            sequences. Higher values speed up batch processing. By default,
            automatically detects and uses all available CPU cores. Must be at
            least 1. Default: auto-detect all cores.

    Note:
        Prodigal is optimized for prokaryotic genomes. For eukaryotic gene
        prediction, use specialized eukaryotic gene finders.
    """

    meta_mode: bool = ConfigField(
        title="Meta Mode",
        default=True,
        description="Use meta mode for short sequences/fragments (True) or single-genome mode (False)",
        advanced=True,
    )
    translation_table: TranslationTable = ConfigField(
        title="Translation Table",
        default="bacterial",
        description="NCBI genetic code for translation (bacterial = table 11, standard = table 1)",
        advanced=True,
    )
    closed_ends: bool = ConfigField(
        title="Closed Ends",
        default=False,
        description="Prevent genes from running off sequence edges (use True for complete circular genomes)",
        advanced=True,
    )
    num_threads: int = ConfigField(
        title="Number of Threads",
        default_factory=lambda: os.cpu_count() or 1,
        ge=1,
        description="Number of threads for parallel processing (default: auto-detect all available cores)",
        hidden=True,
    )


class ProdigalOutput(BaseToolOutput):
    """Output from Prodigal ORF prediction.

    This class encapsulates the results of Prodigal gene prediction, providing
    detailed information about predicted genes including sequences, coordinates,
    and quality metrics.

    Attributes:
        predicted_orfs (list[list[ORF]]): List of ORF results per input
            sequence. Each inner list contains the ORF objects found in a
            single input sequence. The outer list order matches the input sequences.

        num_orfs: Total number of ORFs predicted across all input sequences.
            Computed property derived from predicted_orfs.

        num_orfs_per_sequence: Number of ORFs predicted for each input sequence.
            Computed property derived from predicted_orfs.
    """

    predicted_orfs: list[list[ORF]] = Field(
        default_factory=list,
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
        return ["gff", "csv", "json", "faa", "fna"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "gff"

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
                        # Prodigal header format style
                        # >sequence_id_gene_id start_pos end_pos strand info...
                        header = f">{orf.parent_id}_{orf.orf_id} # {orf.nucleotide_start} # {orf.nucleotide_end} # {1 if orf.strand == '+' else -1} # {orf.description}"
                        seq = orf.amino_acid_sequence if file_format == "faa" else orf.nucleotide_sequence
                        f.write(f"{header}\n{seq}\n")

        elif file_format == "gff":
            with open(path, "w") as f:
                f.write("##gff-version 3\n")
                for seq_results in self.predicted_orfs:
                    if not seq_results:
                        continue
                    f.write(f"# Sequence Data: {seq_results[0].parent_id}\n")
                    for orf in seq_results:
                        # columns: seqid source type start end score strand phase attributes
                        # Use metrics directly or fall back to description if needed
                        partial_val = f"{orf.metrics.get('partial_begin', 0)}{orf.metrics.get('partial_end', 0)}"
                        attributes = (
                            f"ID={orf.parent_id}_{orf.orf_id};"
                            f"partial={partial_val};"
                            f"start_type={orf.metrics.get('start_type')};"
                            f"rbs_motif={orf.metrics.get('rbs_motif')};"
                            f"rbs_spacer={orf.metrics.get('rbs_spacer')};"
                            f"gc_cont={orf.gc_content:.3f}"
                        )
                        f.write(
                            f"{orf.parent_id}\tProdigal_v2.6.3\tCDS\t"
                            f"{orf.nucleotide_start}\t{orf.nucleotide_end}\t.\t"
                            f"{orf.strand}\t0\t{attributes}\n"
                        )
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ProdigalInput(input_sequences=["ATGCGTAAATAA"])


@tool(
    key="prodigal-prediction",
    label="Prodigal ORF Prediction",
    category="orf_prediction",
    input_class=ProdigalInput,
    config_class=ProdigalConfig,
    output_class=ProdigalOutput,
    description="Prokaryotic ORF and gene prediction using Prodigal",
    example_input=example_input,
    iterable_input_field="input_sequences",
    iterable_output_field="predicted_orfs",
    cacheable=True,
)
def run_prodigal_prediction(inputs: ProdigalInput, config: ProdigalConfig, instance: Any = None) -> ProdigalOutput:
    """Predict genes in prokaryotic DNA sequences using Prodigal.

    Uses pyrodigal Python bindings for gene prediction in bacterial and archaeal
    genomes. Prodigal identifies protein-coding genes, including partial genes
    at sequence ends, and provides detailed annotations including ribosome binding
    sites and start codon types.

    Args:
        inputs (ProdigalInput): Validated input containing one or more prokaryotic
            DNA sequences for gene prediction.
        config (ProdigalConfig): Validated Prodigal configuration specifying
            prediction mode (meta vs. single-genome), genetic code, and threading.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ProdigalOutput: Structured output

    Raises:
        ValueError: If input sequences are empty, contain invalid DNA characters,
            or are too short for reliable prediction.
        RuntimeError: If Prodigal prediction fails for any sequence.

    See Also:
        - Prodigal paper: https://doi.org/10.1186/1471-2105-11-119
        - Pyrodigal documentation: https://pyrodigal.readthedocs.io/

    Example:
        >>> inputs = ProdigalInput(input_sequences=["ATGCGTAAATAA"])
        >>> config = ProdigalConfig(meta_mode=True)
        >>> result = run_prodigal_prediction(inputs, config)
        >>> print(f"Found {result.num_orfs} genes")

    Note:
        - Prodigal works best on sequences longer than 20kb for training
        - Use meta mode for short contigs, metagenomic data, or draft genomes
        - Use single-genome mode only for complete genomes (>100kb recommended)
        - Set ``closed_ends=True`` only for complete circular genomes
    """
    output_data = ToolInstance.dispatch(
        "prodigal",
        {
            "device": "cpu",
            "sequences": inputs.input_sequences,
            "config": {
                "meta_mode": config.meta_mode,
                "closed_ends": config.closed_ends,
                "num_threads": config.num_threads,
                "translation_table": TRANSLATION_TABLE_MAP[config.translation_table],
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

    return ProdigalOutput(
        metadata={
            "meta_mode": config.meta_mode,
            "num_threads": config.num_threads,
            "num_input_sequences": len(inputs.input_sequences),
        },
        predicted_orfs=predicted_orfs,
    )
