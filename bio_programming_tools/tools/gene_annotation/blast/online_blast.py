"""Online NCBI BLAST search tool."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd
from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
from pydantic import ConfigDict, Field

from bio_programming_tools.utils.tool_cache import tool_cache
from bio_programming_tools.utils.tool_io import BaseToolInput, BaseToolOutput
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField

BLAST_PROGRAMS = Literal["blastn", "blastp", "blastx", "tblastn", "tblastx"]
BLAST_DATABASES = Literal[
    "nt",  # Nucleotide collection
    "nr",  # Non-redundant protein sequences
    "refseq_rna",  # RefSeq RNA sequences
    "refseq_protein",  # RefSeq protein sequences
    "swissprot",  # Swiss-Prot protein sequences
    "pdb",  # Protein Data Bank
    "pataa",  # Patent protein sequences
    "patnt",  # Patent nucleotide sequences
]

# ============================================================================
# Data Models
# ============================================================================
# Input:
class OnlineBlastInput(BaseToolInput):
    """
    Input object for online NCBI BLAST search.

    This class defines the input parameters required for submitting a BLAST query
    to NCBI's online BLAST service. The query can be provided either as a raw
    sequence string or as a path to a FASTA file.

    Attributes:
        query (str): Either a raw nucleotide/protein sequence string (e.g.,
            ``"ATGCGTAAA"``) or a file path to a FASTA file containing the query
            sequence. If a file path is provided, the first sequence in the file
            will be used as the query.

    Note:
        The file path option will be removed in a future version. TODO
    """

    query: str = Field(
        description="Query sequence or path to FASTA file containing a query sequence"
    )


# Output:
class BlastOutput(BaseToolOutput):
    """Output from BLAST search.

    This class encapsulates the results of a BLAST search, providing both
    structured tabular results and summary statistics. The results are returned
    in the standard BLAST tabular format (outfmt 6).

    Attributes:
        results_df (Optional[pd.DataFrame]): A pandas DataFrame containing BLAST
            results in standard tabular format with the following columns:

            - ``qseqid``: Query sequence identifier
            - ``sseqid``: Subject (database) sequence identifier
            - ``pident``: Percentage of identical matches
            - ``length``: Alignment length (number of positions)
            - ``mismatch``: Number of mismatches
            - ``gapopen``: Number of gap openings
            - ``qstart``: Start position in query sequence
            - ``qend``: End position in query sequence
            - ``sstart``: Start position in subject sequence
            - ``send``: End position in subject sequence
            - ``evalue``: Expected value (E-value)
            - ``bitscore``: Bit score
            TODO: Package these into a custom object, not a pandas DataFrame.

            Returns ``None`` if no hits are found.

        num_hits (int): Total number of alignment hits found in the search. A
            value of 0 indicates no significant alignments were detected.
    """
    results_df: Optional[pd.DataFrame] = ConfigField(
        default=None,
        description="DataFrame with BLAST results",
    )
    num_hits: int = ConfigField(
        default=0,
        description="Number of BLAST hits found",
    )

    @property
    def output_format_options(self) -> List[str]:
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str):
        import warnings

        if self.results_df is None or len(self.results_df) == 0:
            warnings.warn(
                "No BLAST results to export. The search returned no hits.",
                UserWarning,
                stacklevel=2
            )
            return

        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "csv":
            self.results_df.to_csv(path, index=False)

        elif file_format == "json":
            self.results_df.to_json(path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

    model_config = ConfigDict(
        arbitrary_types_allowed=True  # Allow pandas DataFrame
    )


# Output (alias):
OnlineBlastOutput = BlastOutput

# Config:
class OnlineBlastConfig(BaseConfig):
    """Configuration object for online NCBI BLAST search.

    This class defines all configuration parameters for running BLAST searches
    against NCBI's online databases. It allows selection of the BLAST program type,
    target database, and additional search parameters.

    Attributes:
        program (str): The BLAST algorithm to use for the search. Options include:

            - ``"blastn"``: Nucleotide query vs nucleotide database
            - ``"blastp"``: Protein query vs protein database
            - ``"blastx"``: Translated nucleotide query vs protein database
            - ``"tblastn"``: Protein query vs translated nucleotide database
            - ``"tblastx"``: Translated nucleotide query vs translated nucleotide database

        database (str): The NCBI database to search against. Options include:

            - ``"nt"``: Nucleotide collection (all GenBank+EMBL+DDBJ+PDB sequences)
            - ``"nr"``: Non-redundant protein sequences
            - ``"refseq_rna"``: NCBI RefSeq RNA sequences
            - ``"refseq_protein"``: NCBI RefSeq protein sequences
            - ``"swissprot"``: Curated protein sequence database
            - ``"pdb"``: Protein Data Bank sequences
            - ``"pataa"``: Patent protein sequences
            - ``"patnt"``: Patent nucleotide sequences

        additional_params (Dict[str, str | int | float | bool]): Dictionary
            of additional parameters to pass to ``NCBIWWW.qblast``. Common options
            include:

            - ``"expect"``: E-value threshold (default: 10.0)
            - ``"word_size"``: Word size for initial matches (default varies by program)
            - ``"matrix"``: Scoring matrix name for protein searches (e.g., ``"BLOSUM62"``)
            - ``"gapcosts"``: Gap opening and extension costs as "open extend" string
            - ``"nucl_reward"``: Reward for nucleotide match (blastn only)
            - ``"nucl_penalty"``: Penalty for nucleotide mismatch (blastn only)
            - ``"hitlist_size"``: Number of database sequences to keep
    """
    program: BLAST_PROGRAMS = ConfigField(
        title="BLAST Program",
        default="blastn",
        description="Selects which BLAST program is utilized for the search",
    )
    database: BLAST_DATABASES = ConfigField(
        title="BLAST Database",
        default="nt",
        description="Specifies the NCBI database to search",
    )

    additional_params: Dict[str, str | int | float | bool] = ConfigField(
        title="Additional Parameters",
        default_factory=dict,
        description="Additional parameters for NCBIWWW.qblast (e.g., word_size, expect, matrix)",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="online-blast",
    label="Online BLAST Search (NCBI)",
    input=OnlineBlastInput,
    config=OnlineBlastConfig,
    output=OnlineBlastOutput,
    description="Submit query to online NCBI BLAST and return results",
)
@tool_cache("online-blast")
def run_online_blast_search(inputs: OnlineBlastInput, config: OnlineBlastConfig) -> OnlineBlastOutput:
    """
    Submit query to online NCBI BLAST and return results as DataFrame.

    This is the standardized tool interface following the registry pattern.
    Returns structured output with metadata tracking.

    Args:
        inputs (OnlineBlastInput): Validated online BLAST input
        config (OnlineBlastConfig): Validated online BLAST configuration

    Raises:
        RuntimeError: If the online BLAST search fails.

    Returns:
        Structured output with BLAST results DataFrame

    Examples:
        >>> from bio_programming_tools.tools.gene_annotation import run_online_blast_search, OnlineBlastInput, OnlineBlastConfig
        >>> inputs = OnlineBlastInput(query="ATGCGTAAA")
        >>> config = OnlineBlastConfig(
        ...     program="blastn",
        ...     database="nt"
        ... )
        >>> result = run_online_blast_search(inputs, config)
        >>> print(f"Found {result.num_hits} hits")
    """

    # Process query
    if Path(inputs.query).exists():
        seq_record = next(SeqIO.parse(inputs.query, "fasta"))
        query_seq = str(seq_record.seq)
    else:
        query_seq = inputs.query

    # Send BLAST request
    try:
        handle = NCBIWWW.qblast(
            program=config.program,
            database=config.database,
            sequence=query_seq,
            format_type="XML",
            **config.additional_params,
        )
    except Exception as e:
        raise RuntimeError(f"Online BLAST search failed: {e}")

    raw_xml = handle.read()
    handle.close()

    # Parse XML results and convert to DataFrame
    blast_records = list(NCBIXML.parse(io.StringIO(raw_xml)))
    results_df = _blast_results_to_df(blast_records)

    return OnlineBlastOutput(
        metadata={
            "program": config.program,
            "database": config.database,
            "query_length": len(query_seq),
        },
        results_df=results_df,
        num_hits=len(results_df),
    )


# ============================================================================
# Helper Functions
# ============================================================================
def _blast_results_to_df(blast_records) -> pd.DataFrame:
    """Convert Biopython BLAST records to a standard tabular DataFrame.

    Args:
        blast_records: List of Bio.Blast.Record objects

    Returns:
        DataFrame with standard BLAST tabular columns
    """
    # Define standard BLAST cols
    cols = [
        "qseqid",
        "sseqid",
        "pident",
        "length",
        "mismatch",
        "gapopen",
        "qstart",
        "qend",
        "sstart",
        "send",
        "evalue",
        "bitscore",
    ]

    hits = []
    for record in blast_records:
        query_id = record.query.split()[0]
        for alignment in record.alignments:
            for hsp in alignment.hsps:
                align_length = getattr(hsp, "align_length", len(hsp.match))
                identities = hsp.identities
                percent_identity = (
                    (identities / align_length) * 100 if align_length > 0 else 0
                )

                hits.append(
                    {
                        "qseqid": query_id,
                        "sseqid": getattr(alignment, "accession", alignment.hit_id),
                        "pident": percent_identity,
                        "length": align_length,
                        "mismatch": align_length - identities,
                        "gapopen": getattr(hsp, "gaps", 0),
                        "qstart": hsp.query_start,
                        "qend": hsp.query_end,
                        "sstart": hsp.sbjct_start,
                        "send": hsp.sbjct_end,
                        "evalue": hsp.expect,
                        "bitscore": hsp.bits,
                    }
                )

    return pd.DataFrame(hits) if hits else pd.DataFrame(columns=cols)
