"""Local BLAST+ search tool."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Literal

import pandas as pd
from pydantic import Field, field_validator

from bio_programming_tools.utils.tool_cache import tool_cache
from bio_programming_tools.utils.tool_io import BaseToolInput
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField

from .online_blast import BLAST_PROGRAMS, BlastOutput


# ============================================================================
# Data Models
# ============================================================================
# Input:
class LocalBlastInput(BaseToolInput):
    """Input object for local BLAST+ search.

    This class defines the input parameters for running BLAST searches using
    a local installation of BLAST+. Unlike online BLAST, local BLAST requires
    the query to be provided as a FASTA file.

    Attributes:
        query (str): Path to a FASTA file containing one or more query sequences.
            The file must exist and be in valid FASTA format. All sequences
            in the file will be used as queries in the BLAST search.

    Raises:
        ValueError: If the specified query file does not exist.
    """

    query: str = Field(description="Path to FASTA file containing a query sequence")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate that query file exists"""
        if not Path(v).exists():
            raise ValueError(f"Query file not found: {v}")
        return v


# Output:
LocalBlastOutput = BlastOutput

# Config:
class LocalBlastConfig(BaseConfig):
    """Configuration object for local BLAST+ search.

    This class defines all configuration parameters for running BLAST searches
    using a local BLAST+ installation. It provides control over the search algorithm,
    database location, computational resources, and search parameters.

    Attributes:
        program (str): The BLAST algorithm to use for the search. Options include:

            - ``"blastn"``: Nucleotide query vs nucleotide database
            - ``"blastp"``: Protein query vs protein database
            - ``"blastx"``: Translated nucleotide query vs protein database
            - ``"tblastn"``: Protein query vs translated nucleotide database
            - ``"tblastx"``: Translated nucleotide query vs translated nucleotide database

        local_db (str): Path to the local BLAST database (without file extensions).
            This should be the base name created by ``makeblastdb``. For example,
            if ``makeblastdb`` created files ``"mydb.nhr"``, ``"mydb.nin"``,
            ``"mydb.nsq"``, then ``local_db`` should be ``"mydb"`` or
            ``"/full/path/to/mydb"``.

        num_threads (int): Number of CPU threads to use for the search. More threads
            can significantly speed up searches on multi-core systems. Typical
            values range from 1 to the number of available CPU cores.

        additional_params (Dict[str, str | int | float | bool]): Dictionary
            of additional parameters to pass to the BLAST command. Common options
            include:

            - ``"evalue"``: E-value threshold for reporting hits (default: 10.0)
            - ``"word_size"``: Word size for initial matches (default varies by program)
            - ``"max_target_seqs"``: Maximum number of aligned sequences to keep
            - ``"max_hsps"``: Maximum number of HSPs per subject sequence
            - ``"outfmt"``: Output format (default: 6 for tabular)
            - ``"qcov_hsp_perc"``: Minimum query coverage per HSP (percentage)
            - ``"perc_identity"``: Minimum percent identity for hits
            - ``"culling_limit"``: Delete hits that are enveloped by superior hits
            - ``"best_hit_overhang"``: Best Hit algorithm overhang value
            - ``"best_hit_score_edge"``: Best Hit algorithm score edge value
            - ``"subject_besthit"``: Only show best hit per subject sequence
    """
    program: Literal[BLAST_PROGRAMS] = ConfigField(
        title="BLAST Program name",
        default="blastn",
        description="Select which BLAST program to use for the search",
    )
    local_db: str = ConfigField(
        title="Local BLAST Database",
        description="Specifies a path to the local BLAST database to search against",
        json_schema_extra={"advanced": False},
    )

    num_threads: int = ConfigField(
        title="Number of Threads",
        default=4,
        ge=1,
        description="Number of threads to use",
        json_schema_extra={"advanced": False},
    )
    additional_params: Dict[str, str | int | float | bool] = ConfigField(
        title="Additional Parameter Dictionary",
        default_factory=dict,
        description="Additional parameters for BLAST command (e.g., outfmt, max_target_seqs, word_size)",
        json_schema_extra={"advanced": True},
    )


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="local-blast",
    label="Local BLAST",
    input=LocalBlastInput,
    config=LocalBlastConfig,
    output=LocalBlastOutput,
    description="Run BLAST+ locally and return results",
)
@tool_cache("local-blast")
def run_local_blast_search(inputs: LocalBlastInput, config: LocalBlastConfig) -> LocalBlastOutput:
    """
    Run BLAST+ locally and return results as DataFrame.

    This is the standardized tool interface following the registry pattern.
    Returns structured output with metadata tracking.

    Args:
        inputs (LocalBlastInput): Validated local BLAST input
        config (LocalBlastConfig): Validated local BLAST configuration

    Returns:
        Structured output with BLAST results DataFrame

    Raises:
        RuntimeError: If the local BLAST search fails.

    Examples:
        >>> from bio_programming_tools.tools.gene_annotation import run_local_blast_search, LocalBlastInput, LocalBlastConfig
        >>> inputs = LocalBlastInput(query="query.fasta")
        >>> config = LocalBlastConfig(
        ...     local_db="/data/blast/nr",
        ...     program="blastp",
        ...     num_threads=8
        ... )
        >>> result = run_local_blast_search(inputs, config)
        >>> print(f"Found {result.num_hits} hits")
    """
    from bio_programming_tools.utils.env_manager import EnvManager

    venv_manager = EnvManager(model_name="blast")

    input_data = {
        "operation": "local_blast",
        "program": config.program,
        "query_path": inputs.query,
        "db": config.local_db,
        "num_threads": config.num_threads,
        "additional_params": config.additional_params,
    }

    output_data = venv_manager.call_standalone_script_in_venv(
        script_path=Path(__file__).parent / "standalone" / "run.py",
        input_dict=input_data,
        device="cpu",
    )

    # Parse raw tabular output into DataFrame
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

    raw_output = output_data["stdout"]
    if not raw_output.strip():
        results_df = pd.DataFrame(columns=cols)
    else:
        results_df = pd.read_csv(io.StringIO(raw_output), sep="\t", names=cols)

    return LocalBlastOutput(
        metadata={
            "program": config.program,
            "database": config.local_db,
            "num_threads": config.num_threads,
        },
        results_df=results_df,
        num_hits=len(results_df),
    )
