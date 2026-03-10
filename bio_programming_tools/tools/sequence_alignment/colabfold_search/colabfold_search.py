"""
ColabFold MSA Search implementation.

This module provides a standardized interface for generating Multiple Sequence
Alignments (MSAs) using ColabFold's local database search with MMSeqs2.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bio_programming_tools.tools.sequence_alignment.msas import MSA
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField
from bio_programming_tools.utils.tool_io import BaseToolInput, BaseToolOutput

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

# Default cache directory for MSA files
DEFAULT_OUTPUT_DIR = Path.home() / ".cache" / "bio-programming" / "colabfold_search"

# Default database directory (in the same directory as this file)
DEFAULT_DB_DIR = Path(__file__).parent / "databases"

# TODO: In the future, we should remove this hardcoded path and use the DEFAULT_DB_DIR
# - This is a temporary solution for the backend that allows us to use the same database instead of downloading again
CHIMERA_COLABFOLD_DB_LOCATION = "/large_storage/hielab/brk/databases/colabfold"


# Input:


class ColabfoldSearchQuery(BaseModel):
    """Represents a single protein sequence to search for homologs.

    This class defines a query for MSA generation. Each query consists of a
    protein sequence and an optional identifier.

    Attributes:
        sequence (str): Protein sequence to search for homologs. Must be a
            non-empty string containing amino acid characters.
        sequence_id (Optional[str]): Optional identifier for this sequence.
            Used for output file naming and result tracking. If not provided,
            will be auto-generated as seq_0, seq_1, etc.
    """

    sequence: str = Field(description="Protein sequence to search for homologs")
    sequence_id: Optional[str] = Field(
        default=None,
        description="Optional sequence identifier (auto-generated if not provided)",
    )

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        """Validate that sequence is a non-empty string."""
        if not isinstance(v, str):
            raise ValueError(f"Sequence must be a string, got {type(v)}")
        if not v or not v.strip():
            raise ValueError("Sequence cannot be empty")
        return v.strip()


class ColabfoldSearchInput(BaseToolInput):
    """Input object for ColabFold MSA search.

    This class provides a flexible interface for specifying one or more protein
    sequences to search for homologs. After validation, always contains a list
    of ColabfoldSearchQuery instances.

    Supports multiple input formats:
    - List of ColabfoldSearchQuery instances or single ColabfoldSearchQuery instance  (explicit format)
    - List of sequence strings or single sequence string (each auto-assigned seq_0, seq_1, ...)
    - List of tuples or single tuple of the form (sequence, sequence_id)

    Attributes:
        queries (List[ColabfoldSearchQuery]): List of search queries. Each query
            contains a protein sequence and optional identifier. After validation,
            always a list of ColabfoldSearchQuery instances regardless of input format.

    Examples:
        >>> # Simple format - just sequences
        >>> inputs = ColabfoldSearchInput(
        ...     queries=["MVLSPADKTN", "ACDEFGHIKL"]
        ... )
        >>>
        >>> # Explicit format with IDs
        >>> query1 = ColabfoldSearchQuery(
        ...     sequence="MVLSPADKTN",
        ...     sequence_id="protein_A"
        ... )
        >>> inputs = ColabfoldSearchInput(queries=[query1])
    """

    queries: List[ColabfoldSearchQuery] = Field(
        description="List of protein sequences to search for homologs"
    )

    @field_validator("queries", mode="before")
    @classmethod
    def normalize_queries(cls, value):
        """Normalize various input formats to List[ColabfoldSearchQuery]."""

        # If single instance, immediately convert to list
        if not isinstance(value, list):
            value = [value]

        if len(value) == 0:
            raise ValueError("At least one query sequence is required")

        # Validate each query
        validated_queries = []
        for query in value:
            if isinstance(query, str):
                query = ColabfoldSearchQuery(sequence=query)
            elif isinstance(query, tuple):
                query = ColabfoldSearchQuery(sequence=query[0], sequence_id=query[1])
            elif not isinstance(query, ColabfoldSearchQuery):
                raise ValueError(
                    f"Invalid query input: {query}. Must be a string, tuple, or ColabfoldSearchQuery instance."
                )
            validated_queries.append(query)

        return validated_queries

    @model_validator(mode="after")
    def populate_sequence_ids(self):
        """Auto-generate sequence IDs if not provided."""
        # Auto-generate sequence IDs from hash of sequence
        for query in self.queries:
            if query.sequence_id is None:
                query.sequence_id = (
                    "seq_" + hashlib.sha256(query.sequence.encode()).hexdigest()[:10]
                )

        # Ensure that no two queries have the same sequence ID
        seen_ids = set()
        for query in self.queries:
            if query.sequence_id in seen_ids:
                raise ValueError(
                    f"Sequence ID {query.sequence_id} is not unique. Please provide a unique sequence ID for each query."
                )
            seen_ids.add(query.sequence_id)

        return self

    def __len__(self) -> int:
        """Get the number of queries."""
        return len(self.queries)

    def __getitem__(self, index: int) -> ColabfoldSearchQuery:
        """Get a query by index."""
        return self.queries[index]

    def __iter__(self) -> Iterator[ColabfoldSearchQuery]:
        """Iterate over the queries."""
        return iter(self.queries)


# Output:


class ColabfoldSearchResult(BaseModel):
    """Result from searching a single protein sequence.

    Attributes:
        msa (Optional[MSA]): The Multiple Sequence Alignment object containing the homologous sequences.
            None if no homologs were found (only the query sequence would be present).
        sequence_id (str): Identifier for the sequence that was searched.
    """

    msa: Optional[MSA] = Field(
        description="Multiple Sequence Alignment containing homologous sequences, or None if no homologs found"
    )
    sequence_id: str = Field(description="Identifier for the searched sequence")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def num_homologs_found(self) -> int:
        """Number of homologous sequences found in the MSA (count excludes the query sequence
        which will always be the first sequence in the MSA). Returns 0 if msa is None.
        """
        if self.msa is None:
            return 0
        return len(self.msa) - 1


class ColabfoldSearchOutput(BaseToolOutput):
    """Output from ColabFold MSA search.

    This class encapsulates the results of MSA searches for one or more
    input sequences.

    Attributes:
        results (List[ColabfoldSearchResult]): List of search results, one per
            input query. Each result contains the path to the generated A3M file
            and metadata. The order matches the input queries order.
    """

    results: List[ColabfoldSearchResult] = Field(
        description="List of MSA search results"
    )

    @property
    def output_format_options(self) -> List[str]:
        return ["a3m", "fasta"]

    @property
    def output_format_default(self) -> str:
        return "a3m"

    def _export_output(self, export_path: Union[Path, str], file_format: str):
        if file_format not in ["a3m", "fasta"]:
            raise ValueError(f"Unsupported format: {file_format}")

        path = Path(export_path)
        path.mkdir(parents=True, exist_ok=True)
        for res in self.results:
            if res.msa:
                out_file = path / f"{res.sequence_id}.{file_format}"
                if file_format == "a3m":
                    res.msa.to_a3m_file(str(out_file))
                elif file_format == "fasta":
                    res.msa.to_fasta_file(str(out_file))

    def __len__(self) -> int:
        """Get the number of results."""
        return len(self.results)

    def __getitem__(self, index: int) -> ColabfoldSearchResult:
        """Get a result by index."""
        return self.results[index]

    def __iter__(self) -> Iterator[ColabfoldSearchResult]:
        """Iterate over the results."""
        return iter(self.results)


# Config:


class ColabfoldSearchConfig(BaseConfig):
    """Configuration object for ColabFold MSA search.

    This class defines all configuration parameters for running ColabFold's
    MSA search using local MMSeqs2 against a local sequence database or the
    online ColabFold API.

    Attributes:
        search_mode (Literal["local", "remote"]): Mode to use for MSA search.
            Options: "local" (uses local MMSeqs2 database search) or "remote" (uses
            the ColabFold online API). Default: "remote".

        use_metagenomic_db (bool): Whether to include metagenomic/environmental databases
            in the search. Metagenomic databases can improve MSA quality for
            some sequences but increase search time. Supported in both local and remote modes.
            Default: False.

        output_dir (Optional[str]): Directory where output MSA files will be saved.
            A subdirectory named 'msas' will be created to store A3M format
            alignment files. Each sequence will get its own A3M file named
            by its sequence ID. If None, uses the default cache directory
            (~/.cache/bio-programming/colabfold_search). Default: None.

        sensitivity (Optional[float]): Only used if search_mode is "local". MMseqs2 sensitivity
            parameter (1.0-9.0). Higher values increase sensitivity and may find more remote homologs,
            but significantly slow down the search. Lower values are faster but
            may miss distant homologs. If None, uses the default sensitivity (~8). Default: None.

        msa_db_dir (Optional[str]): Only used if search_mode is "local". Path to the local ColabFold/MMSeqs2 database directory.
            This directory should contain the database files downloaded using the
            setup_databases.sh script. To download databases, run: ./setup_databases.sh
            Default: None (uses built-in databases directory).

        database_name (str): Only used if search_mode is "local". Name of the database to use.
            If not provided, the tool will automatically detect the available databases and use one.
            Default: "uniref30_2302_db".

        num_threads (Optional[int]): Only used if search_mode is "local". Number of CPU threads to use for parallel
            processing. If None, automatically detects and uses all available
            CPU cores. Must be at least 1 if specified. Default: None (auto-detect).

        use_gpu (bool): Only used if search_mode is "local". Whether to enable GPU-accelerated search using MMseqs2-GPU.
            Requires GPU databases to be set up using 'GPU=1 ./setup_databases.sh'.
            When enabled, uses all available GPUs for search. Default: False.
            TODO: This is currently not working due to issue with GPU flag in local_msa_search.py

    """

    search_mode: Literal["local", "remote"] = ConfigField(
        title="Search Mode",
        default="remote",
        description="Mode to use for MSA search.",
        hidden=True,
    )
    use_metagenomic_db: bool = ConfigField(
        title="Use Metagenomic Database",
        default=False,
        description="Whether to include metagenomic database in search",
        advanced=True,
    )
    output_dir: Optional[str] = ConfigField(
        title="Output Directory",
        default=None,
        description="Directory for output MSA files (default: ~/.cache/bio-programming/colabfold_search)",
        hidden=True,
    )
    msa_db_dir: str = ConfigField(
        title="MSA Database Directory",
        default=CHIMERA_COLABFOLD_DB_LOCATION,
        description="Path to local ColabFold/MMSeqs2 database directory (default: ./databases in tool directory)",
    )
    database_name: str = ConfigField(
        title="Database Name",
        default="uniref30_2302_db",
        description="Name of the database to use.",
        advanced=True,
    )
    sensitivity: Optional[float] = ConfigField(
        title="MMseqs2 Sensitivity",
        default=None,
        ge=1.0,
        le=9.0,
        description="MMseqs2 parameter. Higher means more hits but slower search. Default matches ColabFold server.",
        advanced=True,
    )
    num_threads: Optional[int] = ConfigField(
        title="Number of Threads",
        default=None,
        ge=1,
        description="Number of CPU threads (None for auto-detect)",
        hidden=True,
    )
    # TODO: Local GPU search is not currently supported
    # use_gpu: bool = ConfigField(
    #     title="Use GPU Acceleration",
    #     default=False,
    #     description="Enable GPU-accelerated search using MMseqs2-GPU (requires GPU databases to be set up with GPU=1)",
    #     hidden=True,
    # )
    # Private field to track if user specified custom db_dir
    _user_specified_db_dir: bool = False
    # Private field to track if user specified custom output_dir
    _user_specified_output_dir: bool = False

    @model_validator(mode="after")
    def set_default_msa_db_dir(self):
        """Set default database directory if not provided and track user specification."""
        if self.msa_db_dir is None:
            self.msa_db_dir = str(DEFAULT_DB_DIR)
            self._user_specified_db_dir = False
        else:
            self._user_specified_db_dir = True
        return self

    @model_validator(mode="after")
    def validate_local_database_dir_and_name(self):
        """If the msa_db_dir is specified, ensure the folder exists."""

        # This check should only run for local search mode
        if self.search_mode != "local":
            return self

        if self.msa_db_dir is None:
            return self

        # Ensure the specified directory exists
        if not Path(self.msa_db_dir).exists():
            raise ValueError(f"msa_db_dir does not exist: {self.msa_db_dir}")
        if not Path(self.msa_db_dir).is_dir():
            raise ValueError(
                f"msa_db_dir exists but is not a directory: {self.msa_db_dir}"
            )

        # Ensure the specified database name is available
        available_databases = detect_available_local_databases(self.msa_db_dir)
        if self.database_name not in available_databases:
            raise ValueError(
                f"database_name does not exist: {self.database_name}. Available databases: {available_databases}"
            )

        return self

    @model_validator(mode="after")
    def set_default_output_dir(self):
        """Set default output directory if not provided and track user specification."""
        if self.output_dir is None:
            self.output_dir = str(DEFAULT_OUTPUT_DIR)
            self._user_specified_output_dir = False
        else:
            self._user_specified_output_dir = True
        return self


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input():
    """Minimal valid input for testing and examples."""
    return ColabfoldSearchInput(queries=["MKTL"])


@tool(
    key="colabfold-search",
    label="ColabFold MSA Search",
    category="sequence_alignment",
    input_class=ColabfoldSearchInput,
    config_class=ColabfoldSearchConfig,
    output_class=ColabfoldSearchOutput,
    description="Generate Multiple Sequence Alignments using ColabFold local database search",
    example_input=example_input,
    iterable_input_field="queries",
    iterable_output_field="results",
    cacheable=True,
)
def run_colabfold_search(
    inputs: ColabfoldSearchInput, config: ColabfoldSearchConfig | None = None,
    instance=None,
) -> ColabfoldSearchOutput:
    """Generate MSAs for protein sequences using ColabFold search, with options
    for online and local execution.

    Local Execution:
    Searches a local MMSeqs2 database to find homologous sequences and generates
    Multiple Sequence Alignments (MSAs).

    Online Execution:
    Uses the ColabFold online API to search for homologous sequences and generates
    Multiple Sequence Alignments (MSAs). Rate limited.

    Args:
        inputs (ColabfoldSearchInput): Validated input containing sequences to search.
        config (ColabfoldSearchConfig): Configuration with database path and search parameters.

    Returns:
        ColabfoldSearchOutput: List of results containing MSA objects.

    Raises:
        RuntimeError: If colabfold_search command execution fails.
        FileNotFoundError: If colabfold_search is not installed.
        ValueError: If msa_db_dir does not exist.

    Examples:
        >>> inputs = ColabfoldSearchInput(
        ...     queries=["MVLSPADKTN", "MKTAYIAKQR"]
        ... )
        >>> config = ColabfoldSearchConfig(msa_db_dir="/path/to/db")
        >>> output = run_colabfold_search(inputs, config)
        >>> # Access the MSA object directly
        >>> msa = output.results[0].msa
        >>> print(f"Found {msa.num_sequences} homologous sequences")
        >>> # Iterate through aligned sequences
        >>> for seq in msa:
        ...     print(seq)
    """

    if not inputs.queries:
        return ColabfoldSearchOutput(results=[])

    # Cleanup leftover files from previous runs if using default directory and cache is empty
    _cleanup_default_output_dir_if_cache_empty(config)

    # Extract sequences and IDs from queries
    sequences = [query.sequence for query in inputs.queries]
    sequence_ids = [query.sequence_id for query in inputs.queries]

    # Create output directory structure
    os.makedirs(config.output_dir, exist_ok=True)
    msa_out_dir = os.path.join(config.output_dir, "msas")
    os.makedirs(msa_out_dir, exist_ok=True)

    if config.search_mode == "local":
        return _local_search(sequences, sequence_ids, config, msa_out_dir, instance=instance)
    elif config.search_mode == "remote":
        return _remote_search(sequences, sequence_ids, config, msa_out_dir, instance=instance)
    else:
        raise ValueError(f"Invalid search mode: {config.search_mode}")


# ============================================================================
# Helper Functions
# ============================================================================


def _cleanup_default_output_dir_if_cache_empty(
    config: ColabfoldSearchConfig,
) -> None:
    """
    Clean up default output directory if cache is empty.

    This function removes leftover alignment files from previous runs to avoid
    the accumulation of unused alignment files in the default cache directory.

    Args:
        config: ColabFold search configuration

    Notes:
        - Only cleans up if output_dir was not user-specified (using default cache directory ~/.cache/bio-programming/colabfold_search)
        - Only cleans up if the tool cache is empty (no cached entries)
        - Preserves user-specified directories and cached files
    """
    # Only cleanup default directories, never user-specified ones
    if config._user_specified_output_dir:
        return

    from bio_programming_tools.utils.tool_cache import has_cached_entries

    # Only cleanup if cache is empty (no entries to preserve)
    if has_cached_entries("colabfold-search"):
        return

    # Safe to clean up: using default dir and cache is empty
    if os.path.exists(config.output_dir):
        shutil.rmtree(config.output_dir, ignore_errors=True)
        if config.verbose:
            logger.info(
                f"Cleaned up default output directory (cache is empty): {config.output_dir}"
            )


def _count_sequences_in_a3m(a3m_path: str | Path) -> int:
    """Count the number of sequences in an A3M file.

    Args:
        a3m_path: Path to the A3M file

    Returns:
        Number of sequences in the file
    """
    count = 0
    with open(a3m_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count


def _replace_query_header_in_a3m(a3m_path: str | Path, seq_id: str) -> None:
    """Replace the first header line in an A3M file with the real sequence ID.

    Since we use numeric indices in the query FASTA for predictable output
    filenames, this restores the original sequence ID in the A3M content.
    """
    with open(a3m_path, "r") as f:
        lines = f.readlines()
    if lines and lines[0].startswith(">"):
        lines[0] = f">{seq_id}\n"
        with open(a3m_path, "w") as f:
            f.writelines(lines)


def detect_available_local_databases(
    msa_db_dir: str | Path, verbose: bool = False
) -> list[str]:
    """
    Detect and list all available databases in the MSA database directory.

    This function scans the database directory for ColabFold/MMSeqs2 database files
    and returns a list of the available databases names.

    Args:
        msa_db_dir: Path to the ColabFold/MMSeqs2 database directory
        verbose: Whether to print detection information

    Returns:
        List of database names (without .dbtype extension) found in the directory.
    """
    msa_db_dir = Path(msa_db_dir)

    if not msa_db_dir.exists():
        if verbose:
            logger.warning(f"Database directory does not exist: {msa_db_dir}")
        return []

    # Look for main database files ending in _db.dbtype
    # These are the primary database files that can be used with colabfold_search
    dbtype_files = list(msa_db_dir.glob("*_db.dbtype"))

    if not dbtype_files:
        # Fallback: look for any .dbtype file
        all_dbtype = list(msa_db_dir.glob("*.dbtype"))
        if all_dbtype:
            # Filter out auxiliary files (those with suffixes like _seq, _aln, _h)
            dbtype_files = [
                f
                for f in all_dbtype
                if not any(
                    f.stem.endswith(suffix)
                    for suffix in ["_seq", "_aln", "_h", "_seq_h"]
                )
            ]

    if not dbtype_files:
        if verbose:
            logger.warning(f"No database files found in {msa_db_dir}")
        return []

    # Extract database names (remove .dbtype extension) and sort
    db_names = sorted([f.stem for f in dbtype_files])

    if verbose:
        logger.info(f"Detected {len(db_names)} database(s) in {msa_db_dir}:")
        for db_name in db_names:
            logger.info(f"  - {db_name}")

    return db_names


def _local_search(
    sequences: List[str],
    sequence_ids: List[str],
    config: ColabfoldSearchConfig,
    msa_out_dir: str,
    instance=None,
) -> ColabfoldSearchOutput:
    """
    Performs local search for homologous sequences and generates Multiple Sequence Alignments (MSAs).
    """
    logger.debug(f"Generating local MSAs for {len(sequences)} sequence(s)...")

    # Use ToolInstance to run colabfold_search in isolated environment
    from bio_programming_tools.utils.tool_instance import ToolInstance

    # Get the standalone script path
    standalone_script = Path(__file__).parent / "standalone" / "local_msa_search.py"

    # Create temporary FASTA file for colabfold_search
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "query.fasta")
        with open(fasta_path, "w") as f:
            for idx, seq in enumerate(sequences):
                # Use numeric IDs to ensure colabfold_search outputs
                # predictably named files (0.a3m, 1.a3m, etc.)
                f.write(f">{idx}\n{seq}\n")

        # Prepare input data for standalone script
        num_threads = config.num_threads
        if num_threads is None:
            num_threads = len(os.sched_getaffinity(0))

        input_data = {
            "query_fasta_path": str(fasta_path),
            "msa_db_dir": str(config.msa_db_dir),
            "output_dir": str(msa_out_dir),
            "num_threads": num_threads,
            "use_metagenomic_db": config.use_metagenomic_db,
            "sensitivity": config.sensitivity,
            "database_name": config.database_name,
            "use_gpu": False,
            "verbose": config.verbose,
        }

        # Execute colabfold_search via standalone script
        input_data["device"] = "cpu"
        output_data = ToolInstance.dispatch(
            "colabfold_search",
            input_data,
            instance=instance,
            script_path=standalone_script,
            config=config,
        )

        if not output_data.get("success", False):
            error_msg = output_data.get("error", "Unknown error")
            raise RuntimeError(f"colabfold_search failed: {error_msg}")

    # Process results: colabfold_search outputs files as 0.a3m, 1.a3m, etc.
    results = []
    for idx, seq_id in enumerate(sequence_ids):

        # Collect and convert the A3M file to an MSA object
        numbered_a3m = os.path.join(msa_out_dir, f"{idx}.a3m")
        named_a3m = os.path.join(msa_out_dir, f"{seq_id}.a3m")

        # Rename the numbered file to use the sequence ID for better user experience
        if os.path.exists(numbered_a3m):
            # Restore the real sequence ID in the query header (first line)
            # since we used numeric indices in the FASTA input
            _replace_query_header_in_a3m(numbered_a3m, seq_id)
            shutil.move(numbered_a3m, named_a3m)

        # Check if the A3M file contains at least 2 sequences (query + at least one homolog)
        # If only the query sequence is present, return None instead of creating an MSA
        num_sequences = _count_sequences_in_a3m(named_a3m)

        if num_sequences < 2:
            # No homologs found, only the query sequence
            msa = None
        else:
            msa = MSA(aligned_sequences_or_filepath=named_a3m)

        results.append(
            ColabfoldSearchResult(
                msa=msa,
                sequence_id=seq_id,
            )
        )

    return ColabfoldSearchOutput(results=results)


def _remote_search(
    sequences: List[str],
    sequence_ids: List[str],
    config: ColabfoldSearchConfig,
    msa_out_dir: str,
    instance=None,
) -> ColabfoldSearchOutput:
    """
    Performs remote search for homologous sequences and generates Multiple Sequence Alignments (MSAs).
    """
    logger.debug(f"Generating remote MSAs for {len(sequences)} sequence(s)...")

    # Use ToolInstance to run remote search in isolated environment
    from bio_programming_tools.utils.tool_instance import ToolInstance

    # Get the standalone script path
    standalone_script = Path(__file__).parent / "standalone" / "remote_msa_search.py"

    # Prepare input data for standalone script
    input_data = {
        "sequences": sequences,
        "sequence_ids": sequence_ids,
        "output_dir": str(config.output_dir),
        "use_metagenomic_db": config.use_metagenomic_db,
        "verbose": config.verbose,
    }

    # Execute remote search via standalone script
    input_data["device"] = "cpu"
    output_data = ToolInstance.dispatch(
        "colabfold_search",
        input_data,
        instance=instance,
        script_path=standalone_script,
        config=config,
    )

    if not output_data.get("success", False):
        # Check if we have partial results
        num_successful = output_data.get("num_successful", 0)
        num_failed = output_data.get("num_failed", 0)

        if num_successful == 0:
            # Total failure
            error_msg = "Remote MSA search failed for all sequences"
            if "errors" in output_data:
                error_msg += f"\nErrors: {output_data['errors']}"
            raise RuntimeError(error_msg)
        else:
            # Partial failure - log warning but continue
            if config.verbose:
                logger.warning(
                    f"Remote MSA search partially failed: {num_successful} succeeded, {num_failed} failed"
                )
                if "errors" in output_data:
                    for seq_id, error in output_data["errors"].items():
                        logger.warning(f"  {seq_id}: {error}")

    # Process results
    msa_paths = output_data.get("msa_paths", {})
    results = []

    for seq_id in sequence_ids:
        if seq_id in msa_paths:
            msa_path = msa_paths[seq_id]

            # Check if the A3M file contains at least 2 sequences (query + at least one homolog)
            num_sequences = _count_sequences_in_a3m(msa_path)

            if num_sequences < 2:
                # No homologs found, only the query sequence
                msa = None
            else:
                msa = MSA(aligned_sequences_or_filepath=msa_path)

            results.append(
                ColabfoldSearchResult(
                    msa=msa,
                    sequence_id=seq_id,
                )
            )
        else:
            # No MSA generated for this sequence
            if config.verbose:
                logger.warning(f"No MSA generated for sequence {seq_id}")
            results.append(
                ColabfoldSearchResult(
                    msa=None,
                    sequence_id=seq_id,
                )
            )

    return ColabfoldSearchOutput(results=results)
