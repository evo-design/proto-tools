"""proto_tools/tools/sequence_alignment/colabfold_search/colabfold_search.py.

This module provides a standardized interface for generating Multiple Sequence
Alignments (MSAs) using ColabFold's local database search with MMSeqs2.
"""

import hashlib
import logging
import os
import platform
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from proto_tools.tools.sequence_alignment.msas import MSA
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
    has_cached_entries,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


# Default cache directory for MSA files, derived from PROTO_HOME
def _default_output_dir() -> Path:
    from proto_tools.utils.proto_home import get_proto_home

    return get_proto_home() / "colabfold_search"


# Default database directory: registry-driven, resolves to
# `$PROTO_MODEL_CACHE/databases/uniref30_2302/`. Kept under PROTO_MODEL_CACHE
# (not $PROTO_HOME/colabfold_search/) so it survives `_default_output_dir`
# cleanup, and uses a dataset-scoped subdir (not tool-scoped) so the future
# `mmseqs2-homology-search` tool reads the same files without redownloading.
# See proto_tools/databases/.
def _default_msa_db_dir() -> Path:
    from proto_tools.databases import get_dataset_dir

    return get_dataset_dir("uniref30-2302")


# Input:


class ColabfoldSearchQuery(BaseModel):
    """Represents a single protein sequence to search for homologs.

    This class defines a query for MSA generation. Each query consists of a
    protein sequence and an optional identifier.

    Attributes:
        sequence (str): Protein sequence to search for homologs. Must be a
            non-empty string containing amino acid characters.
        sequence_id (str | None): Optional identifier for this sequence.
            Used for output file naming and result tracking. If not provided,
            will be auto-generated as seq_0, seq_1, etc.
    """

    sequence: str = Field(description="Protein sequence to search for homologs")
    sequence_id: str | None = Field(
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
    - List of dicts or single dict of the form ``{"sequence": str, "sequence_id": str | None}``
      (the shape produced by ``model_dump`` when the input is round-tripped through JSON)

    Attributes:
        queries (list[ColabfoldSearchQuery]): List of search queries. Each query
            contains a protein sequence and optional identifier. After validation,
            always a list of ColabfoldSearchQuery instances regardless of input format.

    Examples:
        >>> # Simple format - just sequences
        >>> inputs = ColabfoldSearchInput(queries=["MVLSPADKTN", "ACDEFGHIKL"])
        >>>
        >>> # Explicit format with IDs
        >>> query1 = ColabfoldSearchQuery(sequence="MVLSPADKTN", sequence_id="protein_A")
        >>> inputs = ColabfoldSearchInput(queries=[query1])
    """

    queries: list[ColabfoldSearchQuery] = InputField(description="List of protein sequences to search for homologs")

    @field_validator("queries", mode="before")
    @classmethod
    def normalize_queries(cls, value: Any) -> Any:
        """Normalize various input formats to List[ColabfoldSearchQuery]."""
        # If single instance, immediately convert to list
        if not isinstance(value, list):
            value = [value]

        if len(value) == 0:
            raise ValueError("At least one query sequence is required")

        # Validate each query
        validated_queries = []
        for raw_query in value:
            if isinstance(raw_query, str):
                query = ColabfoldSearchQuery(sequence=raw_query)
            elif isinstance(raw_query, tuple):
                query = ColabfoldSearchQuery(sequence=raw_query[0], sequence_id=raw_query[1])
            elif isinstance(raw_query, dict):
                # JSON round-trip case: model_dump serializes ColabfoldSearchQuery
                # as {"sequence": str, "sequence_id": str | None}.
                query = ColabfoldSearchQuery(**raw_query)
            elif isinstance(raw_query, ColabfoldSearchQuery):
                query = raw_query
            else:
                raise ValueError(
                    f"Invalid query input: {raw_query}. Must be a string, tuple, dict, or ColabfoldSearchQuery instance."
                )
            validated_queries.append(query)

        return validated_queries

    @model_validator(mode="after")
    def populate_sequence_ids(self) -> Any:
        """Auto-generate sequence IDs if not provided."""
        # Auto-generate sequence IDs from hash of sequence
        for query in self.queries:
            if query.sequence_id is None:
                query.sequence_id = "seq_" + hashlib.sha256(query.sequence.encode()).hexdigest()[:10]

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

    def __iter__(self) -> Iterator[ColabfoldSearchQuery]:  # type: ignore[override]
        """Iterate over the queries."""
        return iter(self.queries)


# Output:


class ColabfoldSearchResult(BaseModel):
    """Result from searching a single protein sequence.

    Attributes:
        msa (MSA | None): The Multiple Sequence Alignment object containing the homologous sequences.
            None if no homologs were found (only the query sequence would be present).
        sequence_id (str): Identifier for the sequence that was searched.
    """

    msa: MSA | None = Field(
        description="Multiple Sequence Alignment containing homologous sequences, or None if no homologs found"
    )
    sequence_id: str = Field(description="Identifier for the searched sequence")

    @property
    def num_homologs_found(self) -> int:
        """Number of homologous sequences found in the MSA (count excludes the query sequence.

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
        results (list[ColabfoldSearchResult]): List of search results, one per
            input query. Each result contains the path to the generated A3M file
            and metadata. The order matches the input queries order.
    """

    results: list[ColabfoldSearchResult] = Field(description="List of MSA search results")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["a3m", "fasta"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "a3m"

    def _export_output(self, export_path: Path | str, file_format: str) -> None:
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

    def __iter__(self) -> Iterator[ColabfoldSearchResult]:  # type: ignore[override]
        """Iterate over the results."""
        return iter(self.results)


# Config:


class ColabfoldSearchConfig(BaseConfig):
    """Configuration object for ColabFold MSA search.

    Defines all configuration parameters for running MSA search either against
    a local MMseqs2 database or the online ColabFold API.

    Attributes:
        search_mode (Literal['local', 'remote']): ``"local"`` runs MMseqs2
            against a downloaded DB; ``"remote"`` queries ColabFold's MSA API.
        use_metagenomic_db (bool): Include metagenomic/environmental DBs
            (ColabFoldDB envdb / SPIRE) in the search. Off for speed; upstream
            colabfold defaults this on (``--use-env=1``). Supported in both
            local and remote modes.
        output_dir (str | None): Directory where output MSA files are saved.
            An ``msas`` subdirectory is created to store A3M files, one per
            sequence ID. ``None`` resolves to ``$PROTO_HOME/colabfold_search``.
        sensitivity (float | None): MMseqs2 ``-s`` override (1.0-9.0). Local
            mode only. Ignored under ``use_gpu=True`` (colabfold_search forces
            ungapped prefilter and drops ``-s``). When ``None`` on CPU, falls
            back to colabfold's k-score path (matches the public MSA server).
        msa_db_dir (str | None): Local mode only. Path to the MMseqs2 database
            directory provisioned by ``setup_databases.sh``. ``None`` resolves
            to ``$PROTO_MODEL_CACHE/databases/uniref30_2302/``. Deliberately
            kept *outside* ``output_dir`` so the run-time cleanup in
            ``_cleanup_default_output_dir_if_cache_empty`` cannot delete it.
        database_name (str): Local mode only. MMseqs2 DB stem within
            ``msa_db_dir`` (matches the ``*.dbtype`` file).
        num_threads (int | None): Local mode only. CPU threads for parallel
            search. ``None`` auto-detects all available cores.
        use_gpu (bool): Local mode only. Run MMseqs2-GPU; requires an NVIDIA
            GPU (Turing+), a Linux host, and a ``*.idx_pad`` GPU index built
            via ``mmseqs makepaddedseqdb``. Validators raise ``ValueError`` if
            set with ``search_mode="remote"``, on non-Linux platforms, or
            without the padded DB on disk.
        extra_args (list[str]): Local mode only. Verbatim ``colabfold_search``
            CLI tokens appended after the typed flags (e.g.
            ``["--max-accept", "500"]``). Power-user escape hatch for flags
            not exposed as typed fields above.
        timeout (int | None): Subprocess timeout in seconds. Full database searches
            can take more than 10 minutes. ``None`` waits indefinitely.
    """

    search_mode: Literal["local", "remote"] = ConfigField(
        title="Search Mode",
        default="remote",
        description="`remote` queries ColabFold's MSA API; `local` runs MMseqs2 against a downloaded DB.",
    )
    use_metagenomic_db: bool = ConfigField(
        title="Use Metagenomic Database",
        default=False,
        description="Include metagenomic DBs (envdb/SPIRE). Off by default for speed (upstream = on).",
    )
    output_dir: str | None = ConfigField(
        title="Output Directory",
        default=None,
        description="Directory for output MSA files; resolves to a `$PROTO_HOME/colabfold_search` subdir when None.",
        include_in_key=False,
    )
    msa_db_dir: str | None = ConfigField(
        title="MSA Database Directory",
        default=None,
        description="Local MMseqs2 database directory; resolves to the registry-provisioned location when None.",
    )
    database_name: str = ConfigField(
        title="Database Name",
        default="uniref30_2302_db",
        description="MMseqs2 DB stem within `msa_db_dir` (matches the `*.dbtype` file).",
    )
    sensitivity: float | None = ConfigField(
        title="MMseqs2 Sensitivity",
        default=None,
        ge=1.0,
        le=9.0,
        description="MMseqs2 `-s` (1.0-9.0); ignored on GPU; `None` falls back to colabfold's k-score path.",
    )
    num_threads: int | None = ConfigField(
        title="Number of Threads",
        default=None,
        ge=1,
        description="CPU threads; `None` auto-detects all available cores.",
        include_in_key=False,
    )
    use_gpu: bool = ConfigField(
        title="Use GPU",
        default=False,
        description="GPU-accelerated MMseqs2; requires an NVIDIA GPU (Turing+) on Linux and a GPU-padded DB.",
    )
    extra_args: list[str] = ConfigField(
        title="Extra CLI Arguments",
        default=[],
        description="Verbatim `colabfold_search` CLI tokens for niche flags (e.g. `['--max-accept', '500']`).",
    )
    timeout: int | None = ConfigField(
        title="Timeout",
        default=3600,
        ge=1,
        description="Subprocess timeout in seconds; full-database searches can exceed 10 minutes.",
        include_in_key=False,
    )
    # Private field to track if user specified custom db_dir
    _user_specified_db_dir: bool = False
    # Private field to track if user specified custom output_dir
    _user_specified_output_dir: bool = False

    @model_validator(mode="after")
    def set_default_msa_db_dir(self) -> Any:
        """Resolve `msa_db_dir` default to $PROTO_MODEL_CACHE/databases/uniref30_2302/ when not user-specified.

        `_user_specified_db_dir` reflects whether the resolved value differs
        from the registry default; it can't reliably distinguish None (unset)
        from an explicit pass of the same default path, but that doesn't matter
        for the error-message hint logic below.
        """
        default_resolved = str(_default_msa_db_dir())
        if self.msa_db_dir is None:
            self.msa_db_dir = default_resolved
            self._user_specified_db_dir = False
        else:
            self._user_specified_db_dir = self.msa_db_dir != default_resolved
        return self

    @model_validator(mode="after")
    def validate_local_database_dir_and_name(self) -> Any:
        """If the msa_db_dir is specified, ensure the folder exists."""
        # This check should only run for local search mode
        if self.search_mode != "local":
            return self

        # `set_default_msa_db_dir` (declared above) always resolves None → default path.
        assert self.msa_db_dir is not None

        # Ensure the specified directory exists
        if not Path(self.msa_db_dir).exists():
            hint = (
                " Provision by running "
                "`proto_tools/tools/sequence_alignment/colabfold_search/setup_databases.sh "
                f"{self.msa_db_dir}` "
                "or set `msa_db_dir` to a pre-built database location."
                if not self._user_specified_db_dir
                else ""
            )
            raise ValueError(f"msa_db_dir does not exist: {self.msa_db_dir}.{hint}")
        if not Path(self.msa_db_dir).is_dir():
            raise ValueError(f"msa_db_dir exists but is not a directory: {self.msa_db_dir}")

        # Ensure the specified database name is available
        available_databases = detect_available_local_databases(self.msa_db_dir)
        if self.database_name not in available_databases:
            raise ValueError(
                f"database_name does not exist: {self.database_name}. Available databases: {available_databases}"
            )

        return self

    @model_validator(mode="after")
    def set_default_output_dir(self) -> Any:
        """Set default output directory if not provided and track user specification."""
        if self.output_dir is None:
            self.output_dir = str(_default_output_dir())
            self._user_specified_output_dir = False
        else:
            self._user_specified_output_dir = True
        return self

    @model_validator(mode="after")
    def validate_use_gpu_requires_local(self) -> Any:
        """Validate that use_gpu is only set with local search mode."""
        if self.use_gpu and self.search_mode != "local":
            raise ValueError(
                "use_gpu=True requires search_mode='local' (GPU acceleration is not available for remote search)"
            )
        return self

    @model_validator(mode="after")
    def validate_use_gpu_platform(self) -> Any:
        """Validate that use_gpu is only set on Linux where the GPU binary is available."""
        if self.use_gpu and platform.system() != "Linux":
            raise ValueError(
                f"use_gpu=True requires Linux (current platform: {platform.system()} {platform.machine()}). "
                "GPU-accelerated MMseqs2 is only available for Linux."
            )
        return self

    @model_validator(mode="after")
    def validate_use_gpu_database(self) -> Any:
        """Validate that the database has been formatted for GPU search."""
        if not self.use_gpu or self.search_mode != "local":
            return self
        assert self.msa_db_dir is not None  # resolved by `set_default_msa_db_dir`
        idx_pad = Path(self.msa_db_dir) / f"{self.database_name}.idx_pad"
        # Both the padded data file and its ``.dbtype`` must exist; without
        # the latter, ``easy-search`` falls through to its FASTA-input branch.
        if not idx_pad.exists() or not Path(f"{idx_pad}.dbtype").is_file():
            raise ValueError(
                f"use_gpu=True requires a GPU-formatted database, but a complete padded DB "
                f"({self.database_name}.idx_pad with sibling .dbtype) was not found in "
                f"{self.msa_db_dir}. Create it with: "
                f"mmseqs makepaddedseqdb {self.database_name} {self.database_name}.idx_pad"
            )
        return self

    @property
    def gpus_per_instance(self) -> int:
        """Number of GPUs the configured search uses.

        Returns 1 when ``use_gpu=True`` (MMseqs2-GPU is invoked with ``--gpu 1``),
        else 0 (CPU search). Override is required despite ``BaseConfig`` deriving
        the default from the ``device`` string: this tool always sets
        ``device='cpu'`` (mmseqs handles its own GPU detection internally), so the
        real GPU signal is the ``use_gpu`` flag, not the device field.
        """
        return 1 if self.use_gpu else 0

    def effective_timeout(self) -> int | None:
        """Drop the cap when ``use_metagenomic_db=True`` and the user did not set timeout (envdb/SPIRE searches run for hours)."""
        if self.use_metagenomic_db and "timeout" not in self.model_fields_set:
            return None
        return self.timeout


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ColabfoldSearchInput(queries=["MKTL"])  # type: ignore[list-item]


@tool(
    key="colabfold-search",
    label="ColabFold MSA Search",
    category="sequence_alignment",
    input_class=ColabfoldSearchInput,
    config_class=ColabfoldSearchConfig,
    output_class=ColabfoldSearchOutput,
    description="Generate Multiple Sequence Alignments via ColabFold (local MMseqs2 DB or remote API)",
    example_input=example_input,
    iterable_input_field="queries",
    iterable_output_field="results",
    cacheable=True,
)
def run_colabfold_search(
    inputs: ColabfoldSearchInput,
    config: ColabfoldSearchConfig,
    instance: Any = None,
) -> ColabfoldSearchOutput:
    """Generate MSAs for protein sequences using ColabFold search, with options.

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

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ColabfoldSearchOutput: List of results containing MSA objects.

    Raises:
        RuntimeError: If colabfold_search command execution fails.
        FileNotFoundError: If colabfold_search is not installed.
        ValueError: If msa_db_dir does not exist.

    Examples:
        >>> inputs = ColabfoldSearchInput(queries=["MVLSPADKTN", "MKTAYIAKQR"])
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
    os.makedirs(config.output_dir, exist_ok=True)  # type: ignore[arg-type]
    msa_out_dir = os.path.join(config.output_dir, "msas")  # type: ignore[arg-type]
    os.makedirs(msa_out_dir, exist_ok=True)

    if config.search_mode == "local":
        return _local_search(sequences, sequence_ids, config, msa_out_dir, instance=instance)  # type: ignore[arg-type]
    if config.search_mode == "remote":
        return _remote_search(sequences, sequence_ids, config, msa_out_dir, instance=instance)  # type: ignore[arg-type]
    raise ValueError(f"colabfold-search: invalid search_mode {config.search_mode!r}; expected 'local' or 'remote'")


# ============================================================================
# Helper Functions
# ============================================================================


def _cleanup_default_output_dir_if_cache_empty(
    config: ColabfoldSearchConfig,
) -> None:
    """Clean up default output directory if cache is empty.

    This function removes leftover alignment files from previous runs to avoid
    the accumulation of unused alignment files in the default cache directory.

    Args:
        config (ColabfoldSearchConfig): ColabFold search configuration

    Notes:
        - Only cleans up if output_dir was not user-specified (using default cache directory $PROTO_HOME/colabfold_search)
        - Only cleans up if the tool cache is empty (no cached entries)
        - Preserves user-specified directories and cached files
    """
    # Only cleanup default directories, never user-specified ones
    if config._user_specified_output_dir:
        return

    # Only cleanup if cache is empty (no entries to preserve)
    if has_cached_entries("colabfold-search"):
        return

    # Safe to clean up: using default dir and cache is empty
    if os.path.exists(config.output_dir):  # type: ignore[arg-type]
        shutil.rmtree(config.output_dir, ignore_errors=True)  # type: ignore[arg-type]
        if config.verbose:
            logger.info(f"Cleaned up default output directory (cache is empty): {config.output_dir}")


def _count_sequences_in_a3m(a3m_path: str | Path) -> int:
    """Count the number of sequences in an A3M file.

    Args:
        a3m_path (str | Path): Path to the A3M file

    Returns:
        int: Number of sequences in the file
    """
    count = 0
    with open(a3m_path) as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count


def _replace_query_header_in_a3m(a3m_path: str | Path, seq_id: str) -> None:
    """Replace the first header line in an A3M file with the real sequence ID.

    Since we use numeric indices in the query FASTA for predictable output
    filenames, this restores the original sequence ID in the A3M content.

    Args:
        a3m_path (str | Path): Path to the A3M file to modify.
        seq_id (str): Sequence identifier to set as the query header.
    """
    with open(a3m_path) as f:
        lines = f.readlines()
    if lines and lines[0].startswith(">"):
        lines[0] = f">{seq_id}\n"
        with open(a3m_path, "w") as f:
            f.writelines(lines)


def detect_available_local_databases(msa_db_dir: str | Path, verbose: bool = False) -> list[str]:
    """Detect and list all available databases in the MSA database directory.

    This function scans the database directory for ColabFold/MMSeqs2 database files
    and returns a list of the available databases names.

    Args:
        msa_db_dir (str | Path): Path to the ColabFold/MMSeqs2 database directory
        verbose (bool): Whether to print detection information

    Returns:
        list[str]: List of database names (without .dbtype extension) found in the directory.
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
                f for f in all_dbtype if not any(f.stem.endswith(suffix) for suffix in ["_seq", "_aln", "_h", "_seq_h"])
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
    sequences: list[str],
    sequence_ids: list[str],
    config: ColabfoldSearchConfig,
    msa_out_dir: str,
    instance: Any = None,
) -> ColabfoldSearchOutput:
    """Performs local search for homologous sequences and generates Multiple Sequence Alignments (MSAs)."""
    logger.debug(f"Generating local MSAs for {len(sequences)} sequence(s)...")

    # Use ToolInstance to run colabfold_search in isolated environment

    # Get the standalone script path
    standalone_script = Path(__file__).parent / "standalone" / "local_msa_search.py"

    # Create temporary FASTA file for colabfold_search
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "query.fasta")
        with open(fasta_path, "w") as f:
            f.writelines(f">{idx}\n{seq}\n" for idx, seq in enumerate(sequences))

        # Prepare input data for standalone script
        num_threads = config.num_threads
        if num_threads is None:
            num_threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1

        input_data = {
            "query_fasta_path": str(fasta_path),
            "msa_db_dir": str(config.msa_db_dir),
            "output_dir": str(msa_out_dir),
            "num_threads": num_threads,
            "use_metagenomic_db": config.use_metagenomic_db,
            "sensitivity": config.sensitivity,
            "database_name": config.database_name,
            "use_gpu": config.use_gpu,
            "extra_args": list(config.extra_args),
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

        msa = None if num_sequences < 2 else MSA.from_file(named_a3m)

        results.append(
            ColabfoldSearchResult(
                msa=msa,
                sequence_id=seq_id,
            )
        )

    return ColabfoldSearchOutput(results=results)


def _remote_search(
    sequences: list[str],
    sequence_ids: list[str],
    config: ColabfoldSearchConfig,
    msa_out_dir: str,  # noqa: ARG001 — required by tool interface
    instance: Any = None,
) -> ColabfoldSearchOutput:
    """Performs remote search for homologous sequences and generates Multiple Sequence Alignments (MSAs)."""
    logger.debug(f"Generating remote MSAs for {len(sequences)} sequence(s)...")

    # Use ToolInstance to run remote search in isolated environment

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
            error_msg = f"colabfold-search: remote MSA search failed for all {num_failed} sequence(s)"
            if "errors" in output_data:
                error_msg += f"; errors: {output_data['errors']}"
            raise RuntimeError(error_msg)
        # Partial failure - log warning but continue
        if config.verbose:
            logger.warning(f"Remote MSA search partially failed: {num_successful} succeeded, {num_failed} failed")
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

            msa = None if num_sequences < 2 else MSA.from_file(msa_path)

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
