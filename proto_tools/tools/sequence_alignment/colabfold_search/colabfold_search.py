"""proto_tools/tools/sequence_alignment/colabfold_search/colabfold_search.py.

This module provides a standardized interface for generating Multiple Sequence
Alignments (MSAs) using ColabFold's local database search with MMSeqs2.
"""

import logging
import os
import platform
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from proto_tools.entities.msa import MSA
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
    """One search unit; one or more chains searched together.

    ``sequences`` is always normalized to a ``list[str]``: a bare ``str`` is
    accepted and wrapped into a one-element list. A single chain is searched via
    the unpaired API endpoint; two or more chains are submitted in a single
    ``use_pairing=True`` call and the resulting per-chain MSAs are row-aligned
    across the group by taxonomy.

    Attributes:
        sequences (list[str]): The chain sequence(s) for this query. One chain is
            an unpaired search; two or more is one taxonomy-paired group. A bare
            ``str`` is accepted and normalized to ``[str]``.
    """

    sequences: list[str] = Field(
        title="Sequences",
        description="Chain sequence(s) for this query; one is unpaired, two or more is one paired group.",
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def validate_sequences(cls, v: Any) -> Any:
        """Normalize a bare ``str`` to ``[str]`` and validate non-empty, stripped chains."""
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            raise ValueError(f"sequences must be str or list[str], got {type(v).__name__}")
        if len(v) < 1:
            raise ValueError("sequences must contain at least one chain")
        cleaned: list[str] = []
        for item in v:
            if not isinstance(item, str):
                raise ValueError(f"sequences elements must be strings, got {type(item).__name__}")
            stripped = item.strip()
            if not stripped:
                raise ValueError("Sequence cannot be empty")
            cleaned.append(stripped)
        return cleaned

    @property
    def is_paired(self) -> bool:
        """True when this query represents a multi-chain paired group."""
        return len(self.sequences) > 1

    @property
    def chain_count(self) -> int:
        """Number of chains in this query (1 for unpaired, N for paired)."""
        return len(self.sequences)


class ColabfoldSearchInput(BaseToolInput):
    """Input object for ColabFold MSA search.

    A list of :class:`ColabfoldSearchQuery` units. Each unit is either:

    - A single sequence (unpaired): one independent search.
    - A list of sequences (paired): submitted together as one
      ``use_pairing=True`` API call; the resulting per-chain MSAs are
      row-aligned across the group by taxonomy.

    Supports multiple input shapes per outer-list element:

    - Sequence string → unpaired query
    - ``{"sequences": ...}`` dict (round-tripped JSON shape)
    - :class:`ColabfoldSearchQuery` instance (explicit, may be either kind)
    - **List of sequence strings** → one taxonomy-paired group

    Attributes:
        queries (list[ColabfoldSearchQuery]): Search queries in original input order.
            Each query is unpaired (one chain) or paired (``q.is_paired``, two or
            more chains). Results are returned parallel to this list.

    Examples:
        >>> # Three independent unpaired searches
        >>> ColabfoldSearchInput(queries=["MVLSPADKTN", "ACDEFGHIKL", "MKTAYIAKQR"])
        >>>
        >>> # One unpaired + one paired group of two heterocomplex chains
        >>> ColabfoldSearchInput(queries=["MVLSPADKTN", ["MKTAYIAKQR", "GSSGSSGSS"]])
        >>>
        >>> # Single paired group of three chains
        >>> ColabfoldSearchInput(queries=[["SEQ_A", "SEQ_B", "SEQ_C"]])
    """

    queries: list[ColabfoldSearchQuery] = InputField(
        title="Queries",
        description="List of search queries; each is one sequence (unpaired) or a list of sequences (paired group).",
    )

    @field_validator("queries", mode="before")
    @classmethod
    def normalize_queries(cls, value: Any) -> Any:
        """Normalize various input formats to a list of ColabfoldSearchQuery."""
        if not isinstance(value, list):
            value = [value]
        if len(value) == 0:
            raise ValueError("At least one query is required")

        return [_coerce_query(raw) for raw in value]

    def __len__(self) -> int:
        """Get the number of queries."""
        return len(self.queries)

    def __getitem__(self, index: int) -> ColabfoldSearchQuery:
        """Get a query by index."""
        return self.queries[index]

    def __iter__(self) -> Iterator[ColabfoldSearchQuery]:  # type: ignore[override]
        """Iterate over the queries."""
        return iter(self.queries)


def _coerce_query(raw: Any) -> ColabfoldSearchQuery:
    """Coerce a single outer-list element into a ColabfoldSearchQuery.

    Accepted shapes:
        - str → unpaired query
        - ``{"sequences": ...}`` dict (round-tripped JSON shape) → ColabfoldSearchQuery(**raw)
        - ColabfoldSearchQuery instance → passed through
        - list of strings → one paired query over those chains
    """
    if isinstance(raw, ColabfoldSearchQuery):
        return raw
    if isinstance(raw, str):
        return ColabfoldSearchQuery(sequences=[raw])
    if isinstance(raw, dict):
        return ColabfoldSearchQuery(**raw)
    if isinstance(raw, list):
        return ColabfoldSearchQuery(sequences=raw)
    raise ValueError(f"Invalid query input: {raw!r}. Must be a string, dict, list, or ColabfoldSearchQuery.")


# Output:


class ColabfoldSearchResult(BaseModel):
    """Result for one :class:`ColabfoldSearchQuery`, returned in the input query order.

    Both fields are always lists, parallel to each other and to the query's
    chains: index ``i`` is chain ``i``. An unpaired query has one element; a
    paired query has one per chain, row-aligned across chains by taxonomy.
    Supports ``len()``, indexing, and iteration over the per-chain MSAs.

    Attributes:
        query_sequences (list[str]): The query chain sequence(s) this result is for.
        msas (list[MSA | None]): One MSA per chain (``None`` for a chain with no
            homologs beyond the query row).
    """

    query_sequences: list[str] = Field(
        title="Query Sequences", description="The query chain sequence(s) this result is for."
    )
    msas: list[MSA | None] = Field(
        title="MSA Results",
        description="One MSA per chain (None when a chain has no homologs); row-aligned for paired queries.",
    )

    @property
    def num_homologs_found(self) -> int:
        """Homolog count of the first chain's MSA, excluding the query row.

        Paired chains share the same row count by construction, so the first
        chain is representative. Returns 0 when that chain has no MSA.
        """
        first = self.msas[0] if self.msas else None
        return len(first) - 1 if first is not None else 0

    def __len__(self) -> int:
        """Number of chains in this query."""
        return len(self.msas)

    def __getitem__(self, index: int) -> MSA | None:
        """Get a chain's MSA by index."""
        return self.msas[index]

    def __iter__(self) -> Iterator[MSA | None]:  # type: ignore[override]
        """Iterate over per-chain MSAs."""
        return iter(self.msas)


class ColabfoldSearchOutput(BaseToolOutput):
    """Output from ColabFold MSA search.

    This class encapsulates the results of MSA searches for one or more
    input sequences.

    Attributes:
        results (list[ColabfoldSearchResult]): List of search results, one per
            input query. Each result contains the path to the generated A3M file
            and metadata. The order matches the input queries order.
    """

    results: list[ColabfoldSearchResult] = Field(title="MSA Results", description="List of MSA search results")

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
        for query_idx, res in enumerate(self.results):
            # Name files by query index; paired queries (>1 chain) also get a chain index.
            paired = len(res.msas) > 1
            for chain_idx, msa in enumerate(res.msas):
                if msa is None:
                    continue
                stem = f"query_{query_idx}_chain_{chain_idx}" if paired else f"query_{query_idx}"
                out_file = path / f"{stem}.{file_format}"
                if file_format == "a3m":
                    msa.to_a3m_file(str(out_file))
                elif file_format == "fasta":
                    msa.to_fasta_file(str(out_file))

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

    @model_validator(mode="after")
    def warn_extra_args_in_remote_mode(self) -> "ColabfoldSearchConfig":
        """Warn when ``extra_args`` is set in remote mode (it has no effect there).

        The remote ColabFold MMseqs2 API does not accept arbitrary CLI tokens, so
        ``extra_args`` is silently dropped on the remote path. Emit a log warning
        rather than hard-erroring, since the rest of the configuration is still
        valid for a remote search.
        """
        if self.search_mode == "remote" and self.extra_args:
            logger.warning("Config field 'extra_args' is local-only and will be ignored in remote search mode.")
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
        >>> # Each result is a list of per-chain MSAs; an unpaired query has one.
        >>> msa = output.results[0][0]
        >>> print(f"Found {msa.num_sequences} homologous sequences")
        >>> # Iterate through aligned sequences
        >>> for seq in msa:
        ...     print(seq)
    """
    if not inputs.queries:
        return ColabfoldSearchOutput(results=[])

    # Cleanup leftover files from previous runs if using default directory and cache is empty
    _cleanup_default_output_dir_if_cache_empty(config)

    # Create output directory structure
    os.makedirs(config.output_dir, exist_ok=True)  # type: ignore[arg-type]
    msa_out_dir = os.path.join(config.output_dir, "msas")  # type: ignore[arg-type]
    os.makedirs(msa_out_dir, exist_ok=True)

    if config.search_mode not in ("local", "remote"):
        raise ValueError(f"colabfold-search: invalid search_mode {config.search_mode!r}; expected 'local' or 'remote'")

    unpaired_indices = [i for i, q in enumerate(inputs.queries) if not q.is_paired]
    paired_indices = [i for i, q in enumerate(inputs.queries) if q.is_paired]

    if paired_indices and config.search_mode == "local":
        raise NotImplementedError(
            "colabfold-search: paired queries (list-shaped sequences) require search_mode='remote'; "
            "local paired support is pending env-pairing DB provisioning."
        )

    # Results are assembled parallel to inputs.queries (by position).
    ordered: list[ColabfoldSearchResult | None] = [None] * len(inputs.queries)

    if unpaired_indices:
        # An unpaired query has exactly one chain; pass the bare sequence string.
        unpaired_seqs = [inputs.queries[i].sequences[0] for i in unpaired_indices]
        if config.search_mode == "local":
            unpaired_out = _local_search(unpaired_seqs, config, msa_out_dir, instance=instance)
        else:
            unpaired_out = _remote_search(unpaired_seqs, config, msa_out_dir, instance=instance)
        for query_idx, result in zip(unpaired_indices, unpaired_out.results, strict=True):
            ordered[query_idx] = result

    for query_idx in paired_indices:
        ordered[query_idx] = _remote_search_paired(inputs.queries[query_idx], config, instance=instance)

    # Every slot must be filled; results stay parallel to inputs.queries.
    assert all(r is not None for r in ordered), "internal: missing result for at least one query"
    results = [r for r in ordered if r is not None]  # narrows None out for the type checker
    return ColabfoldSearchOutput(results=results)


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


def _replace_query_header_in_a3m(a3m_path: str | Path, label: str) -> None:
    """Rewrite the query (first) header line in an A3M to ``label``, leaving homologs untouched.

    Local mode uses numeric FASTA indices and remote mode gets ColabFold's internal
    index (e.g. ``101``) for the query row; both are normalized to a stable label.
    """
    with open(a3m_path) as f:
        lines = f.readlines()
    if lines and lines[0].startswith(">"):
        lines[0] = f">{label}\n"
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
    config: ColabfoldSearchConfig,
    msa_out_dir: str,
    instance: Any = None,
) -> ColabfoldSearchOutput:
    """Performs local search for homologous sequences and generates Multiple Sequence Alignments (MSAs).

    Returns results parallel to ``sequences`` (by position).
    """
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

    # colabfold_search writes one file per query, named by FASTA index (0.a3m, 1.a3m, ...).
    results = []
    for idx, seq in enumerate(sequences):
        numbered_a3m = os.path.join(msa_out_dir, f"{idx}.a3m")
        if os.path.exists(numbered_a3m):
            _replace_query_header_in_a3m(numbered_a3m, "query")

        # Only the query row means no homologs were found → no MSA.
        num_sequences = _count_sequences_in_a3m(numbered_a3m) if os.path.exists(numbered_a3m) else 0
        msa = None if num_sequences < 2 else MSA.from_file(numbered_a3m)
        results.append(ColabfoldSearchResult(query_sequences=[seq], msas=[msa]))

    return ColabfoldSearchOutput(results=results)


def _remote_search(
    sequences: list[str],
    config: ColabfoldSearchConfig,
    msa_out_dir: str,  # noqa: ARG001 — required by tool interface
    instance: Any = None,
) -> ColabfoldSearchOutput:
    """Performs remote search for homologous sequences and generates Multiple Sequence Alignments (MSAs).

    The standalone keys ``msa_paths`` by query index; results are returned
    parallel to ``sequences`` (by position).
    """
    logger.debug(f"Generating remote MSAs for {len(sequences)} sequence(s)...")

    standalone_script = Path(__file__).parent / "standalone" / "remote_msa_search.py"

    input_data = {
        "queries": [{"sequences": seq} for seq in sequences],
        "output_dir": str(config.output_dir),
        "use_metagenomic_db": config.use_metagenomic_db,
        "verbose": config.verbose,
        "device": "cpu",
    }

    output_data = ToolInstance.dispatch(
        "colabfold_search",
        input_data,
        instance=instance,
        script_path=standalone_script,
        config=config,
    )

    if not output_data.get("success", False):
        num_successful = output_data.get("num_successful", 0)
        num_failed = output_data.get("num_failed", 0)
        if num_successful == 0:
            error_msg = f"colabfold-search: remote MSA search failed for all {num_failed} sequence(s)"
            if "errors" in output_data:
                error_msg += f"; errors: {output_data['errors']}"
            raise RuntimeError(error_msg)
        if config.verbose:
            logger.warning(f"Remote MSA search partially failed: {num_successful} succeeded, {num_failed} failed")
            if "errors" in output_data:
                for label, error in output_data["errors"].items():
                    logger.warning(f"  {label}: {error}")

    # Standalone keys msa_paths by query index (as str).
    msa_paths = output_data.get("msa_paths", {})
    results = []
    for idx, seq in enumerate(sequences):
        msa_path = msa_paths.get(str(idx))
        if msa_path is None:
            if config.verbose:
                logger.warning(f"No MSA generated for query {idx}")
            results.append(ColabfoldSearchResult(query_sequences=[seq], msas=[None]))
            continue
        # Remote A3M labels the query row ">101"; normalize it to a stable label.
        _replace_query_header_in_a3m(msa_path, "query")
        num_sequences = _count_sequences_in_a3m(msa_path)
        msa = None if num_sequences < 2 else MSA.from_file(msa_path)
        results.append(ColabfoldSearchResult(query_sequences=[seq], msas=[msa]))

    return ColabfoldSearchOutput(results=results)


def _remote_search_paired(
    query: ColabfoldSearchQuery,
    config: ColabfoldSearchConfig,
    instance: Any = None,
) -> ColabfoldSearchResult:
    r"""Submit one paired query to the remote API; return one result with row-aligned MSAs.

    Hits the ``ticket/pair`` endpoint via ``run_mmseqs2(use_pairing=True)``. The API
    returns one ``pair.a3m`` containing N chain blocks separated by ``\x00``; rows
    are taxonomy-paired across blocks by position. ``result.msas`` is a list of N
    MSAs in input chain order.
    """
    assert query.is_paired, "_remote_search_paired requires a list-sequences query"
    assert isinstance(query.sequences, list)

    logger.debug(f"Generating paired remote MSAs ({len(query.sequences)} chains)...")
    standalone_script = Path(__file__).parent / "standalone" / "remote_msa_search.py"

    input_data = {
        "queries": [query.model_dump()],
        "output_dir": str(config.output_dir),
        "use_metagenomic_db": config.use_metagenomic_db,
        "verbose": config.verbose,
        "device": "cpu",
    }

    output_data = ToolInstance.dispatch(
        "colabfold_search",
        input_data,
        instance=instance,
        script_path=standalone_script,
        config=config,
    )

    if not output_data.get("success", False):
        errors = output_data.get("errors", {})
        raise RuntimeError(f"colabfold-search: paired query failed; errors: {errors!r}")

    # Standalone keys paired_msa_paths by query index → per-chain paths in input order.
    chain_paths = output_data.get("paired_msa_paths", {}).get("0", [])
    if len(chain_paths) != len(query.sequences):
        raise RuntimeError(
            f"colabfold-search: paired standalone returned {len(chain_paths)} chain MSA path(s), "
            f"expected {len(query.sequences)} (one per chain)."
        )

    per_chain_msas: list[MSA | None] = []
    for chain_idx, msa_path in enumerate(chain_paths):
        num_seqs = _count_sequences_in_a3m(msa_path)
        if num_seqs < 2:
            if config.verbose:
                logger.warning(f"No paired MSA produced for chain {chain_idx}")
            per_chain_msas.append(None)
        else:
            per_chain_msas.append(MSA.from_file(msa_path))

    # Partial pairing (some chains found, others empty) breaks row-alignment for consumers.
    n_found = sum(m is not None for m in per_chain_msas)
    if 0 < n_found < len(per_chain_msas):
        raise RuntimeError(
            f"colabfold-search: paired query produced partial MSAs "
            f"({len(per_chain_msas) - n_found} of {len(per_chain_msas)} chains failed); "
            "paired downstream consumers require all per-chain MSAs to be present."
        )

    return ColabfoldSearchResult(query_sequences=query.sequences, msas=per_chain_msas)
