"""Generalized MMseqs2-based homology search for MSA generation.

The MSA-generation entry point of the unified ``mmseqs2`` toolkit, alongside
``mmseqs2-search-proteins`` / ``mmseqs2-search-genomes`` / ``mmseqs2-clustering``.
Replaces ``colabfold-search`` in structure predictors. This implementation
supports protein search via the registry's UniRef30 entry, GPU by default,
and unpaired MSAs.
"""

import hashlib
import logging
import os
import platform
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from proto_tools.databases import DatasetEntry, DatasetRegistry, get_dataset_dir
from proto_tools.tools.sequence_alignment.msas import MSA
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Input
# ============================================================================


class Mmseqs2HomologySearchQuery(BaseModel):
    """One sequence to search for homologs.

    Attributes:
        sequence (str): Amino acid (or, in future phases, nucleotide)
            sequence to search.
        sequence_id (str | None): Optional identifier. Auto-generated from
            a hash of the sequence when not provided.
        molecule_type (Literal["protein", "rna", "dna"] | None): Sequence
            type. When ``None``, inferred from the dataset's ``molecule_type``.
            Setting this explicitly lets the validator catch protein-vs-nucleic
            mismatches against the configured datasets.
    """

    sequence: str = Field(title="Sequence", description="Sequence to search for homologs")
    sequence_id: str | None = Field(default=None, title="Sequence ID", description="Optional sequence identifier")
    molecule_type: Literal["protein", "rna", "dna"] | None = Field(
        default=None,
        title="Molecule Type",
        description="Sequence type; inferred from datasets when None",
    )

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("sequence must be a non-empty string")
        return v.strip()


class Mmseqs2HomologySearchInput(BaseToolInput):
    """Input for ``mmseqs2-homology-search``.

    Queries are organized into groups. Each top-level item is a *group*:

    - A flat ``Mmseqs2HomologySearchQuery`` (or string / tuple sugar) is a
      *singleton* group — one chain, unpaired MSA.
    - A list of queries is a *paired* group — multiple chains whose MSAs
      should be row-synchronized by taxonomy.

    Phase 3 supports singleton groups only. Paired groups are validator-rejected;
    full support lands when the paired-MSA local path is implemented.

    Attributes:
        queries (list[Mmseqs2HomologySearchQuery | list[Mmseqs2HomologySearchQuery]]):
            List of query groups, in input order.
    """

    queries: list[Mmseqs2HomologySearchQuery | list[Mmseqs2HomologySearchQuery]] = InputField(
        title="Query Groups",
        description="Query groups (flat = singleton/unpaired; nested list = paired chains)",
    )

    @field_validator("queries", mode="before")
    @classmethod
    def _normalize_queries(cls, value: Any) -> Any:
        """Accept strings, tuples, Query objects, or lists of any of these as group items."""
        if not isinstance(value, list):
            value = [value]
        if not value:
            raise ValueError("At least one query group is required")

        normalized: list[Any] = []
        for raw_group in value:
            # Inner list (paired group)
            if isinstance(raw_group, list):
                inner = [_coerce_query(q) for q in raw_group]
                if not inner:
                    raise ValueError("Paired query group cannot be empty")
                normalized.append(inner)
            else:
                normalized.append(_coerce_query(raw_group))
        return normalized

    @model_validator(mode="after")
    def _populate_sequence_ids_and_check_uniqueness(self) -> Any:
        """Auto-generate sequence_ids where missing and enforce global uniqueness."""
        seen: set[str] = set()
        for group in self.queries:
            members = group if isinstance(group, list) else [group]
            for q in members:
                if q.sequence_id is None:
                    q.sequence_id = "seq_" + hashlib.sha256(q.sequence.encode()).hexdigest()[:10]
                if q.sequence_id in seen:
                    raise ValueError(f"Duplicate sequence_id: {q.sequence_id!r} (must be globally unique)")
                seen.add(q.sequence_id)
        return self

    @model_validator(mode="after")
    def _reject_paired_groups_in_phase_3(self) -> Any:
        """Phase 3: reject paired groups; full support lands with the paired-MSA local path."""
        for i, group in enumerate(self.queries):
            if isinstance(group, list) and len(group) >= 2:
                raise ValueError(
                    f"Paired-MSA queries (group #{i} has {len(group)} chains) are not yet supported. "
                    "For unpaired multimer search, pass each chain as a singleton (top-level) entry."
                )
        return self

    def all_queries(self) -> list[Mmseqs2HomologySearchQuery]:
        """Flatten groups → list of every query in input order."""
        flat: list[Mmseqs2HomologySearchQuery] = []
        for group in self.queries:
            if isinstance(group, list):
                flat.extend(group)
            else:
                flat.append(group)
        return flat

    def __len__(self) -> int:
        """Number of groups (matches output `results` length)."""
        return len(self.queries)


def _coerce_query(raw: Any) -> Mmseqs2HomologySearchQuery:
    """Convert string / tuple / dict / Query → Mmseqs2HomologySearchQuery."""
    if isinstance(raw, Mmseqs2HomologySearchQuery):
        return raw
    if isinstance(raw, str):
        return Mmseqs2HomologySearchQuery(sequence=raw)
    if isinstance(raw, tuple) and len(raw) == 2:
        return Mmseqs2HomologySearchQuery(sequence=raw[0], sequence_id=raw[1])
    if isinstance(raw, dict):
        # JSON round-trip case: model_dump serializes Mmseqs2HomologySearchQuery
        # as {"sequence": str, "sequence_id": str | None}.
        return Mmseqs2HomologySearchQuery(**raw)
    raise ValueError(
        f"Invalid query item: {raw!r}. Expected str, (sequence, id) tuple, dict, or Mmseqs2HomologySearchQuery."
    )


# ============================================================================
# Output
# ============================================================================


class Mmseqs2HomologySearchResult(BaseModel):
    """Per-group search result.

    A *singleton* group produces one entry in each of ``sequence_ids``, ``msas``,
    and ``num_homologs_found``, with ``paired_msas == [None]``. A *paired* group
    (Phase 4+) produces one entry per chain, with ``paired_msas`` row-synchronized
    across the group's chains.

    Attributes:
        sequence_ids (list[str]): Identifiers for the chains in this group.
        msas (list[MSA | None]): Per-chain unpaired MSAs. ``None`` when no
            homologs were found beyond the query itself.
        paired_msas (list[MSA | None]): Per-chain paired MSAs (Phase 4+). For
            singleton groups, always ``[None]``.
        datasets_searched (list[str]): Registry keys of datasets hit for this group.
        num_homologs_found (list[int]): Number of homologs per chain (excludes
            the query itself).
    """

    sequence_ids: list[str] = Field(title="Sequence IDs", description="Chain identifiers in this group")
    msas: list[MSA | None] = Field(title="Unpaired MSAs", description="Per-chain unpaired MSAs")
    paired_msas: list[MSA | None] = Field(
        title="Paired MSAs", description="Per-chain paired MSAs (Phase 4+); None in Phase 3"
    )
    datasets_searched: list[str] = Field(
        title="Datasets Searched",
        description="Registry keys of datasets searched for this group",
    )
    num_homologs_found: list[int] = Field(
        title="Homologs Per Chain",
        description="Homolog count per chain (excludes query)",
    )


class Mmseqs2HomologySearchOutput(BaseToolOutput):
    """Output for ``mmseqs2-homology-search``.

    Attributes:
        results (list[Mmseqs2HomologySearchResult]): One result per input
            group (matches the order of ``Mmseqs2HomologySearchInput.queries``).
    """

    results: list[Mmseqs2HomologySearchResult] = Field(
        title="Per-Group Results", description="One result per input group"
    )

    @property
    def output_format_options(self) -> list[str]:
        """A3M / FASTA exports per chain per group."""
        return ["a3m", "fasta"]

    @property
    def output_format_default(self) -> str:
        """Default export is A3M (matches structure-predictor inputs)."""
        return "a3m"

    def _export_output(self, export_path: Path | str, file_format: str) -> None:
        if file_format not in ("a3m", "fasta"):
            raise ValueError(f"Unsupported format: {file_format}")
        path = Path(export_path)
        path.mkdir(parents=True, exist_ok=True)
        for result in self.results:
            for seq_id, msa in zip(result.sequence_ids, result.msas, strict=True):
                if msa is None:
                    continue
                out = path / f"{seq_id}.{file_format}"
                if file_format == "a3m":
                    msa.to_a3m_file(str(out))
                else:
                    msa.to_fasta_file(str(out))

    def __len__(self) -> int:
        """Number of groups (matches input `queries` length)."""
        return len(self.results)

    def __getitem__(self, idx: int) -> Mmseqs2HomologySearchResult:
        """Index into results."""
        return self.results[idx]

    def __iter__(self) -> Iterator[Mmseqs2HomologySearchResult]:  # type: ignore[override]
        """Iterate over results."""
        return iter(self.results)


# ============================================================================
# Config
# ============================================================================


class Mmseqs2HomologySearchConfig(BaseConfig):
    """Configuration for ``mmseqs2-homology-search``.

    Attributes:
        datasets (list[str]): Registered dataset keys; Phase 3 accepts
            exactly one protein dataset.
        use_gpu (bool): Run MMseqs2-GPU; requires a ``.idx_pad`` index,
            an NVIDIA GPU (Turing+), and a Linux host.
        pairing_strategy (Literal["greedy", "complete"]): Paired-MSA strategy
            (used in Phase 4; ignored in Phase 3).
        sensitivity (float | None): MMseqs2 ``-s`` override; ignored under
            ``use_gpu=True``. ``None`` uses the dataset's registered default.
        num_threads (int | None): CPU threads; ``None`` auto-detects all cores.
        timeout (int | None): Subprocess timeout in seconds. ``None`` waits indefinitely.

    Note:
        A3M files are written to a per-call temporary directory and parsed
        into in-memory ``MSA`` objects on the result. The temp dir is
        cleaned up after the call returns. Use ``result.export(path, "a3m")``
        to materialize files at a chosen location — same persistence API
        as every other proto-tool with file outputs.
    """

    datasets: list[str] = ConfigField(
        title="Datasets",
        default=["uniref30-2302"],
        description="Registered dataset keys (e.g. `uniref30-2302`). Phase 3: exactly one protein dataset.",
    )
    use_gpu: bool = ConfigField(
        title="Use GPU",
        default=True,
        description="Use MMseqs2-GPU; requires a `.idx_pad` index, an NVIDIA GPU (Turing+), and a Linux host.",
    )
    pairing_strategy: Literal["greedy", "complete"] = ConfigField(
        title="Pairing Strategy",
        default="greedy",
        description="Paired-MSA strategy (forward-compat field; not yet wired up).",
    )
    sensitivity: float | None = ConfigField(
        title="MMseqs2 Sensitivity",
        default=None,
        ge=1.0,
        le=9.0,
        description="MMseqs2 `-s` override (1.0-9.0); ignored on GPU; `None` uses the dataset's default.",
    )
    num_threads: int | None = ConfigField(
        title="Number of Threads",
        default=None,
        ge=1,
        description="CPU threads; `None` auto-detects all available cores.",
        include_in_key=False,
    )
    timeout: int | None = ConfigField(
        title="Timeout",
        default=3600,
        ge=1,
        description="Subprocess timeout in seconds; full-database searches can exceed 10 minutes.",
        include_in_key=False,
    )

    @model_validator(mode="after")
    def _validate_datasets_registered(self) -> Any:
        """Every dataset key must be in the registry."""
        unknown = [name for name in self.datasets if name not in DatasetRegistry.list_all()]
        if unknown:
            available = ", ".join(DatasetRegistry.list_all()) or "<none>"
            raise ValueError(f"Unknown dataset(s) {unknown}. Registered: {available}")
        return self

    @model_validator(mode="after")
    def _validate_single_dataset_phase_3(self) -> Any:
        """Phase 3 supports exactly one dataset; multi-dataset is a follow-up."""
        if len(self.datasets) != 1:
            raise ValueError(
                f"Phase 3 supports exactly one dataset, got {len(self.datasets)}: {self.datasets}. "
                "Multi-dataset support (UniRef30 + envdb + BFD merge) ships in a follow-up PR."
            )
        return self

    @model_validator(mode="after")
    def _validate_uniform_molecule_type(self) -> Any:
        """All datasets must share a molecule type (so queries can target them as one)."""
        types = {DatasetRegistry.get(name).molecule_type for name in self.datasets}
        if len(types) > 1:
            raise ValueError(
                f"Datasets must share molecule_type, got {sorted(types)}. "
                "Mix protein and nucleotide datasets in separate calls."
            )
        return self

    @model_validator(mode="after")
    def _validate_a3m_adapter_supported_in_phase_3(self) -> Any:
        """Phase 3 only handles ColabFold-style profile DBs.

        The standalone wraps colabfold_search's iterative pipeline, which only
        works against UniRef30 / ColabFoldDB envdb. AF3-style and RNA datasets
        need a different MMseqs2 invocation pattern, landing in Phase 4. Block
        them at config time with a clear hint.
        """
        unsupported = [
            (name, DatasetRegistry.get(name).a3m_adapter)
            for name in self.datasets
            if DatasetRegistry.get(name).a3m_adapter != "colabfold"
        ]
        if unsupported:
            raise ValueError(
                f"Phase 3 only supports datasets with a3m_adapter='colabfold' (UniRef30, "
                f"ColabFoldDB envdb). Got: {unsupported}. AF3-style protein and RNA "
                "datasets are registered and provisionable but not yet searchable — "
                "tracked for Phase 4 of the mmseqs2-homology-search rollout."
            )
        return self

    @model_validator(mode="after")
    def _validate_use_gpu_platform(self) -> Any:
        """GPU search requires Linux (the GPU MMseqs2 binary is Linux-only)."""
        if self.use_gpu and platform.system() != "Linux":
            raise ValueError(
                f"use_gpu=True requires Linux (current: {platform.system()} {platform.machine()}). "
                "Set use_gpu=False to fall back to CPU search."
            )
        return self

    @model_validator(mode="after")
    def _validate_use_gpu_supported_by_datasets(self) -> Any:
        """Every selected dataset must support GPU when use_gpu=True."""
        if not self.use_gpu:
            return self
        unsupported = [name for name in self.datasets if not DatasetRegistry.get(name).supports_gpu]
        if unsupported:
            raise ValueError(
                f"use_gpu=True but datasets {unsupported} declare supports_gpu=False. "
                "Set use_gpu=False or pick a GPU-capable dataset."
            )
        return self

    @property
    def gpus_per_instance(self) -> int:
        """Number of GPUs the configured search uses (1 if GPU, else 0)."""
        return 1 if self.use_gpu else 0

    @classmethod
    def minimal(cls, **kwargs: Any) -> "Mmseqs2HomologySearchConfig":
        """Cheap-mode defaults for env-report and seed-reproducibility tests.

        Points at the in-tree ``tiny-test-colabfold`` fixture DB so the test
        infrastructure never needs the ~100 GB UniRef30 tarball, and forces
        CPU search so coverage isn't gated on a GPU being present.
        """
        kwargs.setdefault("datasets", ["tiny-test-colabfold"])
        kwargs.setdefault("use_gpu", False)
        return super().minimal(**kwargs)  # type: ignore[return-value]


# ============================================================================
# Tool implementation
# ============================================================================


def example_input() -> Mmseqs2HomologySearchInput:
    """Minimal valid input for testing — single human ubiquitin chain."""
    return Mmseqs2HomologySearchInput(
        queries=[
            Mmseqs2HomologySearchQuery(
                sequence="MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
                sequence_id="ubiquitin_human",
            )
        ],
    )


@tool(
    key="mmseqs2-homology-search",
    label="MMseqs2 Homology Search",
    category="sequence_alignment",
    input_class=Mmseqs2HomologySearchInput,
    config_class=Mmseqs2HomologySearchConfig,
    output_class=Mmseqs2HomologySearchOutput,
    description="Generate MSAs by searching protein sequences against MMseqs2-indexed databases (GPU by default).",
    example_input=example_input,
    iterable_input_field="queries",
    iterable_output_field="results",
    cacheable=True,
    device_count="<=1",
)
def run_mmseqs2_homology_search(
    inputs: Mmseqs2HomologySearchInput,
    config: Mmseqs2HomologySearchConfig,
    instance: Any = None,
) -> Mmseqs2HomologySearchOutput:
    """Execute homology search against the configured registered dataset(s).

    Args:
        inputs (Mmseqs2HomologySearchInput): Query groups (singleton only in Phase 3).
        config (Mmseqs2HomologySearchConfig): Search configuration; ``datasets``
            picks the registered DB(s), ``use_gpu`` toggles MMseqs2-GPU.
        instance (Any): Optional persistent ``ToolInstance`` for batch workloads.

    Returns:
        Mmseqs2HomologySearchOutput: One result per input group, with per-chain
            ``msas`` and (in Phase 4+) ``paired_msas``.
    """
    # Phase 3 validators guarantee exactly one dataset, all singleton groups
    dataset_name = config.datasets[0]
    entry: DatasetEntry = DatasetRegistry.get(dataset_name)
    cache_dir = get_dataset_dir(dataset_name)

    # Disk checks happen here (not as Config validators) so Config()
    # construction stays cheap and works in CI / fresh dev machines without
    # any datasets provisioned. The user only hits this error when they
    # actually try to dispatch a search.
    if entry.auto_provision and not _is_provisioned(entry, cache_dir, require_idx_pad=config.use_gpu):
        # Tiny in-tree fixture entries build themselves on first dispatch so
        # CI / smoke tests don't need a separate provisioning step.
        _auto_provision(dataset_name, cache_dir, require_idx_pad=config.use_gpu)
    _check_dataset_provisioned(dataset_name, entry, cache_dir, require_idx_pad=config.use_gpu)

    flat_queries = inputs.all_queries()
    sequences = [q.sequence for q in flat_queries]

    num_threads = config.num_threads
    if num_threads is None:
        num_threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1

    # Sensitivity: user override > registry default
    sensitivity = config.sensitivity if config.sensitivity is not None else entry.mmseqs_flags.sensitivity

    # Inner colabfold_search timeout fires 30s before the framework's outer
    # ToolInstance timeout so the standalone returns a structured error with
    # explicit subprocess cleanup instead of being hard-killed mid-call.
    # When the outer timeout is unbounded (None), the inner is also unbounded.
    _OUTER_TIMEOUT_GRACE_S = 30
    inner_timeout = None if config.timeout is None else max(1, config.timeout - _OUTER_TIMEOUT_GRACE_S)

    # A3M files are intermediates: the standalone writes them, the tool layer
    # parses them into in-memory MSA objects, then the tempdir auto-cleans.
    # Persistence goes through `result.export(path, "a3m")` — same pattern as
    # every other tool with file outputs.
    with tempfile.TemporaryDirectory(prefix="mmseqs2_homology_search_") as tmp_dir_str:
        msa_out_dir = Path(tmp_dir_str)
        payload = {
            "operation": "homology_search",
            "sequences": sequences,
            "dataset_dir": str(cache_dir),
            "db_prefix": entry.db_prefix,
            "output_dir": str(msa_out_dir),
            "num_threads": num_threads,
            "use_gpu": config.use_gpu,
            "verbose": config.verbose,
            "sensitivity": sensitivity,
            "prefilter_mode": entry.mmseqs_flags.prefilter_mode,
            "max_seqs": entry.mmseqs_flags.max_seqs,
            "extra_args": list(entry.mmseqs_flags.extra_args),
            "colabfold_timeout": inner_timeout,
            "device": "cuda" if config.use_gpu else "cpu",
        }

        output = ToolInstance.dispatch(
            "mmseqs2",
            payload,
            instance=instance,
            config=config,
        )

        if not output.get("success", False):
            raise RuntimeError(f"mmseqs2-homology-search failed: {output.get('error', 'unknown error')}")

        # Parse A3Ms back into MSA objects, one per query. The standalone writes
        # outputs as ``__q{idx}.a3m`` (internal header, decoupled from user
        # sequence_id) so that user-chosen IDs can never collide with
        # colabfold_search's positional file naming. We rewrite each file's
        # query header to the user's sequence_id and parse fully into memory
        # before the tempdir is cleaned up on context exit.
        results: list[Mmseqs2HomologySearchResult] = []
        for idx, query in enumerate(flat_queries):
            assert query.sequence_id is not None  # populated by Mmseqs2HomologySearchInput validator
            a3m_path = _rename_a3m_to_sequence_id(msa_out_dir, idx, query.sequence_id)

            msa: MSA | None = None
            homologs = 0
            if a3m_path is not None:
                num_seqs = _count_sequences_in_a3m(a3m_path)
                if num_seqs >= 2:
                    msa = MSA.from_file(str(a3m_path))
                    homologs = num_seqs - 1

            results.append(
                Mmseqs2HomologySearchResult(
                    sequence_ids=[query.sequence_id],
                    msas=[msa],
                    paired_msas=[None],  # Phase 3 only emits singletons
                    datasets_searched=[dataset_name],
                    num_homologs_found=[homologs],
                )
            )

    return Mmseqs2HomologySearchOutput(results=results)


# ============================================================================
# Helpers
# ============================================================================


def _count_sequences_in_a3m(a3m_path: Path) -> int:
    """Count ``>``-prefixed header lines in an A3M file."""
    count = 0
    with open(a3m_path) as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count


def _replace_query_header_in_a3m(a3m_path: Path, seq_id: str) -> None:
    """Rewrite the first header line of an A3M file to use ``seq_id``.

    The standalone subprocess writes A3Ms with internal ``__q{idx}`` headers
    (decoupled from user sequence_ids — see ``standalone/run.py``). This
    rewrites the first header to the user's sequence_id so downstream
    consumers see the real identifier. All other homolog headers are untouched.
    """
    lines = a3m_path.read_text().split("\n")
    for i, line in enumerate(lines):
        if line.startswith(">"):
            lines[i] = f">{seq_id}"
            break
    a3m_path.write_text("\n".join(lines))


def _rename_a3m_to_sequence_id(msa_out_dir: Path, idx: int, sequence_id: str) -> Path | None:
    """Materialize a per-query A3M under the user's sequence_id.

    Renames ``msa_out_dir / f"__q{idx}.a3m"`` to ``msa_out_dir / f"{sequence_id}.a3m"``
    and rewrites its first FASTA header line to ``>{sequence_id}``. Source
    files use the ``__q{idx}`` prefix specifically so that user-chosen
    ``sequence_id``s — including adversarial values that look like our
    internal index naming (e.g. ``"0"``, ``"1"``) — can never collide with
    one another's output.

    Args:
        msa_out_dir (Path): Per-run output directory containing ``__q{idx}.a3m`` files.
        idx (int): Position of the query in the flattened query list.
        sequence_id (str): User-supplied identifier used as the renamed
            file's stem and the rewritten query header.

    Returns:
        Path | None: Path to the renamed file (always
            ``msa_out_dir / f"{sequence_id}.a3m"``), or ``None`` when
            colabfold_search produced no output for this query (e.g. the
            ``no_hits_fallback`` path didn't fire and the file is missing).
    """
    internal = msa_out_dir / f"__q{idx}.a3m"
    if not internal.exists():
        return None
    _replace_query_header_in_a3m(internal, sequence_id)
    named = msa_out_dir / f"{sequence_id}.a3m"
    internal.rename(named)
    return named


def _is_provisioned(entry: DatasetEntry, cache_dir: Path, *, require_idx_pad: bool) -> bool:
    """Cheap, side-effect-free check for whether the indexed DB exists on disk."""
    if not (cache_dir / f"{entry.db_prefix}.dbtype").is_file():
        return False
    if require_idx_pad and entry.gpu_padded_marker:
        # A complete padded DB has both the marker and its ``.dbtype`` companion.
        marker = cache_dir / entry.gpu_padded_marker
        if not marker.exists() or not Path(f"{marker}.dbtype").is_file():
            return False
    return True


def _auto_provision(name: str, cache_dir: Path, *, require_idx_pad: bool) -> None:
    """Provision an ``auto_provision=True`` entry by running its download + index recipe.

    Reuses the standalone env's ``mmseqs`` binary (built lazily via
    ``ToolInstance.ensure_ready``) so callers don't need ``mmseqs`` on PATH.
    Reserved for tiny fixture datasets — production entries declare
    ``auto_provision=False`` and require ``setup_databases.py`` instead.

    Concurrency: holds an advisory ``flock`` on a sibling lockfile across the
    download + index steps so parallel pytest-xdist workers (or any other
    concurrent caller) serialize cleanly instead of racing on ``mmseqs mvdb``
    / ``tar`` against a half-written cache dir. The second arrival blocks,
    then re-checks ``_is_provisioned`` and skips its own work. The re-check
    uses the caller's ``require_idx_pad`` so a partial cpu-only provision
    on disk doesn't fool a use_gpu=True caller into skipping the rebuild.

    Refuses to run under GitHub Actions (``GITHUB_ACTIONS=true``) so CI
    runners never silently pull fixture tarballs from the public mirror.
    """
    if os.environ.get("GITHUB_ACTIONS") == "true":
        raise FileNotFoundError(
            f"Dataset {name!r} is auto_provision but auto-provisioning is disabled "
            "under GITHUB_ACTIONS=true (CI runners shouldn't silently download "
            "fixture tarballs). Provision manually with setup_databases.py if needed."
        )

    import fcntl
    import subprocess

    from proto_tools.databases import DatasetRegistry
    from proto_tools.tools.sequence_alignment.mmseqs2.setup_databases import (
        _is_provisioned as _setup_db_is_provisioned,
    )
    from proto_tools.tools.sequence_alignment.mmseqs2.setup_databases import (
        provision,
    )
    from proto_tools.utils.tool_instance import ToolInstance

    instance = ToolInstance("mmseqs2")
    instance.ensure_ready()
    env_bin = instance.env_path / "bin"

    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir.parent / f".{cache_dir.name}.provision.lock"

    logger.info("Auto-provisioning fixture dataset %r at %s", name, cache_dir)

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        entry = DatasetRegistry.get(name)
        if _is_provisioned(entry, cache_dir, require_idx_pad=require_idx_pad):
            logger.info("Auto-provision skipped for %r: another worker already provisioned it", name)
            return

        if require_idx_pad and entry.gpu_padded_marker and _setup_db_is_provisioned(entry, cache_dir):
            mmseqs_bin = env_bin / "mmseqs"
            logger.info(
                "Auto-provision fast-path for %r: building missing GPU-padded index (%s) only",
                name,
                entry.gpu_padded_marker,
            )
            subprocess.run(  # noqa: S603 — args from trusted registry + env-resolved bin
                [str(mmseqs_bin), "makepaddedseqdb", entry.db_prefix, entry.gpu_padded_marker],
                cwd=cache_dir,
                check=True,
            )
            return

        # The mmseqs env carries the ``mmseqs`` binary; the download step inside
        # ``provision()`` also needs ``curl`` (or wget/aria2c). On hosts that
        # don't ship one (e.g. CentOS 7), ``ToolInstance._ensure_foundation_env``
        # provisions a shared micromamba env with curl/git/gcc — prepend its
        # ``bin/`` here so the download tool is reachable, mirroring what
        # ``ToolInstance._setup`` does for the setup.sh subprocess. Returns
        # None when the host already provides the tools, in which case nothing
        # extra is added.
        #
        # NOTE: PATH mutation is process-global and not thread-safe. Safe today
        # because auto_provision only fires from a single test thread; if this
        # ever runs from a ToolPool worker, swap to passing env_bin into a
        # provision() variant that resolves mmseqs explicitly.
        foundation_path = ToolInstance._ensure_foundation_env()
        path_prefix = str(env_bin)
        if foundation_path is not None:
            path_prefix = f"{path_prefix}{os.pathsep}{foundation_path / 'bin'}"
        original_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{path_prefix}{os.pathsep}{original_path}"
        try:
            provision(name, force=True)
        finally:
            os.environ["PATH"] = original_path


def _check_dataset_provisioned(name: str, entry: DatasetEntry, cache_dir: Path, *, require_idx_pad: bool) -> None:
    """Raise with a provisioning hint if ``name`` isn't on disk.

    Run-time check (not a Config validator) so Config construction stays
    cheap on machines without datasets provisioned. Verifies the indexed DB
    exists, and (when ``require_idx_pad=True``) the GPU-padded index too.
    """
    if not (cache_dir / f"{entry.db_prefix}.dbtype").is_file():
        cmd_path = cache_dir
        cmd_name = name.replace("-", "_")
        raise FileNotFoundError(
            f"Dataset {name!r} not provisioned on disk: expected {cache_dir}.\n"
            "Provision with:\n"
            "  bash proto_tools/tools/sequence_alignment/colabfold_search/setup_databases.sh \\\n"
            f'    "{cmd_path}" {cmd_name} colabfold_envdb_202108 1\n'
            "(see proto_tools/tools/sequence_alignment/colabfold_search/README.md → Local Database Setup)"
        )
    if require_idx_pad and entry.gpu_padded_marker:
        marker = cache_dir / entry.gpu_padded_marker
        if not marker.exists() or not Path(f"{marker}.dbtype").is_file():
            raise FileNotFoundError(
                f"Dataset {name!r} is missing a complete GPU-padded DB "
                f"({entry.gpu_padded_marker} with sibling .dbtype) at {cache_dir}.\n"
                "Build it with: mmseqs makepaddedseqdb <db_prefix> <gpu_padded_marker>\n"
                "(setup_databases.py runs this automatically; rerun or set use_gpu=False)"
            )
