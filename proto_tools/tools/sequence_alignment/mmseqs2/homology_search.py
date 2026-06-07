"""Generalized MMseqs2-based homology search for MSA generation.

The MSA-generation entry point of the unified ``mmseqs2`` toolkit, alongside
``mmseqs2-search-proteins`` / ``mmseqs2-search-genomes`` / ``mmseqs2-clustering``.
Searches protein queries against the registry's UniRef30 entry (GPU by default),
producing unpaired MSAs for singleton queries and taxonomy-paired MSAs for
multi-chain groups.
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
from proto_tools.entities.msa import MSA
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
    resolve_num_threads,
)

logger = logging.getLogger(__name__)

# colabfold_search `--pairing_strategy` maps to mmseqs `pairaln --pairing-mode`: greedy=0, complete=1.
_PAIRING_MODE_INT = {"greedy": 0, "complete": 1}

# datasets_searched label for remote results (the ColabFold API uses its own hosted UniRef30).
_REMOTE_DATASET_LABEL = "colabfold-remote"

# Local metagenomic search adds this registered envdb as colabfold_search's --db3 (--use-env 1).
_METAGENOMIC_DATASET = "colabfold-envdb-202108"


# ============================================================================
# Input
# ============================================================================


class Mmseqs2HomologySearchQuery(BaseModel):
    """One sequence to search for homologs.

    Attributes:
        sequence (str): Amino acid sequence to search.
        sequence_id (str | None): Optional identifier. Auto-generated from
            a hash of the sequence when not provided.
        molecule_type (Literal["protein", "rna", "dna"] | None): Sequence
            type. When ``None`` (the default), inferred from the dataset's
            ``molecule_type``.
    """

    sequence: str = Field(title="Sequence", description="Sequence to search for homologs")
    sequence_id: str | None = Field(default=None, title="Sequence ID", description="Optional sequence identifier")
    molecule_type: Literal["protein", "rna", "dna"] | None = Field(
        default=None,
        title="Molecule Type",
        description="Sequence type; inferred from the dataset when None",
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
      are row-synchronized by taxonomy.

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
    produces one entry per chain, with ``paired_msas`` row-synchronized across the
    group's chains by taxonomy.

    Attributes:
        sequence_ids (list[str]): Identifiers for the chains in this group.
        msas (list[MSA | None]): Per-chain unpaired MSAs. ``None`` when no
            homologs were found beyond the query itself.
        paired_msas (list[MSA | None]): Per-chain taxonomy-paired MSAs,
            row-aligned across the group's chains. ``[None]`` for singleton
            groups (nothing to pair).
        datasets_searched (list[str]): Registry keys of datasets hit for this group.
        num_homologs_found (list[int]): Number of homologs per chain (excludes
            the query itself).
    """

    sequence_ids: list[str] = Field(title="Sequence IDs", description="Chain identifiers in this group")
    msas: list[MSA | None] = Field(title="Unpaired MSAs", description="Per-chain unpaired MSAs")
    paired_msas: list[MSA | None] = Field(
        title="Paired MSAs", description="Per-chain taxonomy-paired MSAs; [None] for singleton groups"
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
        search_mode (Literal["local", "remote"]): ``"local"`` runs MMseqs2
            against a registry-provisioned DB on disk; ``"remote"`` (the
            default) queries the ColabFold MSA API over the network and needs
            no local DB.
        dataset (Literal["colabfold-envdb-202108", "uniref30-2302"]): Local-only
            (ignored when remote). Registered key of the searchable reference
            database; one ColabFold protein DB.
        use_gpu (bool): Local-only (ignored when remote). Run MMseqs2-GPU;
            requires a ``.idx_pad`` index, an NVIDIA GPU (Turing+), and a
            Linux host.
        use_metagenomic_db (bool): Include the metagenomic/environmental DB
            (ColabFoldDB envdb) to deepen unpaired MSAs. Works in both modes;
            local mode requires the ``colabfold-envdb-202108`` dataset
            provisioned. Default ``False``. Does not affect cross-chain pairing.
        pairing_strategy (Literal["greedy", "complete"]): Cross-chain pairing
            strategy for paired (multi-chain) groups. ``"greedy"`` pairs a
            species found in at least two chains; ``"complete"`` only pairs a
            species present in every chain. Ignored for singleton groups, and
            (remote-mode only) the API always uses its own greedy pairing.
        sensitivity (float | None): Local-only (ignored when remote). MMseqs2
            ``-s`` override; ignored under ``use_gpu=True``. ``None`` uses the
            dataset's registered default.
        num_threads (int | None): Local-only. CPU threads; ``None`` auto-detects
            all cores.
        timeout (int | None): Subprocess timeout in seconds. ``None`` waits indefinitely.

    Note:
        A3M files are written to a per-call temporary directory and parsed
        into in-memory ``MSA`` objects on the result. The temp dir is
        cleaned up after the call returns. Use ``result.export(path, "a3m")``
        to materialize files at a chosen location — same persistence API
        as every other proto-tool with file outputs.
    """

    search_mode: Literal["local", "remote"] = ConfigField(
        title="Search Mode",
        default="remote",
        description="`remote` queries ColabFold's MSA API; `local` runs MMseqs2 against a registry-provisioned DB.",
    )
    dataset: Literal["colabfold-envdb-202108", "uniref30-2302"] = ConfigField(
        title="Dataset",
        default="uniref30-2302",
        description="Registered ColabFold protein database to search (e.g. `uniref30-2302`); one per call.",
    )
    use_gpu: bool = ConfigField(
        title="Use GPU",
        default=True,
        description="Use MMseqs2-GPU; requires a `.idx_pad` index, an NVIDIA GPU (Turing+), and a Linux host.",
    )
    use_metagenomic_db: bool = ConfigField(
        title="Use Metagenomic Database",
        default=False,
        description="Include the metagenomic/environmental DB (ColabFoldDB envdb); local mode needs it provisioned.",
    )
    pairing_strategy: Literal["greedy", "complete"] = ConfigField(
        title="Pairing Strategy",
        default="greedy",
        description="Cross-chain pairing for paired groups: `greedy` (species in >=2 chains) or `complete` (all).",
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
    def _validate_use_gpu_platform(self) -> Any:
        """GPU search requires Linux (the GPU MMseqs2 binary is Linux-only); use_gpu is ignored when remote."""
        if self.use_gpu and self.search_mode == "local" and platform.system() != "Linux":
            raise ValueError(
                f"use_gpu=True requires Linux (current: {platform.system()} {platform.machine()}). "
                "Set use_gpu=False to fall back to CPU search."
            )
        return self

    @property
    def gpus_per_instance(self) -> int:
        """Number of GPUs the configured search uses (1 for local GPU search, else 0).

        Remote search runs over the network, so it claims no GPU even though
        ``use_gpu`` keeps its (ignored) local-mode default.
        """
        return 1 if (self.use_gpu and self.search_mode == "local") else 0

    @classmethod
    def minimal(cls, **kwargs: Any) -> "Mmseqs2HomologySearchConfig":
        """Cheap-mode defaults for construct-time test infrastructure.

        Forces CPU search so construction isn't gated on a GPU/Linux host,
        keeps the default ``uniref30-2302`` (a valid ``dataset`` Literal value),
        and pins ``search_mode="local"`` so env-report / seed tests never hit
        the network. The in-tree ``tiny-test-colabfold`` fixture is excluded
        from the product Literal, so tests that actually run a search against
        it build the config via ``model_construct`` rather than through
        ``minimal``.
        """
        kwargs.setdefault("search_mode", "local")
        kwargs.setdefault("dataset", "uniref30-2302")
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
    """Execute homology search against the configured registered dataset.

    Args:
        inputs (Mmseqs2HomologySearchInput): Query groups; singletons yield
            unpaired MSAs, multi-chain groups yield taxonomy-paired MSAs.
        config (Mmseqs2HomologySearchConfig): Search configuration; ``dataset``
            picks the registered DB, ``use_gpu`` toggles MMseqs2-GPU,
            ``pairing_strategy`` controls cross-chain pairing.
        instance (Any): Optional persistent ``ToolInstance`` for batch workloads.

    Returns:
        Mmseqs2HomologySearchOutput: One result per input group, with per-chain
            ``msas`` and (for paired groups) ``paired_msas``.
    """
    # Remote mode (the default) hits the ColabFold MSA API and ignores dataset/use_gpu/sensitivity.
    if config.search_mode == "remote":
        return _run_remote_homology_search(inputs, config, instance)

    # The Literal type guarantees a colabfold-style protein dataset.
    dataset_name = config.dataset
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

    # Metagenomic search adds the envdb as colabfold_search's --db3; gate on it being provisioned (with a GPU-padded index too when use_gpu).
    env_dataset_dir: str | None = None
    env_db_prefix: str | None = None
    if config.use_metagenomic_db:
        env_entry = DatasetRegistry.get(_METAGENOMIC_DATASET)
        env_cache = get_dataset_dir(_METAGENOMIC_DATASET)
        _check_dataset_provisioned(_METAGENOMIC_DATASET, env_entry, env_cache, require_idx_pad=config.use_gpu)
        env_dataset_dir = str(env_cache)
        env_db_prefix = env_entry.db_prefix

    num_threads = resolve_num_threads(config.num_threads)

    # Sensitivity: user override > registry default
    sensitivity = config.sensitivity if config.sensitivity is not None else entry.mmseqs_flags.sensitivity

    # Inner colabfold_search timeout fires 30s before the framework's outer
    # ToolInstance timeout so the standalone returns a structured error with
    # explicit subprocess cleanup instead of being hard-killed mid-call.
    # When the outer timeout is unbounded (None), the inner is also unbounded.
    _OUTER_TIMEOUT_GRACE_S = 30
    inner_timeout = None if config.timeout is None else max(1, config.timeout - _OUTER_TIMEOUT_GRACE_S)

    # Shared dispatch payload; each search adds sequences/output_dir/pairing_strategy.
    base_payload = {
        "operation": "homology_search",
        "dataset_dir": str(cache_dir),
        "db_prefix": entry.db_prefix,
        "num_threads": num_threads,
        "use_gpu": config.use_gpu,
        "verbose": config.verbose,
        "sensitivity": sensitivity,
        "prefilter_mode": entry.mmseqs_flags.prefilter_mode,
        "max_seqs": entry.mmseqs_flags.max_seqs,
        "extra_args": list(entry.mmseqs_flags.extra_args),
        "colabfold_timeout": inner_timeout,
        "device": "cuda" if config.use_gpu else "cpu",
        "use_metagenomic_db": config.use_metagenomic_db,
        "env_dataset_dir": env_dataset_dir,
        "env_db_prefix": env_db_prefix,
    }

    def _dispatch(sequences: list[str], output_dir: Path, pairing_strategy: int | None) -> None:
        payload = {
            **base_payload,
            "sequences": sequences,
            "output_dir": str(output_dir),
            "pairing_strategy": pairing_strategy,
        }
        out = ToolInstance.dispatch("mmseqs2", payload, instance=instance, config=config)
        if not out.get("success", False):
            raise RuntimeError(f"mmseqs2-homology-search failed: {out.get('error', 'unknown error')}")

    # Singletons share one batched unpaired search; each paired group (>=2 chains) dispatches on its own.
    singletons: list[tuple[int, Mmseqs2HomologySearchQuery]] = []
    paired_groups: list[tuple[int, list[Mmseqs2HomologySearchQuery]]] = []
    for group_idx, group in enumerate(inputs.queries):
        members = group if isinstance(group, list) else [group]
        if isinstance(group, list) and len(members) >= 2:
            paired_groups.append((group_idx, members))
        else:
            singletons.append((group_idx, members[0]))

    pairing_int = _PAIRING_MODE_INT[config.pairing_strategy]

    # A3M files are intermediates: the standalone writes them, the tool layer
    # parses them into in-memory MSA objects, then the tempdir auto-cleans.
    # Persistence goes through `result.export(path, "a3m")` — same pattern as
    # every other tool with file outputs.
    results: list[Mmseqs2HomologySearchResult | None] = [None] * len(inputs.queries)
    with tempfile.TemporaryDirectory(prefix="mmseqs2_homology_search_") as tmp_dir_str:
        tmp_root = Path(tmp_dir_str)

        # Batched unpaired search over all singletons; outputs use internal __q{idx}.a3m names, renamed to sequence_id below.
        if singletons:
            unpaired_dir = tmp_root / "unpaired"
            unpaired_dir.mkdir()
            _dispatch([q.sequence for _, q in singletons], unpaired_dir, None)
            for flat_idx, (group_idx, query) in enumerate(singletons):
                assert query.sequence_id is not None  # populated by the input validator
                a3m_path = _rename_a3m_to_sequence_id(unpaired_dir, flat_idx, query.sequence_id)
                msa, homologs = _parse_a3m(a3m_path)
                results[group_idx] = Mmseqs2HomologySearchResult(
                    sequence_ids=[query.sequence_id],
                    msas=[msa],
                    paired_msas=[None],
                    datasets_searched=[dataset_name],
                    num_homologs_found=[homologs],
                )

        # One paired search per multi-chain group; standalone emits per-chain {i}.a3m (unpaired) and {i}.paired.a3m (row-aligned).
        for group_idx, members in paired_groups:
            group_dir = tmp_root / f"paired_{group_idx}"
            group_dir.mkdir()
            _dispatch([q.sequence for q in members], group_dir, pairing_int)
            results[group_idx] = _assemble_paired_result(group_dir, members, dataset_name)

    assert all(r is not None for r in results), "internal: missing result for at least one group"
    return Mmseqs2HomologySearchOutput(results=[r for r in results if r is not None])


# ============================================================================
# Remote mode (ColabFold API)
# ============================================================================


def _dispatch_remote(
    remote_queries: list[dict[str, Any]],
    output_dir: Path,
    config: Mmseqs2HomologySearchConfig,
    instance: Any,
    script_path: Path,
) -> dict[str, Any]:
    """Dispatch the ColabFold remote standalone for a batch of queries.

    ``remote_queries`` items are ``{"sequences": str}`` (unpaired) or
    ``{"sequences": [str, ...]}`` (one paired group). ``use_metagenomic_db``
    forwards to the ColabFold API's ``use_env`` (off by default).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "queries": remote_queries,
        "output_dir": str(output_dir),
        "use_metagenomic_db": config.use_metagenomic_db,
        "verbose": config.verbose,
        "device": "cpu",
    }
    return ToolInstance.dispatch("mmseqs2", payload, instance=instance, script_path=script_path, config=config)


def _parse_remote_a3m(path: str | None, seq_id: str) -> tuple[MSA | None, int]:
    """Normalize a remote A3M's query header to ``seq_id`` and parse it into ``(MSA, homolog_count)``."""
    if path is None:
        return None, 0
    a3m = Path(path)
    if not a3m.exists():
        return None, 0
    _replace_query_header_in_a3m(a3m, seq_id)
    return _parse_a3m(a3m)


def _remote_paired_group(
    members: list[Mmseqs2HomologySearchQuery],
    group_dir: Path,
    config: Mmseqs2HomologySearchConfig,
    instance: Any,
    script_path: Path,
) -> Mmseqs2HomologySearchResult:
    """Run one paired group through the remote API; fall back to unpaired when no pairing.

    The standalone returns one row-aligned A3M per chain under
    ``paired_msa_paths["0"]``. When pairing finds nothing (every chain query-only),
    a second unpaired call supplies per-chain ``msas`` and ``paired_msas`` stays
    ``[None, ...]`` — the same fallback the local paired path uses. Remote never
    computes separate unpaired MSAs alongside a successful pairing, so on success
    ``msas`` is ``[None, ...]`` and consumers read ``paired_msas``.
    """
    sequence_ids: list[str] = []
    for query in members:
        assert query.sequence_id is not None  # populated by the input validator
        sequence_ids.append(query.sequence_id)

    out = _dispatch_remote([{"sequences": [q.sequence for q in members]}], group_dir, config, instance, script_path)
    if not out.get("success", False):
        raise RuntimeError(f"mmseqs2-homology-search: remote paired search failed: {out.get('errors')}")

    chain_paths = out.get("paired_msa_paths", {}).get("0", [])
    if len(chain_paths) != len(members):
        raise RuntimeError(
            f"mmseqs2-homology-search: remote paired standalone returned {len(chain_paths)} chain MSA path(s), "
            f"expected {len(members)} (one per chain)."
        )

    paired_msas: list[MSA | None] = []
    paired_homologs: list[int] = []
    for query, path in zip(members, chain_paths, strict=True):
        msa, homologs = _parse_remote_a3m(path, query.sequence_id)  # type: ignore[arg-type]
        paired_msas.append(msa)
        paired_homologs.append(homologs)

    # Any pairing => return the row-aligned paired MSAs (remote computes no separate unpaired set).
    if any(m is not None for m in paired_msas):
        return Mmseqs2HomologySearchResult(
            sequence_ids=sequence_ids,
            msas=[None] * len(members),
            paired_msas=paired_msas,
            datasets_searched=[_REMOTE_DATASET_LABEL],
            num_homologs_found=paired_homologs,
        )

    # No shared taxonomy: a second unpaired call supplies per-chain MSAs; paired_msas stays None.
    fb = _dispatch_remote(
        [{"sequences": q.sequence} for q in members], group_dir / "unpaired_fallback", config, instance, script_path
    )
    fb_paths = fb.get("msa_paths", {})
    msas: list[MSA | None] = []
    homolog_counts: list[int] = []
    for chain_idx, query in enumerate(members):
        msa, homologs = _parse_remote_a3m(fb_paths.get(str(chain_idx)), query.sequence_id)  # type: ignore[arg-type]
        msas.append(msa)
        homolog_counts.append(homologs)
    return Mmseqs2HomologySearchResult(
        sequence_ids=sequence_ids,
        msas=msas,
        paired_msas=[None] * len(members),
        datasets_searched=[_REMOTE_DATASET_LABEL],
        num_homologs_found=homolog_counts,
    )


def _run_remote_homology_search(
    inputs: Mmseqs2HomologySearchInput,
    config: Mmseqs2HomologySearchConfig,
    instance: Any,
) -> Mmseqs2HomologySearchOutput:
    """Generate MSAs via the ColabFold remote API (no local DB / GPU / provisioning).

    Singletons share one batched unpaired API call; each paired (multi-chain)
    group makes its own ``use_pairing=True`` call (see ``_remote_paired_group``).
    ``dataset``/``use_gpu``/``sensitivity`` are ignored in this mode.
    """
    remote_script = Path(__file__).parent / "standalone" / "remote_msa_search.py"

    singletons: list[tuple[int, Mmseqs2HomologySearchQuery]] = []
    paired_groups: list[tuple[int, list[Mmseqs2HomologySearchQuery]]] = []
    for group_idx, group in enumerate(inputs.queries):
        members = group if isinstance(group, list) else [group]
        if isinstance(group, list) and len(members) >= 2:
            paired_groups.append((group_idx, members))
        else:
            singletons.append((group_idx, members[0]))

    results: list[Mmseqs2HomologySearchResult | None] = [None] * len(inputs.queries)
    with tempfile.TemporaryDirectory(prefix="mmseqs2_remote_homology_search_") as tmp_dir_str:
        tmp_root = Path(tmp_dir_str)

        # One batched unpaired API call for all singletons; standalone keys msa_paths by submission index.
        if singletons:
            out = _dispatch_remote(
                [{"sequences": q.sequence} for _, q in singletons],
                tmp_root / "unpaired",
                config,
                instance,
                remote_script,
            )
            if not out.get("success", False) and out.get("num_successful", 0) == 0:
                raise RuntimeError(
                    f"mmseqs2-homology-search: remote search failed for all queries: {out.get('errors')}"
                )
            msa_paths = out.get("msa_paths", {})
            for batch_idx, (group_idx, query) in enumerate(singletons):
                assert query.sequence_id is not None  # populated by the input validator
                msa, homologs = _parse_remote_a3m(msa_paths.get(str(batch_idx)), query.sequence_id)
                results[group_idx] = Mmseqs2HomologySearchResult(
                    sequence_ids=[query.sequence_id],
                    msas=[msa],
                    paired_msas=[None],
                    datasets_searched=[_REMOTE_DATASET_LABEL],
                    num_homologs_found=[homologs],
                )

        # One paired API call per multi-chain group.
        for group_idx, members in paired_groups:
            results[group_idx] = _remote_paired_group(
                members, tmp_root / f"paired_{group_idx}", config, instance, remote_script
            )

    assert all(r is not None for r in results), "internal: missing result for at least one group"
    return Mmseqs2HomologySearchOutput(results=[r for r in results if r is not None])


# ============================================================================
# Helpers
# ============================================================================


def _parse_a3m(a3m_path: Path | None) -> tuple[MSA | None, int]:
    """Parse an A3M into an ``(MSA, homolog_count)`` pair.

    Returns ``(None, 0)`` when the file is missing or holds only the query row
    (no homologs); otherwise the parsed MSA and its homolog count (rows beyond
    the query).
    """
    if a3m_path is None or not a3m_path.exists():
        return None, 0
    num_seqs = _count_sequences_in_a3m(a3m_path)
    if num_seqs < 2:
        return None, 0
    return MSA.from_file(str(a3m_path)), num_seqs - 1


def _assemble_paired_result(
    group_dir: Path,
    members: list[Mmseqs2HomologySearchQuery],
    dataset_name: str,
) -> Mmseqs2HomologySearchResult:
    """Build a per-group result from a paired search's per-chain A3M files.

    The standalone unpacks one ``{i}.a3m`` (unpaired) and ``{i}.paired.a3m``
    (row-aligned paired) per chain, keyed by chain index ``i``. Each file's
    query header is rewritten to the chain's sequence_id before parsing. Both the
    unpaired ``msas`` and ``paired_msas`` are returned, so a consumer can fall back
    to the unpaired MSAs when pairing found nothing (all ``paired_msas`` None).
    """
    sequence_ids: list[str] = []
    msas: list[MSA | None] = []
    paired_msas: list[MSA | None] = []
    num_homologs: list[int] = []
    for chain_idx, query in enumerate(members):
        assert query.sequence_id is not None  # populated by the input validator
        sequence_ids.append(query.sequence_id)
        unpaired_path = group_dir / f"{chain_idx}.a3m"
        paired_path = group_dir / f"{chain_idx}.paired.a3m"
        for path in (unpaired_path, paired_path):
            if path.exists():
                _replace_query_header_in_a3m(path, query.sequence_id)
        unpaired_msa, homologs = _parse_a3m(unpaired_path)
        paired_msa, _ = _parse_a3m(paired_path)
        # Defensive: catch a corrupt standalone output before mis-keyed MSAs reach a predictor.
        for kind, msa in (("unpaired", unpaired_msa), ("paired", paired_msa)):
            if msa is not None and msa.original_sequences[0] != query.sequence:
                raise RuntimeError(
                    f"{kind} MSA for chain {chain_idx} ({query.sequence_id!r}) has query row "
                    f"{msa.original_sequences[0]!r}, expected {query.sequence!r}"
                )
        msas.append(unpaired_msa)
        paired_msas.append(paired_msa)
        num_homologs.append(homologs)

    # Returns unpaired ``msas`` and ``paired_msas`` separately, so a consumer falls back to msas when pairing found nothing.
    return Mmseqs2HomologySearchResult(
        sequence_ids=sequence_ids,
        msas=msas,
        paired_msas=paired_msas,
        datasets_searched=[dataset_name],
        num_homologs_found=num_homologs,
    )


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
        raise FileNotFoundError(
            f"Dataset {name!r} not provisioned on disk: expected {cache_dir}.\n"
            "Provision with:\n"
            f"  python -m proto_tools.tools.sequence_alignment.mmseqs2.setup_databases {name}\n"
            "(see proto_tools/tools/sequence_alignment/mmseqs2/README.md)"
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
