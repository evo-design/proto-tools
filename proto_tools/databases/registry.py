"""Dataset registry for MMseqs2-based homology search tools.

One source of truth for downloadable sequence databases consumed by
``colabfold-search`` and ``mmseqs2-homology-search``. Holds per-dataset
metadata — molecule type, download URLs, index recipe, MMseqs2 flags,
GPU/pairing capability — and resolves the on-disk cache location under
the databases root (``$PROTO_DATABASES_DIR``, else ``$PROTO_MODEL_CACHE/databases/``);
see :func:`get_databases_root`.

This module defines the schemas and a simple lookup registry. Dataset
provisioning is implemented by the MMseqs2 homology-search tooling.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

# ============================================================================
# Schemas
# ============================================================================


class DownloadSpec(BaseModel):
    """One file to fetch as part of a dataset.

    Attributes:
        url (str): HTTPS URL to fetch from.
        filename (str): Local filename under the dataset's cache directory.
            Used for resume and post-download verification.
        sha256 (str | None): Optional SHA-256 of the downloaded file for
            integrity check.
        required (bool): When False, download failure is non-fatal
            (e.g. taxonomy files needed only for paired-MSA workflows).
    """

    model_config = ConfigDict(extra="forbid")

    url: str = Field(description="HTTPS URL to fetch from")
    filename: str = Field(description="Local filename under the dataset cache dir")
    sha256: str | None = Field(default=None, description="Optional SHA-256 checksum")
    required: bool = Field(default=True, description="Whether failure is fatal")


class IndexStep(BaseModel):
    """One command to run after download, in order.

    Each step is an MMseqs2 invocation (or similar) that transforms the raw
    download into usable index files. Steps are resolved in the tool's
    standalone env at ``ensure()`` time.

    Attributes:
        command (list[str]): Argv for the step. Template placeholders are
            substituted at runtime: ``{name}`` → the dataset name, and
            ``{split_memory_limit}`` → a cgroup-aware ``mmseqs`` memory cap in
            bytes (use it on memory-hungry steps like ``createindex``).
        description (str): One-line explanation for logs / error messages.
    """

    model_config = ConfigDict(extra="forbid")

    command: list[str] = Field(description="Argv (supports {name} and {split_memory_limit} placeholders)")
    description: str = Field(description="Human-readable step description")


class IndexRecipe(BaseModel):
    """How to turn downloads into an indexed MMseqs2 database.

    Attributes:
        steps (list[IndexStep]): Commands to run in order (extract, createdb,
            makepaddedseqdb, createtaxdb, etc.).
        output_files (list[str]): Files whose presence marks the dataset as
            indexed. Supports ``{name}`` substitution.
    """

    model_config = ConfigDict(extra="forbid")

    steps: list[IndexStep] = Field(default_factory=list)
    output_files: list[str] = Field(default_factory=list)


class MmseqsFlags(BaseModel):
    """Search-time MMseqs2 parameters, baked into the dataset entry.

    Registry-driven defaults keep the tool layer thin: the generalized
    ``mmseqs2-homology-search`` tool does not branch on molecule type or
    dataset, it just dereferences these flags.

    Attributes:
        sensitivity (float): MMseqs2 ``-s`` parameter (1.0 fast, 7.5 very sensitive).
        prefilter_mode (int): MMseqs2 prefilter mode (0 kmer, 1 ungapped, 2 exhaustive).
        max_seqs (int): Maximum results per query allowed through the prefilter.
        extra_args (list[str]): Escape hatch for dataset-specific quirks.
            Prefer dedicated fields when a pattern recurs.
    """

    model_config = ConfigDict(extra="forbid")

    sensitivity: float = Field(default=8.0, description="MMseqs2 -s sensitivity")
    prefilter_mode: int = Field(default=0, description="0=kmer, 1=ungapped, 2=exhaustive")
    max_seqs: int = Field(default=300, description="Max prefilter results per query")
    extra_args: list[str] = Field(default_factory=list, description="Escape-hatch extra flags")


class DatasetEntry(BaseModel):
    """One searchable homology database.

    Attributes:
        name (str): Registry key, kebab-case (e.g. ``"uniref30-2302"``).
        molecule_type (Literal["protein", "rna", "dna"]): Sequence type.
        display_name (str): Human-readable name for UI.
        description (str): One-line description for UI.
        citation_doi (str | None): Paper DOI if applicable.
        urls (list[DownloadSpec]): Files to fetch.
        total_download_bytes (int): Sum of all ``urls`` sizes for precheck.
        total_disk_bytes (int): Post-extract, post-index size estimate.
        index_recipe (IndexRecipe): Steps to produce the indexed DB.
        mmseqs_flags (MmseqsFlags): Search-time MMseqs2 parameters.
        db_prefix (str): Filename prefix of the final DB files on disk
            (e.g. ``"uniref30_2302_db"`` → ``{cache_dir}/uniref30_2302_db*``).
        supports_gpu (bool): Whether a GPU-padded index is produced
            (``.idx_pad`` file present after indexing).
        min_gpu_memory_gb (float | None): Minimum GPU memory for GPU search.
            None when the dataset is CPU-only or negligible.
        gpu_padded_marker (str | None): Filename whose presence signals the
            GPU-padded index has been built. Set this for ColabFold-style
            entries that produce a separate ``<db_prefix>.idx_pad`` sibling.
            Leave ``None`` for AF3-style entries where ``db_prefix`` is itself
            the padded DB (i.e. ``mmseqs makepaddedseqdb`` writes there
            directly), in which case ``<db_prefix>.dbtype`` is sufficient.
        a3m_adapter (Literal["colabfold", "plain", "rna"]): Which A3M-writer
            convention the ``mmseqs2-homology-search`` tool uses to stitch
            m8 hits into an A3M for this dataset.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    molecule_type: Literal["protein", "rna", "dna"]
    display_name: str
    description: str
    citation_doi: str | None = None

    urls: list[DownloadSpec]
    total_download_bytes: int
    total_disk_bytes: int

    index_recipe: IndexRecipe
    mmseqs_flags: MmseqsFlags = Field(default_factory=MmseqsFlags)

    db_prefix: str
    supports_gpu: bool
    min_gpu_memory_gb: float | None = None
    gpu_padded_marker: str | None = Field(
        default=None,
        description=(
            "Filename signaling the GPU-padded index has been built. "
            "Set for ColabFold-style entries (separate <db_prefix>.idx_pad "
            "sibling); leave None for AF3-style entries where db_prefix is "
            "itself the padded DB."
        ),
    )
    a3m_adapter: Literal["colabfold", "plain", "rna"] = "colabfold"
    auto_provision: bool = Field(
        default=False,
        description=(
            "When True, the tool layer transparently runs the entry's download + "
            "``index_recipe.steps`` on first dispatch instead of erroring with the "
            "``setup_databases.py`` hint. Reserved for small test/fixture entries "
            "(typically a few hundred MB or less); production datasets are too "
            "large to provision implicitly."
        ),
    )


# ============================================================================
# Registry
# ============================================================================


class DatasetRegistry:
    """Lookup API over the set of registered dataset entries.

    Entries self-register at import time by calling :meth:`register` from
    ``proto_tools.databases.entries.*`` modules.
    """

    _entries: ClassVar[dict[str, DatasetEntry]] = {}

    @classmethod
    def register(cls, entry: DatasetEntry) -> None:
        """Register a dataset entry. Raises on duplicate ``name``."""
        if entry.name in cls._entries:
            raise ValueError(
                f"DatasetRegistry: dataset {entry.name!r} already registered "
                f"(existing molecule_type={cls._entries[entry.name].molecule_type!r})"
            )
        cls._entries[entry.name] = entry

    @classmethod
    def get(cls, name: str) -> DatasetEntry:
        """Return the entry for ``name``. Raises ``KeyError`` if missing."""
        if name not in cls._entries:
            available = ", ".join(sorted(cls._entries)) or "<none>"
            raise KeyError(f"Unknown dataset {name!r}. Registered: {available}")
        return cls._entries[name]

    @classmethod
    def list_all(cls) -> list[str]:
        """All registered dataset names, sorted."""
        return sorted(cls._entries)

    @classmethod
    def by_molecule_type(cls, molecule_type: Literal["protein", "rna", "dna"]) -> list[DatasetEntry]:
        """All entries of the given molecule type."""
        return [e for e in cls._entries.values() if e.molecule_type == molecule_type]


# ============================================================================
# Path resolution
# ============================================================================


def get_databases_root() -> Path:
    """Return the root directory holding provisioned datasets.

    Resolution order:

    1. ``PROTO_DATABASES_DIR`` — used verbatim as the databases root. Lets large
       datasets live on a separate (e.g. scratch or shared) filesystem from model
       weights, and applies to both provisioning and runtime resolution.
    2. ``$PROTO_MODEL_CACHE/databases/`` — co-located with model weights.
    3. ``$PROTO_HOME/proto_model_cache/databases/`` — the all-in-one default.

    Datasets are read-only once indexed, so NFS-mounting the root shares them
    across users (same pattern as ``PROTO_MODEL_CACHE`` for weights).
    """
    explicit = os.environ.get("PROTO_DATABASES_DIR")
    if explicit:
        return Path(explicit)

    from proto_tools.utils.proto_home import get_proto_home

    model_cache = os.environ.get("PROTO_MODEL_CACHE") or str(get_proto_home() / "proto_model_cache")
    return Path(model_cache) / "databases"


def dataset_slug(name: str) -> str:
    """Convert a registry name to its filesystem / MMseqs filename slug.

    Kebab-case → snake_case (e.g. ``"uniref30-2302"`` → ``"uniref30_2302"``).
    The slug is used both for the cache directory name and for ``{name}``
    placeholder substitution in :class:`IndexStep` commands and
    :attr:`IndexRecipe.output_files`. Single source of truth so the
    convention can't drift between the registry, the provisioning CLI,
    and tests.
    """
    return name.replace("-", "_")


def get_dataset_dir(name: str) -> Path:
    """Return the on-disk cache directory for a registered dataset.

    Resolves to ``<databases_root>/{name_slug}/`` (see :func:`get_databases_root`
    for how the root is chosen) where ``name_slug`` is :func:`dataset_slug` of the
    dataset's name (kebab-case with ``-`` replaced by ``_``, to match MMseqs2
    filename conventions).

    The directory may not exist yet — this is a pure path helper, not an
    ``ensure``. Materialization belongs to ``DatasetManager.ensure`` (TBD,
    shipping with ``mmseqs2-homology-search``).

    Args:
        name (str): Registered dataset key (kebab-case, e.g. ``"uniref30-2302"``).

    Returns:
        Path: Absolute path to the dataset's cache directory.

    Raises:
        KeyError: If ``name`` is not a registered dataset.
    """
    entry = DatasetRegistry.get(name)
    return get_databases_root() / dataset_slug(entry.name)
