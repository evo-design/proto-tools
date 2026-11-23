"""Registry of downloadable datasets: reference genomes, MMseqs2 search databases, and the like.

Every dataset is a set of files to download plus a recipe that prepares them, described by a
:class:`DatasetEntry`. Kinds of dataset that a tool consumes in its own way subclass it, as
:class:`~proto_tools.databases.mmseqs2.MMseqs2DatabaseEntry` does for MMseqs2 search databases.
Datasets are cached under the databases root (``$PROTO_DATABASES_DIR``, else
``$PROTO_MODEL_CACHE/databases/``); see :func:`get_databases_root`. Provisioning them lives in
:mod:`proto_tools.databases.provision`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar, Literal, TypeVar, overload

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
        expected_bytes (int | None): Size of the completed file. When set, a
            cached file of a different size is discarded and fetched again.
        sha256 (str | None): Optional SHA-256 of the downloaded file, checked
            once after a download completes.
        required (bool): When False, download failure is non-fatal
            (e.g. taxonomy files needed only for paired-MSA workflows).
    """

    model_config = ConfigDict(extra="forbid")

    url: str = Field(description="HTTPS URL to fetch from")
    filename: str = Field(description="Local filename under the dataset cache dir")
    expected_bytes: int | None = Field(default=None, description="Size of the completed file")
    sha256: str | None = Field(default=None, description="Optional SHA-256 checksum")
    required: bool = Field(default=True, description="Whether failure is fatal")


class BuildStep(BaseModel):
    """One command run in the dataset's cache directory after its downloads complete.

    Attributes:
        command (list[str]): Argv for the step. Template placeholders are
            substituted at runtime: ``{name}`` → the dataset's slug, and
            ``{split_memory_limit}`` → a cgroup-aware ``mmseqs`` memory cap
            (use it on memory-hungry steps like ``createindex``).
        description (str): One-line explanation for logs / error messages.
    """

    model_config = ConfigDict(extra="forbid")

    command: list[str] = Field(description="Argv (supports {name} and {split_memory_limit} placeholders)")
    description: str = Field(description="Human-readable step description")


class BuildRecipe(BaseModel):
    """How to turn a dataset's downloads into the files its consumers read.

    Attributes:
        steps (list[BuildStep]): Commands to run in order (decompress, ``mmseqs createdb``, etc.).
        output_files (list[str]): Files whose presence marks the dataset as provisioned.
            Supports ``{name}`` substitution and glob patterns.
    """

    model_config = ConfigDict(extra="forbid")

    steps: list[BuildStep] = Field(default_factory=list)
    output_files: list[str] = Field(min_length=1)


class DatasetEntry(BaseModel):
    """One downloadable dataset.

    Attributes:
        name (str): Registry key, kebab-case (e.g. ``"uniref30-2302"``).
        molecule_type (Literal["protein", "rna", "dna"]): Sequence type.
        display_name (str): Human-readable name for UI.
        description (str): One-line description for UI.
        citation_doi (str | None): Paper DOI if applicable.
        downloads (list[DownloadSpec]): Files to fetch.
        total_download_bytes (int): Sum of all ``downloads`` sizes for precheck.
        total_disk_bytes (int): Size on disk once built.
        build (BuildRecipe): Steps that prepare the downloads, and the files they produce.
        auto_provision (bool): Provision on first use instead of requiring the
            ``setup_databases`` CLI. Reserved for datasets small enough to fetch unasked.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    molecule_type: Literal["protein", "rna", "dna"]
    display_name: str
    description: str
    citation_doi: str | None = None

    downloads: list[DownloadSpec]
    total_download_bytes: int
    total_disk_bytes: int

    build: BuildRecipe
    auto_provision: bool = Field(
        default=False,
        description=(
            "When True, the dataset is downloaded and built on first use instead of erroring with "
            "the ``setup_databases`` hint. Reserved for datasets small enough to fetch unasked; "
            "production search databases are too large to provision implicitly."
        ),
    )


# ============================================================================
# Registry
# ============================================================================

EntryT = TypeVar("EntryT", bound=DatasetEntry)


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
            raise ValueError(f"DatasetRegistry: dataset {entry.name!r} already registered")
        cls._entries[entry.name] = entry

    @overload
    @classmethod
    def get(cls, name: str) -> DatasetEntry: ...

    @overload
    @classmethod
    def get(cls, name: str, entry_type: type[EntryT]) -> EntryT: ...

    @classmethod
    def get(cls, name: str, entry_type: type[DatasetEntry] = DatasetEntry) -> DatasetEntry:
        """Return the entry for ``name``, requiring it to be an ``entry_type``.

        Args:
            name (str): Registered dataset key.
            entry_type (type[DatasetEntry]): The kind of dataset the caller can use.

        Returns:
            DatasetEntry: The registered entry, an instance of ``entry_type``.

        Raises:
            KeyError: If ``name`` is not registered.
            ValueError: If the entry is not an ``entry_type``.
        """
        if name not in cls._entries:
            available = ", ".join(sorted(cls._entries)) or "<none>"
            raise KeyError(f"Unknown dataset {name!r}. Registered: {available}")
        entry = cls._entries[name]
        if not isinstance(entry, entry_type):
            raise ValueError(f"Dataset {name!r} is not an {entry_type.__name__}")
        return entry

    @classmethod
    def list_all(cls, entry_type: type[DatasetEntry] = DatasetEntry) -> list[str]:
        """Return the sorted names of registered datasets that are an ``entry_type``."""
        return sorted(name for name, entry in cls._entries.items() if isinstance(entry, entry_type))


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

    Datasets are read-only once built, so NFS-mounting the root shares them
    across users (same pattern as ``PROTO_MODEL_CACHE`` for weights).
    """
    explicit = os.environ.get("PROTO_DATABASES_DIR")
    if explicit:
        return Path(explicit)

    from proto_tools.utils.proto_home import get_proto_home

    model_cache = os.environ.get("PROTO_MODEL_CACHE") or str(get_proto_home() / "proto_model_cache")
    return Path(model_cache) / "databases"


def dataset_slug(name: str) -> str:
    """Convert a registry name to its filesystem slug.

    Kebab-case → snake_case (e.g. ``"uniref30-2302"`` → ``"uniref30_2302"``).
    The slug is used both for the cache directory name and for ``{name}``
    placeholder substitution in :class:`BuildStep` commands and
    :attr:`BuildRecipe.output_files`. Single source of truth so the
    convention can't drift between the registry, the provisioning CLI,
    and tests.
    """
    return name.replace("-", "_")


def get_dataset_dir(name: str) -> Path:
    """Return the on-disk cache directory for a registered dataset.

    Resolves to ``<databases_root>/{name_slug}/`` (see :func:`get_databases_root`
    for how the root is chosen) where ``name_slug`` is :func:`dataset_slug` of the
    dataset's name.

    The directory may not exist yet — this is a pure path helper, not a
    provisioner; see :mod:`proto_tools.databases.provision` for that.

    Args:
        name (str): Registered dataset key (kebab-case, e.g. ``"uniref30-2302"``).

    Returns:
        Path: Absolute path to the dataset's cache directory.

    Raises:
        KeyError: If ``name`` is not a registered dataset.
    """
    entry = DatasetRegistry.get(name)
    return get_databases_root() / dataset_slug(entry.name)
