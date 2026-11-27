"""MMseqs2 search databases: the dataset kind the MMseqs2 toolkit searches.

Defined here rather than in the toolkit because entries in :mod:`proto_tools.databases.entries`
import it, and the registry must not depend on a tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from proto_tools.databases.registry import DatasetEntry


class MMseqs2SearchFlags(BaseModel):
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


class MMseqs2DatabaseEntry(DatasetEntry):
    """A dataset built into an MMseqs2 database that the MMseqs2 tools search.

    Attributes:
        db_prefix (str): Stem of the built database in the cache directory
            (e.g. ``"uniref30_2302_db"`` → ``{cache_dir}/uniref30_2302_db*``).
        gpu_db (str | None): Stem that GPU search (``--gpu 1``) reads, or None for a CPU-only
            database. Equal to ``db_prefix`` when the database is itself padded (AF3-style);
            a separate ``<db_prefix>.idx_pad`` sibling for ColabFold-style databases.
        a3m_adapter (Literal["colabfold", "plain", "rna"]): Which A3M-writer convention
            ``mmseqs2-homology-search`` uses to stitch hits into an A3M for this database.
        search_flags (MMseqs2SearchFlags): Search-time MMseqs2 parameters.
    """

    db_prefix: str
    gpu_db: str | None = None
    a3m_adapter: Literal["colabfold", "plain", "rna"]
    search_flags: MMseqs2SearchFlags = Field(default_factory=MMseqs2SearchFlags)

    def search_db(self, cache_dir: Path, *, gpu: bool) -> Path:
        """Return the database stem to search, the padded one for GPU search.

        Args:
            cache_dir (Path): The dataset's cache directory.
            gpu (bool): Whether the search runs on GPU.

        Returns:
            Path: The stem MMseqs2 is given as its target database.

        Raises:
            ValueError: If ``gpu`` is set and this database has no GPU index.
        """
        if not gpu:
            return cache_dir / self.db_prefix
        if self.gpu_db is None:
            raise ValueError(f"Dataset {self.name!r} does not support GPU search")
        return cache_dir / self.gpu_db

    def is_built(self, cache_dir: Path, *, gpu: bool) -> bool:
        """Return whether the database, and its GPU index when ``gpu``, is complete on disk.

        Args:
            cache_dir (Path): The dataset's cache directory.
            gpu (bool): Whether to also require the database GPU search reads.

        Returns:
            bool: True when every required MMseqs2 database has its ``.dbtype`` file.
        """
        if not Path(f"{cache_dir / self.db_prefix}.dbtype").is_file():
            return False
        if not gpu or self.gpu_db == self.db_prefix:
            return True
        if self.gpu_db is None:
            return False
        # A separate padded sibling is a database of its own, complete with its data file and .dbtype.
        padded = cache_dir / self.gpu_db
        return padded.exists() and Path(f"{padded}.dbtype").is_file()
