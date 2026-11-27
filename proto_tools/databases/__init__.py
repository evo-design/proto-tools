"""Registry of downloadable datasets shared across tools (reference genomes, MMseqs2 search databases).

See :mod:`proto_tools.databases.registry` for the base ``DatasetEntry`` schema and
:func:`get_dataset_dir` path helper, :mod:`proto_tools.databases.mmseqs2` for the MMseqs2 search
database subtype, and :mod:`proto_tools.databases.provision` for downloading and building them.
Entries live in :mod:`proto_tools.databases.entries`; importing this package registers all of them.
"""

from proto_tools.databases import entries  # noqa: F401  (registers all entries)
from proto_tools.databases.mmseqs2 import MMseqs2DatabaseEntry, MMseqs2SearchFlags
from proto_tools.databases.registry import (
    BuildRecipe,
    BuildStep,
    DatasetEntry,
    DatasetRegistry,
    DownloadSpec,
    dataset_slug,
    get_databases_root,
    get_dataset_dir,
)

__all__ = [
    "BuildRecipe",
    "BuildStep",
    "DatasetEntry",
    "DatasetRegistry",
    "DownloadSpec",
    "MMseqs2DatabaseEntry",
    "MMseqs2SearchFlags",
    "dataset_slug",
    "get_databases_root",
    "get_dataset_dir",
]
