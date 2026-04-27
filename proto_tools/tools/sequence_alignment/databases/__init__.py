"""Dataset registry for MMseqs2-based homology search tools.

See :mod:`proto_tools.tools.sequence_alignment.databases.registry` for the
``DatasetEntry`` schema and :func:`get_dataset_dir` path helper. Entries live
in :mod:`proto_tools.tools.sequence_alignment.databases.entries`; importing
this package registers all of them.
"""

from proto_tools.tools.sequence_alignment.databases import entries  # noqa: F401  (registers all entries)
from proto_tools.tools.sequence_alignment.databases.registry import (
    DatasetEntry,
    DatasetRegistry,
    DownloadSpec,
    IndexRecipe,
    IndexStep,
    MmseqsFlags,
    dataset_slug,
    get_databases_root,
    get_dataset_dir,
)

__all__ = [
    "DatasetEntry",
    "DatasetRegistry",
    "DownloadSpec",
    "IndexRecipe",
    "IndexStep",
    "MmseqsFlags",
    "dataset_slug",
    "get_databases_root",
    "get_dataset_dir",
]
