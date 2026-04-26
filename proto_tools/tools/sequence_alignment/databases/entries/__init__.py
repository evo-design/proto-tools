"""Dataset entries — one module per registered homology database.

Each module defines a ``DatasetEntry`` and calls
``DatasetRegistry.register(ENTRY)`` at import time. Importing this package
triggers registration of every entry.
"""

from proto_tools.tools.sequence_alignment.databases.entries import uniref30_2302  # noqa: F401

__all__: list[str] = []
