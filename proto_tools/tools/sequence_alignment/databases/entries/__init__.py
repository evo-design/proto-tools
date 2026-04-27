"""Dataset entries — one module per registered homology database.

Each module defines a ``DatasetEntry`` and calls
``DatasetRegistry.register(ENTRY)`` at import time. Importing this package
triggers registration of every entry.
"""

# All entries below are import-for-side-effect: each module registers itself
# with DatasetRegistry at import time.
from proto_tools.tools.sequence_alignment.databases.entries import (
    colabfold_envdb_202108,  # noqa: F401
    mgnify_2022_05,  # noqa: F401
    nt_rna_2023_02_23_90_80,  # noqa: F401
    pdb_seqres_2022_09_28,  # noqa: F401
    rfam_14_9_90_80,  # noqa: F401
    rnacentral_active_90_80,  # noqa: F401
    small_bfd,  # noqa: F401
    uniprot_2021_04,  # noqa: F401
    uniref30_2302,  # noqa: F401
    uniref90_2022_05,  # noqa: F401
)

__all__: list[str] = []
