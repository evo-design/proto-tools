"""Sequence alignment and MSA tools."""

from proto_tools.tools.sequence_alignment.blast import (
    BlastHit,
    BlastSearchConfig,
    BlastSearchInput,
    BlastSearchOutput,
    CreateBlastDbConfig,
    CreateBlastDbInput,
    CreateBlastDbOutput,
    run_blast_search,
    run_create_blast_db,
)
from proto_tools.tools.sequence_alignment.colabfold_search import (
    ColabfoldSearchConfig,
    ColabfoldSearchInput,
    ColabfoldSearchOutput,
    run_colabfold_search,
)
from proto_tools.tools.sequence_alignment.mafft import MafftConfig, MafftInput, MafftOutput, run_mafft_align
from proto_tools.tools.sequence_alignment.mmseqs2_homology_search import (
    Mmseqs2HomologySearchConfig,
    Mmseqs2HomologySearchInput,
    Mmseqs2HomologySearchOutput,
    Mmseqs2HomologySearchQuery,
    Mmseqs2HomologySearchResult,
    run_mmseqs2_homology_search,
)
from proto_tools.tools.sequence_alignment.msas import MSA

__all__ = [
    # Schemas
    "MSA",
    # BLAST
    "run_blast_search",
    "run_create_blast_db",
    "BlastHit",
    "BlastSearchInput",
    "BlastSearchConfig",
    "BlastSearchOutput",
    "CreateBlastDbInput",
    "CreateBlastDbConfig",
    "CreateBlastDbOutput",
    # MAFFT
    "run_mafft_align",
    "MafftInput",
    "MafftConfig",
    "MafftOutput",
    # ColabFold Search
    "run_colabfold_search",
    "ColabfoldSearchInput",
    "ColabfoldSearchConfig",
    "ColabfoldSearchOutput",
    # MMseqs2 Homology Search
    "run_mmseqs2_homology_search",
    "Mmseqs2HomologySearchInput",
    "Mmseqs2HomologySearchConfig",
    "Mmseqs2HomologySearchOutput",
    "Mmseqs2HomologySearchQuery",
    "Mmseqs2HomologySearchResult",
]
