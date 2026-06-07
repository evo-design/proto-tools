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
from proto_tools.tools.sequence_alignment.mafft import MafftConfig, MafftInput, MafftOutput, run_mafft_align
from proto_tools.tools.sequence_alignment.mmseqs2 import (
    Mmseqs2ClusteringConfig,
    Mmseqs2ClusteringInput,
    Mmseqs2ClusteringOutput,
    Mmseqs2ClusterMember,
    Mmseqs2ClusterResult,
    Mmseqs2Hit,
    Mmseqs2HomologySearchConfig,
    Mmseqs2HomologySearchInput,
    Mmseqs2HomologySearchOutput,
    Mmseqs2HomologySearchQuery,
    Mmseqs2HomologySearchResult,
    Mmseqs2SearchGenomesConfig,
    Mmseqs2SearchGenomesInput,
    Mmseqs2SearchGenomesOutput,
    Mmseqs2SearchProteinsConfig,
    Mmseqs2SearchProteinsInput,
    Mmseqs2SearchProteinsOutput,
    Mmseqs2SequenceSearchResult,
    run_mmseqs2_clustering,
    run_mmseqs2_homology_search,
    run_mmseqs2_search_genomes,
    run_mmseqs2_search_proteins,
)

__all__ = [
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
    # MMseqs2 shared schemas
    "Mmseqs2Hit",
    "Mmseqs2SequenceSearchResult",
    "Mmseqs2ClusterMember",
    "Mmseqs2ClusterResult",
    # MMseqs2 protein search
    "run_mmseqs2_search_proteins",
    "Mmseqs2SearchProteinsInput",
    "Mmseqs2SearchProteinsConfig",
    "Mmseqs2SearchProteinsOutput",
    # MMseqs2 genome search
    "run_mmseqs2_search_genomes",
    "Mmseqs2SearchGenomesInput",
    "Mmseqs2SearchGenomesConfig",
    "Mmseqs2SearchGenomesOutput",
    # MMseqs2 clustering
    "run_mmseqs2_clustering",
    "Mmseqs2ClusteringInput",
    "Mmseqs2ClusteringConfig",
    "Mmseqs2ClusteringOutput",
    # MMseqs2 Homology Search (MSA)
    "run_mmseqs2_homology_search",
    "Mmseqs2HomologySearchInput",
    "Mmseqs2HomologySearchConfig",
    "Mmseqs2HomologySearchOutput",
    "Mmseqs2HomologySearchQuery",
    "Mmseqs2HomologySearchResult",
]
