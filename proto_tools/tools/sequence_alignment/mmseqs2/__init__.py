"""MMseqs2 toolkit: protein/genome search, clustering, and MSA-style homology search.

Four registered tools backed by a single MMseqs2 install (and ColabFold for the
homology-search MSA pipeline):

- ``mmseqs2-search-proteins`` — protein-vs-database search via ``mmseqs easy-search``.
- ``mmseqs2-search-genomes`` — nucleotide genome-to-genome search workflow.
- ``mmseqs2-clustering`` — sequence clustering via greedy set-cover.
- ``mmseqs2-homology-search`` — MSA generation for structure predictors (the
  ColabFold-style iterative pipeline).
"""

from proto_tools.tools.sequence_alignment.mmseqs2.clustering import (
    Mmseqs2ClusteringConfig,
    Mmseqs2ClusteringInput,
    Mmseqs2ClusteringOutput,
    Mmseqs2ClusterMember,
    Mmseqs2ClusterResult,
    run_mmseqs2_clustering,
)
from proto_tools.tools.sequence_alignment.mmseqs2.homology_search import (
    Mmseqs2HomologySearchConfig,
    Mmseqs2HomologySearchInput,
    Mmseqs2HomologySearchOutput,
    Mmseqs2HomologySearchQuery,
    Mmseqs2HomologySearchResult,
    run_mmseqs2_homology_search,
)
from proto_tools.tools.sequence_alignment.mmseqs2.search_genomes import (
    Mmseqs2SearchGenomesConfig,
    Mmseqs2SearchGenomesInput,
    Mmseqs2SearchGenomesOutput,
    run_mmseqs2_search_genomes,
)
from proto_tools.tools.sequence_alignment.mmseqs2.search_proteins import (
    Mmseqs2Hit,
    Mmseqs2SearchProteinsConfig,
    Mmseqs2SearchProteinsInput,
    Mmseqs2SearchProteinsOutput,
    Mmseqs2SequenceSearchResult,
    run_mmseqs2_search_proteins,
)

__all__ = [
    # Shared models
    "Mmseqs2Hit",
    "Mmseqs2SequenceSearchResult",
    "Mmseqs2ClusterMember",
    "Mmseqs2ClusterResult",
    # Protein search
    "run_mmseqs2_search_proteins",
    "Mmseqs2SearchProteinsInput",
    "Mmseqs2SearchProteinsConfig",
    "Mmseqs2SearchProteinsOutput",
    # Genome search
    "run_mmseqs2_search_genomes",
    "Mmseqs2SearchGenomesInput",
    "Mmseqs2SearchGenomesConfig",
    "Mmseqs2SearchGenomesOutput",
    # Clustering
    "run_mmseqs2_clustering",
    "Mmseqs2ClusteringInput",
    "Mmseqs2ClusteringConfig",
    "Mmseqs2ClusteringOutput",
    # Homology search (MSA)
    "run_mmseqs2_homology_search",
    "Mmseqs2HomologySearchInput",
    "Mmseqs2HomologySearchConfig",
    "Mmseqs2HomologySearchOutput",
    "Mmseqs2HomologySearchQuery",
    "Mmseqs2HomologySearchResult",
]
