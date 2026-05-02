"""Gene annotation and sequence search tools."""

# BLAST tools
from proto_tools.tools.gene_annotation.blast import (
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

# CRISPRtracrRNA tools
from proto_tools.tools.gene_annotation.crispr_tracr_rna import (
    CrisprTracrRNAConfig,
    CrisprTracrRNAInput,
    CrisprTracrRNAOutput,
    CrisprTracrRNAPrediction,
    CrisprTracrRNASequenceResult,
    run_crispr_tracr_rna,
)

# MinCED tools
from proto_tools.tools.gene_annotation.minced import (
    CrisprArray,
    CrisprRepeatSpacer,
    MincedConfig,
    MincedInput,
    MincedOutput,
    MincedSequenceResult,
    run_minced,
)

# MMseqs2 tools
from proto_tools.tools.gene_annotation.mmseqs import (
    MmseqsClusteringConfig,
    MmseqsClusteringInput,
    MmseqsClusteringOutput,
    MmseqsClusterMember,
    MmseqsClusterResult,
    MmseqsHit,
    MmseqsSearchGenomesConfig,
    MmseqsSearchGenomesInput,
    MmseqsSearchGenomesOutput,
    MmseqsSearchProteinsConfig,
    MmseqsSearchProteinsInput,
    MmseqsSearchProteinsOutput,
    MmseqsSequenceSearchResult,
    run_mmseqs_clustering,
    run_mmseqs_search_genomes,
    run_mmseqs_search_proteins,
)

# Promoter Calculator tools
from proto_tools.tools.gene_annotation.promoter_calculator import (
    PromoterCalculatorConfig,
    PromoterCalculatorInput,
    PromoterCalculatorOutput,
    PromoterCalculatorSequenceResult,
    PromoterPrediction,
    run_promoter_calculator,
)

# PyHMMER tools
from proto_tools.tools.gene_annotation.pyhmmer import (
    DomainHit,
    PyHmmerConfig,
    PyHmmscanConfig,
    PyHmmscanInput,
    PyHmmscanOutput,
    PyHmmsearchConfig,
    PyHmmsearchInput,
    PyHmmsearchOutput,
    PyJackhmmerConfig,
    PyJackhmmerInput,
    PyJackhmmerOutput,
    PyNhmmerConfig,
    PyNhmmerInput,
    PyNhmmerOutput,
    PyPhmmerConfig,
    PyPhmmerInput,
    PyPhmmerOutput,
    SequenceHit,
    run_pyhmmer_hmmscan,
    run_pyhmmer_hmmsearch,
    run_pyhmmer_jackhmmer,
    run_pyhmmer_nhmmer,
    run_pyhmmer_phmmer,
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
    # PyHMMER
    "run_pyhmmer_hmmsearch",
    "run_pyhmmer_hmmscan",
    "run_pyhmmer_phmmer",
    "run_pyhmmer_nhmmer",
    "run_pyhmmer_jackhmmer",
    "PyHmmsearchInput",
    "PyHmmsearchConfig",
    "PyHmmsearchOutput",
    "PyHmmscanInput",
    "PyHmmscanConfig",
    "PyHmmscanOutput",
    "PyPhmmerInput",
    "PyPhmmerConfig",
    "PyPhmmerOutput",
    "PyNhmmerInput",
    "PyNhmmerConfig",
    "PyNhmmerOutput",
    "PyJackhmmerInput",
    "PyJackhmmerConfig",
    "PyJackhmmerOutput",
    "SequenceHit",
    "DomainHit",
    "PyHmmerConfig",
    # MMseqs2 schema classes
    "MmseqsHit",
    "MmseqsSequenceSearchResult",
    "MmseqsClusterMember",
    "MmseqsClusterResult",
    # MMseqs2 protein search
    "run_mmseqs_search_proteins",
    "MmseqsSearchProteinsInput",
    "MmseqsSearchProteinsConfig",
    "MmseqsSearchProteinsOutput",
    # MMseqs2 genome search
    "run_mmseqs_search_genomes",
    "MmseqsSearchGenomesInput",
    "MmseqsSearchGenomesConfig",
    "MmseqsSearchGenomesOutput",
    # MMseqs2 clustering
    "run_mmseqs_clustering",
    "MmseqsClusteringInput",
    "MmseqsClusteringConfig",
    "MmseqsClusteringOutput",
    # CRISPRtracrRNA
    "CrisprTracrRNAConfig",
    "CrisprTracrRNAInput",
    "CrisprTracrRNAOutput",
    "CrisprTracrRNAPrediction",
    "CrisprTracrRNASequenceResult",
    "run_crispr_tracr_rna",
    # MinCED
    "run_minced",
    "MincedInput",
    "MincedConfig",
    "MincedOutput",
    "MincedSequenceResult",
    "CrisprArray",
    "CrisprRepeatSpacer",
    # Promoter Calculator
    "run_promoter_calculator",
    "PromoterCalculatorInput",
    "PromoterCalculatorConfig",
    "PromoterCalculatorOutput",
    "PromoterCalculatorSequenceResult",
    "PromoterPrediction",
]
