"""Database retrieval tools for biological sequences and structures."""

# AlphaFold DB fetch
from proto_tools.tools.database_retrieval.alphafold_db import (
    AlphaFoldDBFetchConfig,
    AlphaFoldDBFetchInput,
    AlphaFoldDBFetchOutput,
    run_alphafold_db_fetch,
)

# AlphaMissense fetch
from proto_tools.tools.database_retrieval.alphamissense import (
    AlphaMissenseClass,
    AlphaMissenseFetchConfig,
    AlphaMissenseFetchInput,
    AlphaMissenseFetchOutput,
    AlphaMissensePrediction,
    run_alphamissense_fetch,
)

# NCBI Entrez tools
from proto_tools.tools.database_retrieval.ncbi import (
    NCBIEfetchConfig,
    NCBIEfetchInput,
    NCBIEfetchOutput,
    NCBIEsearchConfig,
    NCBIEsearchInput,
    NCBIEsearchOutput,
    NCBIEsummaryConfig,
    NCBIEsummaryInput,
    NCBIEsummaryOutput,
    NCBIFastaRecord,
    NCBIFetchConfig,
    run_ncbi_efetch,
    run_ncbi_esearch,
    run_ncbi_esummary,
)

# PDB tools
from proto_tools.tools.database_retrieval.pdb import (
    PdbChain,
    PdbFetchConfig,
    PdbFetchEntryConfig,
    PdbFetchEntryInput,
    PdbFetchEntryOutput,
    PdbFetchFastaConfig,
    PdbFetchFastaInput,
    PdbFetchFastaOutput,
    run_pdb_fetch_entry,
    run_pdb_fetch_fasta,
)

# PubChem fetch
from proto_tools.tools.database_retrieval.pubchem import (
    PubChemFetchConfig,
    PubChemFetchInput,
    PubChemFetchOutput,
    PubChemProperty,
    run_pubchem_fetch,
)

# Multi-source sequence fetch (orchestrator)
from proto_tools.tools.database_retrieval.sequence_fetch import (
    FetchedSequence,
    FetchedStructure,
    SequenceFetchConfig,
    SequenceFetchInput,
    SequenceFetchOutput,
    SequenceFetchRequest,
    SequenceFetchResult,
    run_sequence_fetch,
)

# UniProt fetch
from proto_tools.tools.database_retrieval.uniprot import (
    UniProtFetchConfig,
    UniProtFetchInput,
    UniProtFetchOutput,
    run_uniprot_fetch,
)

__all__ = [
    # AlphaFold DB fetch
    "AlphaFoldDBFetchConfig",
    "AlphaFoldDBFetchInput",
    "AlphaFoldDBFetchOutput",
    "run_alphafold_db_fetch",
    # AlphaMissense fetch
    "AlphaMissenseClass",
    "AlphaMissenseFetchConfig",
    "AlphaMissenseFetchInput",
    "AlphaMissenseFetchOutput",
    "AlphaMissensePrediction",
    "run_alphamissense_fetch",
    # PubChem fetch
    "PubChemFetchConfig",
    "PubChemFetchInput",
    "PubChemFetchOutput",
    "PubChemProperty",
    "run_pubchem_fetch",
    # NCBI Entrez tools
    "NCBIFastaRecord",
    "NCBIFetchConfig",
    "NCBIEsearchConfig",
    "NCBIEsearchInput",
    "NCBIEsearchOutput",
    "run_ncbi_esearch",
    "NCBIEsummaryConfig",
    "NCBIEsummaryInput",
    "NCBIEsummaryOutput",
    "run_ncbi_esummary",
    "NCBIEfetchConfig",
    "NCBIEfetchInput",
    "NCBIEfetchOutput",
    "run_ncbi_efetch",
    # UniProt fetch
    "UniProtFetchConfig",
    "UniProtFetchInput",
    "UniProtFetchOutput",
    "run_uniprot_fetch",
    # PDB tools
    "PdbChain",
    "PdbFetchConfig",
    "PdbFetchEntryConfig",
    "PdbFetchEntryInput",
    "PdbFetchEntryOutput",
    "run_pdb_fetch_entry",
    "PdbFetchFastaConfig",
    "PdbFetchFastaInput",
    "PdbFetchFastaOutput",
    "run_pdb_fetch_fasta",
    # Sequence fetch (orchestrator)
    "FetchedSequence",
    "FetchedStructure",
    "SequenceFetchConfig",
    "SequenceFetchInput",
    "SequenceFetchOutput",
    "SequenceFetchRequest",
    "SequenceFetchResult",
    "run_sequence_fetch",
]
