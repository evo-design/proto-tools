"""UniProt protein record retrieval."""

from proto_tools.tools.database_retrieval.uniprot.uniprot_fetch import (
    UniProtFetchConfig,
    UniProtFetchInput,
    UniProtFetchOutput,
    run_uniprot_fetch,
)

__all__ = [
    "UniProtFetchConfig",
    "UniProtFetchInput",
    "UniProtFetchOutput",
    "run_uniprot_fetch",
]
