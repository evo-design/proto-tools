"""UniProt protein record retrieval and search."""

from proto_tools.tools.database_retrieval.uniprot.uniprot_fetch import (
    UniProtFetchConfig,
    UniProtFetchInput,
    UniProtFetchOutput,
    run_uniprot_fetch,
)
from proto_tools.tools.database_retrieval.uniprot.uniprot_search import (
    UniProtSearchConfig,
    UniProtSearchHit,
    UniProtSearchInput,
    UniProtSearchOutput,
    UniProtSearchResult,
    run_uniprot_search,
)

__all__ = [
    "UniProtFetchConfig",
    "UniProtFetchInput",
    "UniProtFetchOutput",
    "run_uniprot_fetch",
    "UniProtSearchConfig",
    "UniProtSearchHit",
    "UniProtSearchInput",
    "UniProtSearchOutput",
    "UniProtSearchResult",
    "run_uniprot_search",
]
