"""AlphaMissense pathogenicity score retrieval (human variants)."""

from proto_tools.tools.database_retrieval.alphamissense_db.alphamissense_db_fetch import (
    AlphaMissenseClass,
    AlphaMissenseDBFetchConfig,
    AlphaMissenseDBFetchInput,
    AlphaMissenseDBFetchOutput,
    AlphaMissensePrediction,
    run_alphamissense_db_fetch,
)

__all__ = [
    "AlphaMissenseClass",
    "AlphaMissenseDBFetchConfig",
    "AlphaMissenseDBFetchInput",
    "AlphaMissenseDBFetchOutput",
    "AlphaMissensePrediction",
    "run_alphamissense_db_fetch",
]
