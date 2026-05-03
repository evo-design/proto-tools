"""AlphaMissense pathogenicity score retrieval (human variants)."""

from proto_tools.tools.database_retrieval.alphamissense.alphamissense_fetch import (
    AlphaMissenseClass,
    AlphaMissenseFetchConfig,
    AlphaMissenseFetchInput,
    AlphaMissenseFetchOutput,
    AlphaMissensePrediction,
    run_alphamissense_fetch,
)

__all__ = [
    "AlphaMissenseClass",
    "AlphaMissenseFetchConfig",
    "AlphaMissenseFetchInput",
    "AlphaMissenseFetchOutput",
    "AlphaMissensePrediction",
    "run_alphamissense_fetch",
]
