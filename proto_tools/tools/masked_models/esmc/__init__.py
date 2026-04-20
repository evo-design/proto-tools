"""ESM C (Cambrian) protein language model — embedding-focused."""

from proto_tools.tools.masked_models.esmc.esmc_embeddings import (
    ESMCEmbeddingsConfig,
    ESMCEmbeddingsInput,
    ESMCEmbeddingsOutput,
    run_esmc_embeddings,
)

__all__ = [
    "ESMCEmbeddingsInput",
    "ESMCEmbeddingsConfig",
    "ESMCEmbeddingsOutput",
    "run_esmc_embeddings",
]
