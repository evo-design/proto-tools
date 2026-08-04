"""ESM C (Cambrian) protein language model — embeddings and sparse autoencoder features."""

from proto_tools.tools.masked_models.esmc.esmc_embeddings import (
    ESMCEmbeddingsConfig,
    ESMCEmbeddingsInput,
    ESMCEmbeddingsOutput,
    run_esmc_embeddings,
)
from proto_tools.tools.masked_models.esmc.esmc_sae_features import (
    ESMCSAEFeaturesConfig,
    ESMCSAEFeaturesInput,
    ESMCSAEFeaturesOutput,
    SAELayerFeatures,
    SequenceSAEFeatures,
    resolve_sae_repo,
    run_esmc_sae_features,
)
from proto_tools.tools.masked_models.esmc.helpers import (
    DESCRIBED_SAE_REPO,
    describe_sae_features,
)

__all__ = [
    # Embeddings
    "ESMCEmbeddingsInput",
    "ESMCEmbeddingsConfig",
    "ESMCEmbeddingsOutput",
    "run_esmc_embeddings",
    # SAE Features
    "DESCRIBED_SAE_REPO",
    "ESMCSAEFeaturesConfig",
    "ESMCSAEFeaturesInput",
    "ESMCSAEFeaturesOutput",
    "SAELayerFeatures",
    "SequenceSAEFeatures",
    "describe_sae_features",
    "resolve_sae_repo",
    "run_esmc_sae_features",
]
