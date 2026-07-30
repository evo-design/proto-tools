"""ESM C sparse autoencoder features — interpretable decomposition of ESM C activations."""

from proto_tools.tools.masked_models.esmc_sae.esmc_sae_features import (
    ESMCSAEFeaturesConfig,
    ESMCSAEFeaturesInput,
    ESMCSAEFeaturesOutput,
    SAELayerFeatures,
    SequenceSAEFeatures,
    resolve_sae_repo,
    run_esmc_sae_features,
)
from proto_tools.tools.masked_models.esmc_sae.helpers import (
    DESCRIBED_SAE_REPO,
    describe_sae_features,
)

__all__ = [
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
