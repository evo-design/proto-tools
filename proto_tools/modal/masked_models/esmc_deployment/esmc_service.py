"""ESM C Modal service.

Delegates to proto-tools ``run_esmc_embeddings`` and ``run_esmc_sae_features`` for parameter
validation, environment setup, and model inference. The build-time warmup creates the tool env
and downloads the default checkpoint and the sparse autoencoder, all persisted on the Modal
volume via ``PROTO_MODEL_CACHE``. Non-default checkpoints download on first use.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SCALEDOWN_WINDOW, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import GPU_BASE, with_proto_tools
from proto_tools.modal.gpu_profiles import GPU_DEFAULT
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, ensure_gpu_ready, env_for, run_tool_call


def _warmup() -> None:
    """Deploy-time: warm the default checkpoint and stage the sparse autoencoder."""
    from proto_tools.tools.masked_models.esmc import esmc_embeddings, esmc_sae_features

    esmc_embeddings.run_esmc_embeddings(esmc_embeddings.example_input(), esmc_embeddings.ESMCEmbeddingsConfig())
    esmc_sae_features.run_esmc_sae_features(
        esmc_sae_features.example_input(), esmc_sae_features.ESMCSAEFeaturesConfig()
    )


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("ESMCService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["ESMCService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools(
    {
        "esmc-embedding": "inference",
        "esmc-sae-features": "sae_features",
    }
)
class ESMCService:
    """Modal service for ESM C masked language model inference and sparse-autoencoder features."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("esmc")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("esmc")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def inference(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run ESM C embedding inference on protein sequences.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Mean embeddings, optional logits, and attention masks.
        """
        from proto_tools.tools.masked_models.esmc.esmc_embeddings import (
            ESMCEmbeddingsConfig,
            ESMCEmbeddingsInput,
            run_esmc_embeddings,
        )

        return run_tool_call(
            run_esmc_embeddings,
            ESMCEmbeddingsInput,
            ESMCEmbeddingsConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def sae_features(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Extract sparse-autoencoder features from ESM C representations.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-sequence sparse feature activations.
        """
        from proto_tools.tools.masked_models.esmc.esmc_sae_features import (
            ESMCSAEFeaturesConfig,
            ESMCSAEFeaturesInput,
            run_esmc_sae_features,
        )

        return run_tool_call(
            run_esmc_sae_features,
            ESMCSAEFeaturesInput,
            ESMCSAEFeaturesConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )
