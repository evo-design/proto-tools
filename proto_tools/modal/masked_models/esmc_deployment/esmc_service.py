"""ESM C Modal service.

Delegates to proto-tools ``run_esmc_embeddings`` for parameter validation,
environment setup, and model inference. The build-time warmup creates the
tool env and downloads the default checkpoint, both persisted on the
Modal volume via ``PROTO_MODEL_CACHE``. Non-default checkpoints download
on first use.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SCALEDOWN_WINDOW, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import GPU_BASE, with_proto_tools
from proto_tools.modal.gpu_profiles import GPU_DEFAULT
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, dispatch_tool_call, ensure_gpu_ready, env_for


def _warmup() -> None:
    """Deploy-time: warm only the default checkpoint."""
    from proto_tools.tools.masked_models.esmc.esmc_embeddings import (
        ESMCEmbeddingsConfig,
        example_input,
        run_esmc_embeddings,
    )

    run_esmc_embeddings(example_input(), ESMCEmbeddingsConfig())


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
@register_tools({"esmc-embedding": "inference"})
class ESMCService:
    """Modal service for ESM C masked language model inference."""

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

        inputs = ESMCEmbeddingsInput(**input_dict)
        config = ESMCEmbeddingsConfig(**config_dict)
        return dispatch_tool_call(run_esmc_embeddings, inputs, config, instance=self.instance)
