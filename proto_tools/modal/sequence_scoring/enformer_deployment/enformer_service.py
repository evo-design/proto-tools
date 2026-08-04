"""Enformer Modal service.

Delegates to proto-tools ``run_enformer`` for parameter validation,
environment setup, and model inference.  The build-time warmup creates the
micromamba env and downloads weights, both persisted on the Modal volume via
``PROTO_MODEL_CACHE``.
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
    """Deploy-time: warm only the default species (human)."""
    from proto_tools.tools.sequence_scoring.enformer.enformer_prediction import (
        EnformerConfig,
        example_input,
        run_enformer,
    )

    run_enformer(example_input(), EnformerConfig())


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("EnformerService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["EnformerService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"enformer-prediction": "predict"})
class EnformerService:
    """Modal service for Enformer regulatory sequence scoring."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("enformer")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("enformer")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def predict(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Predict regulatory activity from sequence using Enformer.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Predicted activity matrix with shape [896, num_tracks].
        """
        from proto_tools.tools.sequence_scoring.enformer import (
            EnformerConfig,
            EnformerInput,
            run_enformer,
        )

        return run_tool_call(
            run_enformer, EnformerInput, EnformerConfig, input_dict, config_dict, instance=self.instance
        )
