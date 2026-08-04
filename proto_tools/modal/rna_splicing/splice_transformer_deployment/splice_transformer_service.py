"""SpliceTransformer Modal service.

Delegates to proto-tools ``run_splice_transformer`` for parameter validation,
environment setup, and model inference.  The build-time warmup creates the
micromamba env and downloads weights, both persisted on the Modal volume via
``PROTO_MODEL_CACHE``.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SCALEDOWN_WINDOW, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import GPU_BASE, with_proto_tools
from proto_tools.modal.gpu_profiles import GPU_BASIC
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, ensure_gpu_ready, env_for, run_tool_call


def _warmup() -> None:
    """Deploy-time: warm the default config."""
    from proto_tools.tools.rna_splicing.splice_transformer.splice_transformer import (
        example_input,
        run_splice_transformer,
    )

    run_splice_transformer(example_input())


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_BASIC, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("SpliceTransformerService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_BASIC,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["SpliceTransformerService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"splice-transformer-prediction": "run"})
class SpliceTransformerService:
    """Modal service for SpliceTransformer RNA splicing inference."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("splice_transformer")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("splice_transformer")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def run(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run SpliceTransformer inference on sequences.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Predictions array of shape (batch, target_length, 18)
        """
        from proto_tools.tools.rna_splicing.splice_transformer import (
            SpliceTransformerConfig,
            SpliceTransformerInput,
            run_splice_transformer,
        )

        return run_tool_call(
            run_splice_transformer,
            SpliceTransformerInput,
            SpliceTransformerConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )
