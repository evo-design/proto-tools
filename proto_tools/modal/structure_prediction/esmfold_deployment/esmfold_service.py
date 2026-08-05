"""ESMFold Modal service.

Delegates to proto-tools ``run_esmfold`` for parameter validation,
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
    """Deploy-time: warm the default config."""
    from proto_tools.tools.structure_prediction.esmfold import ESMFoldConfig
    from proto_tools.tools.structure_prediction.esmfold.esmfold import (
        example_input,
        run_esmfold,
    )

    run_esmfold(example_input(), ESMFoldConfig())


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("ESMFoldService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["ESMFoldService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"esmfold-prediction": "predict", "esmfold-gradient": "gradient"})
class ESMFoldService:
    """Modal service for ESMFold structure prediction."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("esmfold")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("esmfold")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def predict(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run ESMFold structure prediction on protein sequences.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Structure prediction results with metrics.
        """
        from proto_tools.tools.structure_prediction.esmfold import (
            ESMFoldConfig,
            ESMFoldInput,
        )
        from proto_tools.tools.structure_prediction.esmfold.esmfold import run_esmfold

        return run_tool_call(run_esmfold, ESMFoldInput, ESMFoldConfig, input_dict, config_dict, instance=self.instance)

    @modal.method()
    def gradient(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Compute ESMFold confidence gradients."""
        from proto_tools.tools.structure_prediction.esmfold.esmfold import (
            ESMFoldGradientConfig,
            ESMFoldGradientInput,
            run_esmfold_gradient,
        )

        return run_tool_call(
            run_esmfold_gradient,
            ESMFoldGradientInput,
            ESMFoldGradientConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )
