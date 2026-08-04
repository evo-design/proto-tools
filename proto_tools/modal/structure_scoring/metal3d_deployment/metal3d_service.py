"""Metal3D Modal service."""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SCALEDOWN_WINDOW, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import GPU_BASE, with_proto_tools
from proto_tools.modal.gpu_profiles import GPU_BASIC
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import (
    RUNTIME_ENV,
    dispatch_tool_call,
    ensure_gpu_ready,
    env_for,
    stage_all_excluded_fixtures,
)


def _warmup() -> None:
    """Deploy-time: create the Metal3D env and warm the default checkpoint."""
    from proto_tools.tools.structure_scoring.metal3d.metal3d_prediction import (
        Metal3DPredictionConfig,
        example_input,
        run_metal3d_prediction,
    )

    run_metal3d_prediction(example_input(), Metal3DPredictionConfig())


image = with_proto_tools(GPU_BASE).env(env_for())
image = (
    stage_all_excluded_fixtures(image)
    .run_function(
        _warmup, gpu=GPU_BASIC, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("Metal3DService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_BASIC,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["Metal3DService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"metal3d-prediction": "predict"})
class Metal3DService:
    """Modal service for Metal3D metal-ion site prediction."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("metal3d")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("metal3d")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def predict(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run Metal3D prediction."""
        from proto_tools.tools.structure_scoring.metal3d.metal3d_prediction import (
            Metal3DPredictionConfig,
            Metal3DPredictionInput,
            run_metal3d_prediction,
        )

        inputs = Metal3DPredictionInput(**input_dict)
        config = Metal3DPredictionConfig(**config_dict)
        return dispatch_tool_call(run_metal3d_prediction, inputs, config, instance=self.instance)
