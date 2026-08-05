"""Chai1 Modal service.

Delegates to proto-tools ``run_chai1`` for parameter validation,
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
    from proto_tools.tools.structure_prediction.chai1 import Chai1Config
    from proto_tools.tools.structure_prediction.chai1.chai1 import (
        example_input,
        run_chai1,
    )

    run_chai1(example_input(), Chai1Config(use_msa=False))


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("Chai1Service")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["Chai1Service"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"chai1-prediction": "predict"})
class Chai1Service:
    """Modal service for Chai1 structure prediction."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("chai1")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("chai1")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def predict(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run Chai1 structure prediction.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Structure prediction results with metrics.
        """
        from proto_tools.tools.structure_prediction.chai1 import (
            Chai1Config,
            Chai1Input,
        )
        from proto_tools.tools.structure_prediction.chai1.chai1 import run_chai1

        return run_tool_call(run_chai1, Chai1Input, Chai1Config, input_dict, config_dict, instance=self.instance)
