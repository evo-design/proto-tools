"""ESMFold2 Modal service.

Delegates to proto-tools ``run_esmfold2`` for parameter validation,
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
from proto_tools.modal.utils import RUNTIME_ENV, dispatch_tool_call, ensure_gpu_ready, env_for


def _warmup() -> None:
    """Deploy-time: warm the default ``esmfold2-fast`` checkpoint."""
    from proto_tools.tools.structure_prediction.esmfold2 import ESMFold2Config
    from proto_tools.tools.structure_prediction.esmfold2.esmfold2 import (
        example_input,
        run_esmfold2,
    )

    run_esmfold2(example_input(), ESMFold2Config())


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    # ESMFold2's biohub weights are HF-gated, so the build-time warmup needs the token
    # (the public-weight structure services warm without it; env_for() omits HF_TOKEN).
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("ESMFold2Service")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["ESMFold2Service"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"esmfold2-prediction": "predict"})
class ESMFold2Service:
    """Modal service for ESMFold2 all-atom complex structure prediction."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("esmfold2")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("esmfold2")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def predict(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run ESMFold2 all-atom complex structure prediction.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Structure prediction results with metrics.
        """
        from proto_tools.tools.structure_prediction.esmfold2 import (
            ESMFold2Config,
            ESMFold2Input,
        )
        from proto_tools.tools.structure_prediction.esmfold2.esmfold2 import run_esmfold2

        inputs = ESMFold2Input(**input_dict)
        config = ESMFold2Config(**config_dict)
        return dispatch_tool_call(run_esmfold2, inputs, config, instance=self.instance)
