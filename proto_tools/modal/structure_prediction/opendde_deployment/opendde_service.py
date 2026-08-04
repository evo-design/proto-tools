"""OpenDDE Modal service.

Delegates to proto-tools ``run_opendde`` for parameter validation, environment setup, and structure
prediction. The build-time warmup creates the tool env and downloads the default checkpoint, both
persisted on the Modal volume via ``PROTO_MODEL_CACHE``.
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
    """Deploy-time: warm only the default checkpoint.

    ``use_msa=False`` so the build stages weights rather than also searching the ColabFold server,
    which is slow, external, and unrelated to whether the image is sound. Matches chai1 and protenix.
    """
    from proto_tools.tools.structure_prediction.opendde.opendde import OpenDDEConfig, example_input, run_opendde

    run_opendde(example_input(), OpenDDEConfig(use_msa=False))


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("OpenDDEService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["OpenDDEService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"opendde-prediction": "predict"})
class OpenDDEService:
    """Modal service for OpenDDE structure prediction."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("opendde")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("opendde")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def predict(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Predict a structure from sequence input.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Predicted structures and confidence metrics.
        """
        from proto_tools.tools.structure_prediction.opendde.opendde import (
            OpenDDEConfig,
            OpenDDEInput,
            run_opendde,
        )

        inputs = OpenDDEInput(**input_dict)
        config = OpenDDEConfig(**config_dict)
        return dispatch_tool_call(run_opendde, inputs, config, instance=self.instance)
