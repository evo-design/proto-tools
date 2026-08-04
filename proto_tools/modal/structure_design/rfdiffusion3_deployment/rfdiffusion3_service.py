"""RFdiffusion3 Modal service.

Delegates to proto-tools ``run_rfdiffusion3`` for parameter validation,
environment setup, and model inference.  The build-time warmup creates the
tool env and downloads weights, persisted on the Modal volume via ``PROTO_MODEL_CACHE``.
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
    """Deploy-time: warm the default config."""
    from proto_tools.tools.structure_design.rfdiffusion3.rfdiffusion3_sample import (
        RFdiffusion3Config,
        example_input,
        run_rfdiffusion3,
    )

    run_rfdiffusion3(example_input(), RFdiffusion3Config())


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("RFdiffusion3Service")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["RFdiffusion3Service"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"rfdiffusion3-design": "generate"})
class RFdiffusion3Service:
    """Modal service for RFdiffusion3 structure design."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("rfdiffusion3")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("rfdiffusion3")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def generate(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run RFdiffusion3 structure generation.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Generated structures with metadata.
        """
        from proto_tools.tools.structure_design.rfdiffusion3.rfdiffusion3_sample import (
            RFdiffusion3Config,
            RFdiffusion3Input,
            run_rfdiffusion3,
        )

        inputs = RFdiffusion3Input(**input_dict)
        config = RFdiffusion3Config(**config_dict)
        return dispatch_tool_call(run_rfdiffusion3, inputs, config, instance=self.instance)
