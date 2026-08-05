"""BioEmu Modal service.

Delegates to proto-tools ``run_bioemu`` for parameter validation,
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
from proto_tools.modal.utils import (
    RUNTIME_ENV,
    ensure_gpu_ready,
    env_for,
    run_tool_call,
    stage_all_excluded_fixtures,
)


def _warmup() -> None:
    """Deploy-time: warm the default config."""
    from proto_tools.tools.structure_dynamics.bioemu.bioemu_sample import (
        BioEmuConfig,
        example_input,
        run_bioemu,
    )

    run_bioemu(example_input(), BioEmuConfig())


image = with_proto_tools(GPU_BASE).env(env_for())
image = (
    stage_all_excluded_fixtures(image)
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("BioEmuService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["BioEmuService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"bioemu-sample": "sample"})
class BioEmuService:
    """Modal service for BioEmu conformational ensemble sampling."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("bioemu")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("bioemu")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def sample(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Sample conformational ensemble with BioEmu.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Generated protein conformational ensembles.
        """
        from proto_tools.tools.structure_dynamics.bioemu import (
            BioEmuConfig,
            BioEmuInput,
            run_bioemu,
        )

        return run_tool_call(run_bioemu, BioEmuInput, BioEmuConfig, input_dict, config_dict, instance=self.instance)
