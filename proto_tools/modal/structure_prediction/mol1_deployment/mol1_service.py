"""Mol-1 prediction service using a persistent standalone worker."""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SCALEDOWN_WINDOW, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import GPU_BASE, with_proto_tools
from proto_tools.modal.gpu_profiles import GPU_DEFAULT
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, ensure_gpu_ready, env_for, run_tool_call


def _warmup() -> None:
    """Build the environment and warm the default checkpoint without an MSA search."""
    from proto_tools.tools.structure_prediction.mol1.mol1_prediction import (
        Mol1Config,
        example_input,
        run_mol1_prediction,
    )

    run_mol1_prediction(example_input(), Mol1Config(use_msa=False))


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)

app = get_app_for_service("Mol1Service")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["Mol1Service"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"mol1-prediction": "predict"})
class Mol1Service:
    """Serve Mol-1 predictions from one persistent worker per container."""

    @modal.enter()
    def setup(self) -> None:
        """Start the standalone worker."""
        ensure_gpu_ready("mol1")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("mol1")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Stop the worker when the container shuts down."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def predict(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Predict structures from serialized complexes.

        Args:
            input_dict (dict[str, Any]): Serialized Mol1Input fields.
            config_dict (dict[str, Any]): Serialized Mol1Config fields.

        Returns:
            dict[str, Any]: Serialized structures and confidence metrics.
        """
        from proto_tools.tools.structure_prediction.mol1.mol1_prediction import (
            Mol1Config,
            Mol1Input,
            run_mol1_prediction,
        )

        return run_tool_call(
            run_mol1_prediction, Mol1Input, Mol1Config, input_dict, config_dict, instance=self.instance
        )
