"""Puffin Modal service.

Delegates to proto-tools ``run_puffin_prediction`` / ``run_puffin_interpretation`` for parameter
validation, environment setup, and inference. The build-time warmup creates the tool env and
downloads the model weights, both persisted on the Modal volume via ``PROTO_MODEL_CACHE``.
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
    """Deploy-time: warm the model weights."""
    from proto_tools.tools.sequence_scoring.puffin.puffin_prediction import (
        PuffinPredictionConfig,
        example_input,
        run_puffin_prediction,
    )

    run_puffin_prediction(example_input(), PuffinPredictionConfig())


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("PuffinService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["PuffinService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools(
    {
        "puffin-interpretation": "interpret",
        "puffin-prediction": "predict",
    }
)
class PuffinService:
    """Modal service for Puffin promoter-activity prediction and interpretation."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("puffin")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("puffin")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def interpret(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Attribute Puffin predictions back to sequence positions and motifs.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-sequence interpretation tracks.
        """
        from proto_tools.tools.sequence_scoring.puffin.puffin_interpretation import (
            PuffinInterpretationConfig,
            PuffinInterpretationInput,
            run_puffin_interpretation,
        )

        inputs = PuffinInterpretationInput(**input_dict)
        config = PuffinInterpretationConfig(**config_dict)
        return dispatch_tool_call(run_puffin_interpretation, inputs, config, instance=self.instance)

    @modal.method()
    def predict(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Predict promoter activity profiles for input sequences.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-sequence predicted activity tracks.
        """
        from proto_tools.tools.sequence_scoring.puffin.puffin_prediction import (
            PuffinPredictionConfig,
            PuffinPredictionInput,
            run_puffin_prediction,
        )

        inputs = PuffinPredictionInput(**input_dict)
        config = PuffinPredictionConfig(**config_dict)
        return dispatch_tool_call(run_puffin_prediction, inputs, config, instance=self.instance)
