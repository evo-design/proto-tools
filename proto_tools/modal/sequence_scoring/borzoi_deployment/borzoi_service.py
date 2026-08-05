"""Borzoi Modal service.

Delegates to proto-tools ``run_borzoi`` and ``run_borzoi_ensemble`` for
parameter validation, environment setup, and model inference.  The build-time
warmup creates the micromamba env and downloads weights, both persisted on
the Modal volume via ``PROTO_MODEL_CACHE``.
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
    """Deploy-time: warm only the default ``(human, replicate=0)``."""
    from proto_tools.tools.sequence_scoring.borzoi.borzoi_prediction import (
        BorzoiConfig,
        example_input,
        run_borzoi,
    )

    run_borzoi(example_input(), BorzoiConfig())


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("BorzoiService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["BorzoiService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"borzoi-prediction": "predict", "borzoi-ensemble": "predict_ensemble"})
class BorzoiService:
    """Modal service for Borzoi sequence scoring inference."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("borzoi")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("borzoi")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def predict(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Predict regulatory activity from sequence using a single Borzoi replicate.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Prediction matrix with shape [num_tracks, 6144].
        """
        from proto_tools.tools.sequence_scoring.borzoi import (
            BorzoiConfig,
            BorzoiInput,
            run_borzoi,
        )

        return run_tool_call(run_borzoi, BorzoiInput, BorzoiConfig, input_dict, config_dict, instance=self.instance)

    @modal.method()
    def predict_ensemble(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Predict regulatory activity using all 4 Borzoi replicates.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Stacked predictions from Borzoi replicates 0-3.
        """
        from proto_tools.tools.sequence_scoring.borzoi import (
            BorzoiEnsembleConfig,
            BorzoiInput,
            run_borzoi_ensemble,
        )

        return run_tool_call(
            run_borzoi_ensemble, BorzoiInput, BorzoiEnsembleConfig, input_dict, config_dict, instance=self.instance
        )
