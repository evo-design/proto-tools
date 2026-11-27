"""PARADE Modal service.

Delegates to proto-tools' three PARADE entry points for parameter validation, environment setup,
and inference. The build-time warmup creates the tool env and runs the pinned-commit checkpoint
download, both persisted on the Modal volume via ``PROTO_MODEL_CACHE``, so the first call does not
pay for it.
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
    """Deploy-time: build the env and stage the checkpoint."""
    from proto_tools.tools.sequence_scoring.parade.parade_activity import (
        ParadeActivityConfig,
        example_input,
        run_parade_activity,
    )

    run_parade_activity(example_input(), ParadeActivityConfig())


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("ParadeService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["ParadeService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools(
    {
        "parade-activity": "activity",
        "parade-gradient": "gradient",
        "parade-stability": "stability",
    }
)
class ParadeService:
    """Modal service for PARADE UTR activity, stability, and gradient prediction."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("parade")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("parade")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def activity(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Predict UTR translational activity for input sequences.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-sequence activity predictions.
        """
        from proto_tools.tools.sequence_scoring.parade.parade_activity import (
            ParadeActivityConfig,
            run_parade_activity,
        )
        from proto_tools.tools.sequence_scoring.parade.shared_data_models import ParadeSequenceInput

        return run_tool_call(
            run_parade_activity,
            ParadeSequenceInput,
            ParadeActivityConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def gradient(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Compute input gradients of a PARADE prediction with respect to sequence.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-position gradient attributions.
        """
        from proto_tools.tools.sequence_scoring.parade.parade_gradient import (
            ParadeGradientConfig,
            ParadeGradientInput,
            run_parade_gradient,
        )

        return run_tool_call(
            run_parade_gradient,
            ParadeGradientInput,
            ParadeGradientConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def stability(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Predict UTR-driven transcript stability for input sequences.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-sequence stability predictions.
        """
        from proto_tools.tools.sequence_scoring.parade.parade_stability import run_parade_stability
        from proto_tools.tools.sequence_scoring.parade.shared_data_models import (
            ParadeCheckpointConfig,
            ParadeSequenceInput,
        )

        return run_tool_call(
            run_parade_stability,
            ParadeSequenceInput,
            ParadeCheckpointConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )
