"""Cherimoya Modal service.

Delegates to proto-tools for validation, environment setup, checkpoint provisioning, and
inference. The build-time warmup creates the standalone tool env and downloads the default
CATv1 checkpoint into the shared Modal model cache via ``PROTO_MODEL_CACHE``; every other
checkpoint is fetched on first use.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SCALEDOWN_WINDOW, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import GPU_BASE, with_proto_tools
from proto_tools.modal.gpu_profiles import GPU_BASIC
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, ensure_gpu_ready, env_for, run_tool_call


def _warmup() -> None:
    """Deploy-time: create the Cherimoya env and provision/load the default checkpoint."""
    from proto_tools.tools.sequence_scoring.cherimoya import CherimoyaScoreConfig, run_cherimoya_score
    from proto_tools.tools.sequence_scoring.cherimoya.cherimoya_score import example_input

    run_cherimoya_score(example_input(), CherimoyaScoreConfig())


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_BASIC, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("CherimoyaService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_BASIC,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["CherimoyaService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"cherimoya-score": "score", "cherimoya-predict": "predict", "cherimoya-gradient": "gradient"})
class CherimoyaService:
    """Modal service for Cherimoya chromatin accessibility scoring, profiles, and gradients."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so loaded checkpoints stay cached across requests."""
        ensure_gpu_ready("cherimoya")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("cherimoya")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def score(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score DNA sequences with Cherimoya models.

        Args:
            input_dict (dict[str, Any]): Serialized ``CherimoyaScoreInput``.
            config_dict (dict[str, Any]): Serialized ``CherimoyaScoreConfig``.

        Returns:
            dict[str, Any]: Per-sequence log counts (and target counts) per model.
        """
        from proto_tools.tools.sequence_scoring.cherimoya import (
            CherimoyaScoreConfig,
            CherimoyaScoreInput,
            run_cherimoya_score,
        )

        return run_tool_call(
            run_cherimoya_score,
            CherimoyaScoreInput,
            CherimoyaScoreConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def predict(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Predict base-resolution profiles with Cherimoya models.

        Args:
            input_dict (dict[str, Any]): Serialized ``CherimoyaPredictInput``.
            config_dict (dict[str, Any]): Serialized ``CherimoyaPredictConfig``.

        Returns:
            dict[str, Any]: Per-sequence 1,000 bp profiles per model.
        """
        from proto_tools.tools.sequence_scoring.cherimoya import (
            CherimoyaPredictConfig,
            CherimoyaPredictInput,
            run_cherimoya_predict,
        )

        return run_tool_call(
            run_cherimoya_predict,
            CherimoyaPredictInput,
            CherimoyaPredictConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def gradient(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Compute Cherimoya objective gradients for relaxed DNA logits.

        Args:
            input_dict (dict[str, Any]): Serialized ``CherimoyaGradientInput``.
            config_dict (dict[str, Any]): Serialized ``CherimoyaGradientConfig``.

        Returns:
            dict[str, Any]: Loss terms and optional gradients with respect to the input logits.
        """
        from proto_tools.tools.sequence_scoring.cherimoya import (
            CherimoyaGradientConfig,
            CherimoyaGradientInput,
            run_cherimoya_gradient,
        )

        return run_tool_call(
            run_cherimoya_gradient,
            CherimoyaGradientInput,
            CherimoyaGradientConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )
