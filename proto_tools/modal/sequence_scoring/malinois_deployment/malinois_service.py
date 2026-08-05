"""Malinois Modal service.

Delegates to proto-tools for validation, environment setup, checkpoint
provisioning, model inference, and relaxed-logit gradients. The build-time
warmup creates the standalone tool env and downloads the Malinois artifact into
the shared Modal model cache via ``PROTO_MODEL_CACHE``.
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
    """Deploy-time: create the Malinois env and provision/load the default checkpoint."""
    from proto_tools.tools.sequence_scoring.malinois import MalinoisScoreConfig, MalinoisScoreInput, run_malinois_score

    run_malinois_score(MalinoisScoreInput(sequences=["A" * 200]), MalinoisScoreConfig(cell_types=["K562"]))


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_BASIC, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("MalinoisService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_BASIC,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["MalinoisService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"malinois-score": "score", "malinois-gradient": "gradient"})
class MalinoisService:
    """Modal service for Malinois MPRA regulatory DNA activity scoring and gradients."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("malinois")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("malinois")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def score(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score regulatory DNA sequences using Malinois.

        Args:
            input_dict (dict[str, Any]): Serialized ``MalinoisScoreInput``.
            config_dict (dict[str, Any]): Serialized ``MalinoisScoreConfig``.

        Returns:
            dict[str, Any]: Per-sequence predictions keyed by requested cell type.
        """
        from proto_tools.tools.sequence_scoring.malinois import (
            MalinoisScoreConfig,
            MalinoisScoreInput,
            run_malinois_score,
        )

        return run_tool_call(
            run_malinois_score, MalinoisScoreInput, MalinoisScoreConfig, input_dict, config_dict, instance=self.instance
        )

    @modal.method()
    def gradient(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Compute Malinois activity gradients for relaxed DNA logits.

        Args:
            input_dict (dict[str, Any]): Serialized ``MalinoisGradientInput``.
            config_dict (dict[str, Any]): Serialized ``MalinoisGradientConfig``.

        Returns:
            dict[str, Any]: Scalar objective terms and optional gradients with respect to input logits.
        """
        from proto_tools.tools.sequence_scoring.malinois import (
            MalinoisGradientConfig,
            MalinoisGradientInput,
            run_malinois_gradient,
        )

        return run_tool_call(
            run_malinois_gradient,
            MalinoisGradientInput,
            MalinoisGradientConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )
