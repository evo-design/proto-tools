"""ProteinMPNN Modal service.

Delegates to proto-tools ``run_proteinmpnn_sample`` and ``run_proteinmpnn_score``
for parameter validation, environment setup, and model inference.  The build-time
warmup creates the tool env and downloads weights for both checkpoints
(proteinmpnn and abmpnn), persisted on the Modal volume via ``PROTO_MODEL_CACHE``.
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
    dispatch_tool_call,
    ensure_gpu_ready,
    env_for,
    stage_all_excluded_fixtures,
)


def _warmup() -> None:
    """Deploy-time: warm only the default model_choice (proteinmpnn)."""
    from proto_tools.tools.inverse_folding.proteinmpnn.proteinmpnn_sample import (
        ProteinMPNNSampleConfig,
        example_input,
        run_proteinmpnn_sample,
    )

    run_proteinmpnn_sample(example_input(), ProteinMPNNSampleConfig())


image = with_proto_tools(GPU_BASE).env(env_for())
image = (
    stage_all_excluded_fixtures(image)
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("ProteinMPNNService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["ProteinMPNNService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"proteinmpnn-sample": "sample", "proteinmpnn-score": "score", "proteinmpnn-gradient": "gradient"})
class ProteinMPNNService:
    """Modal service for ProteinMPNN inverse folding inference."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("proteinmpnn")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("proteinmpnn")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def sample(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Sample sequences from the ProteinMPNN model.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Sampled sequences with scores and sequence identities.
        """
        from proto_tools.tools.inverse_folding.proteinmpnn.proteinmpnn_sample import (
            ProteinMPNNSampleConfig,
            ProteinMPNNSampleInput,
            run_proteinmpnn_sample,
        )

        inputs = ProteinMPNNSampleInput(**input_dict)
        config = ProteinMPNNSampleConfig(**config_dict)
        return dispatch_tool_call(run_proteinmpnn_sample, inputs, config, instance=self.instance)

    @modal.method()
    def score(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score a protein sequence against a structure.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Scoring metrics and optionally per-residue logits.
        """
        from proto_tools.tools.inverse_folding.proteinmpnn.proteinmpnn_score import (
            ProteinMPNNScoringConfig,
            ProteinMPNNScoringInput,
            run_proteinmpnn_score,
        )

        inputs = ProteinMPNNScoringInput(**input_dict)
        config = ProteinMPNNScoringConfig(**config_dict)
        return dispatch_tool_call(run_proteinmpnn_score, inputs, config, instance=self.instance)

    @modal.method()
    def gradient(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Compute ProteinMPNN gradients."""
        from proto_tools.tools.inverse_folding.proteinmpnn.proteinmpnn_gradient import (
            ProteinMPNNGradientConfig,
            ProteinMPNNGradientInput,
            run_proteinmpnn_gradient,
        )

        inputs = ProteinMPNNGradientInput(**input_dict)
        config = ProteinMPNNGradientConfig(**config_dict)
        return dispatch_tool_call(run_proteinmpnn_gradient, inputs, config, instance=self.instance)
