"""IgLM Modal service.

Delegates to proto-tools ``run_iglm_*`` functions for parameter validation,
environment setup, and model inference. IgLM ships its weights inside the
``iglm`` pip package, so the build-time warmup only has to create the tool env.
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
    """Deploy-time: build the tool env and exercise the default checkpoint."""
    from proto_tools.tools.causal_models.iglm.iglm_sample import (
        IgLMSampleConfig,
        example_input,
        run_iglm_sample,
    )

    run_iglm_sample(example_input(), IgLMSampleConfig())


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("IgLMService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["IgLMService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"iglm-sample": "sample", "iglm-infill": "infill", "iglm-score": "score"})
class IgLMService:
    """Modal service for IgLM antibody language model inference."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("iglm")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("iglm")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def sample(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Generate full-length antibody sequences.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Generated sequences bundled per prompt.
        """
        from proto_tools.tools.causal_models.iglm.iglm_sample import (
            IgLMSampleConfig,
            IgLMSampleInput,
            run_iglm_sample,
        )

        return run_tool_call(
            run_iglm_sample, IgLMSampleInput, IgLMSampleConfig, input_dict, config_dict, instance=self.instance
        )

    @modal.method()
    def infill(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Re-design a span of each input antibody sequence.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Re-designed sequences and infilled spans, bundled per parent sequence.
        """
        from proto_tools.tools.causal_models.iglm.iglm_infill import (
            IgLMInfillConfig,
            IgLMInfillInput,
            run_iglm_infill,
        )

        return run_tool_call(
            run_iglm_infill, IgLMInfillInput, IgLMInfillConfig, input_dict, config_dict, instance=self.instance
        )

    @modal.method()
    def score(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score antibody sequences and return likelihood metrics.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Scoring metrics, optional logits, and vocabulary.
        """
        from proto_tools.tools.causal_models.iglm.iglm_score import (
            IgLMScoringConfig,
            IgLMScoringInput,
            run_iglm_score,
        )

        return run_tool_call(
            run_iglm_score, IgLMScoringInput, IgLMScoringConfig, input_dict, config_dict, instance=self.instance
        )
