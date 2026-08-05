"""ProGen2 Modal service.

Delegates to proto-tools ``run_progen2_*`` functions for parameter validation,
environment setup, and model inference.  The build-time warmup creates the
tool env and downloads weights for all checkpoints, both persisted on the
Modal volume via ``PROTO_MODEL_CACHE``.
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
    """Deploy-time: warm only the default checkpoint."""
    from proto_tools.tools.causal_models.progen2.progen2_sample import (
        ProGen2SampleConfig,
        example_input,
        run_progen2_sample,
    )

    run_progen2_sample(example_input(), ProGen2SampleConfig())


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("ProGen2Service")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["ProGen2Service"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"progen2-sample": "sample", "progen2-score": "score"})
class ProGen2Service:
    """Modal service for ProGen2 protein language model inference."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("progen2")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("progen2")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def sample(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Generate protein sequences using ProGen2 model.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Sampled sequences and optional logits.
        """
        from proto_tools.tools.causal_models.progen2.progen2_sample import (
            ProGen2SampleConfig,
            ProGen2SampleInput,
            run_progen2_sample,
        )

        return run_tool_call(
            run_progen2_sample, ProGen2SampleInput, ProGen2SampleConfig, input_dict, config_dict, instance=self.instance
        )

    @modal.method()
    def score(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score protein sequences and return logits and pre-computed metrics.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Logits, scoring metrics, and vocabulary.
        """
        from proto_tools.tools.causal_models.progen2.progen2_score import (
            ProGen2ScoringConfig,
            ProGen2ScoringInput,
            run_progen2_score,
        )

        return run_tool_call(
            run_progen2_score,
            ProGen2ScoringInput,
            ProGen2ScoringConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )
