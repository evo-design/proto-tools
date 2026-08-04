"""Evo1 Modal service.

Delegates to proto-tools ``run_evo1_*`` functions for parameter validation,
environment setup, and model inference.  The build-time warmup creates the
tool env and downloads weights for all checkpoints, both persisted on the
Modal volume via ``PROTO_MODEL_CACHE``.
"""

from pathlib import Path
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
)


def _warmup() -> None:
    """Deploy-time: warm only the default checkpoint."""
    from proto_tools.tools.causal_models.evo1.evo1_sample import (
        Evo1SampleConfig,
        example_input,
        run_evo1_sample,
    )

    run_evo1_sample(example_input(), Evo1SampleConfig())


image = with_proto_tools(GPU_BASE, overrides="evo1", overrides_dir=Path(__file__).parent)
image = (
    image.env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("Evo1Service")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["Evo1Service"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"evo1-sample": "sample", "evo1-score": "score"})
class Evo1Service:
    """Modal service for Evo1 causal language model inference."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("evo1")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("evo1")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def sample(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Generate DNA sequences using Evo1 model.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Sampled sequences and optional logits.
        """
        from proto_tools.tools.causal_models.evo1.evo1_sample import (
            Evo1SampleConfig,
            Evo1SampleInput,
            run_evo1_sample,
        )

        inputs = Evo1SampleInput(**input_dict)
        config = Evo1SampleConfig(**config_dict)
        return dispatch_tool_call(run_evo1_sample, inputs, config, instance=self.instance)

    @modal.method()
    def score(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score DNA sequences and return logits and pre-computed metrics.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Logits, scoring metrics, and vocabulary.
        """
        from proto_tools.tools.causal_models.evo1.evo1_score import (
            Evo1ScoringConfig,
            Evo1ScoringInput,
            run_evo1_score,
        )

        inputs = Evo1ScoringInput(**input_dict)
        config = Evo1ScoringConfig(**config_dict)
        return dispatch_tool_call(run_evo1_score, inputs, config, instance=self.instance)
