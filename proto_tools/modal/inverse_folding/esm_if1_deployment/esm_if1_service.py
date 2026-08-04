"""ESM-IF1 Modal service.

Delegates to proto-tools ``run_esm_if1_sample`` and ``run_esm_if1_score`` for
parameter validation, environment setup, and model inference.  The build-time
warmup creates the tool env and downloads weights, persisted on the Modal
volume via ``PROTO_MODEL_CACHE``.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SCALEDOWN_WINDOW, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import GPU_BASE, with_proto_tools
from proto_tools.modal.gpu_profiles import GPU_BASIC
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
    """Deploy-time: warm only the default weights variant (protein_dpo)."""
    from proto_tools.tools.inverse_folding.esm_if1.esm_if1_sample import (
        ESMIF1SampleConfig,
        example_input,
        run_esm_if1_sample,
    )

    run_esm_if1_sample(example_input(), ESMIF1SampleConfig())


image = with_proto_tools(GPU_BASE).env(env_for())
image = (
    stage_all_excluded_fixtures(image)
    .run_function(
        _warmup, gpu=GPU_BASIC, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("ESMIF1Service")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_BASIC,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["ESMIF1Service"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"esm-if1-sample": "sample", "esm-if1-score": "score"})
class ESMIF1Service:
    """Modal service for ESM-IF1 inverse folding inference."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("esm_if1")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("esm_if1")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def sample(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Sample sequences from the ESM-IF1 model.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Sampled sequences with metrics.
        """
        from proto_tools.tools.inverse_folding.esm_if1.esm_if1_sample import (
            ESMIF1SampleConfig,
            ESMIF1SampleInput,
            run_esm_if1_sample,
        )

        inputs = ESMIF1SampleInput(**input_dict)
        config = ESMIF1SampleConfig(**config_dict)
        return dispatch_tool_call(run_esm_if1_sample, inputs, config, instance=self.instance)

    @modal.method()
    def score(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score sequence-structure pairs with ESM-IF1.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-sequence log-likelihoods and scoring metrics.
        """
        from proto_tools.tools.inverse_folding.esm_if1.esm_if1_score import (
            ESMIF1ScoringConfig,
            ESMIF1ScoringInput,
            run_esm_if1_score,
        )

        inputs = ESMIF1ScoringInput(**input_dict)
        config = ESMIF1ScoringConfig(**config_dict)
        return dispatch_tool_call(run_esm_if1_score, inputs, config, instance=self.instance)
