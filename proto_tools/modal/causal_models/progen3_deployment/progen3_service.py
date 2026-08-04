"""ProGen3 Modal service.

Delegates to proto-tools ``run_progen3_sample`` / ``run_progen3_score`` for parameter validation,
environment setup, and generation. The build-time warmup creates the tool env and downloads the
default ``progen3-762m`` checkpoint, both persisted on the Modal volume via ``PROTO_MODEL_CACHE``.
Larger checkpoints download on first use of the call that names them.
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
    """Deploy-time: warm only the default checkpoint."""
    from proto_tools.tools.causal_models.progen3.progen3_score import (
        ProGen3ScoringConfig,
        example_input,
        run_progen3_score,
    )

    run_progen3_score(example_input(), ProGen3ScoringConfig())


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("ProGen3Service")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["ProGen3Service"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools(
    {
        "progen3-sample": "sample",
        "progen3-score": "score",
    }
)
class ProGen3Service:
    """Modal service for ProGen3 protein sequence generation and scoring."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("progen3")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("progen3")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def sample(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Generate protein sequences with ProGen3.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Sampled sequences and their generation metadata.
        """
        from proto_tools.tools.causal_models.progen3.progen3_sample import (
            ProGen3SampleConfig,
            run_progen3_sample,
        )
        from proto_tools.tools.causal_models.shared_data_models import CausalModelSampleInput

        inputs = CausalModelSampleInput(**input_dict)
        config = ProGen3SampleConfig(**config_dict)
        return dispatch_tool_call(run_progen3_sample, inputs, config, instance=self.instance)

    @modal.method()
    def score(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score protein sequences under ProGen3's causal language model.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-sequence log-likelihoods and perplexities.
        """
        from proto_tools.tools.causal_models.progen3.progen3_score import (
            ProGen3ScoringConfig,
            run_progen3_score,
        )
        from proto_tools.tools.causal_models.shared_data_models import CausalModelScoringInput

        inputs = CausalModelScoringInput(**input_dict)
        config = ProGen3ScoringConfig(**config_dict)
        return dispatch_tool_call(run_progen3_score, inputs, config, instance=self.instance)
