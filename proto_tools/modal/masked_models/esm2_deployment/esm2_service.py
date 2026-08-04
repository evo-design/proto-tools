"""ESM2 Modal service.

Delegates to proto-tools ``run_esm2_*`` functions for parameter validation,
environment setup, and model inference. The build-time warmup creates the
tool env and downloads the default checkpoint, both persisted on the
Modal volume via ``PROTO_MODEL_CACHE``. Non-default checkpoints download
on first use.
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
    from proto_tools.tools.masked_models.esm2.esm2_embeddings import (
        ESM2EmbeddingsConfig,
        example_input,
        run_esm2_embeddings,
    )

    run_esm2_embeddings(example_input(), ESM2EmbeddingsConfig())


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("ESM2Service")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["ESM2Service"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools(
    {"esm2-sample": "sample", "esm2-score": "score", "esm2-embedding": "inference", "esm2-gradient": "gradient"}
)
class ESM2Service:
    """Modal service for ESM2 masked language model inference."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("esm2")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("esm2")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def inference(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run ESM2 embedding inference on sequences.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Mean embeddings, logits, and attention masks.
        """
        from proto_tools.tools.masked_models.esm2.esm2_embeddings import (
            ESM2EmbeddingsConfig,
            ESM2EmbeddingsInput,
            run_esm2_embeddings,
        )

        inputs = ESM2EmbeddingsInput(**input_dict)
        config = ESM2EmbeddingsConfig(**config_dict)
        return dispatch_tool_call(run_esm2_embeddings, inputs, config, instance=self.instance)

    @modal.method()
    def sample(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Sample protein sequences using ESM2.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Sampled sequences and optional logits.
        """
        from proto_tools.tools.masked_models.esm2.esm2_sample import (
            ESM2SampleConfig,
            ESM2SampleInput,
            run_esm2_sample,
        )

        inputs = ESM2SampleInput(**input_dict)
        config = ESM2SampleConfig(**config_dict)
        return dispatch_tool_call(run_esm2_sample, inputs, config, instance=self.instance)

    @modal.method()
    def score(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score protein sequences using ESM2 (MLM pseudo-perplexity).

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Logits, scoring metrics, and vocabulary.
        """
        from proto_tools.tools.masked_models.esm2.esm2_score import (
            ESM2ScoringConfig,
            ESM2ScoringInput,
            run_esm2_score,
        )

        inputs = ESM2ScoringInput(**input_dict)
        config = ESM2ScoringConfig(**config_dict)
        return dispatch_tool_call(run_esm2_score, inputs, config, instance=self.instance)

    @modal.method()
    def gradient(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Compute ESM2 gradients."""
        from proto_tools.tools.masked_models.esm2.esm2_gradient import (
            ESM2GradientConfig,
            ESM2GradientInput,
            run_esm2_gradient,
        )

        inputs = ESM2GradientInput(**input_dict)
        config = ESM2GradientConfig(**config_dict)
        return dispatch_tool_call(run_esm2_gradient, inputs, config, instance=self.instance)
