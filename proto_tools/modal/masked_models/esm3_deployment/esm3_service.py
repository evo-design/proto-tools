"""ESM3 Modal service.

Delegates to proto-tools ``run_esm3_embeddings`` / ``run_esm3_sample`` / ``run_esm3_score`` for
parameter validation, environment setup, and model inference. The build-time warmup creates the
tool env and downloads the default checkpoint, both persisted on the Modal volume via
``PROTO_MODEL_CACHE``. ESM3 weights are gated on HuggingFace, so the build needs a token.
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
    from proto_tools.tools.masked_models.esm3.esm3_embeddings import (
        ESM3EmbeddingsConfig,
        example_input,
        run_esm3_embeddings,
    )

    run_esm3_embeddings(example_input(), ESM3EmbeddingsConfig())


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("ESM3Service")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["ESM3Service"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools(
    {
        "esm3-embedding": "inference",
        "esm3-sample": "sample",
        "esm3-score": "score",
    }
)
class ESM3Service:
    """Modal service for ESM3 masked language model inference."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("esm3")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("esm3")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def inference(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run ESM3 embedding inference on protein sequences.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Mean embeddings, optional logits, and attention masks.
        """
        from proto_tools.tools.masked_models.esm3.esm3_embeddings import (
            ESM3EmbeddingsConfig,
            ESM3EmbeddingsInput,
            run_esm3_embeddings,
        )

        return run_tool_call(
            run_esm3_embeddings,
            ESM3EmbeddingsInput,
            ESM3EmbeddingsConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def sample(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Fill masked positions in protein sequences with ESM3-sampled residues.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Sampled sequences and optional per-position logits.
        """
        from proto_tools.tools.masked_models.esm3.esm3_sample import (
            ESM3SampleConfig,
            ESM3SampleInput,
            run_esm3_sample,
        )

        return run_tool_call(
            run_esm3_sample, ESM3SampleInput, ESM3SampleConfig, input_dict, config_dict, instance=self.instance
        )

    @modal.method()
    def score(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score protein sequences under ESM3's masked language model.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-sequence scoring metrics.
        """
        from proto_tools.tools.masked_models.esm3.esm3_score import (
            ESM3ScoringConfig,
            ESM3ScoringInput,
            run_esm3_score,
        )

        return run_tool_call(
            run_esm3_score, ESM3ScoringInput, ESM3ScoringConfig, input_dict, config_dict, instance=self.instance
        )
