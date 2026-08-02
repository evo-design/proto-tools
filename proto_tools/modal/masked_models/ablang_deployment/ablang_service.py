"""AbLang Modal service.

Delegates to proto-tools ``run_ablang_*`` functions for parameter validation,
environment setup, and model inference.  The build-time warmup creates the
tool env and downloads weights for all three AbLang model variants (heavy,
light, paired), persisted on the Modal volume via ``PROTO_MODEL_CACHE``.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SCALEDOWN_WINDOW, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import GPU_BASE, with_proto_tools
from proto_tools.modal.gpu_profiles import GPU_BASIC
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, dispatch_tool_call, ensure_gpu_ready, env_for


def _warmup() -> None:
    """Deploy-time: warm the paired-antibody variant (ablang2-paired)."""
    from proto_tools.entities.antibody import Antibody
    from proto_tools.tools.masked_models.ablang.ablang_embeddings import (
        AbLangEmbeddingsConfig,
        AbLangEmbeddingsInput,
        run_ablang_embeddings,
    )

    paired = Antibody(heavy_chain="EVQLVESGGGLVQPGG", light_chain="DIVMTQSPSSLSASVG")
    run_ablang_embeddings(
        AbLangEmbeddingsInput(antibodies=[paired]),
        AbLangEmbeddingsConfig(),
    )


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_BASIC, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("AbLangService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_BASIC,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["AbLangService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools(
    {
        "ablang-sample": "sample",
        "ablang-score": "score",
        "ablang-embedding": "inference",
        "ablang-gradient": "gradient",
    }
)
class AbLangService:
    """Modal service for AbLang antibody language model inference."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("ablang")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("ablang")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def sample(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Sample / restore antibody chains using AbLang.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Restored sequences and per-position probabilities.
        """
        from proto_tools.tools.masked_models.ablang.ablang_sample import (
            AbLangSampleConfig,
            AbLangSampleInput,
            run_ablang_sample,
        )

        inputs = AbLangSampleInput(**input_dict)
        config = AbLangSampleConfig(**config_dict)
        return dispatch_tool_call(run_ablang_sample, inputs, config, instance=self.instance)

    @modal.method()
    def score(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score antibody chains using AbLang (MLM pseudo-perplexity).

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Logits and scoring metrics.
        """
        from proto_tools.tools.masked_models.ablang.ablang_score import (
            AbLangScoringConfig,
            AbLangScoringInput,
            run_ablang_score,
        )

        inputs = AbLangScoringInput(**input_dict)
        config = AbLangScoringConfig(**config_dict)
        return dispatch_tool_call(run_ablang_score, inputs, config, instance=self.instance)

    @modal.method()
    def inference(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run AbLang embedding inference on antibody chains.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Mean embeddings per antibody.
        """
        from proto_tools.tools.masked_models.ablang.ablang_embeddings import (
            AbLangEmbeddingsConfig,
            AbLangEmbeddingsInput,
            run_ablang_embeddings,
        )

        inputs = AbLangEmbeddingsInput(**input_dict)
        config = AbLangEmbeddingsConfig(**config_dict)
        return dispatch_tool_call(run_ablang_embeddings, inputs, config, instance=self.instance)

    @modal.method()
    def gradient(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Compute AbLang shifted cross-entropy gradient on relaxed antibody logits.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Gradient w.r.t. input logits, loss, and metrics.
        """
        from proto_tools.tools.masked_models.ablang.ablang_gradient import (
            AbLangGradientConfig,
            AbLangGradientInput,
            run_ablang_gradient,
        )

        inputs = AbLangGradientInput(**input_dict)
        config = AbLangGradientConfig(**config_dict)
        return dispatch_tool_call(run_ablang_gradient, inputs, config, instance=self.instance)
