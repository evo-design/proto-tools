"""gLM2 Modal service.

Delegates to proto-tools ``run_glm2_*`` functions for parameter validation, environment setup,
and model inference. The build-time warmup creates the shared ``glm2_base`` env and downloads the
default checkpoint, persisted on the Modal volume via ``PROTO_MODEL_CACHE``. The smaller checkpoint
downloads on first use.
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
    """Deploy-time: build the tool env and warm the default checkpoint."""
    from proto_tools.tools.masked_models.glm2 import glm2_embeddings

    glm2_embeddings.run_glm2_embeddings(glm2_embeddings.example_input(), glm2_embeddings.GLM2EmbeddingsConfig())


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("GLM2Service")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["GLM2Service"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools(
    {
        "glm2-embedding": "inference",
        "glm2-score": "score",
        "glm2-sample": "sample",
        "glm2-gradient": "gradient",
    }
)
class GLM2Service:
    """Modal service for gLM2 mixed protein/DNA masked language model inference."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("glm2")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("glm2")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def inference(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run gLM2 embedding inference on mixed protein/DNA loci.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Mean embeddings and optional per-position logits.
        """
        from proto_tools.tools.masked_models.glm2 import GLM2EmbeddingsConfig, GLM2EmbeddingsInput, run_glm2_embeddings

        return run_tool_call(
            run_glm2_embeddings,
            GLM2EmbeddingsInput,
            GLM2EmbeddingsConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def score(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score mixed protein/DNA loci by masked pseudo-log-likelihood.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-locus likelihood metrics and optional logits.
        """
        from proto_tools.tools.masked_models.glm2 import GLM2ScoringConfig, GLM2ScoringInput, run_glm2_score

        return run_tool_call(
            run_glm2_score, GLM2ScoringInput, GLM2ScoringConfig, input_dict, config_dict, instance=self.instance
        )

    @modal.method()
    def sample(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Resample masked positions of mixed protein/DNA loci within each position's modality.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Completed loci and optional logits.
        """
        from proto_tools.tools.masked_models.glm2 import GLM2SampleConfig, GLM2SampleInput, run_glm2_sample

        return run_tool_call(
            run_glm2_sample, GLM2SampleInput, GLM2SampleConfig, input_dict, config_dict, instance=self.instance
        )

    @modal.method()
    def gradient(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Compute the masked pseudo-log-likelihood gradient for a relaxed mixed sequence.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Position-aligned gradient, loss, and metrics.
        """
        from proto_tools.tools.masked_models.glm2 import GLM2GradientConfig, GLM2GradientInput, run_glm2_gradient

        return run_tool_call(
            run_glm2_gradient, GLM2GradientInput, GLM2GradientConfig, input_dict, config_dict, instance=self.instance
        )
