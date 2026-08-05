"""CodonFM Modal service.

Delegates all five Encodon operations to their proto-tools run functions for validation,
standalone-environment setup, checkpoint caching, and inference. The build-time warmup creates
the environment and stages the public default checkpoint on the shared model-cache volume.
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
    """Deploy-time: build the environment and warm the default Encodon checkpoint."""
    from proto_tools.tools.masked_models.codonfm.codonfm_fitness import (
        CodonFMFitnessConfig,
        example_input,
        run_codonfm_fitness,
    )

    run_codonfm_fitness(example_input(), CodonFMFitnessConfig())


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup,
        gpu=GPU_DEFAULT,
        volumes={"/weights": MODEL_CACHE},
        secrets=[HF_TOKEN_SECRET],
        include_source=False,
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("CodonFMService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["CodonFMService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools(
    {
        "codonfm-embedding": "inference",
        "codonfm-fitness": "fitness",
        "codonfm-gradient": "gradient",
        "codonfm-sample": "sample",
        "codonfm-score": "score",
    }
)
class CodonFMService:
    """Modal service for CodonFM/Encodon coding-sequence inference."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the checkpoint stays loaded across requests."""
        ensure_gpu_ready("codonfm")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("codonfm")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def inference(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Extract Encodon CLS embeddings from coding sequences."""
        from proto_tools.tools.masked_models.codonfm.codonfm_embeddings import (
            CodonFMEmbeddingsConfig,
            CodonFMEmbeddingsInput,
            run_codonfm_embeddings,
        )

        return run_tool_call(
            run_codonfm_embeddings,
            CodonFMEmbeddingsInput,
            CodonFMEmbeddingsConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def fitness(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score coding-sequence fitness with Encodon's visible-token objective."""
        from proto_tools.tools.masked_models.codonfm.codonfm_fitness import (
            CodonFMFitnessConfig,
            CodonFMFitnessInput,
            run_codonfm_fitness,
        )

        return run_tool_call(
            run_codonfm_fitness,
            CodonFMFitnessInput,
            CodonFMFitnessConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def gradient(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Compute the Encodon masked pseudo-log-likelihood gradient."""
        from proto_tools.tools.masked_models.codonfm.codonfm_gradient import (
            CodonFMGradientConfig,
            CodonFMGradientInput,
            run_codonfm_gradient,
        )

        return run_tool_call(
            run_codonfm_gradient,
            CodonFMGradientInput,
            CodonFMGradientConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def sample(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Resample selected codons with Encodon."""
        from proto_tools.tools.masked_models.codonfm.codonfm_sample import (
            CodonFMSampleConfig,
            CodonFMSampleInput,
            run_codonfm_sample,
        )

        return run_tool_call(
            run_codonfm_sample,
            CodonFMSampleInput,
            CodonFMSampleConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def score(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score reference-versus-alternate codon substitutions with Encodon."""
        from proto_tools.tools.masked_models.codonfm.codonfm_score import (
            CodonFMScoreConfig,
            CodonFMScoreInput,
            run_codonfm_score,
        )

        return run_tool_call(
            run_codonfm_score,
            CodonFMScoreInput,
            CodonFMScoreConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )
