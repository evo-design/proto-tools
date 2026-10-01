"""Minerva Modal service.

Delegates to proto-tools ``run_minerva_*`` functions for parameter validation, environment setup,
and model inference. The build-time warmup creates the shared ``glm2_base`` env and downloads the
default checkpoint, persisted on the Modal volume via ``PROTO_MODEL_CACHE``. The 8k-context
checkpoint downloads on first use.
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
    from proto_tools.tools.masked_models.minerva import minerva_embeddings

    minerva_embeddings.run_minerva_embeddings(
        minerva_embeddings.example_input(), minerva_embeddings.MinervaEmbeddingsConfig()
    )


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("MinervaService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["MinervaService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools(
    {
        "minerva-embedding": "inference",
        "minerva-score": "score",
        "minerva-sample": "sample",
        "minerva-gradient": "gradient",
        "minerva-interactions": "interactions",
    }
)
class MinervaService:
    """Modal service for Minerva mixed protein/DNA masked language model inference and interaction maps."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("minerva")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("minerva")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def inference(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run Minerva embedding inference on mixed protein/DNA loci.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Mean embeddings and optional per-position logits.
        """
        from proto_tools.tools.masked_models.minerva import (
            MinervaEmbeddingsConfig,
            MinervaEmbeddingsInput,
            run_minerva_embeddings,
        )

        return run_tool_call(
            run_minerva_embeddings,
            MinervaEmbeddingsInput,
            MinervaEmbeddingsConfig,
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
        from proto_tools.tools.masked_models.minerva import MinervaScoringConfig, MinervaScoringInput, run_minerva_score

        return run_tool_call(
            run_minerva_score,
            MinervaScoringInput,
            MinervaScoringConfig,
            input_dict,
            config_dict,
            instance=self.instance,
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
        from proto_tools.tools.masked_models.minerva import MinervaSampleConfig, MinervaSampleInput, run_minerva_sample

        return run_tool_call(
            run_minerva_sample, MinervaSampleInput, MinervaSampleConfig, input_dict, config_dict, instance=self.instance
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
        from proto_tools.tools.masked_models.minerva import (
            MinervaGradientConfig,
            MinervaGradientInput,
            run_minerva_gradient,
        )

        return run_tool_call(
            run_minerva_gradient,
            MinervaGradientInput,
            MinervaGradientConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def interactions(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Predict base-pairing, protein-contact, and repeat interaction maps.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-locus interaction maps with axis labels.
        """
        from proto_tools.tools.masked_models.minerva import (
            MinervaInteractionsConfig,
            MinervaInteractionsInput,
            run_minerva_interactions,
        )

        return run_tool_call(
            run_minerva_interactions,
            MinervaInteractionsInput,
            MinervaInteractionsConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )
