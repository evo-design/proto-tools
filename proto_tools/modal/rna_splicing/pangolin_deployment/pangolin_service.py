"""Pangolin Modal service.

Delegates to proto-tools ``run_pangolin_predict`` / ``run_pangolin_score_variants``
for parameter validation, environment setup, and model inference. The build-time
warmup creates the micromamba env and downloads weights, both persisted on the
Modal volume via ``PROTO_MODEL_CACHE``.
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
    """Deploy-time: warm the default config."""
    from proto_tools.tools.rna_splicing.pangolin import PangolinPredictConfig
    from proto_tools.tools.rna_splicing.pangolin.pangolin_predict import (
        example_input,
        run_pangolin_predict,
    )

    run_pangolin_predict(example_input(), PangolinPredictConfig())


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_BASIC, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("PangolinService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_BASIC,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["PangolinService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"pangolin-predict": "predict", "pangolin-score-variants": "score_variants"})
class PangolinService:
    """Modal service for Pangolin splice-site prediction and variant scoring."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("pangolin")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("pangolin")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def predict(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run Pangolin splice-site prediction.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Splice-site prediction results with metrics.
        """
        from proto_tools.tools.rna_splicing.pangolin import (
            PangolinPredictConfig,
            PangolinPredictInput,
        )
        from proto_tools.tools.rna_splicing.pangolin.pangolin_predict import run_pangolin_predict

        inputs = PangolinPredictInput(**input_dict)
        config = PangolinPredictConfig(**config_dict)
        return dispatch_tool_call(run_pangolin_predict, inputs, config, instance=self.instance)

    @modal.method()
    def score_variants(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run Pangolin variant splice-effect scoring.

        Shares the persistent ``pangolin`` worker started in ``setup``.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Variant scoring results with metrics.
        """
        from proto_tools.tools.rna_splicing.pangolin import (
            PangolinScoreVariantsConfig,
            PangolinScoreVariantsInput,
        )
        from proto_tools.tools.rna_splicing.pangolin.pangolin_score_variants import run_pangolin_score_variants

        inputs = PangolinScoreVariantsInput(**input_dict)
        config = PangolinScoreVariantsConfig(**config_dict)
        return dispatch_tool_call(run_pangolin_score_variants, inputs, config, instance=self.instance)
