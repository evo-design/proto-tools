"""AlphaGenome Modal service.

Delegates to proto-tools' six AlphaGenome entry points for parameter validation, environment
setup, and model inference. The build-time warmup creates the tool env and downloads the
``google/alphagenome-all-folds`` checkpoint into the Modal volume via ``PROTO_MODEL_CACHE``,
so the first call after a deploy pays for neither. Those weights are gated on HuggingFace, so
the build needs a token.
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
    """Deploy-time: build the env and pull the gated checkpoint onto the volume."""
    from proto_tools.tools.sequence_scoring.alphagenome.alphagenome_predict_sequences import (
        AlphaGenomePredictSequencesConfig,
        example_input,
        run_alphagenome_predict_sequences,
    )

    run_alphagenome_predict_sequences(example_input(), AlphaGenomePredictSequencesConfig())


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("AlphaGenomeService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["AlphaGenomeService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools(
    {
        "alphagenome-predict-intervals": "predict_intervals",
        "alphagenome-predict-sequences": "predict_sequences",
        "alphagenome-predict-variants": "predict_variants",
        "alphagenome-score-intervals": "score_intervals",
        "alphagenome-score-ism-variants-batch": "score_ism_variants",
        "alphagenome-score-variants": "score_variants",
    }
)
class AlphaGenomeService:
    """Modal service for AlphaGenome regulatory-genomics prediction and scoring."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("alphagenome")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("alphagenome")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def predict_intervals(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Predict regulatory tracks across genomic intervals.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-interval track predictions.
        """
        from proto_tools.tools.sequence_scoring.alphagenome.alphagenome_predict_intervals import (
            AlphaGenomePredictIntervalsConfig,
            AlphaGenomePredictIntervalsInput,
            run_alphagenome_predict_intervals,
        )

        return run_tool_call(
            run_alphagenome_predict_intervals,
            AlphaGenomePredictIntervalsInput,
            AlphaGenomePredictIntervalsConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def predict_sequences(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Predict regulatory tracks directly from supplied sequences.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-sequence track predictions.
        """
        from proto_tools.tools.sequence_scoring.alphagenome.alphagenome_predict_sequences import (
            AlphaGenomePredictSequencesConfig,
            AlphaGenomePredictSequencesInput,
            run_alphagenome_predict_sequences,
        )

        return run_tool_call(
            run_alphagenome_predict_sequences,
            AlphaGenomePredictSequencesInput,
            AlphaGenomePredictSequencesConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def predict_variants(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Predict regulatory tracks for reference and alternate alleles.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-variant reference and alternate track predictions.
        """
        from proto_tools.tools.sequence_scoring.alphagenome.alphagenome_predict_variants import (
            AlphaGenomePredictVariantsConfig,
            AlphaGenomePredictVariantsInput,
            run_alphagenome_predict_variants,
        )

        return run_tool_call(
            run_alphagenome_predict_variants,
            AlphaGenomePredictVariantsInput,
            AlphaGenomePredictVariantsConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def score_intervals(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Summarize predicted tracks over genomic intervals into scores.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-interval scores.
        """
        from proto_tools.tools.sequence_scoring.alphagenome.alphagenome_score_intervals import (
            AlphaGenomeScoreIntervalsConfig,
            AlphaGenomeScoreIntervalsInput,
            run_alphagenome_score_intervals,
        )

        return run_tool_call(
            run_alphagenome_score_intervals,
            AlphaGenomeScoreIntervalsInput,
            AlphaGenomeScoreIntervalsConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def score_ism_variants(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score an in-silico mutagenesis batch of variants.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-variant ISM scores.
        """
        from proto_tools.tools.sequence_scoring.alphagenome.alphagenome_score_ism_variants_batch import (
            AlphaGenomeScoreISMConfig,
            AlphaGenomeScoreISMInput,
            run_alphagenome_score_ism_variants_batch,
        )

        return run_tool_call(
            run_alphagenome_score_ism_variants_batch,
            AlphaGenomeScoreISMInput,
            AlphaGenomeScoreISMConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def score_variants(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score variants by the effect of the alternate allele on predicted tracks.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-variant effect scores.
        """
        from proto_tools.tools.sequence_scoring.alphagenome.alphagenome_score_variants import (
            AlphaGenomeScoreVariantsConfig,
            AlphaGenomeScoreVariantsInput,
            run_alphagenome_score_variants,
        )

        return run_tool_call(
            run_alphagenome_score_variants,
            AlphaGenomeScoreVariantsInput,
            AlphaGenomeScoreVariantsConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )
