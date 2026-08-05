"""SpliceAI Modal service.

Delegates to proto-tools ``run_spliceai_predict`` / ``run_spliceai_score`` for parameter
validation, environment setup, and inference. The build-time warmup goes through the prediction
tool, which needs only the bundled model weights.

``spliceai-score`` additionally needs a reference genome. The build stages every assembly the
config offers by name onto the model-cache volume, so a call naming one finds it already there.
Staging at build rather than on first use is deliberate: provisioning mutates process-global state
and is not safe to run from several request-handling containers racing on a shared volume.

A path in ``reference_fasta`` is refused for a remote device by the config's own
``remote_unsupported_reason`` — it would name a file on the caller's machine, not this container.
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
    """Deploy-time: warm the model, then stage every named assembly onto the volume."""
    from proto_tools.tools.rna_splicing.spliceai.spliceai_predict import (
        SpliceAIPredictConfig,
        example_input,
        run_spliceai_predict,
    )
    from proto_tools.tools.rna_splicing.spliceai.spliceai_score import (
        _GENOME_FASTA,
        SpliceAIScoreConfig,
        SpliceAIScoreInput,
        SpliceAIVariant,
        run_spliceai_score,
    )

    run_spliceai_predict(example_input(), SpliceAIPredictConfig())

    probe = SpliceAIScoreInput(variants=[SpliceAIVariant(chromosome="1", position=100_000, ref="A", alt="C")])
    for assembly in sorted(_GENOME_FASTA):
        run_spliceai_score(probe, SpliceAIScoreConfig(reference_fasta=assembly))


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("SpliceAIService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["SpliceAIService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools(
    {
        "spliceai-predict": "predict",
        "spliceai-score": "score",
    }
)
class SpliceAIService:
    """Modal service for SpliceAI splice-site prediction and variant scoring."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("spliceai")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("spliceai")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def predict(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Predict splice acceptor and donor probabilities along sequences.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-position splice-site probabilities.
        """
        from proto_tools.tools.rna_splicing.spliceai.spliceai_predict import (
            SpliceAIPredictConfig,
            SpliceAIPredictInput,
            run_spliceai_predict,
        )

        return run_tool_call(
            run_spliceai_predict,
            SpliceAIPredictInput,
            SpliceAIPredictConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def score(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score variants for their effect on splicing.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-variant delta scores and positions.
        """
        from proto_tools.tools.rna_splicing.spliceai.spliceai_score import (
            SpliceAIScoreConfig,
            SpliceAIScoreInput,
            run_spliceai_score,
        )

        return run_tool_call(
            run_spliceai_score, SpliceAIScoreInput, SpliceAIScoreConfig, input_dict, config_dict, instance=self.instance
        )
