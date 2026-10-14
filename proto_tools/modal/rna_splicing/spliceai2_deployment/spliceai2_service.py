"""SpliceAI2 Modal service.

Wraps ``run_spliceai2_predict`` and ``run_spliceai2_score``, which handle validation, environment
setup, and inference.

Weights: the ``illumina-ai/SpliceAI2`` checkpoints are gated on HuggingFace. At build time, the
warmup downloads them with the deployer's HF token (the ``HF_TOKEN_SECRET`` Modal secret) into the
deployer's model-cache volume (``PROTO_MODEL_CACHE``).

Reference genomes: ``spliceai2-score`` also needs a reference genome. At build time, a probe scoring
call downloads each assembly the config offers by name to the volume and builds its FASTA index.
Doing this once at build keeps concurrent containers from racing to set up the same files on the
shared volume.

``reference_fasta`` paths are rejected for remote devices, since a path would point to a file on
the caller's machine, not in this container.
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
    """Deploy-time: pull the gated weights, then stage and index every named assembly."""
    from proto_tools.tools.rna_splicing.reference_genome import GENOME_FASTA
    from proto_tools.tools.rna_splicing.spliceai2 import (
        SpliceAI2PredictConfig,
        SpliceAI2ScoreConfig,
        SpliceAI2ScoreInput,
        SpliceAI2Variant,
        run_spliceai2_predict,
        run_spliceai2_score,
    )
    from proto_tools.tools.rna_splicing.spliceai2.spliceai2_predict import example_input

    run_spliceai2_predict(example_input(), SpliceAI2PredictConfig())

    # 1:100000 is C in both the GRCh37 and GRCh38 Ensembl primary assemblies, so the ref check passes.
    probe = SpliceAI2ScoreInput(
        variants=[SpliceAI2Variant(chromosome="1", position=100_000, ref="C", alt="A", strand="+")]
    )
    for assembly in sorted(GENOME_FASTA):
        run_spliceai2_score(probe, SpliceAI2ScoreConfig(reference_fasta=assembly))


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("SpliceAI2Service")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["SpliceAI2Service"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"spliceai2-predict": "predict", "spliceai2-score": "score"})
class SpliceAI2Service:
    """Modal service for SpliceAI2 splice and transcript prediction and variant scoring."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("spliceai2")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("spliceai2")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def predict(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Predict splice sites, junctions, and transcripts along sequences.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-sequence splice site, junction, and transcript predictions.
        """
        from proto_tools.tools.rna_splicing.spliceai2 import (
            SpliceAI2PredictConfig,
            SpliceAI2PredictInput,
            run_spliceai2_predict,
        )

        return run_tool_call(
            run_spliceai2_predict,
            SpliceAI2PredictInput,
            SpliceAI2PredictConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def score(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score variants for their effect on splice site and junction usage.

        Shares the persistent ``spliceai2`` worker started in ``setup``.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-variant site and junction changes with summary scores.
        """
        from proto_tools.tools.rna_splicing.spliceai2 import (
            SpliceAI2ScoreConfig,
            SpliceAI2ScoreInput,
            run_spliceai2_score,
        )

        return run_tool_call(
            run_spliceai2_score,
            SpliceAI2ScoreInput,
            SpliceAI2ScoreConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )
