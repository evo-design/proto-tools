"""FAMPNN Modal service.

Delegates to proto-tools ``run_fampnn_*`` functions for parameter validation,
environment setup, and model inference.  The build-time warmup creates the
tool env and downloads weights for all three checkpoints (design 0.3, packing
0.0, scoring 0.3_cath), persisted on the Modal volume via ``PROTO_MODEL_CACHE``.

The fampnn standalone env declares a custom ``LD_LIBRARY_PATH`` via
``standalone/env_vars.txt``; ``ToolInstance`` applies it to the subprocess
automatically — no service-side handling needed.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SCALEDOWN_WINDOW, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import GPU_BASE, with_proto_tools
from proto_tools.modal.gpu_profiles import GPU_DEFAULT
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import (
    RUNTIME_ENV,
    dispatch_tool_call,
    ensure_gpu_ready,
    env_for,
    stage_all_excluded_fixtures,
)


def _warmup() -> None:
    """Deploy-time: warm only the default sample checkpoint (0.3)."""
    from proto_tools.tools.inverse_folding.fampnn.fampnn_sample import (
        FAMPNNSampleConfig,
        example_input,
        run_fampnn_sample,
    )

    run_fampnn_sample(example_input(), FAMPNNSampleConfig())


image = with_proto_tools(GPU_BASE).env(env_for())
image = (
    stage_all_excluded_fixtures(image)
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("FAMPNNService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["FAMPNNService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools(
    {
        "fampnn-sample": "sample",
        "fampnn-score": "score",
        "fampnn-score-all-mutations": "score_all_mutations",
        "fampnn-pack": "pack",
    }
)
class FAMPNNService:
    """Modal service for FAMPNN inverse folding, packing, and scoring."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("fampnn")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("fampnn")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def sample(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Sample sequences from FAMPNN.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Sampled sequences with per-position metrics.
        """
        from proto_tools.tools.inverse_folding.fampnn.fampnn_sample import (
            FAMPNNSampleConfig,
            FAMPNNSampleInput,
            run_fampnn_sample,
        )

        inputs = FAMPNNSampleInput(**input_dict)
        config = FAMPNNSampleConfig(**config_dict)
        return dispatch_tool_call(run_fampnn_sample, inputs, config, instance=self.instance)

    @modal.method()
    def score(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score specific mutations against a backbone with FAMPNN.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Mutation scores and metrics.
        """
        from proto_tools.tools.inverse_folding.fampnn.fampnn_score import (
            FAMPNNScoreConfig,
            FAMPNNScoreInput,
            run_fampnn_score,
        )

        inputs = FAMPNNScoreInput(**input_dict)
        config = FAMPNNScoreConfig(**config_dict)
        return dispatch_tool_call(run_fampnn_score, inputs, config, instance=self.instance)

    @modal.method()
    def score_all_mutations(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Exhaustively score every single-residue mutation with FAMPNN.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: 20-by-L log-likelihood-ratio grid keyed by position label.
        """
        from proto_tools.tools.inverse_folding.fampnn.fampnn_score_all_mutations import (
            FAMPNNScoreAllMutationsConfig,
            FAMPNNScoreAllMutationsInput,
            run_fampnn_score_all_mutations,
        )

        inputs = FAMPNNScoreAllMutationsInput(**input_dict)
        config = FAMPNNScoreAllMutationsConfig(**config_dict)
        return dispatch_tool_call(run_fampnn_score_all_mutations, inputs, config, instance=self.instance)

    @modal.method()
    def pack(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Pack sidechains onto a fixed backbone with FAMPNN.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Repacked structures and per-residue confidence.
        """
        from proto_tools.tools.inverse_folding.fampnn.fampnn_pack import (
            FAMPNNPackConfig,
            FAMPNNPackInput,
            run_fampnn_pack,
        )

        inputs = FAMPNNPackInput(**input_dict)
        config = FAMPNNPackConfig(**config_dict)
        return dispatch_tool_call(run_fampnn_pack, inputs, config, instance=self.instance)
