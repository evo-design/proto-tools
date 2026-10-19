"""Caliby Modal service.

Delegates to proto-tools ``run_caliby_*`` functions for parameter validation,
environment setup, and model inference. The tool env's ``setup.sh`` downloads every
bundled checkpoint, design and packer alike, so the build-time warmup only has to
create the env and load one model; the packer weights are already on the Modal volume
via ``PROTO_MODEL_CACHE`` when ``pack`` first runs.

All three tools share one service because they share a model family, a tool env, and a
persistent worker, so splitting them would download the same checkpoints twice.
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
    ensure_gpu_ready,
    env_for,
    run_tool_call,
    stage_all_excluded_fixtures,
)


def _warmup() -> None:
    """Deploy-time: build the env and warm the default design checkpoint."""
    from proto_tools.tools.inverse_folding.caliby.caliby_sample import (
        CalibySampleConfig,
        example_input,
        run_caliby_sample,
    )

    run_caliby_sample(example_input(), CalibySampleConfig())


image = with_proto_tools(GPU_BASE).env(env_for())
image = (
    stage_all_excluded_fixtures(image)
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("CalibyService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["CalibyService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools(
    {
        "caliby-sample": "sample",
        "caliby-score": "score",
        "caliby-pack": "pack",
    }
)
class CalibyService:
    """Modal service for Caliby Potts-model sequence design, scoring, and packing."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("caliby")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("caliby")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def sample(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Design sequences for a backbone, optionally conditioned on a conformer ensemble.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Designed sequences with Potts energies and sequence recovery.
        """
        from proto_tools.tools.inverse_folding.caliby.caliby_sample import (
            CalibySampleConfig,
            CalibySampleInput,
            run_caliby_sample,
        )

        return run_tool_call(
            run_caliby_sample, CalibySampleInput, CalibySampleConfig, input_dict, config_dict, instance=self.instance
        )

    @modal.method()
    def score(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score the sequence a structure already carries under the Potts model.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Potts energies, optionally with local conditional energies.
        """
        from proto_tools.tools.inverse_folding.caliby.caliby_score import (
            CalibyScoreConfig,
            CalibyScoreInput,
            run_caliby_score,
        )

        return run_tool_call(
            run_caliby_score, CalibyScoreInput, CalibyScoreConfig, input_dict, config_dict, instance=self.instance
        )

    @modal.method()
    def pack(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Pack sidechains onto a backbone with the diffusion packer.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Packed full-atom structures, one per input structure.
        """
        from proto_tools.tools.inverse_folding.caliby.caliby_pack import (
            CalibyPackConfig,
            CalibyPackInput,
            run_caliby_pack,
        )

        return run_tool_call(
            run_caliby_pack, CalibyPackInput, CalibyPackConfig, input_dict, config_dict, instance=self.instance
        )
