"""Foldmason Modal service.

Delegates to proto-tools ``run_foldmason_msa`` / ``run_foldmason_score_msa`` for parameter
validation, environment setup, and alignment. Foldmason installs from bioconda at build time, so
the warmup pays for the env once and the image layer keeps it. CPU only, with no weights to stage.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import CPU_BASE, with_proto_tools
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, dispatch_tool_call, env_for


def _warmup() -> None:
    """Deploy-time: build the env so the conda install is not paid for on first call."""
    from proto_tools.tools.structure_alignment.foldmason.foldmason_msa import example_input, run_foldmason_msa

    run_foldmason_msa(example_input())


image = (
    with_proto_tools(CPU_BASE)
    .env(env_for())
    .run_function(_warmup, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False)
    .env(RUNTIME_ENV)
)


app = get_app_for_service("FoldmasonService")


@app.cls(
    include_source=False,
    image=image,
    cpu=4,
    timeout=SERVICE_MODAL_TIMEOUTS["FoldmasonService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools(
    {
        "foldmason-msa": "msa",
        "foldmason-score-msa": "score_msa",
    }
)
class FoldmasonService:
    """Modal service for Foldmason structural multiple-sequence alignment."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the binary stays warm across requests."""
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("foldmason")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def msa(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Build a structural multiple-sequence alignment from input structures.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: The alignment and its per-structure records.
        """
        from proto_tools.tools.structure_alignment.foldmason.foldmason_msa import (
            FoldmasonMSAConfig,
            FoldmasonMSAInput,
            run_foldmason_msa,
        )

        inputs = FoldmasonMSAInput(**input_dict)
        config = FoldmasonMSAConfig(**config_dict)
        return dispatch_tool_call(run_foldmason_msa, inputs, config, instance=self.instance)

    @modal.method()
    def score_msa(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score an existing structural multiple-sequence alignment.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Alignment quality scores.
        """
        from proto_tools.tools.structure_alignment.foldmason.foldmason_score_msa import (
            FoldmasonScoreMSAConfig,
            FoldmasonScoreMSAInput,
            run_foldmason_score_msa,
        )

        inputs = FoldmasonScoreMSAInput(**input_dict)
        config = FoldmasonScoreMSAConfig(**config_dict)
        return dispatch_tool_call(run_foldmason_score_msa, inputs, config, instance=self.instance)
