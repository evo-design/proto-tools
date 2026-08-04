"""TMalign Modal service.

Delegates to proto-tools ``run_tmalign`` for parameter validation, binary
compilation, and structure alignment. The build-time warmup installs the
C++ binary and runs a test alignment, both persisted in the image.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import CPU_BASE, with_proto_tools
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import env_for, run_tool_call, stage_all_excluded_fixtures


def _tmalign_warmup() -> None:
    """Deploy-time: warm the default TMalign config."""
    from proto_tools.tools.structure_alignment.tmalign.tmalign import example_input, run_tmalign

    run_tmalign(example_input())


tmalign_image = with_proto_tools(CPU_BASE).env(env_for())
tmalign_image = stage_all_excluded_fixtures(tmalign_image).run_function(
    _tmalign_warmup, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET]
)


app = get_app_for_service("TMalignService")


@app.cls(
    include_source=False,
    image=tmalign_image,
    cpu=2,
    timeout=SERVICE_MODAL_TIMEOUTS["TMalignService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"tmalign-alignment": "align"})
class TMalignService:
    """Modal service for TMalign pairwise protein structure alignment."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the TMalign binary stays warm across requests."""
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("tmalign")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def align(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run TMalign pairwise protein structure alignment.

        Args:
            input_dict (dict[str, Any]): Mapping of input field names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: TM-scores normalized by each structure's length.
        """
        from proto_tools.tools.structure_alignment.tmalign.tmalign import (
            TMalignConfig,
            TMalignInput,
            run_tmalign,
        )

        return run_tool_call(run_tmalign, TMalignInput, TMalignConfig, input_dict, config_dict, instance=self.instance)
