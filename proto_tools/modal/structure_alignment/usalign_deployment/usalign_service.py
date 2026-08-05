"""USalign Modal service.

Delegates to proto-tools ``run_usalign`` for parameter validation, binary
compilation, and structure alignment. Supports monomers, multimers, and
nucleic acids. The build-time warmup installs the C++ binary and runs a
test alignment, both persisted in the image.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import CPU_BASE, with_proto_tools
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, env_for, run_tool_call, stage_all_excluded_fixtures


def _usalign_warmup() -> None:
    """Deploy-time: warm the default USalign config."""
    from proto_tools.tools.structure_alignment.usalign.usalign import example_input, run_usalign

    run_usalign(example_input())


usalign_image = with_proto_tools(CPU_BASE).env(env_for())
usalign_image = (
    stage_all_excluded_fixtures(usalign_image)
    .run_function(_usalign_warmup, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET])
    .env(RUNTIME_ENV)
)


app = get_app_for_service("USalignService")


@app.cls(
    include_source=False,
    image=usalign_image,
    cpu=2,
    timeout=SERVICE_MODAL_TIMEOUTS["USalignService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"usalign-alignment": "align"})
class USalignService:
    """Modal service for USalign universal structure alignment."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the USalign binary stays warm across requests."""
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("usalign")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def align(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run USalign universal structure alignment.

        Supports monomers, multimers, and nucleic acids.

        Args:
            input_dict (dict[str, Any]): Mapping of input field names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: TM-scores normalized by each structure's length.
        """
        from proto_tools.tools.structure_alignment.usalign.usalign import (
            USalignConfig,
            USalignInput,
            run_usalign,
        )

        return run_tool_call(run_usalign, USalignInput, USalignConfig, input_dict, config_dict, instance=self.instance)
