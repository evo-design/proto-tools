"""DSSP Secondary Structure Modal service.

Delegates to proto-tools ``run_dssp_secondary_structure`` for parameter
validation, environment setup, and DSSP helix/sheet/loop assignment from a
PDB structure.  The build-time warmup creates the tool env and verifies
inference on a minimal inline PDB.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import CPU_BASE, with_proto_tools
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import (
    RUNTIME_ENV,
    env_for,
    run_tool_call,
    stage_all_excluded_fixtures,
)


def _warmup() -> None:
    """Deploy-time: warm the default config."""
    from proto_tools.tools.structure_scoring.dssp.dssp import example_input, run_dssp_secondary_structure

    run_dssp_secondary_structure(example_input())


image = (
    with_proto_tools(CPU_BASE)
    # PROTO_MODEL_CACHE unused for pure-CPU tools; set for env consistency with GPU services
    .env(env_for())
)
image = (
    stage_all_excluded_fixtures(image)
    .run_function(_warmup, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False)
    .env(RUNTIME_ENV)
)


app = get_app_for_service("DSSPService")


@app.cls(
    include_source=False,
    image=image,
    cpu=2,
    timeout=SERVICE_MODAL_TIMEOUTS["DSSPService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"dssp-secondary-structure": "compute"})
class DSSPService:
    """Modal service for DSSP secondary-structure assignment from PDB structures."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the DSSP env stays loaded across requests."""
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("dssp")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def compute(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Assign DSSP helix/sheet/loop percentages from PDB structures.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-input secondary-structure percentages (helix_pct, sheet_pct, loop_pct).
        """
        from proto_tools.tools.structure_scoring.dssp.dssp import (
            DSSPSecondaryStructureConfig,
            DSSPSecondaryStructureInput,
            run_dssp_secondary_structure,
        )

        return run_tool_call(
            run_dssp_secondary_structure,
            DSSPSecondaryStructureInput,
            DSSPSecondaryStructureConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )
