"""AutoDock Vina Modal service.

Delegates to proto-tools' Vina entry point for parameter validation, environment setup, and
docking. The build-time warmup creates the tool env so the first call does not pay for it.
CPU only, with no model weights to stage: Vina's cost is the search itself, which scales with
``exhaustiveness`` and the number of ligands in the batch.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import CPU_BASE, with_proto_tools
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, env_for, run_tool_call, stage_all_excluded_fixtures


def _warmup() -> None:
    """Deploy-time: build the env so the Vina install is not paid for on first call."""
    from proto_tools.tools.molecular_docking.vina.vina_docking import example_input, run_vina_docking

    run_vina_docking(example_input())


image = (
    with_proto_tools(CPU_BASE)
    # PROTO_MODEL_CACHE unused for pure-CPU tools; set for env consistency with GPU services
    .env(env_for())
)
# example_input() reads a bundled .pdb receptor, which the wheel's package-data excludes.
image = (
    stage_all_excluded_fixtures(image)
    .run_function(_warmup, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False)
    .env(RUNTIME_ENV)
)


app = get_app_for_service("VinaService")


@app.cls(
    include_source=False,
    image=image,
    cpu=4,
    timeout=SERVICE_MODAL_TIMEOUTS["VinaService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"vina-docking": "dock"})
class VinaService:
    """Modal service for AutoDock Vina ligand docking."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the tool env is prepared once across requests."""
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("vina")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def dock(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Dock ligands into a receptor's search box.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-ligand docked poses and their scores.
        """
        from proto_tools.tools.molecular_docking.vina.vina_docking import (
            VinaDockingConfig,
            VinaDockingInput,
            run_vina_docking,
        )

        return run_tool_call(
            run_vina_docking,
            VinaDockingInput,
            VinaDockingConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )
