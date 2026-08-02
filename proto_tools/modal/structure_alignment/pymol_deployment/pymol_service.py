"""PyMOL RMSD Modal service."""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import CPU_BASE, with_proto_tools
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import (
    RUNTIME_ENV,
    dispatch_tool_call,
    env_for,
    stage_all_excluded_fixtures,
)


def _pymol_warmup() -> None:
    """Deploy-time: warm both alignment methods (cealign + align)."""
    from proto_tools.tools.structure_alignment.pymol_rmsd.pymol_rmsd import (
        PyMOLRMSDConfig,
        example_input,
        run_pymol_rmsd_alignment,
    )

    inputs = example_input()
    run_pymol_rmsd_alignment(inputs, PyMOLRMSDConfig(method="cealign"))
    run_pymol_rmsd_alignment(inputs, PyMOLRMSDConfig(method="align"))


pymol_image = with_proto_tools(CPU_BASE.apt_install("libgl1", "libegl1", "libxrender1", "libxext6", "libsm6")).env(
    env_for()
)
pymol_image = (
    stage_all_excluded_fixtures(pymol_image)
    .run_function(_pymol_warmup, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False)
    .env(RUNTIME_ENV)
)


app = get_app_for_service("PyMOLService")


@app.cls(
    include_source=False,
    image=pymol_image,
    cpu=2,
    timeout=SERVICE_MODAL_TIMEOUTS["PyMOLService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"pymol-rmsd-alignment": "align"})
class PyMOLService:
    """Modal service for PyMOL RMSD structure alignment."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so PyMOL stays warm across requests."""
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("pymol_rmsd")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def align(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run PyMOL RMSD structure alignment."""
        from proto_tools.tools.structure_alignment.pymol_rmsd.pymol_rmsd import (
            PyMOLRMSDConfig,
            PyMOLRMSDInput,
            run_pymol_rmsd_alignment,
        )

        inputs = PyMOLRMSDInput(**input_dict)
        config = PyMOLRMSDConfig(**config_dict)
        return dispatch_tool_call(run_pymol_rmsd_alignment, inputs, config, instance=self.instance)
