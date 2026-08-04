"""ViennaRNA Modal service.

Delegates to proto-tools ``run_viennarna`` for parameter validation, environment setup, and RNA
secondary-structure folding. ViennaRNA installs from PyPI at build time, so the warmup pays for
the env once and the image layer keeps it. CPU only, with no model weights to stage.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import CPU_BASE, with_proto_tools
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, dispatch_tool_call, env_for


def _warmup() -> None:
    """Deploy-time: build the env so the install is not paid for on first call."""
    from proto_tools.tools.structure_prediction.viennarna.viennarna import example_input, run_viennarna

    run_viennarna(example_input())


image = (
    with_proto_tools(CPU_BASE)
    .env(env_for())
    .run_function(_warmup, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False)
    .env(RUNTIME_ENV)
)


app = get_app_for_service("ViennaRNAService")


@app.cls(
    include_source=False,
    image=image,
    cpu=2,
    timeout=SERVICE_MODAL_TIMEOUTS["ViennaRNAService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"viennarna-prediction": "predict"})
class ViennaRNAService:
    """Modal service for ViennaRNA secondary-structure prediction."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the env stays warm across requests."""
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("viennarna")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def predict(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Fold RNA sequences into minimum-free-energy secondary structures.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-sequence structures and free energies.
        """
        from proto_tools.tools.structure_prediction.viennarna.viennarna import (
            ViennaRNAConfig,
            ViennaRNAInput,
            run_viennarna,
        )

        inputs = ViennaRNAInput(**input_dict)
        config = ViennaRNAConfig(**config_dict)
        return dispatch_tool_call(run_viennarna, inputs, config, instance=self.instance)
