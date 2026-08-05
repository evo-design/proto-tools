"""ORF Prediction Modal services.

Delegates to proto-tools ``run_orfipy_prediction`` for parameter validation,
environment setup, and ORF prediction.  The build-time warmup creates the
tool env and verifies inference.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import CPU_BASE, with_proto_tools
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, env_for, run_tool_call


# =============================================================================
# Orfipy
# =============================================================================
def _warmup_orfipy() -> None:
    """Deploy-time: warm the default Orfipy config."""
    from proto_tools.tools.orf_prediction.orfipy.orfipy import (
        example_input,
        run_orfipy_prediction,
    )

    run_orfipy_prediction(example_input())


orfipy_image = (
    with_proto_tools(CPU_BASE)
    # PROTO_MODEL_CACHE unused for pure-CPU tools; set for env consistency with GPU services
    .env(env_for())
    .run_function(_warmup_orfipy, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False)
    .env(RUNTIME_ENV)
)


app = get_app_for_service("OrfipyService")


@app.cls(
    include_source=False,
    image=orfipy_image,
    cpu=2,
    timeout=SERVICE_MODAL_TIMEOUTS["OrfipyService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"orfipy-prediction": "predict"})
class OrfipyService:
    """Modal service for Orfipy ORF prediction."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the tool env stays loaded across requests."""
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("orfipy")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def predict(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run Orfipy ORF prediction on one or more DNA sequences.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Predicted ORFs for each input sequence.
        """
        from proto_tools.tools.orf_prediction.orfipy import (
            OrfipyConfig,
            OrfipyInput,
            run_orfipy_prediction,
        )

        return run_tool_call(
            run_orfipy_prediction, OrfipyInput, OrfipyConfig, input_dict, config_dict, instance=self.instance
        )
