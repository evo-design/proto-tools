"""MinCED Modal service.

Delegates to proto-tools ``run_minced`` for parameter validation,
environment setup, and CRISPR array detection. The image installs Java
because MinCED is distributed as a JAR.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import CPU_BASE, with_proto_tools
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, env_for, run_tool_call


def _warmup() -> None:
    """Deploy-time: install MinCED and verify the Java wrapper starts."""
    from proto_tools.tools.gene_annotation.minced.minced import example_input, run_minced

    run_minced(example_input())


image = (
    with_proto_tools(CPU_BASE.apt_install("openjdk-17-jre-headless"))
    .env(env_for())
    .run_function(_warmup, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False)
    .env(RUNTIME_ENV)
)


app = get_app_for_service("MincedService")


@app.cls(
    include_source=False,
    image=image,
    cpu=2,
    timeout=SERVICE_MODAL_TIMEOUTS["MincedService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"minced-crispr": "detect"})
class MincedService:
    """Modal service for MinCED CRISPR array detection."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the MinCED wrapper stays warm across requests."""
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("minced")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def detect(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Detect CRISPR arrays in nucleotide sequences with MinCED.

        Args:
            input_dict (dict[str, Any]): Mapping of input field names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: MinCED CRISPR array detections for each input sequence.
        """
        from proto_tools.tools.gene_annotation.minced import MincedConfig, MincedInput, run_minced

        return run_tool_call(run_minced, MincedInput, MincedConfig, input_dict, config_dict, instance=self.instance)
