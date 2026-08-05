"""Segmasker Modal service.

Delegates to proto-tools ``run_segmasker`` for parameter validation,
environment setup, and low-complexity region detection.  The build-time
warmup creates the tool env and verifies inference.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import CPU_BASE, with_proto_tools
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, env_for, run_tool_call


def _warmup() -> None:
    """Deploy-time: warm the default config."""
    from proto_tools.tools.sequence_scoring.segmasker.segmasker import (
        example_input,
        run_segmasker,
    )

    run_segmasker(example_input())


image = (
    with_proto_tools(CPU_BASE)
    # PROTO_MODEL_CACHE unused for pure-CPU tools; set for env consistency with GPU services
    .env(env_for())
    .run_function(_warmup, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False)
    .env(RUNTIME_ENV)
)


app = get_app_for_service("SegmaskerService")


@app.cls(
    include_source=False,
    image=image,
    cpu=2,
    timeout=SERVICE_MODAL_TIMEOUTS["SegmaskerService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"segmasker-score": "score"})
class SegmaskerService:
    """Modal service for segmasker low-complexity region detection."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the tool env stays loaded across requests."""
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("segmasker")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def score(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Detect low-complexity regions in protein sequences using segmasker.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Segmasker scoring results with fractions and masked regions.
        """
        from proto_tools.tools.sequence_scoring.segmasker import (
            SegmaskerConfig,
            SegmaskerInput,
            run_segmasker,
        )

        return run_tool_call(
            run_segmasker, SegmaskerInput, SegmaskerConfig, input_dict, config_dict, instance=self.instance
        )
