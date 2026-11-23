"""Structure-metrics Modal service.

The tool computes geometry directly from an input structure. It has no standalone
environment, model weights, or persistent worker. The container imports proto-tools
and calls the function.

Deployment lets large batches run across containers, with up to 256 structures per
chunk. A single call is faster locally. Remote dispatch uses this service when it
is deployed and otherwise falls back to running in the caller's process.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import CPU_BASE, with_proto_tools
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, env_for, run_tool_call


def _warmup() -> None:
    """Deploy-time: run the tool once so a broken image fails the build rather than the first call."""
    from proto_tools.tools.structure_scoring.structure_metrics.structure_metrics import (
        example_input,
        run_structure_metrics,
    )

    run_structure_metrics(example_input())


image = (
    with_proto_tools(CPU_BASE)
    .env(env_for())
    .run_function(_warmup, secrets=[HF_TOKEN_SECRET], include_source=False)
    .env(RUNTIME_ENV)
)


app = get_app_for_service("StructureMetricsService")


@app.cls(
    include_source=False,
    image=image,
    cpu=2,
    timeout=SERVICE_MODAL_TIMEOUTS["StructureMetricsService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"structure-metrics": "compute"})
class StructureMetricsService:
    """Modal service for structural quality metrics."""

    @modal.method()
    def compute(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Compute secondary-structure percentages, longest helix, and gyration radius.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-structure quality metrics, aligned with the input structures.
        """
        from proto_tools.tools.structure_scoring.structure_metrics.structure_metrics import (
            StructureMetricsConfig,
            StructureMetricsInput,
            run_structure_metrics,
        )

        return run_tool_call(
            run_structure_metrics, StructureMetricsInput, StructureMetricsConfig, input_dict, config_dict
        )
