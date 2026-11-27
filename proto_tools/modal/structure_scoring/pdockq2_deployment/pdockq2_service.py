"""pDockQ2 Modal service.

Like structure-metrics, this tool has no standalone environment and no model — it scores an
interface arithmetically from pLDDT and PAE already attached to the structure. The container
imports proto-tools and calls the function; there is no env to build and no worker to hold open.

Note this tool is not iterable, so a call is one structure and gains no fan-out from being here.
It is deployed so that a caller working against a remote device can reach it alongside the tools
that do benefit, rather than having one step of a scoring pipeline behave differently from the rest.
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
    from proto_tools.tools.structure_scoring.pdockq2.pdockq2 import example_input, run_pdockq2

    run_pdockq2(example_input())


image = (
    with_proto_tools(CPU_BASE)
    .env(env_for())
    .run_function(_warmup, secrets=[HF_TOKEN_SECRET], include_source=False)
    .env(RUNTIME_ENV)
)


app = get_app_for_service("PDockQ2Service")


@app.cls(
    include_source=False,
    image=image,
    cpu=2,
    timeout=SERVICE_MODAL_TIMEOUTS["PDockQ2Service"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"pdockq2": "score"})
class PDockQ2Service:
    """Modal service for pDockQ2 interface quality scoring."""

    @modal.method()
    def score(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score a complex interface with pDockQ2.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Overall pDockQ2 plus the per-chain interface breakdown.
        """
        from proto_tools.tools.structure_scoring.pdockq2.pdockq2 import (
            PDockQ2Config,
            PDockQ2Input,
            run_pdockq2,
        )

        return run_tool_call(run_pdockq2, PDockQ2Input, PDockQ2Config, input_dict, config_dict)
