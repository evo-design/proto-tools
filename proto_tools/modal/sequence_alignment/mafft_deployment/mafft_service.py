"""MAFFT alignment Modal service.

Delegates to proto-tools ``run_mafft_align`` for parameter validation,
environment setup, and binary installation.  The build-time warmup installs
the MAFFT binary and runs a test alignment, both persisted in the image layer.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import CPU_BASE, with_proto_tools
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, dispatch_tool_call, env_for


def _warmup() -> None:
    """Deploy-time: warm the default config."""
    from proto_tools.tools.sequence_alignment.mafft.mafft import (
        example_input,
        run_mafft_align,
    )

    run_mafft_align(example_input())


image = (
    with_proto_tools(CPU_BASE)
    .env(env_for())
    .run_function(_warmup, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False)
    .env(RUNTIME_ENV)
)


app = get_app_for_service("MafftAlignService")


@app.cls(
    include_source=False,
    image=image,
    cpu=2,
    timeout=SERVICE_MODAL_TIMEOUTS["MafftAlignService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"mafft-align": "align"})
class MafftAlignService:
    """Modal service for MAFFT multiple sequence alignment."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the MAFFT binary stays warm across requests."""
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("mafft")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def align(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run MAFFT multiple sequence alignment.

        Args:
            input_dict (dict[str, Any]): Mapping of input field names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Aligned sequences with alignment metadata.
        """
        from proto_tools.tools.sequence_alignment.mafft.mafft import (
            MafftConfig,
            MafftInput,
            run_mafft_align,
        )

        inputs = MafftInput(**input_dict)
        config = MafftConfig(**config_dict)
        return dispatch_tool_call(run_mafft_align, inputs, config, instance=self.instance)
