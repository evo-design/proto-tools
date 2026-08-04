"""CCD lookup Modal service.

Delegates to proto-tools ``run_ccd_lookup`` for parameter validation, environment setup, and
retrieval. The build-time warmup builds the tool env and stages the Chemical Component Dictionary
under ``PROTO_MODEL_CACHE``, so the first call does not pay for the download. CPU only.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import CPU_BASE, with_proto_tools
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, env_for, run_tool_call


def _warmup() -> None:
    """Deploy-time: build the env and stage the dictionary."""
    from proto_tools.tools.database_retrieval.ccd_lookup.ccd_lookup import example_input, run_ccd_lookup

    run_ccd_lookup(example_input())


image = (
    with_proto_tools(CPU_BASE)
    .env(env_for())
    .run_function(_warmup, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False)
    .env(RUNTIME_ENV)
)


app = get_app_for_service("CcdLookupService")


@app.cls(
    include_source=False,
    image=image,
    cpu=2,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["CcdLookupService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"ccd-lookup": "lookup"})
class CcdLookupService:
    """Modal service for Chemical Component Dictionary lookups."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the dictionary stays parsed across requests."""
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("ccd_lookup")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def lookup(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Look up ligand definitions in the Chemical Component Dictionary.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-identifier chemical component records.
        """
        from proto_tools.tools.database_retrieval.ccd_lookup.ccd_lookup import (
            CcdLookupConfig,
            CcdLookupInput,
            run_ccd_lookup,
        )

        return run_tool_call(
            run_ccd_lookup, CcdLookupInput, CcdLookupConfig, input_dict, config_dict, instance=self.instance
        )
