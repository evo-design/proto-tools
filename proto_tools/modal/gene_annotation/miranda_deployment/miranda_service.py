"""miRanda Modal service.

Delegates to proto-tools ``run_miranda_scan`` for parameter validation, environment setup, and
target scanning. The miRanda binary installs from bioconda at build time, so the warmup pays for
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
    """Deploy-time: build the env so the conda install is not paid for on first call."""
    from proto_tools.tools.gene_annotation.miranda.miranda_scan import (
        MirandaConfig,
        example_input,
        run_miranda_scan,
    )

    # `mirna_queries` has no default and dispatch refuses without it; `minimal()` supplies the
    # bundled bantam query, so the warmup is a real scan rather than a validation error.
    run_miranda_scan(example_input(), MirandaConfig.minimal())


image = (
    with_proto_tools(CPU_BASE)
    .env(env_for())
    .run_function(_warmup, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False)
    .env(RUNTIME_ENV)
)


app = get_app_for_service("MirandaService")


@app.cls(
    include_source=False,
    image=image,
    cpu=2,
    timeout=SERVICE_MODAL_TIMEOUTS["MirandaService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"miranda-scan": "scan"})
class MirandaService:
    """Modal service for miRanda microRNA target scanning."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the binary stays warm across requests."""
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("miranda")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def scan(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Scan target sequences for microRNA binding sites.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-target hits with alignment scores and energies.
        """
        from proto_tools.tools.gene_annotation.miranda.miranda_scan import (
            MirandaConfig,
            MirandaInput,
            run_miranda_scan,
        )

        inputs = MirandaInput(**input_dict)
        config = MirandaConfig(**config_dict)
        return dispatch_tool_call(run_miranda_scan, inputs, config, instance=self.instance)
