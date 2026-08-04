"""Promoter Calculator Modal service.

Delegates to proto-tools ``run_promoter_calculator`` for parameter validation, environment setup,
and promoter scoring. The package installs from bioconda at build time, so the warmup pays for the
env once and the image layer keeps it. CPU only, with no model weights to stage.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import CPU_BASE, with_proto_tools
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, env_for, run_tool_call


def _warmup() -> None:
    """Deploy-time: build the env so the conda install is not paid for on first call."""
    from proto_tools.tools.gene_annotation.promoter_calculator.promoter_calculator import (
        example_input,
        run_promoter_calculator,
    )

    run_promoter_calculator(example_input())


image = (
    with_proto_tools(CPU_BASE)
    .env(env_for())
    .run_function(_warmup, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False)
    .env(RUNTIME_ENV)
)


app = get_app_for_service("PromoterCalculatorService")


@app.cls(
    include_source=False,
    image=image,
    cpu=4,
    timeout=SERVICE_MODAL_TIMEOUTS["PromoterCalculatorService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"promoter-calculator": "calculate"})
class PromoterCalculatorService:
    """Modal service for bacterial promoter strength prediction."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the env stays warm across requests."""
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("promoter_calculator")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def calculate(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Predict sigma70 promoter strength across nucleotide sequences.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-sequence promoter predictions with transcription rates.
        """
        from proto_tools.tools.gene_annotation.promoter_calculator.promoter_calculator import (
            PromoterCalculatorConfig,
            PromoterCalculatorInput,
            run_promoter_calculator,
        )

        return run_tool_call(
            run_promoter_calculator,
            PromoterCalculatorInput,
            PromoterCalculatorConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )
