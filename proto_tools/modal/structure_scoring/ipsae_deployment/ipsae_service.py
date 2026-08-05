"""IPSAE Modal service."""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import CPU_BASE, with_proto_tools
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import (
    RUNTIME_ENV,
    env_for,
    run_tool_call,
    stage_all_excluded_fixtures,
)


def _warmup() -> None:
    from proto_tools.tools.structure_scoring.ipsae.ipsae_scoring import example_input, run_ipsae_scoring

    run_ipsae_scoring(example_input())


image = with_proto_tools(CPU_BASE).env(env_for())
image = (
    stage_all_excluded_fixtures(image)
    .run_function(_warmup, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False)
    .env(RUNTIME_ENV)
)


app = get_app_for_service("IPSAEService")


@app.cls(
    include_source=False,
    image=image,
    cpu=2,
    timeout=SERVICE_MODAL_TIMEOUTS["IPSAEService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"ipsae-scoring": "score"})
class IPSAEService:
    """Modal service for IPSAE scoring."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent IPSAE worker."""
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("ipsae")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def score(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run IPSAE scoring."""
        from proto_tools.tools.structure_scoring.ipsae.ipsae_scoring import (
            IPSAEScoringConfig,
            IPSAEScoringInput,
            run_ipsae_scoring,
        )

        return run_tool_call(
            run_ipsae_scoring, IPSAEScoringInput, IPSAEScoringConfig, input_dict, config_dict, instance=self.instance
        )
