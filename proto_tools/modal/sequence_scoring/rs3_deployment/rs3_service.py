"""Rule Set 3 Modal service."""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import CPU_BASE, with_proto_tools
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, env_for, run_tool_call


def _warmup() -> None:
    from proto_tools.tools.sequence_scoring.rs3.rs3_score import example_input, run_rs3_score

    run_rs3_score(example_input())


image = (
    with_proto_tools(CPU_BASE)
    .env(env_for())
    .run_function(_warmup, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False)
    .env(RUNTIME_ENV)
)
app = get_app_for_service("RS3Service")


@app.cls(
    include_source=False,
    image=image,
    cpu=2,
    timeout=SERVICE_MODAL_TIMEOUTS["RS3Service"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"rs3-score": "score"})
class RS3Service:
    """Hosted CPU service for Rule Set 3 inference."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent isolated RS3 worker."""
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("rs3")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker during container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def score(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Validate inputs and return Rule Set 3 scores."""
        from proto_tools.tools.sequence_scoring.rs3 import RS3ScoreConfig, RS3ScoreInput, run_rs3_score

        return run_tool_call(
            run_rs3_score, RS3ScoreInput, RS3ScoreConfig, input_dict, config_dict, instance=self.instance
        )
