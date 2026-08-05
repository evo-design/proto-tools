"""Primer3 Modal service.

Delegates to proto-tools ``run_primer3_thermodynamics`` for parameter validation,
environment setup, and oligo thermodynamics. primer3-py installs from PyPI at
build time, so the warmup pays for the env once and the image layer keeps it.
CPU only, with no model weights to stage.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import CPU_BASE, with_proto_tools
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, env_for, run_tool_call


def _warmup() -> None:
    """Deploy-time: build the env so the install is not paid for on first call."""
    from proto_tools.tools.sequence_scoring.primer3.primer3_thermodynamics import (
        example_input,
        run_primer3_thermodynamics,
    )

    run_primer3_thermodynamics(example_input())


image = (
    with_proto_tools(CPU_BASE)
    # PROTO_MODEL_CACHE unused for pure-CPU tools; set for env consistency with GPU services
    .env(env_for())
    .run_function(_warmup, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False)
    .env(RUNTIME_ENV)
)


app = get_app_for_service("Primer3Service")


@app.cls(
    include_source=False,
    image=image,
    cpu=2,
    timeout=SERVICE_MODAL_TIMEOUTS["Primer3Service"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"primer3-thermodynamics": "score"})
class Primer3Service:
    """Modal service for Primer3 oligonucleotide thermodynamics."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the tool env stays loaded across requests."""
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("primer3")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def score(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score DNA oligos for melting temperature, hairpin/dimer stability, GC content, and GC-clamp.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-oligo thermodynamic scores.
        """
        from proto_tools.tools.sequence_scoring.primer3 import (
            Primer3ThermodynamicsConfig,
            Primer3ThermodynamicsInput,
            run_primer3_thermodynamics,
        )

        return run_tool_call(
            run_primer3_thermodynamics,
            Primer3ThermodynamicsInput,
            Primer3ThermodynamicsConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )
