"""PyHMMER Modal service.

Delegates to proto-tools' three sequence-query PyHMMER entry points for parameter validation,
environment setup, and search. ``pyhmmer-hmmscan`` and ``pyhmmer-hmmsearch`` are absent by design:
both read an HMM database or profile file from the caller's own disk, so they are declared
``local_only`` and cannot be served from a container. CPU only, with no weights to stage.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import CPU_BASE, with_proto_tools
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, dispatch_tool_call, env_for


def _warmup() -> None:
    """Deploy-time: build the env so the install is not paid for on first call."""
    from proto_tools.tools.gene_annotation.pyhmmer.phmmer import example_input, run_pyhmmer_phmmer

    run_pyhmmer_phmmer(example_input())


image = (
    with_proto_tools(CPU_BASE)
    .env(env_for())
    .run_function(_warmup, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False)
    .env(RUNTIME_ENV)
)


app = get_app_for_service("PyHmmerService")


@app.cls(
    include_source=False,
    image=image,
    cpu=4,
    timeout=SERVICE_MODAL_TIMEOUTS["PyHmmerService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools(
    {
        "pyhmmer-jackhmmer": "jackhmmer",
        "pyhmmer-nhmmer": "nhmmer",
        "pyhmmer-phmmer": "phmmer",
    }
)
class PyHmmerService:
    """Modal service for PyHMMER sequence-query searches."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the env stays warm across requests."""
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("pyhmmer")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def jackhmmer(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run an iterative jackhmmer search of query sequences against a target set.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-query hits with alignment statistics.
        """
        from proto_tools.tools.gene_annotation.pyhmmer.jackhmmer import (
            PyJackhmmerConfig,
            PyJackhmmerInput,
            run_pyhmmer_jackhmmer,
        )

        inputs = PyJackhmmerInput(**input_dict)
        config = PyJackhmmerConfig(**config_dict)
        return dispatch_tool_call(run_pyhmmer_jackhmmer, inputs, config, instance=self.instance)

    @modal.method()
    def nhmmer(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Search nucleotide query sequences against nucleotide targets.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-query hits with alignment statistics.
        """
        from proto_tools.tools.gene_annotation.pyhmmer.nhmmer import (
            PyNhmmerConfig,
            PyNhmmerInput,
            run_pyhmmer_nhmmer,
        )

        inputs = PyNhmmerInput(**input_dict)
        config = PyNhmmerConfig(**config_dict)
        return dispatch_tool_call(run_pyhmmer_nhmmer, inputs, config, instance=self.instance)

    @modal.method()
    def phmmer(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Search protein query sequences against a protein target set.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-query hits with alignment statistics.
        """
        from proto_tools.tools.gene_annotation.pyhmmer.phmmer import PyPhmmerInput, run_pyhmmer_phmmer
        from proto_tools.tools.gene_annotation.pyhmmer.shared_data_models import PyHmmerConfig

        inputs = PyPhmmerInput(**input_dict)
        config = PyHmmerConfig(**config_dict)
        return dispatch_tool_call(run_pyhmmer_phmmer, inputs, config, instance=self.instance)
