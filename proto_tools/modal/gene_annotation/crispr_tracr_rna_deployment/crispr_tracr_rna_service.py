"""CRISPRtracrRNA Modal service.

Delegates to proto-tools ``run_crispr_tracr_rna`` for parameter validation,
environment setup (micromamba conda env with bioconda packages), and
tracrRNA prediction.  The build-time warmup creates the heavy conda_deps
environment and runs a test prediction, both persisted in the image.

Note: CRISPRtracrRNA only supports Linux x86_64 due to vmatch and
several bioconda packages lacking aarch64 builds.
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
    from proto_tools.tools.gene_annotation.crispr_tracr_rna.crispr_tracr_rna import (
        example_input,
        run_crispr_tracr_rna,
    )

    run_crispr_tracr_rna(example_input())


image = (
    with_proto_tools(CPU_BASE)
    .env(env_for())
    .run_function(_warmup, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False)
    .env(RUNTIME_ENV)
)


app = get_app_for_service("CrisprTracrRNAService")


@app.cls(
    include_source=False,
    image=image,
    cpu=4,
    timeout=SERVICE_MODAL_TIMEOUTS["CrisprTracrRNAService"],
    retries=SERVICE_RETRIES,
    volumes={"/weights": MODEL_CACHE},
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"crispr-tracr-rna": "crispr_tracr_rna"})
class CrisprTracrRNAService:
    """Modal service for CRISPRtracrRNA tracrRNA prediction."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the conda env stays warm across requests."""
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("crispr_tracr_rna")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def crispr_tracr_rna(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Predict tracrRNA sequences from nucleotide CRISPR loci.

        Args:
            input_dict (dict[str, Any]): Mapping of input field names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Predicted tracrRNA sequences and associated metadata.
        """
        from proto_tools.tools.gene_annotation.crispr_tracr_rna.crispr_tracr_rna import (
            CrisprTracrRNAConfig,
            CrisprTracrRNAInput,
            run_crispr_tracr_rna,
        )

        inputs = CrisprTracrRNAInput(**input_dict)
        config = CrisprTracrRNAConfig(**config_dict)
        return dispatch_tool_call(run_crispr_tracr_rna, inputs, config, instance=self.instance)
