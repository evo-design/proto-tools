"""AlphaFold2 Modal service.

Wraps two proto-tools entry points from the ``alphafold2`` toolkit:

* ``run_alphafold2`` (structure prediction) from
  ``proto_tools.tools.structure_prediction.alphafold2.alphafold2``
* ``run_alphafold2_gradient`` (binder design via ColabDesign) from
  ``proto_tools.tools.structure_prediction.alphafold2.alphafold2_gradient``

The build-time warmup runs a single AF2 prediction against the tool's example
input so that JAX is initialized and weights are downloaded onto the shared
``bio-model-cache`` volume via ``PROTO_MODEL_CACHE``.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SCALEDOWN_WINDOW, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import GPU_BASE, with_proto_tools
from proto_tools.modal.gpu_profiles import GPU_DEFAULT
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, ensure_gpu_ready, env_for, run_tool_call


def _warmup() -> None:
    """Deploy-time: warm the weights without searching for an MSA.

    ``use_msa`` defaults to True, so leaving the config off made the build depend on the ColabFold
    server — slow, external, and unrelated to whether the image is sound. Every other structure
    predictor already states this; alphafold2 was relying on something short-circuiting the search
    for a four-residue input, which is not a property to build on.
    """
    from proto_tools.tools.structure_prediction.alphafold2 import AlphaFold2Config
    from proto_tools.tools.structure_prediction.alphafold2.alphafold2 import (
        example_input,
        run_alphafold2,
    )

    run_alphafold2(example_input(), AlphaFold2Config(use_msa=False))


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("AlphaFold2Service")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["AlphaFold2Service"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"alphafold2-prediction": "predict", "alphafold2-gradient": "gradient"})
class AlphaFold2Service:
    """Modal service for AlphaFold2 structure prediction and binder design."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so AF2 weights stay loaded across requests."""
        ensure_gpu_ready("alphafold2")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("alphafold2")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def predict(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run AlphaFold2 structure prediction.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Structure prediction results with metrics.
        """
        from proto_tools.tools.structure_prediction.alphafold2.alphafold2 import (
            AlphaFold2Config,
            AlphaFold2Input,
            run_alphafold2,
        )

        return run_tool_call(
            run_alphafold2, AlphaFold2Input, AlphaFold2Config, input_dict, config_dict, instance=self.instance
        )

    @modal.method()
    def gradient(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Compute AlphaFold2 binder-design loss and gradient via ColabDesign.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Binder-design loss, gradient, predicted structure, and metrics.
        """
        from proto_tools.tools.structure_prediction.alphafold2.alphafold2_gradient import (
            AlphaFold2GradientConfig,
            AlphaFold2GradientInput,
            run_alphafold2_gradient,
        )

        return run_tool_call(
            run_alphafold2_gradient,
            AlphaFold2GradientInput,
            AlphaFold2GradientConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )
