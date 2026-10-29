"""RFD4-Proteina Modal service.

Delegates to proto-tools ``run_rfd4_design`` and ``run_rfd4_inverse_folding`` for
parameter validation, environment setup, and model inference. The build-time warmup
creates the tool env and downloads the ``nvidia/RFD4-Proteina`` denoiser and
autoencoder (~2.8 GB, ungated), persisted on the Modal volume via ``PROTO_MODEL_CACHE``.

Both tools share one service because they share a model bundle, a tool env, and a
persistent worker; inverse folding only selects a different upstream generation preset,
so splitting them would download the same weights twice.
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
    """Deploy-time: build the env and warm the default checkpoint.

    Deliberately the cheap sampling path rather than the production default: this shares
    ``run_function``'s one-hour timeout with a micromamba CUDA toolkit install, the
    ``rfd4-proteina[all-cuda12]`` wheels, and a ~2.8 GB weight download, and the point here
    is to populate the env and the HuggingFace cache, not to produce a usable design. Both
    registered tools resolve the same toolkit and checkpoint bundle, so one warmup covers them.
    """
    from proto_tools.tools.structure_design.rfd4.rfd4_design import (
        RFD4DesignConfig,
        example_input,
        run_rfd4_design,
    )

    run_rfd4_design(example_input(), RFD4DesignConfig(recycle_strategy=None, nsteps=20))


image = (
    with_proto_tools(GPU_BASE)
    .env(env_for())
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("RFD4Service")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["RFD4Service"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"rfd4-design": "design", "rfd4-inverse-folding": "inverse_folding"})
class RFD4Service:
    """Modal service for RFD4-Proteina design and inverse folding."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("rfd4")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("rfd4")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def design(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Co-generate sequence and all-atom structure for one or more design specs.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Designed structures with per-design metadata.
        """
        from proto_tools.tools.structure_design.rfd4.rfd4_design import (
            RFD4DesignConfig,
            RFD4Input,
            run_rfd4_design,
        )

        return run_tool_call(
            run_rfd4_design, RFD4Input, RFD4DesignConfig, input_dict, config_dict, instance=self.instance
        )

    @modal.method()
    def inverse_folding(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Design sequence and sidechains on a fixed input backbone.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Repacked all-atom structures carrying the designed sequences.
        """
        from proto_tools.tools.structure_design.rfd4.rfd4_design import RFD4Input
        from proto_tools.tools.structure_design.rfd4.rfd4_inverse_folding import (
            RFD4InverseFoldingConfig,
            run_rfd4_inverse_folding,
        )

        return run_tool_call(
            run_rfd4_inverse_folding,
            RFD4Input,
            RFD4InverseFoldingConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )
