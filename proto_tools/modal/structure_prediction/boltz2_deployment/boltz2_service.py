"""Boltz2 Modal service.

Delegates to proto-tools ``run_boltz2`` for parameter validation,
environment setup, and model inference.  The build-time warmup creates the
micromamba env and downloads weights, both persisted on the Modal volume via
``PROTO_MODEL_CACHE``.
"""

from pathlib import Path
from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SCALEDOWN_WINDOW, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import GPU_BASE, with_proto_tools
from proto_tools.modal.gpu_profiles import GPU_DEFAULT
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import (
    RUNTIME_ENV,
    dispatch_tool_call,
    ensure_gpu_ready,
    env_for,
)

_BOLTZ2_CACHE_DIR = Path("/weights/boltz2")


def _warmup() -> None:
    """Deploy-time: warm the default config.

    Clears the boltz2 weights dir on failure — a partial HuggingFace
    download persisted on the volume would otherwise poison every
    subsequent build.
    """
    import shutil

    from proto_tools.tools.structure_prediction.boltz2 import Boltz2Config
    from proto_tools.tools.structure_prediction.boltz2.boltz2 import (
        example_input,
        run_boltz2,
    )

    try:
        run_boltz2(example_input(), Boltz2Config(use_msa=False, verbose=3))
    except Exception:
        if _BOLTZ2_CACHE_DIR.exists():
            # Image build has no logging handlers; stdout is what reaches the build log.
            print(f"Clearing {_BOLTZ2_CACHE_DIR} after warmup failure")  # noqa: T201
            shutil.rmtree(_BOLTZ2_CACHE_DIR)
        raise


image = with_proto_tools(GPU_BASE, overrides="boltz2", overrides_dir=Path(__file__).parent)
image = (
    image.env(
        {
            **env_for(),
            # HF_HOME routes HuggingFace-backed weights (boltz downloads its own
            # checkpoints from HF) onto the persistent Modal volume. Restored
            # from the pre-refactor service; without it, weights land in the
            # container's ephemeral filesystem.
            "HF_HOME": "/weights",
        }
    )
    .run_function(
        _warmup, gpu=GPU_DEFAULT, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("Boltz2Service")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["Boltz2Service"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"boltz2-prediction": "predict", "boltz2-affinity": "affinity"})
class Boltz2Service:
    """Modal service for Boltz2 structure and binding-affinity prediction."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the model stays loaded across requests."""
        ensure_gpu_ready("boltz2")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("boltz2")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def predict(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run Boltz2 structure prediction.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Structure prediction results with metrics.
        """
        from proto_tools.tools.structure_prediction.boltz2 import (
            Boltz2Config,
            Boltz2Input,
        )
        from proto_tools.tools.structure_prediction.boltz2.boltz2 import run_boltz2

        inputs = Boltz2Input(**input_dict)
        config = Boltz2Config(**config_dict)
        return dispatch_tool_call(run_boltz2, inputs, config, instance=self.instance)

    @modal.method()
    def affinity(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run Boltz2 binding-affinity prediction.

        Shares the persistent ``boltz2`` worker started in ``setup`` — ``run_boltz2_affinity``
        dispatches to the same toolkit.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Affinity prediction results with metrics.
        """
        from proto_tools.tools.structure_prediction.boltz2 import (
            Boltz2AffinityConfig,
            Boltz2AffinityInput,
        )
        from proto_tools.tools.structure_prediction.boltz2.boltz2_affinity import run_boltz2_affinity

        inputs = Boltz2AffinityInput(**input_dict)
        config = Boltz2AffinityConfig(**config_dict)
        return dispatch_tool_call(run_boltz2_affinity, inputs, config, instance=self.instance)
