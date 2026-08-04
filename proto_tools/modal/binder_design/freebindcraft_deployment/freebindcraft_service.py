"""FreeBindCraft Modal service (PyRosetta-free binder design — MIT, hostable for commercial use)."""

from pathlib import Path
from typing import Any

import modal

from proto_tools.modal.app import (
    HF_TOKEN_SECRET,
    MODEL_CACHE,
    SCALEDOWN_WINDOW,
    get_app_for_service,
    retries_for_service,
)
from proto_tools.modal.base_images import with_dependencies, with_proto_tools
from proto_tools.modal.gpu_profiles import GPU_DEFAULT
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import (
    RUNTIME_ENV,
    ensure_gpu_ready,
    env_for,
    run_tool_call,
)

# Mirrors FreeBindCraft's own Dockerfile; the shared GPU_BASE segfaults in the first compiled AF2 pass.
FREEBINDCRAFT_BASE = with_dependencies(
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04", add_python="3.12").apt_install(
        "curl", "git", "libgfortran5", "build-essential", "pkg-config", "wget"
    )
)


def _warmup() -> None:
    """Materialize the FreeBindCraft standalone env at image build time."""
    from proto_tools.utils.tool_instance import ToolInstance

    ToolInstance("freebindcraft")._ensure_env()


# FreeBindCraft's native standalone is already PyRosetta-free, so the override no-ops.
image = with_proto_tools(FREEBINDCRAFT_BASE, overrides="freebindcraft", overrides_dir=Path(__file__).parent)
image = (
    image.env(
        {
            **env_for(),
            # uv falls back to PyPI when the cu128 index lacks torch==2.6.*.
            "UV_INDEX_STRATEGY": "unsafe-best-match",
        }
    )
    .run_function(
        _warmup,
        gpu=GPU_DEFAULT,
        volumes={"/weights": MODEL_CACHE},
        timeout=3600,
        secrets=[HF_TOKEN_SECRET],
        include_source=False,
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("FreeBindCraftService")


@app.cls(
    include_source=False,
    image=image,
    gpu=GPU_DEFAULT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/weights": MODEL_CACHE},
    timeout=SERVICE_MODAL_TIMEOUTS["FreeBindCraftService"],
    retries=retries_for_service("FreeBindCraftService"),
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"freebindcraft-design": "design"})
class FreeBindCraftService:
    """Modal service for FreeBindCraft PyRosetta-free binder design."""

    @modal.enter()
    def setup(self) -> None:
        """Open a persistent ``freebindcraft`` worker."""
        ensure_gpu_ready("freebindcraft")
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("freebindcraft")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def design(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Run one FreeBindCraft binder-design campaign.

        Args:
            input_dict (dict[str, Any]): Serialized ``FreeBindCraftInput`` fields.
            config_dict (dict[str, Any]): Serialized ``FreeBindCraftConfig`` fields.

        Returns:
            dict[str, Any]: Serialized ``FreeBindCraftOutput``.
        """
        from proto_tools.tools.binder_design.freebindcraft import (
            FreeBindCraftConfig,
            FreeBindCraftInput,
            run_freebindcraft_design,
        )

        return run_tool_call(
            run_freebindcraft_design,
            FreeBindCraftInput,
            FreeBindCraftConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )
