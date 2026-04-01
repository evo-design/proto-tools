"""Standalone inference script for mock JAX tool.

This script runs in an isolated environment and implements a minimal JAX
model for testing DeviceManager functionality. Follows real JAX/Flax
conventions: params are a dict pytree separated from the model function,
and device movement uses jax.device_put() on the params pytree (matching
ProteinMPNN, Flax, Haiku, etc.).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

# JAX memory settings - prevent preallocation so nvidia-smi reflects actual usage
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
from standalone_helpers import (
    get_jax_memory_stats,
    move_model_to_device,
    resolve_jax_device,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Minimal JAX Model (Flax-style: params pytree + apply function)
# ============================================================================


def _init_params(
    input_size: int = 4,
    hidden_size: int = 128,
    output_size: int = 4,
    memory_mb: int = 512,
    device: jax.Device | None = None,
) -> dict[str, Any]:
    """Initialize model params as a dict pytree (Flax/Haiku convention).

    Returns a nested dict of jax arrays, placed on the target device.
    Includes a memory buffer to simulate realistic GPU memory usage.
    """
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    params = {
        "layer1": {
            "weights": jax.random.normal(k1, (input_size, hidden_size)) * 0.01,
            "bias": jnp.zeros(hidden_size),
        },
        "layer2": {
            "weights": jax.random.normal(k2, (hidden_size, output_size)) * 0.01,
            "bias": jnp.zeros(output_size),
        },
        # Buffer to consume target GPU memory (~512MB by default)
        "memory_buffer": jnp.zeros((memory_mb * 1024 * 1024) // 4),
    }

    if device is not None:
        params = jax.device_put(params, device)

    return params


def _apply(params: dict[str, Any], x: jax.Array) -> jax.Array:
    """Forward pass using params pytree (pure function, no model state)."""
    h = jnp.maximum(0, x @ params["layer1"]["weights"] + params["layer1"]["bias"])
    return h @ params["layer2"]["weights"] + params["layer2"]["bias"]


# ============================================================================
# Model Wrapper
# ============================================================================


class MockJAXToolModel:
    """Wrapper managing params pytree and device placement.

    Follows the ProteinMPNN pattern: params are a dict pytree moved via
    move_model_to_device(), which uses jax.device_put() natively on dicts.
    """

    def __init__(self, hidden_size: int = 128, memory_mb: int = 512, device: str = "cuda"):
        """Initialize MockJAXToolModel."""
        self.hidden_size = hidden_size
        self.memory_mb = memory_mb
        self.device_str = device
        self._jax_device = resolve_jax_device(device)
        self.params = _init_params(
            input_size=4,
            hidden_size=hidden_size,
            output_size=4,
            memory_mb=memory_mb,
            device=self._jax_device,
        )
        self._loaded = True
        logger.info(
            "Loaded mock JAX model on %s (hidden_size=%d, memory_mb=%d)",
            device,
            hidden_size,
            memory_mb,
        )

    def to_device(self, device: str) -> None:
        """Move params to new device via move_model_to_device (dict pytree)."""
        if self.device_str == device:
            return

        old_device = self.device_str
        self.params = move_model_to_device(self.params, old_device, device)
        self._jax_device = resolve_jax_device(device)
        self.device_str = device
        logger.info("Moved params to %s via device_put (from %s)", device, old_device)

    def run(self, data: list[float]) -> dict[str, Any]:
        """Run inference on input data."""
        x = jax.device_put(jnp.array(data, dtype=jnp.float32), self._jax_device)
        output = _apply(self.params, x)
        return {
            "result": output.tolist(),
            "device_used": self.device_str,
        }


# ============================================================================
# Global Model Instance
# ============================================================================

_model: MockJAXToolModel | None = None


# ============================================================================
# Dispatch
# ============================================================================


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model

    data = input_dict.get("data", [1.0, 2.0, 3.0, 4.0])
    device = input_dict.get("device", "cuda")
    hidden_size = input_dict.get("hidden_size", 128)
    memory_mb = input_dict.get("memory_mb", 512)

    if _model is None:
        _model = MockJAXToolModel(
            hidden_size=hidden_size,
            memory_mb=memory_mb,
            device=device,
        )

    return _model.run(data)


def to_device(device: str) -> dict[str, Any]:
    """Move model to specified device (called by DeviceManager)."""
    global _model
    if _model is not None and _model._loaded:
        _model.to_device(device)
        return {"success": True, "device": device}
    return {"success": True, "device": device, "note": "model not loaded yet"}


def get_memory_stats() -> dict[str, Any]:
    """Get memory statistics (called by DeviceManager)."""
    global _model
    if _model is None:
        return {"available": False, "framework": "jax", "reason": "Model not loaded"}

    # Resolve device index for stats
    device_idx = 0
    if _model._jax_device is not None and _model._jax_device.platform == "gpu":
        device_idx = _model._jax_device.id

    return get_jax_memory_stats(device_idx)  # type: ignore[no-any-return]


# ============================================================================
# Main Entry Point
# ============================================================================


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
