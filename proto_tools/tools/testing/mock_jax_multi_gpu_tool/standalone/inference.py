"""Standalone inference script for mock JAX multi-GPU tool.

Two param pytrees on separate GPUs, following Flax/Haiku conventions.
Uses real JAX with jax.device_put() for device placement, same pattern
as ProteinMPNN and other real JAX tools.
"""

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
    set_jax_seed,
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
    seed: int = 0,
) -> dict[str, Any]:
    """Initialize model params as a dict pytree (Flax/Haiku convention)."""
    key = jax.random.PRNGKey(seed)
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
# Multi-GPU Model Wrapper
# ============================================================================


def _parse_device_pair(device_str: str) -> tuple[str, str]:
    """Parse a multi-device string into two individual device strings."""
    if device_str == "cpu":
        return ("cpu", "cpu")

    if "," in device_str:
        parts = device_str.split(",")
        if len(parts) != 2:
            raise ValueError(f"Expected exactly 2 devices for multi-GPU tool, got {len(parts)}: {device_str}")
        device_a = parts[0].strip()
        device_b = parts[1].strip()
        if not device_b.startswith("cuda:"):
            device_b = f"cuda:{device_b}"
        return (device_a, device_b)

    return (device_str, device_str)


class MockJAXMultiGPUToolModel:
    """Wrapper managing two param pytrees on separate devices.

    Each "model" is a params dict placed on a different GPU, following
    the Flax/Haiku convention of separated params and apply functions.
    """

    def __init__(self, hidden_size: int = 128, memory_mb: int = 512, device: str = "cuda:0,cuda:1", seed: int = 0):
        """Initialize MockJAXMultiGPUToolModel."""
        self.hidden_size = hidden_size
        self.memory_mb = memory_mb
        self.device_a, self.device_b = _parse_device_pair(device)
        self._jax_device_a = resolve_jax_device(self.device_a)
        self._jax_device_b = resolve_jax_device(self.device_b)
        self.params_a = _init_params(
            input_size=4,
            hidden_size=hidden_size,
            output_size=4,
            memory_mb=memory_mb,
            device=self._jax_device_a,
            seed=seed,
        )
        self.params_b = _init_params(
            input_size=4,
            hidden_size=hidden_size,
            output_size=4,
            memory_mb=memory_mb,
            device=self._jax_device_b,
            seed=seed,
        )
        self._loaded = True
        logger.info(
            "Loaded param pytrees: params_a on %s, params_b on %s (hidden_size=%d, memory_mb=%d)",
            self.device_a,
            self.device_b,
            hidden_size,
            memory_mb,
        )

    def to_device(self, device: str) -> None:
        """Move both param pytrees to new devices via move_model_to_device."""
        new_device_a, new_device_b = _parse_device_pair(device)

        if self.device_a != new_device_a:
            self.params_a = move_model_to_device(self.params_a, self.device_a, new_device_a)
            self._jax_device_a = resolve_jax_device(new_device_a)
            self.device_a = new_device_a

        if self.device_b != new_device_b:
            self.params_b = move_model_to_device(self.params_b, self.device_b, new_device_b)
            self._jax_device_b = resolve_jax_device(new_device_b)
            self.device_b = new_device_b

        logger.info("Moved params: params_a on %s, params_b on %s", self.device_a, self.device_b)

    def run(self, data: list[float]) -> dict[str, Any]:
        """Run inference on both param sets."""
        x_a = jax.device_put(jnp.array(data, dtype=jnp.float32), self._jax_device_a)
        x_b = jax.device_put(jnp.array(data, dtype=jnp.float32), self._jax_device_b)
        result_a = _apply(self.params_a, x_a).tolist()
        result_b = _apply(self.params_b, x_b).tolist()
        return {
            "result_model_a": result_a,
            "result_model_b": result_b,
            "devices_used": [self.device_a, self.device_b],
        }


# ============================================================================
# Global Model Instance
# ============================================================================

_model: MockJAXMultiGPUToolModel | None = None


# ============================================================================
# Dispatch
# ============================================================================


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model

    data = input_dict["data"]
    device = input_dict["device"]
    hidden_size = input_dict["hidden_size"]
    memory_mb = input_dict["memory_mb"]
    seed = input_dict["seed"]

    set_jax_seed(seed)

    if _model is None:
        _model = MockJAXMultiGPUToolModel(
            hidden_size=hidden_size,
            memory_mb=memory_mb,
            device=device,
            seed=seed,
        )
    else:
        _model.params_a = _init_params(
            input_size=4,
            hidden_size=_model.hidden_size,
            output_size=4,
            memory_mb=_model.memory_mb,
            device=_model._jax_device_a,
            seed=seed,
        )
        _model.params_b = _init_params(
            input_size=4,
            hidden_size=_model.hidden_size,
            output_size=4,
            memory_mb=_model.memory_mb,
            device=_model._jax_device_b,
            seed=seed,
        )

    return _model.run(data)


def to_device(device: str) -> dict[str, Any]:
    """Move both param sets to specified devices (called by DeviceManager)."""
    global _model
    if _model is not None and _model._loaded:
        _model.to_device(device)
        return {"success": True, "device": device}
    return {"success": True, "device": device, "note": "models not loaded yet"}


def get_memory_stats() -> dict[str, Any]:
    """Get memory statistics from both devices."""
    global _model
    if _model is None:
        return {"available": False, "framework": "jax", "reason": "Models not loaded"}
    return get_jax_memory_stats(device_index=0)  # type: ignore[no-any-return]


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
