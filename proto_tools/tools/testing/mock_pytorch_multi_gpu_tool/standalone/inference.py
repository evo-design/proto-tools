"""Standalone inference script for mock multi-GPU PyTorch tool.

This script runs in an isolated environment and implements two minimal PyTorch
models for testing multi-device management. Each model lives on a separate GPU.
It's designed to be fast (<1s) while exercising multi-device allocation,
movement, and eviction code paths.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import torch
import torch.nn as nn
from standalone_helpers import get_pytorch_memory_stats, move_model_to_device

logger = logging.getLogger(__name__)


# ============================================================================
# Minimal PyTorch Model
# ============================================================================


class TinyModel(nn.Module):  # type: ignore[misc]
    """Minimal PyTorch model that allocates a realistic amount of GPU memory."""

    def __init__(self, input_size: int = 4, hidden_size: int = 128, output_size: int = 4, memory_mb: int = 512):
        """Initialize TinyModel."""
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        # Allocate buffer to consume target GPU memory
        num_floats = (memory_mb * 1024 * 1024) // 4
        self.register_buffer("_memory_buffer", torch.zeros(num_floats))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


# ============================================================================
# Multi-GPU Model Wrapper
# ============================================================================


def _parse_device_pair(device_str: str) -> tuple[str, str]:
    """Parse a multi-device string into two individual device strings.

    Supports formats:
        "cudax2"          -> ("cuda:0", "cuda:1")  [auto-allocated by DeviceManager]
        "cuda:0,cuda:1"   -> ("cuda:0", "cuda:1")
        "cuda:0,1"        -> ("cuda:0", "cuda:1")
        "cpu"             -> ("cpu", "cpu")

    Note: In practice, DeviceManager resolves "cudax2" to specific indices
    before it reaches the worker subprocess. This function handles the
    resolved format.
    """
    if device_str == "cpu":
        return ("cpu", "cpu")

    # Handle comma-separated devices: "cuda:0,cuda:1" or "cuda:0,1"
    if "," in device_str:
        parts = device_str.split(",")
        if len(parts) != 2:
            raise ValueError(f"Expected exactly 2 devices for multi-GPU tool, got {len(parts)}: {device_str}")

        device_a = parts[0].strip()
        device_b = parts[1].strip()

        # Handle shorthand "cuda:0,1" -> "cuda:0", "cuda:1"
        if not device_b.startswith("cuda:"):
            device_b = f"cuda:{device_b}"

        return (device_a, device_b)

    # Single device fallback (e.g., during CPU offload)
    return (device_str, device_str)


class MockPyTorchMultiGPUToolModel:
    """Wrapper managing two tiny PyTorch models on separate devices."""

    def __init__(self, hidden_size: int = 128, memory_mb: int = 512, device: str = "cuda:0,cuda:1"):
        """Initialize MockPyTorchMultiGPUToolModel."""
        self.hidden_size = hidden_size
        self.memory_mb = memory_mb
        self.device_a, self.device_b = _parse_device_pair(device)

        # Resolve to actual torch devices
        self._device_a = torch.device(self.device_a if torch.cuda.is_available() else "cpu")
        self._device_b = torch.device(self.device_b if torch.cuda.is_available() else "cpu")

        # Create two separate models, one per device
        self.model_a = TinyModel(input_size=4, hidden_size=hidden_size, output_size=4, memory_mb=memory_mb)
        self.model_b = TinyModel(input_size=4, hidden_size=hidden_size, output_size=4, memory_mb=memory_mb)

        self.model_a.to(self._device_a)
        self.model_b.to(self._device_b)

        self._loaded = True
        logger.info(
            "Loaded TinyModel pair: model_a on %s, model_b on %s (hidden_size=%d)",
            self._device_a,
            self._device_b,
            hidden_size,
        )

    def to_device(self, device: str) -> None:
        """Move both models to new devices.

        Uses move_model_to_device helper for proper GPU memory cleanup.
        """
        new_device_a, new_device_b = _parse_device_pair(device)

        # Move model A
        self.model_a = move_model_to_device(self.model_a, self.device_a, new_device_a)
        self.device_a = new_device_a
        self._device_a = torch.device(new_device_a if torch.cuda.is_available() else "cpu")

        # Move model B
        self.model_b = move_model_to_device(self.model_b, self.device_b, new_device_b)
        self.device_b = new_device_b
        self._device_b = torch.device(new_device_b if torch.cuda.is_available() else "cpu")

        logger.info(
            "Moved models: model_a to %s, model_b to %s",
            self._device_a,
            self._device_b,
        )

    def run(self, data: list[float]) -> dict[str, Any]:
        """Run inference on both models with the same input data."""
        # Feed data through model A
        x_a = torch.tensor(data, dtype=torch.float32, device=self._device_a)
        with torch.no_grad():
            output_a = self.model_a(x_a)
        result_a = output_a.cpu().tolist()

        # Feed data through model B
        x_b = torch.tensor(data, dtype=torch.float32, device=self._device_b)
        with torch.no_grad():
            output_b = self.model_b(x_b)
        result_b = output_b.cpu().tolist()

        return {
            "result_model_a": result_a,
            "result_model_b": result_b,
            "devices_used": [str(self._device_a), str(self._device_b)],
        }


# ============================================================================
# Global Model Instance
# ============================================================================

_model: MockPyTorchMultiGPUToolModel | None = None


# ============================================================================
# Dispatch
# ============================================================================


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model

    data = input_dict.get("data", [1.0, 2.0, 3.0, 4.0])
    device = input_dict.get("device", "cuda:0,cuda:1")
    hidden_size = input_dict.get("hidden_size", 128)
    memory_mb = input_dict.get("memory_mb", 512)

    if _model is None:
        _model = MockPyTorchMultiGPUToolModel(
            hidden_size=hidden_size,
            memory_mb=memory_mb,
            device=device,
        )

    return _model.run(data)


def to_device(device: str) -> dict[str, Any]:
    """Move both models to specified devices (called by DeviceManager)."""
    global _model

    if _model is not None and _model._loaded:
        _model.to_device(device)
        return {"success": True, "device": device}
    return {"success": True, "device": device, "note": "models not loaded yet"}


def get_memory_stats() -> dict[str, Any]:
    """Get memory statistics from both devices."""
    global _model

    if _model is None:
        return {"available": False, "framework": "pytorch", "reason": "Models not loaded"}

    stats = {"available": True, "framework": "pytorch", "devices": {}}

    for label, device in [("model_a", _model._device_a), ("model_b", _model._device_b)]:
        if device.type == "cuda":
            device_idx = device.index if device.index is not None else 0
            stats["devices"][label] = get_pytorch_memory_stats(device_idx)  # type: ignore[index]
        else:
            stats["devices"][label] = {"device": str(device), "type": "cpu"}  # type: ignore[index]

    return stats


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
