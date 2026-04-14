"""Standalone inference script for mock PyTorch tool.

This script runs in an isolated environment and implements a minimal PyTorch
model for testing DeviceManager and ToolPool functionality. It's designed to be
fast (<1s) while still exercising device management, memory tracking, and
worker protocols.
"""

import json
import logging
import sys
from typing import Any

import torch
import torch.nn as nn

# Import standalone helpers (auto-copied by worker bootstrap)
from standalone_helpers import get_pytorch_memory_stats, move_model_to_device, set_torch_seed

logger = logging.getLogger(__name__)


# ============================================================================
# Minimal PyTorch Model
# ============================================================================


class TinyModel(nn.Module):  # type: ignore[misc]
    """Minimal PyTorch model that allocates a realistic amount of GPU memory.

    Uses small linear layers for computation and a large buffer tensor to
    simulate realistic GPU memory usage (~512MB by default).
    """

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


class MockPyTorchToolModel:
    """Wrapper for the tiny PyTorch model."""

    def __init__(self, hidden_size: int = 128, memory_mb: int = 512, device: str = "cuda"):
        """Initialize MockPyTorchToolModel."""
        self.hidden_size = hidden_size
        self.memory_mb = memory_mb
        self.device_str = device
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = TinyModel(input_size=4, hidden_size=hidden_size, output_size=4, memory_mb=memory_mb)
        self.model.to(self._device)
        self._loaded = True
        logger.info(f"Loaded TinyModel on {self._device} (hidden_size={hidden_size}, memory_mb={memory_mb})")

    def to_device(self, device: str) -> None:
        """Move model to specified device with proper GPU memory cleanup."""
        new_device = device if torch.cuda.is_available() else "cpu"
        self.model = move_model_to_device(self.model, self.device_str, new_device)
        self._device = torch.device(new_device)
        self.device_str = new_device
        logger.info(f"Moved model to {self._device}")

    def run(self, data: list[float]) -> dict[str, Any]:
        """Run inference on a single data item."""
        x = torch.tensor(data, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            output = self.model(x)
        return {
            "result": output.cpu().tolist(),
            "device_used": str(self._device),
        }


# ============================================================================
# Global Model Instance
# ============================================================================

_model: MockPyTorchToolModel | None = None


# ============================================================================
# Dispatch
# ============================================================================


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model

    # Extract parameters
    data_items = input_dict["data_items"]
    device = input_dict["device"]
    hidden_size = input_dict["hidden_size"]
    memory_mb = input_dict["memory_mb"]
    seed = input_dict["seed"]
    set_torch_seed(seed)

    # Initialize model if needed
    if _model is None:
        _model = MockPyTorchToolModel(
            hidden_size=hidden_size,
            memory_mb=memory_mb,
            device=device,
        )

    # Run inference on each data item
    results = [_model.run(item) for item in data_items]

    return {"results": results}


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
        return {"available": False, "framework": "pytorch", "reason": "Model not loaded"}

    # Use standalone helper for PyTorch memory stats
    device_idx = 0
    if isinstance(_model._device, torch.device) and _model._device.type == "cuda" and _model._device.index is not None:
        device_idx = _model._device.index

    return get_pytorch_memory_stats(device_idx)  # type: ignore[no-any-return]


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
