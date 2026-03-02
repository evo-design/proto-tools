"""Mock multi-GPU PyTorch tool for testing multi-device management."""
from __future__ import annotations

from .mock_pytorch_multi_gpu_tool import (
    MockPyTorchMultiGPUToolConfig,
    MockPyTorchMultiGPUToolInput,
    MockPyTorchMultiGPUToolOutput,
    run_mock_pytorch_multi_gpu_tool,
)

__all__ = [
    "MockPyTorchMultiGPUToolConfig",
    "MockPyTorchMultiGPUToolInput",
    "MockPyTorchMultiGPUToolOutput",
    "run_mock_pytorch_multi_gpu_tool",
]
