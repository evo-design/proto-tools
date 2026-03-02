"""Mock CLI multi-GPU subprocess tool for testing multi-device subprocess routing."""
from __future__ import annotations

from .mock_cli_multi_gpu_tool import (
    MockCLIMultiGPUToolConfig,
    MockCLIMultiGPUToolInput,
    MockCLIMultiGPUToolOutput,
    run_mock_cli_multi_gpu_tool,
)

__all__ = [
    "MockCLIMultiGPUToolConfig",
    "MockCLIMultiGPUToolInput",
    "MockCLIMultiGPUToolOutput",
    "run_mock_cli_multi_gpu_tool",
]
