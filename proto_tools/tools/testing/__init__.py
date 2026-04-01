"""Testing tools for infrastructure testing."""

from __future__ import annotations

from proto_tools.tools.testing.mock_cli_multi_gpu_tool import (
    MockCLIMultiGPUToolConfig,
    MockCLIMultiGPUToolInput,
    MockCLIMultiGPUToolOutput,
    run_mock_cli_multi_gpu_tool,
)
from proto_tools.tools.testing.mock_cli_tool import (
    MockCLIToolConfig,
    MockCLIToolInput,
    MockCLIToolOutput,
    run_mock_cli_tool,
)
from proto_tools.tools.testing.mock_jax_multi_gpu_tool import (
    MockJAXMultiGPUToolConfig,
    MockJAXMultiGPUToolInput,
    MockJAXMultiGPUToolOutput,
    run_mock_jax_multi_gpu_tool,
)
from proto_tools.tools.testing.mock_jax_tool import (
    MockJAXToolConfig,
    MockJAXToolInput,
    MockJAXToolOutput,
    run_mock_jax_tool,
)
from proto_tools.tools.testing.mock_pytorch_multi_gpu_tool import (
    MockPyTorchMultiGPUToolConfig,
    MockPyTorchMultiGPUToolInput,
    MockPyTorchMultiGPUToolOutput,
    run_mock_pytorch_multi_gpu_tool,
)
from proto_tools.tools.testing.mock_pytorch_tool import (
    MockPyTorchToolConfig,
    MockPyTorchToolInput,
    MockPyTorchToolOutput,
    MockPyTorchToolResult,
    run_mock_pytorch_tool,
)

__all__ = [
    "MockCLIMultiGPUToolConfig",
    "MockCLIMultiGPUToolInput",
    "MockCLIMultiGPUToolOutput",
    "run_mock_cli_multi_gpu_tool",
    "MockCLIToolConfig",
    "MockCLIToolInput",
    "MockCLIToolOutput",
    "run_mock_cli_tool",
    "MockJAXMultiGPUToolConfig",
    "MockJAXMultiGPUToolInput",
    "MockJAXMultiGPUToolOutput",
    "run_mock_jax_multi_gpu_tool",
    "MockJAXToolConfig",
    "MockJAXToolInput",
    "MockJAXToolOutput",
    "run_mock_jax_tool",
    "MockPyTorchMultiGPUToolConfig",
    "MockPyTorchMultiGPUToolInput",
    "MockPyTorchMultiGPUToolOutput",
    "run_mock_pytorch_multi_gpu_tool",
    "MockPyTorchToolConfig",
    "MockPyTorchToolInput",
    "MockPyTorchToolOutput",
    "MockPyTorchToolResult",
    "run_mock_pytorch_tool",
]
