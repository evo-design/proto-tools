"""Mock pure-CPU tool for ToolPool fan-out integration testing."""

from proto_tools.tools.testing.mock_cpu_tool.mock_cpu_tool import (
    MockCPUToolConfig,
    MockCPUToolInput,
    MockCPUToolOutput,
    MockCPUToolResult,
    run_mock_cpu_tool,
)

__all__ = [
    "MockCPUToolConfig",
    "MockCPUToolInput",
    "MockCPUToolOutput",
    "MockCPUToolResult",
    "run_mock_cpu_tool",
]
