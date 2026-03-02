"""Mock PyTorch tool for testing DeviceManager functionality.

This is a minimal PyTorch tool designed for fast testing of device management,
memory tracking, and worker lifecycle. It loads a tiny model in <1 second while
still exercising all DeviceManager code paths.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField
from bio_programming_tools.utils.tool_io import BaseToolInput, BaseToolOutput


# ============================================================================
# Input / Config / Output
# ============================================================================


class MockPyTorchToolInput(BaseToolInput):
    """Input for mock PyTorch tool.

    Fields:
        data: Input data to pass through the model.
    """

    data: list[float] = Field(
        default=[1.0, 2.0, 3.0, 4.0],
        description="Input data to pass through the model (any length)",
    )


class MockPyTorchToolConfig(BaseConfig):
    """Config for mock PyTorch tool.

    Fields:
        device: Device to run on.
        hidden_size: Hidden layer size.
        memory_mb: GPU memory to allocate via buffer.
    """

    device: str = ConfigField(
        default="cuda",
        title="Device",
        description="Device to run on",
        hidden=True,
    )

    hidden_size: int = ConfigField(
        default=128,
        title="Hidden Size",
        description="Hidden layer size (affects memory)",
        reload_on_change=True,
    )

    memory_mb: int = ConfigField(
        default=512,
        title="Memory (MB)",
        description="GPU memory to allocate via buffer tensor (MB)",
        reload_on_change=True,
    )


class MockPyTorchToolOutput(BaseToolOutput):
    """Output from mock PyTorch tool.

    Fields:
        result: Output from the model.
        device_used: Device the model ran on.
    """

    result: list[float] = Field(
        description="Output from the model"
    )

    device_used: str = Field(
        description="Device the model ran on"
    )

    @property
    def output_format_options(self) -> list[str]:
        return ["json", "txt"]

    @property
    def output_format_default(self) -> str:
        return "json"

    def _export_output(self, export_path: str | Path, file_format: str):
        from pathlib import Path
        import json

        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "json":
            with open(path, "w") as f:
                json.dump({"result": self.result, "device_used": self.device_used}, f, indent=2)
        elif file_format == "txt":
            with open(path, "w") as f:
                f.write(f"Device: {self.device_used}\n")
                f.write(f"Result: {self.result}\n")
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input():
    """Minimal valid input for testing and examples."""
    return MockPyTorchToolInput()


@tool(
    key="mock-pytorch-tool-run",
    label="Mock PyTorch Tool",
    category="testing",
    input_class=MockPyTorchToolInput,
    config_class=MockPyTorchToolConfig,
    output_class=MockPyTorchToolOutput,
    description="Minimal PyTorch tool for testing device management",
    uses_gpu=True,
    device_count="1",
    example_input=example_input,
)
def run_mock_pytorch_tool(
    inputs: MockPyTorchToolInput,
    config: MockPyTorchToolConfig | None = None,
    instance=None,
) -> MockPyTorchToolOutput:
    """Run mock PyTorch tool (minimal model for fast testing)."""
    from bio_programming_tools.utils.tool_instance import ToolInstance

    result = ToolInstance.dispatch(
        "mock_pytorch_tool",
        {
            "data": inputs.data,
            "device": config.device,
            "hidden_size": config.hidden_size,
            "memory_mb": config.memory_mb,
        },
        instance=instance,
        reload_on=type(config).reload_fields(),
    )

    return MockPyTorchToolOutput(**result)
