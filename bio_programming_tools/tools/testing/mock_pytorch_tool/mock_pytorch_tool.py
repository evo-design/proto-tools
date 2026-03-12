"""Mock PyTorch tool for testing DeviceManager and ToolPool functionality.

This is a minimal PyTorch tool designed for fast testing of device management,
memory tracking, worker lifecycle, and parallel fan-out. It loads a tiny model
in <1 second while still exercising all DeviceManager and ToolPool code paths.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField
from bio_programming_tools.utils.tool_io import BaseToolInput, BaseToolOutput, InputField


# ============================================================================
# Input / Config / Output
# ============================================================================


class MockPyTorchToolInput(BaseToolInput):
    """Input for mock PyTorch tool.

    Fields:
        data_items: List of data vectors to process through the model.
    """

    data_items: list[list[float]] = InputField(
        default=[[1.0, 2.0, 3.0, 4.0]],
        description="List of data items to process (each is a 4-element float vector)",
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
        include_in_key=False,
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


class MockPyTorchToolResult(BaseModel):
    """Single result from mock PyTorch tool."""

    result: list[float] = Field(description="Output from the model")
    device_used: str = Field(description="Device the model ran on")


class MockPyTorchToolOutput(BaseToolOutput):
    """Output from mock PyTorch tool.

    Fields:
        results: List of results, one per input data item.
    """

    results: list[MockPyTorchToolResult] = Field(
        description="Results from the model, one per input data item"
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
                json.dump(
                    {"results": [r.model_dump() for r in self.results]},
                    f,
                    indent=2,
                )
        elif file_format == "txt":
            with open(path, "w") as f:
                for i, r in enumerate(self.results):
                    f.write(f"Item {i}: device={r.device_used}, result={r.result}\n")
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
    description="Minimal PyTorch tool for testing device management and ToolPool",
    uses_gpu=True,
    device_count="1",
    example_input=example_input,
    iterable_input_field="data_items",
    iterable_output_field="results",
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
            "data_items": [list(item) for item in inputs.data_items],
            "device": config.device,
            "hidden_size": config.hidden_size,
            "memory_mb": config.memory_mb,
        },
        instance=instance,
        config=config,
    )

    return MockPyTorchToolOutput(**result)
