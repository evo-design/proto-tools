"""proto_tools/tools/testing/mock_jax_tool/mock_jax_tool.py

This is a minimal JAX-pattern tool designed for testing device management with
JAX semantics: no in-place .to(), model reload on device change, and
get_jax_memory_stats() reporting."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)

# ============================================================================
# Input / Config / Output
# ============================================================================


class MockJAXToolInput(BaseToolInput):
    """Input for mock JAX tool.

    Attributes:
        data (list[float]): Input data to pass through the model.
    """

    data: list[float] = InputField(
        default=[1.0, 2.0, 3.0, 4.0],
        description="Input data to pass through the model (any length)",
    )


class MockJAXToolConfig(BaseConfig):
    """Config for mock JAX tool.

    Attributes:
        device (str): Device to run on.
        hidden_size (int): Hidden layer size.
        memory_mb (int): GPU memory to allocate via buffer.
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
        description="GPU memory to allocate via buffer array (MB)",
        reload_on_change=True,
    )


class MockJAXToolOutput(BaseToolOutput):
    """Output from mock JAX tool.

    Attributes:
        result (list[float]): Output from the model.
        device_used (str): Device the model ran on.
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
        import json
        from pathlib import Path

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
    return MockJAXToolInput()


@tool(
    key="mock-jax-tool-run",
    label="Mock JAX Tool",
    category="testing",
    input_class=MockJAXToolInput,
    config_class=MockJAXToolConfig,
    output_class=MockJAXToolOutput,
    description="Minimal JAX-pattern tool for testing device management",
    uses_gpu=True,
    device_count="1",
    example_input=example_input,
)
def run_mock_jax_tool(
    inputs: MockJAXToolInput,
    config: MockJAXToolConfig | None = None,
    instance=None,
) -> MockJAXToolOutput:
    """Run mock JAX tool (minimal model with JAX device semantics)."""

    result = ToolInstance.dispatch(
        "mock_jax_tool",
        {
            "data": inputs.data,
            "device": config.device,
            "hidden_size": config.hidden_size,
            "memory_mb": config.memory_mb,
        },
        instance=instance,
        config=config,
    )

    return MockJAXToolOutput(**result)
