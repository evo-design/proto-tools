"""proto_tools/tools/testing/mock_jax_multi_gpu_tool/mock_jax_multi_gpu_tool.py

This is a minimal 2-GPU JAX-pattern tool designed for testing multi-device
allocation, movement, and eviction with JAX semantics: model reload on
device change, no in-place .to()."""
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


class MockJAXMultiGPUToolInput(BaseToolInput):
    """Input for mock JAX multi-GPU tool.

    Attributes:
        data (list[float]): Input data to pass through both models.
    """

    data: list[float] = InputField(
        default=[1.0, 2.0, 3.0, 4.0],
        description="Input data to pass through both models (any length)",
    )


class MockJAXMultiGPUToolConfig(BaseConfig):
    """Config for mock JAX multi-GPU tool.

    Attributes:
        device (str): Device specification for 2 GPUs.
        hidden_size (int): Hidden layer size per model.
        memory_mb (int): GPU memory to allocate per model via buffer.
    """

    device: str = ConfigField(
        default="cudax2",
        title="Device",
        description="Device spec for 2 GPUs (cudax2, cuda:0,1)",
        hidden=True,
        include_in_key=False,
    )

    hidden_size: int = ConfigField(
        default=128,
        title="Hidden Size",
        description="Hidden layer size per model (affects memory)",
        reload_on_change=True,
    )

    memory_mb: int = ConfigField(
        default=512,
        title="Memory (MB)",
        description="GPU memory to allocate per model via buffer array (MB)",
        reload_on_change=True,
    )


class MockJAXMultiGPUToolOutput(BaseToolOutput):
    """Output from mock JAX multi-GPU tool.

    Attributes:
        result_model_a (list[float]): Output from model A (first GPU).
        result_model_b (list[float]): Output from model B (second GPU).
        devices_used (list[str]): Devices each model ran on.
    """

    result_model_a: list[float] = Field(
        description="Output from model A (first GPU)"
    )

    result_model_b: list[float] = Field(
        description="Output from model B (second GPU)"
    )

    devices_used: list[str] = Field(
        description="Devices each model ran on [device_a, device_b]"
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
                json.dump(
                    {
                        "result_model_a": self.result_model_a,
                        "result_model_b": self.result_model_b,
                        "devices_used": self.devices_used,
                    },
                    f,
                    indent=2,
                )
        elif file_format == "txt":
            with open(path, "w") as f:
                f.write(f"Devices: {self.devices_used}\n")
                f.write(f"Result A: {self.result_model_a}\n")
                f.write(f"Result B: {self.result_model_b}\n")
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input():
    """Minimal valid input for testing and examples."""
    return MockJAXMultiGPUToolInput()


@tool(
    key="mock-jax-multi-gpu-tool-run",
    label="Mock JAX Multi-GPU Tool",
    category="testing",
    input_class=MockJAXMultiGPUToolInput,
    config_class=MockJAXMultiGPUToolConfig,
    output_class=MockJAXMultiGPUToolOutput,
    description="Minimal 2-GPU JAX-pattern tool for testing multi-device management",
    uses_gpu=True,
    device_count="2",
    example_input=example_input,
)
def run_mock_jax_multi_gpu_tool(
    inputs: MockJAXMultiGPUToolInput,
    config: MockJAXMultiGPUToolConfig | None = None,
    instance=None,
) -> MockJAXMultiGPUToolOutput:
    """Run mock JAX multi-GPU tool (two minimal models with JAX semantics)."""

    result = ToolInstance.dispatch(
        "mock_jax_multi_gpu_tool",
        {
            "data": inputs.data,
            "device": config.device,
            "hidden_size": config.hidden_size,
            "memory_mb": config.memory_mb,
        },
        instance=instance,
        config=config,
    )

    return MockJAXMultiGPUToolOutput(**result)
