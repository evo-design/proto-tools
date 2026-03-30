"""bio_programming_tools/tools/testing/mock_cli_multi_gpu_tool/mock_cli_multi_gpu_tool.py

This is a minimal 2-GPU CLI-pattern tool that spawns subprocesses for inference,
matching tools like Boltz2 with multi-GPU support. It uses
get_subprocess_device_env() for device routing with comma-separated device strings."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import (
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


class MockCLIMultiGPUToolInput(BaseToolInput):
    """Input for mock CLI multi-GPU tool.

    Attributes:
        data (list[float]): Input data to pass through the CLI subprocess.
    """

    data: list[float] = InputField(
        default=[1.0, 2.0, 3.0, 4.0],
        description="Input data to pass through the CLI subprocess",
    )


class MockCLIMultiGPUToolConfig(BaseConfig):
    """Config for mock CLI multi-GPU tool.

    Attributes:
        device (str): Device specification for 2 GPUs.
        scale_factor (float): Scale factor applied to input data.
    """

    device: str = ConfigField(
        default="cudax2",
        title="Device",
        description="Device spec for 2 GPUs (cudax2, cuda:0,1)",
        hidden=True,
        include_in_key=False,
    )

    scale_factor: float = ConfigField(
        default=2.0,
        title="Scale Factor",
        description="Scale factor applied to input data",
    )


class MockCLIMultiGPUToolOutput(BaseToolOutput):
    """Output from mock CLI multi-GPU tool.

    Attributes:
        result (list[float]): Output from the CLI subprocess.
        device_used (str): Device string passed to subprocess.
        cuda_visible_devices (str): CUDA_VISIBLE_DEVICES passed to subprocess.
    """

    result: list[float] = Field(
        description="Output from the CLI subprocess"
    )

    device_used: str = Field(
        description="Device string passed to subprocess"
    )

    cuda_visible_devices: str = Field(
        description="CUDA_VISIBLE_DEVICES passed to subprocess"
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
                        "result": self.result,
                        "device_used": self.device_used,
                        "cuda_visible_devices": self.cuda_visible_devices,
                    },
                    f,
                    indent=2,
                )
        elif file_format == "txt":
            with open(path, "w") as f:
                f.write(f"Device: {self.device_used}\n")
                f.write(f"CUDA_VISIBLE_DEVICES: {self.cuda_visible_devices}\n")
                f.write(f"Result: {self.result}\n")
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input():
    """Minimal valid input for testing and examples."""
    return MockCLIMultiGPUToolInput()


@tool(
    key="mock-cli-multi-gpu-tool-run",
    label="Mock CLI Multi-GPU Tool",
    category="testing",
    input_class=MockCLIMultiGPUToolInput,
    config_class=MockCLIMultiGPUToolConfig,
    output_class=MockCLIMultiGPUToolOutput,
    description="Minimal 2-GPU CLI subprocess tool for testing device routing",
    uses_gpu=True,
    device_count="2",
    example_input=example_input,
)
def run_mock_cli_multi_gpu_tool(
    inputs: MockCLIMultiGPUToolInput,
    config: MockCLIMultiGPUToolConfig | None = None,
    instance=None,
) -> MockCLIMultiGPUToolOutput:
    """Run mock CLI multi-GPU tool (subprocess-based with 2-GPU routing)."""

    result = ToolInstance.dispatch(
        "mock_cli_multi_gpu_tool",
        {
            "data": inputs.data,
            "device": config.device,
            "scale_factor": config.scale_factor,
        },
        instance=instance,
        config=config,
    )

    return MockCLIMultiGPUToolOutput(**result)
