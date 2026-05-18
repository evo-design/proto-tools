"""proto_tools/tools/testing/mock_cli_tool/mock_cli_tool.py.

This is a minimal CLI-pattern tool that spawns a subprocess for inference,
matching the pattern used by tools like Boltz2, RFDiffusion3, and Protenix.
It uses get_subprocess_device_env() for device routing.
"""

import json
from pathlib import Path
from typing import Any

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


class MockCLIToolInput(BaseToolInput):
    """Input for mock CLI tool.

    Attributes:
        data (list[float]): Input data to pass through the CLI subprocess.
    """

    data: list[float] = InputField(
        default=[1.0, 2.0, 3.0, 4.0],
        description="Input data to pass through the CLI subprocess",
    )


class MockCLIToolConfig(BaseConfig):
    """Config for mock CLI tool.

    Attributes:
        device (str): Device to run on.
        scale_factor (float): Scale factor applied to input data.
    """

    device: str = ConfigField(
        default="cuda",
        title="Device",
        description="Device to run on",
        include_in_key=False,
    )

    scale_factor: float = ConfigField(
        default=2.0,
        title="Scale Factor",
        description="Scale factor applied to input data",
    )


class MockCLIToolOutput(BaseToolOutput):
    """Output from mock CLI tool.

    Attributes:
        result (list[float]): Output from the CLI subprocess.
        device_used (str): Device the subprocess ran on.
        cuda_visible_devices (str): CUDA_VISIBLE_DEVICES passed to subprocess.
    """

    result: list[float] = Field(description="Output from the CLI subprocess")

    device_used: str = Field(description="Device the subprocess ran on")

    cuda_visible_devices: str = Field(description="CUDA_VISIBLE_DEVICES passed to subprocess")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "txt"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
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


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return MockCLIToolInput()


@tool(
    key="mock-cli-tool-run",
    label="Mock CLI Tool",
    category="testing",
    input_class=MockCLIToolInput,
    config_class=MockCLIToolConfig,
    output_class=MockCLIToolOutput,
    description="Minimal CLI subprocess tool for testing device routing",
    uses_gpu=True,
    device_count="1",
    example_input=example_input,
)
def run_mock_cli_tool(
    inputs: MockCLIToolInput,
    config: MockCLIToolConfig,
    instance: Any = None,
) -> MockCLIToolOutput:
    """Run mock CLI tool (subprocess-based for testing device routing)."""
    result = ToolInstance.dispatch(
        "mock_cli_tool",
        {
            "data": inputs.data,
            "device": config.device,
            "scale_factor": config.scale_factor,
        },
        instance=instance,
        config=config,
    )

    return MockCLIToolOutput(**result)
