"""proto_tools/tools/testing/mock_cpu_tool/mock_cpu_tool.py.

Pure-stdlib CPU-only mock tool used to integration-test ToolPool's CPU
fan-out path. Each call reports per-item: the worker subprocess PID, the
``OMP_NUM_THREADS`` env var the worker observed, and a per-worker stable
id (``{pid}-{startup_uuid}``) that lets tests verify both that distinct
subprocesses ran and that workers stayed warm across dispatches.

Counterpart to ``mock_pytorch_tool`` — same iterable-input shape, but no
torch/jax dependency so the env builds in seconds and no GPU is required.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    InputField,
    ToolInstance,
)

# ============================================================================
# Input / Config / Output
# ============================================================================


class MockCPUToolInput(BaseToolInput):
    """Input for mock CPU tool.

    Attributes:
        items (list[int]): List of integer items to process. Each item gets
            its own result row carrying the worker's pid / OMP setting / id.
    """

    items: list[int] = InputField(
        default=[1, 2, 3, 4],
        title="Items",
        description="List of integer items to process",
    )


class MockCPUToolConfig(BaseConfig):
    """Config for mock CPU tool.

    Overrides ``cpus_per_instance`` to ``1`` so ToolPool fans this mock out
    across its CPU budget — that's the whole point of the tool (testing
    fan-out end-to-end). Tests that exercise the explicit short-circuit
    override back to ``None`` in a subclass.
    """

    @property
    def cpus_per_instance(self) -> int | None:
        """Opt in to ToolPool CPU fan-out — this is the mock used to test it."""
        return 1


class MockCPUToolResult(BaseModel):
    """Per-item result from mock CPU tool.

    Attributes:
        item (int): The input item this result corresponds to.
        pid (int): OS process id of the worker subprocess that handled it.
        omp_num_threads (str): Value of ``OMP_NUM_THREADS`` observed by the
            worker subprocess (``"(unset)"`` if absent). Tests assert this
            equals ``cpus_per_instance`` to confirm thread-budget pinning.
        process_unique_id (str): ``{pid}-{startup_uuid}`` — stable across
            requests within the same persistent worker, distinct across
            worker subprocesses. Tests use this to count how many workers
            actually ran, and to verify warm-worker reuse across dispatches.
    """

    item: int = Field(title="Item", description="The input item this result corresponds to")
    pid: int = Field(title="PID", description="OS pid of the worker subprocess")
    omp_num_threads: str = Field(title="OMP Num Threads", description="OMP_NUM_THREADS observed by the worker")
    process_unique_id: str = Field(
        title="Process Unique ID", description="{pid}-{startup_uuid} — distinct per worker, stable per worker"
    )


class MockCPUToolOutput(BaseToolOutput):
    """Output from mock CPU tool.

    Attributes:
        results (list[MockCPUToolResult]): One result per input item, in input order.
    """

    results: list[MockCPUToolResult] = Field(title="Results", description="Per-item results, in input order")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")
        if file_format == "json":
            with open(path, "w") as f:
                json.dump({"results": [r.model_dump() for r in self.results]}, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return MockCPUToolInput()


@tool(
    key="mock-cpu-tool-run",
    label="Mock CPU Tool",
    category="testing",
    input_class=MockCPUToolInput,
    config_class=MockCPUToolConfig,
    output_class=MockCPUToolOutput,
    description="Pure-stdlib CPU mock tool for ToolPool CPU fan-out integration tests",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_field="items",
    iterable_output_field="results",
)
def run_mock_cpu_tool(
    inputs: MockCPUToolInput,
    config: MockCPUToolConfig,
    instance: Any = None,
) -> MockCPUToolOutput:
    """Run mock CPU tool — pure stdlib, no model load."""
    result = ToolInstance.dispatch(
        "mock_cpu_tool",
        {
            "items": list(inputs.items),
            "device": config.device,
        },
        instance=instance,
        config=config,
    )

    return MockCPUToolOutput(**result)
