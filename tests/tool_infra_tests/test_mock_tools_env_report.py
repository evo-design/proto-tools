"""tests/tool_infra_tests/test_mock_tools_env_report.py

Environment report smoke tests for mock PyTorch tools."""
import pytest

from proto_tools.tools.testing.mock_pytorch_multi_gpu_tool import (
    MockPyTorchMultiGPUToolInput,
    run_mock_pytorch_multi_gpu_tool,
)
from proto_tools.tools.testing.mock_pytorch_tool import (
    MockPyTorchToolInput,
    run_mock_pytorch_tool,
)
from proto_tools.utils.device import number_of_visible_gpus


@pytest.mark.include_in_env_report(category="mock")
@pytest.mark.uses_gpu
def test_mock_pytorch_tool_env_report():
    """Smoke test: mock single-GPU PyTorch tool builds and runs."""
    result = run_mock_pytorch_tool(MockPyTorchToolInput())
    assert result.success
    assert len(result.results) == 1


@pytest.mark.include_in_env_report(category="mock")
@pytest.mark.uses_gpu
def test_mock_pytorch_multi_gpu_tool_env_report():
    """Smoke test: mock multi-GPU PyTorch tool builds and runs."""
    if number_of_visible_gpus() < 2:
        pytest.skip("Requires 2+ GPUs")
    result = run_mock_pytorch_multi_gpu_tool(MockPyTorchMultiGPUToolInput())
    assert result.success
