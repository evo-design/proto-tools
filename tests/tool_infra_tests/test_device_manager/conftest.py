"""Shared fixtures for DeviceManager tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from bio_programming_tools.utils.device_manager import (
    DeviceManager,
)


@pytest.fixture
def mock_2_gpus():
    """Mock number_of_visible_gpus to return 2 for test duration."""
    with patch(
        "bio_programming_tools.utils.device_manager.number_of_visible_gpus",
        return_value=2,
    ) as mock:
        yield mock


@pytest.fixture
def mock_0_gpus():
    """Mock number_of_visible_gpus to return 0 for test duration."""
    with patch(
        "bio_programming_tools.utils.device_manager.number_of_visible_gpus",
        return_value=0,
    ) as mock:
        yield mock


@pytest.fixture
def device_manager(mock_2_gpus):
    """Create a fresh DeviceManager instance for each test with 2 GPUs."""
    # Reset singleton
    DeviceManager.reset_instance()
    dm = DeviceManager.get_instance()
    yield dm
    # Clean up
    DeviceManager.reset_instance()


@pytest.fixture
def no_gpus_manager(mock_0_gpus):
    """Create DeviceManager with no GPUs available."""
    DeviceManager.reset_instance()
    dm = DeviceManager.get_instance()
    yield dm
    DeviceManager.reset_instance()


@pytest.fixture
def mock_1_gpu():
    """Mock number_of_visible_gpus to return 1 for test duration."""
    with patch(
        "bio_programming_tools.utils.device_manager.number_of_visible_gpus",
        return_value=1,
    ) as mock:
        yield mock


@pytest.fixture
def device_manager_1gpu(mock_1_gpu):
    """Create a fresh DeviceManager instance for each test with 1 GPU."""
    DeviceManager.reset_instance()
    dm = DeviceManager.get_instance()
    yield dm
    DeviceManager.reset_instance()


@pytest.fixture
def mock_callback():
    """Create a mock eviction callback that tracks calls."""
    calls = []

    def callback(action: str) -> None:
        calls.append(action)

    callback.calls = calls  # Attach calls list for inspection
    return callback
