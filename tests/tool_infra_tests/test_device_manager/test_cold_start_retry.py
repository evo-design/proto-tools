"""Cold-start GPU-detection retry (the cloud runtime GPU container readiness race).

On a freshly-scheduled the cloud runtime GPU container, ``nvidia-smi`` can transiently report
0 GPUs before the driver is ready. DeviceManager re-polls a few times before
declaring "no GPUs visible" — but only when nvidia-smi exists, so a real CPU host
still fails fast.
"""

from unittest.mock import patch

import pytest

from proto_tools.utils.device_manager import _GPU_DETECT_RETRIES, DeviceManager


@pytest.fixture
def fresh_manager():
    """A fresh singleton DeviceManager per test (no GPU mock baked in)."""
    DeviceManager.reset_instance()
    yield DeviceManager.get_instance()
    DeviceManager.reset_instance()


def _noop_callback(action):  # eviction-callback interface
    return None


def test_transient_nvidia_smi_miss_recovers(fresh_manager):
    """GPU host: first detection returns 0, retry sees the GPU → allocation succeeds."""
    calls = {"n": 0}

    def count_gpus():
        calls["n"] += 1
        return 0 if calls["n"] == 1 else 2  # first poll races the driver, then 2 GPUs

    with (
        patch("proto_tools.utils.device_manager.nvidia_smi_present", return_value=True),
        patch("proto_tools.utils.device_manager.number_of_visible_gpus", side_effect=count_gpus),
        patch("proto_tools.utils.device_manager.is_exclusive_process_mode", return_value=False),
        patch("proto_tools.utils.device_manager.time.sleep") as mock_sleep,
    ):
        device = fresh_manager.request_device("esmfold", "inst1", device="cuda", eviction_callback=_noop_callback)

    assert device.startswith("cuda:")
    mock_sleep.assert_called()  # it retried rather than raising on the first 0


def test_cpu_host_raises_immediately_without_retry(fresh_manager):
    """Real CPU host (nvidia-smi absent): raise at once, never sleep/retry."""
    with (
        patch("proto_tools.utils.device_manager.nvidia_smi_present", return_value=False),
        patch("proto_tools.utils.device_manager.number_of_visible_gpus", return_value=0),
        patch("proto_tools.utils.device_manager.time.sleep") as mock_sleep,
    ):
        with pytest.raises(RuntimeError, match="no GPUs visible"):
            fresh_manager.request_device("esmfold", "inst1", device="cuda", eviction_callback=_noop_callback)

    mock_sleep.assert_not_called()  # no cold-start retry on a host with no nvidia-smi


def test_persistent_gpu_host_miss_raises_after_exhausting_retries(fresh_manager):
    """GPU host where GPUs never appear: retry the full budget, then raise."""
    with (
        patch("proto_tools.utils.device_manager.nvidia_smi_present", return_value=True),
        patch("proto_tools.utils.device_manager.number_of_visible_gpus", return_value=0),
        patch("proto_tools.utils.device_manager.time.sleep") as mock_sleep,
    ):
        with pytest.raises(RuntimeError, match="no GPUs visible"):
            fresh_manager.request_device("esmfold", "inst1", device="cuda", eviction_callback=_noop_callback)

    assert mock_sleep.call_count == _GPU_DETECT_RETRIES


def test_lease_api_retries_transient_miss(fresh_manager):
    """The lease API (second raise site) also retries a transient cold-start miss."""
    calls = {"n": 0}

    def count_gpus():
        calls["n"] += 1
        return 0 if calls["n"] == 1 else 2

    with (
        patch("proto_tools.utils.device_manager.nvidia_smi_present", return_value=True),
        patch("proto_tools.utils.device_manager.number_of_visible_gpus", side_effect=count_gpus),
        patch("proto_tools.utils.device_manager.is_exclusive_process_mode", return_value=False),
        patch("proto_tools.utils.device_manager.time.sleep") as mock_sleep,
    ):
        with fresh_manager.lease("esmfold", device="cuda") as device:
            assert device.startswith("cuda:")

    mock_sleep.assert_called()  # retried rather than raising on the first 0
