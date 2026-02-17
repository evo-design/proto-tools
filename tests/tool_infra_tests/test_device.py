from __future__ import annotations

from unittest.mock import patch

import pytest

from bio_programming_tools.utils.device import determine_visible_devices

GPU_COUNT_PATH = "bio_programming_tools.utils.device.number_of_available_gpus"


class TestDetermineVisibleDevices:
    """Tests for determine_visible_devices()."""

    def test_cpu_returns_empty(self):
        assert determine_visible_devices("cpu") == ""

    def test_cuda_returns_zero(self):
        assert determine_visible_devices("cuda") == "0"

    @patch(GPU_COUNT_PATH, return_value=2)
    def test_cuda_colon_valid_index(self, _mock):
        assert determine_visible_devices("cuda:0") == "0"
        assert determine_visible_devices("cuda:1") == "1"

    @patch(GPU_COUNT_PATH, return_value=1)
    def test_cuda_colon_index_out_of_range(self, _mock):
        with pytest.raises(ValueError, match="Device index 1"):
            determine_visible_devices("cuda:1")

    @patch(GPU_COUNT_PATH, return_value=2)
    def test_cuda_colon_index_exceeds_gpu_count(self, _mock):
        with pytest.raises(ValueError, match="Device index 3"):
            determine_visible_devices("cuda:3")

    def test_cuda_colon_non_integer_index(self):
        with pytest.raises(ValueError):
            determine_visible_devices("cuda:abc")

    @patch(GPU_COUNT_PATH, return_value=2)
    def test_plain_int_string_valid(self, _mock):
        assert determine_visible_devices("0") == "0"
        assert determine_visible_devices("1") == "1"

    @patch(GPU_COUNT_PATH, return_value=1)
    def test_plain_int_string_out_of_range(self, _mock):
        with pytest.raises(ValueError):
            determine_visible_devices("2")

    def test_nonsense_string_raises(self):
        with pytest.raises(ValueError, match="Invalid device"):
            determine_visible_devices("tpu")
