"""Tests for device string parsing and CUDA_VISIBLE_DEVICES generation."""

import os

import pytest

from bio_programming_tools.utils.device import (
    DeviceSpec,
    PROTO_DEFAULT_CONCURRENCY,
    determine_visible_devices,
    number_of_available_gpus,
    number_of_physical_gpus,
    number_of_visible_gpus,
    parse_device_string,
)

# ── parse_device_string() ────────────────────────────────────────────────────


def test_parse_device_string_cpu():
    """Test parsing CPU device string."""
    spec = parse_device_string("cpu")
    assert spec.kind == "cpu"
    assert spec.devices == ["cpu"]
    assert spec.count == 1
    assert spec.concurrency == 0


def test_parse_device_string_single_auto():
    """Test parsing single auto-allocate GPU."""
    spec = parse_device_string("cuda")
    assert spec.kind == "cuda"
    assert spec.devices is None
    assert spec.count == 1


def test_parse_device_string_single_explicit():
    """Test parsing single explicit GPU."""
    spec = parse_device_string("cuda:0")
    assert spec.devices == ["cuda:0"]
    assert spec.count == 1

    spec = parse_device_string("cuda:2")
    assert spec.devices == ["cuda:2"]
    assert spec.count == 1


def test_parse_device_string_multi_auto():
    """Test parsing multi-GPU auto-allocate."""
    spec = parse_device_string("cudax2")
    assert spec.devices is None
    assert spec.count == 2

    spec = parse_device_string("cudax4")
    assert spec.devices is None
    assert spec.count == 4


def test_parse_device_string_multi_explicit_shorthand():
    """Test parsing multi-GPU explicit shorthand syntax."""
    spec = parse_device_string("cuda:0,1")
    assert spec.devices == ["cuda:0", "cuda:1"]
    assert spec.count == 2

    spec = parse_device_string("cuda:2,3,4")
    assert spec.devices == ["cuda:2", "cuda:3", "cuda:4"]
    assert spec.count == 3


def test_parse_device_string_multi_explicit_verbose():
    """Test parsing multi-GPU explicit verbose syntax."""
    spec = parse_device_string("cuda:0,cuda:1")
    assert spec.devices == ["cuda:0", "cuda:1"]
    assert spec.count == 2

    spec = parse_device_string("cuda:1,cuda:3,cuda:5")
    assert spec.devices == ["cuda:1", "cuda:3", "cuda:5"]
    assert spec.count == 3


def test_parse_device_string_invalid_zero_count():
    """Test that cudax0 raises ValueError."""
    with pytest.raises(ValueError, match="must be >= 1"):
        parse_device_string("cudax0")


def test_parse_device_string_invalid_negative_count():
    """Test that negative count raises ValueError."""
    with pytest.raises(ValueError, match="must be >= 1"):
        parse_device_string("cudax-1")


def test_parse_device_string_invalid_missing_prefix():
    """Test that shorthand without prefix raises ValueError."""
    with pytest.raises(ValueError, match="shorthand .* without prefix"):
        parse_device_string("0,1")


def test_parse_device_string_invalid_malformed():
    """Test that malformed strings raise ValueError."""
    with pytest.raises(ValueError, match="Invalid device"):
        parse_device_string("gpu:0")

    with pytest.raises(ValueError, match="Invalid device"):
        parse_device_string("cuda:")


def test_parse_device_string_whitespace_handling():
    """Test that whitespace is properly stripped."""
    spec = parse_device_string("  cudax2  ")
    assert spec.devices is None
    assert spec.count == 2

    spec = parse_device_string(" cuda:0, 1 ")
    assert spec.devices == ["cuda:0", "cuda:1"]
    assert spec.count == 2


# ============================================================================
# parse_device_string() Tests — Proto
# ============================================================================


def test_parse_proto_default():
    """Bare 'proto' gets default concurrency."""
    spec = parse_device_string("proto")
    assert spec.kind == "proto"
    assert spec.devices is None
    assert spec.count == 1
    assert spec.concurrency == PROTO_DEFAULT_CONCURRENCY


def test_parse_proto_colon():
    """'proto:64' sets concurrency via colon syntax."""
    spec = parse_device_string("proto:64")
    assert spec.kind == "proto"
    assert spec.concurrency == 64


def test_parse_proto_x_suffix():
    """'protox64' sets concurrency via x-suffix syntax."""
    spec = parse_device_string("protox64")
    assert spec.kind == "proto"
    assert spec.concurrency == 64


def test_parse_proto_colon_and_x_equivalent():
    """'proto:32' and 'protox32' produce equivalent DeviceSpecs."""
    assert parse_device_string("proto:32") == parse_device_string("protox32")


def test_parse_proto_invalid_zero():
    """Proto concurrency of 0 raises ValueError."""
    with pytest.raises(ValueError, match="must be >= 1"):
        parse_device_string("proto:0")


def test_parse_proto_invalid_negative():
    """Negative proto concurrency raises ValueError."""
    with pytest.raises(ValueError, match="must be >= 1"):
        parse_device_string("protox-1")


def test_parse_proto_invalid_text():
    """Non-numeric proto suffix raises ValueError."""
    with pytest.raises(ValueError, match="count must be integer"):
        parse_device_string("proto:abc")


def test_parse_cuda_returns_devicespec():
    """All CUDA variants return DeviceSpec with kind='cuda'."""
    for device_str in ("cuda", "cudax2", "cuda:0", "cuda:0,1"):
        spec = parse_device_string(device_str)
        assert isinstance(spec, DeviceSpec)
        assert spec.kind == "cuda"


def test_devicespec_frozen():
    """DeviceSpec is immutable."""
    spec = parse_device_string("cuda")
    with pytest.raises(AttributeError):
        spec.kind = "proto"


def _expected_physical(*logical_indices: int) -> str:
    """Map logical CUDA indices to the expected physical device string.

    When CUDA_VISIBLE_DEVICES is set (e.g. "5,6,7"), logical index 0 maps to
    physical "5", logical 1 to "6", etc. Without it, logical == physical.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd and cvd.strip():
        devices = [d.strip() for d in cvd.split(",")]
        return ",".join(devices[i] for i in logical_indices)
    return ",".join(str(i) for i in logical_indices)


# ── determine_visible_devices() ──────────────────────────────────────────────


def test_determine_visible_devices_cpu():
    """Test CUDA_VISIBLE_DEVICES for CPU."""
    assert determine_visible_devices("cpu") == ""


def test_determine_visible_devices_single_auto():
    """Test CUDA_VISIBLE_DEVICES for single auto-allocate."""
    assert determine_visible_devices("cuda") == _expected_physical(0)


@pytest.mark.uses_gpu
def test_determine_visible_devices_single_explicit():
    """Test CUDA_VISIBLE_DEVICES for single explicit GPU."""
    result = determine_visible_devices("cuda:0")
    assert result == _expected_physical(0)


@pytest.mark.uses_gpu(3)
def test_determine_visible_devices_multi_auto():
    """Test CUDA_VISIBLE_DEVICES for multi-GPU auto-allocate."""
    # Note: This is a fallback case - DeviceManager should resolve cudax2 to specific devices
    result = determine_visible_devices("cudax2")
    assert result == _expected_physical(0, 1)

    result = determine_visible_devices("cudax3")
    assert result == _expected_physical(0, 1, 2)


def test_determine_visible_devices_multi_explicit_shorthand():
    """Test CUDA_VISIBLE_DEVICES for multi-GPU shorthand."""
    num_gpus = number_of_available_gpus()
    if num_gpus < 2:
        pytest.skip(f"Test requires 2+ GPUs, found {num_gpus}")

    result = determine_visible_devices("cuda:0,1")
    assert result == _expected_physical(0, 1)

    if num_gpus >= 5:
        result = determine_visible_devices("cuda:2,3,4")
        assert result == _expected_physical(2, 3, 4)


def test_determine_visible_devices_multi_explicit_verbose():
    """Test CUDA_VISIBLE_DEVICES for multi-GPU verbose syntax."""
    num_gpus = number_of_available_gpus()
    if num_gpus < 2:
        pytest.skip(f"Test requires 2+ GPUs, found {num_gpus}")

    result = determine_visible_devices("cuda:0,cuda:1")
    assert result == _expected_physical(0, 1)

    if num_gpus >= 4:
        result = determine_visible_devices("cuda:1,cuda:3")
        assert result == _expected_physical(1, 3)


def test_determine_visible_devices_invalid_index_exceeds_gpus():
    """Test that device index exceeding available GPUs raises ValueError."""
    # This test assumes system has fewer than 100 GPUs.
    # Error message depends on whether any GPUs exist at all.
    with pytest.raises(ValueError, match="exceeds|no GPUs detected"):
        determine_visible_devices("cuda:100")

    with pytest.raises(ValueError, match="exceeds|no GPUs detected"):
        determine_visible_devices("cuda:50,51")


# ── Device count requirement parsing ─────────────────────────────────────────


def test_parse_device_count_exact():
    """Test parsing exact device count specifications."""
    from bio_programming_tools.utils.device import parse_device_count_requirement

    # Single device
    result = parse_device_count_requirement("1")
    assert result == {"min": 1, "max": 1}

    # Two devices
    result = parse_device_count_requirement("2")
    assert result == {"min": 2, "max": 2}


def test_parse_device_count_range():
    """Test parsing range device count specifications."""
    from bio_programming_tools.utils.device import parse_device_count_requirement

    # 1-2 devices
    result = parse_device_count_requirement("1-2")
    assert result == {"min": 1, "max": 2}

    # 1-4 devices
    result = parse_device_count_requirement("1-4")
    assert result == {"min": 1, "max": 4}


def test_parse_device_count_open_ended():
    """Test parsing open-ended device count specifications."""
    from bio_programming_tools.utils.device import parse_device_count_requirement

    # At least 1 device
    result = parse_device_count_requirement(">=1")
    assert result == {"min": 1, "max": None}

    # At most 2 devices
    result = parse_device_count_requirement("<=2")
    assert result == {"min": None, "max": 2}


def test_parse_device_count_combined():
    """Test parsing combined device count specifications."""
    from bio_programming_tools.utils.device import parse_device_count_requirement

    # 1 to 4 devices
    result = parse_device_count_requirement(">=1,<=4")
    assert result == {"min": 1, "max": 4}

    # Order shouldn't matter
    result = parse_device_count_requirement("<=4,>=1")
    assert result == {"min": 1, "max": 4}


def test_parse_device_count_invalid():
    """Test that invalid device count specifications raise ValueError."""
    from bio_programming_tools.utils.device import parse_device_count_requirement

    # Invalid format
    with pytest.raises(ValueError, match="Invalid device count"):
        parse_device_count_requirement("abc")

    # Negative count (will be parsed as invalid range)
    with pytest.raises(ValueError, match="Invalid device count"):
        parse_device_count_requirement("-1")

    # Invalid range (min > max)
    with pytest.raises(ValueError, match="cannot exceed maximum"):
        parse_device_count_requirement("3-1")

    # Zero count
    with pytest.raises(ValueError, match="must be >= 1"):
        parse_device_count_requirement("0")


def test_validate_device_allocation_within_range():
    """Test validation passes when allocation is within range."""
    from bio_programming_tools.utils.device import validate_device_allocation

    # Exact match
    validate_device_allocation(1, "1", "test-tool")

    # Within range
    validate_device_allocation(1, "1-2", "test-tool")
    validate_device_allocation(2, "1-2", "test-tool")

    # Open-ended lower bound
    validate_device_allocation(5, ">=1", "test-tool")


def test_validate_device_allocation_over_allocation(caplog):
    """Test validation warns on over-allocation (more than max)."""
    import logging

    from bio_programming_tools.utils.device import validate_device_allocation

    with caplog.at_level(logging.WARNING):
        # Requesting 2 devices for tool that needs exactly 1
        validate_device_allocation(2, "1", "test-tool")
        assert "requires at most 1 device(s), but 2 requested" in caplog.text

        caplog.clear()

        # Requesting 3 devices for tool that needs at most 2
        validate_device_allocation(3, "<=2", "test-tool")
        assert "requires at most 2 device(s), but 3 requested" in caplog.text


def test_validate_device_allocation_under_allocation():
    """Test validation raises error on under-allocation (fewer than min)."""
    from bio_programming_tools.utils.device import validate_device_allocation

    # Requesting 1 device for tool that needs 2
    with pytest.raises(ValueError, match="requires at least 2"):
        validate_device_allocation(1, "2", "test-tool")

    # Requesting 1 device for tool that needs at least 2
    with pytest.raises(ValueError, match="requires at least 2"):
        validate_device_allocation(1, ">=2", "test-tool")

    # Requesting 0 devices for tool that needs 1-2
    with pytest.raises(ValueError, match="requires at least 1"):
        validate_device_allocation(0, "1-2", "test-tool")


# ── CUDA_VISIBLE_DEVICES mapping ─────────────────────────────────────────────


def test_determine_visible_devices_with_parent_cuda_visible_devices(monkeypatch):
    """Test that determine_visible_devices correctly maps logical to physical devices."""
    # Simulate parent process with CUDA_VISIBLE_DEVICES=3,5,7
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5,7")
    
    # Logical cuda:0 should map to physical GPU 3
    assert determine_visible_devices("cuda:0") == "3"
    
    # Logical cuda:1 should map to physical GPU 5
    assert determine_visible_devices("cuda:1") == "5"
    
    # Logical cuda:2 should map to physical GPU 7
    assert determine_visible_devices("cuda:2") == "7"
    
    # Multi-device: logical cuda:0,1 should map to physical 3,5
    assert determine_visible_devices("cuda:0,1") == "3,5"
    
    # Multi-device: logical cuda:1,2 should map to physical 5,7
    assert determine_visible_devices("cuda:1,2") == "5,7"


def test_determine_visible_devices_with_single_parent_device(monkeypatch):
    """Test mapping when parent has only one visible device."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    
    # Only cuda:0 should be valid
    assert determine_visible_devices("cuda:0") == "4"


def test_determine_visible_devices_auto_allocate_with_parent(monkeypatch):
    """Test auto-allocation respects parent CUDA_VISIBLE_DEVICES."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3,4")
    
    # Auto-allocate single GPU should use first visible device
    assert determine_visible_devices("cuda") == "2"


def test_determine_visible_devices_parent_with_spaces(monkeypatch):
    """Test that parent CUDA_VISIBLE_DEVICES with spaces is handled correctly."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", " 3, 5 , 7 ")
    
    assert determine_visible_devices("cuda:0") == "3"
    assert determine_visible_devices("cuda:1") == "5"
    assert determine_visible_devices("cuda:2") == "7"


def test_determine_visible_devices_empty_parent(monkeypatch):
    """Test behavior when CUDA_VISIBLE_DEVICES is set but empty."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    # Empty CUDA_VISIBLE_DEVICES means no GPUs are visible, so cuda:0 should fail
    with pytest.raises(ValueError, match="no GPUs detected"):
        determine_visible_devices("cuda:0")


def test_determine_visible_devices_cpu_ignores_parent(monkeypatch):
    """Test that CPU device ignores parent CUDA_VISIBLE_DEVICES."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5,7")

    assert determine_visible_devices("cpu") == ""


# ── determine_visible_devices() — List Input ─────────────────────────────────


@pytest.mark.uses_gpu
def test_determine_visible_devices_list_basic():
    """List of CUDA device strings returns deduplicated physical indices."""
    num_gpus = number_of_available_gpus()
    if num_gpus < 2:
        pytest.skip(f"Test requires 2+ GPUs, found {num_gpus}")

    result = determine_visible_devices(["cuda:0", "cuda:1"])
    assert result == _expected_physical(0, 1)


def test_determine_visible_devices_list_skips_non_cuda():
    """Non-CUDA entries (proto, cpu) are skipped in list input."""
    num_gpus = number_of_available_gpus()
    if num_gpus < 1:
        pytest.skip(f"Test requires 1+ GPU, found {num_gpus}")

    result = determine_visible_devices(["cuda:0", "proto"])
    assert result == _expected_physical(0)


def test_determine_visible_devices_list_proto_only():
    """List with only proto/cpu entries returns empty string."""
    result = determine_visible_devices(["proto", "cpu"])
    assert result == ""


def test_determine_visible_devices_list_with_parent(monkeypatch):
    """List input respects CUDA_VISIBLE_DEVICES mapping."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5,7")

    result = determine_visible_devices(["cuda:0", "cuda:2"])
    assert result == "3,7"


def test_determine_visible_devices_list_deduplicates(monkeypatch):
    """Duplicate CUDA indices are deduplicated in output."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5,7")

    result = determine_visible_devices(["cuda:0", "cuda:0", "cuda:1"])
    assert result == "3,5"


def test_determine_visible_devices_list_invalid_index():
    """List with an out-of-range CUDA index raises ValueError."""
    with pytest.raises(ValueError, match="exceeds|no GPUs"):
        determine_visible_devices(["cuda:0", "cuda:100"])


def test_determine_visible_devices_list_no_gpus(monkeypatch):
    """List with CUDA devices on a no-GPU system raises ValueError."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    with pytest.raises(ValueError, match="no GPUs detected"):
        determine_visible_devices(["cuda:0"])


def test_determine_visible_devices_list_int_entries():
    """List of int entries validates correctly."""
    num_gpus = number_of_available_gpus()
    if num_gpus < 2:
        pytest.skip(f"Test requires 2+ GPUs, found {num_gpus}")

    result = determine_visible_devices([0, 1])
    assert result == _expected_physical(0, 1)


def test_determine_visible_devices_list_int_invalid():
    """List with out-of-range int raises ValueError."""
    with pytest.raises(ValueError, match="exceeds|no GPUs"):
        determine_visible_devices([100])


# ── number_of_physical_gpus() and number_of_visible_gpus() ───────────────────


def test_number_of_physical_gpus_returns_positive_or_zero():
    """Test that number_of_physical_gpus() returns non-negative count."""
    count = number_of_physical_gpus()
    assert count >= 0


def test_number_of_visible_gpus_without_cuda_visible_devices(monkeypatch):
    """Test number_of_visible_gpus() without CUDA_VISIBLE_DEVICES set."""
    # Remove CUDA_VISIBLE_DEVICES if it exists
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    
    # Should return same as physical GPUs
    physical = number_of_physical_gpus()
    visible = number_of_visible_gpus()
    assert visible == physical


def test_number_of_visible_gpus_with_cuda_visible_devices(monkeypatch):
    """Test number_of_visible_gpus() with CUDA_VISIBLE_DEVICES set."""
    # Set CUDA_VISIBLE_DEVICES to a subset of GPUs
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2,5")
    
    # Should return count of devices in CUDA_VISIBLE_DEVICES
    visible = number_of_visible_gpus()
    assert visible == 3


def test_number_of_visible_gpus_with_single_device(monkeypatch):
    """Test number_of_visible_gpus() with single GPU visible."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    
    visible = number_of_visible_gpus()
    assert visible == 1


def test_number_of_visible_gpus_with_empty_cuda_visible_devices(monkeypatch):
    """Test number_of_visible_gpus() with empty CUDA_VISIBLE_DEVICES (no GPUs visible)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    
    visible = number_of_visible_gpus()
    assert visible == 0


@pytest.mark.uses_gpu
def test_number_of_visible_gpus_with_invalid_indices(monkeypatch, caplog):
    """Test number_of_visible_gpus() warns when CUDA_VISIBLE_DEVICES has invalid indices."""
    import logging
    
    # Set CUDA_VISIBLE_DEVICES with indices that exceed physical GPU count
    # (assuming system has < 100 GPUs)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,99,100")
    
    with caplog.at_level(logging.WARNING):
        visible = number_of_visible_gpus()
        # Should still return the count (3 devices specified)
        assert visible == 3
        # Should log warning about invalid indices
        assert "non-existent GPU(s)" in caplog.text


def test_number_of_visible_gpus_backward_compat():
    """Test that number_of_available_gpus() is alias for number_of_visible_gpus()."""
    # Both should return same value
    assert number_of_available_gpus() == number_of_visible_gpus()
