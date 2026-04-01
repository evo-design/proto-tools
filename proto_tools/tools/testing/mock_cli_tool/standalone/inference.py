"""Standalone inference script for mock CLI subprocess tool.

This script mimics CLI-based tools (Boltz2, RFDiffusion3, Protenix) that spawn
subprocesses for inference. It uses get_subprocess_device_env() for correct
CUDA_VISIBLE_DEVICES routing and subprocess.run for the actual "CLI" call.

The subprocess is a simple Python one-liner that scales input data, lightweight
but exercises the full subprocess device routing path.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from typing import Any

from standalone_helpers import get_subprocess_device_env

logger = logging.getLogger(__name__)


# ============================================================================
# CLI Model Wrapper
# ============================================================================


class MockCLIToolModel:
    """Wrapper that spawns CLI subprocesses for inference."""

    def __init__(self) -> None:
        """Initialize MockCLIToolModel."""
        self._loaded = True
        logger.info("MockCLIToolModel initialized")

    def __call__(
        self,
        data: list[float],
        scale_factor: float = 2.0,
        device: str = "cuda:0",
    ) -> dict[str, Any]:
        """Run inference via CLI subprocess."""
        # Build a simple CLI command that scales data
        data_json = json.dumps(data)
        cmd = [
            sys.executable,
            "-c",
            f"import json, os; "
            f"data = {data_json}; "
            f"scale = {scale_factor}; "
            f"result = [x * scale for x in data]; "
            f"cvd = os.environ.get('CUDA_VISIBLE_DEVICES', ''); "
            f"print(json.dumps({{'result': result, 'cuda_visible_devices': cvd}}))",
        ]

        # Get subprocess environment with correct CUDA_VISIBLE_DEVICES mapping
        env = get_subprocess_device_env(device)

        logger.debug("Running CLI command with device=%s", device)
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

        if proc.returncode != 0:
            raise RuntimeError(f"CLI subprocess failed: {proc.stderr}")

        output = json.loads(proc.stdout.strip())
        output["device_used"] = device
        return output  # type: ignore[no-any-return]


# ============================================================================
# Global Model Instance
# ============================================================================

_model: MockCLIToolModel | None = None


# ============================================================================
# Dispatch
# ============================================================================


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model

    if _model is None:
        _model = MockCLIToolModel()

    data = input_dict.get("data", [1.0, 2.0, 3.0, 4.0])
    device = input_dict.get("device", "cuda:0")
    scale_factor = input_dict.get("scale_factor", 2.0)

    return _model(data=data, scale_factor=scale_factor, device=device)


def to_device(device: str) -> dict[str, Any]:
    """Passthrough for CLI tool; automatically unloads after each call."""
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    # CLI tools don't hold models in memory; report as unavailable
    return {"available": False, "framework": "pytorch", "reason": "CLI tool, no persistent model"}


# ============================================================================
# Main Entry Point
# ============================================================================


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
