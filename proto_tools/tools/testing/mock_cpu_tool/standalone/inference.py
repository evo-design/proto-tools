"""Standalone inference script for mock CPU tool.

Pure stdlib — no third-party deps, no model load. On each request the
worker reports its own pid, the ``OMP_NUM_THREADS`` env var it observed
at request time, and a stable ``process_unique_id`` (``{pid}-{startup_uuid}``)
per item. The startup uuid is computed once at module import so it survives
across multiple requests within the same persistent worker subprocess and
lets tests distinguish "fresh subprocess" from "warm reuse".
"""

import json
import os
import sys
import uuid
from typing import Any

# Auto-copied by worker bootstrap; same convention as the other mocks.
from standalone_helpers import get_logger

logger = get_logger(__name__)

# Stable per-worker id: same across all requests this subprocess handles,
# distinct across subprocesses. Module import happens once per worker.
_WORKER_UUID = uuid.uuid4().hex[:8]


# ============================================================================
# Dispatch
# ============================================================================


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    items = input_dict["items"]
    pid = os.getpid()
    omp = os.environ.get("OMP_NUM_THREADS", "(unset)")
    process_unique_id = f"{pid}-{_WORKER_UUID}"

    logger.info(
        "mock_cpu_tool: handling %d item(s) (pid=%d, OMP_NUM_THREADS=%s)",
        len(items),
        pid,
        omp,
    )

    return {
        "results": [
            {
                "item": item,
                "pid": pid,
                "omp_num_threads": omp,
                "process_unique_id": process_unique_id,
            }
            for item in items
        ]
    }


def to_device(device: str) -> dict[str, Any]:
    """No-op for a CPU-only tool — required by the DeviceManager protocol."""
    return {"success": True, "device": device, "note": "CPU tool, nothing to move"}


def get_memory_stats() -> dict[str, Any]:
    """No GPU memory to report — required by the DeviceManager protocol."""
    return {"available": False, "framework": "cpu", "reason": "CPU tool"}


# ============================================================================
# Main Entry Point
# ============================================================================


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("mock_cpu_tool: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
