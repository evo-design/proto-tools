"""Promoter Calculator standalone runner for ToolInstance venv execution.

Usage (called by ToolInstance, not directly):
    python run.py <input.json> <output.json>
"""

import json
import sys
from typing import Any

from standalone_helpers import get_logger

logger = get_logger(__name__)


# =============================================================================
# Implementation
# =============================================================================
def _get(prediction: Any, name: str) -> Any:
    # Upstream returns each promoter as a dict in some paths and as an object in
    # others; read both transparently.
    if isinstance(prediction, dict):
        return prediction[name]
    return getattr(prediction, name)


def _prediction_to_dict(prediction: Any) -> dict[str, Any]:
    """Project an upstream promoter result down to the contract's schema fields."""
    return {
        "tss_name": str(_get(prediction, "TSS_name")),
        "tss": int(_get(prediction, "TSS")),
        "strand": str(_get(prediction, "strand")),
        "dG_total": float(_get(prediction, "dG_total")),
        "Tx_rate": float(_get(prediction, "Tx_rate")),
        "promoter_sequence": str(_get(prediction, "promoter_sequence")),
        "length": int(_get(prediction, "length")),
        "UP_position": [int(x) for x in _get(prediction, "UP_position")],
        "hex35_position": [int(x) for x in _get(prediction, "hex35_position")],
        "spacer_position": [int(x) for x in _get(prediction, "spacer_position")],
        "hex10_position": [int(x) for x in _get(prediction, "hex10_position")],
        "disc_position": [int(x) for x in _get(prediction, "disc_position")],
    }


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Run the promoter calculator on each input sequence and emit per-sequence results."""
    from promoter_calculator.wrapper import promoter_calculator  # type: ignore[import-untyped]

    sequences = input_dict["sequences"]
    sequence_ids = input_dict["sequence_ids"]
    config = input_dict.get("config", {})
    threads = int(config["threads"])
    circular = bool(config["circular"])

    results = []
    for seq, seq_id in zip(sequences, sequence_ids, strict=True):
        # verbosity=0 pinned: any non-zero level prints progress to stdout, which
        # corrupts the subprocess JSON contract.
        predictions = promoter_calculator(seq, threads=threads, verbosity=0, circular=circular)
        results.append(
            {
                "sequence_id": seq_id,
                "predictions": [_prediction_to_dict(p) for p in predictions],
            }
        )

    return {"results": results}


# =============================================================================
# Device / memory protocol (CPU-only tool)
# =============================================================================
def to_device(device: str) -> dict[str, Any]:
    """No-op for CPU tools — workers unload between calls."""
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}


def get_memory_stats() -> dict[str, Any]:
    """No persistent state for this CPU tool."""
    return {"available": False, "framework": "cpu", "note": "CPU tool"}


# =============================================================================
# Entry point (called by ToolInstance)
# =============================================================================
if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        input_data = json.load(f)
    output = dispatch(input_data)
    with open(sys.argv[2], "w") as f:
        json.dump(output, f)
