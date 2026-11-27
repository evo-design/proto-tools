"""Standalone Rule Set 3 inference runner."""

import json
import sys
from typing import Any

from standalone_helpers import get_logger

logger = get_logger(__name__)


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Return sequence-only Rule Set 3 predictions."""
    from rs3.seq import predict_seq  # type: ignore[import-not-found]

    scores = predict_seq(input_dict["sequences"], sequence_tracr=input_dict["tracr_rna"])
    return {"scores": [float(score) for score in scores]}


def to_device(device: str) -> dict[str, Any]:
    """Confirm CPU placement for the model runner."""
    del device
    return {"success": True, "device": "cpu"}


def get_memory_stats() -> dict[str, Any]:
    """Report that GPU memory accounting is inapplicable."""
    return {"available": False, "framework": "cpu"}


if __name__ == "__main__":
    with open(sys.argv[1]) as handle:
        input_data = json.load(handle)
    with open(sys.argv[2], "w") as handle:
        json.dump(dispatch(input_data), handle)
