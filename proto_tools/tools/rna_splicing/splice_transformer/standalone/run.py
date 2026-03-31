"""
SpliceTransformer standalone runner for ToolInstance venv execution.

Handles tissue-specific splice site prediction using the SpliceTransformer model.
Communicates via JSON input/output files (ToolInstance pattern).

Usage (called by ToolInstance, not directly):
    python run.py <input.json> <output.json>
"""

from __future__ import annotations

import json
import sys


# =============================================================================
# Main Entry Point
# =============================================================================
def run_splice_transformer(input_data: dict) -> dict:
    """Run SpliceTransformer inference on sequences.

    Args:
        input_data: Dict with keys: target_seqs, left_contexts, right_contexts,
                    context_length, device, verbose

    Returns:
        Dict with key: prediction (nested list, JSON-serializable)
    """
    from inference import SpliceTransformerModel

    context_length = input_data.get("context_length", 4000)
    device = input_data.get("device", "cuda")
    verbose = input_data.get("verbose", False)

    model = SpliceTransformerModel(context_length=context_length)

    prediction = model(
        target_seqs=input_data["target_seqs"],
        left_contexts=input_data["left_contexts"],
        right_contexts=input_data["right_contexts"],
        device=device,
        verbose=verbose,
    )

    # Always unload model after inference (no in-process caching in subprocess mode)
    model.unload()

    # Convert numpy array to nested list for JSON serialization
    return {"prediction": prediction.tolist()}


# =============================================================================
# Entry point (called by ToolInstance)
# =============================================================================

def to_device(device: str) -> dict:
    """Passthrough for CLI tool - automatically unloads after each call."""
    # CLI tool that spawns subprocesses and naturally unloads after each call
    # This is a passthrough for standardization with other tools
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            f"Usage: python {sys.argv[0]} <input_json_path> <output_json_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    with open(input_json_path, "r") as f:
        input_data = json.load(f)

    output_data = run_splice_transformer(input_data)

    with open(output_json_path, "w") as f:
        json.dump(output_data, f)
