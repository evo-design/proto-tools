"""Enformer standalone inference implementation for venv execution."""

import json
import logging
import sys
from typing import Any

import torch
from standalone_helpers import serialize_output

logger = logging.getLogger(__name__)

ENFORMER_CONTEXT = 196_608
ENFORMER_OUTPUT = 896


class EnformerModel:
    """Enformer model for regulatory activity and gene expression prediction.

    Uses the EleutherAI/enformer-official-rough model for human and mouse predictions.
    """

    def __init__(self) -> None:
        """Initialize Enformer model wrapper."""
        self._loaded = False

    def load(self, device: str, verbose: bool = False) -> None:
        """Load Enformer model to device."""
        from enformer_pytorch import from_pretrained

        if verbose:
            logger.info(f"Loading Enformer model on {device}")

        self.model = from_pretrained("EleutherAI/enformer-official-rough").to(device)
        self.device = device
        self._loaded = True

        if verbose:
            logger.info("Enformer model loaded successfully")

    def unload(self, verbose: bool = False) -> None:
        """Move model to CPU and clear CUDA cache."""
        if hasattr(self, "model") and hasattr(self, "device"):
            if verbose:
                logger.info(f"Moving model from {self.device} to CPU")
            self.to_device("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to_device(self, device: str) -> dict[str, Any]:  # type: ignore[return]
        """Move model to a different device."""
        if not self._loaded:
            raise RuntimeError("Cannot move unloaded model to device. Call load() first.")

        if self.device != device:
            self.model = self.model.to(device)
            self.device = device

    def __call__(
        self,
        sequence: str,
        output_tracks: list[int],
        species: str,
        device: str = "cuda",
        verbose: bool = False,
    ) -> torch.Tensor:
        """Run Enformer inference on a DNA sequence.

        Args:
            sequence: DNA sequence (must be ENFORMER_CONTEXT=196,608 bp)
            output_tracks: List of track indices to extract
            species: Species to predict for ('human' or 'mouse')
            device: Device to run on (defaults to cuda if available)
            verbose: Whether to print status messages

        Returns:
            Predicted regulatory activity tensor
        """
        # Lazy load on first call or device change
        if not self._loaded:
            self.load(device, verbose=verbose)
        elif self.device != device:
            self.to_device(device)

        # Prepare input (Enformer handles 'N' as index 4)
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        input_ids = torch.tensor([mapping.get(char, 4) for char in sequence]).unsqueeze(0).to(device)

        # Run prediction
        with torch.inference_mode():
            output = self.model(input_ids)

        # Extract prediction for specified tracks and species
        prediction = output[species][0][:, output_tracks]

        self.unload(verbose=verbose)
        return prediction


# ============================================================================
# Dispatch
# ============================================================================
_model: EnformerModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = EnformerModel()

    operation = input_dict["operation"]
    if operation == "predict":
        prediction = _model(
            sequence=input_dict["sequence"],
            output_tracks=input_dict["output_tracks"],
            species=input_dict["species"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
        )
        return {
            "prediction": prediction,
            "applied_species": input_dict["species"],
        }
    raise ValueError(f"Unknown operation: {operation}")


def to_device(device: str) -> dict[str, Any]:
    """Move model to specified device (called by DeviceManager)."""
    global _model
    if _model is not None and _model._loaded:
        _model.to_device(device)
        return {"success": True, "device": device}
    # Model not loaded yet - will use device on next call
    return {"success": True, "device": device, "note": "model not loaded yet"}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    from standalone_helpers import get_pytorch_memory_stats

    global _model
    device = _model.device if _model and hasattr(_model, "device") else 0
    return get_pytorch_memory_stats(device)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
