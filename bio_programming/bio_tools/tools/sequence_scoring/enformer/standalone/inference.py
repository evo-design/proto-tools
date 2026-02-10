"""
Enformer standalone inference implementation for venv execution.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import List

import torch

logger = logging.getLogger(__name__)

ENFORMER_CONTEXT = 196_608
ENFORMER_OUTPUT = 896


class EnformerModel:
    """
    Enformer model for regulatory activity and gene expression prediction.

    Uses the EleutherAI/enformer-official-rough model for human and mouse predictions.
    """

    def __init__(self):
        """Initialize Enformer model wrapper."""
        self._loaded = False

    def load(self, device: str, verbose: bool = False):
        """Load Enformer model to device."""
        from enformer_pytorch import from_pretrained

        if verbose:
            logger.info(f"Loading Enformer model on {device}")

        self.model = from_pretrained("EleutherAI/enformer-official-rough").to(device)
        self.device = device
        self._loaded = True

        if verbose:
            logger.info("Enformer model loaded successfully")

    def unload(self, verbose: bool = False):
        """Move model to CPU and clear CUDA cache."""
        if hasattr(self, "model") and hasattr(self, "device"):
            if verbose:
                logger.info(f"Moving model from {self.device} to CPU")
            self.to_device("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to_device(self, device: str):
        """Move model to a different device."""
        if not self._loaded:
            raise RuntimeError("Cannot move unloaded model to device. Call load() first.")

        if self.device != device:
            self.model = self.model.to(device)
            self.device = device

    def __call__(
        self,
        sequence: str,
        output_tracks: List[int],
        species: str,
        device: str = "cuda",
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Run Enformer inference on a DNA sequence.

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
        input_ids = torch.tensor(
            [mapping.get(char, 4) for char in sequence]
        ).unsqueeze(0).to(device)

        # Run prediction
        with torch.inference_mode():
            output = self.model(input_ids)

        # Extract prediction for specified tracks and species
        prediction = output[species][0][:, output_tracks]

        self.unload(verbose=verbose)
        return prediction


# Standalone script entry point for venv execution
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python inference.py <input_json_path> <output_json_path>")

    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    with open(input_json_path, "r") as f:
        input_data = json.load(f)

    # Create model
    model = EnformerModel()

    # Run inference
    prediction = model(
        sequence=input_data["sequence"],
        output_tracks=input_data["output_tracks"],
        species=input_data.get("species", "human"),
        device=input_data.get("device", "cuda"),
        verbose=input_data.get("verbose", False),
    )

    # Prepare output
    output_data = {
        "prediction": prediction.tolist() if hasattr(prediction, "tolist") else prediction,
        "applied_species": input_data.get("species", "human"),
    }

    with open(output_json_path, "w") as f:
        json.dump(output_data, f)
