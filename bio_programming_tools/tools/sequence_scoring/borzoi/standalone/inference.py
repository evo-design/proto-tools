"""
Borzoi standalone inference implementation for venv execution.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import List

import torch

logger = logging.getLogger(__name__)

BORZOI_CONTEXT = 524_288
BORZOI_OUTPUT = 6_144


class BorzoiModel:
    """
    Borzoi model for regulatory activity prediction.

    Supports human and mouse models with optional FlashAttention.
    """

    def __init__(
        self,
        species: str = "human",
        replicate: str = "0",
        use_flash_attn: bool = True,
    ):
        """
        Initialize Borzoi model wrapper.

        Args:
            species: Species to predict for ('human' or 'mouse')
            replicate: Replicate to use (e.g., '0', '1', '2', '3')
            use_flash_attn: Whether to use FlashAttention
        """
        if species == "mouse" and use_flash_attn:
            raise ValueError(
                "FlashAttention (flashzoi) is not available for mouse models. "
                "Please set use_flash_attn=False."
            )

        self.species = species
        self.replicate = replicate
        self.use_flash_attn = use_flash_attn
        self._loaded = False

        # Determine model name
        prefix = "flashzoi" if use_flash_attn else "borzoi"
        self.model_name = f"johahi/{prefix}-replicate-{replicate}"
        if species == "mouse":
            self.model_name += "-mouse"

    def load(self, device: str, verbose: bool = False):
        """Load Borzoi model to device."""
        from borzoi_pytorch import Borzoi

        if verbose:
            logger.info(f"Loading Borzoi model: {self.model_name} on {device}")

        self.model = Borzoi.from_pretrained(self.model_name).to(device)
        self.device = device
        self._loaded = True

        if verbose:
            logger.info("Borzoi model loaded successfully")

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
        avg_output_tracks: bool = True,
        device: str = "cuda",
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Run Borzoi inference on a DNA sequence.

        Args:
            sequence: DNA sequence (must be BORZOI_CONTEXT=524,288 bp)
            output_tracks: List of track indices to extract
            avg_output_tracks: Whether to average across output tracks
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

        # Prepare input
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        indices = [mapping[char] for char in sequence]
        onehot = torch.zeros(4, len(sequence), device=device)
        onehot[indices, range(len(sequence))] = 1

        # Run prediction with autocast
        with torch.amp.autocast(device):
            with torch.inference_mode():
                output = self.model(onehot.unsqueeze(0), is_human=(self.species == "human"))

        # Process output - always return 2D (num_tracks, positions)
        if avg_output_tracks:
            prediction = output[0][output_tracks, :].mean(0, keepdim=True)
        else:
            prediction = output[0][output_tracks, :]

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

    # Extract initialization parameters
    species = input_data.get("species", "human")
    replicate = input_data.get("replicate", "0")
    use_flash_attn = input_data.get("use_flash_attn", True)

    # Auto-detect flash-attn availability
    if use_flash_attn:
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            logger.warning(
                "flash-attn not installed — falling back to use_flash_attn=False (standard borzoi model)"
            )
            use_flash_attn = False

    # Create model
    model = BorzoiModel(
        species=species,
        replicate=replicate,
        use_flash_attn=use_flash_attn,
    )

    # Run inference
    prediction = model(
        sequence=input_data["sequence"],
        output_tracks=input_data["output_tracks"],
        avg_output_tracks=input_data.get("avg_output_tracks", True),
        device=input_data.get("device", "cuda"),
        verbose=input_data.get("verbose", False),
    )

    # Prepare output
    output_data = {
        "prediction": prediction.tolist() if hasattr(prediction, "tolist") else prediction,
        "applied_species": species,
        "applied_replicate": replicate,
    }

    with open(output_json_path, "w") as f:
        json.dump(output_data, f)
