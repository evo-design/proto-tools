"""Borzoi standalone inference implementation for venv execution."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import torch

logger = logging.getLogger(__name__)

BORZOI_CONTEXT = 524_288
BORZOI_OUTPUT = 6_144


class BorzoiModel:
    """Borzoi model for regulatory activity prediction.

    Supports human and mouse models with optional FlashAttention.
    """

    def __init__(
        self,
        species: str = "human",
        replicate: str = "0",
        use_flash_attn: bool = True,
    ):
        """Initialize Borzoi model wrapper.

        Args:
            species: Species to predict for ('human' or 'mouse')
            replicate: Replicate to use (e.g., '0', '1', '2', '3')
            use_flash_attn: Whether to use FlashAttention
        """
        if species == "mouse" and use_flash_attn:
            raise ValueError(
                "FlashAttention (flashzoi) is not available for mouse models. Please set use_flash_attn=False."
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

    def load(self, device: str, verbose: bool = False) -> None:
        """Load Borzoi model to device."""
        from borzoi_pytorch import Borzoi

        if verbose:
            logger.info(f"Loading Borzoi model: {self.model_name} on {device}")

        self.model = Borzoi.from_pretrained(self.model_name).to(device)
        self.device = device
        self._loaded = True

        if verbose:
            logger.info("Borzoi model loaded successfully")

    def unload(self, verbose: bool = False) -> None:
        """Move model to CPU and clear CUDA cache."""
        if hasattr(self, "model") and hasattr(self, "device"):
            if verbose:
                logger.info(f"Moving model from {self.device} to CPU")
            self.to_device("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _reload_to_device(self, model: Any, old_device: str, new_device: str) -> Any:  # noqa: ARG002 — required by device transition callback signature
        """Custom move function: reload the model onto the target GPU.

        Borzoi uses flash-attn/Triton kernels that cannot run on CPU, so
        CPU→GPU transitions require a full reload rather than a simple .to().
        """
        self.load(new_device)
        return self.model

    def to_device(self, device: str) -> dict[str, Any]:  # type: ignore[return]
        """Move model to a different device.

        GPU→GPU and GPU→CPU use standard .to(). CPU→GPU requires a full
        reload because flash-attn/Triton kernels need fresh GPU initialization.
        """
        from standalone_helpers import move_model_to_device

        if not self._loaded:
            raise RuntimeError("Cannot move unloaded model to device. Call load() first.")

        if self.device == device:
            return  # type: ignore[return-value]

        if self.device == "cpu" and device.startswith("cuda"):
            # CPU→GPU: reload needed for flash-attn/Triton
            self.model = move_model_to_device(
                self.model,
                self.device,
                device,
                custom_move_fn=self._reload_to_device,
            )
        else:
            # GPU→GPU or GPU→CPU: standard .to() works
            self.model = move_model_to_device(self.model, self.device, device)
            self.device = device

    def __call__(
        self,
        sequence: str,
        output_tracks: list[int],
        avg_output_tracks: bool = True,
        device: str = "cuda",
        verbose: bool = False,
    ) -> torch.Tensor:
        """Run Borzoi inference on a DNA sequence.

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
        with torch.amp.autocast(device), torch.inference_mode():
            output = self.model(onehot.unsqueeze(0), is_human=(self.species == "human"))

        # Process output - always return 2D (num_tracks, positions)
        if avg_output_tracks:
            prediction = output[0][output_tracks, :].mean(0, keepdim=True)
        else:
            prediction = output[0][output_tracks, :]

        return prediction


def _serialize_output(value: Any) -> Any:
    """Recursively serialize tensors and arrays to JSON-safe types."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: _serialize_output(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_output(v) for v in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


# ============================================================================
# Dispatch
# ============================================================================
_model: BorzoiModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        species = input_dict.get("species", "human")
        replicate = input_dict.get("replicate", "0")
        use_flash_attn = input_dict.get("use_flash_attn", True)

        # Auto-detect flash-attn availability
        if use_flash_attn:
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                logger.warning("flash-attn not installed, falling back to use_flash_attn=False (standard borzoi model)")
                use_flash_attn = False

        _model = BorzoiModel(
            species=species,
            replicate=replicate,
            use_flash_attn=use_flash_attn,
        )

    operation = input_dict.get("operation", "predict")
    if operation == "predict":
        prediction = _model(
            sequence=input_dict["sequence"],
            output_tracks=input_dict["output_tracks"],
            avg_output_tracks=input_dict.get("avg_output_tracks", True),
            device=input_dict.get("device", "cuda"),
            verbose=input_dict.get("verbose", False),
        )
        return {
            "prediction": prediction,
            "applied_species": _model.species,
            "applied_replicate": _model.replicate,
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
        json.dump(_serialize_output(result), f)
