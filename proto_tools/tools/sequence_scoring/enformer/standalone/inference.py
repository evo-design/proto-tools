"""Enformer standalone inference implementation for venv execution."""

import json
import logging
import sys
from typing import Any, cast

import torch
from standalone_helpers import serialize_output, set_torch_seed

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
        self.model: Any = None
        self.device: str | None = None

    def load(self, device: str, verbose: bool = False) -> None:
        """Load Enformer model to device."""
        from enformer_pytorch import from_pretrained

        if verbose:
            logger.info(f"Loading Enformer model on {device}")

        repo = "EleutherAI/enformer-official-rough"
        try:
            self.model = from_pretrained(repo).to(device)
        except OSError as e:
            raise RuntimeError(f"enformer: HF weight load from {repo!r} failed: {e}") from e
        self.device = device
        self._loaded = True

        if verbose:
            logger.info("Enformer model loaded successfully")

    def unload(self, verbose: bool = False) -> None:
        """Move model to CPU and clear CUDA cache."""
        if self._loaded:
            if verbose:
                logger.info(f"Moving model from {self.device} to CPU")
            self.to_device("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to_device(self, device: str) -> None:
        """Move model to a different device."""
        from standalone_helpers import move_model_to_device

        if not self._loaded:
            raise ValueError("enformer: cannot move unloaded model to device — call load() first")

        if self.device != device:
            self.model = move_model_to_device(self.model, self.device, device)
            self.device = device

    def _predict_sequences(
        self,
        sequences: list[str],
        output_tracks: list[int],
        species: str,
        device: str = "cuda",
        verbose: bool = False,
    ) -> torch.Tensor:
        """Run Enformer inference on a batch of DNA sequences.

        Args:
            sequences: DNA sequences to score. Each sequence must be
                ENFORMER_CONTEXT=196,608 bp.
            output_tracks: Track indices to extract from the selected species
                output head.
            species: Species output head to use, either ``"human"`` or
                ``"mouse"``.
            device: Device to run inference on.
            verbose: Whether to log status messages.

        Returns:
            Prediction tensor with shape ``[batch, ENFORMER_OUTPUT, num_tracks]``.
        """
        # Lazy load on first call or device change
        if not self._loaded:
            self.load(device, verbose=verbose)
        elif self.device != device:
            self.to_device(device)

        # Prepare input (Enformer handles 'N' as index 4)
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        input_ids = torch.tensor([[mapping.get(char, 4) for char in sequence] for sequence in sequences]).to(device)

        # Run prediction
        with torch.inference_mode():
            output = self.model(input_ids)

        # Extract prediction for specified tracks and species
        return cast(torch.Tensor, output[species][:, :, output_tracks])


# ============================================================================
# Dispatch
# ============================================================================
_model: EnformerModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    set_torch_seed(input_dict["seed"])
    if _model is None:
        _model = EnformerModel()

    operation = input_dict["operation"]
    if operation == "predict":
        sequences = input_dict["sequences"]
        batch_size = max(1, int(input_dict.get("batch_size", len(sequences))))
        predictions = []
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            predictions.append(
                _model._predict_sequences(
                    sequences=chunk,
                    output_tracks=input_dict["output_tracks"],
                    species=input_dict["species"],
                    device=input_dict["device"],
                    verbose=input_dict["verbose"],
                )
                .detach()
                .cpu()
            )
        return {
            "predictions": torch.cat(predictions, dim=0),
            "applied_species": input_dict["species"],
        }
    raise ValueError(f"enformer: unknown operation {operation!r}; valid: ['predict']")


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
        raise ValueError("enformer: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
