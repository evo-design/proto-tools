"""Standalone inference entry point for Splice Transformer RNA splicing prediction."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from standalone_helpers import move_model_to_device

logger = logging.getLogger(__name__)

_STANDALONE_DIR = Path(__file__).resolve().parent
if str(_STANDALONE_DIR) not in sys.path:
    # Needed for persistent-worker imports where script dir is not auto-added.
    sys.path.insert(0, str(_STANDALONE_DIR))


def _resolve_local_checkpoint_path() -> Path | None:
    """Resolve a local SpliceTransformer checkpoint path if available."""
    explicit_path = os.environ.get("SPLICE_TRANSFORMER_CHECKPOINT", "").strip()
    if explicit_path:
        checkpoint_path = Path(explicit_path).expanduser().resolve()
        if checkpoint_path.exists():
            return checkpoint_path
        raise FileNotFoundError(f"SPLICE_TRANSFORMER_CHECKPOINT is set but file does not exist: {checkpoint_path}")

    return None


class SpliceTransformerModel:
    """Manages Splice Transformer model loading and batch inference."""

    def __init__(self, context_length: int = 4000):
        """Initialize SpliceTransformerModel."""
        self.context_length = context_length
        self.device = None
        self.model = None
        self._loaded = False

    def __call__(
        self,
        target_seqs: list[str],
        left_contexts: list[str],
        right_contexts: list[str],
        device: str = "cuda",
        verbose: bool = False,
    ) -> np.ndarray:
        """Run SpliceTransformer inference on sequences with contexts.

        Args:
            target_seqs: Target sequences to make predictions on
            left_contexts: Left context sequences (must be context_length long)
            right_contexts: Right context sequences (must be context_length long)
            device: Device to run inference on
            verbose: Whether to print status messages

        Returns:
            Predictions of shape (batch, target_length, 18)
        """
        # Lazy load on first call or device change
        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

        assert len(target_seqs) == len(left_contexts) == len(right_contexts), (
            "Number of targets must be the same as the number of left and right contexts"
        )

        seqs_tokenized = []
        for target, left, right in zip(target_seqs, left_contexts, right_contexts, strict=False):
            assert len(left) == len(right) == self.context_length, (
                f"Length of left and right contexts must be {self.context_length}, got {len(left)} and {len(right)}"
            )
            seq = left + target + right
            seqs_tokenized.append(self._one_hot_encode(seq))
        seqs_tokenized = np.stack(seqs_tokenized)  # type: ignore[assignment]

        return self._calc_batched_sequence(seqs_tokenized)  # type: ignore[arg-type]  # (batch, target_length, 18)

    def _one_hot_encode(self, seq: str) -> Any:
        """Parse input RNA sequence into one-hot-encoding format."""
        IN_MAP = np.asarray([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        seq = seq.upper().replace("A", "1").replace("C", "2")
        seq = seq.replace("G", "3").replace("T", "4").replace("U", "4").replace("N", "0")
        seq = np.asarray(list(map(int, list(seq))))  # type: ignore[assignment]
        return IN_MAP[seq.astype("int8")]  # type: ignore[attr-defined]

    def _post_decorate(self, outputs: torch.Tensor) -> Any:
        outputs[:, :3, :] = torch.nn.functional.softmax(outputs[:, :3, :], dim=1)
        outputs[:, 3:, :] = torch.sigmoid(outputs[:, 3:, :])
        return outputs

    def _step(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run model forward pass.

        Args:
            inputs: Encoded sequence tensor of shape (batch, 4, length)

        Returns:
            Model output tensor
        """
        assert len(inputs.size()) == 3
        with torch.no_grad():
            out = self.model(inputs).cpu().detach()  # type: ignore[misc]
            return self._post_decorate(out)

    def _calc_single_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Calculate model output for a single sequence.

        Args:
            seq: One-hot encoded sequence array of shape (length, 4)

        Returns:
            Model output array
        """
        seq = torch.tensor(seq).to(self.device)
        seq = seq.unsqueeze(0).transpose(1, 2)  # type: ignore[attr-defined]
        res = self._step(seq.float())  # type: ignore[attr-defined]
        return res[0].transpose(0, 1).numpy()  # type: ignore[no-any-return]

    def _calc_batched_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Calculate model output for multiple sequences.

        Args:
            seq: One-hot encoded sequence array of shape (batch, length, 4)

        Returns:
            Model output array
        """
        seq = torch.tensor(seq).to(self.device)
        seq = seq.transpose(1, 2)  # 　(batch, length, 4) -> (batch, 4, length)
        res = self._step(seq.float())  # type: ignore[attr-defined]
        return res.transpose(1, 2).numpy()  # type: ignore[no-any-return]  # 　(batch, 4, length) -> (batch, length, 4)

    # ============================================================================
    # Model Loading & Device Management
    # ============================================================================
    def _fix_state_dict_keys(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Fix mismatched weight keys in checkpoint."""
        return {
            key.replace("attn.pos_emb.weights_", "attn.pos_emb.weights."): value for key, value in state_dict.items()
        }

    def load(self, device: str = "cuda", verbose: bool = False) -> None:  # noqa: ARG002 — required by tool interface
        """Load SpliceTransformer model to device."""
        logger.debug(f"Loading SpliceTransformer (context_length={self.context_length}) on {device}")

        from model import SpTransformer

        self.model = (
            SpTransformer(
                128,
                context_len=self.context_length,
                tissue_num=15,
                max_seq_len=8192,
                attn_depth=8,
                training=False,
            )
            .to(device)
            .eval()
        )

        # Prefer a local checkpoint if available, then fallback to HuggingFace Hub.
        local_model_path = _resolve_local_checkpoint_path()
        if local_model_path is not None:
            model_path = str(local_model_path)
            logger.debug(f"Using local SpliceTransformer checkpoint: {model_path}")
        else:
            logger.debug("Downloading SpliceTransformer checkpoint from HuggingFace Hub...")
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(
                repo_id="brianhie/SpTransformer",
                filename="SpTransformer_pytorch.ckpt",
            )

        # Load and fix state dict
        save_dict = torch.load(model_path, map_location=device)
        state_dict = self._fix_state_dict_keys(save_dict["state_dict"])
        self.model.load_state_dict(state_dict)  # type: ignore[attr-defined]

        self.device = device  # type: ignore[assignment]
        self._loaded = True

        logger.debug("SpliceTransformer model loaded successfully")

    def to_device(self, device: str) -> None:
        """Move model to a different device."""
        if not self._loaded:
            raise RuntimeError("Cannot move unloaded model to device. Call load() first.")

        if self.device != device:
            self.model = move_model_to_device(self.model, self.device, device)
            self.device = device  # type: ignore[assignment]

    def unload(self, verbose: bool = False) -> None:  # noqa: ARG002 — required by tool interface
        """Move model to CPU to free GPU memory."""
        if self._loaded and self.device != "cpu":
            logger.debug("Unloading SpliceTransformer from GPU")

            self.model = self.model.to("cpu")  # type: ignore[attr-defined]
            self.device = "cpu"  # type: ignore[assignment]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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
_model: SpliceTransformerModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = SpliceTransformerModel(
            context_length=input_dict.get("context_length", 4000),
        )

    operation = input_dict.get("operation", "predict")
    if operation == "predict":
        prediction = _model(
            target_seqs=input_dict["target_seqs"],
            left_contexts=input_dict["left_contexts"],
            right_contexts=input_dict["right_contexts"],
            device=input_dict.get("device", "cuda"),
            verbose=input_dict.get("verbose", False),
        )
        return {"prediction": prediction}
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
