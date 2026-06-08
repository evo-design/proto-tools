"""Local ESM C (Cambrian) inference implementation.

Embedding-focused masked language model. Ships in the same ``esm`` package as
ESM3 and shares the ``biohub_esm`` env.
"""

import json
import logging
import sys
from typing import Any, Literal

import torch
from standalone_helpers import AMINO_ACIDS_LIST, get_logger, serialize_output
from tqdm import tqdm

logger = get_logger(__name__)

# Suppress esm library INFO logs
logging.getLogger("esm").setLevel(logging.ERROR)
logging.getLogger("esm.sdk").setLevel(logging.ERROR)

ESMC_MODEL_CHECKPOINTS = Literal["esmc_300m", "esmc_600m"]


class ESMCModel:
    """ESM C (Cambrian) model for protein sequence embeddings and logits.

    Embedding-focused masked language model from EvolutionaryScale. Both the
    300M (open-license) and 600M (non-commercial) variants are supported.
    """

    def __init__(self, model_checkpoint: ESMC_MODEL_CHECKPOINTS = "esmc_300m"):
        """Initialize ESM C model wrapper.

        Args:
            model_checkpoint: ESM C checkpoint name (e.g., ``"esmc_300m"``).
        """
        self._loaded = False
        self.model_checkpoint = model_checkpoint
        self.tokenizer: Any = None
        self.amino_acid_token_ids: Any = None
        self.device: str | None = None
        self.model: Any = None

    def __call__(
        self,
        sequences: list[str],
        batch_size: int = 1,
        device: str = "cuda",
        verbose: bool = False,
        return_logits: bool = False,
        repr_layer: int = -1,
    ) -> dict[str, torch.Tensor]:
        """Run ESM C inference on protein sequences.

        Args:
            sequences: Protein sequences.
            batch_size: Sequences per GPU forward pass. Larger batches are
                faster but use more memory.
            device: Device to run on.
            verbose: Whether to print progress.
            return_logits: Whether to return per-position amino-acid logits.
            repr_layer: Hidden-state layer index for embeddings. ``-1`` returns the
                final post-norm representation (model output ``embeddings``); other
                indices select from ``hidden_states``.

        Returns:
            Dictionary with mean_embeddings, attention_masks, and optionally logits.
        """
        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

        if not sequences:
            raise ValueError("esmc: __call__ requires at least one sequence")
        if any(len(seq) == 0 for seq in sequences):
            raise ValueError("esmc: __call__ does not support empty sequences")

        max_seq_len = max(len(seq) for seq in sequences)
        batches = [sequences[i : i + batch_size] for i in range(0, len(sequences), batch_size)]

        all_mean_embeddings: list[torch.Tensor] = []
        all_attention_masks: list[torch.Tensor] = []
        all_logits: list[torch.Tensor] | None = [] if return_logits else None

        for batch_sequences in tqdm(
            batches, desc="ESM C inference", unit="batch", total=len(batches), disable=not verbose
        ):
            # Tokenize the batch with padding so we can run a single forward pass
            batch_inputs = self.tokenizer(
                batch_sequences,
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            input_ids = batch_inputs["input_ids"].to(device)
            attention_mask_raw = batch_inputs["attention_mask"].to(device)

            with torch.inference_mode():
                outputs = self.model(sequence_tokens=input_ids)

                # -1 = post-norm last layer; other indices = pre-norm per-block.
                source = outputs.embeddings if repr_layer == -1 else outputs.hidden_states[repr_layer]
                embeddings = source[:, 1:-1, :]  # strip BOS/EOS
                attention_mask = attention_mask_raw[:, 1:-1]

                # Mean-pool over valid (non-pad) positions
                masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
                seq_lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                batch_mean_embeddings = masked_embeddings.sum(dim=1) / seq_lengths
                all_mean_embeddings.append(batch_mean_embeddings)

                # Pad attention mask to a uniform max_seq_len across batches
                pad_len = max_seq_len - embeddings.size(1)
                if pad_len > 0:
                    attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)
                all_attention_masks.append(attention_mask)

                # Per-position logits (BOS/EOS stripped, AAs only). The model always emits sequence_logits,
                # but the post-processing (slice/index/pad/cat) + GPU residency is skipped when not requested.
                if all_logits is not None:
                    logits = outputs.sequence_logits[:, 1:-1, :]
                    logits = logits[:, :, self.amino_acid_token_ids]
                    if pad_len > 0:
                        logits = torch.nn.functional.pad(logits, (0, 0, 0, pad_len), value=0.0)
                    all_logits.append(logits)

        return {
            "mean_embeddings": torch.cat(all_mean_embeddings, dim=0),
            "logits": torch.cat(all_logits, dim=0) if all_logits is not None else None,
            "attention_masks": torch.cat(all_attention_masks, dim=0),
        }

    # ============================================================================
    # Helper Functions
    # ============================================================================
    def load(self, device: str, verbose: bool = False) -> None:
        """Load ESM C model and tokenizer to device."""
        from esm.models.esmc import ESMC

        if verbose:
            logger.info(f"Loading ESM C model: {self.model_checkpoint} on {device}")

        try:
            self.model = ESMC.from_pretrained(self.model_checkpoint, device=torch.device(device)).eval()
        except OSError as e:
            raise RuntimeError(f"esmc: HF weight load from {self.model_checkpoint!r} failed: {e}") from e
        self.tokenizer = self.model.tokenizer

        self.amino_acid_token_ids = torch.tensor(
            [self.tokenizer.get_vocab()[aa] for aa in AMINO_ACIDS_LIST],
            device=device,
        )
        self.device = device
        self._loaded = True

        if verbose:
            logger.info("ESM C model loaded successfully")

    def to_device(self, device: str) -> None:
        """Move model to a different device."""
        from standalone_helpers import move_model_to_device

        if not self._loaded:
            raise ValueError("esmc: cannot move unloaded model to device — call load() first")

        if self.device != device:
            self.model = move_model_to_device(self.model, self.device, device)
            self.amino_acid_token_ids = self.amino_acid_token_ids.to(device)
            self.device = device

    def unload(self, verbose: bool = False) -> None:
        """Move model to CPU to free GPU memory."""
        if self._loaded and self.device != "cpu":
            if verbose:
                logger.info(f"Unloading {self.__class__.__name__} from GPU")

            self.model = self.model.to("cpu")
            self.amino_acid_token_ids = self.amino_acid_token_ids.to("cpu")
            self.device = "cpu"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ============================================================================
# Dispatch
# ============================================================================
_model: ESMCModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = ESMCModel(model_checkpoint=input_dict["model_checkpoint"])

    operation = input_dict["operation"]
    if operation in ("embeddings", "inference"):
        return _model(
            sequences=input_dict["sequences"],
            batch_size=input_dict["batch_size"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            return_logits=input_dict["return_logits"],
            repr_layer=input_dict["repr_layer"],
        )
    raise ValueError(f"esmc: unknown operation {operation!r}; valid: ['embeddings', 'inference']")


def to_device(device: str) -> dict[str, Any]:
    """Move model to specified device (called by DeviceManager)."""
    global _model
    if _model is not None and _model._loaded:
        _model.to_device(device)
        return {"success": True, "device": device}
    return {"success": True, "device": device, "note": "model not loaded yet"}


def get_memory_stats() -> dict[str, Any]:
    """Return PyTorch memory stats for the model's device (used by ToolPool)."""
    from standalone_helpers import get_pytorch_memory_stats

    device = _model.device if _model and hasattr(_model, "device") else 0
    return get_pytorch_memory_stats(device)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("esmc: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
