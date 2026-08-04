"""Local ESM C (Cambrian) inference implementation.

Embedding-focused masked language model, plus sparse-autoencoder feature
extraction over its activations. Ships in the same ``esm`` package as ESM3 and
shares the ``biohub_esm`` env.

``ESMCModel`` loads the ``esm``-package weights for embeddings. ``ESMCSAEModel``
loads the Transformers-format backbone, which is what the published SAEs are
defined against. Each is constructed only when its operation is first requested.
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

ESMC_MODEL_CHECKPOINTS = Literal["esmc_300m", "esmc_600m", "esmc_6b"]


class ESMCModel:
    """ESM C (Cambrian) model for protein sequence embeddings and logits.

    Embedding-focused masked language model from Biohub. The 300M, 600M, and 6B
    variants are all supported.
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


# HuggingFace repos holding the Transformers-format backbone weights. These are
# distinct from the `esm`-package repos ``ESMCModel`` above loads.
BACKBONE_REPOS = {
    "esmc_300m": "biohub/ESMC-300M",
    "esmc_600m": "biohub/ESMC-600M",
    "esmc_6b": "biohub/ESMC-6B",
}


class ESMCSAEModel:
    """ESM C backbone with sparse autoencoders attached to selected layers."""

    def __init__(
        self,
        model_checkpoint: str = "esmc_300m",
        sae_repo: str = "",
        layers: tuple[int, ...] = (),
        backbone: str = "transformers",
    ):
        """Initialize the wrapper.

        Args:
            model_checkpoint: ESM C backbone key, e.g. ``"esmc_300m"``.
            sae_repo: HuggingFace repo id holding the SAE weights.
            layers: Backbone layer indices to attach SAEs to.
            backbone: ``"transformers"`` for the published path, or ``"esm"`` to read
                activations from the esmc toolkit's weights.
        """
        self._loaded = False
        self.model_checkpoint = model_checkpoint
        self.sae_repo = sae_repo
        self.layers = tuple(layers)
        self.backbone = backbone
        self.sae: Any = None
        self.device: str | None = None
        self.model: Any = None
        self.tokenizer: Any = None

    def __call__(
        self,
        sequences: list[str],
        batch_size: int = 1,
        device: str = "cuda",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Extract SAE features for each sequence.

        Args:
            sequences: Protein sequences.
            batch_size: Sequences per forward pass.
            device: Device to run on.
            verbose: Whether to show progress.

        Returns:
            Dictionary with a ``features`` list holding one entry per input
            sequence, keyed by layer index, each with ``indices`` and
            ``magnitudes`` arrays of shape (residues, k).
        """
        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

        if not sequences:
            raise ValueError("esmc_sae: __call__ requires at least one sequence")
        if any(len(seq) == 0 for seq in sequences):
            raise ValueError("esmc_sae: __call__ does not support empty sequences")

        batches = [sequences[i : i + batch_size] for i in range(0, len(sequences), batch_size)]
        features: list[dict[str, dict[str, list[list[float]]]]] = []

        if self.backbone == "esm":
            for batch in tqdm(
                batches, desc="ESM C SAE inference", unit="batch", total=len(batches), disable=not verbose
            ):
                features.extend(self._esm_batch(batch))
            return {"features": features}

        for batch in tqdm(batches, desc="ESM C SAE inference", unit="batch", total=len(batches), disable=not verbose):
            inputs = self.tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(**inputs)

            # The SAE stack emits one row per *unpadded* token, concatenated
            # across the batch, so rows must be split by each sequence's real
            # token count rather than reshaped to (batch, padded_len, codebook).
            token_counts = inputs["attention_mask"].sum(dim=1).tolist()
            features.extend(self._split_batch(outputs["sae_outputs"], token_counts))

        return {"features": features}

    def _esm_batch(self, batch: list[str]) -> list[dict[str, dict[str, list[list[float]]]]]:
        """Run one batch through the esm-package backbone and apply the SAEs directly.

        The esm package exposes ``hidden_states`` as (layer, batch, seq, d) with padding
        preserved, and its stack omits the embedding output that Transformers keeps at
        index 0. A Transformers ``layer{N}`` SAE therefore reads ``hidden_states[N - 1]``;
        ``test_esm_backbone_layer_offset`` pins that offset.

        Args:
            batch: Protein sequences.

        Returns:
            One dict per sequence, keyed by layer index as a string.
        """
        input_ids = self.model._tokenize(batch)
        token_counts = (input_ids != self.model.tokenizer.pad_token_id).sum(dim=1).tolist()

        with torch.inference_mode():
            hidden_states = self.model(input_ids).hidden_states

        per_sequence: list[dict[str, dict[str, list[list[float]]]]] = [{} for _ in batch]
        for layer in self.layers:
            layer_module = self.sae.layers[str(layer)]
            source = hidden_states[layer - 1].to(dtype=next(layer_module.parameters()).dtype)
            with torch.inference_mode():
                magnitudes = layer_module(source).feature_magnitudes

            for seq_idx, count in enumerate(token_counts):
                # Strip the start and end tokens to align with the input residues.
                rows = magnitudes[seq_idx, 1 : count - 1]
                sorted_magnitudes, indices = torch.sort(rows, dim=-1, descending=True)
                active = (rows != 0).sum(dim=-1)
                keep = int(active.max().item()) if active.numel() else 0
                per_sequence[seq_idx][str(layer)] = {
                    "indices": serialize_output(indices[:, :keep].to(torch.int64)),
                    "magnitudes": serialize_output(sorted_magnitudes[:, :keep].to(torch.float32)),
                }
        return per_sequence

    def _split_batch(
        self, sae_outputs: dict[str, torch.Tensor], token_counts: list[int]
    ) -> list[dict[str, dict[str, list[list[float]]]]]:
        """Split concatenated SAE rows into per-sequence top-k features.

        Args:
            sae_outputs: Mapping of ``layer{N}`` to a sparse (total_tokens, codebook)
                tensor covering every unpadded token in the batch.
            token_counts: Real token count per sequence, including start/end tokens.

        Returns:
            One dict per sequence, keyed by layer index as a string.
        """
        per_sequence: list[dict[str, dict[str, list[list[float]]]]] = [{} for _ in token_counts]

        for layer in self.layers:
            dense = sae_outputs[f"layer{layer}"]
            if dense.is_sparse:
                dense = dense.to_dense()

            offset = 0
            for seq_idx, count in enumerate(token_counts):
                # Strip the start and end tokens to align with the input residues.
                rows = dense[offset + 1 : offset + count - 1]
                offset += count

                magnitudes, indices = torch.sort(rows, dim=-1, descending=True)
                active = (rows != 0).sum(dim=-1)
                keep = int(active.max().item()) if active.numel() else 0

                per_sequence[seq_idx][str(layer)] = {
                    "indices": serialize_output(indices[:, :keep].to(torch.int64)),
                    "magnitudes": serialize_output(magnitudes[:, :keep].to(torch.float32)),
                }

        return per_sequence

    # ============================================================================
    # Helper Functions
    # ============================================================================
    def load(self, device: str, verbose: bool = False) -> None:
        """Load the backbone and SAE layers onto the device."""
        from transformers.models.esmc.modeling_esmc_sae import ESMCSAEModel as HFSAEModel

        logger.update_status(f"Loading {self.model_checkpoint}")
        if self.backbone == "esm":
            from esm.models.esmc import ESMC

            if verbose:
                logger.info(f"Loading ESM C backbone {self.model_checkpoint} (esm package) on {device}")
            self.model = ESMC.from_pretrained(self.model_checkpoint, device=torch.device(device)).eval()
            self.tokenizer = self.model.tokenizer
        else:
            from transformers import AutoModel, AutoTokenizer

            repo = BACKBONE_REPOS[self.model_checkpoint]
            if verbose:
                logger.info(f"Loading ESM C backbone {repo} on {device}")
            self.model = AutoModel.from_pretrained(repo, device_map=device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(repo)

        # Fetch only the requested layers; the all-layer SAE repos hold one file
        # per backbone layer and the 6B collection is tens of gigabytes.
        allow_patterns = ["config.json"] + [f"layer_{layer}.safetensors" for layer in self.layers]
        logger.update_status(f"Loading SAE layers {list(self.layers)}")
        device_for_sae = self.model.device if hasattr(self.model, "device") else device
        sae = HFSAEModel.from_pretrained(self.sae_repo, allow_patterns=allow_patterns, device=device_for_sae)
        sae.initialize_layers(list(self.layers))
        self.sae = sae
        # The esm path applies the SAE modules itself; only Transformers hooks them in.
        if self.backbone != "esm":
            self.model.add_sae_models([sae.layers[str(layer)] for layer in self.layers])

        self.device = device
        self._loaded = True

        if verbose:
            logger.info(f"ESM C SAE model loaded ({self.sae_repo}, layers {list(self.layers)})")

    def to_device(self, device: str) -> None:
        """Move the model to a different device."""
        from standalone_helpers import move_model_to_device

        if not self._loaded:
            raise ValueError("esmc_sae: cannot move unloaded model to device — call load() first")

        if self.device != device:
            self.model = move_model_to_device(self.model, self.device, device)
            self.device = device

    def unload(self, verbose: bool = False) -> None:
        """Move the model to CPU to free GPU memory."""
        if self._loaded and self.device != "cpu":
            if verbose:
                logger.info(f"Unloading {self.__class__.__name__} from GPU")

            self.model = self.model.to("cpu")
            self.device = "cpu"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ============================================================================
# Dispatch
# ============================================================================
_model: ESMCModel | None = None
_sae_model: ESMCSAEModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model, _sae_model

    operation = input_dict["operation"]

    if operation in ("embeddings", "inference"):
        if _model is None:
            _model = ESMCModel(model_checkpoint=input_dict["model_checkpoint"])
        return _model(
            sequences=input_dict["sequences"],
            batch_size=input_dict["batch_size"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            return_logits=input_dict["return_logits"],
            repr_layer=input_dict["repr_layer"],
        )

    if operation == "sae_features":
        layers = tuple(input_dict["layers"])
        backbone = input_dict.get("backbone", "transformers")
        # The SAE stack is bound to one repo/layer set/backbone, so a request for a
        # different combination needs a fresh model rather than a reused one.
        if (
            _sae_model is None
            or _sae_model.sae_repo != input_dict["sae_repo"]
            or _sae_model.layers != layers
            or _sae_model.backbone != backbone
        ):
            _sae_model = ESMCSAEModel(
                model_checkpoint=input_dict["model_checkpoint"],
                sae_repo=input_dict["sae_repo"],
                layers=layers,
                backbone=backbone,
            )
        return _sae_model(
            sequences=input_dict["sequences"],
            batch_size=input_dict["batch_size"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
        )

    raise ValueError(f"esmc: unknown operation {operation!r}; valid: ['embeddings', 'inference', 'sae_features']")


def to_device(device: str) -> dict[str, Any]:
    """Move whichever models are loaded to the specified device (called by DeviceManager)."""
    moved = [m for m in (_model, _sae_model) if m is not None and m._loaded]
    for model in moved:
        model.to_device(device)
    if moved:
        return {"success": True, "device": device}
    return {"success": True, "device": device, "note": "model not loaded yet"}


def get_memory_stats() -> dict[str, Any]:
    """Return PyTorch memory stats for the loaded model's device (used by ToolPool)."""
    from standalone_helpers import get_pytorch_memory_stats

    loaded = next((m for m in (_model, _sae_model) if m is not None and hasattr(m, "device")), None)
    device = loaded.device if loaded is not None else 0
    return get_pytorch_memory_stats(device)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("esmc: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
