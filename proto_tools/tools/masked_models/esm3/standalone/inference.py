"""Local ESM3 inference implementation."""

from __future__ import annotations

import json
import logging
import math
import sys
from logging import getLogger
from typing import Any, Literal

import torch
from tqdm import tqdm

logger = getLogger(__name__)

# Suppress esm library INFO logs
logging.getLogger("esm").setLevel(logging.ERROR)
logging.getLogger("esm.sdk").setLevel(logging.ERROR)

AMINO_ACIDS_LIST: list[str] = list("ACDEFGHIKLMNPQRSTVWY")
ESM3_MODEL_CHECKPOINTS = Literal["esm3_sm_open_v1",]


class ESM3Model:
    """ESM3 model for protein sequence embeddings, logits, and structure prediction.

    Multi-modal protein language model from EvolutionaryScale.
    """

    def __init__(self, model_checkpoint: ESM3_MODEL_CHECKPOINTS = "esm3_sm_open_v1"):
        """Initialize ESM3 model wrapper.

        Args:
            model_checkpoint: ESM3 checkpoint name (e.g., "esm3_sm_open_v1")
        """
        self._loaded = False
        self.model_checkpoint = model_checkpoint
        self.tokenizer = None
        self.amino_acid_token_ids = None
        self.device = None
        self.model = None

    def __call__(
        self,
        sequences: list[str],
        batch_size: int = 128,
        device: str = "cuda",
        verbose: bool = False,
        return_logits: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run ESM3 inference on protein sequences.

        Args:
            sequences: Protein sequences
            batch_size: Sequences per GPU forward pass. Larger batches are
                faster but use more memory.
            device: Device to run on
            verbose: Whether to print progress
            return_logits: Whether to return logits

        Returns:
            Dictionary with mean_embeddings, attention_masks, and optionally logits
        """
        # Lazy load on first call or device change
        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

        # Validate input sequences
        if not sequences:
            raise ValueError("ESM3Model.__call__ requires at least one sequence.")
        if any(len(seq) == 0 for seq in sequences):
            raise ValueError("ESM3Model.__call__ does not support empty sequences.")

        # Get the max sequence length
        max_seq_len = max(len(seq) for seq in sequences)

        # Split the sequences into batches
        batches = [sequences[i : i + batch_size] for i in range(0, len(sequences), batch_size)]

        all_mean_embeddings = []
        all_logits = []
        all_attention_masks = []

        # For each batch
        for batch_sequences in tqdm(batches, desc="ESM3 inference", unit="batch", total=len(batches)):
            # Tokenize the batch
            batch_inputs = self.tokenizer(  # type: ignore[misc]
                batch_sequences,
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            # Move the inputs to the correct device
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

            # Forward pass
            with torch.inference_mode():
                batch_outputs = self.model(  # type: ignore[misc]
                    sequence_tokens=batch_inputs["input_ids"],
                )

                # Append the embeddings
                embeddings = batch_outputs.embeddings[:, 1:-1, :]  # Remove special tokens
                attention_mask = batch_inputs["attention_mask"][:, 1:-1]

                # Average over embeddings
                masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
                seq_lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                batch_mean_embeddings = masked_embeddings.sum(dim=1) / seq_lengths
                all_mean_embeddings.append(batch_mean_embeddings)

                # Extract logits
                logits = batch_outputs.sequence_logits[:, 1:-1, :]  # Remove special tokens
                # Only keep the logits corresponding to the amino acids
                logits = logits[:, :, self.amino_acid_token_ids]

                # Determine padding length
                additional_padding_len = max_seq_len - embeddings.size(1)

                if additional_padding_len > 0:
                    # Pad attention_mask
                    attention_mask = torch.nn.functional.pad(attention_mask, (0, additional_padding_len), value=0)
                    # Pad logits
                    logits = torch.nn.functional.pad(logits, (0, 0, 0, additional_padding_len), value=0.0)

                # Save attention mask and logits
                all_attention_masks.append(attention_mask)
                all_logits.append(logits)

        # Concatenate along the batch dimension
        all_mean_embeddings = torch.cat(all_mean_embeddings, dim=0)
        all_logits = torch.cat(all_logits, dim=0)
        all_attention_masks = torch.cat(all_attention_masks, dim=0)

        return {
            "mean_embeddings": all_mean_embeddings,
            "logits": all_logits if return_logits else None,
            "attention_masks": all_attention_masks,
        }

    def predict_structure(
        self,
        sequences: list[str],
        batch_size: int = 40,
        device: str = "cuda",
        verbose: bool = False,
    ) -> list[dict[str, Any]]:
        """Predict 3D structures for protein sequences.

        Args:
            sequences: Protein sequences
            batch_size: Batch size for structure prediction
            device: Device to run on

            verbose: Whether to enable verbose logging output.

        Returns:
            List of structure dictionaries
        """
        # Lazy import ESM3 dependencies
        from esm.sdk.api import ESMProtein, GenerationConfig

        # Lazy load on first call or device change
        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

        # Split the sequences into batches
        max_batch_size = min(batch_size, len(sequences))
        batches = [sequences[i : i + max_batch_size] for i in range(0, len(sequences), max_batch_size)]

        all_structures = []  # type: ignore[var-annotated]

        # For each batch
        for batch_sequences in tqdm(
            batches, desc="Predicting structures with ESM3", unit="sequence batch", total=len(batches)
        ):
            # Create protein and config objects
            esm3_proteins = [ESMProtein(sequence=seq) for seq in batch_sequences]
            structure_configs = [GenerationConfig(track="structure")] * len(esm3_proteins)

            # Generate the structures
            structures = self.model.batch_generate(  # type: ignore[attr-defined]
                inputs=esm3_proteins,
                configs=structure_configs,
            )

            # Unpack predicted structures
            all_structures.extend(
                {
                    "sequence": struct.sequence,
                    "pdb_string": struct.to_pdb_string(),
                    "avg_plddt": struct.plddt.mean().item(),
                    "ptm": struct.ptm.item(),
                }
                for struct in structures
            )

        return all_structures

    def sample(
        self,
        sequences: list[str],
        temperature: float,
        batch_size: int = 1,
        device: str = "cuda",
        verbose: bool = False,
        return_logits: bool = False,
    ) -> dict[str, Any]:
        """Sample amino acids at masked positions ('_') in protein sequences.

        Receives sequences with '_' at positions to be sampled, injects
        ``[MASK]`` tokens at those positions, runs a forward pass, and samples
        replacement amino acids from the model's predictions.

        Args:
            sequences: Protein sequences with '_' at positions to sample.
            temperature: Sampling temperature for amino acid selection.
            batch_size: Sequences per GPU forward pass.
            device: Device to run on.
            verbose: Whether to print progress.
            return_logits: Whether to return per-position AA logits.

        Returns:
            Dictionary with "sequences" and optionally "logits".
        """
        device_obj = torch.device(device)

        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

        # Record mask positions, replace '_' with the model's mask token
        mask_token = self.tokenizer.mask_token  # type: ignore[attr-defined]
        mask_positions: list[list[int]] = []
        tokenizer_sequences: list[str] = []
        for seq in sequences:
            positions = [i for i, c in enumerate(seq) if c == "_"]
            mask_positions.append(positions)
            tokenizer_sequences.append(seq.replace("_", mask_token))

        # Batch the forward passes
        batch_ranges = [(i, min(i + batch_size, len(sequences))) for i in range(0, len(sequences), batch_size)]

        all_sampled: list[str] = []
        all_logits: list[torch.Tensor] = []

        for start, end in tqdm(
            batch_ranges,
            desc="ESM3 sampling",
            unit="batch",
            disable=not verbose,
        ):
            batch_tok_seqs = tokenizer_sequences[start:end]
            batch_masks = mask_positions[start:end]
            batch_originals = sequences[start:end]

            # Tokenize (mask tokens are handled natively by the tokenizer)
            batch_inputs = self.tokenizer(  # type: ignore[misc]
                batch_tok_seqs,
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            input_ids = batch_inputs["input_ids"].to(device_obj)

            # Forward pass
            with torch.inference_mode():
                outputs = self.model(sequence_tokens=input_ids)  # type: ignore[misc]
                # AA logits: remove BOS/EOS, keep only standard amino acids
                aa_logits = outputs.sequence_logits[:, 1:-1, :][:, :, self.amino_acid_token_ids]

            # Sample at mask positions for each sequence
            for seq_idx, (orig_seq, positions) in enumerate(zip(batch_originals, batch_masks, strict=False)):
                if not positions:
                    all_sampled.append(orig_seq)
                else:
                    pos_t = torch.tensor(positions, device=device_obj)
                    pos_logits = aa_logits[seq_idx, pos_t]
                    scaled = pos_logits / max(temperature, 1e-8)
                    probs = torch.softmax(scaled, dim=-1)
                    sampled_idx = torch.multinomial(probs, 1).squeeze(-1)

                    chars = list(orig_seq)
                    for pos, aa_idx in zip(positions, sampled_idx.tolist(), strict=False):
                        chars[pos] = AMINO_ACIDS_LIST[aa_idx]
                    all_sampled.append("".join(chars))

            if return_logits:
                for seq_idx, orig_seq in enumerate(batch_originals):
                    all_logits.append(aa_logits[seq_idx, : len(orig_seq)])

        return {
            "sequences": all_sampled,
            "logits": all_logits if return_logits else None,
        }

    def score(
        self,
        sequences: list[str],
        batch_size: int = 32,
        device: str = "cuda",
        verbose: bool = False,
        return_logits: bool = True,
    ) -> dict[str, list[Any]]:
        """Score protein sequences using ESM3 with MLM pseudo-perplexity.

        Computes pseudo-perplexity by masking each position individually and
        computing P(x_i | x_{-i}). Uses batching for efficiency. Requires L forward
        passes per sequence of length L (batched).

        Ambiguous amino acids (X, B, Z, etc.) are excluded from the perplexity
        calculation using the industry-standard exclusion strategy. Only positions
        with standard amino acids contribute to log-likelihood and perplexity.

        Args:
            sequences: List of protein sequences to score
            batch_size: Masked variants per forward pass. Larger batches are
                faster but use more memory.
            device: Device to run on
            verbose: Whether to print progress
            return_logits: Whether to include logits in the output

        Returns:
            Dictionary with:
                - "logits": List of logits tensors (seq_len, vocab_size=20) if return_logits=True
                - "metrics": List of metric dicts with log_likelihood, avg_log_likelihood, perplexity
                - "vocab": List of 20 standard amino acid characters
        """
        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

        # Validate sequences for scoring
        if not sequences:
            raise ValueError("ESM3Model.score requires at least one non-empty sequence.")

        all_logits = []
        all_metrics = []

        for seq in tqdm(sequences, desc="ESM3 scoring", unit="sequence", total=len(sequences)):
            if len(seq) == 0:
                raise ValueError("ESM3Model.score does not support empty sequences.")

            # Compute MLM pseudo-perplexity and collect logits from masked positions
            # Returns (log_prob, logits, valid_count) where valid_count excludes ambiguous AAs
            log_prob, logits, valid_count = self._compute_mlm_score(seq, batch_size)

            # Average over valid positions only (exclusion strategy for ambiguous AAs)
            if valid_count == 0:
                raise ValueError(f"No valid characters found for ESM3 scoring in sequence: {seq}")
            avg_ll = log_prob / valid_count

            all_logits.append(logits)
            all_metrics.append(
                {
                    "log_likelihood": log_prob,
                    "avg_log_likelihood": avg_ll,
                    "perplexity": math.exp(-avg_ll),
                }
            )

        return {
            "logits": all_logits if return_logits else None,  # type: ignore[dict-item]
            "metrics": all_metrics,
            "vocab": AMINO_ACIDS_LIST,  # Return AA-only vocab (20 tokens)
        }

    def _compute_mlm_score(self, seq: str, batch_size: int) -> tuple[float, torch.Tensor, int]:
        """Compute MLM pseudo-perplexity by masking each position.

        This method performs L forward passes (batched) for a sequence of length L,
        collecting the logits P(aa | context without position i) at each masked position.

        Ambiguous amino acids (X, B, Z, etc.) are excluded from the log-likelihood
        calculation but their logits are still returned (with values only for
        standard amino acids).

        Args:
            seq: Protein sequence
            batch_size: Number of masked variants per forward pass

        Returns:
            Tuple of (total_log_probability, logits_tensor, valid_count) where:
                - total_log_probability: Sum of log probs for standard AA positions only
                - logits_tensor: Shape (seq_len, vocab_size=20) with AA-only logits
                - valid_count: Number of standard AA positions (excludes ambiguous)
        """
        # Tokenize once
        encoded = self.tokenizer(seq, add_special_tokens=True, return_tensors="pt")  # type: ignore[misc]
        original_ids = encoded["input_ids"].to(self.device)

        # Create all masked variants (L variants for sequence of length L)
        masked_ids = original_ids.repeat(len(seq), 1)
        for pos in range(len(seq)):
            masked_ids[pos, pos + 1] = self.tokenizer.mask_token_id  # type: ignore[attr-defined]  # +1 for BOS token

        # Get true token IDs directly from tokenized input
        true_token_ids = original_ids[0, 1 : 1 + len(seq)]  # Token positions: [BOS] + seq + [EOS]

        # Create mask for valid (standard AA) positions - exclusion strategy
        valid_mask = torch.tensor([aa in AMINO_ACIDS_LIST for aa in seq], device=self.device)
        valid_count = int(valid_mask.sum().item())

        # Initialize outputs
        total_log_prob = 0.0
        all_position_logits = []

        # Process in batches
        for batch_start in range(0, len(seq), batch_size):
            batch_end = min(batch_start + batch_size, len(seq))
            batch_ids = masked_ids[batch_start:batch_end]

            with torch.inference_mode():
                outputs = self.model(sequence_tokens=batch_ids)  # type: ignore[misc]

            # Extract logits at masked positions
            batch_indices = torch.arange(batch_end - batch_start, device=self.device)
            positions = torch.arange(batch_start, batch_end, device=self.device) + 1  # +1 for BOS
            position_logits = outputs.sequence_logits[batch_indices, positions]

            # Filter logits to AA-only (20 standard amino acids)
            aa_position_logits = position_logits[:, self.amino_acid_token_ids]
            all_position_logits.append(aa_position_logits)

            # Compute log probs for scoring (only over standard AAs)
            log_probs = torch.log_softmax(position_logits, dim=-1)

            # Get log probs for the observed tokens at the masked positions
            # Only include positions with standard AAs in the total (exclusion strategy)
            batch_true_ids = true_token_ids[batch_start:batch_end]
            true_log_probs = log_probs[batch_indices, batch_true_ids]

            # Apply valid mask to exclude ambiguous AAs from log-likelihood
            batch_valid_mask = valid_mask[batch_start:batch_end].float()
            total_log_prob += (true_log_probs * batch_valid_mask).sum().item()

        # Concatenate logits from all batches: (seq_len, vocab_size=20)
        logits = torch.cat(all_position_logits, dim=0)
        return total_log_prob, logits, valid_count

    # ============================================================================
    # Helper Functions
    # ============================================================================
    def load(self, device: str, verbose: bool = False) -> None:
        """Load ESM3 model and tokenizer to device."""
        # Lazy import ESM3 dependencies
        from esm.models.esm3 import ESM3
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

        if verbose:
            logger.info(f"Loading ESM3 model: {self.model_checkpoint} on {device}")

        # Load model and tokenizer
        self.model = ESM3.from_pretrained(self.model_checkpoint, device=torch.device(device)).eval()
        self.tokenizer = EsmSequenceTokenizer()

        self.amino_acid_token_ids = torch.tensor(
            [self.tokenizer.get_vocab()[aa] for aa in AMINO_ACIDS_LIST],  # type: ignore[attr-defined]
            device=device,
        )
        self.device = device  # type: ignore[assignment]
        self._loaded = True

        if verbose:
            logger.info("ESM3 model loaded successfully")

    def to_device(self, device: str) -> dict[str, Any]:  # type: ignore[return]
        """Move model to a different device."""
        if not self._loaded:
            raise RuntimeError("Cannot move unloaded model to device. Call load() first.")

        if self.device != device:
            self.model = self.model.to(device)  # type: ignore[attr-defined]
            self.amino_acid_token_ids = self.amino_acid_token_ids.to(device)  # type: ignore[attr-defined]
            self.device = device  # type: ignore[assignment]

    def unload(self, verbose: bool = False) -> None:
        """Move model to CPU to free GPU memory."""
        if self._loaded and self.device != "cpu":
            if verbose:
                logger.info(f"Unloading {self.__class__.__name__} from GPU")

            self.model = self.model.to("cpu")  # type: ignore[attr-defined]
            self.amino_acid_token_ids = self.amino_acid_token_ids.to("cpu")  # type: ignore[attr-defined]
            self.device = "cpu"  # type: ignore[assignment]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _serialize_output(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: _serialize_output(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_output(v) for v in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "tolist"):
        # .tolist() handles CPU transfer internally for CUDA tensors,
        # avoiding an unnecessary intermediate CPU tensor allocation.
        return value.tolist()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value


# ============================================================================
# Dispatch
# ============================================================================
_model: ESM3Model | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = ESM3Model(
            model_checkpoint=input_dict.get("model_checkpoint", "esm3_sm_open_v1"),
        )

    operation = input_dict.get("operation", "embeddings")
    if operation in ("embeddings", "inference"):
        return _model(
            sequences=input_dict.get("sequences", []),
            batch_size=input_dict.get("batch_size", 128),
            device=input_dict.get("device", "cuda"),
            verbose=input_dict.get("verbose", False),
            return_logits=input_dict.get("return_logits", False),
        )
    if operation == "sample":
        return _model.sample(
            sequences=input_dict.get("sequences", []),
            temperature=input_dict.get("temperature", 1.0),
            batch_size=input_dict.get("batch_size", 1),
            device=input_dict.get("device", "cuda"),
            verbose=input_dict.get("verbose", False),
            return_logits=input_dict.get("return_logits", False),
        )
    if operation == "score":
        return _model.score(
            sequences=input_dict.get("sequences", []),
            batch_size=input_dict.get("batch_size", 32),
            device=input_dict.get("device", "cuda"),
            verbose=input_dict.get("verbose", False),
            return_logits=input_dict.get("return_logits", False),
        )
    if operation == "predict_structure":
        return _model.predict_structure(  # type: ignore[return-value]
            sequences=input_dict.get("sequences", []),
            batch_size=input_dict.get("batch_size", 128),
            device=input_dict.get("device", "cuda"),
            verbose=input_dict.get("verbose", False),
        )
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
