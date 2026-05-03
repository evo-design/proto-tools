"""Local ProGen2 inference implementation.

Uses Hugging Face models from https://huggingface.co/hugohrban/
Based on the ProGen2-finetuning repository: https://github.com/hugohrban/ProGen2-finetuning
"""

import json
import sys
from logging import getLogger
from typing import Any, Literal

import torch
from standalone_helpers import move_model_to_device, serialize_output, set_torch_seed
from tqdm import tqdm

logger = getLogger(__name__)

HUGGINGFACE_REPO_PREFIX = "hugohrban"
# Tokenizer: 0=pad, 1=bos, 2=eos, 3='1', 4='2', 5-29=A..Z (no J). logits[t,j] -> PROGEN2_VOCAB[j].
PROGEN2_PAD_TOKEN = "<|pad|>"  # ID 0
PROGEN2_BOS_TOKEN = "<|bos|>"  # ID 1
PROGEN2_EOS_TOKEN = "<|eos|>"  # ID 2
PROGEN2_START_TOKEN = "1"  # ID 3
PROGEN2_END_TOKEN = "2"  # ID 4
PROGEN2_VOCAB: list[str] = [
    PROGEN2_PAD_TOKEN,
    PROGEN2_BOS_TOKEN,
    PROGEN2_EOS_TOKEN,
    PROGEN2_START_TOKEN,
    PROGEN2_END_TOKEN,
    *list("ABCDEFGHIKLMNOPQRSTUVWXYZ"),
]
PROGEN2_FIRST_AA_TOKEN = 5
PROGEN2_LAST_AA_TOKEN = 29
PROGEN2_SEQUENCE_CHARS = PROGEN2_VOCAB[PROGEN2_FIRST_AA_TOKEN : PROGEN2_LAST_AA_TOKEN + 1]  # Amino acid chars only

PROGEN2_MODEL_CHECKPOINTS = Literal[
    "progen2-small",  # 151M parameters
    "progen2-medium",  # 754M parameters
    "progen2-oas",  # 754M parameters, trained on OAS antibody sequences
    "progen2-large",  # 2B parameters
    "progen2-BFD90",  # 2B parameters, trained on BFD90
    "progen2-xlarge",  # 6B parameters
]


class ProGen2Model:
    """ProGen2 model wrapper for protein sequence generation.

    Handles model loading, device management, batched generation and scoring.
    """

    def __init__(
        self,
        model_checkpoint: PROGEN2_MODEL_CHECKPOINTS = "progen2-large",
        local_path: str | None = None,
    ):
        """Initialize ProGen2 model wrapper.

        Args:
            model_checkpoint: ProGen2 checkpoint name
            local_path: Optional local path to model weights
        """
        self._loaded = False
        self.model_checkpoint = model_checkpoint
        self.local_path = local_path
        self.tokenizer: Any = None
        self.device: str | None = None
        self.model: Any = None
        self.pad_token_id: int | None = None

    def load(self, device: str, verbose: bool = False) -> None:
        """Load ProGen2 model and tokenizer to device."""
        if verbose:
            logger.info(f"Loading ProGen2 model: {self.model_checkpoint} on {device}")

        from tokenizers import Tokenizer
        from transformers import AutoModelForCausalLM

        model_path = self.local_path or f"{HUGGINGFACE_REPO_PREFIX}/{self.model_checkpoint}"

        try:
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype="auto",
                )
                .to(device)
                .eval()
            )
        except OSError as e:
            raise RuntimeError(f"progen2: HF weight load from {model_path!r} failed: {e}") from e

        try:
            self.tokenizer = Tokenizer.from_pretrained(model_path)
        except OSError as e:
            raise RuntimeError(f"progen2: HF tokenizer load from {model_path!r} failed: {e}") from e
        self.device = device
        self.pad_token_id = self.tokenizer.token_to_id(PROGEN2_PAD_TOKEN)  # ID 0
        self._loaded = True

        if verbose:
            logger.info("ProGen2 model loaded successfully")

    def to_device(self, device: str) -> None:
        """Move model to a different device."""
        if not self._loaded:
            raise ValueError("progen2: cannot move unloaded model to device — call load() first")
        if self.device != device:
            self.model = move_model_to_device(self.model, self.device, device)
            self.device = device

    def unload(self, verbose: bool = False) -> None:
        """Unload model to free GPU memory."""
        if self._loaded and self.device != "cpu":
            if verbose:
                logger.info(f"Unloading {self.__class__.__name__} from GPU")
            self.model = self.model.to("cpu")
            self.device = "cpu"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _prepare_batch(
        self,
        sequences: list[str],
        pad_left: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Tokenize and pad sequences into a batch with attention mask."""
        if not sequences:
            raise ValueError("progen2: cannot prepare empty batch")
        assert self.pad_token_id is not None, "Model not loaded; call load() first"
        encodings = self.tokenizer.encode_batch(sequences)
        token_lists = [e.ids for e in encodings]
        lengths = [len(t) for t in token_lists]
        max_len = max(lengths)

        input_ids = torch.full((len(sequences), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)

        for i, (tokens, length) in enumerate(zip(token_lists, lengths, strict=False)):
            if pad_left:
                input_ids[i, max_len - length :] = torch.tensor(tokens, dtype=torch.long)
                attention_mask[i, max_len - length :] = 1
            else:
                input_ids[i, :length] = torch.tensor(tokens, dtype=torch.long)
                attention_mask[i, :length] = 1

        return input_ids.to(self.device), attention_mask.to(self.device), lengths

    def _truncate_at_terminals(self, sequence: str) -> str:
        """Truncate sequence at the first terminal token ('1' or '2') after position 0."""
        terminals = [PROGEN2_START_TOKEN, PROGEN2_END_TOKEN]
        # Start search from index 1 to avoid catching the start token if it exists at index 0.
        indices = [sequence.find(t, 1) for t in terminals]
        valid_indices = [i for i in indices if i != -1]
        if valid_indices:
            return sequence[: min(valid_indices) + 1]
        return sequence

    def sample(
        self,
        prompts: list[str],
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 0,
        max_length: int = 256,
        num_return_sequences: int = 1,
        truncate_at_stop: bool = True,
        strip_special_tokens: bool = True,
        prepend_prompt: bool = True,
        device: str = "cuda",
        verbose: bool = False,
        batch_size: int = 1,
        return_logits: bool = False,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Sample protein sequences using ProGen2 with batched generation.

        Args:
            prompts: List of prompt sequences
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling (0 to disable)
            max_length: Maximum total sequence length
            num_return_sequences: Sequences per prompt
            truncate_at_stop: Whether to stop at EOS tokens
            strip_special_tokens: Whether to remove start and stop tokens
            prepend_prompt: Whether to include prompt in output
            device: Device to run on
            verbose: Verbose logging
            batch_size: Number of sequences per GPU forward pass. Larger batches
                are faster but use more memory.
            return_logits: Whether to return per-position logits for generated tokens.
            seed: Random seed for reproducibility.

        Returns:
            Dictionary with:
                - "sequences": List of generated strings
                - "logits": List of logits tensors (if return_logits=True), else None
        """
        # Lazy load.
        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

        # Seed after load so each dispatch enters sampling with the same RNG state.
        set_torch_seed(seed)

        if not prompts:
            raise ValueError("progen2: cannot sample from empty prompt list")

        # Batch processing logic
        all_sequences: list[str] = []
        all_logits: list[torch.Tensor | None] = []
        batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]

        with torch.no_grad():
            for batch_prompts in tqdm(batches, desc="ProGen2 Sequence Generation", unit="batch", disable=not verbose):
                # Left-pad for generation
                input_ids, attention_mask, _ = self._prepare_batch(batch_prompts, pad_left=True)

                output = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=temperature,
                    max_length=max_length,
                    top_p=top_p,
                    top_k=top_k if top_k > 0 else None,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=return_logits,
                )
                tokens_batch = output.sequences

                # Stack scores into (batch * num_return, num_generated_tokens, vocab_size)
                # output.scores is a tuple of tensors, each of shape (batch * num_return, vocab_size)
                stacked_logits = torch.stack(output.scores, dim=1) if return_logits and output.scores else None

                # Decode: generate returns (batch * num_return_sequences, seq_len)
                for prompt_idx, prompt in enumerate(batch_prompts):
                    for seq_idx in range(num_return_sequences):
                        flat_idx = prompt_idx * num_return_sequences + seq_idx
                        token_ids = tokens_batch[flat_idx].tolist()

                        # Strip padding tokens
                        token_ids = [t for t in token_ids if t != self.pad_token_id]
                        seq = self.tokenizer.decode(token_ids)

                        if truncate_at_stop:
                            seq = self._truncate_at_terminals(seq)

                        # Be careful removing prompt if generation modified it or if special tokens overlap
                        if not prepend_prompt and seq.startswith(prompt):
                            seq = seq[len(prompt) :]

                        if strip_special_tokens:
                            seq = seq.lstrip(PROGEN2_START_TOKEN).rstrip(PROGEN2_END_TOKEN)

                        all_sequences.append(seq)

                        # Collect logits for this sequence
                        # Note: model outputs 32 logits but tokenizer has 30 tokens
                        if stacked_logits is not None:
                            all_logits.append(stacked_logits[flat_idx, :, : len(PROGEN2_VOCAB)].float().cpu())
                        else:
                            all_logits.append(None)

        return {
            "sequences": all_sequences,
            "logits": all_logits if return_logits else None,
        }

    def score(
        self,
        sequences: list[str],
        device: str = "cuda",
        verbose: bool = False,
        batch_size: int = 1,
        return_logits: bool = False,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Score protein sequences by computing logits and metrics via forward pass.

        Sequences are batched with padding and attention masking. Metrics are
        computed only over non-padded tokens, and optionally only over amino acid
        tokens (excluding special tokens like START, END, PAD).

        Args:
            sequences: List of protein sequences to score
            device: Device to run on
            verbose: Whether to print status messages
            batch_size: Number of sequences per GPU forward pass. Larger batches
                are faster but use more memory.
            return_logits: Whether to include logits in the output
            seed: Random seed. Scoring is deterministic given the model state,
                but we still seed RNGs/cudnn flags so consecutive calls in a
                persistent worker behave identically regardless of call order.

        Returns:
            Dictionary with:
                - "logits": List of logits tensors (seq_len, vocab_size) if return_logits=True
                - "metrics": List of metric dicts with log_likelihood, avg_log_likelihood, perplexity
                - "vocab": Vocabulary list
        """
        # Lazy load on first call or device change
        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

        set_torch_seed(seed)

        if not sequences:
            raise ValueError("progen2: cannot score empty sequence list")

        # Ensure sequences have start token
        normalized_seqs = [
            seq if seq.startswith(PROGEN2_START_TOKEN) else PROGEN2_START_TOKEN + seq for seq in sequences
        ]

        # Batch processing logic
        all_logits: list[torch.Tensor] = []
        all_metrics: list[dict[str, float]] = []
        batches = [normalized_seqs[i : i + batch_size] for i in range(0, len(normalized_seqs), batch_size)]

        # Run inference
        with torch.inference_mode():
            for batch_idx, batch_sequences in enumerate(
                tqdm(batches, desc="ProGen2 Sequence Scoring", unit="batch", total=len(batches))
            ):
                if verbose:
                    logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch_sequences)} sequences)")

                input_ids, _, lengths = self._prepare_batch(batch_sequences, pad_left=False)

                outputs = self.model(input_ids)
                logits = outputs.logits  # [batch, seq_len, vocab_size]

                # Shift for autoregressive scoring: predict position i from positions 0..i-1
                # logits[:, :-1] predicts tokens at positions 1, 2, ..., n-1
                # targets[:, 1:] are the actual tokens at positions 1, 2, ..., n-1
                shifted_logits = logits[:, :-1, :].float()  # [batch, seq_len - 1, vocab_size]
                shifted_targets = input_ids[:, 1:]  # [batch, seq_len - 1]

                # Compute log probabilities
                log_probs = torch.log_softmax(shifted_logits, dim=-1)  # [batch, seq_len - 1, vocab_size]
                batch_log_probs = log_probs.gather(2, shifted_targets.unsqueeze(2)).squeeze(2)  # [batch, seq_len - 1]

                for i, length in enumerate(lengths):
                    seq_log_probs = batch_log_probs[i, : length - 1]  # [seq_len - 1]

                    # Only count amino acid tokens (indices 5-29), not special tokens
                    # This matches the original ProGen2 paper's perplexity calculation
                    seq_targets = input_ids[i, 1:length]
                    aa_mask = (seq_targets >= PROGEN2_FIRST_AA_TOKEN) & (seq_targets <= PROGEN2_LAST_AA_TOKEN)
                    seq_log_probs = seq_log_probs[aa_mask]

                    all_metrics.append(
                        {
                            "log_likelihood": seq_log_probs.sum().item(),
                            "avg_log_likelihood": seq_log_probs.mean().item(),
                            "perplexity": torch.exp(-seq_log_probs.mean()).item(),
                        }
                    )
                    # Return unpadded logits (full vocabulary)
                    # Note: model outputs 32 logits but tokenizer has 30 tokens
                    all_logits.append(logits[i, :length, : len(PROGEN2_VOCAB)].cpu())

        return {
            "logits": all_logits if return_logits else None,
            "metrics": all_metrics,
            "vocab": PROGEN2_VOCAB,
        }


# ============================================================================
# Dispatch
# ============================================================================
_model: ProGen2Model | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = ProGen2Model(
            model_checkpoint=input_dict["model_checkpoint"],
            local_path=input_dict.get("local_path"),
        )

    operation = input_dict["operation"]
    if operation == "sample":
        return _model.sample(
            prompts=input_dict["prompts"],
            temperature=input_dict["temperature"],
            top_p=input_dict["top_p"],
            top_k=input_dict["top_k"],
            max_length=input_dict["max_length"],
            truncate_at_stop=input_dict["truncate_at_stop"],
            strip_special_tokens=input_dict["strip_special_tokens"],
            prepend_prompt=input_dict["prepend_prompt"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            batch_size=input_dict["batch_size"],
            return_logits=input_dict["return_logits"],
            seed=input_dict["seed"],
        )
    if operation == "score":
        return _model.score(
            sequences=input_dict["sequences"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            batch_size=input_dict["batch_size"],
            return_logits=input_dict["return_logits"],
            seed=input_dict["seed"],
        )
    raise ValueError(f"progen2: unknown operation {operation!r}; valid: ['sample', 'score']")


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
        raise ValueError("progen2: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
