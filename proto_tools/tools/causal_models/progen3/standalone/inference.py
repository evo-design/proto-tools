"""ProGen3 standalone inference."""

import json
import os
import re
import sys
from collections import defaultdict
from copy import deepcopy
from typing import Any

from standalone_helpers import get_logger, log_likelihood_metrics, serialize_output, set_torch_seed

logger = get_logger(__name__)

HUGGINGFACE_REPO_PREFIX = "Profluent-Bio"

# Tokenizer order (vocab size 34): 0-5 specials (<pad>, <bos>, <eos>, <bos_glm>,
# <eos_span>, <mask>), 6-7 direction markers ("1", "2"), 8-33 amino acid letters A-Z.
PROGEN3_VOCAB: list[str] = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<bos_glm>",
    "<eos_span>",
    "<mask>",
    "1",
    "2",
    *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
]


class ProGen3Model:
    """ProGen3 model wrapper for protein sequence generation and scoring."""

    def __init__(
        self,
        model_checkpoint: str = "progen3-762m",
        local_path: str | None = None,
    ):
        """Initialize ProGen3 model wrapper."""
        self._loaded = False
        self.model_checkpoint = model_checkpoint
        self.local_path = local_path
        self.device: str | None = None
        self.model: Any = None

    def load(self, device: str, verbose: bool = False) -> None:
        """Load ProGen3 model to device.

        Restricts CUDA_VISIBLE_DEVICES before loading so that progen3's
        internal ``dist.get_device()`` (which calls ``torch.cuda.current_device()``)
        returns the correct (and only visible) GPU.
        """
        if verbose:
            logger.info(f"Loading ProGen3 model: {self.model_checkpoint} on {device}")

        # Restrict GPU visibility to the allocated device. progen3's scorer
        # and generator use dist.get_device() → torch.cuda.current_device()
        # internally for tensor placement, so we must ensure only the
        # allocated GPU is visible.
        from standalone_helpers import get_subprocess_device_env

        restricted_env = get_subprocess_device_env(device)
        os.environ["CUDA_VISIBLE_DEVICES"] = restricted_env["CUDA_VISIBLE_DEVICES"]
        logger.debug(f"Restricted CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']} for device {device}")

        import torch
        from progen3.modeling import ProGen3ForCausalLM

        model_path = self.local_path or f"{HUGGINGFACE_REPO_PREFIX}/{self.model_checkpoint}"

        try:
            self.model = ProGen3ForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            )
        except OSError as e:
            raise RuntimeError(f"progen3: HF weight load from {model_path!r} failed: {e}") from e
        # After CUDA_VISIBLE_DEVICES restriction, cuda:0 is the allocated GPU
        self.model.eval().to("cuda:0")
        self.device = device
        self._loaded = True

        if verbose:
            logger.info("ProGen3 model loaded successfully")

    def to_device(self, device: str) -> None:
        """Move model to a different device.

        GPU → CPU uses standard PyTorch ``.to("cpu")``.
        CPU → GPU (or GPU → GPU) fully reloads so that
        CUDA_VISIBLE_DEVICES is updated for progen3's internal
        ``dist.get_device()`` calls.
        """
        if not self._loaded:
            raise ValueError("progen3: cannot move unloaded model to device — call load() first")
        if self.device == device:
            return
        if device == "cpu":
            from standalone_helpers import move_model_to_device

            self.model = move_model_to_device(self.model, self.device, device)
            self.device = device
        else:
            # Reload to update CUDA_VISIBLE_DEVICES for the new GPU
            self._loaded = False
            self.model = None
            self.load(device)

    def sample(
        self,
        prompts: list[str],
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        min_new_tokens: int = 1,
        batch_size: int = 1,
        device: str = "cuda",
        verbose: bool = False,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Sample protein sequences using ProGen3.

        Args:
            prompts (list[str]): Prompt sequences starting with "1" (forward) or "2" (reverse).
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling parameter.
            max_new_tokens (int): Maximum new tokens to generate.
            min_new_tokens (int): Minimum new tokens to generate.
            batch_size (int): Maximum number of same-length prompts per
                generation batch.
            device (str): Device to run on.
            verbose (bool): Whether to log progress.
            seed (int | None): Random seed for reproducibility.

        Returns:
            dict[str, Any]: Dict with "sequences" key containing list of generated sequences.
        """
        set_torch_seed(seed)

        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

        from progen3.generator import ProGen3Generator

        # +512 accounts for prompt tokens; progen3's BPE tokenizer maps
        # each amino acid to ~1 token so this covers prompts up to ~512 residues.
        max_batch_tokens = max(batch_size * (max_new_tokens + 512), 65536)
        generator = ProGen3Generator(
            self.model,
            max_batch_tokens=max_batch_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        prompt_infos = [self._prepare_generation_prompt(generator, idx, prompt) for idx, prompt in enumerate(prompts)]
        outputs: dict[int, dict[str, str]] = {}

        for chunk_start in range(0, len(prompt_infos), batch_size):
            chunk = prompt_infos[chunk_start : chunk_start + batch_size]
            grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
            for info in chunk:
                grouped[(info["num_input_tokens"], info["end_token_id"])].append(info)

            for group in grouped.values():
                if verbose:
                    logger.info(f"Generating ProGen3 batch with {len(group)} prompt(s)")
                for info, generated in zip(
                    group,
                    self._generate_batch(
                        generator,
                        group,
                        min_new_tokens=min_new_tokens,
                        max_new_tokens=max_new_tokens,
                    ),
                    strict=True,
                ):
                    outputs[info["index"]] = generated

        generated_outputs = [outputs[i] for i in range(len(prompt_infos))]

        return {
            "sequences": [output["sequence"] for output in generated_outputs],
            "prompts": [output["prompt"] for output in generated_outputs],
            "directions": [output["direction"] for output in generated_outputs],
        }

    def _prepare_generation_prompt(self, generator: Any, index: int, raw_prompt: str) -> dict[str, Any]:
        """Prepare one prompt for generation."""
        from progen3.batch_preparer import assert_valid_instance, is_glm_instance
        from progen3.generator import parse_directed_prompt

        prompt, direction = parse_directed_prompt(raw_prompt)
        assert_valid_instance(prompt)

        generation_kwargs = generator.batch_preparer.get_generation_kwargs(prompt, direction == "rev")
        input_encoding = {key: value.squeeze(0) for key, value in generation_kwargs.items()}

        end_token = "<eos_span>" if is_glm_instance(prompt) else "<eos>"
        end_token_id = generator.batch_preparer.tokenizer.token_to_id(end_token)

        return {
            "index": index,
            "raw_prompt": raw_prompt,
            "prompt": prompt,
            "aa_prompt": raw_prompt[1:],
            "direction": direction,
            "input_encoding": input_encoding,
            "num_input_tokens": len(input_encoding["input_ids"]),
            "end_token_id": end_token_id,
        }

    def _generate_batch(
        self,
        generator: Any,
        prompt_infos: list[dict[str, Any]],
        *,
        min_new_tokens: int,
        max_new_tokens: int,
    ) -> list[dict[str, str]]:
        """Generate one same-length prompt batch."""
        import torch
        from progen3.common import dist
        from progen3.common.dist import generate
        from progen3.generator import compile_generation

        input_encoding = {
            key: torch.stack([info["input_encoding"][key] for info in prompt_infos]).to(device=dist.get_device())
            for key in ("input_ids", "sequence_ids", "position_ids")
        }
        num_input_tokens = prompt_infos[0]["num_input_tokens"]

        gen_config = deepcopy(generator.default_gen_config)
        gen_config.eos_token_id = prompt_infos[0]["end_token_id"]
        # Mirror upstream: generated token limits include completion terminators.
        gen_config.max_new_tokens = max_new_tokens + 2
        gen_config.min_new_tokens = min_new_tokens + 2
        gen_config.num_return_sequences = 1

        cached_length = num_input_tokens - 1
        key_value_cache = None
        if cached_length > 0:
            with torch.no_grad():
                cached_encoding = {key: value[:, :cached_length] for key, value in input_encoding.items()}
                key_value_cache = self.model(**cached_encoding, use_cache=True, return_dict=True).past_key_values
            torch.cuda.empty_cache()

        outputs = generate(self.model, **input_encoding, generation_config=gen_config, past_key_values=key_value_cache)

        completions = outputs.sequences[:, num_input_tokens:].tolist()
        decoded_completions = generator.batch_preparer.tokenizer.decode_batch(
            completions,
            skip_special_tokens=False,
        )

        generated = []
        for info, decoded_completion in zip(prompt_infos, decoded_completions, strict=True):
            sequence = compile_generation(info["prompt"], decoded_completion, info["direction"])
            if sequence is None:
                match = re.match(r"[A-Z]*", decoded_completion)
                stripped = match.group(0) if match else ""
                sequence = (
                    info["aa_prompt"] + stripped if info["direction"] == "fwd" else stripped[::-1] + info["aa_prompt"]
                )
                logger.debug(
                    "compile_generation returned None for prompt '%s'; "
                    "falling back to manual compilation (%d residues)",
                    info["raw_prompt"],
                    len(stripped),
                )
            generated.append(
                {
                    "sequence": sequence,
                    "prompt": info["aa_prompt"],
                    "direction": info["direction"],
                }
            )

        return generated

    def score(
        self,
        sequences: list[str],
        device: str = "cuda",
        verbose: bool = False,
        batch_size: int = 1,
        return_logits: bool = False,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Score protein sequences using bidirectional likelihood.

        Returns aggregate metrics and per-position log-likelihoods for each
        sequence. Per-position scores include forward, reverse, and
        bidirectional (averaged) values.

        Args:
            sequences (list[str]): Protein sequences to score (N-to-C direction).
            device (str): Device to run on.
            verbose (bool): Whether to log progress.
            batch_size (int): Batch size for scoring.
            return_logits (bool): Whether to include forward-pass per-position logits in
                the output. Only the forward (N→C) pass is returned, matching the
                evo/progen2 convention; bidirectional info is already exposed via
                ``per_position_metrics``.
            seed (int | None): Random seed. Scoring is deterministic given the
                model state, but we still seed RNGs/cudnn flags so consecutive
                calls in a persistent worker behave identically regardless of
                call order.

        Returns:
            dict[str, Any]: Dict with ``metrics``, ``per_position_metrics``, optional
                ``logits`` (per-sequence tensor of shape ``(tokenized_len, vocab_size)``
                when ``return_logits=True``), and ``vocab``.
        """
        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

        set_torch_seed(seed)

        max_batch_tokens = max(
            batch_size * max(len(s) for s in sequences) * 2,
            65536,
        )
        scorer = _PerPositionScorer(self.model, max_batch_tokens=max_batch_tokens)

        result = scorer.score_batch_with_positions(sequences, return_logits=return_logits)

        # Stay consistent with the per-position output by deriving the triple from those values.
        metrics = []
        for per_pos in result["per_position_metrics"]:
            valid = [x for x in per_pos["log_likelihood"] if x is not None]
            # No scorable positions (e.g. a 1-residue sequence) → neutral sentinel, not div-by-zero.
            if valid:
                metrics.append(log_likelihood_metrics(sum(valid) / len(valid), len(valid)))
            else:
                metrics.append(log_likelihood_metrics(0.0, 0))

        return {
            "metrics": metrics,
            "per_position_metrics": result["per_position_metrics"],
            "logits": result.get("logits"),
            "vocab": PROGEN3_VOCAB,
        }


class _PerPositionScorer:
    """Wraps ProGen3Scorer to extract per-position log-likelihoods.

    Duplicates the forward pass logic from ProGen3Scorer._log_likelihoods
    to return both reduced and per-position NLL values.
    """

    def __init__(self, model: Any, max_batch_tokens: int = 65536) -> None:
        import torch
        from progen3.batch_preparer import ProGen3BatchPreparer
        from progen3.common import dist

        self.model = model
        self.max_batch_tokens = max_batch_tokens
        self.batch_preparer = ProGen3BatchPreparer()
        self._torch = torch
        self._dist = dist
        self.model.eval()

    def _log_likelihoods_with_positions(self, model_forward_kwargs: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
        """Compute both aggregate and per-position log-likelihoods.

        Returns:
            tuple: ``(reduced_ll, per_token_ll, target_mask, raw_logits)`` where:
                - reduced_ll: shape [batch], aggregate LL per sequence
                - per_token_ll: shape [batch, seq_len-1], per-token LL
                - target_mask: shape [batch, seq_len-1], bool mask for valid tokens
                - raw_logits: shape [batch, seq_len, vocab_size], pre-shift forward logits
        """
        torch = self._torch
        nn = torch.nn

        output = self.model(
            input_ids=model_forward_kwargs["input_ids"],
            labels=model_forward_kwargs["labels"],
            sequence_ids=model_forward_kwargs["sequence_ids"],
            position_ids=model_forward_kwargs["position_ids"],
            return_dict=True,
        )
        labels = model_forward_kwargs["labels"]
        target_mask = (labels != self.model.config.pad_token_id)[..., 1:].contiguous()

        targets = labels[..., 1:].contiguous()
        logits = output.logits[..., :-1, :].contiguous().to(torch.float32)
        flat_logits = logits.view(-1, logits.shape[-1])
        nll = nn.functional.cross_entropy(
            flat_logits,
            targets.view(-1),
            reduction="none",
        ).view(targets.shape)

        per_token_ll = -(nll * target_mask.to(nll)).detach()
        reduced_nll = (nll * target_mask.to(nll)).sum(dim=1)

        return -reduced_nll.detach(), per_token_ll, target_mask.detach(), output.logits.detach()

    def score_batch_with_positions(self, sequences: list[str], return_logits: bool = False) -> dict[str, Any]:
        """Score sequences and return aggregate + per-position metrics."""
        torch = self._torch
        dist = self._dist
        device = dist.get_device()

        with torch.no_grad():
            return self._score_batch_impl(sequences, device, return_logits=return_logits)

    def _score_batch_impl(self, sequences: list[str], device: Any, return_logits: bool = False) -> dict[str, Any]:
        """Internal implementation for score_batch_with_positions."""
        kwargs_fwd = self.batch_preparer.get_batch_kwargs(
            sequences,
            device=device,
            reverse=False,
        )
        reduced_fwd, per_pos_fwd, mask_fwd, logits_fwd = self._log_likelihoods_with_positions(kwargs_fwd)

        kwargs_rev = self.batch_preparer.get_batch_kwargs(
            sequences,
            device=device,
            reverse=True,
        )
        # Reverse-pass logits aren't returned (matches evo/progen2 forward-only convention).
        reduced_rev, per_pos_rev, mask_rev, _ = self._log_likelihoods_with_positions(kwargs_rev)

        scores: dict[str, Any] = {"log_likelihood": [], "perplexity": []}
        all_per_position: list[dict[str, list[float | None]]] = []

        for i in range(len(sequences)):
            # Aggregate scores (same as upstream)
            ll = (reduced_fwd[i] + reduced_rev[i]) / 2
            scores["log_likelihood"].append(ll)
            scores["perplexity"].append(self._torch.exp(-ll))

            seq_len = len(sequences[i])

            # Extract per-position forward LL (strip special tokens). Token layout:
            # [<bos>, marker, AA_0..AA_{L-1}, end, <eos>]; after shift-by-1, valid indices are 1..L
            # (predicting AA_0 from <bos>+marker through end_marker). Position 0 has no forward score.
            fwd_mask = mask_fwd[i].bool()
            fwd_vals = per_pos_fwd[i][fwd_mask].cpu().tolist()
            # fwd_vals has L values: LL for predicting tokens at positions 1..L
            # (AA_0 through end_marker). We only want the L-1 AA predictions.
            # The last value is the end_marker prediction — drop it.
            fwd_aa = fwd_vals[: seq_len - 1] if len(fwd_vals) >= seq_len else fwd_vals

            # Extract per-position reverse LL (strip special tokens, reverse order).
            rev_mask = mask_rev[i].bool()
            rev_vals = per_pos_rev[i][rev_mask].cpu().tolist()
            rev_aa = rev_vals[: seq_len - 1] if len(rev_vals) >= seq_len else rev_vals
            rev_aa = list(reversed(rev_aa))

            # Build length-L arrays with None at endpoints
            fwd_full: list[float | None] = [None] + fwd_aa + ([None] * (seq_len - 1 - len(fwd_aa)))
            rev_full: list[float | None] = rev_aa + ([None] * (seq_len - 1 - len(rev_aa))) + [None]
            fwd_full = fwd_full[:seq_len]
            rev_full = rev_full[:seq_len]

            # Bidirectional: average where both exist
            bidir: list[float | None] = []
            for f, r in zip(fwd_full, rev_full, strict=True):
                if f is not None and r is not None:
                    bidir.append((f + r) / 2)
                elif f is not None:
                    bidir.append(f)
                elif r is not None:
                    bidir.append(r)
                else:
                    bidir.append(None)

            all_per_position.append(
                {
                    "forward_log_likelihood": fwd_full,
                    "reverse_log_likelihood": rev_full,
                    "log_likelihood": bidir,
                }
            )

        scores["per_position_metrics"] = all_per_position

        if return_logits:
            # Per-sequence forward logits sliced to the unpadded tokenized length;
            # bf16 → float32 because numpy-backed serialization has no bf16 dtype.
            pad_id = self.model.config.pad_token_id
            labels_fwd = kwargs_fwd["labels"]
            all_logits = []
            for i in range(len(sequences)):
                length = int((labels_fwd[i] != pad_id).sum().item())
                all_logits.append(logits_fwd[i, :length, : len(PROGEN3_VOCAB)].float().cpu())
            scores["logits"] = all_logits

        return scores


# ============================================================================
# Dispatch
# ============================================================================
_model: ProGen3Model | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = ProGen3Model(
            model_checkpoint=input_dict["model_checkpoint"],
            local_path=input_dict.get("local_path"),
        )

    operation = input_dict["operation"]
    if operation == "sample":
        return _model.sample(
            prompts=input_dict["prompts"],
            temperature=input_dict["temperature"],
            top_p=input_dict["top_p"],
            max_new_tokens=input_dict["max_new_tokens"],
            min_new_tokens=input_dict["min_new_tokens"],
            batch_size=input_dict["batch_size"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
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
    raise ValueError(f"progen3: unknown operation {operation!r}; valid: ['sample', 'score']")


def to_device(device: str) -> dict[str, Any]:
    """Move model to specified device (called by DeviceManager).

    Updates CUDA_VISIBLE_DEVICES when moving to a GPU device so that
    progen3's internal dist.get_device() returns the correct device.
    """
    global _model
    if _model is not None and _model._loaded:
        _model.to_device(device)
        return {"success": True, "device": device}
    return {"success": True, "device": device, "note": "model not loaded yet"}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    from standalone_helpers import get_pytorch_memory_stats

    global _model
    device = _model.device if _model and hasattr(_model, "device") else 0
    result: dict[str, Any] = get_pytorch_memory_stats(device)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("progen3: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
