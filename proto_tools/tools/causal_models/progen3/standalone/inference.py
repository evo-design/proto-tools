"""ProGen3 standalone inference."""

import json
import os
import re
import sys
from logging import getLogger
from typing import Any

from standalone_helpers import serialize_output, set_torch_seed

logger = getLogger(__name__)

HUGGINGFACE_REPO_PREFIX = "Profluent-Bio"


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
        logger.info(f"Restricted CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']} for device {device}")

        import torch
        from progen3.modeling import ProGen3ForCausalLM

        model_path = self.local_path or f"{HUGGINGFACE_REPO_PREFIX}/{self.model_checkpoint}"

        self.model = ProGen3ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
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
            raise RuntimeError("Cannot move unloaded model. Call load() first.")
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
        num_sequences: int = 1,
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
            num_sequences (int): Number of sequences per prompt.
            batch_size (int): Batch size for generation.
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

        all_sequences = []
        all_prompts = []
        all_directions = []

        for prompt in prompts:
            direction = "fwd" if prompt.startswith("1") else "rev"
            aa_prompt = prompt[1:]

            results = list(
                generator.generate(
                    prompt=prompt,
                    num_sequences=num_sequences,
                    min_new_tokens=min_new_tokens,
                    max_new_tokens=max_new_tokens,
                )
            )
            for r in results:
                if r.sequence is not None:
                    seq = r.sequence
                else:
                    # compile_generation() returned None (e.g. truncated at
                    # max_new_tokens before <eos>). Strip special tokens and
                    # compile manually.
                    stripped = re.sub(r"[^A-Z]", "", r.generation)
                    seq = aa_prompt + stripped if direction == "fwd" else stripped[::-1] + aa_prompt
                    logger.warning(
                        "compile_generation returned None for prompt '%s'; "
                        "falling back to manual compilation (%d residues)",
                        prompt,
                        len(stripped),
                    )
                all_sequences.append(seq)
                all_prompts.append(aa_prompt)
                all_directions.append(direction)

        return {
            "sequences": all_sequences,
            "prompts": all_prompts,
            "directions": all_directions,
        }

    def score(
        self,
        sequences: list[str],
        device: str = "cuda",
        verbose: bool = False,
        batch_size: int = 1,
        reduction: str = "mean",
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
            reduction (str): How to aggregate per-token log-likelihoods ("mean" or "sum").
            seed (int | None): Random seed. Scoring is deterministic given the
                model state, but we still seed RNGs/cudnn flags so consecutive
                calls in a persistent worker behave identically regardless of
                call order.

        Returns:
            dict[str, Any]: Dict with "metrics" and "per_position_metrics" keys.
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
        scorer = _PerPositionScorer(
            self.model,
            max_batch_tokens=max_batch_tokens,
            reduction=reduction,
        )

        result = scorer.score_batch_with_positions(sequences)

        metrics = []
        for i in range(len(sequences)):
            ll = result["log_likelihood"][i]
            ppl = result["perplexity"][i]
            ll_val = ll.item() if hasattr(ll, "item") else float(ll)
            ppl_val = ppl.item() if hasattr(ppl, "item") else float(ppl)

            avg_ll = ll_val

            metrics.append(
                {
                    "log_likelihood": ll_val,
                    "avg_log_likelihood": avg_ll,
                    "perplexity": ppl_val,
                }
            )

        return {
            "metrics": metrics,
            "per_position_metrics": result["per_position_metrics"],
        }


class _PerPositionScorer:
    """Wraps ProGen3Scorer to extract per-position log-likelihoods.

    Duplicates the forward pass logic from ProGen3Scorer._log_likelihoods
    to return both reduced and per-position NLL values.
    """

    def __init__(self, model: Any, max_batch_tokens: int = 65536, reduction: str = "mean") -> None:
        import torch
        from progen3.batch_preparer import ProGen3BatchPreparer
        from progen3.common import dist

        self.model = model
        self.max_batch_tokens = max_batch_tokens
        self.reduction = reduction
        self.batch_preparer = ProGen3BatchPreparer()
        self._torch = torch
        self._dist = dist
        self.model.eval()

    def _log_likelihoods_with_positions(self, model_forward_kwargs: dict[str, Any]) -> tuple[Any, Any, Any]:
        """Compute both aggregate and per-position log-likelihoods.

        Returns:
            tuple: (reduced_ll, per_token_ll, target_mask) where:
                - reduced_ll: shape [batch], aggregate LL per sequence
                - per_token_ll: shape [batch, seq_len-1], per-token LL
                - target_mask: shape [batch, seq_len-1], bool mask for valid tokens
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
        if self.reduction == "mean":
            reduced_nll = reduced_nll / target_mask.sum(dim=1)

        return -reduced_nll.detach(), per_token_ll, target_mask.detach()

    def score_batch_with_positions(self, sequences: list[str]) -> dict[str, Any]:
        """Score sequences and return aggregate + per-position metrics."""
        torch = self._torch
        dist = self._dist
        device = dist.get_device()

        with torch.no_grad():
            return self._score_batch_impl(sequences, device)

    def _score_batch_impl(self, sequences: list[str], device: Any) -> dict[str, Any]:
        """Internal implementation for score_batch_with_positions."""
        kwargs_fwd = self.batch_preparer.get_batch_kwargs(
            sequences,
            device=device,
            reverse=False,
        )
        reduced_fwd, per_pos_fwd, mask_fwd = self._log_likelihoods_with_positions(kwargs_fwd)

        kwargs_rev = self.batch_preparer.get_batch_kwargs(
            sequences,
            device=device,
            reverse=True,
        )
        reduced_rev, per_pos_rev, mask_rev = self._log_likelihoods_with_positions(kwargs_rev)

        scores: dict[str, Any] = {"log_likelihood": [], "perplexity": []}
        all_per_position: list[dict[str, list[float | None]]] = []

        for i in range(len(sequences)):
            # Aggregate scores (same as upstream)
            ll = (reduced_fwd[i] + reduced_rev[i]) / 2
            scores["log_likelihood"].append(ll)
            scores["perplexity"].append(self._torch.exp(-ll))

            seq_len = len(sequences[i])

            # Extract per-position forward LL (strip special tokens).
            # Token layout: [<bos>, direction_marker, AA_0, ..., AA_{L-1}, end_marker, <eos>]
            # After shift-by-1 in NLL computation, valid AA positions start at index 1
            # (predicting AA_0 from <bos>+marker) through index L (predicting end_marker).
            # We want indices 1..L which give LL for AA positions 1..L-1,
            # with position 0 having no forward score.
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
            num_sequences=input_dict["num_sequences"],
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
            reduction=input_dict["reduction"],
            seed=input_dict["seed"],
        )
    raise ValueError(f"Unknown operation: {operation}")


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
        raise ValueError("Usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
