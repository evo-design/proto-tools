"""Iterative-refinement sampling on token tensors."""

import math
from collections.abc import Callable

import torch


def _cosine_schedule(t: torch.Tensor) -> torch.Tensor:
    """Cosine schedule: t in [0, 1] -> fraction still masked."""
    return torch.cos(t * math.pi * 0.5)


def _linear_schedule(t: torch.Tensor) -> torch.Tensor:
    """Linear schedule: 1 - t (uniform commit rate)."""
    return 1.0 - t


_NOISE_SCHEDULES: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "cosine": _cosine_schedule,
    "linear": _linear_schedule,
}


def _annealed_temperature(step: int, num_steps: int, initial_T: float) -> float:
    """Cool toward ~0 across rounds."""
    step_ratio = step / max(1, num_steps - 1)
    return max(initial_T - step_ratio, 0.001) ** 2


def _sample_with_top_p(
    logits: torch.Tensor,  # (B, L, vocab)
    top_p: float,
    temperature: float,
    valid_token_mask: torch.Tensor,  # (vocab,) bool
) -> torch.Tensor:  # (B, L)
    """Temperature → invalid-token mask → optional top-p → multinomial."""
    scaled = logits / max(temperature, 1e-8)
    scaled = scaled.masked_fill(~valid_token_mask, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, dim=-1, descending=True)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        nucleus = cumulative <= top_p
        nucleus[..., 0] = True  # always keep the top-1
        keep = torch.zeros_like(scaled, dtype=torch.bool)
        keep.scatter_(-1, sorted_indices, nucleus)
        scaled = scaled.masked_fill(~keep, float("-inf"))

    probs = torch.softmax(scaled, dim=-1)
    flat = probs.view(-1, probs.size(-1))
    sampled = torch.multinomial(flat, 1).squeeze(-1)
    return sampled.view(probs.shape[:-1])


def _commit_mask(
    still_masked: torch.Tensor,  # (L,) bool
    num_to_commit: int,
    strategy: str,
    entropy: torch.Tensor | None,  # (L,) when strategy=="entropy"
) -> torch.Tensor:
    """Pick num_to_commit positions from the still-masked set."""
    out = torch.zeros_like(still_masked, dtype=torch.bool)
    if num_to_commit <= 0:
        return out

    if strategy == "entropy" and entropy is not None:
        masked_entropy = entropy.masked_fill(~still_masked, torch.finfo(entropy.dtype).max)
        _, indices = masked_entropy.topk(num_to_commit, largest=False)
        out.scatter_(-1, indices, True)
        return out & still_masked

    masked_positions = still_masked.nonzero(as_tuple=True)[0]
    n = min(num_to_commit, len(masked_positions))
    rnd = torch.randperm(len(masked_positions), device=still_masked.device)[:n]
    out[masked_positions[rnd]] = True
    return out


def iterative_sample(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    initial_tokens: torch.Tensor,
    mask_token_id: int,
    valid_token_ids: list[int],
    *,
    num_steps: int,
    schedule: str = "cosine",
    strategy: str = "random",
    temperature: float = 1.0,
    top_p: float = 1.0,
    temperature_annealing: bool = True,
) -> torch.Tensor:
    """Iteratively refine ``initial_tokens`` over ``num_steps`` rounds.

    Args:
        forward_fn (Callable[[torch.Tensor], torch.Tensor]): ``(B, L)`` tokens →
            ``(B, L, vocab)`` logits.
        initial_tokens (torch.Tensor): Token tensor with ``mask_token_id`` at
            positions to fill.
        mask_token_id (int): Vocab ID of the mask token.
        valid_token_ids (list[int]): Vocab IDs that may be sampled; others are
            masked to ``-inf``.
        num_steps (int): Number of refinement rounds.
        schedule (str): ``"cosine"`` or ``"linear"``.
        strategy (str): ``"random"`` or ``"entropy"`` (commit lowest-entropy first).
        temperature (float): Initial sampling temperature.
        top_p (float): Nucleus threshold; ``1.0`` disables.
        temperature_annealing (bool): If ``True``, anneal toward 0 across rounds.

    Returns:
        torch.Tensor: ``(B, L)`` tokens with originally-masked positions filled.
    """
    schedule_fn = _NOISE_SCHEDULES[schedule]
    tokens = initial_tokens.clone()  # (B, L)
    device = tokens.device
    # Fixed denominator for the schedule: count of mask positions before any commit.
    total_to_sample = (tokens == mask_token_id).sum(dim=-1).float()  # (B,)
    valid_token_mask: torch.Tensor | None = None  # (vocab,) — built lazily on first forward

    for step in range(num_steps):
        with torch.inference_mode():
            logits = forward_fn(tokens)  # (B, L, vocab)

        if valid_token_mask is None:
            valid_token_mask = torch.zeros(logits.size(-1), dtype=torch.bool, device=device)
            valid_token_mask[valid_token_ids] = True

        # Anneal toward 0 if requested; otherwise hold at T0.
        T = _annealed_temperature(step, num_steps, temperature) if temperature_annealing else temperature
        candidates = _sample_with_top_p(logits, top_p, T, valid_token_mask)  # (B, L)

        # Per-position entropy is only needed when strategy="entropy".
        entropy: torch.Tensor | None = None
        if strategy == "entropy":
            scaled = (logits / max(T, 1e-8)).masked_fill(~valid_token_mask, float("-inf"))
            probs = torch.softmax(scaled, dim=-1)  # (B, L, vocab)
            entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=-1)  # (B, L)

        # Schedule maps progress in [0, 1] to fraction still masked AFTER this step.
        perc_after = float(schedule_fn(torch.tensor((step + 1) / num_steps, device=device)).item())

        # Per-prompt commit: schedule + strategy decide which still-masked
        # positions adopt their sampled candidate this round.
        for b in range(tokens.size(0)):
            still_masked = tokens[b] == mask_token_id  # (L,) bool
            still_count = int(still_masked.sum().item())
            if still_count == 0:
                continue
            num_to_commit = still_count - int(perc_after * total_to_sample[b].item() + 0.1)
            commit = _commit_mask(  # (L,) bool
                still_masked,
                num_to_commit,
                strategy,
                entropy[b] if entropy is not None else None,
            )
            tokens[b] = torch.where(commit, candidates[b], tokens[b])

    return tokens
