"""Differentiable ESMFold2 inversion structure objectives over a predicted distogram.

The quantities here are reimplemented from the ESMFold2 binder-design recipe published by the
Chan Zuckerberg Biohub under the MIT license, at
https://github.com/bviggiano/esm/blob/1ad44674ef25f5a783273751ee3261617f4c5504/cookbook/tutorials/binder_design.py
The "Algorithm N" labels in the docstrings below refer to that reference, which is the authority
on what each quantity means; the implementations are our own.

Every function takes distogram logits shaped ``(batch, length, length, bins)`` over the full
complex, with the target occupying the leading positions and the binder the trailing ones, and
returns a per-example tensor shaped ``(batch,)`` that is differentiable with respect to them.
"""

import torch
from standalone_helpers import get_logger

logger = get_logger(__name__)

#: Distogram bin count and the span its interior boundaries cover, in angstroms.
DISTOGRAM_BINS = 128
_FIRST_EDGE = 2.0
_LAST_EDGE = 52.0
#: The outermost bins are open-ended, so they are given finite stand-in edges to take a midpoint.
_BELOW_FIRST = 1.0
_BEYOND_LAST = _LAST_EDGE + 5.0

#: Radius-of-gyration reference for a packed globular protein: ``_RG_COEFFICIENT * n ** _RG_EXPONENT``.
_RG_COEFFICIENT = 2.38
_RG_EXPONENT = 0.365
#: Pair distances saturate for the gyration term; beyond this they say nothing about compactness.
_RG_DISTANCE_CLAMP = 27.0


def bin_midpoints(device: torch.device | None = None) -> torch.Tensor:
    """Distance in angstroms represented by each distogram bin.

    The model emits ``DISTOGRAM_BINS`` bins whose interior boundaries are evenly spaced over
    ``[_FIRST_EDGE, _LAST_EDGE]``. The first and last bins are open-ended, so finite stand-in
    edges stand in for their outer sides and every bin is summarized by the midpoint of its span.

    Args:
        device (torch.device | None): Device to build the tensor on; ``None`` means CPU.

    Returns:
        torch.Tensor: Shape ``(DISTOGRAM_BINS,)`` of bin midpoint distances in angstroms.
    """
    interior = torch.linspace(_FIRST_EDGE, _LAST_EDGE, DISTOGRAM_BINS - 1, device=device)
    edges = torch.cat(
        (
            torch.tensor([_BELOW_FIRST], device=device),
            interior,
            torch.tensor([_BEYOND_LAST], device=device),
        )
    )
    return (edges[:-1] + edges[1:]) / 2


def contact_entropy(distogram_logits: torch.Tensor, midpoints: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Cross-entropy of the predicted distance distribution against its in-contact restriction.

    Renormalizing the distribution over only the bins nearer than ``cutoff`` gives the shape it
    would have if the pair were certainly in contact. Scoring the unrestricted log-probabilities
    against that restriction is low when the model already places the pair in contact and high
    when it does not, so minimizing it drives residues together.

    Args:
        distogram_logits (torch.Tensor): Shape ``(batch, length, length, bins)``.
        midpoints (torch.Tensor): Bin midpoint distances, shape ``(bins,)``.
        cutoff (float): Distance in angstroms within which a pair counts as in contact.

    Returns:
        torch.Tensor: Per-pair scores, shape ``(batch, length, length)``.
    """
    in_contact = midpoints < cutoff
    contact_probabilities = torch.softmax(distogram_logits.masked_fill(~in_contact, -torch.inf), dim=-1)
    log_probabilities = torch.log_softmax(distogram_logits, dim=-1)
    return -(contact_probabilities * log_probabilities).sum(-1)


def mean_of_smallest(values: torch.Tensor, mask: torch.Tensor, count: int) -> torch.Tensor:
    """Average the ``count`` smallest unmasked values along the last axis.

    Taking only the best few contacts per residue is what makes the contact objectives local: a
    residue is satisfied by a handful of close partners rather than by its average distance to
    everything. Where fewer than ``count`` entries are unmasked, the mean is over those available,
    and where none are, the result is zero.

    Args:
        values (torch.Tensor): Values to reduce, any shape.
        mask (torch.Tensor): Broadcastable boolean mask selecting eligible entries.
        count (int): How many of the smallest entries to average.

    Returns:
        torch.Tensor: ``values`` reduced along its last axis.
    """
    eligible = torch.where(mask.bool(), values, torch.inf)
    smallest, _ = torch.topk(eligible, min(count, values.shape[-1]), dim=-1, largest=False)
    taken = torch.isfinite(smallest)
    return torch.where(taken, smallest, 0.0).sum(-1) / taken.sum(-1).clamp(min=1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average the unmasked values along the last axis, yielding zero where none are unmasked.

    Args:
        values (torch.Tensor): Values to reduce, any shape.
        mask (torch.Tensor): Broadcastable boolean mask selecting eligible entries.

    Returns:
        torch.Tensor: ``values`` reduced along its last axis.
    """
    selected = mask.bool()
    return torch.where(selected, values, 0.0).sum(-1) / selected.sum(-1).clamp(min=1)


def _binder_positions(length: int, binder_length: int, device: torch.device) -> torch.Tensor:
    """Boolean mask over complex positions that is true on the trailing binder residues."""
    positions = torch.zeros(length, dtype=torch.bool, device=device)
    positions[length - binder_length :] = True
    return positions


def _contact_loss(
    distogram_logits: torch.Tensor,
    midpoints: torch.Tensor,
    *,
    contacts_per_residue: int,
    minimum_separation: int,
    cutoff: float,
    scored_positions: torch.Tensor,
    partner_positions: torch.Tensor,
) -> torch.Tensor:
    """Average, over scored residues, the best few contacts each makes with eligible partners.

    Algorithm 12. ``scored_positions`` selects which residues are averaged over and
    ``partner_positions`` which may serve as their contacts, so the same reduction expresses both
    an intra-chain objective and an interface one.

    Args:
        distogram_logits (torch.Tensor): Shape ``(batch, length, length, bins)``.
        midpoints (torch.Tensor): Bin midpoint distances, shape ``(bins,)``.
        contacts_per_residue (int): How many contacts each scored residue is credited with.
        minimum_separation (int): Sequence separation a partner must exceed; 0 disables the check,
            which is what an interface objective wants since the chains are not sequential.
        cutoff (float): Contact distance in angstroms.
        scored_positions (torch.Tensor): Boolean mask, shape ``(length,)``, of residues to average.
        partner_positions (torch.Tensor): Boolean mask, shape ``(length,)``, of eligible partners.

    Returns:
        torch.Tensor: Per-example losses, shape ``(batch,)``.
    """
    per_pair = contact_entropy(distogram_logits, midpoints, cutoff)
    eligible = partner_positions.expand_as(per_pair)
    if minimum_separation > 0:
        index = torch.arange(distogram_logits.shape[1], device=distogram_logits.device)
        far_enough = (index[:, None] - index[None, :]).abs() >= minimum_separation
        eligible = eligible & far_enough
    per_residue = mean_of_smallest(per_pair, eligible, contacts_per_residue)
    return masked_mean(per_residue, scored_positions)


def intra_contact_loss(distogram_logits: torch.Tensor, binder_length: int, midpoints: torch.Tensor) -> torch.Tensor:
    """Binder self-contact: each binder residue's two best contacts within its own chain.

    The separation floor of nine keeps trivially-close neighbours from satisfying the objective,
    so it rewards tertiary packing rather than an extended chain.

    Args:
        distogram_logits (torch.Tensor): Shape ``(batch, length, length, bins)``.
        binder_length (int): Number of trailing positions that are the binder.
        midpoints (torch.Tensor): Bin midpoint distances, shape ``(bins,)``.

    Returns:
        torch.Tensor: Per-example losses, shape ``(batch,)``.
    """
    binder = _binder_positions(distogram_logits.shape[1], binder_length, distogram_logits.device)
    return _contact_loss(
        distogram_logits,
        midpoints,
        contacts_per_residue=2,
        minimum_separation=9,
        cutoff=14.0,
        scored_positions=binder,
        partner_positions=binder,
    )


def inter_contact_loss(distogram_logits: torch.Tensor, binder_length: int, midpoints: torch.Tensor) -> torch.Tensor:
    """Interface contact: each target residue's single best contact with the binder.

    Scored over the target and partnered against the binder, so it rewards the binder reaching
    the target rather than the target rearranging.

    Args:
        distogram_logits (torch.Tensor): Shape ``(batch, length, length, bins)``.
        binder_length (int): Number of trailing positions that are the binder.
        midpoints (torch.Tensor): Bin midpoint distances, shape ``(bins,)``.

    Returns:
        torch.Tensor: Per-example losses, shape ``(batch,)``.
    """
    binder = _binder_positions(distogram_logits.shape[1], binder_length, distogram_logits.device)
    return _contact_loss(
        distogram_logits,
        midpoints,
        contacts_per_residue=1,
        minimum_separation=0,
        cutoff=22.0,
        scored_positions=~binder,
        partner_positions=binder,
    )


def globularity_loss(distogram_logits: torch.Tensor, binder_length: int, midpoints: torch.Tensor) -> torch.Tensor:
    """Algorithm 13. Excess radius of gyration over a packed globular protein of the same length.

    The expected squared distance between binder residue pairs gives a radius of gyration, which
    is compared against the empirical scaling for a packed protein. An ELU leaves compact designs
    almost unpenalized while growing linearly as the design extends.

    Args:
        distogram_logits (torch.Tensor): Shape ``(batch, length, length, bins)``.
        binder_length (int): Number of trailing positions that are the binder.
        midpoints (torch.Tensor): Bin midpoint distances, shape ``(bins,)``.

    Returns:
        torch.Tensor: Per-example losses, shape ``(batch,)``.
    """
    binder_pairs = distogram_logits[:, -binder_length:, -binder_length:, :]
    saturated = midpoints.clamp(max=_RG_DISTANCE_CLAMP)
    expected_square = (torch.softmax(binder_pairs, dim=-1) * saturated.square()).sum(-1)
    # Each unordered pair once, via the strict lower triangle.
    total = torch.tril(expected_square, diagonal=-1).sum(dim=(1, 2))
    radius = torch.sqrt(total / (binder_length * binder_length))
    packed = _RG_COEFFICIENT * binder_length**_RG_EXPONENT
    return torch.nn.functional.elu(radius - packed)


def epitope_loss(
    distogram_logits: torch.Tensor,
    epitope_mask: torch.Tensor,
    binder_length: int,
    midpoints: torch.Tensor,
    contact_distance: float,
) -> torch.Tensor:
    """Drive the binder onto the requested target hotspots.

    Each hotspot is scored by the surprisal of its best few binder contacts being within
    ``contact_distance``, then averaged over the hotspots. Unlike the interface objective this is
    directed at named residues, which is how a caller aims a binder at a particular epitope.

    Args:
        distogram_logits (torch.Tensor): Shape ``(batch, length, length, bins)``.
        epitope_mask (torch.Tensor): Boolean mask, shape ``(batch, length)``, true at hotspots.
        binder_length (int): Number of trailing positions that are the binder.
        midpoints (torch.Tensor): Bin midpoint distances, shape ``(bins,)``.
        contact_distance (float): Distance in angstroms defining a hotspot-binder contact.

    Returns:
        torch.Tensor: Per-example losses, shape ``(batch,)``.

    Raises:
        ValueError: If ``epitope_mask`` does not match the complex that was folded.
    """
    batch, length = distogram_logits.shape[0], distogram_logits.shape[1]
    if epitope_mask.shape != (batch, length):
        raise ValueError(f"epitope mask {tuple(epitope_mask.shape)} does not match complex {(batch, length)}")
    binder = _binder_positions(length, binder_length, distogram_logits.device)
    hotspot_to_binder = epitope_mask[:, :, None] & binder[None, None, :]
    in_contact = (torch.softmax(distogram_logits, dim=-1) * (midpoints < contact_distance)).sum(-1)
    surprisal = -torch.log(in_contact + 1e-8)
    per_hotspot = mean_of_smallest(surprisal, hotspot_to_binder, count=5)
    return masked_mean(per_hotspot, epitope_mask)


#: Fraction of scored positions masked in each pseudo-perplexity pass.
MASK_FRACTION = 0.15


def straight_through(relaxed: torch.Tensor) -> torch.Tensor:
    """Hard one-hot in the forward pass, relaxed distribution in the backward one.

    The language model is trained on discrete tokens, so feeding it the relaxed distribution
    directly would score a state it never sees. Passing the argmax forward while routing the
    derivative through the distribution keeps the score honest and the objective differentiable.

    Args:
        relaxed (torch.Tensor): Distribution over the last axis.

    Returns:
        torch.Tensor: One-hot values carrying ``relaxed``'s gradient.
    """
    hard = torch.nn.functional.one_hot(relaxed.argmax(-1), relaxed.shape[-1]).to(relaxed.dtype)
    return relaxed + (hard - relaxed).detach()


def pseudo_perplexity_nll(
    masked_lm: object,
    design: torch.Tensor,
    scored: torch.Tensor,
    *,
    trunk_to_lm: torch.Tensor,
    cls_id: int,
    eos_id: int,
    mask_id: int,
    vocab_size: int,
    amino_acid_offset: int,
    passes: int,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Algorithm 14. Masked-language-model pseudo-perplexity of the designed chain.

    True pseudo-perplexity masks every position in turn, which costs one forward pass per
    residue. This approximates it by masking a random fraction of the scored positions over a few
    passes and averaging the negative log-likelihood at the masked positions only, which is what
    makes it affordable inside a design loop.

    The model is duck-typed so this module needs no model library at import time; it must expose
    ``esmc.embed.weight``, ``esmc.transformer`` and ``lm_head``.

    Args:
        masked_lm (object): Masked-LM exposing the attributes above.
        design (torch.Tensor): Relaxed designed chain, shape ``(batch, length, 20)``, in
            folding-trunk column order.
        scored (torch.Tensor): Boolean mask, shape ``(batch, length)`` or ``(length,)``, of the
            positions eligible to be masked and scored.
        trunk_to_lm (torch.Tensor): Permutation matrix, shape ``(20, 20)``, mapping folding-trunk
            amino-acid columns onto the language model's.
        cls_id (int): Language-model start token id.
        eos_id (int): Language-model end token id.
        mask_id (int): Language-model mask token id.
        vocab_size (int): Language-model vocabulary size.
        amino_acid_offset (int): Index of the first amino-acid column in that vocabulary.
        passes (int): Independent mask draws to average over.
        batch_size (int): Masked sequences per forward pass.
        generator (torch.Generator | None): Draws the masks; ``None`` uses global RNG.

    Returns:
        torch.Tensor: Per-example negative log-likelihoods, shape ``(batch,)``.

    Raises:
        ValueError: If ``scored`` selects no positions for some example, or does not match
            ``design``.
    """
    batch, length, _ = design.shape
    device, dtype = design.device, masked_lm.esmc.embed.weight.dtype  # type: ignore[attr-defined]

    if scored.ndim == 1:
        scored = scored.unsqueeze(0).expand(batch, -1)
    if scored.shape != (batch, length):
        raise ValueError(f"scored mask {tuple(scored.shape)} does not match design {(batch, length)}")
    scored = scored.to(device=device, dtype=torch.bool)

    # The language model scores its own column order, and is fed one-hot tokens with the
    # sequence wrapped in start and end markers.
    in_lm_order = design @ trunk_to_lm
    tokens = torch.zeros((batch, length + 2, vocab_size), dtype=dtype, device=device)
    tokens[:, 0, cls_id] = 1
    tokens[:, -1, eos_id] = 1
    tokens[:, 1:-1, amino_acid_offset : amino_acid_offset + 20] = straight_through(in_lm_order).to(dtype)

    mask_token = torch.zeros(vocab_size, dtype=dtype, device=device)
    mask_token[mask_id] = 1

    drawn: list[torch.Tensor] = []
    masked_rows: list[torch.Tensor] = []
    for index in range(batch):
        eligible = scored[index].nonzero(as_tuple=False).flatten()
        if eligible.numel() == 0:
            raise ValueError(f"pseudo-perplexity scored mask selected no positions for example {index}")
        per_pass = max(1, int(MASK_FRACTION * eligible.numel() + 0.999))
        # Lowest random scores pick the masked subset, independently per pass.
        choice = torch.rand((passes, eligible.numel()), device=device, generator=generator)
        chosen = choice.topk(per_pass, dim=-1, largest=False).indices
        pass_mask = torch.zeros((passes, length), dtype=torch.bool, device=device)
        pass_mask[torch.arange(passes, device=device)[:, None], eligible[chosen]] = True

        rows = tokens[index : index + 1].repeat(passes, 1, 1)
        row, column = pass_mask.nonzero(as_tuple=True)
        rows[row, column + 1] = mask_token  # +1 skips the start marker
        drawn.append(pass_mask)
        masked_rows.append(rows)

    stacked = torch.cat(masked_rows, dim=0)
    logits = torch.cat(
        [
            _lm_logits(masked_lm, stacked[start : start + batch_size], device)
            for start in range(0, stacked.shape[0], batch_size)
        ],
        dim=0,
    )

    losses = []
    for index, pass_mask in enumerate(drawn):
        window = logits[index * passes : (index + 1) * passes]
        log_probabilities = window.log_softmax(-1)[:, 1:-1, amino_acid_offset : amino_acid_offset + 20]
        target = in_lm_order[index].to(log_probabilities.dtype).unsqueeze(0)
        per_position = -(log_probabilities * target).sum(-1)
        losses.append(per_position[pass_mask].mean())
    return torch.stack(losses, dim=0)


def _lm_logits(masked_lm: object, tokens: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run one chunk of one-hot token rows through the masked LM and return its vocabulary logits."""
    esmc = masked_lm.esmc  # type: ignore[attr-defined]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        hidden, *_ = esmc.transformer(
            tokens @ esmc.embed.weight.to(tokens.dtype),
            sequence_id=None,
            layers_to_collect=[],
            output_attentions=False,
        )
        return masked_lm.lm_head(hidden)  # type: ignore[attr-defined]
