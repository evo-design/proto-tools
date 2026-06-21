"""proto_tools/transforms/masking/base.py.

Masking strategies for masked language model sampling.
"""

import logging
from collections.abc import Callable
from typing import Any, Literal, get_args

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from proto_tools.transforms.masking.maskers import (
    MASKERS,
    Masker,
    MaskingInput,
    MaskingMethod,
    compatible_methods,
)
from proto_tools.utils import ConfigField
from proto_tools.utils.sequence import validate_positions_list

logger = logging.getLogger(__name__)

# The token that marks a position for re-design. Downstream sampling tools
# replace each "_" with a predicted amino acid.
MASK_TOKEN = "_"  # noqa: S105 -- not a password


# ============================================================================
# Helpers
# ============================================================================


def mutable_mask(seq: str, fixed: list[int] | None = None) -> list[bool]:
    """Return a boolean mask where True = designable (eligible for masking).

    Args:
        seq (str): Protein sequence (e.g. "MKTLLIFLA").
        fixed (list[int] | None): 1-indexed positions to keep unchanged. Applied uniformly
            to every sequence in a batch by the caller.

    Returns:
        list[bool]: List of bools, one per character in *seq*.
    """
    fixed_set = set(fixed) if fixed else set()
    return [
        # Position is designable if it's not already masked
        # and not in the fixed set (convert 0-indexed i to 1-indexed)
        c != MASK_TOKEN and (i + 1) not in fixed_set
        for i, c in enumerate(seq)
    ]


def apply_mask(seq: str, positions: list[int]) -> str:
    """Replace characters at 0-indexed *positions* with MASK_TOKEN."""
    chars = list(seq)
    for p in positions:
        chars[p] = MASK_TOKEN
    return "".join(chars)


def validate_enough_mutable(
    seq: str,
    mutable: list[bool],
    count: int,
    seq_index: int,
) -> None:
    """Raise if we're asked to mask more positions than are available."""
    n_mutable = sum(mutable)
    if count > n_mutable:
        raise ValueError(
            f"Sequence {seq_index}: requested {count} mutations but only "
            f"{n_mutable} mutable positions available (len={len(seq)})"
        )


def _resolve_count(
    num_mutations: int | None,
    mask_fraction: float | None,
    n_designable: int,
) -> int:
    """Determine how many positions to mask for a single sequence.

    Priority: num_mutations (exact) > mask_fraction (proportional) > 30% (default).
    mask_fraction is applied to *designable* count, not full sequence length.

    Args:
        num_mutations (int | None): Exact number of positions to mask, or None.
        mask_fraction (float | None): Fraction of designable positions to mask, or None.
        n_designable (int): Number of designable (mutable) positions in the sequence.
    """
    if num_mutations is not None:
        return num_mutations
    if mask_fraction is not None:
        # Always mask at least 1 position when designable positions exist,
        # even if round(n * frac) would be 0 (e.g. 2 positions × 0.1 = 0.2 → 0).  # noqa: RUF003
        count = round(n_designable * mask_fraction)
        return max(1, count) if n_designable > 0 else 0
    # Default: mask 30% of designable positions (at least 1)
    return max(1, round(n_designable * 0.3)) if n_designable > 0 else 0


def _validate_mutation_spec(
    num_mutations: int | None,
    mask_fraction: float | None,
) -> None:
    """Ensure num_mutations and mask_fraction aren't both set."""
    if num_mutations is not None and mask_fraction is not None:
        raise ValueError("Set num_mutations or mask_fraction, not both")


def weighted_sample(
    eligible: list[int],
    scores: list[float],
    k: int,
    rng: np.random.RandomState | None = None,
) -> list[int]:
    """Softmax-weighted sampling without replacement (numpy-accelerated).

    Used by MaskingStrategy to select positions based on scores.
    Uniform scores degrade to uniform random sampling.

    Args:
        eligible (list[int]): 0-indexed position indices to sample from.
        scores (list[float]): One score per position. Higher means more likely to be picked.
        k (int): How many positions to select.
        rng (np.random.RandomState | None): Random number generator for reproducibility.
            If None, uses the global numpy RNG.

    Returns:
        list[int]: List of *k* 0-indexed positions drawn from *eligible*.
    """
    if k >= len(eligible):
        return list(eligible)

    # Numerically stable softmax → probability weights
    s = np.asarray(scores, dtype=np.float64)
    s = s - s.max()
    weights = np.exp(s)
    weights /= weights.sum()

    # numpy's multinomial-style sampling without replacement
    if rng is None:
        chosen_idx = np.random.choice(len(eligible), size=k, replace=False, p=weights)
    else:
        chosen_idx = rng.choice(len(eligible), size=k, replace=False, p=weights)
    return [eligible[i] for i in chosen_idx]


def apply_masking_strategy(config: Any, inputs: Any, position_score_fn: Any = None) -> Any:
    """Apply a masking strategy to tool inputs, skipping if already masked.

    Intended to be called from a sample config's ``preprocess()`` hook.
    Reads ``config.masking_strategy`` and applies it to ``inputs.sequences``.
    If any input sequence already contains ``MASK_TOKEN`` (``_``), the
    sequences are left unchanged.

    Warns when sequences are already masked but a non-default strategy
    was explicitly provided.

    Args:
        config (Any): The tool's config object (has ``.masking_strategy`` and ``.seed``).
        inputs (Any): A ``BaseToolInput``-like object with a ``sequences`` field
            and a ``model_copy(update=...)`` method (Pydantic model).
        position_score_fn (Any): Optional callable that takes sequences and returns
            per-position scores. Built by ``build_position_score_fn()`` in
            the tool's ``preprocess()`` hook.

    Returns:
        Any: The (possibly updated) inputs object.
    """
    strategy = config.masking_strategy

    already_masked = any(MASK_TOKEN in seq for seq in inputs.sequences)
    if already_masked:
        if strategy != type(strategy)():
            logger.warning(
                "Sequences already contain mask tokens ('_'); ignoring "
                "custom masking_strategy. Remove '_' tokens from input "
                "sequences to use the masking strategy, or omit "
                "masking_strategy to silence this warning."
            )
    else:
        inputs = inputs.model_copy(
            update={
                "sequences": strategy.mask(
                    inputs.sequences,
                    position_score_fn=position_score_fn,
                    seed=config.seed,
                )
            }
        )
    return inputs


# ============================================================================
# Masking strategies: RandomMaskingStrategy (base) + MaskingStrategy (model-informed)
# ============================================================================


class RandomMaskingStrategy(BaseModel):
    """Random masking strategy: select positions to mask for re-design.

    The base (model-free) tier: positions are chosen uniformly at random.
    Sampling tools with no scoring model (e.g. the mutagenesis samplers) use
    this type directly. Tools with a model use :class:`MaskingStrategy`, which
    adds the model-informed methods.

    How many positions to mask (mutually exclusive, pick one):

    - ``num_mutations``: exact count (e.g. ``num_mutations=3`` → mask 3).
    - ``mask_fraction``: proportion of designable positions
      (e.g. ``mask_fraction=0.15`` → mask ~15%).
    - Neither: masks ~30% of designable positions (default).

    Attributes:
        method (Literal['random']): Position-selection method. Always
            ``"random"`` for this tier (uniform selection, no model).
        num_mutations (int | None): Exact number of positions to mask per sequence.
        mask_fraction (float | None): Fraction of designable positions to mask
            (e.g. 0.15 for ~15%).
        fixed_positions (list[int] | None): 1-indexed positions that must NOT be
            masked. Applied uniformly to all sequences.

    Examples::

        RandomMaskingStrategy().mask(["MKTLLIFLA"])
        # → ["M_TLLIFLA"]  (random, ~30%)

        RandomMaskingStrategy(mask_fraction=0.5, fixed_positions=[1]).mask(["MKTLLIFLA"])
        # → ["M_TL_I__A"]  (roughly half of designable positions masked)

        ``mask_fraction`` is applied over designable positions after fixed
        positions are excluded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # -- Private state (excluded from serialization, equality, schema) ---------
    _masker: Masker | None = PrivateAttr(default=None)

    # -- Scoring method --------------------------------------------------------
    method: Literal["random"] = ConfigField(
        default="random",
        title="Scoring Method",
        description="Position scoring. 'random': uniform selection (no model needed).",
    )

    # -- How many positions to mask (set one or neither, not both) -------------
    num_mutations: int | None = ConfigField(
        default=None,
        ge=1,
        title="Num Mutations",
        description="Exact number of positions to mask per sequence.",
    )
    mask_fraction: float | None = ConfigField(
        default=None,
        gt=0.0,
        le=1.0,
        title="Mask Fraction",
        description="Fraction of designable positions to mask (e.g. 0.15 for ~15%).",
    )

    # -- Which positions to protect from masking -------------------------------
    fixed_positions: list[int] | None = ConfigField(
        default=None,
        title="Fixed Positions",
        description="1-indexed positions that must NOT be masked. Applied uniformly to all sequences.",
    )

    # -- Validators ------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _validate_fixed_positions(cls, data: Any) -> Any:
        """Coerce [] to None; otherwise validate via the shared 1-indexed positions helper."""
        if isinstance(data, dict):
            positions = data.get("fixed_positions")
            if isinstance(positions, list):
                if not positions:
                    data = {**data, "fixed_positions": None}
                else:
                    data = {
                        **data,
                        "fixed_positions": validate_positions_list(
                            positions,
                            label="fixed_positions",
                            logger_obj=logger,
                        ),
                    }
        return data

    @model_validator(mode="after")
    def _validate_mutation_spec(self) -> Any:
        """Ensure num_mutations and mask_fraction aren't both set."""
        _validate_mutation_spec(self.num_mutations, self.mask_fraction)
        return self

    # -- Core API --------------------------------------------------------------

    def mask(
        self,
        sequences: list[str],
        position_score_fn: Callable[..., Any] | None = None,
        seed: int | None = None,
    ) -> list[str]:
        """Apply the masking strategy to a batch of sequences.

        The masker is created on first call and reused on subsequent calls,
        allowing stateful maskers to accumulate information across calls.

        Args:
            sequences (list[str]): Protein sequences to mask.
            position_score_fn (Callable[..., Any] | None): Callable that takes sequences and returns
                per-position scores. Required for model-based methods
                (entropy, max-logit); ignored for ``random``. When called from a
                tool's ``preprocess()``, this is built by
                ``build_position_score_fn()``. When called standalone with a
                model-based method, it must be supplied explicitly.
            seed (int | None): Random seed for reproducible masking. Creates a seeded
                RandomState for position selection. If None, uses global numpy RNG.

        Returns:
            list[str]: List of masked sequences with ``_`` at selected positions.
        """
        # Lazy-init the masker (persists for the lifetime of this strategy)
        if self._masker is None:
            object.__setattr__(
                self,
                "_masker",
                MASKERS[self.method](strategy=self),
            )

        # Model-based methods need a scorer; standalone has no model to build one from.
        if position_score_fn is None and MASKERS[self.method].requires is not None:
            raise ValueError(
                f"method='{self.method}' requires a position_score_fn. Run it through a "
                f"sampling tool that supplies one (its preprocess builds it), or pass "
                f"position_score_fn= explicitly when calling mask() standalone."
            )

        assert self._masker is not None
        all_scores = self._masker.score(sequences, position_score_fn=position_score_fn)

        # Create seeded RNG if seed is provided
        rng = np.random.RandomState(seed) if seed is not None else None

        # temperature exists only on the model-informed tier (random scores are uniform).
        temperature = getattr(self, "temperature", 1.0)

        results = []
        for i, seq in enumerate(sequences):
            mutable = mutable_mask(seq, self.fixed_positions)
            eligible = [j for j, m in enumerate(mutable) if m]
            count = _resolve_count(
                self.num_mutations,
                self.mask_fraction,
                len(eligible),
            )
            validate_enough_mutable(seq, mutable, count, i)

            scores = [all_scores[i][j] / temperature for j in eligible]
            chosen = weighted_sample(eligible, scores, count, rng=rng)
            results.append(apply_mask(seq, chosen))
        return results


class MaskingStrategy(RandomMaskingStrategy):
    """Model-informed masking strategy: select positions to mask for re-design.

    Extends :class:`RandomMaskingStrategy` with scoring methods that consume
    per-position model logits. Sampling tools that have a scoring model (e.g.
    ESM2/ESM3 sampling) type their ``masking_strategy`` field to this class.
    The logits always come from the **sampling tool's own model** — masking and
    resampling use the same model and checkpoint (see
    :func:`build_position_score_fn`).

    The ``method`` field selects the scoring behavior:

    - ``"random"`` (default): uniform random selection (no model needed).
    - ``"entropy"``: score by Shannon entropy from model logits (high
      uncertainty → more likely to mask).
    - ``"max-logit"``: score by negated max-logit (low model confidence
      → more likely to mask).

    Attributes:
        method (MaskingMethod): Scoring method for position selection.
            ``"random"``: uniform random, ``"entropy"``: highest model uncertainty,
            ``"max-logit"``: lowest model confidence.
        temperature (float): Temperature for position selection. < 1.0 is greedy,
            1.0 uses scores as-is, > 1.0 is more uniform. Only affects
            model-based methods.

    Examples::

        MaskingStrategy(method="entropy", num_mutations=3)  # used via a sampling tool
        # masks the 3 highest-entropy positions, scored by the tool's model
    """

    # Deliberate widening of the base's Literal["random"]; mypy flags field-override as invariant.
    method: MaskingMethod = ConfigField(  # type: ignore[assignment]
        default="random",
        title="Scoring Method",
        description="Position scoring: 'random' (uniform), 'entropy' (most uncertain), 'max-logit' (low confidence)",
    )

    # -- Score temperature (only relevant for entropy / max-logit) -------------
    temperature: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Selection Temperature",
        description="Position-selection temperature: <1 greedy, 1 use scores as-is, >1 more uniform.",
    )


# Drift guard: each tier's exposed methods must equal what the capability model derives.
assert set(get_args(RandomMaskingStrategy.model_fields["method"].annotation)) == set(compatible_methods(frozenset())), (
    "RandomMaskingStrategy.method must match compatible_methods(frozenset())"
)
assert set(get_args(MaskingStrategy.model_fields["method"].annotation)) == set(
    compatible_methods(frozenset({MaskingInput.LOGITS}))
), "MaskingStrategy.method must match compatible_methods({LOGITS})"


# ============================================================================
# build_position_score_fn: shared callable factory for tool preprocess hooks
# ============================================================================


def build_position_score_fn(
    sampling_model: str,
    masking_strategy: RandomMaskingStrategy,
    device: str,
    sampling_checkpoint: str | None = None,
) -> Callable[..., Any] | None:
    """Build a position_score_fn callable for model-based masking methods.

    Called from a sampling tool's ``preprocess()`` to create a function
    that the masker can call to obtain per-position logits. The callable
    dispatches through the standard tool path, so it benefits from
    ``ToolInstance.persist()`` worker reuse automatically.

    Scoring always uses the **sampling tool's own** model and checkpoint, so
    masking and resampling stay on the same weights. Passing ``sampling_checkpoint``
    keeps the shared persistent worker from restarting between the preprocess
    scoring call and the main sampling call (both are ``reload_on_change`` on
    ``model_checkpoint``).

    Args:
        sampling_model (str): The sampling tool's model name (e.g. ``"esm2"``).
            Selects which embeddings tool provides the logits.
        masking_strategy (RandomMaskingStrategy): The strategy object; its
            ``method`` decides whether logits are needed at all.
        device (str): Device string from the sampling config (e.g. ``"cuda"``,
            ``"cuda:1"``). Passed through to the embeddings tool.
        sampling_checkpoint (str | None): The sampling tool's selected checkpoint,
            threaded so scoring uses identical weights. ``None`` uses the
            embeddings tool's default.

    Returns:
        Callable[..., Any] | None: A callable ``(sequences: list[str]) -> list[list[list[float]]]``
            that returns per-position logits, or ``None`` if the masking
            method doesn't need logits (e.g. random).
    """
    masker_cls = MASKERS[masking_strategy.method]
    if masker_cls.requires is None:
        return None

    config_kwargs: dict[str, Any] = {"device": device, "return_logits": True}
    if sampling_checkpoint is not None:
        config_kwargs["model_checkpoint"] = sampling_checkpoint

    if sampling_model == "esm2":

        def position_score_fn(sequences: list[str]) -> list[list[list[float]]]:
            from proto_tools.tools.masked_models.esm2.esm2_embeddings import (
                ESM2EmbeddingsConfig,
                ESM2EmbeddingsInput,
                run_esm2_embeddings,
            )

            result = run_esm2_embeddings(
                ESM2EmbeddingsInput(sequences=sequences),
                ESM2EmbeddingsConfig(**config_kwargs),
            )
            return [r.logits for r in result.results]

        return position_score_fn

    if sampling_model == "esm3":

        def position_score_fn(sequences: list[str]) -> list[list[list[float]]]:
            from proto_tools.tools.masked_models.esm3.esm3_embeddings import (
                ESM3EmbeddingsConfig,
                ESM3EmbeddingsInput,
                run_esm3_embeddings,
            )

            result = run_esm3_embeddings(
                ESM3EmbeddingsInput(sequences=sequences),
                ESM3EmbeddingsConfig(**config_kwargs),
            )
            return [r.logits for r in result.results]

        return position_score_fn

    raise ValueError(f"Unknown model: {sampling_model}")
