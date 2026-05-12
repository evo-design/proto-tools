"""proto_tools/transforms/masking/base.py.

Masking strategies for masked language model sampling.
"""

import logging
import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from proto_tools.transforms.masking.maskers import MASKERS, Masker, MaskingMethod
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
        if strategy != MaskingStrategy():
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
# MaskingStrategy: single class for all masking strategies
# ============================================================================


class MaskingStrategy(BaseModel):
    """Masking strategy: select positions to mask for re-design.

    The ``method`` field selects the scoring behavior:

    - ``"random"`` (default): uniform random selection.
    - ``"entropy"``: score by Shannon entropy from model logits (high
      uncertainty → more likely to mask).
    - ``"max-logit"``: score by negated max-logit (low model confidence
      → more likely to mask).

    How many positions to mask (mutually exclusive, pick one):

    - ``num_mutations``: exact count (e.g. ``num_mutations=3`` → mask 3).
    - ``mask_fraction``: proportion of designable positions
      (e.g. ``mask_fraction=0.15`` → mask ~15%).
    - Neither: masks ~30% of designable positions (default).

    Attributes:
        method (MaskingMethod): Scoring method for position selection.
            ``"random"``: uniform random, ``"entropy"``: highest model uncertainty,
            ``"max-logit"``: lowest model confidence.
        num_mutations (int | None): Exact number of positions to mask per sequence.
        mask_fraction (float | None): Fraction of designable positions to mask
            (e.g. 0.15 for ~15%).
        fixed_positions (list[int] | None): 1-indexed positions that must NOT be
            masked. Applied uniformly to all sequences.
        temperature (float): Temperature for position selection. < 1.0 is greedy,
            1.0 uses scores as-is, > 1.0 is more uniform.
        model_name (Literal['esm2', 'esm3'] | None): Which masked model to use
            for scoring. Defaults to the sampling tool's model when unset.
        model_checkpoint (str | None): Model checkpoint override (uses tool default
            if None).

    Examples::

        MaskingStrategy().mask(["MKTLLIFLA"])
        # → ["M_TLLIFLA"]  (random, ~30%)

        MaskingStrategy(method="entropy", model_name="esm2", num_mutations=3).mask(["MKTLLIFLA"])
        # → ["MK_LL_F_A"]  (3 highest-entropy positions)

        MaskingStrategy(mask_fraction=0.5, fixed_positions=[1]).mask(["MKTLLIFLA"])
        # → ["M_TL_I__A"]  (roughly half of designable positions masked)

        ``mask_fraction`` is applied over designable positions after fixed
        positions are excluded.
    """

    model_config = ConfigDict(frozen=True)

    # -- Private state (excluded from serialization, equality, schema) ---------
    _masker: Masker | None = PrivateAttr(default=None)

    # -- Scoring method --------------------------------------------------------
    method: MaskingMethod = ConfigField(
        default="random",
        description=(
            "Scoring method for position selection. 'random': uniform random. "
            "'entropy': highest model uncertainty. 'max-logit': lowest model confidence."
        ),
    )

    # -- How many positions to mask (set one or neither, not both) -------------
    num_mutations: int | None = ConfigField(
        default=None,
        ge=1,
        xor_group="mask_amount",
        description="Exact number of positions to mask per sequence.",
    )
    mask_fraction: float | None = ConfigField(
        default=None,
        gt=0.0,
        le=1.0,
        xor_group="mask_amount",
        description="Fraction of designable positions to mask (e.g. 0.15 for ~15%).",
    )

    # -- Which positions to protect from masking -------------------------------
    fixed_positions: list[int] | None = ConfigField(
        default=None,
        description="1-indexed positions that must NOT be masked. Applied uniformly to all sequences.",
    )

    # -- Score temperature (only relevant for entropy / max-logit) --------------
    temperature: float = ConfigField(
        default=1.0,
        gt=0.0,
        description=(
            "Temperature for position selection. < 1.0: greedy (prefer "
            "highest-scored positions). = 1.0: use scores as-is. > 1.0: "
            "more uniform (ignore score differences)."
        ),
        depends_on={"method": ["entropy", "max-logit"]},
        advanced=True,
    )

    # -- Model-based method fields (only relevant for entropy / max-logit) -----
    model_name: Literal["esm2", "esm3"] | None = ConfigField(
        default=None,
        description="Which masked model to use for scoring. Defaults to the sampling tool's model when unset.",
        depends_on={"method": ["entropy", "max-logit"]},
        advanced=True,
    )
    model_checkpoint: str | None = ConfigField(
        default=None,
        description="Model checkpoint override (uses tool default if None).",
        depends_on={"method": ["entropy", "max-logit"]},
        hidden=True,
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

    @model_validator(mode="after")
    def _validate_method_fields(self) -> Any:
        """Validate model fields against the masker class."""
        masker_cls = MASKERS[self.method]
        if masker_cls.supported_models is None:
            if self.model_name is not None or self.model_checkpoint is not None:
                warnings.warn(f"model_name/model_checkpoint are ignored for method='{self.method}'", stacklevel=2)
        else:
            if self.model_name is not None and self.model_name not in masker_cls.supported_models:
                raise ValueError(
                    f"model '{self.model_name}' not supported for "
                    f"method='{self.method}'. "
                    f"Supported: {masker_cls.supported_models}"
                )
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
                (entropy, max-logit). When called from a tool's
                ``preprocess()``, this is built by
                ``build_position_score_fn()``. When called standalone,
                auto-built from ``model_name`` / ``model_checkpoint``
                if not provided.
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

        # Auto-build position_score_fn for standalone usage
        if position_score_fn is None and MASKERS[self.method].supported_models is not None:
            if self.model_name is None:
                raise ValueError(
                    f"model_name required for method='{self.method}' when "
                    f"used standalone (e.g. MaskingStrategy(method="
                    f"'{self.method}', model_name='esm2'))"
                )
            from proto_tools.utils.device import number_of_visible_gpus

            device = "cuda" if number_of_visible_gpus() > 0 else "cpu"
            position_score_fn = build_position_score_fn(self.model_name, self, device=device)

        assert self._masker is not None
        all_scores = self._masker.score(sequences, position_score_fn=position_score_fn)

        # Create seeded RNG if seed is provided
        rng = np.random.RandomState(seed) if seed is not None else None

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

            scores = [all_scores[i][j] / self.temperature for j in eligible]
            chosen = weighted_sample(eligible, scores, count, rng=rng)
            results.append(apply_mask(seq, chosen))
        return results


# ============================================================================
# build_position_score_fn: shared callable factory for tool preprocess hooks
# ============================================================================


def build_position_score_fn(
    sampling_model: str,
    masking_strategy: MaskingStrategy,
    device: str,
) -> Callable[..., Any] | None:
    """Build a position_score_fn callable for model-based masking methods.

    Called from a sampling tool's ``preprocess()`` to create a function
    that the masker can call to obtain per-position logits. The callable
    dispatches through the standard tool path, so it benefits from
    ``ToolInstance.persist()`` worker reuse automatically.

    Args:
        sampling_model (str): The sampling tool's model name (e.g. ``"esm2"``).
            Used as default when ``masking_strategy.model_name`` is unset.
        masking_strategy (MaskingStrategy): The strategy object (provides ``model_name``
            and ``model_checkpoint`` overrides).
        device (str): Device string from the sampling config (e.g. ``"cuda"``,
            ``"cuda:1"``). Passed through to the embeddings tool.

    Returns:
        Callable[..., Any] | None: A callable ``(sequences: list[str]) -> list[list[list[float]]]``
            that returns per-position logits, or ``None`` if the masking
            method doesn't need logits (e.g. random).
    """
    masker_cls = MASKERS[masking_strategy.method]
    if masker_cls.supported_models is None:
        return None

    model_name = masking_strategy.model_name or sampling_model
    config_kwargs: dict[str, Any] = {"device": device, "return_logits": True}
    if masking_strategy.model_checkpoint is not None:
        config_kwargs["model_checkpoint"] = masking_strategy.model_checkpoint

    if model_name == "esm2":

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

    if model_name == "esm3":

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

    raise ValueError(f"Unknown model: {model_name}")
