"""proto_tools/transforms/masking/maskers.py.

Each masker has:
- ``supported_models``: which models it works with (None = no model needed)
- ``__init__(strategy)``: stores the strategy reference
- ``score(sequences, position_score_fn)``: returns per-position scores

The ``MaskingMethod`` Literal and ``MASKERS`` dict are the single source of
truth for valid method names.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Literal, get_args

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# MaskingMethod: single source of truth for valid method names
# ============================================================================

MaskingMethod = Literal["random", "entropy", "max-logit"]


# ============================================================================
# Masker ABC
# ============================================================================


class Masker(ABC):
    """Abstract base class for masking strategies.

    Subclasses must implement :meth:`score` to return per-position scores
    for a batch of sequences. Higher scores mean the position is more
    likely to be masked.

    Attributes:
        supported_models (list[str] | None): List of model names this masker works with
            (e.g. ``["esm2", "esm3"]``), or ``None`` if no model is needed.

    Instance attributes:
        strategy: The :class:`MaskingStrategy` that owns this masker.
            Provides access to user-configured fields (method, model_name,
            model_checkpoint, etc.).

    To implement a new masker:

    1. Subclass ``Masker``
    2. Set ``supported_models`` (``None`` if no model needed)
    3. Implement ``score()``
    4. Register in the ``MASKERS`` dict and add the method name to
       ``MaskingMethod``

    Example::

        class MyCustomMasker(Masker):
            supported_models = None  # no model needed

            def score(self, sequences, position_score_fn=None):
                # Return higher scores for positions you want masked
                return [[1.0 if c == "A" else 0.0 for c in seq] for seq in sequences]
    """

    supported_models: list[str] | None = None

    def __init__(self, strategy: Any) -> None:
        """Store the owning MaskingStrategy."""
        self.strategy = strategy

    @abstractmethod
    def score(
        self,
        sequences: list[str],
        position_score_fn: Callable[..., Any] | None = None,
    ) -> list[list[float]]:
        """Score all positions for all sequences.

        Args:
            sequences (list[str]): Protein sequences to score.
            position_score_fn (Callable[..., Any] | None): Callable that takes a list of sequences
                and returns per-position scores. Required for model-based
                maskers; ``None`` for maskers that don't use a model.

        Returns:
            list[list[float]]: List of per-position scores, one list per sequence.
                Higher = more likely to mask.
        """


# ============================================================================
# Masker implementations
# ============================================================================


class RandomMasker(Masker):
    """Uniform random selection; all positions scored equally."""

    supported_models = None

    def score(
        self,
        sequences: list[str],
        position_score_fn: Callable[..., Any] | None = None,  # noqa: ARG002 — required by abstract Masker interface
    ) -> list[list[float]]:
        """Score all positions equally (uniform zero scores, no model)."""
        return [[0.0] * len(seq) for seq in sequences]


class EntropyMasker(Masker):
    """Score positions by Shannon entropy. High uncertainty leads to masking."""

    supported_models: ClassVar[list[str]] = ["esm2", "esm3"]  # type: ignore[misc]

    def score(
        self,
        sequences: list[str],
        position_score_fn: Callable[..., Any] | None = None,
    ) -> list[list[float]]:
        """Compute per-position Shannon entropy from model logits."""
        logits = position_score_fn(sequences)  # type: ignore[misc]
        all_scores = []
        for seq_logits in logits:
            # (seq_len, vocab_size)  # noqa: ERA001 -- tensor shape annotation
            arr = np.array(seq_logits)
            # Numerically stable softmax
            arr = arr - arr.max(axis=-1, keepdims=True)
            probs = np.exp(arr)
            probs /= probs.sum(axis=-1, keepdims=True)
            # Shannon entropy: H = -sum(p * log(p))
            entropies = -np.sum(probs * np.log(np.maximum(probs, 1e-30)), axis=-1)
            all_scores.append(entropies.tolist())
        return all_scores


class MaxLogitMasker(Masker):
    """Score positions by negated max-logit. Low confidence leads to masking."""

    supported_models: ClassVar[list[str]] = ["esm2", "esm3"]  # type: ignore[misc]

    def score(
        self,
        sequences: list[str],
        position_score_fn: Callable[..., Any] | None = None,
    ) -> list[list[float]]:
        """Compute negated max-logit per position from model logits."""
        logits = position_score_fn(sequences)  # type: ignore[misc]
        all_scores = []
        for seq_logits in logits:
            arr = np.array(seq_logits)
            # Negate so low confidence → high score → more likely to mask
            all_scores.append((-arr.max(axis=-1)).tolist())
        return all_scores


# ============================================================================
# MASKERS: the lookup table
# ============================================================================

MASKERS: dict[str, type[Masker]] = {
    "random": RandomMasker,
    "entropy": EntropyMasker,
    "max-logit": MaxLogitMasker,
}

assert set(MASKERS.keys()) == set(get_args(MaskingMethod)), (  # noqa: S101 -- module-level registry consistency check
    f"MASKERS keys {set(MASKERS.keys())} don't match MaskingMethod values {set(get_args(MaskingMethod))}"
)
