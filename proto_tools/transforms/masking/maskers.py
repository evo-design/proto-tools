"""proto_tools/transforms/masking/maskers.py.

Each masker has:
- ``requires``: which external input it needs to score positions
  (``None`` = no input needed, e.g. uniform random)
- ``__init__(strategy)``: stores the strategy reference
- ``score(sequences, position_score_fn)``: returns per-position scores

The ``MaskingMethod`` Literal and ``MASKERS`` dict are the single source of
truth for valid method names. ``Masker.requires`` plus a sampling tool's
declared ``masking_inputs`` determine which methods that tool supports
(see :func:`compatible_methods`).
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any, ClassVar, Literal, get_args

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# MaskingInput: external inputs a masker may need to score positions
# ============================================================================


class MaskingInput(str, Enum):
    """An external input a masker needs to score positions.

    A sampling tool declares the set it can supply via ``masking_inputs``;
    a masker declares the one it needs via :attr:`Masker.requires`. A method
    is available on a tool iff the tool provides what the masker requires
    (see :func:`compatible_methods`). Extend this enum to add new
    model-/data-informed maskers (e.g. ``MSA``, ``STRUCTURE``).
    """

    LOGITS = "logits"  # per-position model logits


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
        requires (MaskingInput | None): The external input this masker needs to
            score positions (e.g. ``MaskingInput.LOGITS``), or ``None`` if it
            needs no input (e.g. uniform random). A sampling tool exposes this
            masker's method only if its ``masking_inputs`` provides ``requires``.

    Instance attributes:
        strategy: The masking strategy (:class:`RandomMaskingStrategy` or a
            subclass) that owns this masker. Provides access to user-configured
            fields (method, num_mutations, etc.).

    To implement a new masker:

    1. Subclass ``Masker``
    2. Set ``requires`` (``None`` if no input needed; else a ``MaskingInput``)
    3. Implement ``score()``
    4. Register in the ``MASKERS`` dict and add the method name to
       ``MaskingMethod``
    5. Have sampling tools that can supply ``requires`` add it to their
       ``masking_inputs`` (and widen their ``masking_strategy`` field type)

    Example::

        class MyCustomMasker(Masker):
            requires = None  # no input needed

            def score(self, sequences, position_score_fn=None):
                # Return higher scores for positions you want masked
                return [[1.0 if c == "A" else 0.0 for c in seq] for seq in sequences]
    """

    requires: ClassVar[MaskingInput | None] = None

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

    requires = None

    def score(
        self,
        sequences: list[str],
        position_score_fn: Callable[..., Any] | None = None,  # noqa: ARG002 — required by abstract Masker interface
    ) -> list[list[float]]:
        """Score all positions equally (uniform zero scores, no model)."""
        return [[0.0] * len(seq) for seq in sequences]


class EntropyMasker(Masker):
    """Score positions by Shannon entropy. High uncertainty leads to masking."""

    requires = MaskingInput.LOGITS

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

    requires = MaskingInput.LOGITS

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


def compatible_methods(provides: "frozenset[MaskingInput]") -> list[str]:
    """Return the masking methods available to a tool that supplies ``provides``.

    A method is available iff its masker needs no input (``requires is None``,
    e.g. ``random``) or the tool supplies the required input. This is the
    single derivation behind both the per-tool ``masking_strategy`` field type
    and any compatibility introspection.

    Args:
        provides (frozenset[MaskingInput]): The inputs the sampling tool can
            supply (its ``masking_inputs``). Empty for tools with no model.

    Returns:
        list[str]: Method names from ``MASKERS``, in registry order.
    """
    methods: list[str] = []
    for name, masker in MASKERS.items():
        req = masker.requires
        if req is None or req in provides:
            methods.append(name)
    return methods
