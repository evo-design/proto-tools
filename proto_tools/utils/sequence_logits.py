"""proto_tools/utils/sequence_logits.py.

Relaxed sequence representations shared by the gradient tools.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import ClassVar

from pydantic import BaseModel, RootModel, model_validator
from typing_extensions import Self

from proto_tools.utils.sequence import DNA_NUCLEOTIDES, PROTEIN_AMINO_ACIDS

# Shapes below use L = sequence positions (rows) and V = vocabulary size (columns).

# Absolute tolerance on the row-sum check in probability mode (``temperature=None``).
PROBABILITY_SUM_TOLERANCE = 1e-6


class SequenceLogits(RootModel[list[list[float]]]):
    """A relaxed ``(L, V)`` sequence state over one fixed vocabulary.

    Subclasses set ``VOCAB`` to the column symbols, so ``V`` is its length and ``L`` is the
    number of rows. Construction validates shape, rectangularity, and finiteness. The
    probability-simplex rules are separate, since they apply only in probability mode; a
    gradient input runs them through :meth:`validate_probabilities` when its ``temperature``
    is ``None``.

    Parses from and serializes to ordinary nested lists, so the JSON request shape callers
    already send is unchanged.
    """

    VOCAB: ClassVar[Sequence[str]] = ()

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        """Check for a non-empty, rectangular ``(L, V)`` grid of finite numbers."""
        # root: (L, V) -> rejects empty, ragged, and non-finite matrices.
        vocab = type(self).VOCAB
        if not vocab:
            raise ValueError(
                f"{type(self).__name__} defines no VOCAB; use a vocabulary-specific SequenceLogits subclass"
            )
        if not self.root:
            raise ValueError("logits must contain at least one position")
        width = len(vocab)  # V
        for index, row in enumerate(self.root):
            if len(row) != width:
                raise ValueError(f"logits row {index + 1} must have {width} columns, got {len(row)}")
            if not all(math.isfinite(value) for value in row):
                raise ValueError(f"logits row {index + 1} must contain only finite numbers")
        return self

    def __len__(self) -> int:
        """Return the number of positions."""
        # -> L
        return len(self.root)

    def __iter__(self) -> Iterator[list[float]]:  # type: ignore[override]
        """Iterate over position rows, rather than Pydantic's ``(field, value)`` pairs."""
        # -> L rows, each (V,)
        return iter(self.root)

    def __getitem__(self, index: int) -> list[float]:
        """Return one position row."""
        # -> (V,)
        return self.root[index]

    @property
    def shape(self) -> tuple[int, int]:
        """Return the ``(L, V)`` matrix shape."""
        # -> (L, V)
        return len(self.root), len(type(self).VOCAB)

    @property
    def vocab(self) -> list[str]:
        """Return the column symbols, in column order."""
        # -> V symbols, column-ordered
        return list(type(self).VOCAB)

    def validate_allowed_mask(self, allowed: list[list[bool]], *, label: str = "logits") -> None:
        """Check an allowed-token mask is an ``(L, V)`` boolean grid matching this matrix.

        Without the width check a short or empty mask row reads as all-``False``, which the
        validators would take for a deliberate fixed-context marker and skip.

        Args:
            allowed (list[list[bool]]): Candidate mask of permitted columns per position.
            label (str): Field path used in error messages.

        Raises:
            ValueError: If the mask does not have one ``V``-wide row per position.
        """
        # allowed: must be exactly (L, V) to be interpretable at all.
        length, width = self.shape
        if len(allowed) != length:
            raise ValueError(f"{label}: allowed-token mask must have {length} rows, got {len(allowed)}")
        for index, row in enumerate(allowed):
            if len(row) != width:
                raise ValueError(
                    f"{label}: allowed-token mask row {index + 1} must have {width} columns, got {len(row)}"
                )

    def validate_fixed_rows(self, *, allowed: list[list[bool]], label: str = "logits") -> None:
        """Require rows with no permitted column to be exactly zero.

        A row whose mask is entirely ``False`` is fixed context: it is not a design variable, so
        it carries no state in either temperature mode.

        Args:
            allowed (list[list[bool]]): ``(L, V)`` mask of permitted columns per position.
            label (str): Field path used in error messages.

        Raises:
            ValueError: If the mask is not ``(L, V)``, or a fixed-context row holds a nonzero value.
        """
        # allowed: (L, V) bool -> every all-False row of root must be exactly zero.
        self.validate_allowed_mask(allowed, label=label)
        for index, (row, mask) in enumerate(zip(self.root, allowed, strict=True)):
            if not any(mask) and any(value != 0.0 for value in row):
                raise ValueError(f"Fixed context row {index + 1} must contain zeros")

    def validate_probabilities(self, *, allowed: list[list[bool]] | None = None, label: str = "logits") -> None:
        """Require every active row to be a probability distribution over its permitted columns.

        Rows whose mask is entirely ``False`` are fixed context and are skipped; see
        :meth:`validate_fixed_rows`, which covers them in both temperature modes.

        Args:
            allowed (list[list[bool]] | None): ``(L, V)`` mask of permitted columns per position.
                ``None`` permits every column at every position.
            label (str): Field path used in error messages.

        Raises:
            ValueError: If the mask is not ``(L, V)``, or an active row is negative, carries mass
                outside its permitted columns, or does not sum to 1.0 within
                :data:`PROBABILITY_SUM_TOLERANCE`.
        """
        # allowed: (L, V) bool | None -> each active row must be a simplex over its permitted columns.
        if allowed is not None:
            self.validate_allowed_mask(allowed, label=label)
        for index, row in enumerate(self.root):
            mask = allowed[index] if allowed is not None else None  # (V,) or None
            if mask is not None and not any(mask):
                continue
            position = index + 1
            if any(value < 0.0 for value in row):
                raise ValueError(f"{label} row {position} must be nonnegative when temperature is None")
            if mask is not None and any(
                value != 0.0 for value, permitted in zip(row, mask, strict=True) if not permitted
            ):
                raise ValueError(
                    f"{label} row {position} must leave its non-permitted columns at zero when temperature is None"
                )
            # rel_tol is pinned to 0 so the check is purely absolute, as the constant says.
            if not math.isclose(sum(row), 1.0, rel_tol=0.0, abs_tol=PROBABILITY_SUM_TOLERANCE):
                raise ValueError(f"{label} row {position} must be a probability distribution when temperature is None")

    @classmethod
    def from_probabilities(cls, probabilities: Sequence[Sequence[float]], *, temperature: float = 1.0) -> Self:
        """Build logits that a softmax at ``temperature`` turns back into ``probabilities``.

        Returns ``temperature * log(p)``, which ``softmax(logits / temperature)`` inverts exactly.

        This is the numerical primitive. Prefer a gradient input's own ``from_probabilities()``,
        which also stores the matching ``temperature`` so the conversion and the reading cannot
        disagree; pairing this result with a different temperature reshapes the distribution.

        Args:
            probabilities (Sequence[Sequence[float]]): ``(L, V)`` rows, each a strictly positive
                probability distribution.
            temperature (float): Positive, finite temperature the result will be paired with.

        Returns:
            Self: Logits reproducing ``probabilities`` under ``softmax(logits / temperature)``.

        Raises:
            ValueError: If ``temperature`` is not positive and finite, or a row is not a strictly
                positive probability distribution.
        """
        # probabilities: (L, V) simplex -> (L, V) logits, T * log(p).
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f"temperature must be positive and finite, got {temperature!r}")
        distribution = cls([[float(value) for value in row] for row in probabilities])  # (L, V), validated below
        distribution.validate_probabilities(label="probabilities")
        for index, row in enumerate(distribution.root):
            # log(0) is -inf, which cannot cross JSON; refuse rather than floor the distribution.
            zeros = [column for column, value in enumerate(row) if value == 0.0]
            if zeros:
                raise ValueError(
                    f"probabilities row {index + 1} has exact zeros at columns {zeros}, which map to infinite "
                    "logits. Pass the distribution directly with temperature=None to preserve exact zeros; "
                    "permanently forbidden tokens are to be expressed through a masked-softmax interface."
                )
        return cls([[temperature * math.log(value) for value in row] for row in distribution.root])


class ProteinSequenceLogits(SequenceLogits):
    """A relaxed ``(L, 20)`` protein state in canonical amino-acid order ``ACDEFGHIKLMNPQRSTVWY``."""

    VOCAB: ClassVar[Sequence[str]] = PROTEIN_AMINO_ACIDS


class NucleotideSequenceLogits(SequenceLogits):
    """A relaxed ``(L, 4)`` DNA state in ``A,C,G,T`` order."""

    VOCAB: ClassVar[Sequence[str]] = DNA_NUCLEOTIDES


def iter_sequence_logits(model: BaseModel) -> Iterator[tuple[str, SequenceLogits]]:
    """Yield ``(path, matrix)`` for every relaxed matrix reachable from a model's fields.

    Walks direct :class:`SequenceLogits` fields, lists of them, and nested models that expose
    their own ``iter_sequence_logits()``. Other nested models are not traversed, so an input
    carrying a ``Structure`` or similar is not walked needlessly.

    Args:
        model (BaseModel): Model whose fields to inspect.

    Yields:
        tuple[str, SequenceLogits]: Dotted field path (``"logits"``, ``"antibody.heavy_chain"``,
            ``"logits[0]"``) and the matrix found there.
    """
    # -> one (path, (L, V) matrix) pair per relaxed field, recursing into bundles only.
    for name in type(model).model_fields:
        value = getattr(model, name, None)
        if isinstance(value, SequenceLogits):
            yield name, value
        elif callable(getattr(value, "iter_sequence_logits", None)):
            # A composite field declares its own matrices, so nested models opt in rather than
            # every BaseModel being walked.
            for path, matrix in value.iter_sequence_logits():
                yield f"{name}.{path}", matrix
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, SequenceLogits):
                    yield f"{name}[{index}]", item


__all__ = [
    "PROBABILITY_SUM_TOLERANCE",
    "NucleotideSequenceLogits",
    "ProteinSequenceLogits",
    "SequenceLogits",
    "iter_sequence_logits",
]
