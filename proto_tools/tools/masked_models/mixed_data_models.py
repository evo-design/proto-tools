"""Shared contracts for mixed protein/DNA masked language models."""

import json
import math
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from standalone_helpers.mixed_sequence import one_hot_mixed_logits, tokenize_mixed_sequence
from standalone_helpers.vocabulary import (
    AMBIGUOUS_PROTEIN_TOKENS,
    DNA_TOKENS,
    MIXED_VOCAB,
    PROTEIN_TOKENS,
)

from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelEmbeddingsOutput,
    MaskedModelSample,
    MaskedModelSampleOutput,
    MaskedModelScoringMetrics,
    MaskedModelScoringOutput,
    SequenceEmbedding,
)
from proto_tools.transforms.masking import MASK_TOKEN
from proto_tools.utils import (
    BaseToolInput,
    GradientInput,
    GradientOutput,
    InputField,
    SequenceLogits,
    coerce_to_item_list,
)

MixedModality = Literal["protein", "dna"]
# Biological tokens that fix a neighboring mask's modality, including ambiguous protein symbols.
MIXED_VOCAB_SET = frozenset(PROTEIN_TOKENS + AMBIGUOUS_PROTEIN_TOKENS + DNA_TOKENS)


class MixedSequenceInput(BaseToolInput):
    """Prepared protein/DNA loci in the upstream case-sensitive notation.

    Attributes:
        sequences (list[str]): Uppercase proteins and lowercase DNA, optionally separated
            by ``+``/``-`` strand markers (``<+>``/``<->`` are stored as ``+``/``-``).
            A single string is normalized to a list.
    """

    ALLOW_MASKS: ClassVar[bool] = False
    sequences: list[str] = InputField(
        title="Sequences",
        description="Prepared loci: uppercase protein, lowercase acgt, +/- strand markers (<+>/<-> accepted)",
        min_length=1,
        examples=["+MKTL+acgt-ACDE"],
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value: Any) -> Any:
        """Accept a single locus or a list of loci."""
        return [value] if isinstance(value, str) else value

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, sequences: list[str]) -> list[str]:
        """Validate tokens and store one-character strand markers, without changing case."""
        return ["".join(tokenize_mixed_sequence(sequence, allow_masks=cls.ALLOW_MASKS)) for sequence in sequences]

    def __len__(self) -> int:
        """Return the total number of model tokens."""
        return sum(len(sequence) for sequence in self.sequences)


def _implied_mask_modality(sequence: str, index: int) -> MixedModality | None:
    """Return the modality a mask's context requires, or None when the context is ambiguous."""
    start = index
    while start > 0 and sequence[start - 1] == MASK_TOKEN:
        start -= 1
    end = index
    while end < len(sequence) - 1 and sequence[end + 1] == MASK_TOKEN:
        end += 1
    left = sequence[start - 1] if start > 0 else None
    right = sequence[end + 1] if end < len(sequence) - 1 else None
    # A '-' marker only starts protein elements.
    if left == "-":
        return "protein"
    # A mask run between two same-modality tokens within one element shares their modality.
    if left is not None and right is not None and left in MIXED_VOCAB_SET and right in MIXED_VOCAB_SET:
        left_modality = _token_modality(left)
        if left_modality == _token_modality(right):
            return left_modality
    return None


def _token_modality(token: str) -> MixedModality:
    """Classify a biological token as protein (uppercase) or DNA (lowercase)."""
    return "dna" if token in DNA_TOKENS else "protein"


class MixedSequenceSampleInput(MixedSequenceInput):
    """Prepared loci and explicit modality metadata for any premasked sites.

    Attributes:
        sequences (list[str]): Prepared loci, optionally carrying ``_`` masks.
        mask_modalities (list[dict[int, MixedModality]] | None): One mapping per locus,
            assigning every mask's 1-indexed token position to protein or DNA. Positions
            index the ``+``/``-`` form. Omit for fully specified sequences.
    """

    ALLOW_MASKS: ClassVar[bool] = True
    sequences: list[str] = InputField(
        title="Sequences",
        description="Prepared mixed loci; '_' marks editable sites with explicit mask_modalities",
        min_length=1,
        examples=["+MKTL+acgt"],
    )
    mask_modalities: list[dict[int, MixedModality]] | None = InputField(
        default=None,
        title="Mask Modalities",
        description="Per-locus maps from 1-indexed masked token positions to protein or dna",
    )

    @model_validator(mode="after")
    def validate_mask_modalities(self) -> "MixedSequenceSampleInput":
        """Require exact mask coverage and keep the parallel input lists aligned."""
        modalities = self.mask_modalities
        if modalities is None:
            modalities = [{} for _ in self.sequences]
        if len(modalities) != len(self.sequences):
            raise ValueError("mask_modalities must have one mapping per sequence")
        for index, (sequence, mapping) in enumerate(zip(self.sequences, modalities, strict=True)):
            masked = {i + 1 for i, token in enumerate(sequence) if token == MASK_TOKEN}
            if set(mapping) != masked:
                raise ValueError(
                    f"mask_modalities[{index}] must label exactly the masked token positions {sorted(masked)}"
                )
            for position, modality in mapping.items():
                expected = _implied_mask_modality(sequence, position - 1)
                if expected is not None and modality != expected:
                    raise ValueError(f"mask_modalities[{index}][{position}] must be {expected!r} from its context")
        object.__setattr__(self, "mask_modalities", modalities)
        return self


def mixed_allowed_mask(sequence: str, length: int) -> list[list[bool]]:
    """Build the per-token mask of permitted columns for a mixed template.

    Args:
        sequence (str): Prepared template, normalized to one-character strand markers.
        length (int): Number of relaxed rows the mask must cover.

    Returns:
        list[list[bool]]: ``(L, 24)`` mask — protein tokens take columns 0-19, DNA tokens 20-23,
            and markers or ambiguous symbols take none.

    Raises:
        ValueError: If the template is not token-aligned with ``length``, or fixes every position.
    """
    tokens = list("".join(tokenize_mixed_sequence(sequence)))
    if len(tokens) != length:
        raise ValueError("logits must have one row per model token, including strand markers")
    width = len(MIXED_VOCAB)
    mask: list[list[bool]] = []
    for token in tokens:
        row = [False] * width
        if token in PROTEIN_TOKENS:
            row[:20] = [True] * 20
        elif token in DNA_TOKENS:
            row[20:] = [True] * (width - 20)
        mask.append(row)
    if not any(any(row) for row in mask):
        raise ValueError("Gradient input requires at least one canonical protein or DNA token")
    return mask


class MixedSequenceLogits(SequenceLogits):
    """A relaxed ``(L, 24)`` mixed protein/DNA state in ``ACDEFGHIKLMNPQRSTVWYacgt`` order."""

    VOCAB: ClassVar[Sequence[str]] = tuple(MIXED_VOCAB)


class MixedSequenceGradientCandidate(BaseModel):
    """One candidate: a relaxed state together with the template that fixes its modalities.

    The template belongs to the candidate, not the call, so a batch may mix protein-heavy and
    DNA-heavy candidates and each one's mask is derived from its own sequence.

    Attributes:
        sequence (str): Template fixing each position's modality, with one token per ``logits``
            row -- strand markers included, since each occupies a row of its own. ``<+>`` and
            ``<->`` are accepted and stored as ``+``/``-``. An uppercase letter marks a protein
            position (columns 0-19), lowercase marks DNA (columns 20-23), and a marker or ``X``
            fixes the position, so its row must be zero.
        logits (MixedSequenceLogits): One 24-column row per model token in
            ``ACDEFGHIKLMNPQRSTVWYacgt`` order. Fixed context rows must be zero.
    """

    sequence: str = Field(
        title="Sequence",
        description="Template fixing each position's modality; one token per logits row, markers included",
    )
    logits: MixedSequenceLogits = Field(
        title="Logits",
        description="Token-aligned (L, 24) state in ACDEFGHIKLMNPQRSTVWYacgt order; fixed rows are zero",
    )

    @field_validator("sequence")
    @classmethod
    def normalize_sequence(cls, sequence: str) -> str:
        """Validate tokens and store one-character strand markers."""
        return "".join(tokenize_mixed_sequence(sequence))

    def iter_sequence_logits(self) -> Iterator[tuple[str, SequenceLogits]]:
        """Expose this item's relaxed state to the shared validation hooks.

        Yields:
            tuple[str, SequenceLogits]: The field name and this item's matrix.
        """
        yield "logits", self.logits

    def allowed_mask(self) -> list[list[bool]]:
        """Return the permitted columns per position, from this item's own template.

        Returns:
            list[list[bool]]: Protein tokens take columns 0-19, DNA tokens 20-23, strand markers
                and ambiguous symbols none, which marks them as fixed context.
        """
        return mixed_allowed_mask(self.sequence, len(self.logits))


class MixedSequenceGradientInput(GradientInput):
    """One or many relaxed candidates, each aligned to its own mixed-sequence template.

    Attributes:
        inputs (list[MixedSequenceGradientCandidate]): One template-and-state pair per candidate.
        temperature (float | None): Softmax temperature shared by every item. The default ``1.0``
            reads each state as unnormalized and applies ``softmax(logits / T)``, returning
            logit-space gradients; ``None`` reads them as modality-restricted probabilities and
            returns probability-space gradients.
    """

    inputs: list[MixedSequenceGradientCandidate] = InputField(
        title="Relaxed Candidates",
        description="One template-and-state pair per candidate",
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def normalize_items(cls, value: Any) -> Any:
        """Accept one candidate as well as a list of them."""
        return coerce_to_item_list(value)

    def allowed_tokens(
        self,
        path: str,
        logits: SequenceLogits,  # noqa: ARG002 — sized from the item
    ) -> list[list[bool]]:
        """Restrict each position to its own item's template modality.

        The path names which item the matrix came from, so a batch never masks one candidate
        with another's template.

        Args:
            path (str): Field path of the matrix, as ``inputs[i].logits``.
            logits (SequenceLogits): The token-aligned relaxed state.

        Returns:
            list[list[bool]]: Per-token permitted columns for that item.

        Raises:
            ValueError: If the path does not name an item, or the state is not token-aligned.
        """
        match = re.fullmatch(r"inputs\[(\d+)\]\.logits", path)
        if match is None:
            raise ValueError(f"mixed gradient input cannot mask an unexpected relaxed field {path!r}")
        return self.inputs[int(match.group(1))].allowed_mask()

    @classmethod
    def build_candidate(
        cls,
        probabilities: Sequence[Sequence[float]],
        *,
        temperature: float = 1.0,
        **fields: Any,
    ) -> "MixedSequenceGradientCandidate":
        """Build one candidate whose logits reproduce modality-restricted ``probabilities``.

        Each row must be a distribution over its own modality block, with exact zeros outside
        it — which plain logits cannot express, since ``log(0)`` is not finite. The backend
        normalizes within the permitted block, so only the permitted columns need converting
        and the rest are left at zero, where the modality mask discards them anyway.

        The template belongs to the candidate, so ``sequence`` arrives here either shared across
        the call or named in this candidate's own mapping, and the mask comes from that template.

        Args:
            probabilities (Sequence[Sequence[float]]): ``(L, 24)`` token-aligned rows, strictly
                positive within each row's modality block and zero outside it.
            temperature (float): Positive, finite temperature the matrix is converted at.
            fields (Any): This candidate's remaining fields; must include ``sequence``.

        Returns:
            MixedSequenceGradientCandidate: Candidate reproducing ``probabilities`` within each
                modality block.

        Raises:
            ValueError: If ``sequence`` is missing, or a row is not a valid modality-restricted
                distribution with strictly positive permitted entries.
        """
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f"temperature must be positive and finite, got {temperature!r}")
        if "sequence" not in fields:
            raise ValueError("mixed gradient from_probabilities requires the sequence template")
        mask = mixed_allowed_mask(fields["sequence"], len(probabilities))
        MixedSequenceLogits([[float(v) for v in row] for row in probabilities]).validate_probabilities(
            allowed=mask, label="probabilities"
        )

        rows: list[list[float]] = []
        for index, (row, permitted) in enumerate(zip(probabilities, mask, strict=True)):
            converted = [0.0] * len(MIXED_VOCAB)
            for column, allowed in enumerate(permitted):
                if not allowed:
                    continue
                if row[column] <= 0.0:
                    raise ValueError(
                        f"probabilities row {index + 1} column {column} must be strictly positive; "
                        "pass the distribution directly with temperature=None to keep exact zeros"
                    )
                converted[column] = temperature * math.log(row[column])
            rows.append(converted)
        return MixedSequenceGradientCandidate(
            sequence=fields.pop("sequence"), logits=MixedSequenceLogits(rows), **fields
        )


class MixedSequenceEmbedding(SequenceEmbedding):
    """A pooled embedding with optional per-position logits.

    Attributes:
        vocab (list[str]): Column order of optional biological logits.
        logits (list[list[float]] | None): Optional per-token protein/DNA logits.
    """

    logits: list[list[float]] | None = Field(
        default=None, title="Logits", description="Optional per-token biological logits in vocabulary order"
    )
    vocab: list[str] = Field(
        default_factory=lambda: list(MIXED_VOCAB), title="Vocabulary", description="Biological-logit column order"
    )


class MixedEmbeddingsOutput(MaskedModelEmbeddingsOutput):
    """Embeddings and labeled logits, one bundle per input locus.

    Attributes:
        results (list[MixedSequenceEmbedding]): Per-locus embedding bundles in input order.
    """

    results: list[MixedSequenceEmbedding] = Field(title="Results", description="Embedding bundles in input order")  # type: ignore[assignment]

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Preserve vocabulary and optional logits in JSON; reuse pooled-vector exports."""
        if file_format == "json":
            Path(str(export_path) + ".json").write_text(json.dumps([r.model_dump() for r in self.results]))
        else:
            super()._export_output(export_path, file_format)


class MixedSequenceSample(MaskedModelSample):
    """One sampled locus with labeled biological logits.

    Attributes:
        sequence (str): Completed mixed protein/DNA locus with original strand markers.
        logits (list[list[float]] | None): Optional per-token protein/DNA logits.
        vocab (list[str]): Column order of optional biological logits.
    """

    sequence: str = Field(title="Sequence", description="Completed mixed protein/DNA locus with strand markers")
    logits: list[list[float]] | None = Field(
        default=None, title="Logits", description="Optional per-token biological logits with shape (L, 24)"
    )
    vocab: list[str] = Field(
        default_factory=lambda: list(MIXED_VOCAB), title="Vocabulary", description="Biological-logit column order"
    )


class MixedSampleOutput(MaskedModelSampleOutput):
    """Sampled loci and optional logits, retaining case and marker placement.

    Attributes:
        results (list[MixedSequenceSample]): Per-locus sampling bundles in input order.
    """

    results: list[MixedSequenceSample] = Field(title="Results", description="Sampled locus bundles in input order")  # type: ignore[assignment]

    @property
    def output_format_options(self) -> list[str]:
        """Mixed strings are exported as JSON or text, rather than biological FASTA."""
        return ["json", "txt"]

    @property
    def output_format_default(self) -> str:
        """Return the default export format."""
        return "json"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Preserve vocabulary and optional logits in JSON exports."""
        if file_format == "json":
            Path(str(export_path) + ".json").write_text(json.dumps([r.model_dump() for r in self.results]))
        else:
            super()._export_output(export_path, file_format)


class MixedScoringMetrics(MaskedModelScoringMetrics):
    """Canonical masked-PLL metrics with explicit target-position metadata.

    Attributes:
        scored_positions (list[int]): 1-indexed canonical protein/DNA target token positions.
    """

    scored_positions: list[int] = Field(
        title="Scored Positions", description="1-indexed model-token positions contributing to the masked PLL metrics"
    )


class MixedScoringOutput(MaskedModelScoringOutput):
    """Per-locus masked-PLL metrics and optional labeled logits.

    Attributes:
        scores (list[MixedScoringMetrics]): Per-locus scores in input order.
    """

    scores: list[MixedScoringMetrics] = Field(title="Scores", description="Per-locus scores in input order")  # type: ignore[assignment]

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Preserve scored-position metadata in JSON; reuse the shared metric CSV export."""
        if file_format == "json":
            Path(str(export_path) + ".json").write_text(json.dumps([r.model_dump() for r in self.scores]))
        else:
            super()._export_output(export_path, file_format)


class MixedGradientOutput(GradientOutput):
    """Token-aligned masked-PLL gradient with fixed-context rows preserved.

    Attributes:
        results (list[GradientResult]): One per input item, each holding its derivative of the
            masked NLL with respect to its input state (``None`` in forward-only mode), its mean
            masked negative log-likelihood, and its masked-PLL metrics and checkpoint metadata.
        vocab (list[str]): Protein/DNA column order shared by input and gradient.
    """

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Export the gradient, loss, metrics, and vocabulary as JSON."""
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        payload = self.model_dump(include={"results", "vocab"}, mode="json")
        Path(str(export_path) + ".json").write_text(json.dumps(payload))


__all__ = [
    "MIXED_VOCAB",
    "MixedEmbeddingsOutput",
    "MixedGradientOutput",
    "MixedSampleOutput",
    "MixedScoringMetrics",
    "MixedScoringOutput",
    "MixedSequenceEmbedding",
    "MixedSequenceGradientInput",
    "MixedSequenceInput",
    "MixedSequenceLogits",
    "mixed_allowed_mask",
    "MixedSequenceSample",
    "MixedSequenceSampleInput",
    "one_hot_mixed_logits",
]
