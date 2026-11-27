"""Antibody entity."""

from collections.abc import Iterator

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from proto_tools.utils.sequence_logits import ProteinSequenceLogits, SequenceLogits


class Antibody(BaseModel):
    """An antibody represented by its chain sequences.

    At least one of ``heavy_chain`` or ``light_chain`` must be provided.

    Attributes:
        heavy_chain (str | None): Heavy chain amino-acid sequence.
        light_chain (str | None): Light chain amino-acid sequence.
    """

    model_config = ConfigDict(extra="forbid")

    heavy_chain: str | None = Field(
        default=None,
        title="Heavy Chain",
        description="Heavy chain amino-acid sequence",
    )
    light_chain: str | None = Field(
        default=None,
        title="Light Chain",
        description="Light chain amino-acid sequence",
    )

    @model_validator(mode="after")
    def validate_at_least_one_chain(self) -> Self:
        """Ensure at least one chain is provided."""
        if self.heavy_chain is None and self.light_chain is None:
            raise ValueError("At least one of heavy_chain or light_chain must be provided")
        return self

    def to_sequence(self) -> str:
        """Convert to the pipe-separated string format used by AbLang inference.

        Returns:
            str: ``"heavy|light"`` for paired, or the single chain sequence.
        """
        if self.heavy_chain is not None and self.light_chain is not None:
            return f"{self.heavy_chain}|{self.light_chain}"
        if self.heavy_chain is not None:
            return self.heavy_chain
        assert self.light_chain is not None, "model_validator guarantees at least one chain is set"
        return self.light_chain


class AntibodyLogits(BaseModel):
    """An antibody represented by relaxed distributions over amino acids.

    At least one of ``heavy_chain`` or ``light_chain`` must be provided. Both chains are
    ordinary relaxed sequence matrices, and :meth:`iter_sequence_logits` exposes them so a
    gradient input composing this model validates each chain through the shared contract.

    Attributes:
        heavy_chain (ProteinSequenceLogits | None): Heavy chain relaxed state with shape
            (Lh, 20) in canonical amino-acid order.
        light_chain (ProteinSequenceLogits | None): Light chain relaxed state with shape
            (Ll, 20) in canonical amino-acid order.
    """

    model_config = ConfigDict(extra="forbid")

    heavy_chain: ProteinSequenceLogits | None = Field(
        default=None,
        title="Heavy Chain Logits",
        description="Heavy chain relaxed state with shape (Lh, 20) in canonical amino-acid order",
    )
    light_chain: ProteinSequenceLogits | None = Field(
        default=None,
        title="Light Chain Logits",
        description="Light chain relaxed state with shape (Ll, 20) in canonical amino-acid order",
    )

    @model_validator(mode="after")
    def validate_at_least_one_chain(self) -> Self:
        """Ensure at least one chain is provided."""
        if self.heavy_chain is None and self.light_chain is None:
            raise ValueError("At least one of heavy_chain or light_chain must be provided")
        return self

    def iter_sequence_logits(self) -> Iterator[tuple[str, SequenceLogits]]:
        """Yield ``(chain_name, matrix)`` for each populated chain.

        Gradient inputs call this to find the matrices a composite field holds, so the chains
        are validated individually against the shared contract.

        Yields:
            tuple[str, SequenceLogits]: Chain field name and its ``(L, 20)`` relaxed state.
        """
        for name in ("heavy_chain", "light_chain"):
            chain = getattr(self, name)
            if chain is not None:
                yield name, chain
