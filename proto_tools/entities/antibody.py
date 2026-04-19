"""Antibody entity."""

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing_extensions import Self

from proto_tools.utils.sequence import PROTEIN_AMINO_ACIDS


class Antibody(BaseModel):
    """An antibody represented by its chain sequences.

    At least one of ``heavy_chain`` or ``light_chain`` must be provided.

    Attributes:
        heavy_chain (str | None): Heavy chain amino-acid sequence.
        light_chain (str | None): Light chain amino-acid sequence.
    """

    model_config = ConfigDict(extra="forbid")

    heavy_chain: str | None = None
    light_chain: str | None = None

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
    """An antibody represented by relaxed logit distributions over amino acids.

    At least one of ``heavy_chain`` or ``light_chain`` must be provided.

    Attributes:
        heavy_chain (list[list[float]] | None): Heavy chain logits with shape
            (Lh, 20) in canonical amino-acid order.
        light_chain (list[list[float]] | None): Light chain logits with shape
            (Ll, 20) in canonical amino-acid order.
    """

    model_config = ConfigDict(extra="forbid")

    heavy_chain: list[list[float]] | None = None
    light_chain: list[list[float]] | None = None

    @model_validator(mode="after")
    def validate_at_least_one_chain(self) -> Self:
        """Ensure at least one chain is provided."""
        if self.heavy_chain is None and self.light_chain is None:
            raise ValueError("At least one of heavy_chain or light_chain must be provided")
        return self

    @field_validator("heavy_chain", "light_chain")
    @classmethod
    def validate_chain_logits(cls, v: list[list[float]] | None, info: Any) -> list[list[float]] | None:
        """Ensure chain logits are a non-empty rectangular L x 20 matrix."""
        if v is None:
            return v
        if not v:
            raise ValueError(f"{info.field_name} must contain at least one position")
        expected_width = len(PROTEIN_AMINO_ACIDS)
        for idx, row in enumerate(v):
            if len(row) != expected_width:
                raise ValueError(f"{info.field_name} row {idx} must have {expected_width} columns, got {len(row)}")
        return v
