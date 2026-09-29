"""Token-aligned sequence interaction probability maps."""

from pydantic import BaseModel, Field, model_validator


class SequenceInteractionMap(BaseModel):
    """A dense probability matrix with explicitly labeled token axes.

    Labels may represent residues, bases, model tokens, or numbered positions.
    Values need not be symmetric or have a zero diagonal.

    Attributes:
        tokens (list[str]): Ordered labels for both matrix axes.
        values (list[list[float]]): Square matrix of finite interaction probabilities.
    """

    tokens: list[str] = Field(title="Tokens", description="Ordered labels for both matrix axes", min_length=1)
    values: list[list[float]] = Field(
        title="Values", description="Square token-by-token interaction probability matrix"
    )

    @model_validator(mode="after")
    def validate_matrix(self) -> "SequenceInteractionMap":
        """Validate square shape, axis alignment, and finite probability values."""
        length = len(self.tokens)
        if len(self.values) != length or any(len(row) != length for row in self.values):
            raise ValueError("Interaction matrix must be square and match its token axis")
        if any(not 0.0 <= value <= 1.0 for row in self.values for value in row):
            raise ValueError("Interaction probabilities must be finite values in [0, 1]")
        return self


class SequenceInteractions(BaseModel):
    """Named interaction channels for one input sequence.

    Attributes:
        maps (dict[str, SequenceInteractionMap]): Named interaction probability maps.
    """

    maps: dict[str, SequenceInteractionMap] = Field(title="Maps", description="Named interaction probability maps")
