"""Shared data and output models for sequence interaction probability maps."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from proto_tools.utils.tool_io import BaseToolOutput


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


class SequenceInteractionsOutput(BaseToolOutput):
    """Dense interaction maps, one bundle per input sequence.

    Attributes:
        results (list[SequenceInteractions]): Interaction bundles in input order.
    """

    results: list[SequenceInteractions] = Field(title="Results", description="Interaction bundles in input order")

    @property
    def output_format_options(self) -> list[str]:
        """Return supported export formats."""
        return ["json", "npz"]

    @property
    def output_format_default(self) -> str:
        """Prefer compact archives for dense interaction matrices."""
        return "npz"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Export matrices and token labels without pickle-dependent object arrays.

        Raises:
            ValueError: If channel names produce duplicate NPZ keys or the format is unsupported.
        """
        if file_format == "json":
            Path(str(export_path) + ".json").write_text(json.dumps([r.model_dump() for r in self.results]))
        elif file_format == "npz":
            import numpy as np

            arrays: dict[str, Any] = {}
            for index, result in enumerate(self.results):
                for channel, interaction_map in result.maps.items():
                    values_key = f"{index}_{channel}"
                    tokens_key = f"{index}_{channel}_tokens"
                    for key in (values_key, tokens_key):
                        if key in arrays:
                            raise ValueError(f"Interaction channel names produce duplicate NPZ key: {key}")
                    arrays[values_key] = np.asarray(interaction_map.values, dtype=np.float32)
                    arrays[tokens_key] = np.asarray(interaction_map.tokens, dtype=str)
            np.savez_compressed(str(export_path) + ".npz", **arrays)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
