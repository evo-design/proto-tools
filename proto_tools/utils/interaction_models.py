"""Shared output model for sequence interaction maps."""

import json
from pathlib import Path
from typing import Any

from pydantic import Field

from proto_tools.entities.sequence_interactions import SequenceInteractions
from proto_tools.utils.tool_io import BaseToolOutput


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
