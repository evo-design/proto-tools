"""Shared constants, tissue mappings, and config base for Pangolin tools."""

from typing import Literal

from pydantic import field_validator

from proto_tools.utils import BaseConfig, ConfigField, return_invalid_dna_chars

# Pangolin requires 5000 bp of flanking context on each side of the predicted region.
PANGOLIN_FLANK = 5000

PangolinTissue = Literal["HEART", "LIVER", "BRAIN", "TESTIS"]

# Tissue -> (checkpoint index, output channel), mirroring the upstream CLI: load
# checkpoints final.{1,2,3}.{index}.3.v2 and read the P(splice) head (channels
# [1, 4, 7, 10]) — not the separate "usage" heads.
TISSUE_SPEC: dict[PangolinTissue, tuple[int, int]] = {
    "HEART": (0, 1),
    "LIVER": (2, 4),
    "BRAIN": (4, 7),
    "TESTIS": (6, 10),
}


def validate_dna(sequence: str) -> str:
    """Validate and normalize a DNA sequence (A/C/G/T/N, uppercased)."""
    if not sequence or not sequence.strip():
        raise ValueError("Sequence cannot be empty")
    sequence = sequence.upper()
    invalid = return_invalid_dna_chars(sequence, additional_valid_chars="N")
    if invalid:
        raise ValueError(f"Invalid nucleotide characters in sequence: {', '.join(sorted(invalid))}")
    return sequence


class PangolinConfig(BaseConfig):
    """Shared configuration for Pangolin tools.

    Attributes:
        tissues (list[PangolinTissue]): Tissues whose splice predictions are
            ensembled. Defaults to all four Pangolin tissues.
        device (str): Device to run the model on. Override of ``BaseConfig.device``
            because Pangolin is a GPU tool (default ``cuda``).
    """

    tissues: list[PangolinTissue] = ConfigField(
        title="Tissues",
        default=["HEART", "LIVER", "BRAIN", "TESTIS"],
        description="Tissues whose splice predictions are ensembled (HEART, LIVER, BRAIN, TESTIS)",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        include_in_key=False,
    )

    @field_validator("tissues")
    @classmethod
    def validate_tissues(cls, tissues: list[str]) -> list[str]:
        """Uppercase, validate, and deduplicate requested tissues (order preserved)."""
        seen: set[str] = set()
        normalized: list[str] = []
        for name in tissues:
            upper = name.strip().upper()
            if upper not in TISSUE_SPEC:
                raise ValueError(f"Unknown tissue {name!r}; valid: {sorted(TISSUE_SPEC)}")
            if upper not in seen:
                seen.add(upper)
                normalized.append(upper)
        if not normalized:
            raise ValueError("At least one tissue must be selected")
        return normalized
