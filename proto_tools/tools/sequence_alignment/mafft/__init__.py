"""MAFFT multiple sequence alignment."""

from proto_tools.tools.sequence_alignment.mafft.mafft import MafftConfig, MafftInput, MafftOutput, run_mafft_align

__all__ = [
    "MafftInput",
    "MafftConfig",
    "MafftOutput",
    "run_mafft_align",
]
