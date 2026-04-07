"""Open reading frame prediction tools."""

from proto_tools.tools.orf_prediction.orf import ORF
from proto_tools.tools.orf_prediction.orfipy import OrfipyConfig, OrfipyInput, OrfipyOutput, run_orfipy_prediction
from proto_tools.tools.orf_prediction.prodigal import (
    TRANSLATION_TABLE_MAP,
    ProdigalConfig,
    ProdigalInput,
    ProdigalOutput,
    TranslationTable,
    run_prodigal_prediction,
)

__all__ = [
    "ORF",
    # Orfipy
    "OrfipyInput",
    "OrfipyConfig",
    "OrfipyOutput",
    "run_orfipy_prediction",
    # Prodigal
    "TRANSLATION_TABLE_MAP",
    "ProdigalInput",
    "ProdigalConfig",
    "ProdigalOutput",
    "TranslationTable",
    "run_prodigal_prediction",
]
