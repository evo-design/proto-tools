"""
Shared utilities for bio_programming_tools.

Config, helpers, sequence validation, I/O, caching, env management, device, logging.
"""
from .base_config import BaseConfig, ConfigField
from .device import determine_visible_devices, number_of_available_gpus, use_cloud_gpu
from .tool_instance import ToolInstance
from .helpers import (
    DNA_NUCLEOTIDES,
    PROTEIN_AMINO_ACIDS,
    RNA_NUCLEOTIDES,
    calculate_gc_content,
    detect_sequence_type,
    resolve_sequence_ids,
    return_invalid_dna_chars,
    return_invalid_nucleotide_chars,
    return_invalid_protein_chars,
    return_invalid_rna_chars,
    validate_smiles,
)
from .logging_config import get_logger, setup_logging
from .tool_cache import (
    ToolCache,
    clear_cache,
    clear_tool_cache,
    get_cache_info,
    tool_cache,
    tool_cache_iterable,
)
from .tool_io import BaseToolInput, BaseToolOutput, ToolExecutionError

__all__ = [
    # Config
    "BaseConfig",
    "ConfigField",
    # Helpers & sequence validation
    "resolve_sequence_ids",
    "calculate_gc_content",
    "detect_sequence_type",
    "return_invalid_dna_chars",
    "return_invalid_nucleotide_chars",
    "return_invalid_protein_chars",
    "return_invalid_rna_chars",
    "validate_smiles",
    "DNA_NUCLEOTIDES",
    "RNA_NUCLEOTIDES",
    "PROTEIN_AMINO_ACIDS",
    # I/O
    "BaseToolInput",
    "BaseToolOutput",
    "ToolExecutionError",
    # Caching
    "tool_cache",
    "tool_cache_iterable",
    "clear_cache",
    "clear_tool_cache",
    "get_cache_info",
    "ToolCache",
    # Tool instance management
    "ToolInstance",
    # Device
    "use_cloud_gpu",
    "determine_visible_devices",
    "number_of_available_gpus",
    # Logging
    "get_logger",
    "setup_logging",
]
