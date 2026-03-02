"""
Shared utilities for bio_programming_tools.

Config, helpers, sequence validation, I/O, caching, env management, device, logging.
"""
from .base_config import BaseConfig, ConfigField
from .device import (
    determine_visible_devices,
    display_gpu_memory_usage,
    get_gpu_memory_info,
    get_gpu_process_memory,
    number_of_available_gpus,
)
from .device_manager import (
    AllocationType,
    DeviceManager,
    OffloadStrategy,
    SUPPORTED_DEVICE_PREFIXES,
)
from .http_session import build_http_session
from .system_info import (
    capture_parent_env,
    capture_subprocess_env,
    clear_captured_env,
    collect_system_info,
    get_captured_env,
    get_git_info,
    get_gpu_info,
    get_parent_process_env,
    get_platform_id,
    get_platform_info,
)
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
    # System info
    "get_platform_info",
    "get_gpu_info",
    "get_parent_process_env",
    "get_platform_id",
    "get_git_info",
    "collect_system_info",
    # Environment capture
    "capture_parent_env",
    "capture_subprocess_env",
    "get_captured_env",
    "clear_captured_env",
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
    "determine_visible_devices",
    "number_of_available_gpus",
    "get_gpu_memory_info",
    "get_gpu_process_memory",
    "display_gpu_memory_usage",
    # Device management
    "AllocationType",
    "DeviceManager",
    "OffloadStrategy",
    "SUPPORTED_DEVICE_PREFIXES",
    # HTTP
    "build_http_session",
    # Logging
    "get_logger",
    "setup_logging",
]
