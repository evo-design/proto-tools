"""
tool_registry.py

Tool registry for managing tool discovery and schema generation.

Provides a decorator-based API for registering tools with metadata and
automatic schema generation for API/client integration.
"""
from __future__ import annotations

import logging
import time
import traceback
import warnings
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Type

from pydantic import Field

logger = logging.getLogger(__name__)

# List of warning message substrings to ignore (noisy warnings from dependencies)
IGNORED_WARNING_SUBSTRINGS = [
    "get_autocast_gpu_dtype",
    "get_autocast_dtype",
]

from bio_programming.bio_tools.tools.utils import BaseConfig, BaseRegistry, BaseSpec
from bio_programming.bio_tools.tools.infra.tool_io import BaseToolInput, BaseToolOutput


class ToolSpec(BaseSpec):
    """
    Specification for a registered tool.

    Extends BaseSpec with tool-specific metadata for discovery and schema generation.
    """

    # Private fields - excluded from serialization
    input_model: Type[BaseToolInput] = Field(exclude=True)
    output_model: Type[BaseToolOutput] = Field(exclude=True)
    function: Callable = Field(exclude=True)


class ToolRegistry(BaseRegistry[ToolSpec]):
    """
    Registry for tool discovery and schema generation.

    Inherits common registry functionality from BaseRegistry and adds
    tool-specific metadata.

    Public Methods:
    - register(): Decorator to register tool functions
    - list_all(): List tools with metadata and schemas
    - get(): Get tool spec by key (inherited)
    - get_schema(): Get JSON schema for tool configuration (inherited)
    - count(): Get number of registered tools (inherited)

    Examples:
        Registration (in tool files):
        >>> @tool(
        ...     key="blast-search",
        ...     label="BLAST Search",
        ...     input=BLASTInput,
        ...     config=BLASTConfig,
        ...     output=BLASTOutput,
        ...     description="Protein similarity search using BLAST",
        ... )
        ... def run_blast(config: BLASTConfig) -> BLASTOutput:
        ...     # Tool implementation
        ...     pass

        API/Client Usage:
        >>> # List all available tools
        >>> tools = ToolRegistry.list_all()
        >>>
        >>> # Get form schema
        >>> schema = ToolRegistry.get_schema("blast-search")

        Direct Usage:
        >>> # Call tool function directly
        >>> from bio_programming.bio_tools.tools.gene_annotation import run_blast, BLASTConfig
        >>> config = BLASTConfig(query_sequences=["MVLSP"], database="/data/nr")
        >>> result = run_blast(config)
    """

    # Each registry subclass must have its own _registry dict
    _registry: Dict[str, ToolSpec] = {}

    @classmethod
    def register(
        cls,
        key: str,
        label: str,
        input: Type[BaseToolInput],
        config: Type[BaseConfig],
        output: Type[BaseToolOutput],
        description: str,
    ):
        """
        Decorator to register a tool function and wrap execution with metadata tracking.

        This decorator:
        1. Registers the tool in the registry with its metadata
        2. Wraps the function to automatically track execution time
        3. Handles errors and returns standardized output with success/failure status
        4. Populates metadata fields (tool_id, execution_time, success, timestamp)

        Args:
            key: Unique identifier (e.g., "blast-search", "esm3-embedding")
            label: Readable display name (e.g., "BLAST Search", "ESM3 Embedding")
            input: Pydantic model class for primary input validation
            config: Pydantic model class for tool configuration validation
            output: Pydantic model class for tool output validation
            description: Readable description

        Returns:
            Decorator that wraps the function with metadata tracking

        Examples:
            >>> @tool(
            ...     key="blast-search",
            ...     label="BLAST Search",
            ...     input=BLASTInput,
            ...     config=BLASTConfig,
            ...     output=BLASTOutput,
            ...     description="Protein similarity search using BLAST",
            ... )
            ... def run_blast(inputs: BLASTInput, config: BLASTConfig) -> BLASTOutput:
            ...     # Tool only needs to return result data
            ...     return BLASTOutput(hits=[...])
        """
        def decorator(func: Callable):
            # Prevent duplicate registration using base class helper
            cls._check_duplicate(key, func.__name__)

            @wraps(func)
            def wrapper(inputs: BaseToolInput, config: BaseConfig) -> BaseToolOutput:
                """Wrapper that tracks execution and populates metadata."""
                start_time = time.time()
                logger.debug(f"Tool {key}: starting execution")

                # Capture warnings during execution
                with warnings.catch_warnings(record=True) as warning_list:
                    # Ensure all warnings are captured
                    warnings.simplefilter("always")

                    try:
                        # Execute the tool function
                        result = func(inputs, config)

                        # Populate metadata fields
                        result.tool_id = key
                        result.execution_time = time.time() - start_time
                        result.success = True
                        if result.timestamp is None:
                            result.timestamp = datetime.now()

                        # Add captured warnings to the result (filtered)
                        if warning_list:
                            # Filter out ignored warnings
                            filtered_warnings = [
                                w for w in warning_list
                                if not any(ignored in str(w.message) for ignored in IGNORED_WARNING_SUBSTRINGS)
                            ]

                            if filtered_warnings:
                                captured_warnings = [str(w.message) for w in filtered_warnings]
                                result.warnings = result.warnings + captured_warnings
                                for w in captured_warnings:
                                    logger.warning(f"Tool {key}: {w}")

                                # Re-emit warnings so they still get logged/printed
                                _re_emit_warnings(filtered_warnings)

                        logger.debug(f"Tool {key}: completed in {result.execution_time:.2f}s")
                        return result

                    except Exception as e:
                        # Capture warnings even on failure (filtered)
                        filtered_warnings = [
                            w for w in warning_list
                            if not any(ignored in str(w.message) for ignored in IGNORED_WARNING_SUBSTRINGS)
                        ]
                        captured_warnings = [str(w.message) for w in filtered_warnings]

                        # Re-emit warnings before returning error
                        _re_emit_warnings(filtered_warnings)

                        # Capture traceback
                        full_traceback = traceback.format_exc()
                        logger.error(f"Tool {key}: failed with {type(e).__name__}: {e}")

                        # Create error output using model_construct to bypass validation
                        # (This allows us to create an output with only metadata fields populated)
                        error_output = output.model_construct(
                            tool_id=key,
                            execution_time=time.time() - start_time,
                            success=False,
                            timestamp=datetime.now(),
                            warnings=captured_warnings,
                            errors=[str(e)] + [full_traceback],
                        )
                        return error_output

            # Register the tool spec with the original function
            cls._registry[key] = ToolSpec(
                key=key,
                label=label,
                description=description,
                input_model=input,
                config_model=config,
                output_model=output,
                function=wrapper,
            )
            return wrapper
        return decorator

    @classmethod
    def list_all(cls) -> List[ToolSpec]:
        """
        List all registered tools as Pydantic models.

        Returns list of ToolSpec models that FastAPI automatically serializes to JSON.
        Each spec includes key, label, description, input_model (serialized as JSON Schema),
        config_model (serialized as JSON Schema), output_model (serialized as JSON Schema),
        and function.

        Returns:
            List of ToolSpec Pydantic models

        Examples:
            >>> tools = ToolRegistry.list_all()
            >>> for spec in tools:
            ...     print(f"{spec.label} ({spec.key})")
        """
        return list(cls._registry.values())

    @classmethod
    def get_input_schema(cls, key: str) -> Dict[str, Any]:
        """Get JSON schema for tool inputs."""
        spec = cls.get(key)
        return spec.input_model.model_json_schema()

    @classmethod
    def get_config_schema(cls, key: str) -> Dict[str, Any]:
        """Get JSON schema for tool configuration."""
        spec = cls.get(key)
        return spec.config_model.model_json_schema()

    @classmethod
    def get_output_schema(cls, key: str) -> Dict[str, Any]:
        """Get JSON schema for tool output."""
        spec = cls.get(key)
        return spec.output_model.model_json_schema()

    @classmethod
    def get_schemas(cls, key: str) -> Dict[str, Dict[str, Any]]:
        """Get both input and config schemas."""
        return {
            "inputs": cls.get_input_schema(key),
            "config": cls.get_config_schema(key),
            "output": cls.get_output_schema(key),
        }


def _re_emit_warnings(warning_list: List[Warning]) -> None:
    """Re-emit warnings so they still get logged/printed."""
    for w in warning_list:
        warnings.warn_explicit(
            w.message, w.category, w.filename, w.lineno, w.file, w.line
        )


# Alias for simpler decorator syntax: @tool(...) instead of @tool(...)
tool = ToolRegistry.register
