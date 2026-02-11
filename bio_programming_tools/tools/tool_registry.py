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

from pydantic import BaseModel, Field, field_serializer

logger = logging.getLogger(__name__)

# List of warning message substrings to ignore (noisy warnings from dependencies)
IGNORED_WARNING_SUBSTRINGS = [
    "get_autocast_gpu_dtype",
    "get_autocast_dtype",
]

from bio_programming_tools.utils.tool_io import BaseToolInput, BaseToolOutput
from bio_programming_tools.utils import BaseConfig


class ToolSpec(BaseModel):
    """
    Specification for a registered tool.

    Stores tool metadata in the registry and is automatically serialized
    by FastAPI to JSON for API/client integration.
    """

    # Public fields - exposed in API
    key: str = Field(description="Internal identifier (e.g., 'blast-search')")
    label: str = Field(description="External UI display name (e.g., 'BLAST Search')")
    description: str = Field(description="Detailed description of tool functionality")

    # Configuration model - serialized as JSON Schema
    config_model: Type[BaseModel] = Field(
        description="Pydantic model for configuration validation and schema generation"
    )

    # Private fields - excluded from serialization
    input_model: Type[BaseToolInput] = Field(exclude=True)
    output_model: Type[BaseToolOutput] = Field(exclude=True)
    function: Callable = Field(exclude=True)

    model_config = {
        "arbitrary_types_allowed": True,
    }

    @field_serializer('config_model')
    def serialize_config_model(self, config_model: Type[BaseModel]) -> Dict[str, Any]:
        """Serialize config_model as standard JSON Schema."""
        return config_model.model_json_schema()


class ToolRegistry:
    """
    Registry for tool discovery and schema generation.

    Provides discovery, schema generation, and factory methods for tools.
    Registration happens at import time via the @tool decorator.

    Public Methods:
    - register(): Decorator to register tool functions
    - list_all(): List tools with metadata and schemas
    - get(): Get tool spec by key
    - get_config_schema(): Get JSON schema for tool configuration
    - count(): Get number of registered tools

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
        ...     pass

        API/Client Usage:
        >>> tools = ToolRegistry.list_all()
        >>> schema = ToolRegistry.get_config_schema("blast-search")

        Direct Usage:
        >>> from bio_programming_tools.tools.gene_annotation import run_blast, BLASTConfig
        >>> config = BLASTConfig(query_sequences=["MVLSP"], database="/data/nr")
        >>> result = run_blast(config)
    """

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
        """
        def decorator(func: Callable):
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

        Returns:
            List of ToolSpec Pydantic models
        """
        return list(cls._registry.values())

    @classmethod
    def get(cls, key: str) -> ToolSpec:
        """
        Get tool spec by key.

        Args:
            key: Tool identifier

        Returns:
            Tool specification object

        Raises:
            ValueError: If key not found in registry
        """
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown tool: '{key}'. Available tools: {available}")
        return cls._registry[key]

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
        """Get input, config, and output schemas."""
        return {
            "inputs": cls.get_input_schema(key),
            "config": cls.get_config_schema(key),
            "output": cls.get_output_schema(key),
        }

    @classmethod
    def count(cls) -> int:
        """Get count of registered tools."""
        return len(cls._registry)

    @classmethod
    def _check_duplicate(cls, key: str, attempted_name: str = None) -> None:
        """
        Check for duplicate registration.

        Raises:
            ValueError: If key already exists in registry
        """
        if key in cls._registry:
            existing_label = cls._registry[key].label
            error_msg = f"Tool '{key}' is already registered. Duplicate registration is not allowed."
            if attempted_name:
                error_msg += f"\nExisting: {existing_label}, Attempted: {attempted_name}"
            else:
                error_msg += f"\nExisting tool: {existing_label}"
            raise ValueError(error_msg)


def _re_emit_warnings(warning_list: List[Warning]) -> None:
    """Re-emit warnings so they still get logged/printed."""
    for w in warning_list:
        warnings.warn_explicit(
            w.message, w.category, w.filename, w.lineno, w.file, w.line
        )


# Alias for simpler decorator syntax: @tool(...) instead of @ToolRegistry.register(...)
tool = ToolRegistry.register
