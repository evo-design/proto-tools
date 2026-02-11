"""
tool_io.py

Base input/output classes for standardized tool results with metadata tracking.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ToolExecutionError(Exception):
    """Exception raised when a tool execution fails."""

    def __init__(self, message: str):
        self.message = message
        # Extract the last line and append to the top
        last_line = self.message.splitlines()[-1]
        super().__init__(
            f"Attempt to access field of tool output after failure: {last_line}\n{self.message}"
        )

class BaseToolInput(BaseModel):
    """Base class for primary tool inputs.

    All tools should extend this with tool-specific primary input fields.

    Parameters that are not primary inputs should use the
    bio_programming_tools.utils.BaseConfig class.
    """

    model_config = ConfigDict(
        extra="forbid",  # Catch typos in config keys
        validate_assignment=True,  # Validate on field updates
        use_enum_values=True,  # Serialize enums as values
        validate_default=True,  # Validate default values
    )

class BaseToolOutput(BaseModel, ABC):
    """
    Base class for all tool outputs with standardized metadata.

    All tools should extend this with tool-specific result fields. Provides
    execution tracking (tool_id, execution_time, timestamp, success, warnings).

    NOTE: Metadata fields (tool_id, execution_time, success) are optional during
    construction within tool functions. The @tool decorator
    automatically populates these fields when wrapping tool execution.

    Example:
        >>> class BLASTOutput(BaseToolOutput):
        ...     hits: List[Dict[str, Any]]
        ...
        >>> # Within a tool function (metadata populated by decorator):
        >>> result = BLASTOutput(
        ...     hits=[{"accession": "P12345", "evalue": 1e-50}]
        ... )
        >>>
        >>> # Direct construction (testing/manual use):
        >>> result = BLASTOutput(
        ...     hits=[{"accession": "P12345", "evalue": 1e-50}]
        ... )
    """

    # Universal metadata fields (optional during construction, populated by decorator)
    tool_id: Optional[str] = Field(
        default=None,
        description="Unique tool identifier (e.g., 'blast-search', 'esm3-embedding')",
    )
    execution_time: Optional[float] = Field(
        default=None, description="Execution time in seconds", ge=0.0
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Execution timestamp"
    )
    success: Optional[bool] = Field(default=None, description="Whether execution succeeded")

    # Optional metadata fields
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal warnings generated during execution",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Fatal error messages generated during execution",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional tool-specific metadata"
    )

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_assignment=True,
        use_enum_values=True,
        validate_default=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "tool_id": "blast-search",
                    "execution_time": 45.2,
                    "timestamp": "2025-10-06T10:30:00Z",
                    "success": True,
                    "warnings": [],
                    "errors": [],
                    "metadata": {"blast_version": "2.14.0", "database_size": 1000000},
                }
            ]
        },
    )

    def __getattr__(self, name: str):
        """Raise ToolExecutionError when accessing unset fields after failure."""
        # Check if we're trying to access a model field that wasn't set
        if name in self.__class__.model_fields:
            # Check if tool execution failed
            success = object.__getattribute__(self, "__dict__").get("success", True)
            if success is False:
                errors = object.__getattribute__(self, "__dict__").get("errors", [])

                if errors:
                    raise ToolExecutionError("\nError Messages:\n" + "\n".join(errors))
                else:
                    raise ToolExecutionError("Tool failed with no error messages recorded")

        # Default behavior for truly missing attributes
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __str__(self) -> str:
        """Human-readable string representation with formatted errors."""
        execution_time_str = f"{self.execution_time:.4f}s" if self.execution_time is not None else "N/A"
        tool_id_str = self.tool_id if self.tool_id is not None else "unknown"

        if self.success:
            return f"{tool_id_str}: Success (executed in {execution_time_str})"
        else:
            error_section = "\n" + "=" * 80 + "\n"
            error_section += f"{tool_id_str}: TOOL FAILURE after {execution_time_str}\n"
            error_section += "=" * 80 + "\n"
            for i, error in enumerate(self.errors, 1):
                error_section += f"\nError {i}:\n{error}\n"
            if self.warnings:
                error_section += "\nWarnings:\n"
                for warning in self.warnings:
                    error_section += f"  - {warning}\n"
            error_section += "=" * 80
            return error_section

    def __repr__(self) -> str:
        """Machine-readable string representation with errors."""
        return f"{self.__class__.__name__}({', '.join(list(self.model_dump().keys()))})"

    @property
    @abstractmethod
    def output_format_options(self) -> List[str]:
        """List of valid file formats for exporting the tool output."""
        return []

    @property
    @abstractmethod
    def output_format_default(self) -> str:
        """Default file format for exporting the tool output."""
        return ""

    @abstractmethod
    def _export_output(self, export_path: Path | str, file_format: str):
        """
        Exports the tool output to a file or directory of files.

        Must be implemented by all subclasses of BaseToolOutput.

        Args:
            export_path: Path to the output directory or file location
            file_format: Format of the file to export the output to
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def export(
        self,
        name: str,
        export_path: Optional[Path | str] = None,
        file_format: Optional[str] = None,
    ):
        """Export the tool output to a file or directory containing files.

        Args:
            name: Name of the output file or directory (without extension)
            export_path: Optional path to where the output directory/file will be saved
                If None, the output directory/file will be saved to the current working directory
            file_format: Format files will be exported to. If None, uses
                default file format (default should be defined by subclass).
        """

        if export_path is None:
            export_path = Path.cwd()
        else:
            os.makedirs(export_path, exist_ok=True)

        if file_format is None:
            file_format = self.output_format_default
        elif file_format not in self.output_format_options:
            raise ValueError(
                f"Invalid file format: {file_format}. Must be one of: {self.output_format_options}"
            )

        export_path = Path(export_path) / f"{str(name).lower()}"
        self._export_output(export_path, file_format)


__all__ = [
    "ToolExecutionError",
    "BaseToolInput",
    "BaseToolOutput",
]
