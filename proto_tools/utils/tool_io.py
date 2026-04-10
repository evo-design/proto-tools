"""proto_tools/utils/tool_io.py.

Base input/output classes for standardized tool results with metadata tracking.
"""

import json
import math
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def InputField(
    default: Any = ...,
    *,
    title: str | None = None,
    description: str | None = None,
    hidden: bool = False,
    advanced: bool = False,
    include_in_key: bool = True,
    **kwargs: Any,
) -> Any:
    """Custom Field wrapper for tool Input classes.

    Adds UI metadata flags to json_schema_extra, matching the pattern
    established by ``ConfigField`` for Config classes.

    Args:
        default (Any): Default value for the field. Use ``...`` for required fields.
        title (str | None): Human-readable title for UI display.
        description (str | None): Description of the field for documentation and UI tooltips.
        hidden (bool): If True, field is hidden from UI completely.
        advanced (bool): If True, field appears in "Advanced" section of UI.
        include_in_key (bool): If False, field is excluded from tool cache key
            generation.
        kwargs: All other standard Pydantic Field arguments (via ``**kwargs``).
    """
    json_schema_extra = kwargs.get("json_schema_extra", {})
    json_schema_extra["hidden"] = hidden
    json_schema_extra["advanced"] = advanced
    json_schema_extra["include_in_key"] = include_in_key
    json_schema_extra["_field_type"] = "InputField"
    kwargs["json_schema_extra"] = json_schema_extra
    return Field(default, title=title, description=description, **kwargs)


class ToolExecutionError(Exception):
    """Exception raised when a tool execution fails."""

    def __init__(self, message: str):
        """Initialize ToolExecutionError."""
        self.message = message
        # Extract the last line and append to the top
        last_line = self.message.splitlines()[-1]
        super().__init__(f"Attempt to access field of tool output after failure: {last_line}\n{self.message}")


def _extra_dict(info: Any) -> dict[str, Any]:
    """Return ``json_schema_extra`` as a plain dict, defaulting to ``{}``."""
    extra = info.json_schema_extra
    return extra if isinstance(extra, dict) else {}


class BaseToolInput(BaseModel):
    """Base class for primary tool inputs.

    All tools should extend this with tool-specific primary input fields.

    Parameters that are not primary inputs should use the
    proto_tools.utils.BaseConfig class.
    """

    model_config = ConfigDict(
        extra="forbid",  # Catch typos in config keys
        validate_assignment=True,  # Validate on field updates
        use_enum_values=True,  # Serialize enums as values
        validate_default=True,  # Validate default values
    )

    @classmethod
    def cache_exclude_fields(cls) -> set[str]:
        """Return field names marked with ``include_in_key=False``."""
        return {name for name, info in cls.model_fields.items() if not _extra_dict(info).get("include_in_key", True)}

    def cache_key(self) -> str:
        """Deterministic string for cache key generation, excluding non-key fields."""
        model_dict = self.model_dump(exclude_none=True, exclude=self.cache_exclude_fields())
        return json.dumps(model_dict, sort_keys=True, default=str)

    @classmethod
    def item_cost(cls, item: Any) -> float:  # noqa: ARG003 — required by validator signature
        """Estimated cost of processing a single item (for ToolPool scheduling).

        Override in subclasses to provide cost-aware scheduling. Default is
        uniform cost (1.0), which degrades LPT to round-robin.

        Args:
            item (Any): A single item from the iterable input field.
        """
        return 1.0


# Metadata fields populated by the @tool decorator — these differ between
# runs and are excluded from output comparisons.
_OUTPUT_METADATA_FIELDS: set[str] = {
    "tool_id",
    "execution_time",
    "timestamp",
    "success",
    "warnings",
    "errors",
    "metadata",
}


class BaseToolOutput(BaseModel, ABC):
    """Base class for all tool outputs with standardized metadata.

    All tools should extend this with tool-specific result fields. Provides
    execution tracking (tool_id, execution_time, timestamp, success, warnings).

    NOTE: Metadata fields (tool_id, execution_time, success) are optional during
    construction within tool functions. The @tool decorator
    automatically populates these fields when wrapping tool execution.

    Attributes:
        tool_id (str | None): Unique tool identifier (e.g., ``"blast-search"``).
        execution_time (float | None): Execution time in seconds.
        timestamp (datetime): Execution timestamp.
        success (bool | None): Whether execution succeeded.
        warnings (list[str]): Non-fatal warnings generated during execution.
        errors (list[str]): Fatal error messages generated during execution.
        metadata (dict[str, Any]): Additional tool-specific metadata.

    Example:
        >>> class BLASTOutput(BaseToolOutput):
        ...     hits: List[Dict[str, Any]]
        >>> # Within a tool function (metadata populated by decorator):
        >>> result = BLASTOutput(hits=[{"accession": "P12345", "evalue": 1e-50}])
        >>>
        >>> # Direct construction (testing/manual use):
        >>> result = BLASTOutput(hits=[{"accession": "P12345", "evalue": 1e-50}])
    """

    # Universal metadata fields (optional during construction, populated by decorator)
    tool_id: str | None = Field(
        default=None,
        description="Unique tool identifier (e.g., 'blast-search', 'esm3-embedding')",
    )
    execution_time: float | None = Field(default=None, description="Execution time in seconds", ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.now, description="Execution timestamp")
    success: bool | None = Field(default=None, description="Whether execution succeeded")

    # Optional metadata fields
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings generated during execution",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Fatal error messages generated during execution",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional tool-specific metadata")

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

    def __getattr__(self, name: str) -> None:
        """Raise ToolExecutionError when accessing unset fields after failure."""
        # Check if we're trying to access a model field that wasn't set
        if name in self.__class__.model_fields:
            # Check if tool execution failed
            success = object.__getattribute__(self, "__dict__").get("success", True)
            if success is False:
                errors = object.__getattribute__(self, "__dict__").get("errors", [])

                if errors:
                    raise ToolExecutionError("\nError Messages:\n" + "\n".join(errors))
                raise ToolExecutionError("Tool failed with no error messages recorded")

        # Default behavior for truly missing attributes
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __str__(self) -> str:
        """Human-readable string representation with formatted errors."""
        execution_time_str = f"{self.execution_time:.4f}s" if self.execution_time is not None else "N/A"
        tool_id_str = self.tool_id if self.tool_id is not None else "unknown"

        if self.success:
            return f"{tool_id_str}: Success (executed in {execution_time_str})"
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
    def output_format_options(self) -> list[str]:
        """List of valid file formats for exporting the tool output."""
        return []

    @property
    @abstractmethod
    def output_format_default(self) -> str:
        """Default file format for exporting the tool output."""
        return ""

    @abstractmethod
    def _export_output(self, export_path: Path | str, file_format: str) -> None:
        """Exports the tool output to a file or directory of files.

        Must be implemented by all subclasses of BaseToolOutput.

        Args:
            export_path (Path | str): Path to the output directory or file location
            file_format (str): Format of the file to export the output to
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def export(
        self,
        name: str,
        export_path: Path | str | None = None,
        file_format: str | None = None,
    ) -> None:
        """Export the tool output to a file or directory containing files.

        Args:
            name (str): Name of the output file or directory (without extension)
            export_path (Path | str | None): Optional path to where the output directory/file will be saved
                If None, the output directory/file will be saved to the current working directory
            file_format (str | None): Format files will be exported to. If None, uses
                default file format (default should be defined by subclass).
        """
        if export_path is None:
            export_path = Path.cwd()
        else:
            os.makedirs(export_path, exist_ok=True)

        if file_format is None:
            file_format = self.output_format_default
        elif file_format not in self.output_format_options:
            raise ValueError(f"Invalid file format: {file_format}. Must be one of: {self.output_format_options}")

        export_path = Path(export_path) / f"{str(name).lower()}"
        self._export_output(export_path, file_format)

    def approx_equal(self, other: "BaseToolOutput", rtol: float = 1e-4, atol: float = 1e-5) -> None:
        """Assert that two tool outputs are approximately equal.

        Compares tool-specific output fields (excluding decorator metadata like
        ``tool_id``, ``execution_time``, ``timestamp``, etc.) with tolerance for
        floating-point differences.

        GPU tools using the same seed produce approximately — not exactly — identical
        results because CUDA operations like ``atomicAdd`` reduce floating-point values
        in non-deterministic order across threads. The results are mathematically
        equivalent but differ at the bit level (~1e-7 to 1e-15). Discrete outputs
        (sequences, labels, structure topology) are compared exactly; floating-point
        outputs (scores, coordinates, pLDDT) use configurable tolerance.

        Args:
            other (BaseToolOutput): The other output to compare against.
            rtol (float): Relative tolerance for float comparison.
            atol (float): Absolute tolerance for float comparison. ``1e-5`` matches
                the GPU float32 forward-pass noise floor for log-likelihood ratios
                and other small-magnitude scores that cluster near zero.

        Raises:
            AssertionError: If the outputs differ, with the path to the first difference.
        """
        for field_name in type(self).model_fields:
            if field_name in _OUTPUT_METADATA_FIELDS:
                continue
            self_val = getattr(self, field_name)
            other_val = getattr(other, field_name)
            _approx_equal_values(self_val, other_val, rtol=rtol, atol=atol, path=f"output.{field_name}")


def _approx_equal_values(a: Any, b: Any, rtol: float, atol: float, path: str) -> None:
    """Recursively compare nested structures with float tolerance.

    Args:
        a (Any): First value.
        b (Any): Second value.
        rtol (float): Relative tolerance for float comparison.
        atol (float): Absolute tolerance for float comparison.
        path (str): Dot-separated path for error messages (e.g. ``output.results[0].score``).

    Raises:
        AssertionError: If values differ, with the path to the first difference.
    """
    if type(a) is not type(b):
        raise AssertionError(f"Type mismatch at {path}: {type(a).__name__} != {type(b).__name__}")

    if isinstance(a, float):
        # Treat ``nan`` vs ``nan`` as equal: tools deliberately emit ``nan`` for
        # undefined metrics (e.g. ligand_interface_sequence_recovery when there
        # is no interface). ``math.isclose`` would otherwise reject it.
        if math.isnan(a) and math.isnan(b):
            return
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            raise AssertionError(f"Float mismatch at {path}: {a} != {b} (rtol={rtol}, atol={atol})")
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError(f"Dict keys differ at {path}: {set(a.keys()) ^ set(b.keys())}")
        for key in a:
            _approx_equal_values(a[key], b[key], rtol, atol, f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f"Length mismatch at {path}: {len(a)} != {len(b)}")
        for i, (ai, bi) in enumerate(zip(a, b, strict=True)):
            _approx_equal_values(ai, bi, rtol, atol, f"{path}[{i}]")
    elif hasattr(a, "approx_equal"):
        # Explicit per-class override takes precedence over generic recursion.
        try:
            a.approx_equal(b, rtol=rtol, atol=atol)
        except AssertionError as e:
            raise AssertionError(f"At {path}: {e}") from None
    elif isinstance(a, BaseModel):
        # Generic fallback for nested Pydantic models: recurse field-by-field so
        # float drift in nested ``metrics`` / ``per_position_metrics`` is compared
        # with tolerance instead of falling through to bit-exact ``BaseModel.__eq__``.
        # Classes that need custom comparison logic can override this by defining
        # their own ``approx_equal`` method (handled by the branch above).
        for field_name in type(a).model_fields:
            _approx_equal_values(
                getattr(a, field_name),
                getattr(b, field_name),
                rtol,
                atol,
                f"{path}.{field_name}",
            )
    elif a != b:
        raise AssertionError(f"Value mismatch at {path}: {a!r} != {b!r}")


__all__ = [
    "BaseToolInput",
    "BaseToolOutput",
    "ToolExecutionError",
]
