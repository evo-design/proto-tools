"""
Base configuration class for all pydantic configs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField

DEFAULT_TIMEOUT = 600  # seconds


def ConfigField(
    default: Any = ...,
    *,
    title: str = None,
    description: str = None,
    advanced: bool = False,
    hidden: bool = False,
    reload_on_change: bool = False,
    **kwargs,
) -> Any:
    """
    Custom Field wrapper that automatically adds metadata flags to json_schema_extra.

    Args:
        advanced: If True, field appears in "Advanced" section of UI
        hidden: If True, field is hidden from UI completely
        reload_on_change: If True, changing this field between persistent
            worker calls triggers a subprocess restart.

        **kwargs: All other standard Pydantic Field arguments

    Usage:
        param: int = Field(default=42, title="Param", description="...", advanced=True)
    """
    json_schema_extra = kwargs.get("json_schema_extra", {})

    json_schema_extra["advanced"] = advanced
    json_schema_extra["hidden"] = hidden
    json_schema_extra["reload_on_change"] = reload_on_change

    kwargs["json_schema_extra"] = json_schema_extra

    return PydanticField(default, title=title, description=description, **kwargs)


class BaseConfig(BaseModel):
    """
    Base configuration class for consistent behavior across all configs (tools, constraints, and generators).

    Attributes:
        verbose: Whether to print status messages.
        device: Device to run the tool on.
        timeout: Maximum execution time in seconds.

    Example:
        >>> class MyToolConfig(BaseConfig):
        ...     param1: int
        ...     param2: str
    """

    model_config = ConfigDict(
        extra="ignore",  # Ignore unknown fields
        validate_assignment=True,  # Validate on field updates
        use_enum_values=True,  # Serialize enums as values
        validate_default=True,  # Validate default values
    )

    verbose: bool = ConfigField(
        title="Verbose",
        default=False,
        description="Whether to print status messages",
        hidden=True,
    )

    @classmethod
    def reload_fields(cls) -> set[str]:
        """Return field names marked with ``reload_on_change=True``."""
        return {
            name
            for name, info in cls.model_fields.items()
            if (info.json_schema_extra or {}).get("reload_on_change", False)
        }

    device: str = ConfigField(
        title="Device",
        default="cpu",
        description="Device to run the tool on (e.g., 'cpu', 'cuda', 'cuda:0')",
        hidden=True,
    )

    timeout: int = ConfigField(
        title="Timeout",
        default=DEFAULT_TIMEOUT,
        ge=1,
        description="Maximum execution time in seconds",
        hidden=True,
    )
