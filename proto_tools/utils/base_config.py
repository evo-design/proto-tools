"""
proto_tools/utils/base_config.py

Base configuration class for all pydantic configs.
"""

from __future__ import annotations

import json
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
    include_in_key: bool = True,
    **kwargs,
) -> Any:
    """
    Custom Field wrapper that automatically adds metadata flags to json_schema_extra.

    Args:
        default (Any): Default value for the field. Use ``...`` for required fields.
        title (str): Human-readable title for UI display.
        description (str): Description of the field for documentation and UI tooltips.
        advanced (bool): If True, field appears in "Advanced" section of UI.
        hidden (bool): If True, field is hidden from UI completely.
        reload_on_change (bool): If True, changing this field between persistent
            worker calls triggers a subprocess restart.
        include_in_key (bool): If False, field is excluded from tool cache key
            generation. Fields that don't affect computation results (device,
            verbose, timeout) should set this to False.
        kwargs: All other standard Pydantic Field arguments (via ``**kwargs``).

    Usage:
        param: int = Field(default=42, title="Param", description="...", advanced=True)
    """
    json_schema_extra = kwargs.get("json_schema_extra", {})

    json_schema_extra["advanced"] = advanced
    json_schema_extra["hidden"] = hidden
    json_schema_extra["reload_on_change"] = reload_on_change
    json_schema_extra["include_in_key"] = include_in_key
    json_schema_extra["_field_type"] = "ConfigField"

    kwargs["json_schema_extra"] = json_schema_extra

    return PydanticField(default, title=title, description=description, **kwargs)


class BaseConfig(BaseModel):
    """
    Base configuration class for consistent behavior across all configs (tools, constraints, and generators).

    Attributes:
        verbose (bool): Whether to print status messages.
        device (str): Device to run the tool on.
        timeout (int): Maximum execution time in seconds.

    Properties:
        devices_per_instance: Number of GPUs each worker needs (default 1).
            Override in tool configs where the active model variant determines
            GPU requirements, e.g. a large checkpoint may need 2 GPUs while
            the small variant fits on 1. ``ToolPool`` reads this at dispatch
            time to group devices into worker slots.

    Example:
        >>> class MyToolConfig(BaseConfig):
        ...     param1: int
        ...     param2: str

        Multi-GPU override::

            class Evo2Config(BaseConfig):
                checkpoint: str = ConfigField(default="7b")

                @property
                def devices_per_instance(self) -> int:
                    return 4 if self.checkpoint == "40b" else 1
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
        include_in_key=False,
    )

    @classmethod
    def reload_fields(cls) -> set[str]:
        """Return field names marked with ``reload_on_change=True``."""
        return {
            name
            for name, info in cls.model_fields.items()
            if (info.json_schema_extra or {}).get("reload_on_change", False)
        }

    @classmethod
    def cache_exclude_fields(cls) -> set[str]:
        """Return field names marked with ``include_in_key=False``."""
        return {
            name
            for name, info in cls.model_fields.items()
            if not (info.json_schema_extra or {}).get("include_in_key", True)
        }

    def cache_key(self) -> str:
        """Deterministic string for cache key generation, excluding non-key fields."""
        model_dict = self.model_dump(
            exclude_none=True, exclude=self.cache_exclude_fields()
        )
        return json.dumps(model_dict, sort_keys=True, default=str)

    device: str = ConfigField(
        title="Device",
        default="cpu",
        description="Device to run the tool on (e.g., 'cpu', 'cuda', 'cuda:0')",
        hidden=True,
        include_in_key=False,
    )

    timeout: int = ConfigField(
        title="Timeout",
        default=DEFAULT_TIMEOUT,
        ge=1,
        description="Maximum execution time in seconds",
        hidden=True,
        include_in_key=False,
    )

    @property
    def devices_per_instance(self) -> int:
        """Number of GPUs each ToolPool worker needs for this configuration.

        ToolPool reads this at dispatch time to group its device list into
        worker slots.  For example, with ``devices=["cuda:0", "cuda:1",
        "cuda:2", "cuda:3"]`` and ``devices_per_instance == 2``, ToolPool
        creates 2 workers: one on ``cuda:0,cuda:1`` and one on
        ``cuda:2,cuda:3``.

        This lives on Config (not ToolSpec) because the required device count
        can depend on a runtime config value, e.g. a large model checkpoint
        may need 4 GPUs while the small variant fits on 1.  Override in tool
        config subclasses where this applies; the default is 1.
        """
        return 1

    def preprocess(self, inputs: BaseToolInput) -> BaseToolInput:
        """Transform inputs before tool execution. Override in subclasses."""
        return inputs
