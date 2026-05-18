"""proto_tools/utils/tool_io.py.

Base input/output classes for standardized tool results with metadata tracking.
"""

import json
import math
import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proto_tools.utils.compressed_array import is_compressed_array
from proto_tools.utils.export_names import sanitize_field

_REMOVED_UI_KWARGS = frozenset({"advanced", "hidden", "depends_on"})


def _reject_removed_ui_kwargs(field_helper: str, kwargs: dict[str, Any]) -> None:
    """Reject UI-presentation kwargs that belong in proto-ui overlays."""
    removed = sorted(_REMOVED_UI_KWARGS.intersection(kwargs))
    if not removed:
        return
    names = ", ".join(removed)
    raise TypeError(
        f"{field_helper} no longer accepts UI-presentation kwarg(s): {names}. "
        "Edit the proto-ui overlay for advanced/hidden/conditional visibility."
    )


def InputField(
    default: Any = ...,
    *,
    title: str | None = None,
    description: str | None = None,
    include_in_key: bool = True,
    xor_group: str | None = None,  # noqa: ARG001 — marker for sibling-field XOR groups; enforced by @model_validator per tool.
    **kwargs: Any,
) -> Any:
    """Custom Field wrapper for tool Input classes.

    Args:
        default (Any): Default value for the field. Use ``...`` for required fields.
        title (str | None): Human-readable title for UI display.
        description (str | None): Description of the field for documentation and UI tooltips.
        include_in_key (bool): If False, field is excluded from tool cache key
            generation.
        xor_group (str | None): Mutual-exclusion group name. Enforce at runtime
            with a ``@model_validator`` on the Input class.
        kwargs: All other standard Pydantic Field arguments (via ``**kwargs``).
    """
    _reject_removed_ui_kwargs("InputField", kwargs)
    json_schema_extra = kwargs.get("json_schema_extra", {})
    json_schema_extra["include_in_key"] = include_in_key
    json_schema_extra["_field_type"] = "InputField"
    kwargs["json_schema_extra"] = json_schema_extra
    return Field(default, title=title, description=description, **kwargs)


class ToolExecutionError(Exception):
    """Exception raised when a tool execution fails."""

    def __init__(self, message: str):
        """Initialize ToolExecutionError."""
        self.message = message
        super().__init__(message)


class MissingAssetError(ToolExecutionError):
    """Raised when a tool's required external asset isn't provisioned on disk.

    Signalled by a tool's setup script via the ``proto_resolve_asset_availability``
    helper (``standalone_helpers.sh``), which emits a
    ``[proto-tools] ASSET_NOT_AVAILABLE: <toolkit>:<asset_kind>`` sentinel and
    exits 64. ``ToolInstance`` recognises that sentinel and raises this
    exception in place of a generic ``RuntimeError`` so the test layer can
    convert it into a skip (rather than a hard failure) on machines that
    haven't been provisioned with gated weights, large databases, etc.

    Attributes:
        toolkit (str): Toolkit directory name (e.g. ``"alphafold3"``).
        asset_kind (str): Kind of missing asset (``"weights"``, ``"database"``,
            ``"dataset"``).
    """

    def __init__(self, toolkit: str, asset_kind: str, details: str = ""):
        """Initialize MissingAssetError."""
        self.toolkit = toolkit
        self.asset_kind = asset_kind
        self.message = f"{toolkit}: {asset_kind} not provisioned"
        if details:
            self.message = f"{self.message}\n{details}"
        # Skip ToolExecutionError.__init__
        Exception.__init__(self, self.message)


def _extra_dict(info: Any) -> dict[str, Any]:
    """Return ``json_schema_extra`` as a plain dict, defaulting to ``{}``."""
    extra = info.json_schema_extra
    return extra if isinstance(extra, dict) else {}


MetricValue = float | int | bool | list[float] | list[float | None] | list[list[float]]


class MetricSpec(TypedDict, total=False):
    """Declarative per-metric metadata describing range, type, and availability.

    All fields are optional (``total=False``). Subclasses of :class:`Metrics`
    declare a ``metric_spec: ClassVar[dict[str, MetricSpec]]`` mapping each
    metric name to one of these dicts.

    Attributes:
        description (str): Human-readable description of the metric.
        availability (str): When the metric is present in the output — e.g.
            ``"always"``, ``"multi-chain input only"``, ``"depends on model output"``.
        type (str): Expected value shape as a string — e.g. ``"float"``,
            ``"int"``, ``"bool"``, ``"list[float]"``, ``"list[float|None]"``,
            ``"list[list[float]]"``.
        min (float | None): Minimum valid value (element-wise for list types);
            ``None`` means unbounded below.
        max (float | None): Maximum valid value (element-wise for list types);
            ``None`` means unbounded above.
        unit (str | None): Unit string for the value, e.g. ``"REU"``, ``"Å"``, ``"bits"``.
    """

    description: str
    availability: str
    type: str
    min: float | None
    max: float | None
    unit: str | None


class Metrics(BaseModel):
    """Dual-access metric container for tool outputs.

    Stores arbitrary metric values via Pydantic's ``extra="allow"``, so values
    are accessible both as attributes (``m.plddt``) and as items (``m["plddt"]``).
    Subclasses may declare additional Pydantic fields for raw model outputs
    (logits, vocab, pdb_source, etc.) — those live in ``__dict__`` and do not
    leak into metric iteration. ``items``, ``keys``, ``values``, and
    ``__contains__`` walk ``__pydantic_extra__`` only.

    Subclasses declare ``metric_spec`` to document ranges, types, and
    availability. Validation against the spec is performed by test helpers
    (``tests/tool_infra_tests/_metric_helpers.py``), not at construction.

    Example:
        >>> class ESM2Score(Metrics):
        ...     metric_spec: ClassVar = {
        ...         "perplexity": {"type": "float", "min": 1.0, "max": None},
        ...         "log_likelihood": {"type": "float", "min": None, "max": 0.0},
        ...     }
        ...     logits: list[list[float]] | None = None  # raw model output
        >>> s = ESM2Score(perplexity=3.14, log_likelihood=-10.5, logits=[[0.1]])
        >>> s.perplexity
        3.14
        >>> s["perplexity"]
        3.14
        >>> s.logits
        [[0.1]]
        >>> list(s.items())
        [('perplexity', 3.14), ('log_likelihood', -10.5)]
        >>> "logits" in s
        False

    Attributes:
        metric_spec (ClassVar[dict[str, MetricSpec]]): Per-subclass declarative
            metadata describing each metric's type, range, and availability.
            Empty on the base class; subclasses override.
        primary_metric (str | None): Name of the metric that best summarizes
            the result overall (e.g. ``"avg_plddt"`` for AlphaFold2). Used by
            downstream UI and reporting to pick a headline value.
    """

    model_config = ConfigDict(extra="allow")

    metric_spec: ClassVar[dict[str, MetricSpec]] = {}
    primary_metric: str | None = None

    def __init__(self, **data: Any) -> None:
        """Accept arbitrary metric keyword arguments alongside declared fields.

        ``extra="allow"`` permits any extra keyword — this explicit ``__init__``
        signature tells type-checkers (mypy/pyright) that unknown kwargs are
        fine when instantiating a ``Metrics`` subclass, which they otherwise
        flag as ``call-arg`` errors.
        """
        super().__init__(**data)

    @model_validator(mode="before")
    @classmethod
    def _exclude_none_values(cls, data: Any) -> Any:
        """Strip ``None``-valued *extras* so absent metrics are truly absent.

        Callers that pass ``Metrics(chain_plddt=None)`` (e.g. when a model
        didn't emit a per-chain metric) get ``"chain_plddt" in metrics == False``
        rather than a sentinel ``None`` showing up in iteration. Declared
        fields are left alone so explicit ``None`` goes through Pydantic's
        normal default/assignment logic — otherwise an explicit
        ``primary_metric=None`` on a subclass with a non-``None`` default
        would silently fall back to the default.
        """
        if isinstance(data, dict):
            declared = set(cls.model_fields)
            return {k: v for k, v in data.items() if v is not None or k in declared}
        return data

    # ── Mapping-style access (walks __pydantic_extra__ only) ─────────────────

    def __getitem__(self, key: str) -> MetricValue:
        """Return the metric value for ``key``.

        Args:
            key (str): The metric name.

        Returns:
            MetricValue: The stored metric value.

        Raises:
            KeyError: If ``key`` is not present in the extras.
        """
        extra = self.__pydantic_extra__ or {}
        if key not in extra:
            raise KeyError(key)
        return extra[key]  # type: ignore[no-any-return]  # Pydantic types extras as dict[str, Any]

    def __setitem__(self, key: str, value: MetricValue) -> None:
        """Set a metric value by name.

        Args:
            key (str): The metric name.
            value (MetricValue): The metric value to store.
        """
        extra = self.__pydantic_extra__
        if extra is None:
            extra = {}
            object.__setattr__(self, "__pydantic_extra__", extra)
        extra[key] = value

    def __contains__(self, key: object) -> bool:
        """Whether ``key`` is a stored metric name."""
        return key in (self.__pydantic_extra__ or {})

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        """Iterate over stored metric names (not values)."""
        return iter(self.__pydantic_extra__ or {})

    def __len__(self) -> int:
        """Return the number of stored metrics."""
        return len(self.__pydantic_extra__ or {})

    def keys(self) -> KeysView[str]:
        """Return a view over stored metric names."""
        return (self.__pydantic_extra__ or {}).keys()

    def values(self) -> ValuesView[MetricValue]:
        """Return a view over stored metric values."""
        return (self.__pydantic_extra__ or {}).values()

    def items(self) -> ItemsView[str, MetricValue]:
        """Return a view over (name, value) pairs of stored metrics."""
        return (self.__pydantic_extra__ or {}).items()

    def get(self, key: str, default: Any = None) -> Any:
        """Return the metric value for ``key`` if present, else ``default``."""
        return (self.__pydantic_extra__ or {}).get(key, default)

    def update(self, other: "Metrics | Mapping[str, MetricValue]") -> None:
        """Merge metrics from ``other`` into this container in place.

        Args:
            other (Metrics | Mapping[str, MetricValue]): Metrics to merge.
                Existing keys are overwritten.
        """
        extra = self.__pydantic_extra__
        if extra is None:
            extra = {}
            object.__setattr__(self, "__pydantic_extra__", extra)
        src = other.__pydantic_extra__ or {} if isinstance(other, Metrics) else other
        extra.update(src)

    @property
    def primary_value(self) -> MetricValue | None:
        """Value of the metric named by ``primary_metric``, or ``None`` if unset.

        Convenience shortcut for ``self[self.primary_metric]`` with a fallback
        when either ``primary_metric`` is not declared or the named metric is
        not present in the container.
        """
        if self.primary_metric and self.primary_metric in self:
            return self[self.primary_metric]
        return None

    def validate_against_spec(self) -> None:
        """Assert spec-declared metrics are present, correctly typed, and in-range.

        Three checks, in order:

        1. **Presence**: every ``metric_spec`` entry whose ``availability`` is
           ``"always"`` must be a stored key on this container. Absent
           always-available metrics are a tool bug, not a quiet miss.
        2. **Type**: when the spec declares a ``type`` string (e.g. ``"float"``,
           ``"list[float|None]"``), the runtime value must match. Type
           mismatches raise immediately — they indicate a tool bug (e.g.
           emitting a numpy scalar instead of a native Python type).
        3. **Range**: each stored value whose name appears in ``metric_spec``
           is checked element-wise against the spec's ``min``/``max`` bounds.
           Keys not declared in the spec are skipped (permissive: tools may
           emit undeclared metrics). ``None`` entries in list-valued metrics
           (per-position gaps) are also skipped.

        This is **not** called at construction time by design: tools stay
        fast and emit whatever they emit. Validation is invoked explicitly
        from tests (see ``tests/tool_infra_tests/_metric_helpers.py``).

        Raises:
            AssertionError: With a precise message naming the offending metric
                and bound (or the missing always-available metric).
        """
        for name, spec in self.metric_spec.items():
            if spec.get("availability") == "always" and name not in self:
                raise AssertionError(
                    f"Metric {name!r} declared as always-available but missing from {type(self).__name__}"
                )
        for name, value in self.items():
            value_spec = self.metric_spec.get(name)
            if value_spec is None:
                continue
            type_str = value_spec.get("type")
            if type_str is not None:
                # Spec-driven dispatch: type check + bounds in one pass
                if type_str == "bool":
                    if not isinstance(value, bool):
                        raise AssertionError(
                            f"Metric {name!r} has type {type(value).__name__} but spec declares 'bool'"
                        )
                elif type_str in ("float", "int"):
                    _check_scalar_in_spec(name, value, value_spec)
                elif type_str.startswith("list"):
                    _check_list_in_spec(name, value, value_spec)
                else:
                    raise AssertionError(f"Metric {name!r} has unrecognized spec type {type_str!r}")
            else:
                # Legacy fallback: runtime type dispatch (no type checking, bounds only)
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    _check_scalar_in_spec(name, value, value_spec)
                elif isinstance(value, list):
                    _check_list_in_spec(name, value, value_spec)


def _check_scalar_in_spec(name: str, value: Any, spec: MetricSpec) -> None:
    """Validate a scalar metric's type and bounds against its spec."""
    type_str = spec.get("type")
    if type_str == "float" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        raise AssertionError(f"Metric {name!r} has type {type(value).__name__} but spec declares 'float'")
    if type_str == "int" and (not isinstance(value, int) or isinstance(value, bool)):
        raise AssertionError(f"Metric {name!r} has type {type(value).__name__} but spec declares 'int'")
    if not isinstance(value, (int, float)):
        return  # non-numeric reached via legacy path; nothing to bound-check
    if isinstance(value, float) and not math.isfinite(value):
        raise AssertionError(f"Metric {name!r}={value} is not finite (NaN/Inf rejected by spec)")
    min_v = spec.get("min")
    max_v = spec.get("max")
    if min_v is not None and value < min_v:
        raise AssertionError(f"Metric {name!r}={value} below declared min {min_v}")
    if max_v is not None and value > max_v:
        raise AssertionError(f"Metric {name!r}={value} above declared max {max_v}")


def _check_list_in_spec(name: str, value: Any, spec: MetricSpec) -> None:
    """Validate a list metric's type and element-wise bounds against its spec.

    When the spec declares a ``type`` string, element types are enforced:

    - ``"list[float]"``: elements must be numeric (``int`` or ``float``).
    - ``"list[float|None]"``: elements may be numeric or ``None``.
    - ``"list[list[float]]"``: elements must be lists of numerics.

    When no ``type`` is declared, falls back to permissive bounds-only
    checking (skips non-numeric entries silently).

    ``None`` entries in ``list[float|None]`` metrics are skipped during
    bounds checking (per-position gaps where the metric is undefined).
    """
    type_str = spec.get("type")
    if not isinstance(value, list):
        raise AssertionError(
            f"Metric {name!r} has type {type(value).__name__} but spec declares {type_str or 'list'!r}"
        )
    strict = type_str is not None
    allow_none = type_str == "list[float|None]"
    expect_inner_list = type_str is not None and type_str.startswith("list[list")
    # Pre-compute the unwrapped inner spec for list[list[float]] to avoid per-element allocation
    inner_spec: MetricSpec | None = (
        {"type": "list[float]", "min": spec.get("min"), "max": spec.get("max")} if expect_inner_list else None
    )
    min_v = spec.get("min")
    max_v = spec.get("max")
    for i, entry in enumerate(value):
        if entry is None:
            if strict and not allow_none:
                raise AssertionError(f"Metric {name!r} element at index {i} is None but spec declares {type_str!r}")
            continue
        if isinstance(entry, list):
            if strict and not expect_inner_list:
                raise AssertionError(f"Metric {name!r} element at index {i} is a list but spec declares {type_str!r}")
            _check_list_in_spec(f"{name}[{i}]", entry, inner_spec if inner_spec is not None else spec)
            continue
        if isinstance(entry, bool):
            if strict:
                raise AssertionError(
                    f"Metric {name!r} element at index {i} has type bool but spec declares {type_str!r}"
                )
            continue
        if not isinstance(entry, (int, float)):
            if strict:
                raise AssertionError(
                    f"Metric {name!r} element at index {i} has type {type(entry).__name__} "
                    f"but spec declares {type_str!r}"
                )
            continue
        if expect_inner_list:
            raise AssertionError(
                f"Metric {name!r} element at index {i} is {type(entry).__name__} but spec expects inner list"
            )
        if isinstance(entry, float) and not math.isfinite(entry):
            raise AssertionError(
                f"Metric {name!r} element at index {i} = {entry} is not finite (NaN/Inf rejected by spec)"
            )
        if min_v is not None and entry < min_v:
            raise AssertionError(f"Metric {name!r} element at index {i} = {entry} below declared min {min_v}")
        if max_v is not None and entry > max_v:
            raise AssertionError(f"Metric {name!r} element at index {i} = {entry} above declared max {max_v}")


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
        success (bool | None): Whether execution succeeded. ``True`` for any
            successful call. ``False`` only when the tool *failed* and
            ``PROTO_CAPTURE_ERRORS=1`` is set; on the default raise path
            failures raise instead of returning an output. See
            ``notes/error-handling.md``.
        warnings (list[str]): Non-fatal warnings generated during execution.
        errors (list[str]): Fatal error messages. Populated only when the
            tool *failed* and the wrapper is in capture mode; empty on
            success and on the default raise path. Each entry is
            ``"TypeName: message"`` followed by the formatted traceback.
            See ``notes/error-handling.md``.
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
    success: bool | None = Field(
        default=None,
        description="Whether execution succeeded. False only under PROTO_CAPTURE_ERRORS=1.",
    )

    # Optional metadata fields
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings generated during execution",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Fatal error messages, populated only under PROTO_CAPTURE_ERRORS=1.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional tool-specific metadata")

    model_config = ConfigDict(
        # Allow serialized computed fields to round-trip.
        extra="ignore",
        arbitrary_types_allowed=True,
        validate_assignment=True,
        use_enum_values=True,
        validate_default=True,
        str_strip_whitespace=True,
        # Surface subclass-narrowing bugs at the construction site.
        revalidate_instances="subclass-instances",
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

    @model_validator(mode="before")
    @classmethod
    def _warn_on_unexpected_fields(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data

        allowed = set(cls.model_fields)
        allowed.update(cls.model_computed_fields)

        fields = list(cls.model_fields.values()) + list(cls.model_computed_fields.values())
        aliases = [field.alias for field in fields if field.alias is not None]
        allowed.update(aliases)

        extras = sorted(set(data) - allowed)
        if extras:
            warnings.warn(
                f"{cls.__name__} ignoring unexpected output field(s): {', '.join(extras)}",
                stacklevel=2,
            )
        return data

    def __getattr__(self, name: str) -> Any:
        """Fallback attribute lookup.

        Three cases, in order:

        1. **Failure-mode escalation** — if ``name`` is a declared field and
           the tool run failed (``success is False``), raise
           :class:`ToolExecutionError` with the captured error messages.
        2. **Forwarding to Metrics** — walk set fields for any
           :class:`Metrics`-typed value and return ``value[name]`` if
           present. Lets ``output.plddt`` shortcut to
           ``output.metrics["plddt"]`` when the output has a ``Metrics`` field.
        3. **Not found** — raise :class:`AttributeError`.

        Args:
            name (str): Attribute name being looked up.

        Returns:
            Any: The forwarded value if found on a ``Metrics`` field.

        Raises:
            ToolExecutionError: If ``name`` is a declared field but the tool failed.
            AttributeError: If ``name`` resolves nowhere.
        """
        model_fields = self.__class__.model_fields

        # (1) Failure-mode escalation for declared fields.
        if name in model_fields:
            success = object.__getattribute__(self, "__dict__").get("success", True)
            if success is False:
                tool_id = object.__getattribute__(self, "__dict__").get("tool_id", type(self).__name__)
                errors = object.__getattribute__(self, "__dict__").get("errors", [])
                if errors:
                    raise ToolExecutionError(
                        f"{tool_id}: cannot read field {name!r} — tool failed: {' | '.join(errors)}"
                    )
                raise ToolExecutionError(f"{tool_id}: cannot read field {name!r} — tool failed (no errors recorded)")

        # (2) Forward to any Metrics-typed field whose container holds ``name``.
        for field_name in model_fields:
            try:
                value = object.__getattribute__(self, field_name)
            except AttributeError:
                continue
            if isinstance(value, Metrics) and name in value:
                return value[name]

        # (3) Not found.
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

        export_path = Path(export_path) / sanitize_field(name)
        self._export_output(export_path, file_format)

    def approx_equal(self, other: "BaseToolOutput", rtol: float = 1e-4, atol: float = 1e-5) -> None:
        """Assert that two tool outputs are approximately equal.

        Compares tool-specific output fields (excluding decorator metadata like
        ``tool_id``, ``execution_time``, ``timestamp``, etc.) with tolerance for
        floating-point differences.

        GPU tools using the same seed produce approximately — not exactly — identical
        results because CUDA operations like ``atomicAdd`` reduce floating-point values
        in non-deterministic order across threads. The results are mathematically
        equivalent but not bit-exact. Observed noise floors: ~1e-5 relative error on
        scalar model outputs (log-likelihoods, embeddings), up to ~1e-3 Å on structure
        coordinates, higher still for MoE models with non-deterministic kernels
        (progen3, alphagenome). Discrete outputs (sequences, labels, structure topology)
        are compared exactly; floating-point outputs (scores, coordinates, pLDDT) use
        the ``rtol`` and ``atol`` defaults below, calibrated to these observed noise
        floors. Tools that drift further are excluded via the ``_SEED_*_EXCLUDED_KEYS``
        sets in ``tests/seed_tests/test_seed_reproducibility.py``.

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
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            raise AssertionError(f"Float mismatch at {path}: {a} != {b} (rtol={rtol}, atol={atol})")
    elif isinstance(a, dict):
        # Compressed array dicts: decompress both and compare as numpy arrays.
        if is_compressed_array(a) and is_compressed_array(b):
            import numpy as np

            from proto_tools.utils.compressed_array import decompress_array

            arr_a = decompress_array(a)
            arr_b = decompress_array(b)
            if arr_a.shape != arr_b.shape:
                raise AssertionError(f"Shape mismatch at {path}: {arr_a.shape} != {arr_b.shape}")
            if not np.allclose(arr_a, arr_b, rtol=rtol, atol=atol, equal_nan=True):
                max_diff = float(np.max(np.abs(arr_a - arr_b)))
                raise AssertionError(f"Array mismatch at {path}: max_diff={max_diff} (rtol={rtol}, atol={atol})")
            return

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
        # float drift in nested ``Metrics`` containers is compared with tolerance
        # instead of falling through to bit-exact ``BaseModel.__eq__``. Classes
        # that need custom comparison logic can override this by defining their
        # own ``approx_equal`` method (handled by the branch above).
        for field_name in type(a).model_fields:
            _approx_equal_values(
                getattr(a, field_name),
                getattr(b, field_name),
                rtol,
                atol,
                f"{path}.{field_name}",
            )
        # ``extra="allow"`` models (e.g. ``Metrics`` subclasses) store metric values
        # in ``__pydantic_extra__``. Compare those too so metric drift is caught.
        extra_a = getattr(a, "__pydantic_extra__", None) or {}
        extra_b = getattr(b, "__pydantic_extra__", None) or {}
        if extra_a.keys() != extra_b.keys():
            raise AssertionError(f"Extras keys differ at {path}: {set(extra_a) ^ set(extra_b)}")
        for key in extra_a:
            _approx_equal_values(extra_a[key], extra_b[key], rtol, atol, f"{path}.{key}")
    elif a != b:
        raise AssertionError(f"Value mismatch at {path}: {a!r} != {b!r}")


__all__ = [
    "BaseToolInput",
    "BaseToolOutput",
    "InputField",
    "MetricSpec",
    "MetricValue",
    "Metrics",
    "MissingAssetError",
    "ToolExecutionError",
]
