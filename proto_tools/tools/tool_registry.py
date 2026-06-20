"""proto_tools/tools/tool_registry.py.

Tool registry for managing tool discovery and schema generation.

Provides a decorator-based API for registering tools with metadata and
automatic schema generation for API integration.
"""

from __future__ import annotations

import contextlib
import difflib
import logging
import math
import os
import re
import time
import traceback
import warnings
from collections.abc import Callable, MutableMapping
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import yaml
from pydantic import BaseModel, Field, field_serializer

if TYPE_CHECKING:
    from proto_tools.utils.tool_docs import ModelDoc, ReadmeSections, ToolReadmeEntry

logger = logging.getLogger(__name__)

# Path to tools directory for citation file discovery
TOOLS_DIR = Path(__file__).parent

# List of warning message substrings to ignore (noisy warnings from dependencies)
IGNORED_WARNING_SUBSTRINGS = [
    "get_autocast_gpu_dtype",
    "get_autocast_dtype",
    # ResourceWarning for unclosed subprocess pipe fds — reaped by the OS, no impact on tool output.
    "unclosed file",
]

# Retry configuration for transient failures (network drops, subprocess crashes)
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # Base delay in seconds (exponential backoff: 2s, 4s, 8s)

# TimeoutError excluded: retrying with the same limit would just time out again
_RETRYABLE_EXCEPTIONS = (ConnectionError,)

# Set PROTO_CAPTURE_ERRORS=1 to capture tool exceptions into success=False outputs instead of raising.
_CAPTURE_ERRORS_ENV_VAR = "PROTO_CAPTURE_ERRORS"


def _is_transient_gpu_acquisition_error(exc: BaseException) -> bool:
    """Return True for a retryable GPU context-acquisition failure under Exclusive_Process mode.

    cuInit / "No visible GPU devices" failures right after a device handoff are a race (the prior
    process has not released its exclusive CUDA context yet), not a real fault. Gated on
    Exclusive_Process so the same error still fails fast under Default mode, where it signals a
    genuine misconfiguration or dead GPU.
    """
    from proto_tools.utils.device import is_exclusive_process_mode, is_gpu_acquisition_error

    return is_gpu_acquisition_error(exc) and is_exclusive_process_mode()


def _should_capture_errors() -> bool:
    """Whether tool exceptions should be captured into the output instead of raised."""
    return os.environ.get(_CAPTURE_ERRORS_ENV_VAR, "0") == "1"


from proto_tools.utils import BaseConfig
from proto_tools.utils.device import (
    parse_device_string,
    validate_device_allocation,
)
from proto_tools.utils.progress import (
    reset_current_tool_function,
    set_current_tool_function,
)
from proto_tools.utils.tool_cache import (
    CacheStripResult,
    _generate_cache_key,
    _program_tool_cache,
    _serialize_for_cache_key,
    cache_stitch_items,
    cache_store_items,
    cache_strip_items,
    deduplicate_items,
)
from proto_tools.utils.tool_instance import ToolInstance
from proto_tools.utils.tool_io import BaseToolInput, BaseToolOutput, Metrics, MissingAssetError

ToolDispatchBackend = Callable[[str, BaseToolInput, BaseConfig], BaseToolOutput | None]

# ---------------------------------------------------------------------------
# Parallel iterable-field helpers
#
# A tool's ``iterable_input_fields`` declares index-parallel input lists (primary
# first, e.g. ``["complexes", "msas"]``). The framework slices/dedups/keys them as
# a group so a parallel sibling (``msas``) stays aligned with the primary and is
# part of the cache key. Single-field tools collapse to today's behavior: a
# one-element bundle keys identically to the bare item.
# ---------------------------------------------------------------------------


class _ParallelItem:
    """One row across a tool's parallel iterable fields, with a composite cache key."""

    __slots__ = ("values",)

    def __init__(self, values: tuple[Any, ...]) -> None:
        self.values = values

    def cache_key(self) -> str:
        """Serialize every parallel field's value so dedup/cache keys reflect all of them."""
        return "\x1f".join(_serialize_for_cache_key(v) for v in self.values)


def _active_iter_fields(inputs: Any, spec: ToolSpec) -> list[str]:
    """Parallel iterable fields present (non-None) on ``inputs``, primary first."""
    return [f for f in (spec.iterable_input_fields or []) if getattr(inputs, f, None) is not None]


def _zip_iter_items(inputs: Any, fields: list[str]) -> list[_ParallelItem]:
    """Zip the parallel field lists into per-row ``_ParallelItem`` bundles.

    Raises ``ValueError`` on a length mismatch. This dedup/cache path runs before a
    tool's ``config.preprocess`` (where the per-tool friendly check lives), so without
    this guard a misaligned sibling would surface as a bare ``IndexError`` (too short)
    or be silently truncated to the primary length (too long).
    """
    cols = [list(getattr(inputs, f)) for f in fields]
    n = len(cols[0])
    for f, col in zip(fields[1:], cols[1:], strict=True):
        if len(col) != n:
            raise ValueError(
                f"Parallel iterable field {f!r} (len {len(col)}) is not aligned with primary {fields[0]!r} (len {n})."
            )
    return [_ParallelItem(tuple(col[i] for col in cols)) for i in range(n)]


def _apply_iter_items(inputs: Any, fields: list[str], items: list[_ParallelItem]) -> Any:
    """Rebuild ``inputs`` with each parallel field set from the bundles (kept aligned)."""
    update: dict[str, Any] = {f: [it.values[j] for it in items] for j, f in enumerate(fields)}
    return inputs.model_copy(update=update)


class ToolSpec(BaseModel):
    """Specification for a registered tool.

    Stores tool metadata in the registry.

    Attributes:
        key (str): Internal identifier (e.g., ``"blast-search"``).
        label (str): External UI display name (e.g., ``"BLAST Search"``).
        category (str): Tool category (e.g., ``"gene_annotation"``).
        description (str): Detailed description of tool functionality.
        uses_gpu (bool): Whether this tool requires a GPU.
        gpu_only (bool): Tool cannot run on CPU. Direct ``device="cpu"`` dispatch
            raises ``ValueError``, and LRU eviction restarts the worker instead
            of offloading. Implies ``uses_gpu=True``. This should generally be False:
            it is only used for models that have issues with CPU loading.
        device_count (str): Expected device count requirement
            (e.g., ``"1"``, ``"1-2"``, ``">=1"``).
        config_model (type[BaseModel]): Pydantic model for configuration validation
            and schema generation.
        input_model (type[BaseToolInput]): Pydantic model class for primary input validation.
        output_model (type[BaseToolOutput]): Pydantic model class for tool output validation.
        metrics_model (type[Metrics] | None): Optional Metrics subclass the tool emits, carrying
            the declarative ``metric_spec`` and ``primary_metric`` surfaced by tool_docs.
        function (Callable[..., Any]): The wrapped tool function.
        source_file (Path): Path to the source file where the tool function is defined.
        example_input (Callable[[], BaseToolInput] | None): Factory returning a minimal
            valid input for testing.
        iterable_input_fields (list[str] | None): Index-parallel input list field names
            (primary first; e.g. ``["complexes", "msas"]`` or a single-field ``["sequences"]``).
            Sliced, deduped, and cache-keyed together so siblings stay aligned to the primary.
            ``None`` for non-iterable tools.
        iterable_output_field (str | None): Output field name containing the iterable list
            of results (for ToolPool fan-out).
        cacheable (bool): Declares that this tool's output is a
            deterministic function of (input, config), making it eligible
            for the program-scoped cache and the framework's iterable
            optimizations (dedup, ``post_process_iterable``). Does not
            command caching: actual caching depends on an active program
            cache and (for stochastic tools) a seed being set.
        stochastic (bool): Outputs depend on ``config.seed``. Three runtime
            effects: (1) unseeded calls skip the cache; (2) iterable
            dispatches skip dedup so duplicate items reach the tool and
            diverge via per-item RNG advancement; (3) cacheable seeded
            calls use the whole-call cache rather than per-item cache.
        post_process_iterable (Callable[[list[Any]], None] | None): Optional in-place
            hook invoked on the stitched ``iterable_output_field`` list after cache
            reconciliation and dedup expansion. Use for batch-level post-processing
            (e.g. UMAP projection of an embedding batch) that must see the full
            request, not the per-item cached subset visible inside the tool function.
    """

    # Public fields - exposed in API
    key: str = Field(description="Internal identifier (e.g., 'blast-search')")
    label: str = Field(description="External UI display name (e.g., 'BLAST Search')")
    category: str = Field(description="Tool category (e.g., 'gene_annotation')")
    description: str = Field(description="Detailed description of tool functionality")
    uses_gpu: bool = Field(default=False, description="Whether this tool requires a GPU")
    gpu_only: bool = Field(
        default=False,
        description=(
            "Tool cannot run on CPU. Direct device='cpu' dispatch raises "
            "ValueError; LRU eviction restarts the worker instead of offloading. "
            "Implies uses_gpu=True."
        ),
    )
    pin_visible_devices: bool = Field(
        default=False,
        description=(
            "Pin the worker's CUDA_VISIBLE_DEVICES to its assigned physical GPU(s); "
            "the worker addresses them as local cuda:0..N-1 and respawns on a "
            "physical-device change. For JAX tools, whose runtime spans every "
            "visible GPU. Implies uses_gpu=True."
        ),
    )
    device_count: str = Field(
        default="1",
        description="Expected device count requirement (e.g., '1', '1-2', '>=1', '<=2')",
    )

    # Configuration model - serialized as JSON Schema
    config_model: type[BaseModel] = Field(
        description="Pydantic model for configuration validation and schema generation"
    )

    # Private fields - excluded from serialization
    input_model: type[BaseToolInput] = Field(exclude=True)
    output_model: type[BaseToolOutput] = Field(exclude=True)
    metrics_model: type[Metrics] | None = Field(default=None, exclude=True)
    function: Callable[..., Any] = Field(exclude=True)
    source_file: Path = Field(
        exclude=True,
        description="Path to the source file where the tool function is defined",
    )
    example_input: Callable[[], BaseToolInput] | None = Field(
        default=None,
        exclude=True,
        description="Factory returning a minimal valid input for testing",
    )
    iterable_input_fields: list[str] | None = Field(
        default=None,
        exclude=True,
        description="Index-parallel input list fields (primary first); sliced/keyed together, None entries skipped",
    )
    iterable_output_field: str | None = Field(
        default=None,
        exclude=True,
        description="Output field name containing the iterable list of results (for ToolPool fan-out)",
    )
    cacheable: bool = Field(
        default=False,
        exclude=True,
        description=(
            "Declares that this tool's output is a deterministic function of "
            "(input, config), making it eligible for the program-scoped cache "
            "and the framework's iterable optimizations (dedup, "
            "post_process_iterable). Does not command caching: actual "
            "caching depends on an active program cache and (for stochastic "
            "tools) a seed being set."
        ),
    )
    stochastic: bool = Field(
        default=False,
        exclude=True,
        description="Whether this tool's outputs depend on the random seed",
    )
    post_process_iterable: Callable[[list[Any]], None] | None = Field(
        default=None,
        exclude=True,
        description=(
            "Optional in-place hook invoked on the stitched ``iterable_output_field`` after "
            "cache reconciliation and dedup expansion. Use for batch-level post-processing "
            "(e.g. UMAP projection of an embedding batch) that must see the full request — "
            "not the per-item cached subset visible inside the tool function."
        ),
    )

    model_config = {
        "arbitrary_types_allowed": True,
    }

    @property
    def has_standalone_env(self) -> bool:
        """Whether this tool ships a ``standalone/`` directory with an isolated env.

        ``True`` for tools that run in a micromamba-based subprocess worker
        (most GPU/ML tools). ``False`` for pure-Python in-process tools
        (e.g. ``random_nucleotide``, ``random_protein``, simple sequence
        utilities) — these cannot be used with ``ToolInstance.persist_tool``.
        """
        return (self.source_file.parent / "standalone").is_dir()

    @property
    def local_cpu(self) -> bool:
        """Whether this tool runs trivially in-process on CPU.

        ``True`` iff ``uses_gpu=False`` and ``has_standalone_env=False``;
        ``device='cloud'`` is a no-op for these tools. Opt out by giving the
        tool a ``standalone/`` directory.
        """
        return not self.uses_gpu and not self.has_standalone_env

    @field_serializer("config_model")
    def serialize_config_model(self, config_model: type[BaseModel]) -> dict[str, Any]:
        """Serialize config_model as standard JSON Schema."""
        return config_model.model_json_schema()


class ToolRegistry:
    """Registry for tool discovery and schema generation.

    Provides discovery, schema generation, and factory methods for tools.
    Registration happens at import time via the @tool decorator.

    Public Methods:
    - register(): Decorator to register tool functions
    - list_all(): List tools with metadata and schemas
    - get(): Get tool spec by key
    - get_config_schema(): Get JSON schema for tool configuration
    - count(): Get number of registered tools

    Examples:
        Registration:
        >>> @tool(
        ...     key="blast-search",
        ...     label="BLAST Search",
        ...     input_class=BlastSearchInput,
        ...     config_class=BlastSearchConfig,
        ...     output_class=BlastSearchOutput,
        ...     description="Protein similarity search using BLAST",
        ... )
        ... def run_blast_search(inputs: BlastSearchInput, config: BlastSearchConfig) -> BlastSearchOutput:
        ...     pass

        API Usage:
        >>> tools = ToolRegistry.list_all()
        >>> schema = ToolRegistry.get_config_schema("blast-search")

        Direct Usage:
        >>> from proto_tools.tools.sequence_alignment import (
        ...     run_blast_search,
        ...     BlastSearchInput,
        ...     BlastSearchConfig,
        ... )
        >>> result = run_blast_search(BlastSearchInput(query="MVLSP"), BlastSearchConfig(database="/data/nr"))
    """

    _registry: ClassVar[dict[str, ToolSpec]] = {}
    _dispatch_backend: ClassVar[ToolDispatchBackend | None] = None

    @classmethod
    def configure_dispatch_backend(cls, backend: ToolDispatchBackend | None) -> None:
        """Install an early dispatch hook for registered tool calls.

        The hook runs after input/config coercion but before local device
        validation, cache lookup, preprocessing, ToolPool fan-out, or standalone
        dispatch. Return a ``BaseToolOutput`` to short-circuit local execution,
        or ``None`` to fall through to the normal dispatch path.

        The ``instance`` argument is not forwarded through this hook.
        """
        cls._dispatch_backend = backend

    @classmethod
    def clear_dispatch_backend(cls) -> None:
        """Clear any configured early dispatch backend."""
        cls._dispatch_backend = None

    @classmethod
    def dispatch_backend_configured(cls) -> bool:
        """Return whether an early dispatch backend is currently installed."""
        return cls._dispatch_backend is not None

    @classmethod
    def register(
        cls,
        key: str,
        label: str,
        category: str,
        input_class: type[BaseToolInput],
        config_class: type[BaseConfig],
        output_class: type[BaseToolOutput],
        description: str,
        metrics_class: type[Metrics] | None = None,
        uses_gpu: bool = False,
        gpu_only: bool = False,
        pin_visible_devices: bool = False,
        device_count: str = "1",
        example_input: Callable[[], BaseToolInput] | None = None,
        iterable_input_fields: list[str] | None = None,
        iterable_output_field: str | None = None,
        cacheable: bool = False,
        stochastic: bool = False,
        post_process_iterable: Callable[[list[Any]], None] | None = None,
    ) -> Callable[[Callable[..., BaseToolOutput]], Callable[..., BaseToolOutput]]:
        """Decorator to register a tool function and wrap execution with metadata tracking.

        This decorator:
        1. Registers the tool in the registry with its metadata
        2. Wraps the function to automatically track execution time
        3. Handles errors and returns standardized output with success/failure status
        4. Populates metadata fields (tool_id, execution_time, success, timestamp)
        5. Validates device allocation against tool requirements

        Args:
            key (str): Unique identifier (e.g., "blast-search", "esm3-embedding")
            label (str): Readable display name (e.g., "BLAST Search", "ESM3 Embedding")
            category (str): Tool category matching directory name (e.g., "gene_annotation")
            input_class (type[BaseToolInput]): Pydantic model class for primary input validation
            config_class (type[BaseConfig]): Pydantic model class for tool configuration validation
            output_class (type[BaseToolOutput]): Pydantic model class for tool output validation
            description (str): Readable description
            metrics_class (type[Metrics] | None): Optional Metrics subclass the tool emits on its
                output. Recorded on the spec so tool_docs can surface the metric_spec table.
            uses_gpu (bool): Whether this tool requires a GPU for execution
            gpu_only (bool): Tool cannot run on CPU. Direct ``device="cpu"``
                dispatch raises ``ValueError``, and LRU eviction restarts the
                worker instead of offloading. Implies ``uses_gpu=True``.
            pin_visible_devices (bool): Pin the worker's ``CUDA_VISIBLE_DEVICES``
                to its assigned physical GPU(s); the worker sees them as local
                ``cuda:0..N-1`` and respawns on a physical-device change instead
                of moving in-process. For JAX tools, whose runtime initializes a
                context on every visible GPU. Implies ``uses_gpu=True``.
            device_count (str): Expected device count (e.g., "1", "1-2", ">=1", "<=2").
                Validates allocation: errors on under-allocation, warns on over-allocation.
            example_input (Callable[[], BaseToolInput] | None): Factory returning a minimal valid
                input for testing and examples.
            iterable_input_fields (list[str] | None): Index-parallel input list field names
                (primary first) batched/sliced/keyed together for ToolPool fan-out and per-item
                caching; a None field is skipped. Use a single-element list for an ordinary
                iterable (e.g. ``["sequences"]``), or several aligned lists for a parallel group
                (e.g. structure-prediction ``["complexes", "msas"]``).
            iterable_output_field (str | None): Output field name containing the iterable list of
                results for ToolPool fan-out and per-item caching.
            cacheable (bool): Declares that this tool's output is a
                deterministic function of (input, config), making it
                eligible for the program-scoped cache and the framework's
                iterable optimizations (dedup, ``post_process_iterable``).
                Does not command caching: actual caching depends on an
                active program cache and (for stochastic tools) a seed
                being set.
            stochastic (bool): Outputs depend on ``config.seed``. Three
                runtime effects: (1) unseeded calls skip the cache; (2)
                iterable dispatches skip dedup so duplicate items reach the
                tool and diverge via per-item RNG advancement; (3) cacheable
                seeded calls use the whole-call cache rather than per-item
                cache.
            post_process_iterable (Callable[[list[Any]], None] | None): Optional
                in-place hook invoked on the stitched ``iterable_output_field``
                list after cache reconciliation. Required when a derived field
                (e.g. UMAP projection) is meaningful only across the full request
                — running it inside the tool function would only see the uncached
                subset and produce incoherent results once stitched against
                cached items.

        Returns:
            Callable[[Callable[..., BaseToolOutput]], Callable[..., BaseToolOutput]]: Decorator
                that wraps the function with metadata tracking.
        """

        def decorator(
            func: Callable[..., BaseToolOutput],
        ) -> Callable[..., BaseToolOutput]:
            cls._check_duplicate(key, func.__name__)

            if (iterable_input_fields is None) != (iterable_output_field is None):
                raise ValueError(
                    f"@tool({key!r}): iterable_input_fields and iterable_output_field must both be set or both None"
                )

            if gpu_only and not uses_gpu:
                raise ValueError(f"@tool({key!r}): gpu_only=True requires uses_gpu=True")

            if pin_visible_devices and not uses_gpu:
                raise ValueError(f"@tool({key!r}): pin_visible_devices=True requires uses_gpu=True")

            source_file = Path(func.__code__.co_filename)

            @wraps(func)
            def wrapper(
                inputs: BaseToolInput,
                config: BaseConfig | None = None,
                instance: str | ToolInstance | None = None,
            ) -> BaseToolOutput:
                """Wrapper that tracks execution and populates metadata.

                ``instance`` is forwarded to ``ToolInstance.dispatch``. A string
                is a **reference** to a cached instance registered via
                ``ToolInstance.get(instance_name=...)`` or
                ``ToolInstance.persist_tool(instance_name=...)``; unknown strings
                raise ``ValueError`` unless the call is inside a
                ``ToolInstance.persist()`` context (which authorizes auto-creation
                under the given name).
                """
                _func_token = set_current_tool_function(func.__name__)
                try:
                    return _wrapper_body(inputs, config, instance)
                finally:
                    reset_current_tool_function(_func_token)

            def _wrapper_body(
                inputs: BaseToolInput,
                config: BaseConfig | None,
                instance: str | ToolInstance | None,
            ) -> BaseToolOutput:
                # Auto-configure logging if no handlers exist (one-time, O(1) after first call)
                from proto_tools.utils.logging_config import _auto_configure_logging

                _auto_configure_logging()

                # Coerce inputs/config to the tool's expected classes if a parent
                # class was passed. Only explicitly-set fields are forwarded so
                # child class defaults take precedence over parent defaults.
                inputs = _coerce_model(inputs, input_class, key, "input")  # type: ignore[assignment]
                if config is None:
                    config = config_class()
                    logger.debug(f"Tool {key}: config not provided, using default config")
                else:
                    config = _coerce_model(config, config_class, key, "config")  # type: ignore[assignment]

                start_time = time.time()
                last_exception: Exception | None = None
                last_traceback = ""
                warning_list = []

                # Prefer registry metadata so runtime spec edits are honored.
                spec = cls._registry.get(key)
                cache_enabled = spec.cacheable if spec is not None else cacheable
                gpu_only_flag = spec.gpu_only if spec is not None else gpu_only
                pin_visible_devices_flag = spec.pin_visible_devices if spec is not None else pin_visible_devices
                stochastic_tool = spec.stochastic if spec is not None else stochastic
                # Unseeded calls to stochastic tools skip cache so repeat dispatches advance RNG.
                runtime_cacheable = cache_enabled and (not stochastic_tool or config.seed is not None)

                deduped = None
                original_items = None
                strip = None
                whole_cache_key = None

                def _finish_dispatched(dispatched: BaseToolOutput) -> BaseToolOutput:
                    dispatched.tool_id = key
                    if dispatched.success is None:
                        dispatched.success = True
                    dispatched.execution_time = time.time() - start_time
                    return _post_dispatch_cache_and_expand(
                        key,
                        spec,
                        runtime_cacheable,
                        stochastic_tool,
                        dispatched,
                        strip,
                        deduped,
                        original_items,
                        whole_cache_key,
                    )

                backend = cls._dispatch_backend
                if backend is not None:
                    try:
                        dispatched: object | None = backend(key, inputs, config)
                    except Exception as e:
                        logger.error(
                            "Tool %s: dispatch backend raised %s: %s",
                            key,
                            type(e).__name__,
                            e,
                        )
                        return _make_error_output_or_raise(
                            output_class,
                            key,
                            start_time,
                            e,
                            traceback.format_exc(),
                        )
                    if dispatched is not None:
                        if not isinstance(dispatched, BaseToolOutput):
                            error = TypeError(
                                f"Tool {key!r} dispatch backend returned {type(dispatched).__name__}, "
                                "expected BaseToolOutput or None."
                            )
                            return _make_error_output_or_raise(
                                output_class,
                                key,
                                start_time,
                                error,
                                "".join(traceback.format_stack()),
                            )
                        return _finish_dispatched(dispatched)

                # Validate device allocation against tool requirements.
                # device="cloud" delegates all resource allocation to the cloud
                # service, so local validation is skipped.
                if hasattr(config, "device"):
                    device_str = str(config.device)
                    if device_str == "cloud":
                        from proto_tools.cloud import (
                            cloud_unhostable_message,
                            dispatch_to_cloud,
                            is_cloud_hostable,
                        )

                        # local_cpu tools have nothing to offload — rewrite device and fall through.
                        if spec is not None and spec.local_cpu:
                            logger.debug(
                                "Tool %s: device='cloud' is a no-op for local_cpu tools; running in-process",
                                key,
                            )
                            config = config.model_copy(update={"device": "cpu"})
                            device_str = "cpu"
                        # Fail fast if the tool's license bars it from Proto's cloud; after the local_cpu no-op.
                        elif not is_cloud_hostable(key):
                            raise ValueError(cloud_unhostable_message(key))
                        # Fail fast on a config the cloud can't run (e.g. a local database/file).
                        elif (reason := config.cloud_unsupported_reason()) is not None:
                            raise ValueError(f"{key}: {reason}")
                        else:
                            try:
                                dispatched = dispatch_to_cloud(key, inputs, config)
                            except Exception as e:
                                logger.error(
                                    "Tool %s: cloud dispatch raised %s: %s",
                                    key,
                                    type(e).__name__,
                                    e,
                                )
                                return _make_error_output_or_raise(
                                    output_class,
                                    key,
                                    start_time,
                                    e,
                                    traceback.format_exc(),
                                )
                            return _finish_dispatched(dispatched)
                    if gpu_only_flag and device_str == "cpu":
                        raise ValueError(
                            f"Tool {key!r} is gpu_only and rejects device='cpu'; use 'cuda', 'cuda:N', or 'cudaxN'"
                        )
                    if device_str != "cpu" and not uses_gpu:
                        logger.warning(
                            "Tool %s does not use a GPU but device=%r was requested; "
                            "the tool will run on CPU regardless",
                            key,
                            device_str,
                        )
                    elif device_str != "cpu":
                        device_spec = parse_device_string(device_str)
                        validate_device_allocation(device_spec.count, device_count, key)

                # --- Dedup + Cache (cacheable tools only) ---
                if runtime_cacheable and spec and spec.iterable_input_fields and not stochastic_tool:
                    assert spec.iterable_output_field is not None  # validated at registration
                    # Iterable path (deterministic only): zip the parallel iterable group into per-row
                    # bundles, dedup, then strip cached items. Stochastic iterables skip to the
                    # whole-call cache so duplicates can diverge via per-item RNG.
                    _iter_fields = _active_iter_fields(inputs, spec)
                    original_items = _zip_iter_items(inputs, _iter_fields)
                    if len(original_items) > 1:
                        deduped = deduplicate_items(original_items, key_fn=_serialize_for_cache_key)
                        if len(deduped.unique_items) < len(original_items):
                            inputs = _apply_iter_items(inputs, _iter_fields, deduped.unique_items)
                            logger.debug(
                                "Tool %s: dedup %d → %d unique items",
                                key,
                                len(original_items),
                                len(deduped.unique_items),
                            )

                    # Strip cached items from the (possibly deduped) input
                    current_items = _zip_iter_items(inputs, _iter_fields)
                    strip = cache_strip_items(key, current_items, config)
                    if strip is not None and strip.all_cached:
                        logger.debug(
                            "[Iterable Cache] %s: full cache hit, skipping dispatch",
                            key,
                        )
                        cached_list = [strip.cached_results[i] for i in range(len(current_items))]
                        # Expand dedup if needed
                        if deduped and len(deduped.unique_items) < len(original_items):
                            cached_list = [cached_list[uid] for _, uid in deduped.index_map]
                        if spec.post_process_iterable is not None:
                            spec.post_process_iterable(cached_list)
                        cache_kwargs: dict[str, Any] = {
                            "tool_id": key,
                            "execution_time": 0.0,
                            "success": True,
                            "timestamp": datetime.now(),
                            "warnings": [],
                            "errors": [],
                            "metadata": {},
                            spec.iterable_output_field: cached_list,
                        }
                        return output_class.model_construct(**cache_kwargs)

                    # Narrow inputs to uncached items only (all parallel fields kept aligned)
                    if strip is not None and strip.uncached_items:
                        inputs = _apply_iter_items(inputs, _iter_fields, strip.uncached_items)

                elif runtime_cacheable:
                    # Whole-output path: check full cache
                    cache = _program_tool_cache.get()
                    if cache is not None:
                        whole_cache_key = _generate_cache_key(key, inputs, config)
                        cached: BaseToolOutput | None = cache.get(key, whole_cache_key)
                        if cached is not None:
                            logger.debug("[Cache Hit] %s: using cached result", key)
                            return cached

                # Share one worker across preprocess + main dispatch. Skipped for
                # in-process tools (no worker to share).
                toolkit = source_file.parent.name
                has_standalone_env = spec is not None and spec.has_standalone_env
                has_custom_preprocess = type(config).preprocess is not BaseConfig.preprocess
                # ``instance`` may be a ToolInstance or a string cache key.
                # Fail fast here (before preprocess runs) if a string references
                # an unregistered instance outside of persist mode — dispatch would
                # raise the same error later, but catching it here avoids running
                # preprocess before the inevitable failure.
                from proto_tools.utils.tool_instance import _resolve_instance_or_raise

                resolved_instance: ToolInstance | None = _resolve_instance_or_raise(instance, toolkit)
                needs_scope = has_standalone_env and (has_custom_preprocess or resolved_instance is not None)
                auto_persist_ctx: contextlib.AbstractContextManager[None] = (
                    ToolInstance._auto_persist_scope(toolkit, instance=resolved_instance)
                    if needs_scope
                    else contextlib.nullcontext()
                )

                with auto_persist_ctx:
                    inputs = config.preprocess(inputs)

                    # --- Dispatch (pool or local) ---

                    # Check for active ToolPool (transparent parallel dispatch)
                    from proto_tools.utils.tool_pool import (
                        get_active_pool,
                        is_pool_executing,
                    )

                    pool = get_active_pool()
                    if pool is not None and not is_pool_executing() and spec and spec.iterable_input_fields is not None:
                        result = pool._parallel_dispatch(key, func, inputs, config)
                        result.tool_id = key
                        if result.success is None:
                            result.success = True
                        result.execution_time = time.time() - start_time
                        return _post_dispatch_cache_and_expand(
                            key,
                            spec,
                            runtime_cacheable,
                            stochastic_tool,
                            result,
                            strip,
                            deduped,
                            original_items,
                            whole_cache_key,
                        )

                    # Scale effective_timeout() by the iterable batch size.
                    if spec is not None and spec.iterable_input_fields is not None:
                        effective = config.effective_timeout()
                        if effective is not None:
                            items = getattr(inputs, spec.iterable_input_fields[0], None)
                            n_items = len(items) if items is not None else 1
                            if n_items > 1:
                                config = config.model_copy(update={"timeout": effective * n_items})

                    # Carry per-invocation flags (key, gpu_only, pin_visible_devices) to ToolInstance for eviction/dispatch.
                    from proto_tools.utils.tool_instance import _current_tool_invocation

                    for attempt in range(1 + MAX_RETRIES):
                        try:
                            if attempt == 0:
                                logger.debug(f"Tool {key}: starting execution")

                            # Capture warnings during execution
                            with warnings.catch_warnings(record=True) as warning_list:
                                # Ensure all warnings are captured
                                warnings.simplefilter("always")

                                # Execute the tool function (ContextVar is read by
                                # ``ToolInstance.dispatch`` / ``_run_persistent``).
                                _inv_token = _current_tool_invocation.set(
                                    {
                                        "key": key,
                                        "gpu_only": gpu_only_flag,
                                        "pin_visible_devices": pin_visible_devices_flag,
                                    }
                                )
                                try:
                                    result = func(inputs, config, instance)
                                finally:
                                    _current_tool_invocation.reset(_inv_token)

                                # Populate metadata fields
                                result.tool_id = key
                                result.execution_time = time.time() - start_time
                                if result.success is None:
                                    result.success = True

                                # Add captured warnings to the result (filtered)
                                if warning_list:
                                    # Filter out ignored warnings
                                    filtered_warnings = [
                                        w
                                        for w in warning_list
                                        if not any(ignored in str(w.message) for ignored in IGNORED_WARNING_SUBSTRINGS)
                                    ]

                                    if filtered_warnings:
                                        captured_warnings = [str(w.message) for w in filtered_warnings]
                                        result.warnings = result.warnings + captured_warnings
                                        for w in captured_warnings:
                                            logger.warning(f"Tool {key}: {w}")

                                        # Re-emit warnings so they still get logged/printed
                                        _re_emit_warnings(filtered_warnings)

                                # Post-dispatch: cache store, stitch, dedup expand
                                result = _post_dispatch_cache_and_expand(
                                    key,
                                    spec,
                                    runtime_cacheable,
                                    stochastic_tool,
                                    result,
                                    strip,
                                    deduped,
                                    original_items,
                                    whole_cache_key,
                                )

                                logger.debug(f"Tool {key}: completed in {result.execution_time:.2f}s")
                                return result

                        except (  # noqa: PERF203 -- retry loop
                            _RETRYABLE_EXCEPTIONS
                        ) as e:
                            last_exception = e
                            last_traceback = traceback.format_exc()
                            if attempt < MAX_RETRIES:
                                delay = RETRY_DELAY * (2**attempt)
                                logger.warning(
                                    f"Tool {key}: transient {type(e).__name__} on attempt "
                                    f"{attempt + 1}/{1 + MAX_RETRIES}, retrying in {delay:.1f}s: {e}"
                                )
                                time.sleep(delay)

                        except Exception as e:
                            if attempt < MAX_RETRIES and _is_transient_gpu_acquisition_error(e):
                                last_exception = e
                                last_traceback = traceback.format_exc()
                                delay = RETRY_DELAY * (2**attempt)
                                logger.warning(
                                    f"Tool {key}: transient GPU acquisition failure (Exclusive_Process) on "
                                    f"attempt {attempt + 1}/{1 + MAX_RETRIES}, retrying in {delay:.1f}s: {e}"
                                )
                                time.sleep(delay)
                                continue

                            # Non-retryable error; raise (or capture per policy)
                            filtered_warnings = [
                                w
                                for w in warning_list
                                if not any(ignored in str(w.message) for ignored in IGNORED_WARNING_SUBSTRINGS)
                            ]
                            captured_warnings = [str(w.message) for w in filtered_warnings]
                            _re_emit_warnings(filtered_warnings)

                            logger.error(f"Tool {key}: failed with {type(e).__name__}: {e}")
                            return _make_error_output_or_raise(
                                output_class,
                                key,
                                start_time,
                                e,
                                traceback.format_exc(),
                                captured_warnings,
                            )

                    # All retries exhausted for a retryable exception
                    assert last_exception is not None  # set on first retry iteration
                    logger.error(
                        f"Tool {key}: failed after {1 + MAX_RETRIES} attempts with "
                        f"{type(last_exception).__name__}: {last_exception}"
                    )
                    return _make_error_output_or_raise(
                        output_class,
                        key,
                        start_time,
                        last_exception,
                        last_traceback,
                    )

            # Register the tool spec with the captured source file
            cls._registry[key] = ToolSpec(
                key=key,
                label=label,
                category=category,
                description=description,
                uses_gpu=uses_gpu,
                gpu_only=gpu_only,
                pin_visible_devices=pin_visible_devices,
                device_count=device_count,
                input_model=input_class,
                config_model=config_class,
                output_model=output_class,
                metrics_model=metrics_class,
                function=wrapper,
                source_file=source_file,
                example_input=example_input,
                iterable_input_fields=iterable_input_fields,
                iterable_output_field=iterable_output_field,
                cacheable=cacheable,
                stochastic=stochastic,
                post_process_iterable=post_process_iterable,
            )
            return wrapper

        return decorator

    @classmethod
    def list_all(cls) -> list[ToolSpec]:
        """List all registered tools as Pydantic models.

        Returns all registered tools as ToolSpec models.

        Returns:
            list[ToolSpec]: List of ToolSpec Pydantic models
        """
        return list(cls._registry.values())

    @classmethod
    def get(cls, key: str) -> ToolSpec:
        """Get tool spec by key.

        Args:
            key (str): Tool identifier

        Returns:
            ToolSpec: Tool specification object

        Raises:
            ValueError: If key not found in registry
        """
        if key not in cls._registry:
            suggestions = difflib.get_close_matches(key, cls._registry.keys(), n=3, cutoff=0.6)
            hint = f"; did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise ValueError(f"Unknown tool {key!r} ({len(cls._registry)} registered){hint}")
        return cls._registry[key]

    @classmethod
    def get_input_schema(cls, key: str) -> dict[str, Any]:
        """Get JSON schema for tool inputs."""
        spec = cls.get(key)
        return spec.input_model.model_json_schema()

    @classmethod
    def get_config_schema(cls, key: str) -> dict[str, Any]:
        """Get JSON schema for tool configuration."""
        spec = cls.get(key)
        return spec.config_model.model_json_schema()

    @classmethod
    def get_output_schema(cls, key: str) -> dict[str, Any]:
        """Get JSON schema for tool output.

        Args:
            key (str): Tool identifier.

        """
        spec = cls.get(key)
        return spec.output_model.model_json_schema()

    @classmethod
    def get_schemas(cls, key: str) -> dict[str, dict[str, Any]]:
        """Get input, config, and output schemas."""
        return {
            "inputs": cls.get_input_schema(key),
            "config": cls.get_config_schema(key),
            "output": cls.get_output_schema(key),
        }

    @classmethod
    def get_readme(cls, key: str) -> str:
        """Return the tool's toolkit README as text.

        See ``proto_tools.utils.tool_docs.get_readme`` for full semantics.
        """
        from proto_tools.utils.tool_docs import get_readme

        return get_readme(key)

    @classmethod
    def get_readme_section(cls, key: str, heading: str, *, include_learning_resources: bool = False) -> str | None:
        """Return one named H2 section's body from the tool's toolkit README.

        Returns None when no H2 with that exact heading text exists. See
        ``proto_tools.utils.tool_docs.get_readme_section``.
        """
        from proto_tools.utils.tool_docs import get_readme_section

        return get_readme_section(key, heading, include_learning_resources=include_learning_resources)

    @classmethod
    def get_readme_sections(cls, key: str, *, include_learning_resources: bool = False) -> ReadmeSections:
        """Return the tool's toolkit README parsed into a typed structure.

        See ``proto_tools.utils.tool_docs.get_readme_sections``.
        """
        from proto_tools.utils.tool_docs import get_readme_sections

        return get_readme_sections(key, include_learning_resources=include_learning_resources)

    @classmethod
    def get_tool_docs(
        cls,
        key: str,
        *,
        include_toolkit_notes: bool = True,
        include_license: bool = True,
    ) -> ToolReadmeEntry | None:
        """Return the tool's H3 subsection from its toolkit README.

        When ``include_toolkit_notes`` is True (default), the returned entry's
        ``toolkit_notes`` field is also populated from the toolkit's
        ``## Toolkit Notes`` section, since those tips apply to every tool in
        the toolkit. When ``include_license`` is True (default), the entry's
        ``license`` field is populated from the toolkit's ``license.yaml`` so
        gating and usage terms come back in the same call. See
        ``proto_tools.utils.tool_docs.get_tool_docs``.
        """
        from proto_tools.utils.tool_docs import get_tool_docs

        return get_tool_docs(
            key,
            include_toolkit_notes=include_toolkit_notes,
            include_license=include_license,
        )

    @classmethod
    def get_example_notebook(cls, key: str) -> str | None:
        """Return the tool's ``examples/example.ipynb`` rendered as markdown + code.

        Returns None when the toolkit has no example notebook. See
        ``proto_tools.utils.tool_docs.get_example_notebook``.
        """
        from proto_tools.utils.tool_docs import get_example_notebook

        return get_example_notebook(key)

    @classmethod
    def get_input_doc(cls, tool: str) -> ModelDoc:
        """Return a ``ModelDoc`` view of the tool's input model.

        Accepts any identifier form (registry key, run-function name, docs
        path, single-tool toolkit name). See
        ``proto_tools.utils.tool_docs._normalize_tool_key`` for the full
        resolution rules.
        """
        from proto_tools.utils.tool_docs import _normalize_tool_key, get_model_doc

        return get_model_doc(cls.get(_normalize_tool_key(tool)).input_model)

    @classmethod
    def get_config_doc(cls, tool: str) -> ModelDoc:
        """Return a ``ModelDoc`` view of the tool's config model.

        Accepts any identifier form (registry key, run-function name, docs
        path, single-tool toolkit name). See
        ``proto_tools.utils.tool_docs._normalize_tool_key``.
        """
        from proto_tools.utils.tool_docs import _normalize_tool_key, get_model_doc

        return get_model_doc(cls.get(_normalize_tool_key(tool)).config_model)

    @classmethod
    def get_output_doc(cls, tool: str) -> ModelDoc:
        """Return a ``ModelDoc`` view of the tool's output model.

        Accepts any identifier form (registry key, run-function name, docs
        path, single-tool toolkit name). See
        ``proto_tools.utils.tool_docs._normalize_tool_key``.
        """
        from proto_tools.utils.tool_docs import _normalize_tool_key, get_model_doc

        spec = cls.get(_normalize_tool_key(tool))
        return get_model_doc(
            spec.output_model,
            metrics_class=spec.metrics_model,
            iterable_output_field=spec.iterable_output_field,
        )

    @classmethod
    def get_example_input(cls, key: str) -> BaseToolInput | None:
        """Get a minimal valid input instance for a tool, or None if not defined."""
        spec = cls.get(key)
        if spec.example_input is None:
            return None
        return spec.example_input()

    @classmethod
    def count(cls) -> int:
        """Get count of registered tools."""
        return len(cls._registry)

    @classmethod
    def list_gpu_tools(cls) -> list[ToolSpec]:
        """List all registered tools that require a GPU."""
        return [spec for spec in cls._registry.values() if spec.uses_gpu]

    @classmethod
    def list_cpu_tools(cls) -> list[ToolSpec]:
        """List all registered tools that do not require a GPU."""
        return [spec for spec in cls._registry.values() if not spec.uses_gpu]

    @classmethod
    def list_local_cpu_tools(cls) -> list[ToolSpec]:
        """List all registered tools that run trivially in-process on CPU (no GPU, no standalone env)."""
        return [spec for spec in cls._registry.values() if spec.local_cpu]

    @classmethod
    def list_categories(cls) -> list[str]:
        """Return the sorted list of categories any registered tool belongs to.

        Returns:
            list[str]: Sorted unique category names (e.g.,
                ``["binder_design", "causal_models", "database_retrieval", ...]``).
        """
        return sorted({spec.category for spec in cls._registry.values()})

    @classmethod
    def list_by_category(cls, category: str) -> list[ToolSpec]:
        """Return every registered tool in a category, sorted by key.

        Args:
            category (str): Category name (e.g., ``"masked_models"``,
                ``"structure_prediction"``). See ``list_categories`` for the
                full set.

        Returns:
            list[ToolSpec]: Tools whose ``category`` matches, sorted by
                registry key. Empty if the category is unknown.
        """
        return sorted(
            (spec for spec in cls._registry.values() if spec.category == category),
            key=lambda s: s.key,
        )

    @classmethod
    def catalog(cls) -> dict[str, list[ToolSpec]]:
        """Return every registered tool grouped by category.

        Returns:
            dict[str, list[ToolSpec]]: Mapping from category name to a
                key-sorted list of ``ToolSpec`` in that category. Categories
                themselves are sorted alphabetically.
        """
        return {category: cls.list_by_category(category) for category in cls.list_categories()}

    @classmethod
    def get_citation(cls, key: str) -> str | None:
        """Get BibTeX citation for a tool by key.

        Args:
            key (str): Tool identifier (e.g., 'evo2-sample', 'blast-search')

        Returns:
            str | None: BibTeX citation string, or None if no citation file exists

        Raises:
            ValueError: If tool key is not found in registry
        """
        # Validate tool exists
        cls.get(key)

        # Find and read citation file
        cite_path = _find_citation_file(key)
        if cite_path is None:
            return None

        return cite_path.read_text().strip()

    @classmethod
    def list_citations(cls) -> dict[str, str]:
        """Get all available citations as {tool_key: bibtex_string}.

        Returns:
            dict[str, str]: Dictionary mapping tool keys to their BibTeX citations.
                Only includes tools that have cite.bib files.
        """
        citations = {}
        for key in cls._registry:
            citation = cls.get_citation(key)
            if citation is not None:
                citations[key] = citation
        return citations

    @classmethod
    def get_doi(cls, key: str) -> str | None:
        """Extract DOI from a tool's cite.bib, if available.

        Args:
            key (str): Tool identifier (e.g., 'evo2-sample', 'blast-search')

        Returns:
            str | None: DOI string (e.g., '10.1038/s41586-024-07487-w'), or None
                if no citation exists or the citation has no DOI field

        Raises:
            ValueError: If tool key is not found in registry
        """
        citation = cls.get_citation(key)
        if citation is None:
            return None
        match = re.search(r'doi\s*=\s*[{"]([^}"]+)[}"]', citation)
        return match.group(1) if match else None

    @classmethod
    def get_links(cls, key: str) -> dict[str, str] | None:
        """Get links.yaml metadata for a tool.

        Returns parsed contents of the tool's links.yaml file, which may contain
        github, image, organizations, and preprint fields. The documentation URL
        is computed separately by ``get_docs_url`` (derived from the tool's
        directory path, not stored in YAML).

        Args:
            key (str): Tool identifier (e.g., 'evo2-sample', 'blast-search')

        Returns:
            dict[str, str] | None: Parsed YAML dict, or None if no links.yaml exists

        Raises:
            ValueError: If tool key is not found in registry
        """
        cls.get(key)
        links_file = _find_links_file(key)
        if links_file is None:
            return None
        with open(links_file) as f:
            data = yaml.safe_load(f)
            if not isinstance(data, dict):
                return None
            return data

    @classmethod
    def get_license(cls, key: str) -> dict[str, Any] | None:
        """Get license.yaml metadata for a tool.

        Returns parsed contents of the tool's license.yaml file.

        Args:
            key (str): Tool identifier (e.g., 'evo2-sample', 'blast-search')

        Returns:
            dict[str, Any] | None: Parsed YAML dict, or None if no license.yaml
                exists for this toolkit.

        Raises:
            ValueError: If tool key is not found in registry
        """
        cls.get(key)
        license_file = _find_license_file(key)
        if license_file is None:
            return None
        with open(license_file) as f:
            data = yaml.safe_load(f)
            if not isinstance(data, dict):
                return None
            return data

    @classmethod
    def list_licenses(cls) -> dict[str, dict[str, Any]]:
        """Get all available licenses as {tool_key: license_dict}.

        Returns:
            dict[str, dict[str, Any]]: Dictionary mapping tool keys to their
                parsed license.yaml contents. Only includes tools that have
                license.yaml files.
        """
        licenses = {}
        for key in cls._registry:
            license_data = cls.get_license(key)
            if license_data is not None:
                licenses[key] = license_data
        return licenses

    @classmethod
    def get_weights_access(cls, key: str) -> str:
        """Return how a tool's model weights are obtained.

        Normalizes the nested ``license.yaml`` ``weights.access`` field into a
        single value to check before calling, without navigating the nested
        mapping or knowing that an absent field means open.

        Args:
            key (str): Tool identifier (e.g., 'esm3-embedding').

        Returns:
            str: ``"open"`` (no extra step), ``"hf-gated"`` (accept the
                provider's terms and set ``HF_TOKEN``), or ``"request"``
                (weights obtained from the provider out of band).

        Raises:
            ValueError: If tool key is not found in registry.
        """
        license_data = cls.get_license(key)
        weights = (license_data or {}).get("weights")
        access = weights.get("access") if isinstance(weights, dict) else None
        return access if isinstance(access, str) else "open"

    @classmethod
    def get_docs_url(cls, key: str) -> str | None:
        """Get the documentation URL for a tool.

        Computes the URL from the tool's directory location: every tool under
        ``proto_tools/tools/{category}/{toolkit}/`` maps deterministically
        to ``https://bio-pro.mintlify.app/tools/{category-kebab}/{tool-kebab}``,
        because proto-docs auto-generates one page per tool dir on each sync.

        Returns ``None`` only when the tool's directory can't be located on
        disk (e.g. a registry entry with no corresponding source tree — not a
        case that should occur for shipped tools).

        Args:
            key (str): Tool identifier (e.g., 'evo2-sample', 'blast-search').

        Returns:
            str | None: Documentation URL, or None if the tool directory
                cannot be resolved.

        Raises:
            ValueError: If tool key is not found in registry.
        """
        spec = cls.get(key)
        normalized = key.replace("-", "_")
        category_dir = TOOLS_DIR / spec.category
        if not category_dir.is_dir():
            return None
        for tool_dir in sorted(category_dir.iterdir()):
            if not tool_dir.is_dir() or tool_dir.name.startswith("_"):
                continue
            if normalized == tool_dir.name or normalized.startswith(tool_dir.name + "_"):
                category_kebab = spec.category.replace("_", "-")
                tool_kebab = tool_dir.name.replace("_", "-")
                return f"https://bio-pro.mintlify.app/tools/{category_kebab}/{tool_kebab}"
        return None

    @classmethod
    def get_example_notebook_path(cls, key: str) -> Path | None:
        """Get the path to a tool's example notebook, if present.

        Returns the filesystem path to ``examples/example.ipynb`` inside the
        tool's directory, or ``None`` if the tool has no example notebook.
        Downstream consumers decide how to render (e.g., convert to a
        GitHub viewer URL).

        Args:
            key (str): Tool identifier (e.g., 'evo2-sample', 'blast-search').

        Returns:
            Path | None: Path to ``example.ipynb``, or None if not present.

        Raises:
            ValueError: If tool key is not found in registry.
        """
        cls.get(key)
        return _find_tool_metadata_file(key, "examples/example.ipynb")

    @classmethod
    def _check_duplicate(cls, key: str, attempted_name: str | None = None) -> None:
        """Check for duplicate registration.

        Args:
            key (str): Tool registry key to check.
            attempted_name (str | None): Name of the function attempting registration.

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


def _coerce_model(instance: BaseModel, expected_class: type[BaseModel], tool_key: str, role: str) -> BaseModel:
    """Coerce a Pydantic model to the expected class if a parent class was passed.

    Only explicitly-set fields are forwarded so child class defaults take
    precedence over parent defaults.

    Args:
        instance (BaseModel): The model instance to coerce.
        expected_class (type[BaseModel]): The expected Pydantic model class.
        tool_key (str): Tool registry key for log messages.
        role (str): "input" or "config" for log messages.

    Returns:
        BaseModel: The coerced model instance, or the original if already correct.

    Raises:
        TypeError: If the instance is not the expected class or a parent of it.
    """
    if isinstance(instance, expected_class):
        return instance
    actual_type = type(instance)
    if not issubclass(expected_class, actual_type):
        raise TypeError(
            f"Tool {tool_key}: {role} must be {expected_class.__name__} (or a parent class), got {actual_type.__name__}"
        )
    coerced = expected_class.model_validate(instance.model_dump(include=instance.model_fields_set))
    logger.warning(
        f"Tool {tool_key}: coerced {role} {actual_type.__name__} to {expected_class.__name__}. "
        f"Use {expected_class.__name__} directly to silence this warning."
    )
    return coerced


def _make_error_output(
    output_class: type[BaseToolOutput],
    key: str,
    start_time: float,
    exception: Exception,
    traceback_str: str,
    warning_strings: list[str] | None = None,
) -> BaseToolOutput:
    """Construct a structured error output for a failed tool execution."""
    return output_class.model_construct(
        tool_id=key,
        execution_time=time.time() - start_time,
        success=False,
        timestamp=datetime.now(),
        warnings=warning_strings or [],
        errors=[f"{type(exception).__name__}: {exception}", traceback_str],
    )


def _make_error_output_or_raise(
    output_class: type[BaseToolOutput],
    key: str,
    start_time: float,
    exception: Exception,
    traceback_str: str,
    warning_strings: list[str] | None = None,
) -> BaseToolOutput:
    """Capture or re-raise a tool exception based on the active capture policy.

    ``MissingAssetError`` always re-raises (the pytest skip hook depends on it).
    Otherwise the env var decides: capture into a ``success=False`` output, or re-raise.

    Args:
        output_class (type[BaseToolOutput]): Output model class for the failed tool.
        key (str): Tool registry key.
        start_time (float): Execution start timestamp from ``time.time()``.
        exception (Exception): The exception raised by the tool.
        traceback_str (str): Formatted traceback string.
        warning_strings (list[str] | None): Warnings captured during execution.

    Returns:
        BaseToolOutput: Structured error output, when capture is enabled.

    Raises:
        Exception: The original exception, when capture is disabled.
    """
    if isinstance(exception, MissingAssetError):
        raise exception
    if _should_capture_errors():
        return _make_error_output(output_class, key, start_time, exception, traceback_str, warning_strings)
    raise exception


def _post_dispatch_cache_and_expand(
    key: str,
    spec: ToolSpec | None,
    is_cacheable: bool,
    stochastic_tool: bool,
    result: BaseToolOutput,
    strip: CacheStripResult | None,
    deduped: Any | None,
    original_items: list[Any] | None,
    whole_cache_key: str | None,
) -> BaseToolOutput:
    """Store results in cache and expand deduped items back to original positions.

    Called after both pool and local dispatch paths to avoid duplicating
    cache-store / stitch / dedup-expand logic.

    Args:
        key (str): Tool registry key.
        spec (ToolSpec | None): Tool specification from the registry.
        is_cacheable (bool): Whether the tool supports caching.
        stochastic_tool (bool): Whether the tool is registered ``stochastic=True``.
            Stochastic iterables skip per-item cache / dedup expansion and use
            the whole-call cache instead.
        result (BaseToolOutput): Tool output to post-process.
        strip (CacheStripResult | None): Cache strip result with cached/uncached item info.
        deduped (Any | None): Deduplication result mapping original to unique indices.
        original_items (list[Any] | None): Original input items before dedup/strip.
        whole_cache_key (str | None): Cache key for whole-output caching path.
    """
    if is_cacheable and spec and spec.iterable_input_fields and not stochastic_tool:
        assert spec.iterable_output_field is not None  # validated at registration
        output_field = spec.iterable_output_field
        # Per-item cache store + stitch (deterministic iterable only)
        computed_items = getattr(result, output_field, [])
        if strip is not None and strip.cached_results:
            cache_store_items(key, strip.cache_keys, computed_items)
            total = len(strip.cached_results) + len(strip.uncached_items)
            stitched = cache_stitch_items(strip, computed_items, total)
            setattr(result, output_field, stitched)
        elif strip is not None:
            # No cached items existed, but we still store the new ones
            cache_store_items(key, strip.cache_keys, computed_items)

        # Expand deduped results back to original positions
        if deduped and original_items and len(deduped.unique_items) < len(original_items):
            unique_results = getattr(result, output_field)
            expanded = [unique_results[uid] for _, uid in deduped.index_map]
            setattr(result, output_field, expanded)

    elif is_cacheable and whole_cache_key is not None:
        # Whole-output cache store (non-iterable OR stochastic iterable)
        cache = _program_tool_cache.get()
        if cache is not None:
            cache.set(key, whole_cache_key, result)

    # Run post_process_iterable for any cacheable iterable, regardless of which
    # cache path was taken. Deterministic iterables see the full stitched
    # batch after dedup expand; stochastic iterables see the natural batch
    # (no dedup happened).
    if is_cacheable and spec is not None and spec.iterable_input_fields and spec.post_process_iterable is not None:
        assert spec.iterable_output_field is not None
        spec.post_process_iterable(getattr(result, spec.iterable_output_field))

    paths = _find_non_finite_paths(result.model_dump())
    if paths:
        logger.warning("tool %s: non-finite floats in output at paths=%s", key, paths)
    return result


def _find_non_finite_paths(obj: Any, _path: str = "", _paths: list[str] | None = None) -> list[str]:
    """Return JSONPath strings for each non-finite float in *obj*; does not mutate.

    Args:
        obj (Any): Value to walk.
        _path (str): Internal — JSONPath prefix.
        _paths (list[str] | None): Internal — accumulator.

    Returns:
        list[str]: JSONPath of each non-finite location.
    """
    if _paths is None:
        _paths = []
    if isinstance(obj, float):
        if not math.isfinite(obj):
            _paths.append(_path or "<root>")
        return _paths
    if isinstance(obj, MutableMapping):
        for k, v in obj.items():
            sub_path = f"{_path}.{k}" if _path else str(k)
            _find_non_finite_paths(v, sub_path, _paths)
        return _paths
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            sub_path = f"{_path}[{i}]"
            _find_non_finite_paths(v, sub_path, _paths)
        return _paths
    return _paths


def _re_emit_warnings(warning_list: list[warnings.WarningMessage]) -> None:
    """Re-emit warnings so they still get logged/printed."""
    for w in warning_list:
        warnings.warn_explicit(
            w.message,  # type: ignore[arg-type]
            w.category,
            w.filename,
            w.lineno,
            w.file,  # type: ignore[arg-type]
            w.line,  # type: ignore[arg-type]
        )


def _find_tool_metadata_file(tool_key: str, filename: str) -> Path | None:
    """Find a metadata file (cite.bib, links.yaml, etc.) in tool directories.

    Maps tool key (e.g., 'evo2-sample') to tool directory (e.g., evo2/)
    and checks for the given filename.

    Args:
        tool_key (str): Tool registry key (e.g., 'evo2-sample', 'blast-search')
        filename (str): Name of file to find (e.g., 'cite.bib', 'links.yaml')

    Returns:
        Path | None: Path to the file, or None if not found
    """
    # Tool keys are kebab-case but directories are snake_case
    normalized_key = tool_key.replace("-", "_")

    for category_dir in TOOLS_DIR.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith("_"):
            continue

        for tool_dir in category_dir.iterdir():
            if not tool_dir.is_dir() or tool_dir.name.startswith("_"):
                continue

            # e.g., 'evo2_sample' matches 'evo2', 'blast_search' matches 'blast'
            if normalized_key == tool_dir.name or normalized_key.startswith(tool_dir.name + "_"):
                file_path = tool_dir / filename
                if file_path.exists():
                    return file_path

    return None


def _find_citation_file(tool_key: str) -> Path | None:
    """Find cite.bib for a tool by searching tool directories."""
    return _find_tool_metadata_file(tool_key, "cite.bib")


def _find_links_file(tool_key: str) -> Path | None:
    """Find links.yaml for a tool by searching tool directories."""
    return _find_tool_metadata_file(tool_key, "links.yaml")


def _find_license_file(tool_key: str) -> Path | None:
    """Find license.yaml for a tool by searching tool directories."""
    return _find_tool_metadata_file(tool_key, "license.yaml")


# Alias for simpler decorator syntax: @tool(...) instead of @ToolRegistry.register(...)
tool = ToolRegistry.register
