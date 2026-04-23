"""proto_tools/tools/tool_registry.py.

Tool registry for managing tool discovery and schema generation.

Provides a decorator-based API for registering tools with metadata and
automatic schema generation for API/client integration.
"""

import contextlib
import inspect
import logging
import re
import time
import traceback
import warnings
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, Field, field_serializer

logger = logging.getLogger(__name__)

# Path to tools directory for citation file discovery
TOOLS_DIR = Path(__file__).parent

# List of warning message substrings to ignore (noisy warnings from dependencies)
IGNORED_WARNING_SUBSTRINGS = [
    "get_autocast_gpu_dtype",
    "get_autocast_dtype",
]

# Retry configuration for transient failures (network drops, subprocess crashes)
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # Base delay in seconds (exponential backoff: 2s, 4s, 8s)

# TimeoutError excluded: retrying with the same limit would just time out again
_RETRYABLE_EXCEPTIONS = (ConnectionError,)

from proto_tools.utils import BaseConfig
from proto_tools.utils.device import (
    parse_device_string,
    validate_device_allocation,
)
from proto_tools.utils.progress import reset_current_tool_function, set_current_tool_function
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
from proto_tools.utils.tool_io import BaseToolInput, BaseToolOutput


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
        function (Callable[..., Any]): The wrapped tool function.
        source_file (Path): Path to the source file where the tool function is defined.
        example_input (Callable[[], BaseToolInput] | None): Factory returning a minimal
            valid input for testing.
        iterable_input_field (str | None): Input field name containing the iterable list
            of items (for ToolPool fan-out).
        iterable_output_field (str | None): Output field name containing the iterable list
            of results (for ToolPool fan-out).
        cacheable (bool): Whether this tool's results should be cached in the
            program-scoped cache.
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
    device_count: str = Field(
        default="1", description="Expected device count requirement (e.g., '1', '1-2', '>=1', '<=2')"
    )

    # Configuration model - serialized as JSON Schema
    config_model: type[BaseModel] = Field(
        description="Pydantic model for configuration validation and schema generation"
    )

    # Private fields - excluded from serialization
    input_model: type[BaseToolInput] = Field(exclude=True)
    output_model: type[BaseToolOutput] = Field(exclude=True)
    function: Callable[..., Any] = Field(exclude=True)
    source_file: Path = Field(exclude=True, description="Path to the source file where the tool function is defined")
    example_input: Callable[[], BaseToolInput] | None = Field(
        default=None,
        exclude=True,
        description="Factory returning a minimal valid input for testing",
    )
    iterable_input_field: str | None = Field(
        default=None,
        exclude=True,
        description="Input field name containing the iterable list of items (for ToolPool fan-out)",
    )
    iterable_output_field: str | None = Field(
        default=None,
        exclude=True,
        description="Output field name containing the iterable list of results (for ToolPool fan-out)",
    )
    cacheable: bool = Field(
        default=False,
        exclude=True,
        description="Whether this tool's results should be cached in the program-scoped cache",
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
        >>> from proto_tools.tools.gene_annotation import run_blast, BLASTConfig
        >>> config = BLASTConfig(query_sequences=["MVLSP"], database="/data/nr")
        >>> result = run_blast(config)
    """

    _registry: ClassVar[dict[str, ToolSpec]] = {}

    @classmethod
    def _try_dispatch(
        cls,
        key: str,  # noqa: ARG003 — required by descriptor protocol
        inputs: BaseToolInput,  # noqa: ARG003 — required by descriptor protocol
        config: BaseConfig | None,  # noqa: ARG003 — required by descriptor protocol
    ) -> BaseToolOutput | None:
        """Extension point for external tool dispatch.

        Monkeypatch this classmethod to route tool calls to external
        services (e.g. HTTP APIs).  Return a ``BaseToolOutput`` to handle
        the call, or ``None`` to fall through to local execution.

        Args:
            key (str): Tool registry key (e.g. ``"esm2"``).
            inputs (BaseToolInput): Tool input payload.
            config (BaseConfig | None): Tool configuration, if any.

        Returns:
            BaseToolOutput | None: Tool output if handled, or ``None``
                to fall through to local execution.
        """
        result: BaseToolOutput | None = None
        return result

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
        uses_gpu: bool = False,
        gpu_only: bool = False,
        device_count: str = "1",
        example_input: Callable[[], BaseToolInput] | None = None,
        iterable_input_field: str | None = None,
        iterable_output_field: str | None = None,
        cacheable: bool = False,
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
            uses_gpu (bool): Whether this tool requires a GPU for execution
            gpu_only (bool): Tool cannot run on CPU. Direct ``device="cpu"``
                dispatch raises ``ValueError``, and LRU eviction restarts the
                worker instead of offloading. Implies ``uses_gpu=True``.
            device_count (str): Expected device count (e.g., "1", "1-2", ">=1", "<=2").
                Validates allocation: errors on under-allocation, warns on over-allocation.
            example_input (Callable[[], BaseToolInput] | None): Factory returning a minimal valid
                input for testing and examples.
            iterable_input_field (str | None): Input field name containing the iterable list of
                items for ToolPool fan-out and per-item caching.
            iterable_output_field (str | None): Output field name containing the iterable list of
                results for ToolPool fan-out and per-item caching.
            cacheable (bool): Whether this tool's results should be cached in the
                program-scoped cache.

        Returns:
            Callable[[Callable[..., BaseToolOutput]], Callable[..., BaseToolOutput]]: Decorator
                that wraps the function with metadata tracking.
        """

        def decorator(func: Callable[..., BaseToolOutput]) -> Callable[..., BaseToolOutput]:
            cls._check_duplicate(key, func.__name__)

            if (iterable_input_field is None) != (iterable_output_field is None):
                raise ValueError(
                    f"Tool '{key}': iterable_input_field and iterable_output_field must both be set or both be None"
                )

            if gpu_only and not uses_gpu:
                raise ValueError(f"Tool '{key}': gpu_only=True requires uses_gpu=True")

            # Capture source file from call stack (find first frame in tools directory)
            stack = inspect.stack()
            source_file = None
            for frame_info in stack:
                filename = frame_info.filename
                # Look for the first frame in the tools directory (skip this registry file)
                if "/tools/" in filename and "tool_registry.py" not in filename:
                    source_file = Path(filename)
                    break

            # Fallback to func's code if we couldn't find it in the stack
            if source_file is None:
                source_file = Path(func.__code__.co_filename)

            @wraps(func)
            def wrapper(
                inputs: BaseToolInput,
                config: BaseConfig | None = None,
                instance: "str | ToolInstance | None" = None,
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
                inputs: BaseToolInput, config: BaseConfig | None, instance: "str | ToolInstance | None"
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
                last_exception = None
                last_traceback = ""
                warning_list = []

                # Resolve the spec once so runtime-mutable flags (``gpu_only``)
                # can be monkeypatched by tests. Fall back to the decorator's
                # captured value if the spec is missing (defensive).
                spec = cls._registry.get(key)
                effective_gpu_only = spec.gpu_only if spec is not None else gpu_only

                # Validate device allocation against tool requirements
                if hasattr(config, "device"):
                    device_str = str(config.device)
                    if effective_gpu_only and device_str == "cpu":
                        raise ValueError(
                            f"Tool {key!r} is gpu_only; cannot run with device='cpu'. "
                            f"Use a CUDA device (e.g. 'cuda' or 'cuda:0')."
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
                deduped = None
                original_items = None
                strip = None
                whole_cache_key = None

                if cacheable and spec and spec.iterable_input_field:
                    assert spec.iterable_output_field is not None  # validated at registration
                    # Iterable path: dedup then strip cached items
                    original_items = list(getattr(inputs, spec.iterable_input_field))
                    if len(original_items) > 1:
                        deduped = deduplicate_items(original_items, key_fn=_serialize_for_cache_key)
                        if len(deduped.unique_items) < len(original_items):
                            inputs = inputs.model_copy(update={spec.iterable_input_field: deduped.unique_items})
                            logger.debug(
                                "Tool %s: dedup %d → %d unique items",
                                key,
                                len(original_items),
                                len(deduped.unique_items),
                            )

                    # Strip cached items from the (possibly deduped) input
                    current_items = list(getattr(inputs, spec.iterable_input_field))
                    strip = cache_strip_items(key, current_items, config)
                    if strip is not None and strip.all_cached:
                        logger.debug("[Iterable Cache] %s: full cache hit, skipping dispatch", key)
                        cached_list = [strip.cached_results[i] for i in range(len(current_items))]
                        # Expand dedup if needed
                        if deduped and len(deduped.unique_items) < len(original_items):
                            cached_list = [cached_list[uid] for _, uid in deduped.index_map]
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

                    # Narrow inputs to uncached items only
                    if strip is not None and strip.uncached_items:
                        inputs = inputs.model_copy(update={spec.iterable_input_field: strip.uncached_items})

                elif cacheable:
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
                    if pool is not None and not is_pool_executing() and spec and spec.iterable_input_field is not None:
                        result = pool._parallel_dispatch(key, func, inputs, config)
                        result.tool_id = key
                        result.success = True
                        result.execution_time = time.time() - start_time
                        return _post_dispatch_cache_and_expand(
                            key,
                            spec,
                            cacheable,
                            result,
                            strip,
                            deduped,
                            original_items,
                            whole_cache_key,
                        )

                    # Extension point: try external dispatch before local execution
                    try:
                        dispatched = cls._try_dispatch(key, inputs, config)
                    except Exception as e:
                        logger.error("Tool %s: _try_dispatch raised %s: %s", key, type(e).__name__, e)
                        return _make_error_output(
                            output_class,
                            key,
                            start_time,
                            e,
                            traceback.format_exc(),
                        )
                    if dispatched is not None:
                        dispatched.tool_id = key
                        dispatched.success = True
                        dispatched.execution_time = time.time() - start_time
                        return _post_dispatch_cache_and_expand(
                            key,
                            spec,
                            cacheable,
                            dispatched,
                            strip,
                            deduped,
                            original_items,
                            whole_cache_key,
                        )

                    # Carry per-invocation flags (key, gpu_only) to ToolInstance so
                    # the eviction callback can restart the worker instead of
                    # attempting an in-process CPU offload when the tool is gpu_only.
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
                                _inv_token = _current_tool_invocation.set({"key": key, "gpu_only": effective_gpu_only})
                                try:
                                    result = func(inputs, config, instance)
                                finally:
                                    _current_tool_invocation.reset(_inv_token)

                                # Populate metadata fields
                                result.tool_id = key
                                result.execution_time = time.time() - start_time
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
                                    cacheable,
                                    result,
                                    strip,
                                    deduped,
                                    original_items,
                                    whole_cache_key,
                                )

                                logger.debug(f"Tool {key}: completed in {result.execution_time:.2f}s")
                                return result

                        except _RETRYABLE_EXCEPTIONS as e:  # noqa: PERF203 -- retry loop
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
                            # Non-retryable error; return immediately
                            filtered_warnings = [
                                w
                                for w in warning_list
                                if not any(ignored in str(w.message) for ignored in IGNORED_WARNING_SUBSTRINGS)
                            ]
                            captured_warnings = [str(w.message) for w in filtered_warnings]
                            _re_emit_warnings(filtered_warnings)

                            logger.error(f"Tool {key}: failed with {type(e).__name__}: {e}")
                            return _make_error_output(
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
                    return _make_error_output(
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
                device_count=device_count,
                input_model=input_class,
                config_model=config_class,
                output_model=output_class,
                function=wrapper,
                source_file=source_file,
                example_input=example_input,
                iterable_input_field=iterable_input_field,
                iterable_output_field=iterable_output_field,
                cacheable=cacheable,
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
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown tool: '{key}'. Available tools: {available}")
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
    def get_tool_categories(cls) -> dict[str, str]:
        """Get mapping of tool names to their categories.

        Extracts tool name from registry key (e.g., 'blast-search' -> 'blast')
        and maps to category. Handles multi-word tool names like 'colabfold_search'.

        Returns:
            dict[str, str]: Dict mapping tool name to category (e.g., {'blast': 'gene_annotation'})
        """
        tool_categories: dict[str, str] = {}
        for spec in cls._registry.values():
            # Extract toolkit from key: 'blast-search' -> 'blast'
            # Handle multi-part names: 'colabfold-search' -> 'colabfold_search'
            key_parts = spec.key.split("-")
            toolkit = "_".join(key_parts[:-1]) if len(key_parts) >= 2 else spec.key
            tool_categories[toolkit] = spec.category
        return tool_categories

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
        errors=[str(exception), traceback_str],
    )


def _post_dispatch_cache_and_expand(
    key: str,
    spec: ToolSpec | None,
    is_cacheable: bool,
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
        result (BaseToolOutput): Tool output to post-process.
        strip (CacheStripResult | None): Cache strip result with cached/uncached item info.
        deduped (Any | None): Deduplication result mapping original to unique indices.
        original_items (list[Any] | None): Original input items before dedup/strip.
        whole_cache_key (str | None): Cache key for whole-output caching path.
    """
    if is_cacheable and spec and spec.iterable_input_field:
        assert spec.iterable_output_field is not None  # validated at registration
        output_field = spec.iterable_output_field
        # Iterable cache store + stitch
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
        # Whole-output cache store
        cache = _program_tool_cache.get()
        if cache is not None:
            cache.set(key, whole_cache_key, result)

    return result


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


# Alias for simpler decorator syntax: @tool(...) instead of @ToolRegistry.register(...)
tool = ToolRegistry.register
