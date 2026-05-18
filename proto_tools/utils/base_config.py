"""proto_tools/utils/base_config.py.

Base configuration class for all pydantic configs.
"""

import json
import random
from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField

from proto_tools.utils.tool_io import BaseToolInput, _extra_dict, _reject_removed_ui_kwargs

DEFAULT_TIMEOUT = 600  # seconds
RANDOM_SEED_UPPER_BOUND = 2**31


def ConfigField(
    default: Any = ...,
    *,
    title: str | None = None,
    description: str | None = None,
    reload_on_change: bool = False,
    include_in_key: bool = True,
    xor_group: str | None = None,  # noqa: ARG001 — marker for sibling-field XOR groups; enforced by @model_validator per tool.
    **kwargs: Any,
) -> Any:
    """Custom Field wrapper that automatically adds metadata flags to json_schema_extra.

    Args:
        default (Any): Default value for the field. Use ``...`` for required fields.
        title (str | None): Human-readable title for UI display.
        description (str | None): Description of the field for documentation and UI tooltips.
        reload_on_change (bool): If True, changing this field between persistent
            worker calls triggers a subprocess restart.
        include_in_key (bool): If False, field is excluded from tool cache key
            generation. Fields that don't affect computation results (device,
            verbose, timeout) should set this to False.
        xor_group (str | None): Mutual-exclusion group name. Enforce at runtime
            with a ``@model_validator`` on the Config class.
        kwargs: All other standard Pydantic Field arguments (via ``**kwargs``).

    Usage:
        param: int = ConfigField(default=42, title="Param", description="...")
    """
    _reject_removed_ui_kwargs("ConfigField", kwargs)
    json_schema_extra = kwargs.get("json_schema_extra", {})

    json_schema_extra["reload_on_change"] = reload_on_change
    json_schema_extra["include_in_key"] = include_in_key
    json_schema_extra["_field_type"] = "ConfigField"

    kwargs["json_schema_extra"] = json_schema_extra

    return PydanticField(default, title=title, description=description, **kwargs)


class BaseConfig(BaseModel):
    """Base configuration class for consistent behavior across all configs (tools, constraints, and generators).

    Attributes:
        verbose (int): Verbosity level (0=quiet, 1=info, 2=debug, 3=raw subprocess stderr).
            ``True`` is coerced to ``1`` and ``False`` to ``0``.
        device (str): Device to run the tool on.
        timeout (int | None): Maximum execution time in seconds. ``None`` waits indefinitely.
        seed (int | None): Random seed. When set, tools run reproducibly up to small
            GPU float noise (see ``BaseToolOutput.approx_equal``), and the seed
            participates in cache keys. When None, cacheable seed-sensitive tools
            skip cache until seeded.

    Properties:
        gpus_per_instance: Number of GPUs each worker needs. Default is
            derived from the ``device`` field via :func:`parse_device_string`
            (``cpu`` → 0, ``cuda`` / ``cuda:N`` → 1, ``cudaxN`` / multi → N,
            ``cloud`` → 1). Override in tool configs where GPU need is
            decoupled from the device string — e.g. a large checkpoint that
            needs 2 GPUs regardless of input device, or a tool with a
            separate ``use_gpu`` flag toggling real GPU work. ``ToolPool``
            reads this at dispatch time to group devices into worker slots.
        cpus_per_instance: Per-instance CPU consumption — drives ToolPool's
            CPU fan-out. See the property's own docstring below for full
            semantics.

    Methods:
        effective_timeout: Timeout the framework enforces. Override when the cap depends on other fields.

    Example:
        >>> class MyToolConfig(BaseConfig):
        ...     param1: int
        ...     param2: str

        Multi-GPU override::

            class Evo2Config(BaseConfig):
                checkpoint: str = ConfigField(default="7b")

                @property
                def gpus_per_instance(self) -> int:
                    return 4 if self.checkpoint == "40b" else 1
    """

    model_config = ConfigDict(
        extra="forbid",  # Reject unknown fields
        validate_assignment=True,  # Validate on field updates
        use_enum_values=True,  # Serialize enums as values
        validate_default=True,  # Validate default values
    )

    verbose: int = ConfigField(
        title="Verbose",
        default=0,
        ge=0,
        le=3,
        description="Verbosity level (0=quiet, 1=info, 2=debug, 3=raw subprocess stderr). True→1, False→0.",
        include_in_key=False,
    )

    @classmethod
    def reload_fields(cls) -> set[str]:
        """Return field names marked with ``reload_on_change=True``."""
        return {name for name, info in cls.model_fields.items() if _extra_dict(info).get("reload_on_change", False)}

    @classmethod
    def cache_exclude_fields(cls) -> set[str]:
        """Return field names marked with ``include_in_key=False``."""
        return {name for name, info in cls.model_fields.items() if not _extra_dict(info).get("include_in_key", True)}

    def cache_key(self) -> str:
        """Deterministic string for cache key generation, excluding non-key fields."""
        model_dict = self.model_dump(exclude_none=True, exclude=self.cache_exclude_fields())
        return json.dumps(model_dict, sort_keys=True, default=str)

    device: str = ConfigField(
        title="Device",
        default="cpu",
        description="Device to run the tool on (e.g., 'cpu', 'cuda', 'cuda:0', 'cloud')",
        include_in_key=False,
    )

    timeout: int | None = ConfigField(
        title="Timeout",
        default=DEFAULT_TIMEOUT,
        ge=1,
        description="Maximum execution time in seconds. None waits indefinitely.",
        include_in_key=False,
    )

    seed: int | None = ConfigField(
        title="Random Seed",
        default=None,
        ge=0,
        lt=2**32,
        description="Random seed for reproducible results. Some cacheable tools gate cache on this field.",
        include_in_key=True,
    )

    @staticmethod
    def get_random_int() -> int:
        """Return a fresh random int in ``[0, 2**31)`` for seeding RNGs.

        Use as a fallback when downstream code requires a concrete int seed:
        ``config.seed if config.seed is not None else config.get_random_int()``.
        """
        return random.randint(0, RANDOM_SEED_UPPER_BOUND - 1)  # noqa: S311 -- not for cryptographic use

    def derive_per_item_seeds(self, n_items: int) -> list[int]:
        """Return ``n_items`` distinct seeds derived from ``self.seed`` (or a fresh random base when unseeded)."""
        base = self.seed if self.seed is not None else self.get_random_int()
        rng = random.Random(base)  # noqa: S311 -- non-cryptographic
        return [rng.randint(0, RANDOM_SEED_UPPER_BOUND - 1) for _ in range(n_items)]

    @property
    def gpus_per_instance(self) -> int:
        """Number of GPUs each ToolPool worker needs for this configuration.

        ToolPool reads this at dispatch time to group its device list into
        worker slots. For example, with ``gpus=["cuda:0", "cuda:1",
        "cuda:2", "cuda:3"]`` and ``gpus_per_instance == 2``, ToolPool
        creates 2 workers: one on ``cuda:0,cuda:1`` and one on
        ``cuda:2,cuda:3``. A return of ``0`` declares the tool doesn't use
        the pool's GPUs at all (CPU-only); ToolPool then either fans out
        across CPU workers (if ``cpus_per_instance`` is a positive int) or
        dispatches as a single direct call (if ``cpus_per_instance`` is ``None``).

        Default is derived from ``self.device`` via :func:`parse_device_string`:
            - ``"cpu"`` → 0 (no GPUs needed)
            - ``"cuda"`` / ``"cuda:N"`` → 1
            - ``"cudaxN"`` / ``"cuda:0,cuda:1"`` → N
            - ``"cloud"`` → 1 (cloud dispatch handled before pool partitioning)

        Override in subclasses when GPU need is decoupled from the device
        string — e.g. a model whose large checkpoint needs 4 GPUs regardless
        of input device, or a tool that toggles real GPU use via a separate
        config flag (see ``ColabfoldSearchConfig.gpus_per_instance``).
        """
        from proto_tools.utils.device import parse_device_string

        spec = parse_device_string(self.device)
        return 0 if spec.kind == "cpu" else spec.count

    @property
    def cpus_per_instance(self) -> int | None:
        """Per-instance CPU consumption — drives ToolPool's CPU fan-out.

        Read by ToolPool only when ``gpus_per_instance == 0`` (CPU mode):

            - ``None`` (default): no fan-out. ToolPool dispatches a single
              direct call with all items and ``pool.cpus`` is ignored — the
              tool stays off the pool's CPU scheduler. This is the safe
              default: spinning up N persistent worker subprocesses (each
              holding its own venv in RAM, each paying a startup tax) only
              pays off when per-call work is heavy enough to amortize that
              cost. For most CPU tools — short per-item compute, internal
              threading (mmseqs2 ``--threads``, mafft), or network IO
              against rate-limited services (NCBI, UniProt, RCSB) — the
              single direct call is the right answer.
            - Positive int N: opt in to fan-out. ToolPool spawns
              ``max(1, pool.cpus // N)`` independent worker subprocesses,
              partitions items via LPT, and pins each worker's
              OMP/MKL/OPENBLAS/NUMEXPR thread budgets to N. Override only
              when (a) per-call work is heavy enough to amortize subprocess
              startup, (b) the tool is single-threaded (or N-threaded) per
              call, and (c) items are embarrassingly parallel.

        Canonical opt-in: PyRosetta (heavy ``init``, multi-second per pose,
        independent poses). Most other CPU tools should leave the default.

        When ``gpus_per_instance > 0`` this property is ignored.
        """
        return None

    @classmethod
    def minimal(cls, **kwargs: Any) -> "BaseConfig":
        """Create a config instance with minimal-cost defaults for smoke testing.

        Returns a valid config that exercises the tool's core logic as cheaply
        as possible — disabling expensive optional features (e.g. MSA generation),
        reducing iteration counts, and lowering sample counts.

        Subclasses override this to set tool-specific minimal defaults using
        ``setdefault`` so callers can still override any field explicitly::

            @classmethod
            def minimal(cls, **kwargs):
                kwargs.setdefault("use_msa", False)
                return super().minimal(**kwargs)

        This is used by parametrized test infrastructure (env-report, seed
        reproducibility) to run every registered tool without tool-specific
        hardcoding in test helpers.

        Args:
            **kwargs (Any): Field values passed to the config constructor. These
                take precedence over minimal defaults set by subclasses.

        Returns:
            BaseConfig: An instance of the config with minimal-cost defaults applied.
        """
        return cls(**kwargs)

    def effective_timeout(self) -> int | None:
        """Return the timeout the framework enforces. Override when the cap depends on other fields.

        Returns:
            int | None: Effective timeout in seconds, or None for no cap.
        """
        return self.timeout

    def preprocess(self, inputs: BaseToolInput) -> BaseToolInput:
        """Transform inputs before tool execution. Override in subclasses."""
        return inputs
