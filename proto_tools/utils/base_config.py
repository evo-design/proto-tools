"""proto_tools/utils/base_config.py.

Base configuration class for all pydantic configs.
"""

import getpass
import json
import os
import random
import socket
from contextvars import ContextVar
from typing import Any, ClassVar

from pydantic import (
    BaseModel,
    ConfigDict,
    ModelWrapValidatorHandler,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pydantic import Field as PydanticField

from proto_tools.utils.device import RemoteDevice
from proto_tools.utils.tool_io import (
    BaseToolInput,
    _extra_dict,
    _reject_removed_ui_kwargs,
    _require_title_and_description,
)

DEFAULT_TIMEOUT = 3600  # seconds (generous default; heavy GPU tools need headroom under load)
RANDOM_SEED_UPPER_BOUND = 2**31

# Set by an environment that hosts tools rather than running on a user's own machine. Read rather
# than sniffed from any one provider, so proto-tools stays unaware of which it is running on.
HOSTED_ENV_VAR = "PROTO_IS_HOSTED_ENV"


def is_hosted_env() -> bool:
    """Whether this process hosts tools for someone else rather than running on their machine."""
    return os.environ.get(HOSTED_ENV_VAR, "") not in ("", "0")


# Overrides who a call says it is when reaching an external service. Unset derives the account and
# host, so one installation is distinguishable from another without anyone configuring anything.
CLIENT_IDENTITY_ENV_VAR = "PROTO_CLIENT_IDENTITY"


def client_identity() -> str:
    """Who is asking, for services that want callers to identify themselves.

    Derived where the call originates, not where it is sent. A hosted process runs on someone's
    behalf, so naming the container would attribute every user's traffic to one source — which is
    what the services asking for identification are trying to avoid.
    """
    configured = os.environ.get(CLIENT_IDENTITY_ENV_VAR, "").strip()
    if configured:
        return configured
    try:
        return f"{getpass.getuser()}@{socket.gethostname()}"
    except Exception:  # an account this process cannot name is not a failure
        return "unknown"


# Payload key carrying framework state across a process boundary. Stripped before validation,
# so a config crossing to a remote worker never presents it as a field.
INTERNAL_STATE_KEY = "_proto_internal"

# ``id()`` of every config frozen for the duration of an in-flight ``preprocess`` call. A
# ContextVar so concurrent per-chunk preprocess in worker threads each guard their own config.
_preprocess_frozen: ContextVar[frozenset[int]] = ContextVar("_preprocess_frozen", default=frozenset())


def ConfigField(
    default: Any = ...,
    *,
    title: str | None = None,
    description: str | None = None,
    reload_on_change: bool = False,
    include_in_key: bool = True,
    **kwargs: Any,
) -> Any:
    """Custom Field wrapper that automatically adds metadata flags to json_schema_extra.

    Args:
        default (Any): Default value. Use ``...`` for required fields.
        title (str | None): Short user-readable title; must be a non-empty string.
        description (str | None): Field description; must be a non-empty string.
        reload_on_change (bool): If True, changing this field between persistent
            worker calls triggers a subprocess restart.
        include_in_key (bool): If False, field is excluded from tool cache key
            generation. Fields that don't affect computation results (device,
            verbose, timeout) should set this to False.
        kwargs: All other standard Pydantic Field arguments.

    Usage:
        param: int = ConfigField(default=42, title="Param", description="...")
    """
    _require_title_and_description("ConfigField", title, description)
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
            ``proto`` → 1). Override in tool configs where GPU need is
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

    # The tool this config belongs to, stamped by ``ToolRegistry.register``; ``None`` until registered.
    tool_key: ClassVar[str | None] = None

    # True once ``preprocess`` has run for this call. Private rather than a field: it describes
    # what the framework has already done, not anything a caller chooses, so it stays out of the
    # published schema, out of the cache key, and out of the constructor.
    _preprocess_completed: bool = PrivateAttr(default=False)

    # Who asked for this call, when it came from somewhere else. ``None`` on the machine that
    # originated it, where the identity is derived instead.
    _client_identity: str | None = PrivateAttr(default=None)

    verbose: int = ConfigField(
        title="Verbose",
        default=0,
        ge=0,
        le=3,
        description="Verbosity level (0=quiet, 1=info, 2=debug, 3=raw subprocess stderr). True→1, False→0.",
        include_in_key=False,
    )

    # Fields a nested config takes from the config holding it when it did not set its own.
    # A nested config is part of one call, so settings that describe how that call should be
    # carried out belong to the whole tree rather than to whichever config was named directly.
    INHERITED_BY_NESTED_CONFIGS: ClassVar[tuple[str, ...]] = ("verbose",)

    @model_validator(mode="after")
    def _inherit_fields_into_nested_configs(self) -> "BaseConfig":
        """Push :attr:`INHERITED_BY_NESTED_CONFIGS` down to nested configs that left them unset.

        ``Boltz2Config(verbose=2)`` makes its MMseqs2 search verbose too, while a nested config
        that names its own value keeps it. Values are written straight into the instance so the
        nested config stays recorded as unset, letting a later change to the outer value flow
        through again.
        """
        for nested in _reachable_configs(self):
            if nested is self:
                continue
            for name in self.INHERITED_BY_NESTED_CONFIGS:
                if name in type(nested).model_fields and name not in nested.model_fields_set:
                    nested.__dict__[name] = getattr(self, name)
        return self

    def for_hosted_env(self) -> "BaseConfig":
        """Return a config this call can honour in a hosted environment.

        A hosted environment runs tools for someone else and cannot stage a large corpus on
        demand, so a setting that depends on one has to give way. Return a copy with that setting
        changed; the default changes nothing, which is what almost every config wants.

        Only reached when ``preprocess`` runs in the hosted process itself. A caller who prepared
        the work on their own machine keeps every setting they chose, since the corpus was
        available where it was actually needed.

        Returns:
            BaseConfig: This config, or a copy adjusted for hosted execution.
        """
        return self

    @model_validator(mode="wrap")
    @classmethod
    def _restore_internal_state(cls, data: Any, handler: ModelWrapValidatorHandler["BaseConfig"]) -> "BaseConfig":
        """Rebuild framework state carried in ``INTERNAL_STATE_KEY`` by :meth:`to_transport_dict`.

        The key is removed before validation so ``extra="forbid"`` never sees it, which lets a
        payload cross a process boundary without the receiving side needing to know about it.
        """
        internal: dict[str, Any] | None = None
        if isinstance(data, dict) and INTERNAL_STATE_KEY in data:
            raw = data[INTERNAL_STATE_KEY]
            internal = raw if isinstance(raw, dict) else None
            data = {k: v for k, v in data.items() if k != INTERNAL_STATE_KEY}
        config = handler(data)
        if internal:
            config._preprocess_completed = bool(internal.get("preprocess_completed", False))
            config._client_identity = internal.get("client_identity")
        return config

    def to_transport_dict(self, **dump_kwargs: Any) -> dict[str, Any]:
        """Serialize for dispatch to another process, including framework state.

        Used instead of :meth:`model_dump` when sending a config to a remote device, so the
        receiver learns that preprocess already ran. Serialization is explicit rather than a
        :func:`model_serializer` override so that :meth:`cache_key`, which dumps the same model,
        cannot pick the state up and split the cache between caller and worker. A sender that
        forgets this method therefore sends no state, and the worker preprocesses as it does today.

        Args:
            dump_kwargs (Any): Passed to :meth:`model_dump`, letting each transport keep its own
                serialization. Defaults to ``mode="json"``.

        Returns:
            dict[str, Any]: The dumped config, carrying ``INTERNAL_STATE_KEY`` when state is set.
        """
        data = self.model_dump(**{"mode": "json", **dump_kwargs})
        # Always sent, so a hosted process can say who it is acting for; the receiver strips the
        # envelope before validation, so a field it does not recognise costs nothing.
        data[INTERNAL_STATE_KEY] = {
            "preprocess_completed": self._preprocess_completed,
            "client_identity": self._client_identity or client_identity(),
        }
        return data

    @classmethod
    def reload_fields(cls) -> set[str]:
        """Return field names marked with ``reload_on_change=True``."""
        return {name for name, info in cls.model_fields.items() if _extra_dict(info).get("reload_on_change", False)}

    @classmethod
    def cache_exclude_fields(cls) -> set[str]:
        """Return field names marked with ``include_in_key=False``."""
        return {name for name, info in cls.model_fields.items() if not _extra_dict(info).get("include_in_key", True)}

    def cache_key(self) -> str:
        """Deterministic string for cache key generation, excluding non-key fields at every level.

        Nested configs are excluded field by field like the outer one, so a setting that does not
        change the result (``verbose``, ``device``, ``timeout``) never splits the cache, whether it
        sits on this config or on one nested inside it.
        """
        model_dict = self.model_dump(exclude_none=True, exclude=_cache_exclude_map(self))
        return json.dumps(model_dict, sort_keys=True, default=str)

    device: str = ConfigField(
        title="Device",
        default="cpu",
        description="Device to run the tool on (e.g., 'cpu', 'cuda', 'cuda:0', 'proto', 'modal')",
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
            - ``"proto"`` / ``"modal"`` → 1 (remote dispatch happens before pool partitioning)

        Override in subclasses when GPU need is decoupled from the device
        string — e.g. a model whose large checkpoint needs 4 GPUs regardless
        of input device, or a tool that toggles real GPU use via a separate
        config flag (see ``Mmseqs2HomologySearchConfig.gpus_per_instance``).
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

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        """Reject malformed device strings at construction rather than at dispatch.

        Parsing is purely syntactic, so this stays independent of the hardware
        actually present: ``"cuda:7"`` validates on a machine with no GPU. Whether
        a remote device can run *this* tool is a separate question, answered at
        dispatch where the tool key is known.

        Args:
            value (str): Device string being validated.

        Returns:
            str: The device string, unchanged.

        Raises:
            ValueError: If the string is not a recognized device form.
        """
        from proto_tools.utils.device import parse_device_string

        parse_device_string(value)
        return value

    def preprocess(self, inputs: BaseToolInput) -> "BaseToolInput | tuple[BaseToolInput, BaseConfig]":
        """Transform inputs before tool execution. Override in subclasses.

        Return the prepared inputs alone, or — when preprocess also resolves a config setting
        that execution must see — ``(inputs, config)``. Never assign to ``self``: return an
        updated copy instead, so preprocess is idempotent and safe to run per chunk when a
        batch is fanned across workers.

        Args:
            inputs (BaseToolInput): Tool inputs to prepare.

        Returns:
            BaseToolInput | tuple[BaseToolInput, BaseConfig]: Prepared inputs, optionally
                paired with the config to execute with.
        """
        return inputs

    def __setattr__(self, name: str, value: Any) -> None:
        """Assign a field, refusing writes to a config frozen for the duration of ``preprocess``."""
        if name in type(self).model_fields and id(self) in _preprocess_frozen.get():
            raise TypeError(
                f"{type(self).__name__}.preprocess assigned to self.{name}. preprocess must not "
                f"mutate its config — the caller's config would carry the change into later "
                f"calls. Return an updated copy alongside the inputs instead:\n"
                f"    return inputs, self.model_copy(update={{{name!r}: ...}})"
            )
        super().__setattr__(name, value)

    def remote_unsupported_reason(self, device: RemoteDevice) -> str | None:  # noqa: ARG002 — overrides use it
        """Reason this config can't run on ``device``, or ``None`` if it can.

        Override in a tool's config to fail fast at dispatch when a setting needs a local
        resource (e.g. a local database or file) that can't be staged to a remote worker.
        The returned message is surfaced to the caller; ``None`` (the default) means compatible.

        Takes the device because remote targets differ in what they can run. A local file
        is unstageable everywhere and should be refused regardless, but a restriction that
        reflects what Proto chooses to host says nothing about a deployment the caller
        owns and pays for — those must branch on ``device``.

        Args:
            device (RemoteDevice): Remote target, ``"proto"`` or ``"modal"``.

        Returns:
            str | None: User-facing reason, or ``None`` when the config is compatible.
        """
        return None


def _cache_exclude_map(model: BaseModel, _visited: set[int] | None = None) -> dict[str, Any]:
    """Build a ``model_dump(exclude=...)`` mapping that drops non-key fields at every level.

    ``cache_exclude_fields()`` names the fields of one config that do not affect its result.
    Applied only to the outer config, a nested config's ``verbose``, ``device``, and ``timeout``
    would still reach the cache key and split it. This walks the model tree and produces the
    nested exclude mapping Pydantic expects, including ``__all__`` entries for configs held in
    lists or dicts.
    """
    _visited = set() if _visited is None else _visited
    if id(model) in _visited:
        return {}
    _visited.add(id(model))

    exclude: dict[str, Any] = {}
    skip = model.cache_exclude_fields() if isinstance(model, BaseConfig) else set()
    for name in type(model).model_fields:
        if name in skip:
            exclude[name] = True
            continue
        value = getattr(model, name, None)
        if isinstance(value, BaseModel):
            if child := _cache_exclude_map(value, _visited):
                exclude[name] = child
        elif isinstance(value, list | tuple | set | frozenset | dict):
            members = value.values() if isinstance(value, dict) else value
            merged: dict[str, Any] = {}
            for member in members:
                if isinstance(member, BaseModel):
                    merged.update(_cache_exclude_map(member, _visited))
            if merged:
                exclude[name] = {"__all__": merged}
    return exclude


def _reachable_configs(config: BaseConfig) -> list[BaseConfig]:
    """Return ``config`` and every :class:`BaseConfig` reachable from it.

    Descends through nested models and through list, tuple, set, and dict fields, so a config
    held in a collection is found alongside one held directly. Every expanded object is
    recorded, so a shared or self-referencing model graph is walked once and terminates.
    """
    found: list[BaseConfig] = []
    visited: set[int] = set()
    pending: list[Any] = [config]
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        if isinstance(current, BaseModel):
            visited.add(id(current))
            if isinstance(current, BaseConfig):
                found.append(current)
            pending.extend(getattr(current, name, None) for name in type(current).model_fields)
        elif isinstance(current, list | tuple | set | frozenset):
            visited.add(id(current))
            pending.extend(current)
        elif isinstance(current, dict):
            visited.add(id(current))
            pending.extend(current.values())
    return found


def _nested_configs(config: BaseConfig) -> set[int]:
    """Collect ``id()`` of ``config`` and every :class:`BaseConfig` reachable from it."""
    return {id(nested) for nested in _reachable_configs(config)}


def _marked_preprocessed(config: BaseConfig) -> BaseConfig:
    """Return a copy of ``config`` recording that preprocess has run.

    A copy rather than the object itself, since ``preprocess`` may return the caller's config
    unchanged. Recording it in place would make the caller's own object claim the work was
    already done, and a second call reusing that object would skip preprocess entirely.

    Args:
        config (BaseConfig): Config to mark.

    Returns:
        BaseConfig: A copy carrying the completed state.
    """
    prepared = config.model_copy()
    prepared._preprocess_completed = True
    return prepared


def run_preprocess(config: BaseConfig, inputs: BaseToolInput) -> tuple[BaseToolInput, BaseConfig]:
    """Run ``config.preprocess`` with the config frozen, returning prepared inputs and config.

    Accepts either return shape: bare prepared inputs (the common case), or
    ``(inputs, config)`` when preprocess also resolves a config setting that execution must
    see. Callers always get the pair, so the two shapes are normalized in one place.

    Freezing makes the no-mutation contract enforced rather than merely documented: a
    ``preprocess`` that assigns to ``self`` (or to a nested config the caller owns) raises at
    the offending line instead of silently changing the caller's config for later calls.

    Args:
        config (BaseConfig): Config whose ``preprocess`` to run.
        inputs (BaseToolInput): Tool inputs to prepare.

    The returned config records that preprocess ran, so a remote worker handed these prepared
    inputs does not repeat the work. See :meth:`BaseConfig.to_transport_dict`.

    Returns:
        tuple[BaseToolInput, BaseConfig]: Prepared inputs and the config to execute with.

    Raises:
        TypeError: If ``preprocess`` returns neither inputs nor an ``(inputs, config)`` pair.
    """
    # Adjusted before freezing, so preprocess sees the config this environment can actually honour.
    if is_hosted_env():
        config = config.for_hosted_env()

    token = _preprocess_frozen.set(_preprocess_frozen.get() | _nested_configs(config))
    try:
        result = config.preprocess(inputs)
    finally:
        _preprocess_frozen.reset(token)

    # A BaseToolInput is never a tuple, so the shape is unambiguous.
    if not isinstance(result, tuple):
        return result, _marked_preprocessed(config)
    if len(result) != 2:
        raise TypeError(
            f"{type(config).__name__}.preprocess returned a {len(result)}-tuple; return either "
            f"inputs, or (inputs, config)."
        )
    prepared, prepared_config = result
    if not isinstance(prepared_config, BaseConfig):
        raise TypeError(
            f"{type(config).__name__}.preprocess returned ({type(prepared).__name__}, "
            f"{type(prepared_config).__name__}); the second element must be a BaseConfig."
        )
    return prepared, _marked_preprocessed(prepared_config)
