"""Per-item seed unroll for ``seed_sensitive`` iterable-input tools."""

from collections.abc import Callable
from typing import Any


def unroll_per_item_seed(
    func: Callable[..., Any],
    inputs: Any,
    config: Any,
    instance: Any,
    iterable_input_field: str,
    iterable_output_field: str,
    *,
    seeds: list[int] | None = None,
    extra_config_update: dict[str, Any] | None = None,
    skip_seed_sensitive_cache: bool = False,
) -> Any:
    """Dispatch ``func`` once per item with a per-item-seed config copy; stitch outputs.

    Args:
        func (Callable[..., Any]): Tool function to invoke per item.
        inputs (Any): Input model containing the iterable field.
        config (Any): Tool config; must expose ``derive_per_item_seeds``.
        instance (Any): Forwarded positionally to ``func``; ``None`` omits.
        iterable_input_field (str): Iterable input field name.
        iterable_output_field (str): Iterable output field name.
        seeds (list[int] | None): Pre-derived seeds; defaults to ``config.derive_per_item_seeds(N)``.
        extra_config_update (dict[str, Any] | None): Extra fields merged into the per-item config.
        skip_seed_sensitive_cache (bool): Whether recursive per-item calls should skip cache.

    Returns:
        Any: Stitched output instance.
    """
    items = list(getattr(inputs, iterable_input_field))
    if not items:
        return _call(func, inputs, config, instance, skip_seed_sensitive_cache)
    if seeds is None:
        seeds = config.derive_per_item_seeds(len(items))
    stitched: list[Any] = []
    warnings_acc: list[str] = []
    errors_acc: list[str] = []
    time_acc = 0.0
    success_acc = True
    # Parallel per-item lists (e.g. logits, scores, psce) stitched if every iteration emits length 1.
    companions: dict[str, list[Any]] = {}
    _scalar_keys = {iterable_output_field, "warnings", "errors", "metadata"}
    last_result: Any = None
    for item, seed in zip(items, seeds, strict=True):
        single_input = inputs.model_copy(update={iterable_input_field: [item]})
        update: dict[str, Any] = {"seed": seed}
        if extra_config_update:
            update.update(extra_config_update)
        single_config = config.model_copy(update=update)
        single_result = _call(func, single_input, single_config, instance, skip_seed_sensitive_cache)
        out_items = list(getattr(single_result, iterable_output_field, []) or [])
        if len(out_items) != 1:
            raise RuntimeError(
                f"unroll_per_item_seed: expected single-item dispatch to return 1 "
                f"{iterable_output_field}, got {len(out_items)}"
            )
        stitched.append(out_items[0])
        warnings_acc.extend(getattr(single_result, "warnings", []) or [])
        errors_acc.extend(getattr(single_result, "errors", []) or [])
        time_acc += getattr(single_result, "execution_time", 0.0) or 0.0
        if getattr(single_result, "success", True) is False:
            success_acc = False
        for fname in getattr(type(single_result), "model_fields", {}):
            if fname in _scalar_keys:
                continue
            val = getattr(single_result, fname, None)
            if isinstance(val, list) and len(val) == 1:
                companions.setdefault(fname, []).append(val[0])
        last_result = single_result
    stitched_update: dict[str, Any] = {
        iterable_output_field: stitched,
        "warnings": warnings_acc,
        "errors": errors_acc,
        "execution_time": time_acc,
        "success": success_acc,
    }
    stitched_update.update({fname: acc for fname, acc in companions.items() if len(acc) == len(items)})
    return last_result.model_copy(update=stitched_update)


def _call(func: Callable[..., Any], inputs: Any, config: Any, instance: Any, skip_seed_sensitive_cache: bool) -> Any:
    """Invoke ``func`` with or without ``instance``."""
    if instance is None:
        if skip_seed_sensitive_cache:
            return func(inputs, config, _skip_seed_sensitive_cache=True)
        return func(inputs, config)
    if skip_seed_sensitive_cache:
        return func(inputs, config, instance, _skip_seed_sensitive_cache=True)
    return func(inputs, config, instance)
