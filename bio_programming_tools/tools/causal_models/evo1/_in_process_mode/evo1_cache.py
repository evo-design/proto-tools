"""Model cache helpers for Evo1."""
from __future__ import annotations

from typing import Any, Dict

from ..standalone.inference import Evo1Model

_evo1_model_cache: Dict[str, Evo1Model] = {}


def get_cached_evo1_model(
    model_name: str,
    device: str = "cuda",
) -> Any:
    """Get or create cached Evo1 model instance."""
    cache_key = f"{model_name}:{device}"
    if cache_key not in _evo1_model_cache:
        _evo1_model_cache[cache_key] = Evo1Model(
            model_name=model_name,
            device=device,
        )
    return _evo1_model_cache[cache_key]


def clear_evo1_cache() -> None:
    """Clear all cached Evo1 models."""
    for model in _evo1_model_cache.values():
        model.unload()
    _evo1_model_cache.clear()
