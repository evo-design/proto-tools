"""Model cache helpers for Evo2."""
from __future__ import annotations

from typing import Dict, Literal, Optional

from ..standalone.inference import Evo2Model

EVO2_MODEL_CHECKPOINTS = Literal[
    "evo2_7b",
    "evo2_40b",
    "evo2_7b_base",
    "evo2_40b_base",
    "evo2_1b_base",
    "evo2_7b_262k",
    "evo2_7b_microviridae",
]

_evo2_model_cache: Dict[str, Evo2Model] = {}


def get_cached_evo2_model(
    model_checkpoint: EVO2_MODEL_CHECKPOINTS,
    local_path: Optional[str] = None,
) -> Evo2Model:
    """Get or create cached Evo2 model instance."""
    cache_key = f"{model_checkpoint}:{local_path}"
    if cache_key not in _evo2_model_cache:
        _evo2_model_cache[cache_key] = Evo2Model(
            model_checkpoint=model_checkpoint,
            local_path=local_path,
        )
    return _evo2_model_cache[cache_key]


def clear_evo2_cache() -> None:
    """Clear all cached Evo2 models."""
    for model in _evo2_model_cache.values():
        model.unload()
    _evo2_model_cache.clear()
