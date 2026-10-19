"""proto_tools/utils/standalone_helpers_source/standalone_helpers/relaxation.py.

Shared continuous relaxation of a sequence state for gradient backends.

Not re-exported from the package ``__init__``: standalones import
``from standalone_helpers.relaxation import relaxed_probs`` directly, so torch-less
environments can still import the package.
"""

import math
from typing import Any

from .proto_logging import get_logger

logger = get_logger(__name__)


def relaxed_probs(logits: Any, *, temperature: float | None, soft: float = 1.0, hard: float = 0.0) -> Any:
    """Relax a ``... x L x V`` sequence state into the probabilities a model consumes.

    ``temperature`` sets the input coordinate system, matching the shared gradient contract:
    a positive float reads ``logits`` as unnormalized and applies ``softmax(logits / T)``, so
    the gradient taken w.r.t. the input leaf is logit-space; ``None`` reads it as a probability
    distribution and passes it through, making that gradient probability-space.

    ``soft`` and ``hard`` are the ColabDesign-style blend: ``soft`` interpolates from the hard
    argmax one-hot (0) to the relaxed probabilities (1), and ``hard`` mixes in a
    straight-through estimator with a hard forward pass and soft backward derivative.

    Args:
        logits (Any): Sequence state tensor whose last axis spans the vocabulary.
        temperature (float | None): Positive softmax temperature, or ``None`` to consume
            ``logits`` as probabilities.
        soft (float): Blend from hard argmax one-hot (0) to relaxed probabilities (1).
        hard (float): Straight-through hard-forward coefficient.

    Returns:
        Any: Relaxed probabilities with the same shape as ``logits``.
    """
    import torch.nn.functional as F

    probs = logits if temperature is None else F.softmax(logits / temperature, dim=-1)
    hard_probs = F.one_hot(probs.argmax(dim=-1), num_classes=probs.shape[-1]).to(probs.dtype)
    relaxed = (1.0 - soft) * hard_probs + soft * probs
    if hard > 0.0:
        straight_through = hard_probs + probs - probs.detach()
        relaxed = hard * straight_through + (1.0 - hard) * relaxed
    return relaxed


def resolve_temperature(value: Any) -> float | None:
    """Coerce a transported temperature to a positive float, or ``None`` for probability mode.

    Args:
        value (Any): Temperature as it arrived in the worker payload.

    Returns:
        float | None: ``None`` for probability mode, otherwise the float temperature.

    Raises:
        ValueError: If the temperature is not positive and finite.
    """
    if value is None:
        return None
    temperature = float(value)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be positive and finite, got {value!r}")
    return temperature
