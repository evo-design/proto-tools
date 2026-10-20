"""proto_tools/utils/standalone_helpers_source/standalone_helpers/relaxation.py.

Shared continuous relaxation of a sequence state for gradient backends.

Not re-exported from the package ``__init__``: standalones import
``from standalone_helpers.relaxation import relaxed_sequence`` directly, so torch-less
environments can still import the package.
"""

import math
from typing import Any

from .proto_logging import get_logger

logger = get_logger(__name__)


def relaxed_sequence(logits: Any, *, temperature: float | None, soft: float = 1.0, hard: float = 0.0) -> Any:
    """Relax a ``... x L x V`` sequence state into the representation a model consumes.

    One definition for every gradient backend::

        P = softmax(logits / temperature)  # or logits itself when temperature is None
        D_ste = one_hot(argmax(P)) - detach(P) + P
        U = (1 - soft) * logits + soft * P
        M = (1 - hard) * U + hard * D_ste

    ``soft`` interpolates from the **raw input matrix** (0) to the softmax probabilities (1);
    ``hard`` mixes in a straight-through estimator whose forward value is a one-hot and whose
    gradient flows through ``P``. ``hard=1`` supersedes ``soft``: for a fixed ``P`` both the
    forward value and the surrogate gradient are independent of it.

    Note the ``(1 - soft)`` branch is the caller's own matrix, not an argmax one-hot. Feeding
    raw, unnormalized values to a model is differentiable but outside its training distribution.

    Args:
        logits (Any): Sequence state whose last axis spans the vocabulary.
        temperature (float | None): Positive softmax temperature, or ``None`` to treat ``logits``
            as probabilities already.
        soft (float): Blend from the raw input matrix (0) to softmax probabilities (1).
        hard (float): Straight-through coefficient; 1 forwards a one-hot.

    Returns:
        Any: Relaxed representation with the same shape as ``logits``.
    """
    import torch.nn.functional as F

    probs = logits if temperature is None else F.softmax(logits / temperature, dim=-1)
    blended = probs if soft == 1.0 else (1.0 - soft) * logits + soft * probs
    if hard > 0.0:
        discrete = F.one_hot(probs.argmax(dim=-1), num_classes=probs.shape[-1]).to(probs.dtype)
        straight_through = discrete - probs.detach() + probs
        blended = (1.0 - hard) * blended + hard * straight_through
    return blended


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
