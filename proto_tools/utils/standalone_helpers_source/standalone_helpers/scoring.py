"""Scoring metric helpers for standalone inference scripts."""

import math

from .proto_logging import get_logger

logger = get_logger(__name__)


def log_likelihood_metrics(avg_log_likelihood: float, seq_len: int) -> dict[str, float]:
    """Return the canonical ``{log_likelihood, avg_log_likelihood, perplexity}`` triple.

    Args:
        avg_log_likelihood (float): Mean per-position log-likelihood (<= 0).
        seq_len (int): Number of valid positions used to compute the mean.

    Returns:
        dict[str, float]: Mapping with ``log_likelihood`` (sum = ``avg * seq_len``,
            <= 0), ``avg_log_likelihood`` (<= 0), and ``perplexity``
            (``exp(-avg_log_likelihood)``, >= 1).
    """
    return {
        "log_likelihood": avg_log_likelihood * seq_len,
        "avg_log_likelihood": avg_log_likelihood,
        "perplexity": math.exp(-avg_log_likelihood),
    }
