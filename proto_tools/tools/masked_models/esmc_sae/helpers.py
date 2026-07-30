"""proto_tools/tools/masked_models/esmc_sae/helpers.py.

Look up Biohub's agent-generated descriptions for ESM C SAE features.
"""

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# Biohub publishes descriptions for exactly one SAE. The endpoint takes only a feature
# index, so an index from any other SAE resolves to an unrelated concept.
DESCRIBED_SAE_REPO = "biohub/ESMC-6B-sae-layer60-k64-codebook16384"

_FEATURE_API = "https://biohub.ai/esm/protein/api/v1alpha1/features/{feature_index}"

# The codebook the described SAE was trained with; indices outside it cannot resolve.
_DESCRIBED_CODEBOOK_SIZE = 16384


@lru_cache(maxsize=16384)
def _fetch_feature(feature_index: int, timeout: float) -> dict[str, Any]:
    """Fetch and cache one feature's record from the ESM Atlas API."""
    import requests

    response = requests.get(_FEATURE_API.format(feature_index=feature_index), timeout=timeout)
    response.raise_for_status()
    return dict(response.json())


def _fetch_feature_or_none(feature_index: int, timeout: float) -> dict[str, Any] | None:
    """Fetch one feature's record, or ``None`` if the API has nothing for that index."""
    try:
        return _fetch_feature(feature_index, timeout)
    except Exception as exc:
        logger.debug("esmc_sae: no description for feature %s: %s", feature_index, exc)
        return None


def describe_sae_features(
    feature_indices: list[int] | tuple[int, ...] | int,
    timeout: float = 30.0,
) -> dict[int, dict[str, Any]]:
    """Look up what ESM C SAE features mean, via Biohub's public feature API.

    Descriptions are agent-generated from activation patterns across UniRef90 and are
    published for a single SAE, ``ESMC-6B-sae-layer60-k64-codebook16384``. Feature
    indices are specific to the SAE that produced them, so passing indices from any
    other SAE returns descriptions of unrelated concepts — the endpoint accepts only an
    index and cannot detect the mismatch. Use this only on features obtained with
    ``model_checkpoint="esmc_6b"``, ``layers=[60]``, ``k=64``, ``codebook_size=16384``.

    Each record carries a short ``label``, a longer ``description``, a ``category``, the
    protein families the feature fires on, and the UniRef90 statistics behind the
    normalization Biohub describes (``uniref90_idf``, ``uniref90_max_activation``).

    Results are cached per index for the life of the process, so repeated lookups of the
    same feature cost one request.

    Args:
        feature_indices (list[int] | tuple[int, ...] | int): Codebook indices to look up.
            A single int is accepted for convenience.
        timeout (float): Seconds to wait per request.

    Returns:
        dict[int, dict[str, Any]]: Each requested index mapped to its record. Indices the
            API does not recognize are omitted, and a warning names them.

    Raises:
        ValueError: If an index falls outside the described SAE's codebook.

    Examples:
        >>> from proto_tools.tools.masked_models.esmc_sae import describe_sae_features
        >>> described = describe_sae_features([1251])
        >>> described[1251]["label"]
        'Histidine kinase transmitter module recognition'

    Note:
        This is an alpha API and its response shape may change.
    """
    if isinstance(feature_indices, int):
        feature_indices = [feature_indices]

    out_of_range = [i for i in feature_indices if not 0 <= i < _DESCRIBED_CODEBOOK_SIZE]
    if out_of_range:
        raise ValueError(
            f"feature indices {out_of_range} fall outside the {_DESCRIBED_CODEBOOK_SIZE}-feature "
            f"codebook of {DESCRIBED_SAE_REPO}, the only SAE with published descriptions. "
            f"Indices from a different SAE cannot be described."
        )

    described: dict[int, dict[str, Any]] = {}
    unavailable: list[int] = []
    for feature_index in dict.fromkeys(feature_indices):
        record = _fetch_feature_or_none(feature_index, timeout)
        if record is None:
            unavailable.append(feature_index)
        else:
            described[feature_index] = record

    if unavailable:
        logger.warning(
            f"No description available for {len(unavailable)} feature(s): {unavailable}. "
            f"Descriptions exist only for {DESCRIBED_SAE_REPO}."
        )
    return described
