"""CPU-side UMAP projection of masked-model embeddings."""

import logging
from collections.abc import Sequence

from proto_tools.tools.masked_models.shared_data_models import Projection2D, SequenceEmbedding

logger = logging.getLogger(__name__)

# Below 4 points a 2D scatter doesn't carry information; we skip the UMAP fit
# and the renderer falls back to its "needs ≥4 sequences" hint.
_MIN_POINTS_FOR_PROJECTION = 4


def compute_projections(
    embeddings: Sequence[Sequence[float]],
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 0,
) -> list[Projection2D] | None:
    """Project embeddings to 2D via UMAP, or return None if too few points."""
    n_points = len(embeddings)
    if n_points < _MIN_POINTS_FOR_PROJECTION:
        return None

    import numpy as np
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, n_points - 1),
        min_dist=min_dist,
        random_state=random_state,
        # UMAP forces single-threaded when random_state is set; being explicit
        # silences the override warning.
        n_jobs=1,
    )
    coords = reducer.fit_transform(np.asarray(embeddings, dtype=np.float32))
    return [Projection2D(x=float(x), y=float(y)) for x, y in coords]


def attach_projections(results: list[SequenceEmbedding]) -> None:
    """Compute UMAP projections for ``results`` and attach them in place."""
    if not results:
        return
    projections = compute_projections([r.mean_embedding for r in results])
    if projections is None:
        # Cache returns stored object references — if a prior call ran a
        # multi-point fit and stored the projections, a small follow-up batch
        # would inherit them. Clear so the renderer falls back to its hint.
        for r in results:
            r.projection = None
        return
    for r, p in zip(results, projections, strict=True):
        r.projection = p
