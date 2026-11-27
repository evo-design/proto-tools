"""Promote Foundry's rigid-alignment SVD to float32 on low-precision inference."""

from importlib import metadata
from typing import Any

import torch
from foundry.utils import alignment  # type: ignore[import-not-found]
from standalone_helpers import get_logger

logger = get_logger(__name__)

_AFFECTED_RC_FOUNDRY_VERSIONS = {"0.2.0"}

_foundry_version: str | None
try:
    _foundry_version = metadata.version("rc-foundry")
except metadata.PackageNotFoundError:
    _foundry_version = None

if _foundry_version in _AFFECTED_RC_FOUNDRY_VERSIONS and not getattr(
    alignment.weighted_rigid_align, "_proto_low_precision_safe", False
):
    _original_weighted_rigid_align = alignment.weighted_rigid_align

    def _low_precision_safe_weighted_rigid_align(
        X_L: Any,
        X_gt_L: Any,
        X_exists_L: Any = None,
        w_L: Any = None,
    ) -> Any:
        """Run the small Kabsch/SVD calculation in a CUDA-supported dtype.

        Foundry executes motif realignment under bfloat16 autocast on recent GPUs,
        but ``torch.linalg.svd`` has no CUDA bfloat16 kernel. The alignment is only a
        3x3 decomposition, so float32 is both inexpensive and numerically preferable;
        cast the aligned coordinates back for the surrounding diffusion state.

        Upcasting the inputs alone is not enough: autocast casts the covariance
        ``einsum`` back to bfloat16 inside the original function, so the SVD still
        received bfloat16 ("svd_cuda_gesvdjBatched" not implemented for 'BFloat16').
        Autocast is switched off for the call, which also covers float32 inputs
        reaching it under autocast.
        """
        output_dtype = X_L.dtype
        with torch.autocast(device_type=X_L.device.type, enabled=False):
            aligned = _original_weighted_rigid_align(
                X_L.float(),
                X_gt_L.float(),
                X_exists_L,
                w_L.float() if isinstance(w_L, torch.Tensor) else w_L,
            )
        return aligned.to(output_dtype)

    _low_precision_safe_weighted_rigid_align._proto_low_precision_safe = True  # type: ignore[attr-defined]
    alignment.weighted_rigid_align = _low_precision_safe_weighted_rigid_align
elif _foundry_version not in _AFFECTED_RC_FOUNDRY_VERSIONS:
    logger.debug("RFdiffusion3 precision compatibility shim is not required for rc-foundry %s", _foundry_version)
