"""Promote Foundry's rigid-alignment SVD to float32 on low-precision inference."""

from importlib import metadata

import torch
from foundry.utils import alignment
from standalone_helpers import get_logger

logger = get_logger(__name__)

_AFFECTED_RC_FOUNDRY_VERSIONS = {"0.2.0"}

try:
    _foundry_version = metadata.version("rc-foundry")
except metadata.PackageNotFoundError:
    _foundry_version = None

if _foundry_version in _AFFECTED_RC_FOUNDRY_VERSIONS and not getattr(
    alignment.weighted_rigid_align, "_proto_low_precision_safe", False
):
    _original_weighted_rigid_align = alignment.weighted_rigid_align

    def _low_precision_safe_weighted_rigid_align(X_L, X_gt_L, X_exists_L=None, w_L=None):
        """Run the small Kabsch/SVD calculation in a CUDA-supported dtype.

        Foundry executes motif realignment under bfloat16 autocast on recent GPUs,
        but ``torch.linalg.svd`` has no CUDA bfloat16 kernel. The alignment is only a
        3x3 decomposition, so float32 is both inexpensive and numerically preferable;
        cast the aligned coordinates back for the surrounding diffusion state.
        """
        output_dtype = X_L.dtype
        if output_dtype in (torch.float16, torch.bfloat16):
            aligned = _original_weighted_rigid_align(
                X_L.float(),
                X_gt_L.float(),
                X_exists_L,
                w_L.float() if isinstance(w_L, torch.Tensor) else w_L,
            )
            return aligned.to(output_dtype)
        return _original_weighted_rigid_align(X_L, X_gt_L, X_exists_L, w_L)

    _low_precision_safe_weighted_rigid_align._proto_low_precision_safe = True
    alignment.weighted_rigid_align = _low_precision_safe_weighted_rigid_align
elif _foundry_version not in _AFFECTED_RC_FOUNDRY_VERSIONS:
    logger.debug("RFdiffusion3 precision compatibility shim is not required for rc-foundry %s", _foundry_version)
