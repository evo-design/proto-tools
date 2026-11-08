"""tests/structure_design_tests/_rfd3_precision_site_env_checks.py.

Checks ``rfd3_precision_site/sitecustomize.py`` against the real torch and Foundry inside the
rfdiffusion3 env, under the bfloat16 autocast rfd3's ``bf16-mixed`` inference runs in. Run through
``run_in_env`` with the sitecustomize path as the only argument; prints ``ok`` when every check
passes.
"""

import importlib.util
import os
import sys
from importlib import metadata
from pathlib import Path

import torch
from foundry.utils import alignment

SITE = Path(sys.argv[1])
DEVICE = os.environ["RUN_IN_ENV_DEVICE"]
DEVICE_TYPE = DEVICE.split(":")[0]


def load_site(name: str) -> object:
    """Execute the sitecustomize file as a fresh module (an env-level sitecustomize may already be imported)."""
    spec = importlib.util.spec_from_file_location(name, SITE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


original = alignment.weighted_rigid_align

# A rigid copy of a point cloud: aligning it back onto the original recovers the original.
torch.manual_seed(0)
X = torch.randn(2, 16, 3, device=DEVICE)
rotation = torch.linalg.qr(torch.randn(3, 3, device=DEVICE))[0]
X_gt = X @ rotation.T + 0.5

# Unpatched, the covariance einsum runs in bfloat16 under autocast and the SVD rejects it.
try:
    with torch.autocast(DEVICE_TYPE, dtype=torch.bfloat16):
        original(X, X_gt)
except (NotImplementedError, RuntimeError) as exc:
    assert "BFloat16" in str(exc), exc
else:
    raise AssertionError("Foundry's alignment now runs under bf16 autocast unpatched; the patch may be unneeded")

# A Foundry release outside the audited set is left alone.
real_version = metadata.version
installed = real_version("rc-foundry")
metadata.version = lambda dist: "0.0.0-unaudited" if dist == "rc-foundry" else real_version(dist)
try:
    load_site("_rfd3_precision_site_unaudited")
finally:
    metadata.version = real_version
assert alignment.weighted_rigid_align is original

# The installed release is patched, and the patched alignment works under bf16 autocast.
site = load_site("_rfd3_precision_site_installed")
assert installed in site._AFFECTED_RC_FOUNDRY_VERSIONS, installed
patched = alignment.weighted_rigid_align
assert patched is not original and patched._proto_low_precision_safe

with torch.autocast(DEVICE_TYPE, dtype=torch.bfloat16):
    aligned = patched(X, X_gt)
    assert torch.is_autocast_enabled(DEVICE_TYPE), "autocast must be restored for the surrounding diffusion step"
assert aligned.dtype == torch.float32, aligned.dtype
assert torch.allclose(aligned, X, atol=1e-3), (aligned - X).abs().max()

# Low-precision inputs come back in their own dtype.
assert patched(X.bfloat16(), X_gt.bfloat16()).dtype == torch.bfloat16

print("ok")
