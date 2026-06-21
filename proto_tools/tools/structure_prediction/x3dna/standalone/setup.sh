#!/bin/bash
# Setup script for the X3DNA fiber standalone environment.
#
# X3DNA v2.4 is user-provisioned: it is distributed under CC-BY-NC-4.0 and gated
# behind a free registration on https://x3dna.org, so it is NOT auto-downloaded.
# Install it once and either set the standard X3DNA environment variable to its
# root, or point PROTO_X3DNA_WEIGHTS_DIR (or PROTO_MODEL_CACHE/x3dna) at it.
set -euo pipefail
source standalone_helpers.sh

# ─── Fail-fast install precheck ─────────────────────────────────────────────
# Honor the standard $X3DNA, then the per-tool override / model cache. If an
# install is found we continue; if the user pointed us somewhere invalid we fail
# (exit 1); if nothing is configured we emit the skip sentinel (exit 64) so the
# env build / tests skip cleanly on un-provisioned hosts.
x3dna_found=""
x3dna_configured=""
for candidate in "${X3DNA:-}" "${PROTO_X3DNA_WEIGHTS_DIR:-}" "${PROTO_MODEL_CACHE:+${PROTO_MODEL_CACHE}/x3dna}"; do
    [ -n "$candidate" ] && x3dna_configured="yes"
    if [ -n "$candidate" ] && [ -x "${candidate}/bin/fiber" ]; then
        x3dna_found="$candidate"
        break
    fi
done

if [ -n "$x3dna_found" ]; then
    echo "[x3dna] Using X3DNA install at ${x3dna_found}"
elif [ -n "$x3dna_configured" ]; then
    echo "ERROR: X3DNA path is set but bin/fiber was not found there." >&2
    echo "Checked X3DNA / PROTO_X3DNA_WEIGHTS_DIR / PROTO_MODEL_CACHE/x3dna; fix the path to an x3dna-v2.4 root." >&2
    exit 1
else
    {
        echo "[proto-tools] ASSET_NOT_AVAILABLE: x3dna:install"
        echo "X3DNA v2.4 (bin/fiber) is not provisioned; it is required by the x3dna-fiber tool."
        echo "License / access: https://x3dna.org (CC-BY-NC-4.0)"
        echo "Provisioning steps:"
        echo "  1. Register at https://x3dna.org and download x3dna-v2.4-<platform>.tar.gz."
        echo "  2. Extract it (ships a prebuilt bin/fiber plus its config/ data)."
        echo "  3. Set X3DNA=/path/to/x3dna-v2.4 (the standard X3DNA variable),"
        echo "     or PROTO_X3DNA_WEIGHTS_DIR=/path/to/x3dna-v2.4."
    } >&2
    exit 64
fi

echo "Installing uv package manager..."
pip install uv

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "X3DNA fiber setup complete!"
