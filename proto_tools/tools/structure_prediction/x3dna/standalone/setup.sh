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
# Search the same locations the runtime resolves: the standard $X3DNA, then the
# managed weights cache. proto_resolve_weights_dir honors the
# PROTO_X3DNA_WEIGHTS_DIR override, PROTO_MODEL_CACHE, and the PROTO_HOME default,
# so a drop-in under the default cache is found without any env var. If an install
# is found we continue; if the user explicitly pointed us somewhere invalid we
# fail (exit 1); if nothing is configured we emit the skip sentinel (exit 64) so
# the env build / tests skip cleanly on un-provisioned hosts.
proto_resolve_weights_dir x3dna   # sets $WEIGHTS_DIR (default cache + overrides)
x3dna_found=""
for candidate in "${X3DNA:-}" "${WEIGHTS_DIR:-}"; do
    if [ -n "$candidate" ] && [ -x "${candidate}/bin/fiber" ]; then
        x3dna_found="$candidate"
        break
    fi
done

# "Configured" = the user explicitly pointed us at an install, so a missing
# binary there is an error rather than an un-provisioned-host skip.
x3dna_configured=""
[ -n "${X3DNA:-}" ] && x3dna_configured="yes"
[ -n "${PROTO_X3DNA_WEIGHTS_DIR:-}" ] && x3dna_configured="yes"

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
        echo "Provisioning steps (see the toolkit's SETUP.md for copy-paste commands):"
        echo "  1. Register at https://x3dna.org and download x3dna-v2.4-<platform>.tar.gz."
        echo "  2. Extract it (ships a prebuilt bin/fiber plus its config/ data)."
        echo "  3. Move the install into the managed cache so bin/fiber is found"
        echo "     automatically (no environment variable needed), or set X3DNA /"
        echo "     PROTO_X3DNA_WEIGHTS_DIR to the install root."
    } >&2
    exit 64
fi

echo "Installing uv package manager..."
pip install uv

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "X3DNA fiber setup complete!"
