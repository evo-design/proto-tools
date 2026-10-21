#!/bin/bash
# Setup script for the netMHCpan standalone environment.
#
# netMHCpan is user-provisioned: DTU distributes it under an academic license that
# requires per-user registration, and the download is not available unauthenticated,
# so it is NOT auto-downloaded. The license also forbids redistribution, so the
# archive is never vendored into this repository or baked into an image.
#
# Install it once and either set NETMHCPAN_HOME (or NMHOME) to the netMHCpan-4.2
# root, or point PROTO_NETMHCPAN_WEIGHTS_DIR (or PROTO_MODEL_CACHE/netmhcpan) at it.
set -euo pipefail
source standalone_helpers.sh

# netMHCpan ships per-platform subdirectories named "<uname -s>_<uname -m>", which the
# upstream wrapper derives the same way.
PLATFORM="$(uname -s)_$(uname -m)"
BIN_RELPATH="${PLATFORM}/bin/netMHCpan-4.2"
DATA_RELPATH="${PLATFORM}/data/MHC_pseudo.dat"

# ─── Fail-fast install precheck ─────────────────────────────────────────────
# Search the same locations the runtime resolves: NETMHCPAN_HOME / NMHOME, then the
# managed weights cache. proto_resolve_weights_dir honors PROTO_NETMHCPAN_WEIGHTS_DIR,
# PROTO_MODEL_CACHE, and the PROTO_HOME default, so a drop-in under the default cache is
# found without any env var. If an install is found we continue; if the user explicitly
# pointed us somewhere invalid we fail (exit 1); if nothing is configured we emit the
# skip sentinel (exit 64) so the env build / tests skip cleanly on un-provisioned hosts.
proto_resolve_weights_dir netmhcpan   # sets $WEIGHTS_DIR (default cache + overrides)
netmhcpan_found=""
netmhcpan_missing_data=""
for candidate in "${NETMHCPAN_HOME:-}" "${NMHOME:-}" "${WEIGHTS_DIR:-}"; do
    if [ -n "$candidate" ] && [ -x "${candidate}/${BIN_RELPATH}" ]; then
        if [ -f "${candidate}/${DATA_RELPATH}" ]; then
            netmhcpan_found="$candidate"
            break
        fi
        netmhcpan_missing_data="$candidate"
    fi
done

# A path set by hand means a missing binary at that path is an error, not an
# un-provisioned host.
netmhcpan_configured=""
[ -n "${NETMHCPAN_HOME:-}" ] && netmhcpan_configured="yes"
[ -n "${NMHOME:-}" ] && netmhcpan_configured="yes"
[ -n "${PROTO_NETMHCPAN_WEIGHTS_DIR:-}" ] && netmhcpan_configured="yes"

if [ -n "$netmhcpan_found" ]; then
    echo "[netmhcpan] Using netMHCpan install at ${netmhcpan_found}"
elif [ -n "$netmhcpan_missing_data" ]; then
    echo "ERROR: netMHCpan at ${netmhcpan_missing_data} is missing its data files (${DATA_RELPATH})." >&2
    echo "The netMHCpan-4.2 archive ships these, so this install is incomplete:" >&2
    echo "re-extract netMHCpan-4.2e<platform>.tar.gz and point at the resulting netMHCpan-4.2 root." >&2
    exit 1
elif [ -n "$netmhcpan_configured" ]; then
    echo "ERROR: a netMHCpan path is set but ${BIN_RELPATH} was not found there." >&2
    echo "Checked NETMHCPAN_HOME / NMHOME / PROTO_NETMHCPAN_WEIGHTS_DIR / PROTO_MODEL_CACHE/netmhcpan;" >&2
    echo "fix the path to a netMHCpan-4.2 root holding ${BIN_RELPATH}." >&2
    exit 1
else
    {
        echo "[proto-tools] ASSET_NOT_AVAILABLE: netmhcpan:install"
        echo "netMHCpan (${BIN_RELPATH}) is not provisioned; it is required by the netmhcpan tools."
        echo "License / access: DTU academic license, free for publicly funded academic/education/"
        echo "research institutions, gated behind per-user registration. Commercial use requires a"
        echo "separate license from health-software@dtu.dk."
        echo "Provisioning steps (see the toolkit's SETUP.md for copy-paste commands):"
        echo "  1. Request the Linux package at"
        echo "     https://services.healthtech.dtu.dk/services/NetMHCpan-4.2/ . After agreeing to"
        echo "     terms of use you will be able to submit a request; an institutional e-mail"
        echo "     address is required, and a download link is e-mailed to you once approved."
        echo "  2. Extract netMHCpan-4.2e<platform>.tar.gz. The networks and threshold tables are"
        echo "     bundled in the archive, so nothing further is downloaded."
        echo "  3. Move the install into the managed cache so ${BIN_RELPATH} is found"
        echo "     automatically (no environment variable needed), or set NETMHCPAN_HOME /"
        echo "     PROTO_NETMHCPAN_WEIGHTS_DIR to the netMHCpan-4.2 root."
        echo "No file in the install needs editing: proto-tools invokes the platform binary"
        echo "directly with -rdir/-tdir, bypassing the tcsh wrapper (whose modification the"
        echo "license forbids, and which hardcodes a DTU-internal path)."
    } >&2
    exit 64
fi

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "netMHCpan setup complete!"
