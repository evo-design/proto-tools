#!/bin/bash
# Setup script for the netMHCpan standalone environment.
#
# netMHCpan is user-provisioned: DTU distributes it under an academic license that
# requires per-user registration, and the download is not available unauthenticated,
# so it is NOT auto-downloaded. The license also forbids redistribution, so the
# archive is never vendored into this repository or baked into an image.
#
# Install it once and either set NETMHCPAN_HOME (or NMHOME) to the netMHCpan-4.1
# root, or point PROTO_NETMHCPAN_WEIGHTS_DIR (or PROTO_MODEL_CACHE/netmhcpan) at it.
set -euo pipefail
source standalone_helpers.sh

# netMHCpan ships per-platform subdirectories named "<uname -s>_<uname -m>", which the
# upstream wrapper derives the same way.
PLATFORM="$(uname -s)_$(uname -m)"
BIN_RELPATH="${PLATFORM}/bin/netMHCpan"
DATA_RELPATH="${PLATFORM}/data/MHC_pseudo.dat"

# The network weights and threshold tables ship separately from the software and, unlike
# the software, are an ungated public download, so they are fetched automatically once a
# licensed install is in place.
DATA_URL="https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/data.tar.gz"

# Fetch and extract the data files into an install root that is missing them.
# Returns non-zero (with an explanation) if it cannot safely do so, leaving the caller to
# print the manual instructions.
fetch_netmhcpan_data() {
    local root="$1" tmp
    # The tarball is version-specific and 4.1 / 4.2 data are not interchangeable, so only
    # provision an install that positively identifies itself as 4.1.
    if [ ! -f "${root}/netMHCpan-4.1.readme" ]; then
        echo "[netmhcpan] Cannot confirm ${root} is a netMHCpan-4.1 install (netMHCpan-4.1.readme" >&2
        echo "[netmhcpan] is absent), so the version-specific data files were not downloaded." >&2
        return 1
    fi
    if [ ! -w "$root" ]; then
        echo "[netmhcpan] ${root} is not writable, so the data files cannot be added there." >&2
        return 1
    fi
    echo "[netmhcpan] Data files are missing; downloading the ungated data.tar.gz (~30 MB)..."
    tmp="$(mktemp -d)"
    if ! curl -fsSL --retry 3 --connect-timeout 20 -o "${tmp}/data.tar.gz" "$DATA_URL"; then
        echo "[netmhcpan] Download failed from ${DATA_URL}" >&2
        rm -rf "$tmp"
        return 1
    fi
    # Extracts to <root>/data. The platform directory ships <PLATFORM>/data as a relative
    # symlink to it, which is the path the tool actually reads.
    if ! tar -xzf "${tmp}/data.tar.gz" -C "$root"; then
        echo "[netmhcpan] Could not extract the data archive into ${root}" >&2
        rm -rf "$tmp"
        return 1
    fi
    rm -rf "$tmp"
    if [ ! -f "${root}/${DATA_RELPATH}" ]; then
        echo "[netmhcpan] Extracted data.tar.gz but ${DATA_RELPATH} still does not resolve." >&2
        return 1
    fi
    echo "[netmhcpan] Data files installed at ${root}/data"
    return 0
}

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
    if fetch_netmhcpan_data "$netmhcpan_missing_data"; then
        netmhcpan_found="$netmhcpan_missing_data"
        echo "[netmhcpan] Using netMHCpan install at ${netmhcpan_found}"
    else
        echo "ERROR: netMHCpan at ${netmhcpan_missing_data} is missing its data files (${DATA_RELPATH})" >&2
        echo "and they could not be downloaded automatically. Fetch ${DATA_URL}" >&2
        echo "and extract it inside the netMHCpan-4.1 directory (it creates the 'data' subdirectory)." >&2
        exit 1
    fi
elif [ -n "$netmhcpan_configured" ]; then
    echo "ERROR: a netMHCpan path is set but ${BIN_RELPATH} was not found there." >&2
    echo "Checked NETMHCPAN_HOME / NMHOME / PROTO_NETMHCPAN_WEIGHTS_DIR / PROTO_MODEL_CACHE/netmhcpan;" >&2
    echo "fix the path to a netMHCpan-4.1 root." >&2
    exit 1
else
    {
        echo "[proto-tools] ASSET_NOT_AVAILABLE: netmhcpan:install"
        echo "netMHCpan (${BIN_RELPATH}) is not provisioned; it is required by the netmhcpan tools."
        echo "License / access: DTU academic license, free for publicly funded academic/education/"
        echo "research institutions, gated behind per-user registration. Commercial use requires a"
        echo "separate license from health-software@dtu.dk."
        echo "Provisioning steps (see the toolkit's SETUP.md for copy-paste commands):"
        echo "  1. Accept the academic license and request the Linux package at"
        echo "     https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/ (an institutional"
        echo "     e-mail address is required; a download link is mailed to you)."
        echo "  2. Extract netMHCpan-4.1b.<platform>.tar.gz. Its network weights are a"
        echo "     separate ungated download that proto-tools fetches automatically, so the"
        echo "     software is the only part you obtain by hand."
        echo "  3. Move the install into the managed cache so ${BIN_RELPATH} is found"
        echo "     automatically (no environment variable needed), or set NETMHCPAN_HOME /"
        echo "     PROTO_NETMHCPAN_WEIGHTS_DIR to the netMHCpan-4.1 root."
        echo "No file in the install needs editing: proto-tools invokes the platform binary"
        echo "directly with -rdir/-tdir, bypassing the tcsh wrapper (whose modification the"
        echo "license forbids, and which hardcodes a DTU-internal path)."
    } >&2
    exit 64
fi

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "netMHCpan setup complete!"
