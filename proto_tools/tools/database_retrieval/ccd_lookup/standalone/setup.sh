#!/bin/bash
# Standalone environment setup for ccd_lookup.
#
# Installs pdbeccdutils + RDKit + gemmi into the isolated micromamba env and
# downloads the wwPDB Chemical Component Dictionary (components.cif) into the
# tool's weights directory. The CCD bundle is ~70MB compressed and ~700MB
# uncompressed; pdbeccdutils prefers the uncompressed mmCIF for fast lookups.

set -euo pipefail

source standalone_helpers.sh

# ----------------------------------------------------------------------------
# Python deps
# ----------------------------------------------------------------------------
pip install uv
uv pip install -r requirements.txt

# ----------------------------------------------------------------------------
# CCD data bundle
# ----------------------------------------------------------------------------
proto_resolve_weights_dir ccd_lookup

CCD_URL="https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"
CCD_GZ="${WEIGHTS_DIR}/components.cif.gz"
CCD_CIF="${WEIGHTS_DIR}/components.cif"

if [ -f "${CCD_CIF}" ]; then
    echo "[ccd_lookup] components.cif already present at ${CCD_CIF}; skipping download."
else
    if [ ! -f "${CCD_GZ}" ]; then
        echo "[ccd_lookup] Downloading CCD bundle from ${CCD_URL}..."
        curl -fsSL -o "${CCD_GZ}" "${CCD_URL}"
    else
        echo "[ccd_lookup] components.cif.gz already present; reusing."
    fi
    echo "[ccd_lookup] Decompressing components.cif.gz..."
    gunzip -k -f "${CCD_GZ}"
    # gunzip -k preserves the .gz; remove it to save ~70MB.
    rm -f "${CCD_GZ}"
fi

echo "[ccd_lookup] Setup complete. CCD data at: ${CCD_CIF}"
