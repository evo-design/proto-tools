#!/bin/bash
set -euo pipefail
source standalone_helpers.sh
proto_ensure_uv

CRISPRITZ_VERSION="2.8.1"
CRISPRITZ_COMMIT="1b4cfca5059bb4fc3230b583badafe64494e2cce"
INSTALL_ROOT="${VIRTUAL_ENV}/share/crispritz-${CRISPRITZ_VERSION}"
SOURCE_DIR="${INSTALL_ROOT}/source"
BIN_DIR="${VIRTUAL_ENV}/bin"

uv pip install -r requirements.txt
mkdir -p "${INSTALL_ROOT}" "${BIN_DIR}"
if [ ! -d "${SOURCE_DIR}/.git" ]; then
    git clone --depth 1 --branch "v${CRISPRITZ_VERSION}" https://github.com/pinellolab/CRISPRitz.git "${SOURCE_DIR}"
fi
ACTUAL_COMMIT="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
if [ "${ACTUAL_COMMIT}" != "${CRISPRITZ_COMMIT}" ]; then
    echo "CRISPRitz source revision mismatch: expected ${CRISPRITZ_COMMIT}, got ${ACTUAL_COMMIT}" >&2
    exit 1
fi

SRC="${SOURCE_DIR}/sourceCode/CRISPR-Cas-Tree"
g++ -std=c++11 -O2 -fopenmp "${SRC}/mainParallel.cpp" -o "${BIN_DIR}/crispritz-buildTST"
g++ -std=c++11 -O2 -fopenmp "${SRC}/searchOnTST.cpp" "${SRC}/detailedOutput.cpp" \
    "${SRC}/convert.cpp" -I "${SRC}/include" -o "${BIN_DIR}/crispritz-searchTST"
