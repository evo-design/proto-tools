#!/bin/bash
# Install the pinned upstream package with portable PyTorch kernels.
set -euo pipefail
source standalone_helpers.sh

proto_install_pytorch

# Keep the hardware-matched Torch build when resolving upstream's dependencies.
TORCH_CONSTRAINT="${VENV_PATH}/mol1-torch-constraint.txt"
python - <<'PY' > "$TORCH_CONSTRAINT"
from importlib.metadata import version
print(f"torch=={version('torch')}")
PY
uv pip install -r requirements.txt --constraint "$TORCH_CONSTRAINT"

proto_resolve_weights_dir mol1
mkdir -p "${WEIGHTS_DIR}/checkpoint"
MOL1_DOWNLOAD_PARTIAL=""
trap 'rm -f "${MOL1_DOWNLOAD_PARTIAL:-}"' EXIT

download_atomic() {
    local url="$1" destination="$2"
    if [ ! -s "$destination" ]; then
        MOL1_DOWNLOAD_PARTIAL="${destination}.partial.$$"
        curl -fSL --retry 3 -o "$MOL1_DOWNLOAD_PARTIAL" "$url"
        mv "$MOL1_DOWNLOAD_PARTIAL" "$destination"
        MOL1_DOWNLOAD_PARTIAL=""
    fi
}

MOL1_RELEASE="https://mol-1-release.oss-cn-shanghai.aliyuncs.com"
if [ ! -f "${WEIGHTS_DIR}/.mol1-common-ready" ]; then
    download_atomic "${MOL1_RELEASE}/common.tar.gz" "${WEIGHTS_DIR}/common.tar.gz"
    python - "$WEIGHTS_DIR" <<'PY'
import sys
import tarfile
from pathlib import Path

root = Path(sys.argv[1])
with tarfile.open(root / "common.tar.gz") as archive:
    archive.extractall(root, filter="data")
for name in ("components.cif", "components.cif.rdkit_mol.pkl", "clusters-by-entity-40.txt", "obsolete_release_date.csv"):
    if not (root / "common" / name).is_file():
        raise FileNotFoundError(f"Mol-1 reference data missing: {name}")
(root / ".mol1-common-ready").touch()
(root / "common.tar.gz").unlink()
PY
fi
download_atomic "${MOL1_RELEASE}/checkpoint/mol-1-base.pt" "${WEIGHTS_DIR}/checkpoint/mol-1-base.pt"

echo "Mol-1 setup complete: ${WEIGHTS_DIR}"
