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

# Official upstream release files. Both are unpickled when the model loads, so
# each download is verified against its digest before anything uses it.
MOL1_RELEASE="https://mol-1-release.oss-cn-shanghai.aliyuncs.com"
MOL1_COMMON_SHA256="f6f953dac716bf89b46799738c85f5e0d4626e0cc905bb805a161671fd5cf898"
MOL1_CHECKPOINT_SHA256="d5b83a86884aa6870565b1a249e8f13c232dfc6dfa311089ab0c7868212b6bd6"

if [ ! -f "${WEIGHTS_DIR}/.mol1-common-ready" ]; then
    proto_download_verified "${MOL1_RELEASE}/common.tar.gz" "${WEIGHTS_DIR}/common.tar.gz" "$MOL1_COMMON_SHA256"
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
proto_download_verified "${MOL1_RELEASE}/checkpoint/mol-1-base.pt" "${WEIGHTS_DIR}/checkpoint/mol-1-base.pt" "$MOL1_CHECKPOINT_SHA256"

echo "Mol-1 setup complete: ${WEIGHTS_DIR}"
