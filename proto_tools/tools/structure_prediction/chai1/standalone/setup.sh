#!/bin/bash
# Setup script for Chai1 standalone environment
set -euo pipefail
source standalone_helpers.sh

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo "ERROR: chai1 setup: not supported on aarch64 (chai_lab==0.6.1 pins torch<2.7 lacks sm_121 + ships x86_64-only TorchScript ESM2)" >&2
    exit 1
fi

echo "Setting up Chai1 standalone environment..."

proto_install_pytorch

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt

echo "Upgrading triton..."
# torch 2.6.0 pins triton==3.2.0, which has a PY_SSIZE_T_CLEAN bug causing
# runtime failures with conda-forge Python 3.12. Upgrade AFTER all other installs
# to prevent uv from downgrading it back to 3.2.0 via torch's dependency.
uv pip install --upgrade triton

# chai_lab downloads its weights on first use; staging them here in the persistent weights dir
# (inference.py points CHAI_DOWNLOADS_DIR at it) means env rebuilds never re-fetch them.
proto_resolve_weights_dir chai1

# The model components come from Chai Discovery's HuggingFace repo, pinned to a commit. These
# match chai_lab's own downloads byte for byte; the digests are the files' HuggingFace LFS ids.
HF_BASE="https://huggingface.co/chaidiscovery/chai-1/resolve/9e998997951bbc1680f83939be7f0f4ef16187b2"
declare -A COMPONENT_SHA256=(
    ["feature_embedding.pt"]="58600fb282827ee105b4bf33845c2eb2dce2a7d2ba7582951f76c9290c66c019"
    ["bond_loss_input_proj.pt"]="98f3b2e415d11e460a684cdbfc213ca80447308022b1953cc5326ad468948851"
    ["token_embedder.pt"]="60cada19fd9c77d8298131a00368d162b6bc7d3c836dd36c715db04d21f7bbde"
    ["trunk.pt"]="b7a08e104455f76491088a5eeaea9bb95a08168c48969b0b892971346e51ccbc"
    ["diffusion_module.pt"]="bd77cf4095ad6b8a4eae7e390d01d24be1ebd819922ca3f28e8744f9945c9cad"
    ["confidence_head.pt"]="82a4ac9f934fdd0e73870150f508cecbe010c1383234bdadcaa70d57323deb6b"
)
mkdir -p "${WEIGHTS_DIR}/models_v2" "${WEIGHTS_DIR}/esm"
for COMPONENT in "${!COMPONENT_SHA256[@]}"; do
    proto_download_verified \
        "${HF_BASE}/${COMPONENT}" \
        "${WEIGHTS_DIR}/models_v2/${COMPONENT}" \
        "${COMPONENT_SHA256[$COMPONENT]}"
done

# The ligand conformer cache and the traced ESM2 encoder are published only on Chai's CDN.
CHAI_ASSETS="https://chaiassets.com/chai1-inference-depencencies"
proto_download_verified \
    "${CHAI_ASSETS}/conformers_v1.apkl" \
    "${WEIGHTS_DIR}/conformers_v1.apkl" \
    "f2161256b565bbd84198da8b3e3d3978ae2f179f778d0995cd135c4c0c57c41f"
proto_download_verified \
    "${CHAI_ASSETS}/esm2/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt" \
    "${WEIGHTS_DIR}/esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt" \
    "074673b97e1c1ff9c3cf949294749dd446ad1c83cee50a466b4841bb4d89c27a"

# Warn if GPU compute capability may be incompatible with chai_lab's pinned torch version.
if command -v nvidia-smi &> /dev/null; then
    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]') || true
    if [ -n "$compute_cap" ]; then
        major=$(echo "$compute_cap" | cut -d. -f1)
        minor=$(echo "$compute_cap" | cut -d. -f2)
        # chai_lab==0.6.1 pins torch<2.7 which only supports compute capability up to 12.0
        if [ "$major" -gt 12 ] || { [ "$major" -eq 12 ] && [ "$minor" -gt 0 ]; }; then
            echo ""
            echo "WARNING: Your GPU has CUDA compute capability ${compute_cap}, but chai_lab==0.6.1"
            echo "pins torch<2.7 which only supports up to compute capability 12.0."
            echo "You may see 'no kernel image is available for execution on the device' errors."
            echo "This is a chai_lab version constraint, not a setup issue."
            echo ""
        fi
    fi
fi

echo "Chai1 setup complete!"
