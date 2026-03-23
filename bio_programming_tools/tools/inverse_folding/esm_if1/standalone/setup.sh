#!/bin/bash
# Setup script for ESM-IF/ProteinDPO standalone environment
set -euo pipefail

echo "Setting up ESM-IF standalone environment..."

echo "Installing uv package manager..."
pip install uv

# Install hardware-aware PyTorch version (from centralized detection)
echo "Installing PyTorch: ${RECOMMENDED_TORCH_SPEC:-torch} (platform: ${DETECTED_COMPUTE_PLATFORM:-unknown})"
uv pip install "${RECOMMENDED_TORCH_SPEC:-torch}" --extra-index-url "${RECOMMENDED_TORCH_INDEX}"

echo "Installing torch-geometric (required by ESM-IF GVP modules)..."
uv pip install torch-geometric

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt

# ============================================================================
# Compatibility patches for ESM-IF with modern dependencies
# ============================================================================
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# 1. torch_scatter shim (ESM-IF imports from torch_scatter but building the
#    C extension requires a compiler; use torch_geometric.utils.scatter instead)
echo "Creating torch_scatter compatibility shim..."
SHIM_DIR="${SITE_PACKAGES}/torch_scatter"
mkdir -p "$SHIM_DIR"
cat > "$SHIM_DIR/__init__.py" << 'SHIMEOF'
"""Compatibility shim for torch_scatter using torch_geometric.utils.scatter."""
from torch_geometric.utils import scatter

def scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")
SHIMEOF

# 2. Patch ESM inverse_folding/util.py for biotite >= 1.0 compatibility
#    (filter_backbone was removed; replace with inline atom name filter)
echo "Patching ESM for biotite compatibility..."
ESM_UTIL="${SITE_PACKAGES}/esm/inverse_folding/util.py"
if [ -f "$ESM_UTIL" ] && grep -q "from biotite.structure import filter_backbone" "$ESM_UTIL"; then
    sed -i 's/from biotite.structure import filter_backbone/import numpy as _np  # patched: filter_backbone removed in biotite >= 1.0/' "$ESM_UTIL"
    sed -i 's/bbmask = filter_backbone(structure)/bbmask = _np.isin(structure.atom_name, ["N", "CA", "C", "O"])/' "$ESM_UTIL"
    echo "Patched ESM util.py for biotite compatibility."
fi

# 3. Patch ESM for biotite >= 1.0 API changes and GPU device handling
echo "Applying ESM compatibility patches..."
ESM_GVP="${SITE_PACKAGES}/esm/inverse_folding/gvp_transformer.py"
python -c "
import os

# --- Patch util.py: biotite PDB/CIF API ---
util_path = '$ESM_UTIL'
if os.path.exists(util_path):
    with open(util_path, 'r') as f:
        content = f.read()
    # PDBxFile -> CIFFile, remove 'with open' wrappers
    content = content.replace(
        \"    if fpath.endswith('cif'):\n        with open(fpath) as fin:\n            pdbxf = pdbx.PDBxFile.read(fin)\n        structure = pdbx.get_structure(pdbxf, model=1)\n    elif fpath.endswith('pdb'):\n        with open(fpath) as fin:\n            pdbf = pdb.PDBFile.read(fin)\n        structure = pdb.get_structure(pdbf, model=1)\",
        \"    if fpath.endswith('cif'):\n        pdbxf = pdbx.CIFFile.read(fpath)\n        structure = pdbx.get_structure(pdbxf, model=1)\n    elif fpath.endswith('pdb'):\n        pdbf = pdb.PDBFile.read(fpath)\n        structure = pdb.get_structure(pdbf, model=1)\"
    )
    # GPU support: move target to device and .cpu() before .numpy() in get_sequence_loss
    content = content.replace(
        'target_padding_mask = (target == alphabet.padding_idx)\n    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)',
        'target_padding_mask = (target == alphabet.padding_idx)\n    device = next(model.parameters()).device\n    target = target.to(device)\n    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)'
    )
    content = content.replace(
        'loss = loss[0].detach().numpy()',
        'loss = loss[0].detach().cpu().numpy()'
    )
    content = content.replace(
        'target_padding_mask = target_padding_mask[0].numpy()',
        'target_padding_mask = target_padding_mask[0].cpu().numpy()'
    )
    # GPU support: move tensors in get_encoder_output
    content = content.replace(
        'coords, confidence, _, _, padding_mask = batch_converter(batch)\n    encoder_out = model.encoder.forward(coords, padding_mask, confidence,',
        'coords, confidence, _, _, padding_mask = batch_converter(batch)\n    device = next(model.parameters()).device\n    coords = coords.to(device)\n    confidence = confidence.to(device)\n    padding_mask = padding_mask.to(device)\n    encoder_out = model.encoder.forward(coords, padding_mask, confidence,'
    )
    with open(util_path, 'w') as f:
        f.write(content)

# --- Patch gvp_transformer.py: GPU device handling ---
gvp_path = '$ESM_GVP'
if os.path.exists(gvp_path):
    with open(gvp_path, 'r') as f:
        content = f.read()
    # Patch forward() to move inputs to device
    content = content.replace(
        '    ):\n        encoder_out = self.encoder(coords, padding_mask, confidence,\n            return_all_hiddens=return_all_hiddens)',
        '    ):\n        # Move tensors to model device (patched for GPU support)\n        device = next(self.parameters()).device\n        coords = coords.to(device)\n        padding_mask = padding_mask.to(device)\n        confidence = confidence.to(device)\n        prev_output_tokens = prev_output_tokens.to(device)\n        encoder_out = self.encoder(coords, padding_mask, confidence,\n            return_all_hiddens=return_all_hiddens)'
    )
    # Patch sample() to move inputs to device
    # Insert device detection + tensor moves right after batch_converter, before torch.full
    content = content.replace(
        '            batch_converter([(coords, confidence, None)])\n        )\n',
        '            batch_converter([(coords, confidence, None)])\n        )\n\n        # Move tensors to model device (patched for GPU support)\n        device = next(self.parameters()).device\n        batch_coords = batch_coords.to(device)\n        confidence = confidence.to(device)\n        padding_mask = padding_mask.to(device)\n',
    )
    content = content.replace(
        'sampled_tokens = torch.full((1, 1+L), mask_idx, dtype=int)',
        'sampled_tokens = torch.full((1, 1+L), mask_idx, dtype=int, device=device)',
    )
    with open(gvp_path, 'w') as f:
        f.write(content)
"
echo "Applied ESM compatibility patches."

# Download model weights
WEIGHTS_DIR="${ESMIF_WEIGHTS_DIR:-${VENV_PATH}/weights}"
mkdir -p "$WEIGHTS_DIR"

# Download ESM-IF1 vanilla weights
ESMIF_WEIGHTS_FILE="${WEIGHTS_DIR}/esm_if1_gvp4_t16_142M_UR50.pt"
if [ ! -f "$ESMIF_WEIGHTS_FILE" ]; then
    echo "Downloading ESM-IF1 vanilla weights..."
    curl -fsSL -o "$ESMIF_WEIGHTS_FILE" \
        "https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt"
else
    echo "ESM-IF1 weights already present."
fi

# Download ProteinDPO weights from Zenodo
PROTEINDPO_WEIGHTS_FILE="${WEIGHTS_DIR}/paired_weights.pt"
if [ ! -f "$PROTEINDPO_WEIGHTS_FILE" ]; then
    echo "Downloading ProteinDPO weights from Zenodo..."
    curl -fsSL -o "${WEIGHTS_DIR}/paired_weights.pt.zip" \
        "https://zenodo.org/records/11218181/files/paired_weights.pt.zip"
    python -c "import zipfile; zipfile.ZipFile('${WEIGHTS_DIR}/paired_weights.pt.zip').extractall('${WEIGHTS_DIR}')"
    rm -f "${WEIGHTS_DIR}/paired_weights.pt.zip"
else
    echo "ProteinDPO weights already present."
fi

echo "ESM-IF setup complete!"
