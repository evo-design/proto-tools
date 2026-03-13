# Model Weights & Cache Directories

Many tools download multi-GB model weights to subdirectories of `$HOME` on first use. On systems with small home directory quotas (common on HPC clusters), this can quickly exhaust disk space.

This guide documents where tools store their weights and how to redirect them to larger storage.

---

## Directories to Redirect

If your home directory has limited space, symlink these directories to a larger filesystem **before** running any tools. Replace `$STORAGE` with a path on a filesystem with sufficient space.

```bash
STORAGE=/path/to/large/disk

# 1. ~/.cache — HuggingFace models, torch hub, pip cache, tool environments
#    This is the single most impactful redirect (~100+ GB for all ML models).
mkdir -p $STORAGE/.cache
ln -sfn $STORAGE/.cache ~/.cache

# 2. ~/.local — pip user installs and package data
mkdir -p $STORAGE/.local
ln -sfn $STORAGE/.local ~/.local

# 3. ~/.model_cache — Boltz2 fallback checkpoint location
mkdir -p $STORAGE/.model_cache
ln -sfn $STORAGE/.model_cache ~/.model_cache

# 4. ~/.foundry — RFdiffusion3, LigandMPNN model weights
mkdir -p $STORAGE/.foundry
ln -sfn $STORAGE/.foundry ~/.foundry

# 5. ~/sampling_so3_cache — RFdiffusion3 SO(3) rotation matrices (~9 MB)
mkdir -p $STORAGE/sampling_so3_cache
ln -sfn $STORAGE/sampling_so3_cache ~/sampling_so3_cache

# 6. ~/checkpoint — Protenix default checkpoint location
mkdir -p $STORAGE/protenix_checkpoint
ln -sfn $STORAGE/protenix_checkpoint ~/checkpoint
```

### Verify

```bash
ls -la ~/.cache ~/.local ~/.model_cache ~/.foundry ~/sampling_so3_cache ~/checkpoint 2>/dev/null
```

---

## Alternative: Environment Variables

Instead of symlinks, you can set environment variables in your shell profile (`.bashrc`, `.zshrc`):

```bash
export HF_HOME=/path/to/large/disk/.cache/huggingface
export TORCH_HOME=/path/to/large/disk/.cache/torch
export XDG_CACHE_HOME=/path/to/large/disk/.cache
export PROTENIX_ROOT_DIR=/path/to/large/disk/protenix
```

---

## Cache Locations Reference

### `~/.cache/huggingface/` — Most ML models

| Tool | Model(s) | Approximate Size |
|------|----------|-----------------|
| Evo2 | evo2-1b, evo2-7b, evo2-20b, evo2-40b | 2-80 GB per checkpoint |
| Evo1 | evo-1-8k-base, evo-1-131k-base | ~14 GB |
| ProGen2 | progen2-small through progen2-xlarge | 0.3-12 GB per checkpoint |
| ESM2 | esm2_t6_8M through esm2_t48_15B | 0.03-30 GB per checkpoint |
| ESM3 | esm3-sm-open-v1 (requires `HF_TOKEN`) | ~3 GB |
| ESMFold | facebook/esmfold_v1 | ~3 GB |
| Enformer | EleutherAI/enformer-official-rough | ~1 GB |
| Borzoi | johahi/borzoi-replicate-\* | ~2 GB per replicate |
| AlphaGenome | google/alphagenome-\* | ~2 GB |
| Chai1 | chai-lab models | ~3 GB |
| SpliceTransformer | brianhie/SpTransformer | ~0.5 GB |
| BioEmu | bioemu-v1.0, bioemu-v1.1 | ~2 GB |

### `~/.model_cache/` — Boltz2

Falls back here when `HF_HOME` is not set. If `HF_HOME` is set, Boltz2 uses `$HF_HOME/boltz/` instead.

### `~/.foundry/` — RFdiffusion3, LigandMPNN

Foundry-based model weights.

### `~/.cache/bio_programming_tools/` — Tool environments

For **non-editable** installs only (`pip install bio-programming-tools`). Contains standalone tool environments and micromamba. Editable installs (`pip install -e .`) store these in the project directory instead.

### Tools with no home directory impact

These tools either store weights inside their standalone venv or have no model weights:

- **In-venv weights:** AlphaFold2 (~3.5 GB), ProteinMPNN (~150 MB)
- **User-provided paths:** AlphaFold3 (requires `model_dir`, `db_dir`, `sif_path`), Protenix (`PROTENIX_ROOT_DIR`)
- **No weights:** BLAST, MMseqs2, PyHMMER, MinCED, CRISPR-TracR, MAFFT, Prodigal, Orfipy, TMAlign, USAlign, SEG/Masker, ViennaRNA, Structure Metrics, PDB/NCBI/UniProt/SeqFetch

---

## Environment Variables Reference

| Variable | Default Location | Purpose |
|----------|-----------------|---------|
| `HF_HOME` | `~/.cache/huggingface` | Override HuggingFace cache root |
| `HF_TOKEN` | `~/.cache/huggingface/token` | Gated model authentication (ESM3, etc.) |
| `TORCH_HOME` | `~/.cache/torch` | Override torch hub cache |
| `XDG_CACHE_HOME` | `~/.cache` | Override all XDG-compliant caches |
| `PROTENIX_ROOT_DIR` | `~/checkpoint/` | Override Protenix checkpoint root |
| `ALPHAGENOME_CHECKPOINT_PATH` | (HuggingFace default) | Override AlphaGenome checkpoint |
| `SPLICE_TRANSFORMER_CHECKPOINT` | (HuggingFace default) | Override SpliceTransformer checkpoint |
