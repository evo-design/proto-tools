#!/bin/bash
# Setup script for AlphaFold3 standalone environment.
#
# Two install paths:
#   1. Sif path: build a portable Apptainer sif locally from the bundled
#      Singularity.def recipe. Output is a single ~4 GB file at
#      $VENV_PATH/alphafold3.sif that caches across env rebuilds and can be
#      copied to other machines. No external registry involved.
#   2. Env path: build AlphaFold3 from source into a micromamba venv (CUDA
#      toolkit + JAX + HMMER + zlib/zstd + AF3 clone + pip install +
#      build_data). Works everywhere micromamba runs.
#
# By default we autodetect which is viable: if the kernel exposes user
# namespaces (required for apptainer's rootless build), we attempt the sif path
# and fall back to env on failure. Otherwise we skip straight to env.
# ALPHAFOLD3_BUILD_SIF={1,0,auto} forces / disables / autodetects respectively.
#
# At runtime, inference.py prefers whichever path is present: if
# $VENV_PATH/alphafold3.sif exists (or the tool config sets sif_path), it runs
# run_alphafold.py inside the image via `apptainer exec`; otherwise it uses the
# in-env Python install.
#
# Both paths share the same model parameters, downloaded up front from the URL
# DeepMind publishes them at.
#
# MSAs are supplied by the caller via the input JSON (proto_tools delegates
# MSA generation to mmseqs2-homology-search), so at runtime we pass
# --norun_data_pipeline and skip the ~630 GB sequence DBs regardless of path.

set -euo pipefail
source standalone_helpers.sh

echo "Setting up AlphaFold3 standalone environment..."

# Download the model parameters (~1 GB) before either install path, so a
# network or disk failure surfaces in seconds rather than after a ~30 min build.
# Kept zstd-compressed because AF3's params loader reads *.bin.zst directly, so
# decompressing would only cost ~3 GB of disk. The weights license permits
# neither commercial use nor redistribution, hence the terms notice below.
proto_resolve_weights_dir alphafold3
AF3_WEIGHTS_URL="https://storage.googleapis.com/alphafold3/af3.bin.zst"
if compgen -G "$WEIGHTS_DIR/*.bin*" >/dev/null; then
  echo "[af3] Model parameters already present in $WEIGHTS_DIR — skipping download"
else
  echo "[af3] Downloading AlphaFold3 model parameters (~1 GB) to $WEIGHTS_DIR ..."
  echo "[af3] Subject to DeepMind's AlphaFold 3 Model Parameters Terms of Use"
  echo "[af3] (non-commercial organizations only; redistribution prohibited):"
  echo "[af3]   https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md"
  # Stage under a dot-prefixed name so an interrupted transfer cannot leave a
  # truncated af3.bin.zst that the *.bin* checks here and in inference.py would
  # both read as provisioned.
  AF3_WEIGHTS_TMP="${WEIGHTS_DIR}/.af3.bin.zst.partial"
  rm -f "$AF3_WEIGHTS_TMP"
  if ! curl -fsSL --retry 3 --retry-delay 5 -o "$AF3_WEIGHTS_TMP" "$AF3_WEIGHTS_URL"; then
    rm -f "$AF3_WEIGHTS_TMP"
    echo "[af3] ERROR: failed to download AlphaFold3 parameters from $AF3_WEIGHTS_URL" >&2
    echo "[af3] Place af3.bin / af3.bin.zst in $WEIGHTS_DIR yourself, or point" >&2
    echo "[af3] PROTO_ALPHAFOLD3_WEIGHTS_DIR at an existing copy." >&2
    exit 1
  fi
  mv "$AF3_WEIGHTS_TMP" "${WEIGHTS_DIR}/af3.bin.zst"
  echo "[af3] Model parameters downloaded to ${WEIGHTS_DIR}/af3.bin.zst"
fi

AF3_VERSION="v3.0.2"
SIF_PATH="${VENV_PATH}/alphafold3.sif"

# ─── BYO sif via PROTO_ALPHAFOLD3_SIF_PATH ──────────────────────────────────
# If the user already has a pre-built sif (e.g. on shared lab storage), point
# this env var at it. We install apptainer + symlink the sif into the venv and
# skip the heavy env build entirely. Mirrors the PROTO_ALPHAFOLD3_WEIGHTS_DIR
# pattern for weights.
if [ -n "${PROTO_ALPHAFOLD3_SIF_PATH:-}" ]; then
  if [ ! -f "$PROTO_ALPHAFOLD3_SIF_PATH" ]; then
    echo "[af3] ERROR: PROTO_ALPHAFOLD3_SIF_PATH points at non-existent file: $PROTO_ALPHAFOLD3_SIF_PATH" >&2
    exit 1
  fi
  echo "[af3] Using pre-built sif from PROTO_ALPHAFOLD3_SIF_PATH=$PROTO_ALPHAFOLD3_SIF_PATH"
  "$MAMBA_BIN" install -p "$VENV_PATH" -c conda-forge -y apptainer
  echo "Installing worker dependencies from requirements.txt..."
  uv pip install -r requirements.txt
  ln -sf "$PROTO_ALPHAFOLD3_SIF_PATH" "$SIF_PATH"
  USE_SIF=1
  AUTO_DETECT=0
  SKIP_BUILD=1
fi

# ─── Decide path: explicit override or autodetect (only if BYO sif not set) ─
BUILD_SIF_PREF="${ALPHAFOLD3_BUILD_SIF:-auto}"
AUTO_DETECT="${AUTO_DETECT:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
if [ "$SKIP_BUILD" -ne 1 ]; then
case "$BUILD_SIF_PREF" in
  1|true|yes) USE_SIF=1 ;;
  0|false|no) USE_SIF=0 ;;
  auto|"")
    AUTO_DETECT=1
    # Apptainer's rootless build needs two things:
    #   (1) kernel user namespaces (detected via `unshare -U`), AND
    #   (2) subuid/subgid mappings for the current user (so the fakeroot-mode
    #       %post script can run apt-get etc. inside the build rootfs).
    # Managed HPC clusters commonly have (1) but not (2), in which case the
    # rootless sif build would fail. We check both up front to avoid ~1 min of
    # wasted build time on those machines before falling back to env path.
    if unshare -U true 2>/dev/null && grep -q "^$(id -un):" /etc/subuid 2>/dev/null; then
      echo "[af3] Autodetect: user namespaces + subuid mappings present — will try sif path"
      USE_SIF=1
    else
      echo "[af3] Autodetect: missing user namespaces and/or subuid mappings — using env path"
      USE_SIF=0
    fi
    ;;
  *)
    echo "[af3] ALPHAFOLD3_BUILD_SIF='$BUILD_SIF_PREF' is not recognised (expected 1/0/auto). Using env path."
    USE_SIF=0
    ;;
esac

# ─── Path 1: local sif build ────────────────────────────────────────────────
if [ "$USE_SIF" -eq 1 ]; then
  echo "[af3] Installing apptainer..."
  if ! "$MAMBA_BIN" install -p "$VENV_PATH" -c conda-forge -y apptainer; then
    if [ "$AUTO_DETECT" -eq 1 ]; then
      echo "[af3] apptainer install failed — falling back to env path"
      USE_SIF=0
    else
      echo "[af3] ERROR: ALPHAFOLD3_BUILD_SIF=1 but apptainer install failed" >&2
      exit 1
    fi
  fi
fi

if [ "$USE_SIF" -eq 1 ]; then
  echo "[af3] Building portable AlphaFold3 sif locally..."

  # Singularity.def lives alongside this script — it mirrors DeepMind's Dockerfile.
  DEF_FILE="$(dirname "$(readlink -f "$0")")/Singularity.def"
  if [ ! -f "$DEF_FILE" ]; then
    echo "[af3] ERROR: Singularity.def not found at $DEF_FILE" >&2
    exit 1
  fi

  # Apptainer's user-namespace build: no root, no Docker daemon required.
  if ! "$VENV_PATH/bin/apptainer" build \
      --build-arg "AF3_VERSION=${AF3_VERSION}" \
      "$SIF_PATH" "$DEF_FILE"; then
    rm -f "$SIF_PATH"
    if [ "$AUTO_DETECT" -eq 1 ]; then
      echo "[af3] sif build failed — falling back to env path"
      USE_SIF=0
    else
      echo "[af3] ERROR: sif build failed (ALPHAFOLD3_BUILD_SIF=1)" >&2
      exit 1
    fi
  else
    echo "[af3] sif built at $SIF_PATH"
  fi
fi

# ─── Path 2: env-based install ──────────────────────────────────────────────
if [ "$USE_SIF" -eq 0 ]; then
  echo "[af3] Using env-based install path..."

  proto_install_cuda_toolkit "${ALPHAFOLD3_CUDA_TOOLKIT_CONSTRAINT:-}"

  echo "Installing worker dependencies from requirements.txt..."
  uv pip install -r requirements.txt

  # AlphaFold3 v3.0.2 pins jax==0.9.1 and jax[cuda12]==0.9.1. Default to that pin
  # unless the user overrides via ALPHAFOLD3_JAX_SPEC.
  ALPHAFOLD3_JAX_SPEC="${ALPHAFOLD3_JAX_SPEC:-jax[cuda12]==0.9.1}"
  export ALPHAFOLD3_JAX_SPEC
  proto_install_jax ALPHAFOLD3

  # AlphaFold3 runtime + build deps:
  #   - hmmer: AF3 calls shutil.which() for jackhmmer/nhmmer/hmmalign/hmmsearch/hmmbuild
  #     at startup regardless of --run_data_pipeline, so HMMER must be on PATH.
  #   - zlib, zstd: required by AF3's cifpp C++ extension at build time (matches the
  #     zlib1g-dev + zstd system deps in DeepMind's Dockerfile).
  echo "Installing HMMER + zlib/zstd dev libs via micromamba..."
  "$MAMBA_BIN" install -p "$VENV_PATH" -c conda-forge -c bioconda -y hmmer zlib zstd

  # Clone and install AlphaFold3 from source (not published to PyPI)
  AF3_REPO_DIR="${VENV_PATH}/src/alphafold3"
  if [ -d "$AF3_REPO_DIR" ]; then
    echo "Updating existing AlphaFold3 clone to $AF3_VERSION..."
    git -C "$AF3_REPO_DIR" fetch --tags --depth 1 origin "$AF3_VERSION"
    git -C "$AF3_REPO_DIR" checkout "$AF3_VERSION"
  else
    echo "Cloning AlphaFold3 ($AF3_VERSION)..."
    mkdir -p "$(dirname "$AF3_REPO_DIR")"
    git clone --depth 1 --branch "$AF3_VERSION" \
      https://github.com/google-deepmind/alphafold3.git "$AF3_REPO_DIR"
  fi

  echo "Installing AlphaFold3 from local clone..."
  # Point g++ at the tool env's include / lib paths so cifpp's #include <zlib.h>
  # resolves to the conda-installed zlib (its CMake find_package only wires up
  # the library, not the include dir, for this particular compile). Mirrors the
  # system deps DeepMind's Dockerfile installs (zlib1g-dev, zstd).
  export CPLUS_INCLUDE_PATH="$VENV_PATH/include:${CPLUS_INCLUDE_PATH:-}"
  export C_INCLUDE_PATH="$VENV_PATH/include:${C_INCLUDE_PATH:-}"
  export LIBRARY_PATH="$VENV_PATH/lib:${LIBRARY_PATH:-}"
  # CMake 4.x (bundled by uv's build isolation) rejects sub-projects declaring
  # cmake_minimum_required(VERSION < 3.5). The forward-compat shim restores the
  # old policy defaults for AF3's FetchContent'd deps (cifpp, pybind11, catch2).
  CMAKE_POLICY_VERSION_MINIMUM=3.5 uv pip install "$AF3_REPO_DIR"

  # AF3's install may pull a jax/jaxlib that conflicts with our CUDA plugin; re-apply pin
  echo "Re-applying JAX spec after AlphaFold3 install..."
  uv pip install --upgrade "$ALPHAFOLD3_JAX_SPEC"

  # Build the chemical-components data file (AF3 Dockerfile runs this post-install).
  # `build_data` is registered as a console-script entry point in AF3's pyproject.toml.
  echo "Building AlphaFold3 chemical components database..."
  build_data

  # Record the repo path so inference.py can locate run_alphafold.py at runtime
  echo "$AF3_REPO_DIR" > "$VENV_PATH/alphafold3_repo_path.txt"
fi

fi  # SKIP_BUILD guard (BYO sif via PROTO_ALPHAFOLD3_SIF_PATH)

if [ "$USE_SIF" -eq 1 ]; then
  echo "AlphaFold3 setup complete! (sif path — image at $SIF_PATH)"
else
  echo "AlphaFold3 setup complete! (env path)"
fi
