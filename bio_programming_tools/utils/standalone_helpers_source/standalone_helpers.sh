#!/bin/bash
# ============================================================================
# Shared shell helper functions for tool standalone setup scripts.
#
# This file is automatically copied to each tool's standalone/ directory
# at runtime alongside standalone_helpers.py.
#
# DO NOT EDIT copies in tools/*/standalone/ — they are overwritten.
# Edit the source at: utils/standalone_helpers_source/standalone_helpers.sh
#
# Usage in setup.sh:
#   source standalone_helpers.sh
#
# Available functions:
#   bpt_install_pytorch [torch_spec]
#   bpt_install_jax [TOOL_PREFIX]
#   bpt_install_cuda_toolkit [constraint] [extra_packages...]
#   bpt_resolve_weights_dir <tool_name>     -> sets $WEIGHTS_DIR
#   bpt_check_gated_hf_repo <repo_id> <license_url> [probe_file]
# ============================================================================


# ---------------------------------------------------------------------------
# bpt_install_pytorch [torch_spec] [extra_packages...]
#
# Install PyTorch using the centralized RECOMMENDED_TORCH_SPEC from
# compute_deps.py. Accepts an optional override for tools that pin versions.
# Additional packages (e.g., torchvision, torchaudio) are installed from the
# same index to ensure version compatibility.
#
# Example:
#   bpt_install_pytorch                          # use recommended spec
#   bpt_install_pytorch "torch==2.6.0"           # pin specific version
#   bpt_install_pytorch "" torchvision            # recommended torch + torchvision
#
# Reference: tools/masked_models/esm2/standalone/setup.sh
# ---------------------------------------------------------------------------
bpt_install_pytorch() {
    local torch_spec="${1:-${RECOMMENDED_TORCH_SPEC:-torch}}"
    shift 2>/dev/null || true
    echo "Installing PyTorch: ${torch_spec} (platform: ${DETECTED_COMPUTE_PLATFORM:-unknown})"
    uv pip install "${torch_spec}" "$@" --extra-index-url "${RECOMMENDED_TORCH_INDEX}" --refresh
}


# ---------------------------------------------------------------------------
# bpt_install_jax [TOOL_PREFIX]
#
# Install JAX with the centralized recommendation. Accepts an optional tool
# prefix for per-tool override env vars (e.g., ALPHAFOLD2_JAX_SPEC).
#
# Example:
#   bpt_install_jax                    # use recommended spec
#   bpt_install_jax ALPHAFOLD2         # check ALPHAFOLD2_JAX_SPEC first
#
# Reference: tools/structure_prediction/alphafold2/standalone/setup.sh
# ---------------------------------------------------------------------------
bpt_install_jax() {
    local tool_prefix="${1:-}"
    local jax_variant jax_spec

    if [ -n "$tool_prefix" ]; then
        local variant_var="${tool_prefix}_JAX_VARIANT"
        local spec_var="${tool_prefix}_JAX_SPEC"
        jax_variant="${!variant_var:-${RECOMMENDED_JAX_VARIANT:-cuda12}}"
        jax_spec="${!spec_var:-${RECOMMENDED_JAX_SPEC:-jax[cuda12]>=0.5,<1}}"
    else
        jax_variant="${RECOMMENDED_JAX_VARIANT:-cuda12}"
        jax_spec="${RECOMMENDED_JAX_SPEC:-jax[cuda12]>=0.5,<1}"
    fi

    echo "Detected platform: ${DETECTED_COMPUTE_PLATFORM:-unknown}"
    echo "Installing JAX: ${jax_spec} (variant: ${jax_variant})"
    uv pip install "${jax_spec}"
}


# ---------------------------------------------------------------------------
# bpt_install_cuda_toolkit [constraint] [extra_packages...]
#
# Install CUDA toolkit locally via micromamba. Installs cuda-toolkit,
# cuda-cudart-dev, and cudnn by default. Accepts optional constraint
# override and extra packages.
#
# Example:
#   bpt_install_cuda_toolkit                         # auto-detect version
#   bpt_install_cuda_toolkit "12.4.*"                # pin CUDA version
#   bpt_install_cuda_toolkit "" cuda-nvcc "gcc=12.*" # add extra packages
#
# Reference: tools/inverse_folding/fampnn/standalone/setup.sh
# ---------------------------------------------------------------------------
bpt_install_cuda_toolkit() {
    local cuda_constraint="${1:-}"
    shift 2>/dev/null || true
    local extra_packages=("$@")

    if [ -z "$cuda_constraint" ]; then
        local cuda_major="${DETECTED_CUDA_VERSION:-12}"
        cuda_constraint="${cuda_major}.*"
    fi

    echo "Installing CUDA toolkit ${cuda_constraint} locally via micromamba..."

    local packages=(
        "cuda-toolkit=${cuda_constraint}"
        "cuda-cudart-dev=${cuda_constraint}"
        "cudnn"
    )
    if [ ${#extra_packages[@]} -gt 0 ]; then
        packages+=("${extra_packages[@]}")
    fi

    if ! "$MAMBA_BIN" create -y -p "$VENV_PATH/cuda_env" -c nvidia -c conda-forge \
        "${packages[@]}"; then
        echo "ERROR: Failed to install CUDA toolkit via micromamba"
        exit 1
    fi
}


# ---------------------------------------------------------------------------
# bpt_resolve_weights_dir <tool_name>
#
# Set the shell variable WEIGHTS_DIR based on BPT_MODEL_CACHE.
# Mirrors the Python resolve_weights_dir() logic in standalone_helpers.py.
#
# Priority:
#   1. BPT_{TOOL_NAME}_WEIGHTS_DIR (per-tool override)
#   2. BPT_MODEL_CACHE:
#      - (default/unset): {PACKAGE_ROOT}/model_cache/{tool_name} (survives env rebuilds)
#      - /absolute/path:  /absolute/path/{tool_name} (shared directory)
#      - IN_ENV:          {VENV_PATH}/model_weight_cache (legacy, per-venv)
#      - NONE:            {VENV_PATH}/weights (fallback)
#
# Example:
#   bpt_resolve_weights_dir fampnn
#   echo "Weights at: $WEIGHTS_DIR"
#
# Reference: tools/inverse_folding/fampnn/standalone/setup.sh
# ---------------------------------------------------------------------------
bpt_resolve_weights_dir() {
    local tool_name="$1"
    local tool_upper
    tool_upper=$(echo "$tool_name" | tr '[:lower:]' '[:upper:]')
    local override_var="BPT_${tool_upper}_WEIGHTS_DIR"
    local override="${!override_var:-}"

    if [ -n "$override" ]; then
        WEIGHTS_DIR="$override"
    elif [ "${BPT_MODEL_CACHE:-}" = "IN_ENV" ]; then
        WEIGHTS_DIR="${VENV_PATH}/model_weight_cache"
    elif [ "${BPT_MODEL_CACHE:-}" = "NONE" ]; then
        WEIGHTS_DIR="${VENV_PATH}/weights"
    elif [ -n "${BPT_MODEL_CACHE:-}" ]; then
        WEIGHTS_DIR="${BPT_MODEL_CACHE}/${tool_name}"
    else
        # Default: repo-local model_cache/ directory
        WEIGHTS_DIR="${PACKAGE_ROOT}/model_cache/${tool_name}"
    fi
    mkdir -p "$WEIGHTS_DIR"
}


# ---------------------------------------------------------------------------
# bpt_check_gated_hf_repo <repo_id> <license_url> [probe_file]
#
# Validate access to a gated HuggingFace repository. Discovers HF tokens
# from env vars, token file, and git-credentials. Exits with a clear error
# message if access is denied.
#
# Example:
#   bpt_check_gated_hf_repo \
#       "EvolutionaryScale/esm3-sm-open-v1" \
#       "https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1"
#
#   bpt_check_gated_hf_repo \
#       "google/alphagenome-all-folds" \
#       "https://huggingface.co/google/alphagenome-all-folds" \
#       "README.md"
#
# Reference: tools/masked_models/esm3/standalone/setup.sh
# ---------------------------------------------------------------------------
bpt_check_gated_hf_repo() {
    local repo_id="$1"
    local license_url="$2"
    local probe_file="${3:-config.json}"

    local hf_token="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
    if [ -z "$hf_token" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
        hf_token="$(cat "$HOME/.cache/huggingface/token")"
    fi
    if [ -z "$hf_token" ] && [ -f "$HOME/.git-credentials" ]; then
        hf_token="$(grep -oP 'https?://[^:]+:\Khf_[^@]+(?=@huggingface\.co)' \
            "$HOME/.git-credentials" 2>/dev/null | head -1)" || true
    fi

    local http_code
    if [ -n "$hf_token" ]; then
        http_code=$(curl -s -o /dev/null -w "%{http_code}" \
            -H "Authorization: Bearer ${hf_token}" \
            "https://huggingface.co/${repo_id}/resolve/main/${probe_file}")
    else
        http_code=$(curl -s -o /dev/null -w "%{http_code}" \
            "https://huggingface.co/${repo_id}/resolve/main/${probe_file}")
    fi

    if [ "$http_code" != "200" ]; then
        echo ""
        echo "============================================================"
        echo "ERROR: Cannot access HuggingFace repo '${repo_id}'"
        echo "============================================================"
        echo ""
        if [ -z "$hf_token" ]; then
            echo "No HuggingFace token found. This is a gated model that"
            echo "requires authentication."
            echo ""
            echo "To fix this:"
            echo "  1. Create a HuggingFace account at https://huggingface.co"
            echo "  2. Accept the model license at:"
            echo "     ${license_url}"
            echo "  3. Create an access token at:"
            echo "     https://huggingface.co/settings/tokens"
            echo "  4. Set the token in your environment:"
            echo "     export HF_TOKEN=hf_..."
            echo "     Or log in with: huggingface-cli login"
        else
            echo "A HuggingFace token was found but access was denied (HTTP ${http_code})."
            echo ""
            echo "To fix this:"
            echo "  1. Visit: ${license_url}"
            echo "  2. Accept the license/terms for this model"
            echo "  3. Re-run the setup"
        fi
        echo ""
        echo "============================================================"
        exit 1
    fi
}
