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
#   proto_install_pytorch [torch_spec]
#   proto_install_jax [TOOL_PREFIX]
#   proto_install_cuda_toolkit [constraint] [extra_packages...]
#   proto_resolve_weights_dir <toolkit>     -> sets $WEIGHTS_DIR
#   proto_check_gated_hf_repo <repo_id> <license_url> [probe_file]
# ============================================================================


# ---------------------------------------------------------------------------
# proto_install_pytorch [torch_spec] [extra_packages...]
#
# Install PyTorch using the centralized RECOMMENDED_TORCH_SPEC from
# compute_deps.py. Accepts an optional override for tools that pin versions.
# Additional packages (e.g., torchvision, torchaudio) are installed from the
# same index to ensure version compatibility.
#
# Example:
#   proto_install_pytorch                          # use recommended spec
#   proto_install_pytorch "torch==2.6.0"           # pin specific version
#   proto_install_pytorch "" torchvision            # recommended torch + torchvision
#
# Reference: tools/masked_models/esm2/standalone/setup.sh
# ---------------------------------------------------------------------------
proto_install_pytorch() {
    local torch_spec="${1:-${RECOMMENDED_TORCH_SPEC:-torch}}"
    shift 2>/dev/null || true
    echo "Installing PyTorch: ${torch_spec} (platform: ${DETECTED_COMPUTE_PLATFORM:-unknown})"
    uv pip install "${torch_spec}" "$@" --extra-index-url "${RECOMMENDED_TORCH_INDEX}" --refresh
}


# ---------------------------------------------------------------------------
# proto_install_jax [TOOL_PREFIX]
#
# Install JAX with the centralized recommendation. Accepts an optional tool
# prefix for per-tool override env vars (e.g., ALPHAFOLD2_JAX_SPEC).
#
# Example:
#   proto_install_jax                    # use recommended spec
#   proto_install_jax ALPHAFOLD2         # check ALPHAFOLD2_JAX_SPEC first
#
# Reference: tools/structure_prediction/alphafold2/standalone/setup.sh
# ---------------------------------------------------------------------------
proto_install_jax() {
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
# proto_install_cuda_toolkit [constraint] [extra_packages...]
#
# Install CUDA toolkit locally via micromamba. Installs cuda-toolkit,
# cuda-cudart-dev, and cudnn by default. Accepts optional constraint
# override and extra packages.
#
# Example:
#   proto_install_cuda_toolkit                         # auto-detect version
#   proto_install_cuda_toolkit "12.4.*"                # pin CUDA version
#   proto_install_cuda_toolkit "" cuda-nvcc "gcc=12.*" # add extra packages
#
# Reference: tools/inverse_folding/fampnn/standalone/setup.sh
# ---------------------------------------------------------------------------
proto_install_cuda_toolkit() {
    local cuda_constraint="${1:-}"
    shift 2>/dev/null || true
    local extra_packages=("$@")

    if [ -z "$cuda_constraint" ]; then
        local cuda_major="${DETECTED_CUDA_VERSION:-12}"
        cuda_constraint="${cuda_major}.*"
    fi

    echo "Installing CUDA toolkit ${cuda_constraint} locally via micromamba..."

    # Pin cuda-version to force all sub-packages to the same CUDA generation.
    # cuda-version is available on conda-forge for versions the nvidia channel lacks.
    local packages=(
        "cuda-version=${cuda_constraint}"
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
# proto_resolve_weights_dir <toolkit>
#
# Set the shell variable WEIGHTS_DIR based on PROTO_MODEL_CACHE.
# Mirrors the Python resolve_weights_dir() logic in standalone_helpers.py.
#
# Priority:
#   1. PROTO_{TOOL_NAME}_WEIGHTS_DIR (per-tool override)
#   2. PROTO_MODEL_CACHE:
#      - (default/unset): {PROTO_HOME}/proto_model_cache/{toolkit}
#      - /absolute/path:  /absolute/path/{toolkit} (shared directory)
#      - IN_ENV:          {VENV_PATH}/model_weight_cache (legacy, per-venv)
#      - NONE:            {VENV_PATH}/weights (fallback)
#
# Example:
#   proto_resolve_weights_dir fampnn
#   echo "Weights at: $WEIGHTS_DIR"
#
# Reference: tools/inverse_folding/fampnn/standalone/setup.sh
# ---------------------------------------------------------------------------
proto_resolve_weights_dir() {
    local toolkit="$1"
    local tool_upper
    tool_upper=$(echo "$toolkit" | tr '[:lower:]' '[:upper:]')
    local override_var="PROTO_${tool_upper}_WEIGHTS_DIR"
    local override="${!override_var:-}"

    if [ -n "$override" ]; then
        WEIGHTS_DIR="$override"
    elif [ "${PROTO_MODEL_CACHE:-}" = "IN_ENV" ]; then
        WEIGHTS_DIR="${VENV_PATH}/model_weight_cache"
    elif [ "${PROTO_MODEL_CACHE:-}" = "NONE" ]; then
        WEIGHTS_DIR="${VENV_PATH}/weights"
    elif [ -n "${PROTO_MODEL_CACHE:-}" ]; then
        WEIGHTS_DIR="${PROTO_MODEL_CACHE}/${toolkit}"
    else
        # Default: PROTO_HOME/proto_model_cache/ directory
        local _proto_home="${PROTO_HOME:-$HOME/.proto}"
        WEIGHTS_DIR="${_proto_home}/proto_model_cache/${toolkit}"
    fi
    mkdir -p "$WEIGHTS_DIR"
}


# ---------------------------------------------------------------------------
# proto_resolve_asset_availability <toolkit> <pattern> [license_url] [asset_kind] [hint]
#
# Standardized fail-fast precheck for tool assets that must be provisioned
# externally (DeepMind-gated weights, NVIDIA NIM models, large databases the
# tool can't auto-download, etc.).
#
# Resolves the asset directory, checks for files matching <pattern>, and on
# failure emits a standard banner plus a machine-readable sentinel line that
# the proto-tools test layer recognises and converts into a *test skip*
# (rather than a failure). This means tools that signal "asset not on disk"
# don't fail CI / smoke runs on machines that haven't been provisioned.
#
# Args:
#   toolkit       Toolkit directory name (e.g. ``alphafold3``). Drives both
#                 the dir resolution (``PROTO_<TOOLKIT>_WEIGHTS_DIR`` and
#                 friends) and the sentinel payload.
#   pattern       Glob (relative to the asset dir) of expected files
#                 (e.g. ``*.bin*`` for AF3, ``*.dbtype`` for an MMseqs2 DB).
#   license_url   Optional URL with download / license instructions; printed
#                 in the banner so the user knows where to go.
#   asset_kind    "weights" (default) | "database" | "dataset". Today only
#                 "weights" is wired to a resolver; other kinds will be added
#                 as their cache layouts are formalised. Used in the sentinel
#                 (``ASSET_NOT_AVAILABLE: <toolkit>:<asset_kind>``).
#   hint          Optional multi-line string with tool-specific provisioning
#                 instructions — DeepMind's 2-3 day approval flow, NVIDIA NIM
#                 key registration, etc. Spliced into BOTH failure banners
#                 (env-var-empty exit 1 and not-provisioned exit 64) so users
#                 see the same guidance regardless of which mode triggered.
#                 Lines are indented 2 spaces under a "Provisioning steps:"
#                 header. Pass via heredoc for readability.
#
# Sets ASSET_DIR to the resolved path on success. Exits 64 with the sentinel
# on the last stderr line on failure (last-line so ``_stderr_tail`` in
# ``tool_instance.py`` always preserves it).
#
# Example:
#   proto_resolve_asset_availability alphafold3 "*.bin*" \
#       "https://github.com/google-deepmind/alphafold3#obtaining-model-parameters" \
#       weights \
#       "$(cat <<'HINT'
#   AlphaFold3 weights are gated by DeepMind's Terms of Use:
#     1. Request access via DeepMind's form (link above).
#     2. Wait 2-3 business days for the approval email.
#     3. Download the weights archive from the emailed link.
#     4. Place af3.bin (or af3.bin.zst) in the resolved directory above.
#   HINT
#   )"
#
# Reference: tools/structure_prediction/alphafold3/standalone/setup.sh
# ---------------------------------------------------------------------------
proto_resolve_asset_availability() {
    local toolkit="$1"
    local pattern="$2"
    local license_url="${3:-}"
    local asset_kind="${4:-weights}"
    local hint="${5:-}"

    local tool_upper override_var override
    tool_upper=$(echo "$toolkit" | tr '[:lower:]' '[:upper:]')
    override_var="PROTO_${tool_upper}_WEIGHTS_DIR"
    override="${!override_var:-}"

    case "$asset_kind" in
        weights)
            proto_resolve_weights_dir "$toolkit"
            ASSET_DIR="$WEIGHTS_DIR"
            ;;
        *)
            echo "ERROR: proto_resolve_asset_availability: asset_kind '$asset_kind' not yet implemented (supported: weights)" >&2
            exit 1
            ;;
    esac

    if compgen -G "$ASSET_DIR/$pattern" >/dev/null; then
        return 0
    fi

    # Local helper to render the optional tool-specific hint with consistent
    # indentation under both failure banners.
    _emit_hint() {
        if [ -z "$hint" ]; then
            return
        fi
        echo ""
        echo "Provisioning steps:"
        while IFS= read -r line; do
            echo "  ${line}"
        done <<< "${hint}"
    }

    # Two failure modes, distinguished by whether the user explicitly pointed
    # us at a directory:
    #
    #   (a) Override env var SET but the path is empty → user-supplied
    #       configuration error. Exit 1 with no sentinel; this should fail
    #       the test (typo, stale path, forgot to copy weights, etc.).
    #
    #   (b) Override env var UNSET → falling through to default cache /
    #       PROTO_MODEL_CACHE. The host just hasn't been provisioned with
    #       optional gated assets. Emit the ASSET_NOT_AVAILABLE sentinel and
    #       exit 64; the test layer converts this into a skip.
    if [ -n "$override" ]; then
        {
            echo ""
            echo "============================================================"
            echo "ERROR: ${toolkit} ${asset_kind} not found at the path you configured"
            echo "============================================================"
            echo "${override_var}=${override}"
            echo "Expected files matching: ${pattern}"
            if [ -n "$license_url" ]; then
                echo ""
                echo "License / access:"
                echo "  ${license_url}"
            fi
            _emit_hint
            echo ""
            echo "Fix: place ${asset_kind} matching '${pattern}' under that directory,"
            echo "or update ${override_var} to point at the correct location."
            echo "============================================================"
        } >&2
        exit 1
    fi

    {
        echo ""
        echo "============================================================"
        echo "${toolkit} ${asset_kind} not provisioned on this host"
        echo "============================================================"
        echo "Resolved location (default cache):"
        echo "  ${ASSET_DIR}"
        echo "Expected files matching: ${pattern}"
        if [ -n "$license_url" ]; then
            echo ""
            echo "License / access:"
            echo "  ${license_url}"
        fi
        _emit_hint
        echo ""
        echo "To enable:"
        echo "  1. Place ${asset_kind} matching '${pattern}' in the directory above, OR"
        echo "  2. Set ${override_var}=/path/to/${asset_kind}/dir and re-run."
        echo "============================================================"
        # Sentinel must be the LAST stderr line so _stderr_tail (last 10
        # non-empty lines) preserves it. The proto-tools test layer parses
        # this exact prefix and converts the failure into a skip.
        echo "[proto-tools] ASSET_NOT_AVAILABLE: ${toolkit}:${asset_kind}"
    } >&2
    exit 64
}


# ---------------------------------------------------------------------------
# proto_check_gated_hf_repo <repo_id> <license_url> [probe_file]
#
# Validate access to a gated HuggingFace repository. Discovers HF tokens
# from env vars, token file, and git-credentials. Exits with a clear error
# message if access is denied.
#
# Example:
#   proto_check_gated_hf_repo \
#       "EvolutionaryScale/esm3-sm-open-v1" \
#       "https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1"
#
#   proto_check_gated_hf_repo \
#       "google/alphagenome-all-folds" \
#       "https://huggingface.co/google/alphagenome-all-folds" \
#       "README.md"
#
# Reference: tools/masked_models/esm3/standalone/setup.sh
# ---------------------------------------------------------------------------
proto_check_gated_hf_repo() {
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
