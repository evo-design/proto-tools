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
        # Cap >= 13 to 12.8 — conda-forge cuda-* packages don't yet ship CUDA 13.
        if [ "$cuda_major" -ge 13 ] 2>/dev/null; then
            echo "proto_install_cuda_toolkit: detected CUDA ${cuda_major}; capping to 12.8 (no conda-forge cuda-13 packages yet)"
            cuda_constraint="12.8"
        else
            cuda_constraint="${cuda_major}.*"
        fi
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

    local cuda_log="/tmp/cuda_install.$$.log"
    if ! "$MAMBA_BIN" create -y -p "$VENV_PATH/cuda_env" -c nvidia -c conda-forge \
        "${packages[@]}" 2>&1 | tee "$cuda_log"; then
        echo "ERROR: proto_install_cuda_toolkit (constraint=${cuda_constraint}) failed at $VENV_PATH/cuda_env via $MAMBA_BIN; tail of log ($cuda_log):" >&2
        tail -5 "$cuda_log" >&2
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
# as the FIRST stderr line on failure, followed by 1-2 lines of context. The
# parser in ``tool_instance.py`` scans the whole output, so position is for
# human readability — the sentinel comes first so it's not buried.
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
        # Sentinel FIRST so callers and humans see the machine-readable signal
        # before context. The proto-tools test layer parses this exact prefix
        # and converts the failure into a skip.
        echo "[proto-tools] ASSET_NOT_AVAILABLE: ${toolkit}:${asset_kind}"
        echo "${toolkit} ${asset_kind} not provisioned at ${ASSET_DIR} (expected files matching '${pattern}'); set ${override_var}=/path or place files there to enable."
        if [ -n "$license_url" ]; then
            echo "License / access: ${license_url}"
        fi
        _emit_hint
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
        local reason
        local token_state
        if [ -n "$hf_token" ]; then
            token_state="present"
        else
            token_state="missing"
        fi
        case "$http_code" in
            401)
                if [ "$token_state" = "present" ]; then
                    reason="unauthorized — token present but rejected (expired/revoked); regenerate at https://huggingface.co/settings/tokens"
                else
                    reason="unauthorized — no HF_TOKEN found; set HF_TOKEN (or run 'hf auth login')"
                fi
                ;;
            403)
                if [ "$token_state" = "present" ]; then
                    reason="forbidden — token present but license not accepted; accept at ${license_url}"
                else
                    reason="forbidden — no HF_TOKEN found and license not accepted; set HF_TOKEN AND accept license at ${license_url}"
                fi
                ;;
            404) reason="repo or file not found" ;;
            429) reason="rate limited" ;;
            000|"") reason="curl could not reach huggingface.co (network/DNS failure)" ;;
            *)   reason="HTTP ${http_code} (token=${token_state})" ;;
        esac
        echo "ERROR: HuggingFace gated-repo check for '${repo_id}/${probe_file}' failed: ${reason}" >&2
        exit 1
    fi
}


# ---------------------------------------------------------------------------
# proto_download_gdrive <file_id> <dest>
#
# Download a Google Drive file by ID to <dest>, handling the >100MB
# confirm interstitial (re-submits the hidden-field form to
# drive.usercontent.google.com with cookies preserved). Self-contained:
# uses only stdlib so it works inside setup.sh subprocesses where
# proto_tools is not on PYTHONPATH.
#
# Example:
#   proto_download_gdrive 1YbTxkn9KuJP2D7U1-6kL1Yimu_4RqSl1 "$CAS_ID_DIR/HMM_sets.tar.gz"
# ---------------------------------------------------------------------------
proto_download_gdrive() {
    python3 -c "
import http.cookiejar, re, sys, urllib.parse, urllib.request

file_id, dest = sys.argv[1], sys.argv[2]
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
response = opener.open(f'https://drive.google.com/uc?export=download&id={file_id}', timeout=30)

if 'text/html' in response.headers.get('Content-Type', ''):
    html = response.read().decode('utf-8', errors='replace')
    action = re.search(r'<form[^>]*action=\"([^\"]+)\"', html)
    fields = dict(re.findall(r'<input type=\"hidden\" name=\"([^\"]+)\" value=\"([^\"]*)\"', html))
    if not action or 'id' not in fields:
        sys.exit(f'Google Drive interstitial for {file_id!r} had no confirm form')
    response = opener.open(f'{action.group(1)}?{urllib.parse.urlencode(fields)}', timeout=30)
    if 'text/html' in response.headers.get('Content-Type', ''):
        sys.exit(f'Google Drive still returning HTML after submitting confirm form for {file_id!r}')

with open(dest, 'wb') as f:
    while True:
        chunk = response.read(8192)
        if not chunk:
            break
        f.write(chunk)
" "$1" "$2"
}
