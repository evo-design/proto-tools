#!/bin/bash
# ============================================================================
# Shared shell helper functions for tool standalone setup scripts.
#
# This file is automatically copied to each tool's standalone/ directory
# at runtime alongside the standalone_helpers/ Python package.
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
#   proto_download_gdrive <file_id> <dest>
#   proto_sha256 <path>
#   proto_download_verified <url> <dest> <sha256>
#   proto_download <url> <dest>
#   proto_download_archive <url> <dest_dir> [tar_flags...]
#   proto_is_truncated_zip <path>
#   proto_install_binary <toolkit>
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
# Mirrors the Python resolve_weights_dir() logic in standalone_helpers/weights.py.
#
# Priority:
#   1. PROTO_{TOOL_NAME}_WEIGHTS_DIR (per-tool override)
#   2. PROTO_MODEL_CACHE:
#      - (default/unset): {PROTO_HOME}/proto_model_cache/{toolkit}/weights
#      - /absolute/path:  /absolute/path/{toolkit}/weights (shared directory)
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
        WEIGHTS_DIR="${PROTO_MODEL_CACHE}/${toolkit}/weights"
    else
        # Default: PROTO_HOME/proto_model_cache/ directory
        local _proto_home="${PROTO_HOME:-$HOME/.proto}"
        WEIGHTS_DIR="${_proto_home}/proto_model_cache/${toolkit}/weights"
    fi
    mkdir -p "$WEIGHTS_DIR"
}


# ---------------------------------------------------------------------------
# proto_ensure_af2_params
#
# Make sure the AlphaFold2 parameters are in alphafold2's weights directory and
# set AF2_PARAMS_DIR to that params/ folder. alphafold2, bindcraft, freebindcraft,
# and germinal all read this one copy (~5.3 GB), so the version below is pinned in
# one place (tests/style_consistency_tests/test_shared_weights_pins.py).
#
# The copy counts as complete when all 15 model files are present; a partial
# extraction is downloaded again. .params_complete records a verified copy.
#
# Example:
#   proto_ensure_af2_params
#   echo "AF2 params at: $AF2_PARAMS_DIR"
#
# Reference: tools/structure_prediction/alphafold2/standalone/setup.sh
# ---------------------------------------------------------------------------
proto_ensure_af2_params() {
    proto_resolve_weights_dir alphafold2
    AF2_PARAMS_DIR="${WEIGHTS_DIR}/params"
    mkdir -p "$AF2_PARAMS_DIR"

    _proto_af2_params_complete() {
        local model variant
        for model in 1 2 3 4 5; do
            for variant in "" "_ptm" "_multimer_v3"; do
                [ -s "${AF2_PARAMS_DIR}/params_model_${model}${variant}.npz" ] || return 1
            done
        done
    }

    if [ -f "${AF2_PARAMS_DIR}/.params_complete" ] || _proto_af2_params_complete; then
        touch "${AF2_PARAMS_DIR}/.params_complete"
        echo "AlphaFold2 parameters already present at ${AF2_PARAMS_DIR}"
        return 0
    fi

    echo "Downloading AlphaFold2 parameters (~5.3 GB) into ${AF2_PARAMS_DIR}..."
    curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar | tar x -C "$AF2_PARAMS_DIR"
    if ! _proto_af2_params_complete; then
        echo "ERROR: AlphaFold2 parameter download into ${AF2_PARAMS_DIR} is incomplete" >&2
        return 1
    fi
    touch "${AF2_PARAMS_DIR}/.params_complete"
}



# ---------------------------------------------------------------------------
# proto_ensure_chai1_weights
#
# Stage Chai-1's weights into the chai1 toolkit's weights directory and set
# CHAI1_WEIGHTS_DIR to it. chai1 and germinal both point CHAI_DOWNLOADS_DIR there,
# so chai_lab never falls back to its own (flaky) downloader. Every file is
# sha256-pinned, so both toolkits load exactly the same weights.
#
# Example:
#   proto_ensure_chai1_weights
#   echo "Chai-1 weights at: $CHAI1_WEIGHTS_DIR"
#
# Reference: tools/structure_prediction/chai1/standalone/setup.sh
# ---------------------------------------------------------------------------
proto_ensure_chai1_weights() {
    proto_resolve_weights_dir chai1
    CHAI1_WEIGHTS_DIR="$WEIGHTS_DIR"
    local HF_BASE CHAI_ASSETS COMPONENT

    # The model components come from Chai Discovery's HuggingFace repo, pinned to a commit. These
    # match chai_lab's own downloads byte for byte; the digests are the files' HuggingFace LFS ids.
    HF_BASE="https://huggingface.co/chaidiscovery/chai-1/resolve/9e998997951bbc1680f83939be7f0f4ef16187b2"
    local -A COMPONENT_SHA256=(
        ["feature_embedding.pt"]="58600fb282827ee105b4bf33845c2eb2dce2a7d2ba7582951f76c9290c66c019"
        ["bond_loss_input_proj.pt"]="98f3b2e415d11e460a684cdbfc213ca80447308022b1953cc5326ad468948851"
        ["token_embedder.pt"]="60cada19fd9c77d8298131a00368d162b6bc7d3c836dd36c715db04d21f7bbde"
        ["trunk.pt"]="b7a08e104455f76491088a5eeaea9bb95a08168c48969b0b892971346e51ccbc"
        ["diffusion_module.pt"]="bd77cf4095ad6b8a4eae7e390d01d24be1ebd819922ca3f28e8744f9945c9cad"
        ["confidence_head.pt"]="82a4ac9f934fdd0e73870150f508cecbe010c1383234bdadcaa70d57323deb6b"
    )
    mkdir -p "${CHAI1_WEIGHTS_DIR}/models_v2" "${CHAI1_WEIGHTS_DIR}/esm"
    for COMPONENT in "${!COMPONENT_SHA256[@]}"; do
        proto_download_verified \
            "${HF_BASE}/${COMPONENT}" \
            "${CHAI1_WEIGHTS_DIR}/models_v2/${COMPONENT}" \
            "${COMPONENT_SHA256[$COMPONENT]}"
    done

    # The ligand conformer cache and the traced ESM2 encoder are published only on Chai's CDN.
    CHAI_ASSETS="https://chaiassets.com/chai1-inference-depencencies"
    proto_download_verified \
        "${CHAI_ASSETS}/conformers_v1.apkl" \
        "${CHAI1_WEIGHTS_DIR}/conformers_v1.apkl" \
        "f2161256b565bbd84198da8b3e3d3978ae2f179f778d0995cd135c4c0c57c41f"
    proto_download_verified \
        "${CHAI_ASSETS}/esm2/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt" \
        "${CHAI1_WEIGHTS_DIR}/esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt" \
        "074673b97e1c1ff9c3cf949294749dd446ad1c83cee50a466b4841bb4d89c27a"
}


# ---------------------------------------------------------------------------
# proto_check_gated_hf_repo <repo_id> <license_url> [probe_file]
#
# Validate access to a gated HuggingFace repository. Discovers HF tokens
# from env vars, token file, and git-credentials. Exits with a clear error
# message if access is denied.
#
# The probe file must be gated: HuggingFace serves a gated repo's README.md
# (and often its LICENSE) to anyone, so probing those passes without a token
# or accepted terms. The default .gitattributes exists in every repo and is
# gated along with the weights.
#
# Example:
#   proto_check_gated_hf_repo \
#       "google/alphagenome-all-folds" \
#       "https://huggingface.co/google/alphagenome-all-folds"
#
# Reference: tools/sequence_scoring/alphagenome/standalone/setup.sh
# ---------------------------------------------------------------------------
proto_check_gated_hf_repo() {
    local repo_id="$1"
    local license_url="$2"
    local probe_file="${3:-.gitattributes}"

    local hf_token="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
    if [ -z "$hf_token" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
        hf_token="$(cat "$HOME/.cache/huggingface/token")"
    fi
    if [ -z "$hf_token" ] && [ -f "$HOME/.git-credentials" ]; then
        hf_token="$(grep -oP 'https?://[^:]+:\Khf_[^@]+(?=@huggingface\.co)' \
            "$HOME/.git-credentials" 2>/dev/null | head -1)" || true
    fi

    # -L follows HF's 307 redirect (resolve/main -> resolve-cache CDN); without it
    # the probe sees 307 and a valid, authorized token is wrongly rejected.
    local http_code
    if [ -n "$hf_token" ]; then
        http_code=$(curl -sL -o /dev/null -w "%{http_code}" \
            -H "Authorization: Bearer ${hf_token}" \
            "https://huggingface.co/${repo_id}/resolve/main/${probe_file}")
    else
        http_code=$(curl -sL -o /dev/null -w "%{http_code}" \
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
                    reason="unauthorized — token present but rejected (expired/revoked); regenerate at https://huggingface.co/settings/tokens, and accept the terms at ${license_url} if you have not already"
                else
                    reason="unauthorized — no HF_TOKEN found; set HF_TOKEN (or run 'hf auth login'), and accept the terms at ${license_url} if you have not already"
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


# ---------------------------------------------------------------------------
# proto_sha256 <path>
#
# Print the SHA-256 of a file, or nothing when it is missing. Uses sha256sum on
# Linux and shasum on macOS.
# ---------------------------------------------------------------------------
proto_sha256() {
    [ -f "$1" ] || return 0
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}


# ---------------------------------------------------------------------------
# proto_with_lock [--progress <path>] <lock_file> <command> [args...]
#
# Run <command> (a program or shell function) while holding an exclusive flock on
# <lock_file>, so setup scripts that share a model cache take turns on a download
# instead of writing the same files at once. This works across processes and, on
# the shared filesystems tested (Lustre $SCRATCH and Oak, NFS $HOME), across
# nodes. Returns <command>'s exit status.
#
# Waiting for another holder:
# - With --progress, the waiter watches <path> (a file, directory, or glob, such as
#   the holder's temp file) and keeps waiting while it grows, however long a big
#   download takes. It gives up only after $PROTO_LOCK_STALL_SECONDS (default 600)
#   without any growth, naming the stalled path. The download helpers abort a
#   transfer only once it has received nothing for PROTO_DOWNLOAD_STALL_SECONDS
#   (default 300), never a slow one that is still receiving data, so a stuck holder
#   normally releases the lock before its waiters give up.
# - Without --progress, it waits until the lock is free.
# - $PROTO_LOCK_TIMEOUT_SECONDS (default 0, off) caps the total wait either way.
# - The lock is checked every $PROTO_LOCK_POLL_SECONDS (default 60) at most; it is
#   taken the moment the holder releases it.
#
# Other behavior:
# - The kernel drops the lock when the holder exits, even on kill -9, so a crashed
#   run leaves no stale lock. The 0-byte lock file is kept: deleting it would let
#   two processes lock different inodes.
# - Re-entrant: a <command> that takes the same <lock_file> again (a helper calling
#   a helper) runs without waiting on itself. Different lock files nest normally.
# - <command> runs in a subshell, so variables it sets do not reach the caller.
# - Without flock(1) (macOS), <command> runs unlocked, as before locking existed.
#
# Name lock files like standalone_helpers' Python ones, ".<file>.download.lock"
# beside a file or ".<dir>.extract.lock" beside a directory, so bash and Python
# callers for the same target wait for each other.
#
# Example:
#   proto_with_lock --progress "${WEIGHTS_DIR}/params" \
#       "${WEIGHTS_DIR}/.params.extract.lock" _fetch_params
# ---------------------------------------------------------------------------
proto_with_lock() {
    local progress=""
    if [ "${1:-}" = "--progress" ]; then
        progress="$2"
        shift 2
    fi
    local lock="$1"
    shift
    case ":${_PROTO_HELD_LOCKS:-}:" in
        *":${lock}:"*)
            "$@"
            return
            ;;
    esac
    mkdir -p "$(dirname "$lock")"
    if ! command -v flock >/dev/null 2>&1; then
        "$@"
        return
    fi
    # fd 199 rather than a low number a setup.sh might already use; each nested
    # call's subshell gets its own, so an outer lock stays held
    (
        _proto_wait_for_lock 199 "$lock" "$progress" || exit 1
        export _PROTO_HELD_LOCKS="${_PROTO_HELD_LOCKS:+${_PROTO_HELD_LOCKS}:}${lock}"
        "$@"
    ) 199>>"$lock"
}


# Take the flock on <fd> (open on <lock>), waiting as proto_with_lock describes.
_proto_wait_for_lock() {
    local fd="$1" lock="$2" progress="$3"
    local poll="${PROTO_LOCK_POLL_SECONDS:-60}" stall="${PROTO_LOCK_STALL_SECONDS:-600}"
    local cap="${PROTO_LOCK_TIMEOUT_SECONDS:-0}"
    local waited=0 idle=0 chunk last="" now

    flock -x -n "$fd" && return 0
    echo "Waiting for another process to finish with $(basename "$lock")..."
    while :; do
        chunk="$poll"
        if [ "$cap" -gt 0 ] && [ $((cap - waited)) -lt "$chunk" ]; then
            chunk=$((cap - waited))
        fi
        if [ "$chunk" -gt 0 ] && flock -x -w "$chunk" "$fd"; then
            return 0
        fi
        waited=$((waited + chunk))
        if [ "$cap" -gt 0 ] && [ "$waited" -ge "$cap" ]; then
            echo "ERROR: timed out after ${waited}s waiting for lock ${lock}" >&2
            return 1
        fi
        [ -n "$progress" ] || continue
        now="$(_proto_progress_size "$progress")"
        if [ "$now" != "$last" ]; then
            last="$now"
            idle=0
        else
            idle=$((idle + chunk))
        fi
        if [ "$idle" -ge "$stall" ]; then
            echo "ERROR: gave up waiting for lock ${lock}: ${progress} made no progress for ${idle}s" \
                "(the holder may be stuck; raise PROTO_LOCK_STALL_SECONDS to wait longer)" >&2
            return 1
        fi
    done
}


# Print the total size of whatever <pattern> (a file, directory, or glob) matches,
# so a waiter can tell whether the lock holder is still writing.
_proto_progress_size() {
    local path total=0 size
    while IFS= read -r path; do
        if [ -d "$path" ]; then
            size="$(du -sk "$path" 2>/dev/null | cut -f1)"
        else
            size="$(wc -c < "$path" 2>/dev/null)"
        fi
        total=$((total + ${size:-0}))
    done < <(compgen -G "$1")
    echo "$total"
}


# Remove temp files a killed earlier download of <dest> left behind. Only call it
# while holding <dest>'s lock: no other downloader of <dest> can be running then.
_proto_remove_stale_temps() {
    local dest="$1" stale
    for stale in "$(dirname "$dest")/.$(basename "$dest")".*.tmp; do
        [ -e "$stale" ] && rm -f "$stale"
    done
    return 0
}


# ---------------------------------------------------------------------------
# proto_download_verified <url> <dest> <sha256>
#
# Download <url> to <dest>, verifying against <sha256>. A file already at <dest>
# is kept only when its digest matches, so a truncated or altered download is
# re-fetched instead of being trusted for merely existing. The transfer lands in
# a temp file of this process's own that is moved into place only once it
# matches, so an interrupted run never leaves something that looks finished.
# Concurrent callers for the same <dest> (two setups sharing a model cache) take
# turns under proto_with_lock, so one downloads and the rest find it verified.
# Returns nonzero if the freshly downloaded file does not match, which aborts a
# `set -e` setup.sh.
#
# Example:
#   proto_download_verified "$URL" "$WEIGHTS_DIR/model.pth" "$MODEL_SHA256"
#
# Reference: tools/structure_scoring/metal3d/standalone/setup.sh
# ---------------------------------------------------------------------------
proto_download_verified() {
    local dest="$2" expected="$3"
    # A verified file needs no lock, so read-only caches keep working
    if [ "$(proto_sha256 "$dest")" = "$expected" ]; then
        echo "$(basename "$dest") already present and verified."
        return 0
    fi
    mkdir -p "$(dirname "$dest")"
    proto_with_lock --progress "$(dirname "$dest")/.$(basename "$dest").*.tmp" \
        "$(dirname "$dest")/.$(basename "$dest").download.lock" _proto_download_verified_locked "$@"
}

_proto_download_verified_locked() {
    local url="$1" dest="$2" expected="$3"
    local actual tmp

    actual="$(proto_sha256 "$dest")"
    if [ "$actual" = "$expected" ]; then
        echo "$(basename "$dest") already present and verified."
        return 0
    fi
    if [ -n "$actual" ]; then
        echo "$(basename "$dest") failed checksum verification; re-downloading."
    fi

    echo "Downloading $(basename "$dest")..."
    _proto_remove_stale_temps "$dest"
    tmp="$(mktemp "$(dirname "$dest")/.$(basename "$dest").XXXXXX.tmp")"
    # Retry transient failures: some upstream CDNs drop large transfers intermittently
    if ! curl -fsSL --retry 5 --retry-delay 10 --retry-all-errors \
        --speed-limit 1 --speed-time "${PROTO_DOWNLOAD_STALL_SECONDS:-300}" -o "$tmp" "$url"; then
        rm -f "$tmp"
        echo "ERROR: failed to download ${url}" >&2
        return 1
    fi
    actual="$(proto_sha256 "$tmp")"
    if [ "$actual" != "$expected" ]; then
        rm -f "$tmp"
        echo "ERROR: checksum mismatch for ${url} (expected ${expected}, got ${actual})" >&2
        return 1
    fi
    mv -f "$tmp" "$dest"
}

# ---------------------------------------------------------------------------
# proto_is_truncated_zip <path>
#
# Succeed when <path> starts like a zip archive but has no end-of-central-directory
# record, i.e. the download was cut off (torch reports this at load time as
# "failed finding central directory"). Torch checkpoints (.pt, .ckpt) and .npz
# files are zip archives; other formats (legacy torch pickles, tarballs) never
# match, so they are not mistaken for truncated files.
# ---------------------------------------------------------------------------
proto_is_truncated_zip() {
    python3 -c "
import sys, zipfile
with open(sys.argv[1], 'rb') as f:
    looks_like_zip = f.read(4) == b'PK\x03\x04'
sys.exit(0 if looks_like_zip and not zipfile.is_zipfile(sys.argv[1]) else 1)
" "$1"
}


# ---------------------------------------------------------------------------
# proto_download <url> <dest>
#
# Download <url> to <dest> when no checksum is published (prefer
# proto_download_verified when one is). The transfer lands in a temp file of this
# process's own that is moved into place only once curl succeeds, so an
# interrupted run never leaves something that looks finished. An existing <dest>
# is kept unless it is empty or a truncated zip-format checkpoint (see
# proto_is_truncated_zip), which is re-downloaded. Concurrent callers for the same
# <dest> take turns under proto_with_lock. Returns nonzero on a failed download,
# which aborts a `set -e` setup.sh.
#
# Example:
#   proto_download "${REPO_BASE}/fampnn_0_3.pt" "${WEIGHTS_DIR}/fampnn_0_3.pt"
# ---------------------------------------------------------------------------
proto_download() {
    local dest="$2"
    # A complete file needs no lock, so read-only caches keep working
    if [ -s "$dest" ] && ! proto_is_truncated_zip "$dest"; then
        echo "$(basename "$dest") already present."
        return 0
    fi
    mkdir -p "$(dirname "$dest")"
    proto_with_lock --progress "$(dirname "$dest")/.$(basename "$dest").*.tmp" \
        "$(dirname "$dest")/.$(basename "$dest").download.lock" _proto_download_locked "$@"
}

_proto_download_locked() {
    local url="$1" dest="$2"
    local tmp

    if [ -s "$dest" ] && ! proto_is_truncated_zip "$dest"; then
        echo "$(basename "$dest") already present."
        return 0
    fi
    if [ -e "$dest" ]; then
        echo "$(basename "$dest") is incomplete; re-downloading."
    fi

    echo "Downloading $(basename "$dest")..."
    _proto_remove_stale_temps "$dest"
    tmp="$(mktemp "$(dirname "$dest")/.$(basename "$dest").XXXXXX.tmp")"
    if ! curl -fsSL --speed-limit 1 --speed-time "${PROTO_DOWNLOAD_STALL_SECONDS:-300}" -o "$tmp" "$url"; then
        rm -f "$tmp"
        echo "ERROR: failed to download ${url}" >&2
        return 1
    fi
    mv -f "$tmp" "$dest"
}


# ---------------------------------------------------------------------------
# proto_download_archive <url> <dest_dir> [tar_flags...]
#
# Stream the tar archive at <url> into <dest_dir> and mark it complete. The
# marker is written only after curl and tar both succeed, so a partial
# extraction is re-extracted on the next run instead of being trusted because
# some of its files exist. Concurrent callers for the same <dest_dir> take turns
# under proto_with_lock, so one extracts and the rest find the marker. Extra
# arguments go to tar (e.g. -z for .tar.gz).
#
# Example:
#   proto_download_archive "$AF2_PARAMS_URL" "${WEIGHTS_DIR}/params"
# ---------------------------------------------------------------------------
proto_download_archive() {
    local dest_dir="$2"
    # A completed extraction needs no lock, so read-only caches keep working
    if [ -f "${dest_dir}/.proto_download_complete" ]; then
        echo "$(basename "$dest_dir") already present."
        return 0
    fi
    mkdir -p "$(dirname "$dest_dir")"
    proto_with_lock --progress "$dest_dir" \
        "$(dirname "$dest_dir")/.$(basename "$dest_dir").extract.lock" _proto_download_archive_locked "$@"
}

_proto_download_archive_locked() {
    local url="$1" dest_dir="$2"
    shift 2
    local marker="${dest_dir}/.proto_download_complete"

    if [ -f "$marker" ]; then
        echo "$(basename "$dest_dir") already present."
        return 0
    fi

    echo "Downloading and extracting $(basename "$url")..."
    mkdir -p "$dest_dir"
    if ! (set -o pipefail; curl -fsSL --speed-limit 1 --speed-time "${PROTO_DOWNLOAD_STALL_SECONDS:-300}" "$url" | tar -x -C "$dest_dir" "$@"); then
        echo "ERROR: failed to download or extract ${url}" >&2
        return 1
    fi
    touch "$marker"
}


# ---------------------------------------------------------------------------
# proto_install_binary <toolkit>
#
# Download the platform-specific binary described by binary_config.py in the
# current directory (the env-def dir setup.sh runs from) into the venv's bin/.
# The installer is proto_tools/utils/install_binary.py, located through
# $PROTO_INSTALL_BINARY; <toolkit> labels its progress and error messages.
#
# Example:
#   proto_install_binary blast
#
# Reference: tools/sequence_alignment/blast/standalone/setup.sh
# ---------------------------------------------------------------------------
proto_install_binary() {
    local toolkit="$1"
    python "${PROTO_INSTALL_BINARY:?PROTO_INSTALL_BINARY is unset; run setup.sh through proto-tools}" "$toolkit"
}
