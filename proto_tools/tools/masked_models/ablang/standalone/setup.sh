#!/bin/bash
# Setup script for AbLang standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up AbLang standalone environment..."

echo "Installing uv package manager..."
pip install uv

proto_install_pytorch

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

# bioconda::anarci pulls HMMER + germline DBs; needed for ablang's align=True path.
"$MAMBA_BIN" install -p "$VENV_PATH" -c conda-forge -c bioconda -y anarci

# Pre-fetch ~825 MB of ablang weights into PROTO_MODEL_CACHE and symlink the
# cache into ablang2's hardcoded $(dirname ablang2.__file__)/model-weights-<name>
# lookup path. Bypasses ablang2.download_model's brittle requests.get (no
# retries) and keeps weights outside the env so they survive rebuilds.
# ablang2's lookup follows symlinks, so the staging is transparent.
# curl is on PATH via the host or the foundation env.
proto_resolve_weights_dir ablang
ABLANG_PKG_DIR=$(python -c "import ablang2, os; print(os.path.dirname(ablang2.__file__))")

prefetch_ablang_model() {
    local model_name="$1"
    local url="$2"
    local model_file="$3"

    local cache_dir="${WEIGHTS_DIR}/model-weights-${model_name}"
    local pkg_link="${ABLANG_PKG_DIR}/model-weights-${model_name}"

    mkdir -p "${cache_dir}"

    if [ -f "${cache_dir}/${model_file}" ]; then
        echo "ablang ${model_name} already cached at ${cache_dir}"
    else
        local tmp_tarball="${cache_dir}/tmp.tar.gz"
        echo "Pre-fetching ablang ${model_name} from ${url} into ${cache_dir}..."

        # --fail flips HTTP 4xx/5xx into nonzero exit. --retry-all-errors retries
        # every error class (default is connect-only). 6 total attempts × 600s
        # max-time + 5 × 30s retry-delay = ~62 min worst-case per model under a
        # hung upstream — long network "hangs" during setup are normal, not a bug.
        # --no-progress-meter (vs --silent) lets curl's retry warnings surface.
        if ! curl --no-progress-meter --show-error --location \
                  --fail --retry 5 --retry-delay 30 --retry-all-errors \
                  --max-time 600 \
                  --output "${tmp_tarball}" \
                  "${url}"; then
            echo "ERROR: pre-fetch of ablang ${model_name} failed after retries" >&2
            rm -f "${tmp_tarball}"
            exit 1
        fi

        # Validate gzip magic (1f 8b) before extracting. Defends against the rare
        # case where curl --fail can't catch the upstream returning HTML at HTTP 200.
        local magic
        magic=$(head -c 2 "${tmp_tarball}" | od -An -tx1 | tr -d ' \n')
        if [ "${magic}" != "1f8b" ]; then
            echo "ERROR: pre-fetched ${model_name} is not gzip (magic bytes: ${magic:-empty})" >&2
            rm -f "${tmp_tarball}"
            exit 1
        fi

        tar -zxf "${tmp_tarball}" -C "${cache_dir}"
        rm -f "${tmp_tarball}"

        if [ ! -f "${cache_dir}/${model_file}" ]; then
            echo "ERROR: extracted ${model_name} missing expected weight file ${model_file}" >&2
            exit 1
        fi
        echo "ablang ${model_name} staged at ${cache_dir}"
    fi

    # (Re-)create the symlink so ablang2.load_model.download_model finds the
    # cached weights at its hardcoded $(dirname ablang2.__file__)/model-weights-<name> path.
    rm -rf "${pkg_link}"
    ln -s "${cache_dir}" "${pkg_link}"
    echo "ablang ${model_name}: linked ${pkg_link} -> ${cache_dir}"
}

prefetch_ablang_model "ablang1-heavy"  "https://opig.stats.ox.ac.uk/data/downloads/ablang-heavy.tar.gz"   "amodel.pt"
prefetch_ablang_model "ablang1-light"  "https://opig.stats.ox.ac.uk/data/downloads/ablang-light.tar.gz"   "amodel.pt"
prefetch_ablang_model "ablang2-paired" "https://zenodo.org/records/10185169/files/ablang2-weights.tar.gz" "model.pt"

echo "AbLang setup complete!"
