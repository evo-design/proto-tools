#!/bin/bash
# Setup script for AbLang standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up AbLang standalone environment..."

proto_install_pytorch

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

# bioconda::anarci pulls HMMER + germline DBs; needed for ablang's align=True path.
# numpy is constrained here too: the solver installs it as a dependency and would override the pip build.
"$MAMBA_BIN" install -p "$VENV_PATH" -c conda-forge -c bioconda -y anarci "numpy<2.5"

# Pre-fetch ~825 MB of ablang weights into PROTO_MODEL_CACHE and symlink the
# cache into ablang2's hardcoded $(dirname ablang2.__file__)/model-weights-<name>
# lookup path. Bypasses ablang2.download_model's brittle requests.get (no
# retries) and keeps weights outside the env so they survive rebuilds.
# ablang2's lookup follows symlinks, so the staging is transparent.
proto_resolve_weights_dir ablang
ABLANG_PKG_DIR=$(python -c "import ablang2, os; print(os.path.dirname(ablang2.__file__))")

prefetch_ablang_model() {
    local model_name="$1"
    local url="$2"
    local model_file="$3"

    local cache_dir="${WEIGHTS_DIR}/model-weights-${model_name}"
    local pkg_link="${ABLANG_PKG_DIR}/model-weights-${model_name}"

    # A cache extracted before proto_download_archive recorded its files has the
    # weights but no record, so check the weight file to keep it.
    if [ -f "${cache_dir}/${model_file}" ]; then
        echo "ablang ${model_name} already cached at ${cache_dir}"
    else
        echo "Pre-fetching ablang ${model_name} from ${url} into ${cache_dir}..."
        proto_download_archive "${url}" "${cache_dir}" -z
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
