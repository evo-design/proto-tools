#!/bin/bash -e
# Setup script for downloading ColabFold databases locally
# This script is adapted from the official ColabFold setup_databases.sh
# https://github.com/sokrypton/ColabFold/blob/main/setup_databases.sh

# Usage:
#   ./setup_databases.sh [WORKDIR] [UNIREF30DB] [CFDB] [SKIP_METAGENOMIC]
#
# Arguments:
#   WORKDIR: Directory to download databases to (default: ./databases)
#   UNIREF30DB: UniRef30 database version (default: uniref30_2302)
#   CFDB: ColabFold environmental database (default: colabfold_envdb_202108)
#   SKIP_METAGENOMIC: Set to 1 to skip downloading metagenomic database (default: 0)
#
# Examples:
#   # Download full databases (UniRef30 + metagenomic, ~200GB)
#   ./setup_databases.sh
#
#   # Download only UniRef30 database (~50GB, faster)
#   ./setup_databases.sh ./databases uniref30_2302 colabfold_envdb_202108 1
#
#   # Download to custom directory
#   ./setup_databases.sh /path/to/my/databases

ARIA_NUM_CONN=8
WORKDIR="${1:-$(dirname "$0")/databases}"
UNIREF30DB="${2:-uniref30_2302}"
CFDB="${3:-colabfold_envdb_202108}"
SKIP_METAGENOMIC="${4:-0}"

# Skip templates and PDB downloads for simplicity (not needed for MSA search)
SKIP_TEMPLATES=1

# Use prebuilt databases that support both CPU and GPU
FAST_PREBUILT_DATABASES=1

# Skip index creation if set
MMSEQS_NO_INDEX=${MMSEQS_NO_INDEX:-}

mkdir -p -- "${WORKDIR}"
cd "${WORKDIR}"

echo "=========================================="
echo "ColabFold Database Setup"
echo "=========================================="
echo "Download directory: ${WORKDIR}"
echo "UniRef30 database: ${UNIREF30DB}"
echo "Environmental database: ${CFDB}"
echo "Skip metagenomic: ${SKIP_METAGENOMIC}"
echo "=========================================="

hasCommand () {
    command -v "$1" >/dev/null 2>&1
}

fail() {
    echo "Error: $1"
    exit 1
}

if ! hasCommand mmseqs; then
  fail "mmseqs command not found in PATH. Please install MMseqs2."
fi

STRATEGY=""
if hasCommand aria2c; then STRATEGY="$STRATEGY ARIA"; fi
if hasCommand curl;   then STRATEGY="$STRATEGY CURL"; fi
if hasCommand wget;   then STRATEGY="$STRATEGY WGET"; fi
if [ "$STRATEGY" = "" ]; then
        fail "No download tool found in PATH. Please install aria2c, curl or wget."
fi

downloadFile() {
    URL="$1"
    OUTPUT="$2"
    set +e
    for i in $STRATEGY; do
        case "$i" in
        ARIA)
            FILENAME=$(basename "${OUTPUT}")
            DIR=$(dirname "${OUTPUT}")
            aria2c --max-connection-per-server="$ARIA_NUM_CONN" --allow-overwrite=true -o "$FILENAME" -d "$DIR" "$URL" && set -e && return 0
            ;;
        CURL)
            curl -L -o "$OUTPUT" "$URL" && set -e && return 0
            ;;
        WGET)
            wget -O "$OUTPUT" "$URL" && set -e && return 0
            ;;
        esac
    done
    set -e
    fail "Could not download $URL to $OUTPUT"
}

# Download databases
if [ ! -f DOWNLOADS_READY ]; then
  echo "Downloading databases..."
  downloadFile "https://opendata.mmseqs.org/colabfold/${UNIREF30DB}.db.tar.gz" "${UNIREF30DB}.tar.gz"

  if [ "${SKIP_METAGENOMIC}" != "1" ]; then
    downloadFile "https://opendata.mmseqs.org/colabfold/${CFDB}.db.tar.gz" "${CFDB}.tar.gz"
  fi

  if [ "${UNIREF30DB}" = "uniref30_2302" ]; then
    downloadFile "https://opendata.mmseqs.org/colabfold/uniref30_2302_newtaxonomy.tar.gz" "uniref30_2302_newtaxonomy.tar.gz"
  fi

  touch DOWNLOADS_READY
  echo "Downloads complete!"
fi

# Make MMseqs2 merge the databases to avoid spamming the folder with files
export MMSEQS_FORCE_MERGE=1

GPU_PAR=""
GPU_INDEX_PAR=""
if [ -n "${GPU}" ]; then
  GPU_PAR="--gpu 1"
  GPU_INDEX_PAR=" --split 1 --index-subset 2"

  if ! mmseqs --help | grep -q 'gpuserver'; then
    echo "Warning: The installed MMseqs2 has no GPU support, continuing with CPU mode"
    GPU=""
  fi
fi

# Setup UniRef30 database
if [ ! -f UNIREF30_READY ]; then
  echo "Setting up UniRef30 database..."
  tar -xzvf "${UNIREF30DB}.tar.gz"

  if [ -z "$MMSEQS_NO_INDEX" ]; then
    echo "Creating index for UniRef30 (this may take a while)..."
    mmseqs createindex "${UNIREF30DB}_db" tmp1 --remove-tmp-files 1 ${GPU_INDEX_PAR}
  fi

  # Update taxonomy and mapping files
  if [ -e "uniref30_2302_newtaxonomy.tar.gz" ]; then
    echo "Updating taxonomy files..."
    tar -xzvf "uniref30_2302_newtaxonomy.tar.gz"
  fi

  if [ -e ${UNIREF30DB}_db_mapping ]; then
    # Create binary, mmap-able taxonomy mapping
    TAXHEADER=$(od -An -N4 -t x4 "${UNIREF30DB}_db_mapping" | tr -d ' ')
    if [ "${TAXHEADER}" != "0c170013" ]; then
      mmseqs createbintaxmapping "${UNIREF30DB}_db_mapping" "${UNIREF30DB}_db_mapping.bin"
      mv -f -- "${UNIREF30DB}_db_mapping.bin" "${UNIREF30DB}_db_mapping"
    fi
    ln -sf ${UNIREF30DB}_db_mapping ${UNIREF30DB}_db.idx_mapping
  fi
  if [ -e ${UNIREF30DB}_db_taxonomy ]; then
    ln -sf ${UNIREF30DB}_db_taxonomy ${UNIREF30DB}_db.idx_taxonomy
  fi

  touch UNIREF30_READY
  echo "UniRef30 database setup complete!"
fi

# Setup ColabFold environmental database
if [ ! -f COLABDB_READY ] && [ "${SKIP_METAGENOMIC}" != "1" ]; then
  echo "Setting up ColabFold environmental database..."
  tar -xzvf "${CFDB}.tar.gz"

  if [ -z "$MMSEQS_NO_INDEX" ]; then
    echo "Creating index for environmental database (this may take a while)..."
    mmseqs createindex "${CFDB}_db" tmp2 --remove-tmp-files 1 ${GPU_INDEX_PAR}
  fi

  touch COLABDB_READY
  echo "Environmental database setup complete!"
fi

echo ""
echo "=========================================="
echo "Database setup complete!"
echo "=========================================="
echo "Location: ${WORKDIR}"
echo ""
echo "To use these databases with bio-programming:"
echo "  config = ColabfoldSearchConfig(msa_db_dir='${WORKDIR}')"
echo ""
if [ "${SKIP_METAGENOMIC}" = "1" ]; then
  echo "Note: Metagenomic database was skipped."
  echo "      To enable, set use_metagenomic_db=False in config."
fi
echo "=========================================="
