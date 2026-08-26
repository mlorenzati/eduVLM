#!/usr/bin/env bash

set -euo pipefail

# ----------------------------------------------------------------------
# HMDB51 downloader/extractor
#
# Structure:
#
#   hmdb51_sta.rar
#       ├── brush_hair.rar
#       ├── cartwheel.rar
#       ├── ...
#       └── wave.rar
#
# This script:
#   1. Downloads hmdb51_sta.rar
#   2. Extracts the internal *.rar archives
#   3. Extracts every internal *.rar archive
#   4. Leaves the final video files in ${DATASET_DIR}
# ----------------------------------------------------------------------

DATASET_DIR="$(pwd -P)"
ARCHIVE="${DATASET_DIR}/hmdb51_sta.rar"

URL="https://huggingface.co/datasets/Serrelab/hmdb51/resolve/main/hmdb51_sta.rar"

echo
echo "=============================================="
echo " HMDB51 download and extraction"
echo "=============================================="
echo
echo "Destination:"
echo "  ${DATASET_DIR}"
echo

# ----------------------------------------------------------------------
# Check dependencies
# ----------------------------------------------------------------------

if ! command -v wget >/dev/null 2>&1; then
    echo "ERROR: wget is not installed."
    echo "Install with:"
    echo "  sudo apt install wget"
    exit 1
fi

if ! command -v unrar >/dev/null 2>&1; then
    echo "unrar not found. Installing..."
    sudo apt update
    sudo apt install -y unrar
fi

# ----------------------------------------------------------------------
# Download main archive
# ----------------------------------------------------------------------

if [[ -f "${ARCHIVE}" ]]; then
    echo "Main archive already exists:"
    echo "  ${ARCHIVE}"
    echo
else
    echo "Downloading HMDB51..."
    echo
    echo "  ${URL}"
    echo

    wget \
        --continue \
        --show-progress \
        --output-document="${ARCHIVE}" \
        "${URL}"

    echo
    echo "Download complete."
fi

# ----------------------------------------------------------------------
# Validate main archive
# ----------------------------------------------------------------------

echo
echo "Checking main archive..."

unrar t "${ARCHIVE}"

echo
echo "Main archive OK."

# ----------------------------------------------------------------------
# Extract internal RAR files
#
# We deliberately extract only the *.rar files first.
# ----------------------------------------------------------------------

echo
echo "=============================================="
echo " Extracting internal RAR archives"
echo "=============================================="
echo

unrar x \
    -o+ \
    "${ARCHIVE}" \
    "${DATASET_DIR}/"

# ----------------------------------------------------------------------
# Find internal RAR files
# ----------------------------------------------------------------------

mapfile -t INTERNAL_RARS < <(
    find "${DATASET_DIR}" \
        -maxdepth 1 \
        -type f \
        -iname "*.rar" \
        ! -samefile "${ARCHIVE}" \
        -print
)

echo
echo "Found ${#INTERNAL_RARS[@]} internal RAR archives."
echo

if [[ ${#INTERNAL_RARS[@]} -eq 0 ]]; then
    echo "WARNING: No internal RAR archives found."
    echo "Check the contents of:"
    echo "  ${DATASET_DIR}"
    exit 1
fi

# ----------------------------------------------------------------------
# Extract each action archive
# ----------------------------------------------------------------------

echo
echo "=============================================="
echo " Extracting action archives"
echo "=============================================="
echo

COUNT=0

for RAR in "${INTERNAL_RARS[@]}"; do

    COUNT=$((COUNT + 1))

    NAME="$(basename "${RAR}")"
    ACTION="${NAME%.rar}"

    echo
    echo "[${COUNT}/${#INTERNAL_RARS[@]}] ${ACTION}"
    echo "  Archive: ${RAR}"

    # Extract into DATASET_DIR.
    #
    # -o+ means overwrite existing files.
    # This makes the script safe to rerun.
    #
    unrar x \
        -o+ \
        "${RAR}" \
        "${DATASET_DIR}/"

    echo "  OK"
done

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------

echo
echo "=============================================="
echo " Extraction complete"
echo "=============================================="
echo

echo "Dataset directory:"
echo "  ${DATASET_DIR}"
echo

echo "Internal RAR archives:"
find "${DATASET_DIR}" \
    -maxdepth 1 \
    -type f \
    -iname "*.rar" \
    ! -samefile "${ARCHIVE}" \
    | wc -l

echo
echo "Video files:"

VIDEO_COUNT="$(
    find "${DATASET_DIR}" \
        -type f \
        \( \
            -iname "*.avi" \
            -o -iname "*.mp4" \
            -o -iname "*.mkv" \
        \) \
        | wc -l
)"

echo "  ${VIDEO_COUNT}"

echo
echo "Example files:"
echo

find "${DATASET_DIR}" \
    -type f \
    \( \
        -iname "*.avi" \
        -o -iname "*.mp4" \
        -o -iname "*.mkv" \
    \) \
    | head -20

echo
echo "Done."