#!/usr/bin/env bash
set -euo pipefail

readonly DATA_DIR="data/raw/compmusic"
readonly ARCHIVE="$DATA_DIR/compmusic-raga-features.zip"
readonly URL="https://zenodo.org/records/7278506/files/Indian%20Art%20Music%20Raga%20Recognition%20Dataset%20%28features%29.zip?download=1"

mkdir -p "$DATA_DIR"
curl --fail --location --continue-at - --output "$ARCHIVE" "$URL"

echo "Downloaded: $ARCHIVE"
echo "Next: verify and inspect the archive before extraction."
