#!/usr/bin/env bash
set -euo pipefail

readonly MODEL_DIR="artifacts/reference/twelveswaras"
readonly BASE_URL="https://huggingface.co/spaces/twelveswaras/twelveswaras/resolve/main/models"

mkdir -p "$MODEL_DIR"
for file in raaga_xgb.json raaga_xgb.classes.json raaga_xgb.calib.json; do
  curl \
    --fail \
    --location \
    --retry 5 \
    --retry-all-errors \
    --continue-at - \
    --output "$MODEL_DIR/$file" \
    "$BASE_URL/$file?download=true"
done

echo "Downloaded reference model to $MODEL_DIR"
