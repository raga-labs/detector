#!/usr/bin/env bash
set -euo pipefail

destination="${1:-data/raw/saraga-metadata}"

if [[ -d "$destination/.git" ]]; then
  git -C "$destination" pull --ff-only
else
  git clone --depth 1 https://github.com/MTG/saraga.git "$destination"
fi

echo "Saraga metadata is available at $destination"
echo "Audit it with: python3 scripts/audit_saraga_metadata.py $destination"
