#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
RAW_DIR=${RAW_DIR:-local_repo/PixArt/output}
GEN_DIR=${GEN_DIR:-local_repo/PixArt_ft/output}
REAL_DIR=${REAL_DIR:-local_repo/PixArt/input/test}

echo "Computing FID for raw model..."
"$PYTHON_BIN" scripts/pixart_scripts/common/compute_fid_for_folder.py --generated_dir "$RAW_DIR" --real_dir "$REAL_DIR" --print_timing_stats
echo "Computing FID for finetuned model..."
"$PYTHON_BIN" scripts/pixart_scripts/common/compute_fid_for_folder.py --generated_dir "$GEN_DIR" --real_dir "$REAL_DIR" --print_timing_stats

