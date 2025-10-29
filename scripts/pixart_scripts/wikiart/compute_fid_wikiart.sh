#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
GEN_DIR=${GEN_DIR:-local_repo/WikiArt_ft/output_new}
COUNT=${COUNT:-100}
START=${START:-0}

"$PYTHON_BIN" scripts/pixart_scripts/common/compute_fid_for_folder.py --generated_dir "$GEN_DIR" --dataset huggan/wikiart --split train --start "$START" --count "$COUNT" --print_timing_stats


